"""Evaluate a trained HOPE policy in-sim and report ``success_rate`` (the only metric).

Runs the policy across many parallel environments, detects each strike (the reference clock reaching
the strike frame), and rolls out the no-spin outgoing ball to decide whether the return succeeded
(racket contact AND net crossing AND opponent-half first bounce). Forehand, backhand and rally rounds
are merged into one number:

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

Scoring contract (kept consistent with the shared ``success_metric`` module):

* positions are mapped from the sim world into the TABLE frame the metric expects (origin at the
  near-side left corner of the table surface, x in [0, length], y in [-width, 0], z = 0 at the
  surface) using the same table placement the training command term uses for its return shaping
  (``table_near_x`` / ``table_surface_z`` / station-centred y);
* the outgoing ball leaves with the racket's ACHIEVED velocity at the strike frame (not the
  commanded target velocity);
* strikes that coincide with an environment reset (time-out / fall) are excluded — a reset re-seeds
  the reference clock, which is not a swing.

This remains the fast in-Isaac estimate; the authoritative physical number comes from
``mujoco_eval_onnx.py`` (real simulated ball). Emits only a machine-readable
``{"success_rate": <float>}`` to stdout (and optionally to --json-out). No thresholds, no exit-code
changes, no other metrics.

Usage:
    python scripts/evaluate.py --checkpoint logs/rsl_rl/agibot_a3_hitter_pingpong/<run>/model_<iter>.pt \
        --num-envs 256 --num-steps 4000
"""

import argparse
import json
import os
import pathlib
import sys
from types import SimpleNamespace


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cfg").is_dir() and (parent / "source" / "whole_body_tracking").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    return str(rooted.resolve()) if rooted.is_file() else str(rooted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint (.pt) to evaluate.")
    parser.add_argument("--task", default="HOPE-HitterPingPong-AgibotA3-v0", help="Gym task id.")
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel environments.")
    parser.add_argument("--num-steps", type=int, default=4000, help="Policy steps to roll out.")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic Isaac evaluation seed.")
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-to-target contact gate (m).")
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include strike-attempt and simulated-time diagnostics in the JSON result.",
    )
    parser.add_argument(
        "--virtual-telemetry-out",
        default=None,
        help=(
            "Write one JSON row per exact-strike virtual-ball attempt, including the sampled "
            "incoming state, planner goal, achieved racket state, rollout result, and failure code."
        ),
    )
    parser.add_argument(
        "--with-rewards",
        action="store_true",
        help="Keep the training RewardManager active; disabled by default for evaluation speed.",
    )
    parser.add_argument("--experiment-name", default="agibot_a3_hitter_pingpong", help="rsl_rl experiment name.")
    parser.add_argument(
        "--algo-config",
        default=None,
        help="Optional PPO YAML (for example cfg/algo/ppo_residual.yaml); default keeps official ppo.yaml.",
    )
    parser.add_argument(
        "--motion-file", default="motions/preprocessed/hope_forehand.npz", help="Forehand clip."
    )
    parser.add_argument(
        "--motion-file-2", default="motions/preprocessed/hope_backhand.npz", help="Backhand clip."
    )
    parser.add_argument(
        "--eval-clip-sequence",
        default=None,
        help="Evaluation-only clip sequence, e.g. '0' for forehand-only or '1' for backhand-only.",
    )
    parser.add_argument(
        "--condition-bh-target",
        action="store_true",
        help=(
            "Diagnostic-only: condition core backhand target v_z on sampled virtual incoming v_z; "
            "leaves tuple, normal, reward, and 110D contract unchanged."
        ),
    )
    parser.add_argument("--condition-k-z", type=float, default=0.75)
    parser.add_argument("--condition-v-ref", type=float, default=0.25)
    parser.add_argument("--condition-delta-max", type=float, default=0.40)
    parser.add_argument(
        "--condition-fh-target",
        action="store_true",
        help=(
            "Diagnostic-only: add a fixed core forehand target velocity xy offset; "
            "leaves BH, tuple, normal, reward, and 110D contract unchanged."
        ),
    )
    parser.add_argument("--condition-fh-dvx", type=float, default=0.0)
    parser.add_argument("--condition-fh-dvy", type=float, default=0.0)
    return parser.parse_args()


def _first_attr(obj, names):
    """Return the first present attribute among ``names`` (else None). Coupling shim for the env API."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app

    status = 0
    try:
        import gymnasium as gym
        import torch

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import importlib
        from omegaconf import OmegaConf

        importlib.import_module("whole_body_tracking.tasks")  # registers the gym tasks
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        # Evaluation must construct the same environment recipe as train.py.  The registered
        # Gym task supplies dataclass defaults, while the task YAML carries the full-pose mocap,
        # action, motion, and domain-randomization overrides used by training.  Omitting this
        # step silently evaluates a different environment (or fails during command construction).
        from train import _apply_task_overrides
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs
        # Pin the pure-NumPy scoring metric to this project.  Without
        # this explicit contract, a stale shell override or a missing local
        # configs/ball_physics.yaml can silently select another checkout or the
        # metric's fallback defaults.
        metric_config_path = _repo_root() / "configs" / "ball_physics.yaml"
        if not metric_config_path.is_file():
            raise FileNotFoundError(f"ball-physics config not found: {metric_config_path}")
        os.environ["HOPE_BALL_PHYSICS_CONFIG"] = str(metric_config_path)
        from whole_body_tracking.utils.success_metric import (
            BallPhysics,
            SuccessRate,
            TableGeometry,
            evaluate_return,
        )

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = int(args.seed)
        task_cfg_path = _repo_root() / "cfg" / "task" / "HOPEPingPong.yaml"
        task_cfg = OmegaConf.load(str(task_cfg_path))
        applied_overrides = []
        _apply_task_overrides(env_cfg, SimpleNamespace(task=task_cfg), applied_overrides)
        if args.condition_bh_target:
            env_cfg.commands.racket_target.vb_target_conditioning = True
            env_cfg.commands.racket_target.vb_target_conditioning_clip_id = 1
            env_cfg.commands.racket_target.vb_target_conditioning_k_z = float(args.condition_k_z)
            env_cfg.commands.racket_target.vb_target_conditioning_v_ref = float(args.condition_v_ref)
            env_cfg.commands.racket_target.vb_target_conditioning_delta_max = float(args.condition_delta_max)
            applied_overrides.extend(
                [
                    "commands.racket_target.vb_target_conditioning = true",
                    "commands.racket_target.vb_target_conditioning_clip_id = 1",
                    f"commands.racket_target.vb_target_conditioning_k_z = {float(args.condition_k_z):g}",
                    f"commands.racket_target.vb_target_conditioning_v_ref = {float(args.condition_v_ref):g}",
                    f"commands.racket_target.vb_target_conditioning_delta_max = {float(args.condition_delta_max):g}",
                ]
            )
        if args.condition_fh_target:
            env_cfg.commands.racket_target.fh_target_conditioning = True
            env_cfg.commands.racket_target.fh_target_conditioning_clip_id = 0
            env_cfg.commands.racket_target.fh_target_conditioning_delta_vx = float(args.condition_fh_dvx)
            env_cfg.commands.racket_target.fh_target_conditioning_delta_vy = float(args.condition_fh_dvy)
            applied_overrides.extend(
                [
                    "commands.racket_target.fh_target_conditioning = true",
                    "commands.racket_target.fh_target_conditioning_clip_id = 0",
                    f"commands.racket_target.fh_target_conditioning_delta_vx = {float(args.condition_fh_dvx):g}",
                    f"commands.racket_target.fh_target_conditioning_delta_vy = {float(args.condition_fh_dvy):g}",
                ]
            )
        if args.eval_clip_sequence is not None:
            sequence = tuple(int(item.strip()) for item in args.eval_clip_sequence.split(",") if item.strip())
            if not sequence:
                raise ValueError("--eval-clip-sequence must contain at least one clip id")
            env_cfg.commands.motion.eval_clip_sequence = sequence
            # The evaluator sequence must control every environment; otherwise the training
            # recipe's fixed cohorts would override part of the requested side-only probe.
            env_cfg.commands.motion.fixed_clip_env_fraction_per_clip = 0.0
            # The physical tuple bank has permanent global env-id cohorts (FH/BH).  A side-only
            # probe intentionally assigns every env to one clip, so disable that cohort-bound
            # mixture and evaluate the same clip-specific core target distribution instead.
            env_cfg.commands.racket_target.venue_tuple_enabled = False
            env_cfg.commands.racket_target.venue_tuple_mix_mode = "recovery_scaled_online_v1"
            env_cfg.commands.racket_target.venue_tuple_final_mix_prob = 0.0
            applied_overrides.append(f"commands.motion.eval_clip_sequence = {sequence}")
            applied_overrides.append("commands.motion.fixed_clip_env_fraction_per_clip = 0.0")
            applied_overrides.append("commands.racket_target.venue_tuple_enabled = False")
            applied_overrides.append("commands.racket_target.venue_tuple_mix_mode = recovery_scaled_online_v1")
            applied_overrides.append("commands.racket_target.venue_tuple_final_mix_prob = 0.0")
        print(
            f"[evaluate] applied {len(applied_overrides)} training task override(s)",
            flush=True,
        )
        clips = [_resolve_motion_path(c) for c in (args.motion_file, args.motion_file_2) if c]
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]

        # The external success metric below does not consume training rewards.  Leaving the
        # RewardManager enabled makes every evaluation step execute the full 50+ term reward
        # stack, including expensive diagnostics, while success_rate only needs strike state,
        # racket state, and the independent outgoing-ball rollout.  Terminations and command
        # updates remain active.  Use --with-rewards only when explicitly auditing reward values.
        if not args.with_rewards:
            disabled_rewards = 0
            for name, value in vars(env_cfg.rewards).items():
                if value is not None:
                    setattr(env_cfg.rewards, name, None)
                    disabled_rewards += 1
            print(
                f"[evaluate] RewardManager disabled for evaluation ({disabled_rewards} terms); "
                "terminations and strike-state updates remain active",
                flush=True,
            )

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        base_env = env.unwrapped
        env = RslRlVecEnvWrapper(env)

        agent_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(load_ppo_params(args.algo_config), args.experiment_name)
        )
        agent_cfg.device = args.device
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if bool(agent_cfg.empirical_normalization) and "obs_norm_state_dict" not in checkpoint_payload:
            print(
                "[evaluate] checkpoint has no observation-normalizer state; "
                "forcing raw-observation evaluation (empirical_normalization=false)",
                flush=True,
            )
            agent_cfg.empirical_normalization = False
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base_env.device)

        physics = BallPhysics.from_config()
        table = TableGeometry.from_config()
        accumulator = SuccessRate()
        contact_count = 0
        net_clear_count = 0
        opponent_bounce_count = 0
        reset_count = 0
        termination_counts = {}
        # Training-side virtual-ball counts.  These are accumulated from the exact
        # per-step strike masks, rather than the command's decayed EMA metrics, so
        # checkpoint sweeps can compare legal/attempt fairly across seeds.
        internal_attempts = 0
        internal_hits = 0
        internal_net_clears = 0
        internal_valid_landings = 0
        internal_legal_landings = 0
        virtual_telemetry_rows = []

        # The racket-target command term exposes the per-strike quantities we score. Attribute names
        # are read defensively (see COUPLING NOTES in the report): the command must expose the racket
        # target position (world), the achieved racket position AND velocity (world), the
        # time-to-strike, and the swing side.
        cmd = base_env.command_manager.get_term("racket_target")
        motion_cmd = base_env.command_manager.get_term("motion")
        env_origins = base_env.scene.env_origins  # (N, 3)

        # Table placement in the ENV-LOCAL frame — the same constants the command term's virtual
        # ball / return shaping uses (tasks/tracking/mdp/hope_commands.py, ``vb_table_near_x`` /
        # ``vb_table_surface_z``). The shared success-metric TABLE frame has its origin at the
        # near-side LEFT (+y) corner of the table surface: x_table = x_env - table_near_x,
        # y_table = y_env - (station_y + width/2), z_table = z_env - table_surface_z. Placement
        # (near_x / surface_z / station y) comes from the command cfg; table DIMENSIONS come from
        # the same TableGeometry the scoring uses (configs/ball_physics.yaml), so a re-fitted
        # table width keeps the two in agreement.
        table_near_x = float(cmd.cfg.vb_table_near_x)
        table_surface_z = float(cmd.cfg.vb_table_surface_z)
        table_half_w = 0.5 * float(table.width)

        def read_state():
            target_pos = _first_attr(cmd, ["racket_target_pos_w", "target_pos_w", "racket_target_w"])
            racket_pos = _first_attr(cmd, ["racket_pos_w", "achieved_racket_pos_w", "current_racket_pos_w"])
            racket_vel = _first_attr(cmd, ["racket_lin_vel_w", "racket_vel_w", "achieved_racket_vel_w"])
            tts = _first_attr(cmd, ["time_to_strike", "tts"])
            swing = _first_attr(cmd, ["swing_sign"])
            missing = [n for n, v in [
                ("racket_target_pos_w", target_pos), ("racket_pos_w", racket_pos),
                ("racket_lin_vel_w", racket_vel), ("time_to_strike", tts), ("swing_sign", swing)]
                if v is None]
            if missing:
                raise AttributeError(
                    "evaluate.py could not read the strike quantities from the 'racket_target' command "
                    f"term (missing: {missing}). Expose these tensors on the command term (world frame, "
                    "shape (N,3) for positions/velocities, (N,) for tts/swing) or adjust read_state()."
                )
            return target_pos, racket_pos, racket_vel, tts, swing

        def to_table_frame(pos_w_row, e):
            """Sim-world position -> the shared metric's table frame for env ``e``."""
            p = (pos_w_row - env_origins[e]).cpu().numpy().astype(float)
            # base_target_pos_w is the commanded base station (world xy) of env e.
            station_y = float((cmd.base_target_pos_w[e, 1] - env_origins[e, 1]).item())
            p[0] -= table_near_x
            p[1] -= station_y + table_half_w   # table centred on the station: left edge -> y = 0
            p[2] -= table_surface_z
            return p

        def _tolist(value, e, *, local_position=False):
            """Convert one live tensor row to JSON-safe data without retaining GPU storage."""
            if value is None:
                return None
            row = value[e]
            if local_position:
                row = row - env_origins[e]
            return row.detach().float().cpu().tolist()

        def _scalar(value, e):
            if value is None:
                return None
            return float(value[e].detach().float().cpu().item())

        def _bool(value, e):
            if value is None:
                return None
            return bool(value[e].detach().cpu().item())

        def _failure_code(fired, net_crossed, net_clear, land_valid, on_opponent, land_xy):
            """Use the first failed physical gate as the stable audit classification."""
            if not fired:
                return "MISS_CAPTURE"
            if not net_crossed:
                return "NO_NET_CROSS"
            if not net_clear:
                return "NET_TOO_LOW"
            if not land_valid:
                return "NO_LANDING_WITHIN_HORIZON"
            lx, ly = float(land_xy[0]), float(land_xy[1])
            if lx <= float(cmd._vb_net_x):
                return "LAND_OWN_HALF"
            if lx > float(cmd._vb_far_x):
                return "LAND_OUT_FAR"
            if abs(ly) > float(cmd._vb_half_w):
                return "LAND_OUT_SIDE"
            if on_opponent:
                return "LEGAL"
            return "LAND_OUT_OTHER"

        obs, _ = env.get_observations()
        prev_tts = read_state()[3].clone()
        for _ in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            target_pos, racket_pos, racket_vel, tts, swing = read_state()
            # ``exact_strike_hit_rate`` is a per-environment one-step mask, not a
            # rate despite its historical name.  ``vb_fired`` is the virtual-ball
            # contact gate for exactly the same strike.  Accumulate raw counts here
            # and derive rates from counts below; do not use the decayed EMA fields.
            exact_strike_mask = getattr(cmd, "metrics", {}).get("exact_strike_hit_rate")
            if exact_strike_mask is not None:
                exact_strike_mask = exact_strike_mask.to(dtype=torch.bool)
                vb_fired = getattr(cmd, "vb_fired", None)
                vb_net_clear = getattr(cmd, "vb_net_clear", None)
                vb_landing_valid = getattr(cmd, "vb_landing_valid", None)
                vb_on_opponent = getattr(cmd, "vb_on_opponent", None)
                if any(value is None for value in (vb_fired, vb_net_clear, vb_landing_valid, vb_on_opponent)):
                    raise RuntimeError(
                        "racket_target does not expose the complete virtual-ball outcome contract"
                    )
                exact_strike_mask = exact_strike_mask & (~dones.reshape(-1).to(dtype=torch.bool))
                internal_attempts += int(exact_strike_mask.sum().item())
                internal_hits += int((exact_strike_mask & vb_fired).sum().item())
                internal_net_clears += int((exact_strike_mask & vb_fired & vb_net_clear).sum().item())
                internal_valid_landings += int((exact_strike_mask & vb_fired & vb_landing_valid).sum().item())
                internal_legal_landings += int(
                    (exact_strike_mask & vb_fired & vb_net_clear & vb_on_opponent).sum().item()
                )
                if args.virtual_telemetry_out:
                    telemetry_ids = exact_strike_mask.nonzero(as_tuple=False).flatten().tolist()
                    incoming_v = getattr(cmd, "vb_vel_in_w", None)
                    incoming_w = getattr(cmd, "vb_spin_in_w", None)
                    landing_xy = getattr(cmd, "vb_landing_xy", None)
                    net_z = getattr(cmd, "vb_net_z", None)
                    net_crossed = getattr(cmd, "vb_net_crossed", None)
                    landing_valid = getattr(cmd, "vb_landing_valid", None)
                    on_opponent = getattr(cmd, "vb_on_opponent", None)
                    venue_selected = getattr(cmd, "_venue_tuple_selected", None)
                    venue_intended = getattr(cmd, "_venue_intended_landing_xy", None)
                    default_intended = getattr(cmd, "_vb_target_xy", None)
                    clip_id = getattr(motion_cmd, "clip_id", None)
                    for e in telemetry_ids:
                        fired = _bool(vb_fired, e)
                        crossed = _bool(net_crossed, e)
                        clear = _bool(vb_net_clear, e)
                        valid = _bool(landing_valid, e)
                        opponent = _bool(on_opponent, e)
                        land = _tolist(landing_xy, e)
                        if land is None:
                            land = [0.0, 0.0]
                        selected = _bool(venue_selected, e) or False
                        if selected and venue_intended is not None:
                            intended = _tolist(venue_intended, e)
                        elif default_intended is not None:
                            intended = default_intended.detach().float().cpu().tolist()
                        else:
                            intended = None
                        virtual_telemetry_rows.append(
                            {
                                "env_id": int(e),
                                "clip_id": int(clip_id[e].detach().cpu().item()) if clip_id is not None else None,
                                "swing_sign": _scalar(getattr(cmd, "swing_sign", None), e),
                                "venue_tuple_selected": selected,
                                "incoming_velocity": _tolist(incoming_v, e),
                                "incoming_spin": _tolist(incoming_w, e),
                                "intended_landing_xy_env": intended,
                                "planner_racket_pos_env": _tolist(
                                    getattr(cmd, "racket_target_pos_w", None), e, local_position=True
                                ),
                                "planner_racket_velocity": _tolist(
                                    getattr(cmd, "racket_target_vel_w", None), e
                                ),
                                "planner_racket_normal": _tolist(
                                    getattr(cmd, "racket_target_normal_w", None), e
                                ),
                                "time_to_strike": _scalar(getattr(cmd, "time_to_strike", None), e),
                                "achieved_racket_pos_env": _tolist(
                                    getattr(cmd, "racket_pos_w", None), e, local_position=True
                                ),
                                "achieved_racket_velocity": _tolist(
                                    getattr(cmd, "racket_lin_vel_w", None), e
                                ),
                                "achieved_racket_normal": _tolist(
                                    getattr(cmd, "racket_normal_w", None), e
                                ),
                                "capture_gate": fired,
                                "net_crossed": crossed,
                                "net_clear": clear,
                                "net_z_env": _scalar(net_z, e),
                                "landing_valid": valid,
                                "landing_xy_env": land,
                                "on_opponent": opponent,
                                "failure_code": _failure_code(
                                    fired, crossed, clear, valid, opponent, land
                                ),
                            }
                        )
            # A strike happens when the reference clock crosses the strike frame (tts: >0 -> <=0).
            # Environments that RESET this step are excluded: a time-out/fall reset re-seeds the
            # clock, and counting it would contaminate the denominator with non-swings.
            reset_now = dones.reshape(-1).to(dtype=torch.bool, device=tts.device)
            reset_count += int(reset_now.sum().item())
            if args.diagnostics:
                termination_manager = getattr(base_env, "termination_manager", None)
                if termination_manager is not None:
                    for term_name in getattr(termination_manager, "_term_names", []):
                        try:
                            term_value = termination_manager.get_term(term_name)
                            termination_counts[term_name] = termination_counts.get(term_name, 0) + int(
                                term_value.to(dtype=torch.bool).sum().item()
                            )
                        except Exception:
                            pass
            struck = (prev_tts > 0.0) & (tts <= 0.0) & (~reset_now)
            idx = struck.nonzero(as_tuple=False).flatten().tolist()
            for e in idx:
                tp = to_table_frame(target_pos[e], e)
                rp = to_table_frame(racket_pos[e], e)
                rv = racket_vel[e].cpu().numpy().astype(float)  # achieved racket velocity at strike
                outcome = evaluate_return(tp, rp, rv, physics, table, contact_radius=args.contact_radius)
                accumulator.add(outcome)
                contact_count += int(outcome.contacted)
                net_clear_count += int(outcome.net_clear)
                opponent_bounce_count += int(outcome.on_opponent)
            prev_tts = tts.clone()

        result = accumulator.as_dict()
        if args.diagnostics:
            # Keep the public external success metric separate from Isaac's internal
            # virtual-ball diagnostics.  The former is an independent no-spin rollout;
            # the latter is the exact training-side venue contact/landing pipeline.
            internal_metric_names = (
                "exact_strike_hit_rate",
                "strike_composite_success_exact",
                "virtual_hit_rate",
                "virtual_contact_rate",
                "virtual_net_clear_rate",
                "virtual_land_valid_rate",
                "virtual_land_inbounds_rate",
                "virtual_legal_rate",
                "virtual_land_err_m",
            )
            internal_metrics = {}
            for name in internal_metric_names:
                value = getattr(cmd, "metrics", {}).get(name)
                if value is None:
                    continue
                if hasattr(value, "detach"):
                    value = value.detach().float().mean().item()
                internal_metrics[name] = float(value)
            for name, value in getattr(cmd, "metrics", {}).items():
                if not any(token in name for token in ("virtual_contact_rate", "virtual_legal_rate", "virtual_over_net_rate")):
                    continue
                if hasattr(value, "detach"):
                    value = value.detach().float().mean().item()
                internal_metrics[name] = float(value)
            # The public virtual_* values are deliberately gated until a minimum EMA sample
            # count.  Expose the raw accumulators too, so a short diagnostic run cannot turn
            # "not enough samples" into a misleading zero rate.
            for name in ("_vb_exact_acc", "_vb_hit_acc", "_vb_net_acc", "_vb_land_valid_acc", "_vb_inb_acc"):
                value = getattr(cmd, name, None)
                if value is not None:
                    internal_metrics[name.lstrip("_")] = float(value)
            result.update(
                {
                    "attempts": int(accumulator.attempts),
                    "successes": int(accumulator.successes),
                    "contacts": int(contact_count),
                    "net_clears": int(net_clear_count),
                    "opponent_bounces": int(opponent_bounce_count),
                    "simulated_seconds": float(args.num_steps * base_env.step_dt),
                    "num_envs": int(args.num_envs),
                    "num_steps": int(args.num_steps),
                    "seed": int(args.seed),
                    "resets": int(reset_count),
                    "reset_rate_per_1k_steps": float(
                        reset_count / max(int(args.num_envs) * int(args.num_steps), 1) * 1000.0
                    ),
                    "internal_virtual_counts": {
                        "attempts": int(internal_attempts),
                        "hits": int(internal_hits),
                        "net_clears": int(internal_net_clears),
                        "valid_landings": int(internal_valid_landings),
                        "legal_landings": int(internal_legal_landings),
                    },
                    "internal_virtual_rates": {
                        "hit_per_attempt": float(internal_hits / max(internal_attempts, 1)),
                        "net_per_attempt": float(internal_net_clears / max(internal_attempts, 1)),
                        "legal_per_attempt": float(internal_legal_landings / max(internal_attempts, 1)),
                        "legal_per_hit": float(internal_legal_landings / max(internal_hits, 1)),
                        "valid_land_per_attempt": float(internal_valid_landings / max(internal_attempts, 1)),
                    },
                    "termination_counts": termination_counts,
                    "isaac_internal_metrics_mean": internal_metrics,
                }
            )
        if args.virtual_telemetry_out:
            telemetry_path = os.path.abspath(args.virtual_telemetry_out)
            os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
            telemetry_payload = {
                "schema_version": 1,
                "checkpoint": checkpoint,
                "seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "num_steps": int(args.num_steps),
                "target_conditioning": {
                    "enabled": bool(args.condition_bh_target),
                    "clip_id": 1,
                    "k_z": float(args.condition_k_z),
                    "v_ref": float(args.condition_v_ref),
                    "delta_max": float(args.condition_delta_max),
                    "fh_enabled": bool(args.condition_fh_target),
                    "fh_clip_id": 0,
                    "fh_delta_vx": float(args.condition_fh_dvx),
                    "fh_delta_vy": float(args.condition_fh_dvy),
                },
                "table_frame": {
                    "env_local_table_near_x": table_near_x,
                    "env_local_net_x": float(cmd._vb_net_x),
                    "env_local_far_x": float(cmd._vb_far_x),
                    "half_width": table_half_w,
                    "surface_z": table_surface_z,
                },
                "rows": virtual_telemetry_rows,
            }
            with open(telemetry_path, "w", encoding="utf-8") as f:
                json.dump(telemetry_payload, f, ensure_ascii=False)
                f.write("\n")
            result["virtual_telemetry_out"] = telemetry_path
        print(json.dumps(result))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f)
                f.write("\n")
        env.close()
    except Exception:
        import traceback

        print("\n[evaluate] ERROR:", flush=True)
        traceback.print_exc()
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

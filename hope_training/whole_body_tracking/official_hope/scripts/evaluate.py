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
        "--ready-telemetry-out",
        default=None,
        help=(
            "Write one JSON record per recovery cycle with deadline component checks, "
            "late/never/unstable classification, numeric state, and residual saturation. "
            "Requires --with-rewards so the deadline gate is available."
        ),
    )
    parser.add_argument(
        "--ready-telemetry-offsets",
        default="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,2.5",
        help="Recovery trace offsets in seconds for --ready-telemetry-out.",
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
        "--task-config",
        default="cfg/task/HOPEPingPong.yaml",
        help=(
            "Task YAML used for evaluation. It is merged on top of the base HOPEPingPong recipe; "
            "use cfg/task/HOPEPingPongStanceCurriculum.yaml for the stance Curriculum-FT policy."
        ),
    )
    parser.add_argument(
        "--motion-file", default="motions/preprocessed/hope_forehand.npz", help="Forehand clip."
    )
    parser.add_argument(
        "--motion-file-2", default="motions/preprocessed/hope_backhand.npz", help="Backhand clip."
    )
    parser.add_argument(
        "--motion-file-3", default=None, help="Optional third clip (serve adaptation clip)."
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
        "--target-strike-interval-s",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Override the wrap scheduler target strike-to-strike interval in seconds. "
            "The motion clip remains full-length; the scheduler solves the successor hold."
        ),
    )
    parser.add_argument(
        "--short-transition-env-fraction",
        type=float,
        default=None,
        help=(
            "Assign this fraction of the highest environment ids to the short-transition "
            "scheduler/clip sequence; 0.0 means all environments are short-transition."
        ),
    )
    parser.add_argument(
        "--transition-clip-sequence",
        default=None,
        help="Short-transition clip sequence, e.g. '0,0,1,1'.",
    )
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
    parser.add_argument(
        "--condition-fh-vx-max",
        type=float,
        default=None,
        help="Optional incoming FH ball-vx upper gate for the horizontal correction (m/s).",
    )
    parser.add_argument(
        "--paired-recipe-mode",
        choices=("off", "capture", "replay"),
        default="off",
        help="Capture or replay a per-environment paired question recipe for causal audits.",
    )
    parser.add_argument(
        "--paired-recipe-path",
        default=None,
        help="JSON path for paired recipe capture/replay.",
    )
    parser.add_argument(
        "--paired-recipe-source-env-id",
        type=int,
        default=None,
        help="Replay one source environment's event sequence in a single-env controlled branch.",
    )
    parser.add_argument(
        "--paired-recipe-source-env-ids",
        default=None,
        help="Comma-separated source environment ids, one per runtime env, for batched branches.",
    )
    parser.add_argument(
        "--paired-recipe-nonstrict",
        action="store_true",
        help="Log topology/clip divergence instead of aborting paired replay (diagnostic only).",
    )
    parser.add_argument(
        "--post-strike-offsets",
        default="0.05,0.10,0.20,0.30,0.50,0.80",
        help="Comma-separated post-contact telemetry offsets in seconds.",
    )
    parser.add_argument(
        "--state-transplant-source",
        default=None,
        help="Baseline telemetry JSON containing post_strike_state_rows for a transplant audit.",
    )
    parser.add_argument(
        "--state-transplant-source-env-id",
        type=int,
        default=None,
        help="Map this source telemetry environment to runtime env 0 for a single-shot branch.",
    )
    parser.add_argument(
        "--state-transplant-source-env-ids",
        default=None,
        help="Comma-separated source telemetry env ids, one per runtime env.",
    )
    parser.add_argument(
        "--state-transplant-offset",
        type=float,
        default=None,
        help="Apply the source state at this post-contact offset (seconds).",
    )
    parser.add_argument(
        "--state-transplant-fields",
        default="root_lin_vel,root_ang_vel",
        help="Comma-separated fields: root_pos,root_quat,root_lin_vel,root_ang_vel,joint_pos,joint_vel.",
    )
    parser.add_argument(
        "--snapshot-branch-mode",
        action="store_true",
        help=(
            "Run the true in-process single-shot branch audit. Requires five environments; "
            "env 0 is the gate snapshot and envs 1..4 are G-ang/G-lin/G-upper/G-allvel."
        ),
    )
    parser.add_argument(
        "--snapshot-branch-source",
        default=None,
        help="Baseline telemetry JSON supplying the counterfactual state at snapshot offset.",
    )
    parser.add_argument(
        "--snapshot-branch-source-env-id",
        type=int,
        default=None,
        help="Source telemetry env id used for the in-process snapshot counterfactual.",
    )
    parser.add_argument(
        "--snapshot-branch-offset",
        type=float,
        default=0.20,
        help="Post-FH snapshot offset in seconds for the in-process branch audit.",
    )
    parser.add_argument(
        "--snapshot-branch-blend-steps",
        type=int,
        default=0,
        help=(
            "If positive, ramp branch velocity interventions over this many control steps; "
            "zero keeps the legacy instantaneous transplant."
        ),
    )
    parser.add_argument(
        "--snapshot-action-replay-steps",
        type=int,
        default=0,
        help=(
            "If positive, replay the baseline policy action on branches for this many steps "
            "after the snapshot; no simulator state transplant is applied."
        ),
    )
    parser.add_argument(
        "--snapshot-action-replay-sequence",
        action="store_true",
        help=(
            "Replay the source baseline policy action at each saved post-strike offset after the "
            "snapshot; requires snapshot branch mode and policy_action telemetry."
        ),
    )
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
    try:
        post_strike_offsets = tuple(
            float(item.strip())
            for item in str(args.post_strike_offsets).split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise ValueError("--post-strike-offsets must be comma-separated seconds") from exc
    if not post_strike_offsets or any(value <= 0.0 for value in post_strike_offsets):
        raise ValueError("--post-strike-offsets must contain positive values")
    try:
        ready_telemetry_offsets = tuple(
            float(item.strip())
            for item in str(args.ready_telemetry_offsets).split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise ValueError("--ready-telemetry-offsets must be comma-separated seconds") from exc
    if args.ready_telemetry_out and (
        not ready_telemetry_offsets
        or any(value < 0.0 for value in ready_telemetry_offsets)
        or tuple(sorted(set(ready_telemetry_offsets))) != ready_telemetry_offsets
    ):
        raise ValueError(
            "--ready-telemetry-offsets must be sorted, unique, and non-negative"
        )
    if args.ready_telemetry_out and not args.with_rewards:
        raise ValueError("--ready-telemetry-out requires --with-rewards")
    if args.state_transplant_source and args.state_transplant_offset is None:
        raise ValueError("--state-transplant-offset is required with --state-transplant-source")
    if args.snapshot_branch_mode:
        if int(args.num_envs) < 5:
            raise ValueError("--snapshot-branch-mode requires at least five environments")
        if not args.snapshot_branch_source:
            raise ValueError("--snapshot-branch-source is required with --snapshot-branch-mode")
        if args.snapshot_branch_source_env_id is None:
            raise ValueError("--snapshot-branch-source-env-id is required with --snapshot-branch-mode")
        if float(args.snapshot_branch_offset) <= 0.0:
            raise ValueError("--snapshot-branch-offset must be positive")
        if args.snapshot_action_replay_sequence and int(args.snapshot_action_replay_steps) > 0:
            raise ValueError(
                "use only one of --snapshot-action-replay-sequence and "
                "--snapshot-action-replay-steps"
            )
    def _parse_id_list(value, option_name):
        if value is None:
            return None
        try:
            values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
        except ValueError as exc:
            raise ValueError(f"{option_name} must be comma-separated integers") from exc
        if not values:
            raise ValueError(f"{option_name} must contain at least one id")
        return values

    paired_recipe_source_env_ids = _parse_id_list(
        args.paired_recipe_source_env_ids, "--paired-recipe-source-env-ids"
    )
    transplant_source_env_ids = _parse_id_list(
        args.state_transplant_source_env_ids, "--state-transplant-source-env-ids"
    )
    if paired_recipe_source_env_ids is not None and args.paired_recipe_source_env_id is not None:
        raise ValueError("use only one of --paired-recipe-source-env-id/--paired-recipe-source-env-ids")
    if transplant_source_env_ids is not None and args.state_transplant_source_env_id is not None:
        raise ValueError(
            "use only one of --state-transplant-source-env-id/--state-transplant-source-env-ids"
        )
    transplant_records = {}
    if args.state_transplant_source:
        transplant_path = pathlib.Path(args.state_transplant_source).expanduser()
        if not transplant_path.is_absolute():
            transplant_path = (_repo_root() / transplant_path).resolve()
        transplant_payload = json.loads(transplant_path.read_text(encoding="utf-8"))
        for row in transplant_payload.get("post_strike_state_rows", []):
            # Reset-before-offset markers share this list but do not carry a
            # sampled offset/state payload.
            if "offset_s" not in row:
                continue
            if transplant_source_env_ids is not None:
                source_to_runtime = {source: runtime for runtime, source in enumerate(transplant_source_env_ids)}
                if int(row["env_id"]) not in source_to_runtime:
                    continue
                runtime_env_id = source_to_runtime[int(row["env_id"])]
            elif (
                args.state_transplant_source_env_id is not None
                and int(row["env_id"]) != int(args.state_transplant_source_env_id)
            ):
                continue
            else:
                runtime_env_id = int(row["env_id"])
            if args.state_transplant_source_env_id is not None:
                runtime_env_id = 0
            key = (
                runtime_env_id,
                int(row["source_paired_recipe_index"]),
                round(float(row["offset_s"]), 6),
            )
            transplant_records[key] = row
    snapshot_baseline_rows = {}
    snapshot_action_rows_by_recipe_offset = {}
    if args.snapshot_branch_mode:
        snapshot_path = pathlib.Path(args.snapshot_branch_source).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = (_repo_root() / snapshot_path).resolve()
        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        source_env_id = int(args.snapshot_branch_source_env_id)
        for row in snapshot_payload.get("post_strike_state_rows", []):
            if "offset_s" not in row:
                continue
            if int(row.get("env_id", -1)) != source_env_id:
                continue
            recipe_index = int(row["source_paired_recipe_index"])
            offset_key = round(float(row["offset_s"]), 6)
            if row.get("policy_action") is not None:
                snapshot_action_rows_by_recipe_offset[(recipe_index, offset_key)] = row
            if offset_key == round(float(args.snapshot_branch_offset), 6):
                snapshot_baseline_rows[recipe_index] = row
        if not snapshot_baseline_rows:
            raise RuntimeError(
                "snapshot baseline telemetry has no post-strike row for the requested "
                f"env={source_env_id}, offset={args.snapshot_branch_offset}"
            )

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
        from train import _apply_friction_curriculum, _apply_task_overrides
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

        # The checkpoint's PPO iteration is part of the stance/friction curriculum contract.
        # Load it before the first environment reset so replay/evaluation starts with the same
        # alpha/beta phase as the saved policy instead of silently falling back to iteration 0.
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_iteration = int(checkpoint_payload.get("iter", 0))

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = int(args.seed)
        base_task_cfg_path = _repo_root() / "cfg" / "task" / "HOPEPingPong.yaml"
        task_cfg_path = pathlib.Path(args.task_config).expanduser()
        if not task_cfg_path.is_absolute():
            task_cfg_path = (_repo_root() / task_cfg_path).resolve()
        if not task_cfg_path.is_file():
            raise FileNotFoundError(f"task config not found: {task_cfg_path}")
        # The stance recipe is a Hydra defaults overlay.  This standalone evaluator does not
        # invoke Hydra composition, so explicitly merge it onto the complete base recipe.
        task_cfg = OmegaConf.load(str(base_task_cfg_path))
        if task_cfg_path.resolve() != base_task_cfg_path.resolve():
            task_cfg = OmegaConf.merge(task_cfg, OmegaConf.load(str(task_cfg_path)))
        applied_overrides = []
        _apply_task_overrides(env_cfg, SimpleNamespace(task=task_cfg), applied_overrides)
        # Evaluation must install the same reset-time feet-only friction event as training.
        # Without this explicit launcher step, the registered task's legacy startup material
        # event remains active and endpoint telemetry cannot be interpreted against the saved
        # curriculum phase.
        _apply_friction_curriculum(
            env_cfg,
            getattr(task_cfg, "friction_curriculum", None),
            applied_overrides,
        )
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
        if args.target_strike_interval_s is not None:
            interval = tuple(float(value) for value in args.target_strike_interval_s)
            env_cfg.commands.racket_target.target_strike_interval_s = interval
            applied_overrides.append(
                f"commands.racket_target.target_strike_interval_s = {interval}"
            )
        if args.short_transition_env_fraction is not None:
            fraction = float(args.short_transition_env_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("--short-transition-env-fraction must lie in [0, 1]")
            env_cfg.commands.motion.short_transition_env_fraction = fraction
            applied_overrides.append(
                f"commands.motion.short_transition_env_fraction = {fraction:g}"
            )
        if args.transition_clip_sequence is not None:
            sequence = tuple(
                int(item.strip())
                for item in args.transition_clip_sequence.split(",")
                if item.strip()
            )
            if not sequence:
                raise ValueError("--transition-clip-sequence must contain at least one clip id")
            env_cfg.commands.motion.transition_clip_sequence = sequence
            applied_overrides.append(
                f"commands.motion.transition_clip_sequence = {sequence}"
            )
        if args.condition_fh_target:
            env_cfg.commands.racket_target.fh_target_conditioning = True
            env_cfg.commands.racket_target.fh_target_conditioning_clip_id = 0
            env_cfg.commands.racket_target.fh_target_conditioning_delta_vx = float(args.condition_fh_dvx)
            env_cfg.commands.racket_target.fh_target_conditioning_delta_vy = float(args.condition_fh_dvy)
            env_cfg.commands.racket_target.fh_target_conditioning_vx_max = (
                None if args.condition_fh_vx_max is None else float(args.condition_fh_vx_max)
            )
        if args.paired_recipe_mode != "off":
            if not args.paired_recipe_path:
                raise ValueError("--paired-recipe-path is required with --paired-recipe-mode")
            env_cfg.commands.racket_target.paired_recipe_mode = str(args.paired_recipe_mode)
            env_cfg.commands.racket_target.paired_recipe_path = str(args.paired_recipe_path)
            env_cfg.commands.racket_target.paired_recipe_source_env_id = args.paired_recipe_source_env_id
            env_cfg.commands.racket_target.paired_recipe_source_env_ids = paired_recipe_source_env_ids
            env_cfg.commands.racket_target.paired_recipe_strict = not bool(
                args.paired_recipe_nonstrict
            )
            applied_overrides.extend(
                [
                    f"commands.racket_target.paired_recipe_mode = {args.paired_recipe_mode!r}",
                    f"commands.racket_target.paired_recipe_path = {args.paired_recipe_path!r}",
                    "commands.racket_target.paired_recipe_source_env_id = "
                    f"{args.paired_recipe_source_env_id!r}",
                    "commands.racket_target.paired_recipe_source_env_ids = "
                    f"{paired_recipe_source_env_ids!r}",
                    "commands.racket_target.paired_recipe_strict = "
                    f"{not bool(args.paired_recipe_nonstrict)!r}",
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
        if args.snapshot_branch_mode:
            # A snapshot branch must differ only in the transplanted physical state.  Random
            # mid-swing clip switches are an unrelated topology intervention and can make the
            # paired event cease to be the normal event-1 wrap in only one branch.
            env_cfg.commands.motion.clip_switch_prob = 0.0
            applied_overrides.append("commands.motion.clip_switch_prob = 0.0")
            # The five in-process branches must share the same plant.  Startup/reset domain
            # randomization is per environment (mass, friction/CoM, joint defaults, PD gains),
            # so copying articulation state alone still leaves different dynamics and makes an
            # action replay branch reset for reasons unrelated to the intervention.  This is a
            # diagnostic-only nominal-plant override; formal evaluation/training DR is untouched.
            for event_name in (
                "physics_material",
                "add_joint_default_pos",
                "base_com",
                "randomize_link_mass",
                "randomize_pd_gains",
            ):
                if hasattr(env_cfg.events, event_name):
                    setattr(env_cfg.events, event_name, None)
                    applied_overrides.append(f"events.{event_name} = None (snapshot nominal plant)")
            policy_obs_cfg = getattr(env_cfg.observations, "policy", None)
            if policy_obs_cfg is not None and hasattr(policy_obs_cfg, "enable_corruption"):
                policy_obs_cfg.enable_corruption = False
                applied_overrides.append("observations.policy.enable_corruption = False (snapshot deterministic obs)")
        print(
            f"[evaluate] applied {len(applied_overrides)} training task override(s)",
            flush=True,
        )
        clips = [
            _resolve_motion_path(c)
            for c in (args.motion_file, args.motion_file_2, args.motion_file_3)
            if c
        ]
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
        base_env._hope_stance_curriculum_iteration = checkpoint_iteration
        print(
            "[evaluate] curriculum iteration aligned to checkpoint: "
            f"{checkpoint_iteration}",
            flush=True,
        )
        env = RslRlVecEnvWrapper(env)

        agent_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(load_ppo_params(args.algo_config), args.experiment_name)
        )
        agent_cfg.device = args.device
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
        policy_module = getattr(getattr(runner, "alg", None), "policy", None)

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
        reset_events = []
        post_strike_state_rows = []
        post_offsets_steps = tuple(
            max(1, int(round(offset / float(base_env.step_dt))))
            for offset in post_strike_offsets
        )
        post_pending = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        post_source_step = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        post_source_recipe_index = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        post_source_episode = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        post_source_strike = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        post_source_clip = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        post_source_corrected = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        post_captured_mask = torch.zeros(
            int(args.num_envs), dtype=torch.long, device=base_env.device
        )
        post_last_reset = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        transplant_events = []
        snapshot_branch_labels = {
            0: "G0",
            1: "G-ang",
            2: "G-lin",
            3: "G-upper",
            4: "G-allvel",
        }
        snapshot_branch_events = []
        snapshot_branch_triggered = False
        snapshot_blend_remaining = torch.zeros(
            int(args.num_envs), dtype=torch.long, device=base_env.device
        )
        snapshot_blend_rows = {}
        snapshot_action_replay_remaining = torch.zeros(
            int(args.num_envs), dtype=torch.long, device=base_env.device
        )
        snapshot_action_replay_rows = {}
        snapshot_action_replay_schedule = {}
        snapshot_action_replay_start_step = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        # In snapshot-branch mode every branch must contribute exactly one row for
        # the fixed next-shot recipe.  Exact-strike telemetry and MISS_CAPTURE
        # fallback rows share this guard so a branch cannot be silently omitted or
        # counted twice when its time-to-strike edge is not observed.
        snapshot_event1_logged = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        telemetry_episode_ids = torch.zeros(
            int(args.num_envs), dtype=torch.long, device=base_env.device
        )
        telemetry_strike_indices = torch.zeros(
            int(args.num_envs), dtype=torch.long, device=base_env.device
        )
        # Strike-clock telemetry.  The first strike after reset has no interval;
        # subsequent strikes in the same episode receive the measured simulator-time
        # delta rather than an inferred command hold value.
        last_strike_global_step = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        strike_interval_steps = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        # READY recovery audit state. A cycle begins at an exact strike and is resolved at the
        # next exact strike or an intervening reset. Normal time limits are not failures by
        # themselves; only a missing strict READY pass or a reset before it is counted.
        ready_pending = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        ready_cycle_passed = torch.zeros(
            int(args.num_envs), dtype=torch.bool, device=base_env.device
        )
        ready_cycle_start_step = torch.full(
            (int(args.num_envs),), -1, dtype=torch.long, device=base_env.device
        )
        ready_deadline_gate_steps = 0
        ready_deadline_strict_pass_steps = 0
        ready_cycles_completed = 0
        ready_cycles_passed = 0
        ready_cycles_failed = 0
        ready_cycles_failed_by_reset = 0
        ready_recovery_times_steps = []

        # Optional per-cycle READY diagnosis.  This is deliberately kept out of the normal
        # evaluator path: it stores sparse recovery traces and deadline snapshots only when the
        # caller explicitly requests --ready-telemetry-out.
        ready_cycle_states = [None for _ in range(int(args.num_envs))]
        ready_cycle_records = []
        ready_next_cycle_id = 0
        ready_trace_offsets_steps = tuple(
            int(round(offset / float(base_env.step_dt))) for offset in ready_telemetry_offsets
        ) if args.ready_telemetry_out else ()
        ready_trace_offset_seconds = (
            {steps: float(offset) for steps, offset in zip(ready_trace_offsets_steps, ready_telemetry_offsets)}
            if args.ready_telemetry_out
            else {}
        )
        ready_component_names = (
            "position",
            "planar_velocity",
            "heading",
            "yaw_rate",
            "tilt",
            "joint_velocity",
        )
        ready_component_metric_names = {
            "position": "position",
            "planar_velocity": "speed",
            "heading": "heading",
            "yaw_rate": "yaw_rate",
            "tilt": "tilt",
            "joint_velocity": "joint_speed",
            "foot_slip": "foot_slip",
        }
        ready_support_names = (
            "left_contact",
            "right_contact",
            "double_support",
            "left_slip",
            "right_slip",
        )

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

        def read_robot_state():
            """Return deploy-relevant robot state at a strike for carry-over auditing."""
            robot = getattr(cmd, "robot", None)
            data = getattr(robot, "data", None)
            if data is None:
                return (None,) * 6
            return tuple(
                _first_attr(data, [name])
                for name in (
                    "root_pos_w",
                    "root_quat_w",
                    "root_lin_vel_w",
                    "root_ang_vel_w",
                    "joint_pos",
                    "joint_vel",
                )
            )

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

        transplant_fields = {
            item.strip() for item in str(args.state_transplant_fields).split(",") if item.strip()
        }
        allowed_transplant_fields = {
            "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"
        }
        unknown_transplant_fields = transplant_fields - allowed_transplant_fields
        if unknown_transplant_fields:
            raise ValueError(
                "unknown --state-transplant-fields: "
                + ", ".join(sorted(unknown_transplant_fields))
            )

        def is_fh_correction_applied(e, clip_id, venue_selected, incoming_v):
            if clip_id is None or incoming_v is None:
                return False
            if int(clip_id[e].detach().cpu().item()) != 0 or _bool(venue_selected, e):
                return False
            if not bool(getattr(cmd.cfg, "fh_target_conditioning", False)):
                return False
            vx_max = getattr(cmd.cfg, "fh_target_conditioning_vx_max", None)
            return vx_max is None or float(incoming_v[e, 0].detach().float().cpu().item()) <= float(vx_max)

        def apply_state_transplant(env_ids, source_rows):
            """Apply selected baseline state fields and return refreshed policy observations."""
            if not source_rows:
                return None
            robot = getattr(cmd, "robot", None)
            if robot is None or not hasattr(robot, "data"):
                raise RuntimeError("state transplant requires racket_target.robot articulation access")
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
            # Isaac Lab exposes these buffers as inference tensors.  Allocate
            # ordinary writable tensors explicitly before editing/transplanting.
            root_state = torch.empty(
                robot.data.root_state_w[ids].shape,
                dtype=robot.data.root_state_w.dtype,
                device=base_env.device,
            )
            root_state.copy_(robot.data.root_state_w[ids])
            joint_pos = torch.empty(
                robot.data.joint_pos[ids].shape,
                dtype=robot.data.joint_pos.dtype,
                device=base_env.device,
            )
            joint_pos.copy_(robot.data.joint_pos[ids])
            joint_vel = torch.empty(
                robot.data.joint_vel[ids].shape,
                dtype=robot.data.joint_vel.dtype,
                device=base_env.device,
            )
            joint_vel.copy_(robot.data.joint_vel[ids])
            for local, env_id in enumerate(env_ids):
                row = source_rows[env_id]
                origin = env_origins[env_id]
                if "root_pos" in transplant_fields:
                    root_state[local, :3] = origin + torch.as_tensor(
                        row["robot_root_pos_env"], dtype=torch.float32, device=base_env.device
                    )
                if "root_quat" in transplant_fields:
                    root_state[local, 3:7] = torch.as_tensor(
                        row["robot_root_quat_w"], dtype=torch.float32, device=base_env.device
                    )
                if "root_lin_vel" in transplant_fields:
                    root_state[local, 7:10] = torch.as_tensor(
                        row["robot_root_lin_vel_w"], dtype=torch.float32, device=base_env.device
                    )
                if "root_ang_vel" in transplant_fields:
                    root_state[local, 10:13] = torch.as_tensor(
                        row["robot_root_ang_vel_w"], dtype=torch.float32, device=base_env.device
                    )
                if "joint_pos" in transplant_fields:
                    joint_pos[local] = torch.as_tensor(
                        row["robot_joint_pos"], dtype=torch.float32, device=base_env.device
                    )
                if "joint_vel" in transplant_fields:
                    joint_vel[local] = torch.as_tensor(
                        row["robot_joint_vel"], dtype=torch.float32, device=base_env.device
                    )
            with torch.inference_mode():
                robot.write_root_state_to_sim(root_state, env_ids=ids)
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
                base_env.sim.forward()
            return base_env.observation_manager.compute()["policy"]

        def clone_snapshot_branches(template_env_id, branch_env_ids, baseline_row):
            """Clone one live simulator state, then apply isolated velocity interventions.

            All branches receive the same gate root pose, joint pose, and velocity state first.
            The baseline telemetry row is then used only for the requested velocity fields.  This
            deliberately avoids a second simulator process and keeps the next recipe event common
            to every branch.
            """
            robot = getattr(cmd, "robot", None)
            if robot is None or not hasattr(robot, "data"):
                raise RuntimeError("snapshot branch mode requires racket_target.robot articulation access")
            branch_env_ids = [int(value) for value in branch_env_ids]
            ids = torch.as_tensor(branch_env_ids, dtype=torch.long, device=base_env.device)
            source_origin = env_origins[int(template_env_id)]
            source_root = robot.data.root_state_w[int(template_env_id)]
            source_joint_pos = robot.data.joint_pos[int(template_env_id)]
            source_joint_vel = robot.data.joint_vel[int(template_env_id)]
            root_state = torch.empty(
                (len(branch_env_ids),) + tuple(robot.data.root_state_w.shape[1:]),
                dtype=robot.data.root_state_w.dtype,
                device=base_env.device,
            )
            joint_pos = torch.empty(
                (len(branch_env_ids),) + tuple(robot.data.joint_pos.shape[1:]),
                dtype=robot.data.joint_pos.dtype,
                device=base_env.device,
            )
            joint_vel = torch.empty(
                (len(branch_env_ids),) + tuple(robot.data.joint_vel.shape[1:]),
                dtype=robot.data.joint_vel.dtype,
                device=base_env.device,
            )
            root_state[:] = source_root.detach()
            root_state[:, :3] = (
                env_origins[ids]
                + (source_root[:3].detach() - source_origin)
            )
            joint_pos[:] = source_joint_pos.detach()
            joint_vel[:] = source_joint_vel.detach()

            field_sets = {
                1: {"root_ang_vel"},
                2: {"root_lin_vel"},
                3: {"joint_vel"},
                4: {"root_lin_vel", "root_ang_vel", "joint_vel"},
            }
            for local, env_id in enumerate(branch_env_ids):
                fields = field_sets.get(env_id, set())
                if (
                    "root_lin_vel" in fields
                    and int(args.snapshot_branch_blend_steps) <= 0
                    and int(args.snapshot_action_replay_steps) <= 0
                    and not args.snapshot_action_replay_sequence
                ):
                    root_state[local, 7:10] = torch.as_tensor(
                        baseline_row["robot_root_lin_vel_w"],
                        dtype=torch.float32,
                        device=base_env.device,
                    )
                if (
                    "root_ang_vel" in fields
                    and int(args.snapshot_branch_blend_steps) <= 0
                    and int(args.snapshot_action_replay_steps) <= 0
                    and not args.snapshot_action_replay_sequence
                ):
                    root_state[local, 10:13] = torch.as_tensor(
                        baseline_row["robot_root_ang_vel_w"],
                        dtype=torch.float32,
                        device=base_env.device,
                    )
                if (
                    "joint_vel" in fields
                    and int(args.snapshot_branch_blend_steps) <= 0
                    and int(args.snapshot_action_replay_steps) <= 0
                    and not args.snapshot_action_replay_sequence
                ):
                    joint_vel[local] = torch.as_tensor(
                        baseline_row["robot_joint_vel"],
                        dtype=torch.float32,
                        device=base_env.device,
                    )
            with torch.inference_mode():
                robot.write_root_state_to_sim(root_state, env_ids=ids)
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
                base_env.sim.forward()
            # The first implementation cloned only the articulation.  Parallel evaluation envs
            # can be at different mocap phases even when paired recipe events are identical, so
            # their next strike clocks would not be common.  Copy the motion-clock state as part
            # of the snapshot as well; this is what makes the next event a genuinely shared shot.
            with torch.inference_mode():
                for attr_name in (
                    "time_steps",
                    "clip_id",
                    "hold_counter",
                    "just_resampled",
                    "_prev_clip_id",
                    "eval_sequence_index",
                    "_eval_sequence_counter",
                ):
                    source_value = getattr(motion_cmd, attr_name, None)
                    if not torch.is_tensor(source_value) or source_value.ndim == 0:
                        continue
                    if source_value.shape[0] != int(args.num_envs):
                        continue
                    writable = torch.empty_like(source_value[ids])
                    writable.copy_(source_value[int(template_env_id)])
                    source_value[ids] = writable
                # The racket-target command keeps a second per-env state machine on top of the
                # motion clock.  Copy its live question and replay cursors as well; otherwise a
                # branch whose pre-snapshot resample happened later can still emit the old event
                # after the physical/motion snapshot, defeating the fixed-next-shot design.
                for attr_name in (
                    "racket_target_pos_w",
                    "racket_target_vel_w",
                    "racket_target_normal_w",
                    "base_target_pos_w",
                    "swing_sign",
                    "vb_vel_in_w",
                    "vb_spin_in_w",
                    "_venue_tuple_selected",
                    "_venue_intended_landing_xy",
                    "_venue_outgoing_velocity_seed",
                    "_venue_tuple_outcome_pending",
                    "_venue_tuple_outcome_clip",
                    "_venue_planner_contact_normal_w",
                    "time_to_strike",
                    "pre_strike",
                    "strike_window",
                    "_paired_recipe_index",
                    "_paired_recipe_current_index",
                    "_paired_recipe_current_is_wrap",
                    "_prev_motion_steps",
                ):
                    source_value = getattr(cmd, attr_name, None)
                    if not torch.is_tensor(source_value) or source_value.ndim == 0:
                        continue
                    if source_value.shape[0] != int(args.num_envs):
                        continue
                    writable = torch.empty_like(source_value[ids])
                    writable.copy_(source_value[int(template_env_id)])
                    source_value[ids] = writable
                # Absolute world-frame command/mocap positions must be translated to each
                # destination environment.  The parallel environments have different origins;
                # copying these tensors verbatim creates the ~2.5 m target-observation mismatch
                # that invalidates an otherwise identical snapshot branch.
                origin_delta = env_origins[ids] - source_origin

                def _translate_env_position(name):
                    value = getattr(cmd, name, None)
                    if not torch.is_tensor(value) or value.ndim == 0:
                        return
                    if value.shape[0] != int(args.num_envs) or value.shape[-1] < 2:
                        return
                    copied = value[int(template_env_id)].detach().clone()
                    copied = copied.unsqueeze(0).expand(len(branch_env_ids), *copied.shape).clone()
                    copied[..., :2] += origin_delta[:, :2]
                    if copied.shape[-1] >= 3:
                        copied[..., 2] += origin_delta[:, 2]
                    value[ids] = copied

                for attr_name in (
                    "racket_target_pos_w",
                    "base_target_pos_w",
                    "delayed_racket_target_pos_w",
                    "_held_pos",
                    "_base_mocap_last_received_pos",
                    "_actor_base_pos_w",
                ):
                    _translate_env_position(attr_name)
                delay_buf = getattr(cmd, "_base_mocap_delay_buf", None)
                if (
                    torch.is_tensor(delay_buf)
                    and delay_buf.ndim >= 3
                    and delay_buf.shape[1] == int(args.num_envs)
                ):
                    copied = delay_buf[:, int(template_env_id)].detach().clone()
                    copied = copied.unsqueeze(1).expand(
                        copied.shape[0], len(branch_env_ids), *copied.shape[1:]
                    ).clone()
                    copied[..., :2] += origin_delta[:, :2]
                    if copied.shape[-1] >= 3:
                        copied[..., 2] += origin_delta[:, 2]
                    delay_buf[:, ids] = copied
                for attr_name in ("_actor_base_quat_w",):
                    value = getattr(cmd, attr_name, None)
                    if (
                        torch.is_tensor(value)
                        and value.ndim >= 2
                        and value.shape[0] == int(args.num_envs)
                    ):
                        value[ids] = value[int(template_env_id)].detach().clone()

                if hasattr(cmd, "_paired_recipe_current"):
                    source_event = cmd._paired_recipe_current[int(template_env_id)]
                    for env_id in branch_env_ids:
                        cmd._paired_recipe_current[env_id] = source_event
                # The policy observation also contains manager-side temporal state.  Copy the
                # action manager history and episode/reset buffers so an otherwise identical
                # branch is not reset merely because its pre-snapshot env had a different age or
                # previous action.
                action_manager = getattr(base_env, "action_manager", None)
                if action_manager is not None:
                    for attr_name in ("_action", "_prev_action"):
                        source_value = getattr(action_manager, attr_name, None)
                        if not torch.is_tensor(source_value) or source_value.ndim == 0:
                            continue
                        if source_value.shape[0] != int(args.num_envs):
                            continue
                        writable = torch.empty_like(source_value[ids])
                        writable.copy_(source_value[int(template_env_id)])
                        source_value[ids] = writable
                    action_term = action_manager.get_term("joint_pos")
                    capture = getattr(action_term, "capture_markov_replay_state", None)
                    restore = getattr(action_term, "restore_markov_replay_state", None)
                    if callable(capture) and callable(restore):
                        source_action_state = capture(
                            torch.as_tensor(
                                [int(template_env_id)],
                                dtype=torch.long,
                                device=base_env.device,
                            )
                        )
                        source_action_state = {
                            name: (
                                value.expand(len(branch_env_ids), *value.shape[1:]).clone()
                                if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == 1
                                else value
                            )
                            for name, value in source_action_state.items()
                        }
                        restore(ids, source_action_state)
                    else:
                        # Compatibility fallback for older action terms without the explicit
                        # replay contract.  These are the buffers consumed by applied_last_action.
                        for attr_name in (
                            "_raw_actions",
                            "_applied_raw_actions",
                            "_unclamped_processed_actions",
                            "_processed_actions",
                            "_executed_feedback",
                        ):
                            source_value = getattr(action_term, attr_name, None)
                            if (
                                torch.is_tensor(source_value)
                                and source_value.ndim >= 2
                                and source_value.shape[0] == int(args.num_envs)
                            ):
                                source_value[ids] = source_value[int(template_env_id)].detach().clone()
                for owner, attr_name in (
                    (base_env, "episode_length_buf"),
                    (base_env, "reset_buf"),
                    (getattr(base_env, "termination_manager", None), "terminated"),
                    (getattr(base_env, "termination_manager", None), "time_outs"),
                ):
                    source_value = getattr(owner, attr_name, None) if owner is not None else None
                    if not torch.is_tensor(source_value) or source_value.ndim == 0:
                        continue
                    if source_value.shape[0] != int(args.num_envs):
                        continue
                    writable = torch.empty_like(source_value[ids])
                    writable.copy_(source_value[int(template_env_id)])
                    source_value[ids] = writable
            if int(args.snapshot_branch_blend_steps) > 0:
                for env_id in branch_env_ids:
                    snapshot_blend_rows[env_id] = baseline_row
                    snapshot_blend_remaining[env_id] = int(args.snapshot_branch_blend_steps)
            if int(args.snapshot_action_replay_steps) > 0:
                if baseline_row.get("policy_action") is None:
                    raise RuntimeError(
                        "snapshot action replay requires baseline telemetry with policy_action"
                    )
                for env_id in branch_env_ids:
                    snapshot_action_replay_rows[env_id] = baseline_row
                    snapshot_action_replay_remaining[env_id] = int(args.snapshot_action_replay_steps)
            if args.snapshot_action_replay_sequence:
                recipe_index = int(baseline_row["source_paired_recipe_index"])
                sequence = {}
                for (row_recipe, row_offset), row in snapshot_action_rows_by_recipe_offset.items():
                    if row_recipe != recipe_index or row.get("policy_action") is None:
                        continue
                    relative_s = float(row_offset) - float(args.snapshot_branch_offset)
                    if relative_s < -1.0e-6:
                        continue
                    relative_step = int(round(relative_s / float(base_env.step_dt)))
                    sequence[relative_step] = row
                if 0 not in sequence and baseline_row.get("policy_action") is not None:
                    sequence[0] = baseline_row
                for env_id in branch_env_ids:
                    snapshot_action_replay_schedule[env_id] = sequence
                    snapshot_action_replay_start_step[env_id] = int(step_index)
            # Keep evaluator-side strike edge detection and event counters aligned with the
            # cloned simulator clock.  Without this, a branch can report the same recipe event
            # at a different external edge even though its command state was synchronized.
            live_tts = read_state()[3]
            prev_tts[ids] = live_tts[int(template_env_id)]
            episode_value = telemetry_episode_ids[int(template_env_id)].clone()
            strike_value = telemetry_strike_indices[int(template_env_id)].clone()
            telemetry_episode_ids[ids] = episode_value
            telemetry_strike_indices[ids] = strike_value
            branch_obs = base_env.observation_manager.compute()["policy"]
            obs_diffs = [
                float(
                    torch.max(
                        torch.abs(branch_obs[env_id] - branch_obs[int(template_env_id)])
                    ).detach().cpu().item()
                )
                for env_id in branch_env_ids
            ]
            obs_term_diffs = []
            obs_manager = base_env.observation_manager
            term_names = getattr(obs_manager, "_group_obs_term_names", {}).get("policy", [])
            term_dims = getattr(obs_manager, "_group_obs_term_dim", {}).get("policy", [])
            term_ranges = []
            cursor = 0
            for name, dims in zip(term_names, term_dims):
                width = 1
                for dim in dims:
                    width *= int(dim)
                term_ranges.append((str(name), cursor, cursor + width))
                cursor += width
            for env_id in branch_env_ids:
                obs_term_diffs.append(
                    {
                        name: float(
                            torch.max(
                                torch.abs(
                                    branch_obs[env_id, start:end]
                                    - branch_obs[int(template_env_id), start:end]
                                )
                            ).detach().cpu().item()
                        )
                        for name, start, end in term_ranges
                    }
                )
            snapshot_branch_events.append(
                {
                    "template_env_id": int(template_env_id),
                    "branch_env_ids": branch_env_ids,
                    "branch_labels": [snapshot_branch_labels[e] for e in branch_env_ids],
                    "baseline_source_env_id": int(args.snapshot_branch_source_env_id),
                    "baseline_source_paired_recipe_indices": sorted(snapshot_baseline_rows),
                    "snapshot_offset_s": float(args.snapshot_branch_offset),
                    "motion_clock_synced": True,
                    "intervention_mode": (
                        "action_replay_sequence"
                        if args.snapshot_action_replay_sequence
                        else
                        "action_replay"
                        if int(args.snapshot_action_replay_steps) > 0
                        else "smooth_velocity_blend"
                        if int(args.snapshot_branch_blend_steps) > 0
                        else "instantaneous_velocity_transplant"
                    ),
                    "blend_steps": int(args.snapshot_branch_blend_steps),
                    "action_replay_steps": int(args.snapshot_action_replay_steps),
                    "action_replay_sequence": bool(args.snapshot_action_replay_sequence),
                    "policy_obs_max_abs_diff_vs_G0": obs_diffs,
                    "policy_obs_term_max_abs_diff_vs_G0": obs_term_diffs,
                    "fields_by_branch": {
                        snapshot_branch_labels[0]: [],
                        snapshot_branch_labels[1]: ["root_ang_vel"],
                        snapshot_branch_labels[2]: ["root_lin_vel"],
                        snapshot_branch_labels[3]: ["joint_vel"],
                        snapshot_branch_labels[4]: ["root_lin_vel", "root_ang_vel", "joint_vel"],
                    },
                }
            )
            return branch_obs

        def apply_snapshot_velocity_blend():
            """Move branch velocity state toward the baseline over several control steps."""
            active_ids = snapshot_blend_remaining.nonzero(as_tuple=False).flatten().tolist()
            if not active_ids:
                return None
            robot = getattr(cmd, "robot", None)
            if robot is None or not hasattr(robot, "data"):
                raise RuntimeError("snapshot velocity blending requires racket_target.robot articulation access")
            ids = torch.as_tensor(active_ids, dtype=torch.long, device=base_env.device)
            root_state = torch.empty_like(robot.data.root_state_w[ids])
            root_state.copy_(robot.data.root_state_w[ids])
            joint_vel = torch.empty_like(robot.data.joint_vel[ids])
            joint_vel.copy_(robot.data.joint_vel[ids])
            field_sets = {
                1: {"root_ang_vel"},
                2: {"root_lin_vel"},
                3: {"joint_vel"},
                4: {"root_lin_vel", "root_ang_vel", "joint_vel"},
            }
            for local, env_id in enumerate(active_ids):
                row = snapshot_blend_rows[env_id]
                fields = field_sets.get(env_id, set())
                alpha = 1.0 / float(max(int(snapshot_blend_remaining[env_id].item()), 1))
                if "root_lin_vel" in fields:
                    target = torch.as_tensor(row["robot_root_lin_vel_w"], dtype=root_state.dtype, device=base_env.device)
                    root_state[local, 7:10] += alpha * (target - root_state[local, 7:10])
                if "root_ang_vel" in fields:
                    target = torch.as_tensor(row["robot_root_ang_vel_w"], dtype=root_state.dtype, device=base_env.device)
                    root_state[local, 10:13] += alpha * (target - root_state[local, 10:13])
                if "joint_vel" in fields:
                    target = torch.as_tensor(row["robot_joint_vel"], dtype=joint_vel.dtype, device=base_env.device)
                    joint_vel[local] += alpha * (target - joint_vel[local])
            with torch.inference_mode():
                robot.write_root_state_to_sim(root_state, env_ids=ids)
                robot.write_joint_state_to_sim(robot.data.joint_pos[ids], joint_vel, env_ids=ids)
                base_env.sim.forward()
            snapshot_blend_remaining[ids] -= 1
            return base_env.observation_manager.compute()["policy"]

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

        def _ready_metric(name, env_id, default=0.0):
            value = getattr(cmd, "metrics", {}).get(name)
            if value is not None:
                scalar = _scalar(value, env_id)
                # The reset-time friction event owns the authoritative sampled values.  During
                # evaluation the command metric buffers can be reinitialized after the event
                # publishes them, so fall back to the event instance rather than exporting a
                # misleading zero coefficient.
                if name in {
                    "friction_mu_static",
                    "friction_mu_dynamic",
                    "friction_beta",
                } and abs(scalar) <= 1.0e-12:
                    event_attr = {
                        "friction_mu_static": "mu_static",
                        "friction_mu_dynamic": "mu_dynamic",
                        "friction_beta": "beta",
                    }[name]
                    try:
                        term_cfg = base_env.event_manager.get_term_cfg("physics_material")
                        event = getattr(term_cfg, "func", None)
                        tensor = getattr(event, event_attr, None)
                        if tensor is not None:
                            return _scalar(tensor, env_id)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                return scalar
            if name in {
                "friction_mu_static",
                "friction_mu_dynamic",
                "friction_beta",
            }:
                event_attr = {
                    "friction_mu_static": "mu_static",
                    "friction_mu_dynamic": "mu_dynamic",
                    "friction_beta": "beta",
                }[name]
                try:
                    term_cfg = base_env.event_manager.get_term_cfg("physics_material")
                    event = getattr(term_cfg, "func", None)
                    tensor = getattr(event, event_attr, None)
                    if tensor is not None:
                        return _scalar(tensor, env_id)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            return float(default)

        def _ready_snapshot(env_id):
            """Capture deploy-facing READY components and continuous values for one env."""
            x_error = _ready_metric("ready_station_x_error", env_id)
            y_error = _ready_metric("ready_station_y_error", env_id)
            base_speed = _ready_metric("ready_station_base_speed", env_id)
            heading_deg = _ready_metric("ready_station_heading_error_deg", env_id)
            yaw_rate = _ready_metric("ready_station_yaw_rate_abs", env_id)
            tilt = _ready_metric("ready_station_tilt", env_id)
            joint_speed = _ready_metric("ready_station_joint_speed", env_id)
            component_pass = {
                "position": abs(x_error) <= 0.10 and abs(y_error) <= 0.10,
                "planar_velocity": base_speed <= 0.20,
                "heading": heading_deg <= 15.0,
                "yaw_rate": yaw_rate <= 0.35,
                "tilt": tilt <= 0.14,
                "joint_velocity": joint_speed <= 0.80,
            }
            all_pass = all(component_pass.get(name, False) for name in ready_component_names)

            left_contact = _ready_metric("ready_left_contact_rate", env_id)
            right_contact = _ready_metric("ready_right_contact_rate", env_id)
            double_support = _ready_metric("ready_double_support_rate", env_id)
            left_slip = _ready_metric("ready_left_foot_slip", env_id)
            right_slip = _ready_metric("ready_right_foot_slip", env_id)
            # The existing READY/settle contract uses 3 cm as the foot-slip margin.  Keep the
            # threshold explicit in the output so downstream analysis cannot confuse it with the
            # deploy monitor's disabled default foot-slip field.
            slip_threshold = float(getattr(cmd.cfg, "step_settle_slip_thresh", 0.03))
            support_pass = {
                "left_contact": left_contact > 0.5,
                "right_contact": right_contact > 0.5,
                "double_support": double_support > 0.5,
                "left_slip": left_slip <= slip_threshold,
                "right_slip": right_slip <= slip_threshold,
            }

            # The reward-side stance/support metrics are intentionally phase-gated.  At the
            # exact deadline that gate can be closed even though the robot is physically in
            # contact, which would turn an unobserved metric into a false support failure.  For
            # audit telemetry, read the two foot links and contact-force sensor directly.
            robot = getattr(cmd, "robot", None)
            try:
                foot_names = ("left_ankle_roll_Link", "right_ankle_roll_Link")
                robot_body_names = tuple(getattr(robot, "body_names", ()))
                robot_foot_ids = [robot_body_names.index(name) for name in foot_names]
                foot_pos = robot.data.body_pos_w[env_id, robot_foot_ids, :2]
                foot_vel = robot.data.body_lin_vel_w[env_id, robot_foot_ids, :2]
                direct_width = float(torch.linalg.norm(foot_pos[0] - foot_pos[1]).detach().cpu().item())
                direct_slip = torch.linalg.norm(foot_vel, dim=-1)
                direct_left_slip = float(direct_slip[0].detach().cpu().item())
                direct_right_slip = float(direct_slip[1].detach().cpu().item())
                sensor = base_env.scene.sensors["contact_forces"]
                sensor_body_names = tuple(getattr(sensor, "body_names", ())) if sensor is not None else ()
                sensor_ids = [sensor_body_names.index(name) for name in foot_names]
                force_vec = sensor.data.net_forces_w[env_id, sensor_ids, :]
                force_mag = torch.linalg.norm(force_vec, dim=-1)
                direct_fz = torch.abs(force_vec[:, 2])
                direct_contact = force_mag > 10.0
                direct_left_contact = float(direct_contact[0].detach().cpu().item())
                direct_right_contact = float(direct_contact[1].detach().cpu().item())
                direct_both = float((direct_contact[0] & direct_contact[1]).detach().cpu().item())
                direct_left_fz = float(direct_fz[0].detach().cpu().item())
                direct_right_fz = float(direct_fz[1].detach().cpu().item())
                force_total = max(direct_left_fz + direct_right_fz, 1.0e-6)
                direct_load_left = direct_left_fz / force_total
                direct_load_right = direct_right_fz / force_total
                action_term = base_env.action_manager.get_term("joint_pos")
                alpha = float(action_term.stance_alpha())
                target_lo = (1.0 - alpha) * 0.25 + alpha * 0.45
                target_hi = (1.0 - alpha) * 0.35 + alpha * 0.55
                direct_width_error = max(target_lo - direct_width, 0.0) + max(direct_width - target_hi, 0.0)
                left_contact = direct_left_contact
                right_contact = direct_right_contact
                double_support = direct_both
                left_slip = direct_left_slip
                right_slip = direct_right_slip
                support_pass = {
                    "left_contact": left_contact > 0.5,
                    "right_contact": right_contact > 0.5,
                    "double_support": double_support > 0.5,
                    "left_slip": left_slip <= slip_threshold,
                    "right_slip": right_slip <= slip_threshold,
                }
            except (AttributeError, KeyError, ValueError, IndexError, TypeError):
                direct_width = _ready_metric("ready_stance_width", env_id)
                direct_width_error = _ready_metric("ready_stance_width_error", env_id)
                direct_left_fz = _ready_metric("ready_left_fz", env_id)
                direct_right_fz = _ready_metric("ready_right_fz", env_id)
                direct_load_left = _ready_metric("ready_load_ratio_left", env_id)
                direct_load_right = _ready_metric("ready_load_ratio_right", env_id)

            data = getattr(getattr(cmd, "robot", None), "data", None)
            quat = getattr(cmd, "base_quat_w", None)
            if quat is not None:
                q = quat[env_id]
                roll = torch.atan2(
                    2.0 * (q[0] * q[1] + q[2] * q[3]),
                    1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]),
                )
                pitch = torch.asin(
                    torch.clamp(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)
                )
                roll = float(roll.detach().cpu().item())
                pitch = float(pitch.detach().cpu().item())
            else:
                roll = pitch = None

            base_ang_vel = None
            joint_vel_max = None
            if data is not None:
                ang = getattr(data, "root_ang_vel_b", None)
                if ang is not None:
                    base_ang_vel = _tolist(ang, env_id)
                jv = getattr(data, "joint_vel", None)
                if jv is not None:
                    joint_vel_max = float(torch.abs(jv[env_id]).max().detach().cpu().item())

            values = {
                "position_error_x": x_error,
                "position_error_y": y_error,
                "base_xy_speed": base_speed,
                "heading_error_deg": heading_deg,
                "yaw_rate": yaw_rate,
                "tilt": tilt,
                "joint_velocity_rms": joint_speed,
                "joint_velocity_max": joint_vel_max,
                "stance_width": direct_width,
                "stance_width_error": direct_width_error,
                "left_fz": direct_left_fz,
                "right_fz": direct_right_fz,
                "load_ratio_left": direct_load_left,
                "load_ratio_right": direct_load_right,
                "left_foot_slip": left_slip,
                "right_foot_slip": right_slip,
                "roll_rad": roll,
                "pitch_rad": pitch,
                "base_angular_velocity_b": base_ang_vel,
                "mu_static": _ready_metric("friction_mu_static", env_id),
                "mu_dynamic": _ready_metric("friction_mu_dynamic", env_id),
                "friction_beta": _ready_metric("friction_beta", env_id),
            }
            return {
                "component_pass": component_pass,
                "support_pass": support_pass,
                "ready_all_pass": bool(all_pass),
                "values": values,
                "support_values": {
                    "left_contact_rate": left_contact,
                    "right_contact_rate": right_contact,
                    "double_support_rate": double_support,
                    "left_foot_slip": left_slip,
                    "right_foot_slip": right_slip,
                    "foot_slip_threshold": slip_threshold,
                },
            }

        def _residual_ratio(env_id, observation):
            """Return residual-mean / raw bound for one env when the policy exposes the contract."""
            if not args.ready_telemetry_out or policy_module is None:
                return None
            mean_components = getattr(policy_module, "_mean_components", None)
            bound = getattr(policy_module, "residual_bound_raw", None)
            active = getattr(policy_module, "residual_active_mask", None)
            if not callable(mean_components) or bound is None or active is None:
                return None
            try:
                _, residual_mean, _ = mean_components(observation)
                bound = bound.to(device=residual_mean.device, dtype=residual_mean.dtype)
                active = active.to(device=residual_mean.device)
                ratio = torch.zeros_like(residual_mean)
                valid = active & (bound > 1.0e-8)
                ratio[:, valid] = torch.abs(residual_mean[:, valid]) / bound[valid]
                return ratio[env_id].detach().cpu().tolist()
            except Exception:
                return None

        def _finalize_ready_cycle(env_id, terminal="next_strike"):
            nonlocal ready_cycle_states
            state = ready_cycle_states[env_id]
            if state is None:
                return
            deadline = state.get("deadline_step")
            first_ready = state.get("first_ready_step")
            stable_ready = state.get("stable_ready_step")
            if deadline is None:
                # The configured reward deadline is only active for its selected station step
                # class.  Such cycles remain useful for traces, but must not be classified as
                # late/never/unstable against a deadline that was never observed.
                classification = "deadline_not_observed"
            elif stable_ready is not None:
                classification = "on_time" if stable_ready <= deadline else "late"
            elif first_ready is None:
                classification = "never_ready"
            else:
                classification = "unstable_ready"
            state["terminal"] = terminal
            state["classification"] = classification
            state["first_ready_time_s"] = (
                (first_ready - state["strike_step"]) * float(base_env.step_dt)
                if first_ready is not None else None
            )
            state["stable_ready_time_s"] = (
                (stable_ready - state["strike_step"]) * float(base_env.step_dt)
                if stable_ready is not None else None
            )
            state["ready_deadline_time_s"] = (
                (deadline - state["strike_step"]) * float(base_env.step_dt)
                if deadline is not None else None
            )
            state["deadline_observed"] = deadline is not None
            deadline_snapshot = state.pop("deadline_snapshot", None)
            if deadline_snapshot is None:
                state["position_pass_at_deadline"] = None
                state["planar_velocity_pass_at_deadline"] = None
                state["heading_pass_at_deadline"] = None
                state["tilt_pass_at_deadline"] = None
                state["joint_velocity_pass_at_deadline"] = None
                state["left_contact_pass_at_deadline"] = None
                state["right_contact_pass_at_deadline"] = None
                state["double_support_pass_at_deadline"] = None
                state["left_slip_pass_at_deadline"] = None
                state["right_slip_pass_at_deadline"] = None
                state["ready_all_pass_at_deadline"] = None
                state["deadline_values"] = None
                state["deadline_support_values"] = None
            else:
                component_pass = deadline_snapshot["component_pass"]
                support_pass = deadline_snapshot["support_pass"]
                state["position_pass_at_deadline"] = component_pass.get("position")
                state["planar_velocity_pass_at_deadline"] = component_pass.get("planar_velocity")
                state["heading_pass_at_deadline"] = component_pass.get("heading")
                state["tilt_pass_at_deadline"] = component_pass.get("tilt")
                state["joint_velocity_pass_at_deadline"] = component_pass.get("joint_velocity")
                state["left_contact_pass_at_deadline"] = support_pass.get("left_contact")
                state["right_contact_pass_at_deadline"] = support_pass.get("right_contact")
                state["double_support_pass_at_deadline"] = support_pass.get("double_support")
                state["left_slip_pass_at_deadline"] = support_pass.get("left_slip")
                state["right_slip_pass_at_deadline"] = support_pass.get("right_slip")
                state["ready_all_pass_at_deadline"] = deadline_snapshot["ready_all_pass"]
                state["deadline_values"] = deadline_snapshot["values"]
                state["deadline_support_values"] = deadline_snapshot["support_values"]
                state["deadline_component_pass"] = component_pass
                state["deadline_support_pass"] = support_pass
            state["residual_saturation_fraction"] = (
                float(state["residual_saturated_samples"])
                / float(
                    max(
                        state["residual_samples"]
                        * int(getattr(policy_module, "residual_active_mask", torch.ones(31)).sum().item()),
                        1,
                    )
                )
            )
            ready_cycle_records.append(state)
            ready_cycle_states[env_id] = None

        obs, _ = env.get_observations()
        prev_tts = read_state()[3].clone()
        prev_actions = None
        for step_index in range(args.num_steps):
            # Capture post-strike state at fixed control-time offsets.  This runs before the next
            # action, so the state is the actual hand-off state seen by the policy.  A transplant,
            # when requested, is applied after recording the native gate state and observations are
            # recomputed before the next policy call.
            transplant_env_ids = []
            transplant_rows = {}
            if args.virtual_telemetry_out:
                current_target_pos, current_racket_pos, current_racket_vel, _, _ = read_state()
                for offset_index, (offset_s, offset_steps) in enumerate(
                    zip(post_strike_offsets, post_offsets_steps)
                ):
                    due = (
                        post_pending
                        & ((step_index - post_source_step) == int(offset_steps))
                        & ((post_captured_mask & (1 << offset_index)) == 0)
                    )
                    for env_id in due.nonzero(as_tuple=False).flatten().tolist():
                        root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel = read_robot_state()
                        row = {
                            "env_id": int(env_id),
                            "offset_s": float(offset_s),
                            "source_global_step": int(post_source_step[env_id].item()),
                            "source_episode_id": int(post_source_episode[env_id].item()),
                            "source_strike_index": int(post_source_strike[env_id].item()),
                            "source_paired_recipe_index": int(post_source_recipe_index[env_id].item()),
                            "source_clip_id": int(post_source_clip[env_id].item()),
                            "source_fh_correction_applied": bool(post_source_corrected[env_id].item()),
                            "robot_root_pos_env": _tolist(root_pos, env_id, local_position=True),
                            "robot_root_quat_w": _tolist(root_quat, env_id),
                            "robot_root_lin_vel_w": _tolist(root_lin_vel, env_id),
                            "robot_root_ang_vel_w": _tolist(root_ang_vel, env_id),
                            "robot_joint_pos": _tolist(joint_pos, env_id),
                            "robot_joint_vel": _tolist(joint_vel, env_id),
                            "racket_pos_env": _tolist(current_racket_pos, env_id, local_position=True),
                            "racket_velocity": _tolist(current_racket_vel, env_id),
                            "policy_action": _tolist(prev_actions, env_id),
                            "policy_action_norm": (
                                float(torch.linalg.vector_norm(prev_actions[env_id]).detach().cpu().item())
                                if prev_actions is not None
                                else None
                            ),
                        }
                        post_strike_state_rows.append(row)
                        post_captured_mask[env_id] |= 1 << offset_index
                        key = (
                            int(env_id),
                            int(post_source_recipe_index[env_id].item()),
                            round(float(offset_s), 6),
                        )
                        if key in transplant_records:
                            transplant_env_ids.append(int(env_id))
                            transplant_rows[int(env_id)] = transplant_records[key]
                    if (
                        args.snapshot_branch_mode
                        and not snapshot_branch_triggered
                        and round(float(offset_s), 6) == round(float(args.snapshot_branch_offset), 6)
                        and bool(due[0].item())
                        and bool(post_source_corrected[0].item())
                    ):
                        recipe_index = int(post_source_recipe_index[0].item())
                        baseline_row = snapshot_baseline_rows.get(recipe_index)
                        if baseline_row is None:
                            raise RuntimeError(
                                "snapshot baseline telemetry is missing recipe index "
                                f"{recipe_index} at offset {args.snapshot_branch_offset}"
                            )
                        # Clone env 0's live gate state to the four intervention branches.  The
                        # command replay is already replicated across envs; only physical state is
                        # changed here, so all five branches face the same next recipe event.
                        obs = clone_snapshot_branches(0, [1, 2, 3, 4], baseline_row)
                        snapshot_branch_triggered = True
            if args.snapshot_branch_mode and int(args.snapshot_branch_blend_steps) > 0:
                blended_obs = apply_snapshot_velocity_blend()
                if blended_obs is not None:
                    obs = blended_obs
            if transplant_env_ids:
                refreshed_obs = apply_state_transplant(transplant_env_ids, transplant_rows)
                if refreshed_obs is not None:
                    obs = refreshed_obs
                    for env_id in transplant_env_ids:
                        transplant_events.append(
                            {
                                "env_id": int(env_id),
                                "source_paired_recipe_index": int(
                                    post_source_recipe_index[env_id].item()
                                ),
                                "offset_s": float(args.state_transplant_offset),
                                "fields": sorted(transplant_fields),
                            }
                        )
            with torch.inference_mode():
                actions = policy(obs).clone()
                residual_ratio_batch = None
                if args.ready_telemetry_out and policy_module is not None:
                    mean_components = getattr(policy_module, "_mean_components", None)
                    bound = getattr(policy_module, "residual_bound_raw", None)
                    active = getattr(policy_module, "residual_active_mask", None)
                    if callable(mean_components) and bound is not None and active is not None:
                        try:
                            _, residual_mean, _ = mean_components(obs)
                            bound = bound.to(
                                device=residual_mean.device,
                                dtype=residual_mean.dtype,
                            )
                            active = active.to(device=residual_mean.device)
                            residual_ratio_batch = torch.zeros_like(residual_mean)
                            valid = active & (bound > 1.0e-8)
                            residual_ratio_batch[:, valid] = (
                                torch.abs(residual_mean[:, valid]) / bound[valid]
                            )
                        except Exception:
                            residual_ratio_batch = None
                fixed_replay_ids = (
                    (snapshot_action_replay_remaining > 0)
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                for env_id in fixed_replay_ids:
                    baseline_action = torch.as_tensor(
                        snapshot_action_replay_rows[env_id]["policy_action"],
                        dtype=actions.dtype,
                        device=actions.device,
                    )
                    actions[env_id] = baseline_action
                if args.snapshot_action_replay_sequence:
                    sequence_replay_ids = []
                    for env_id, schedule in snapshot_action_replay_schedule.items():
                        relative_step = int(step_index) - int(snapshot_action_replay_start_step[env_id].item())
                        row = schedule.get(relative_step)
                        if row is None:
                            continue
                        actions[env_id] = torch.as_tensor(
                            row["policy_action"], dtype=actions.dtype, device=actions.device
                        )
                        sequence_replay_ids.append(int(env_id))
                    replay_ids = sorted(set(fixed_replay_ids).union(sequence_replay_ids))
                else:
                    replay_ids = fixed_replay_ids
                obs, _, dones, _ = env.step(actions)
            if fixed_replay_ids:
                snapshot_action_replay_remaining[fixed_replay_ids] -= 1
            prev_actions = actions.detach().clone()
            target_pos, racket_pos, racket_vel, tts, swing = read_state()
            ready_metrics = getattr(cmd, "metrics", {})
            ready_gate_metric = ready_metrics.get("v11_ready_deadline_gate")
            ready_pass_metric = ready_metrics.get("ready_deadline_strict_pass")
            if ready_gate_metric is not None and ready_pass_metric is not None:
                ready_gate_now = ready_gate_metric > 0.5
                ready_pass_now = ready_gate_now & (ready_pass_metric > 0.5)
                ready_deadline_gate_steps += int(ready_gate_now.sum().item())
                ready_deadline_strict_pass_steps += int(ready_pass_now.sum().item())
                newly_ready = ready_pending & ready_pass_now & (~ready_cycle_passed)
                if bool(newly_ready.any()):
                    ready_recovery_times_steps.extend(
                        (
                            int(step_index) - ready_cycle_start_step[newly_ready]
                        ).to(dtype=torch.float32).cpu().tolist()
                    )
                    ready_cycle_passed[newly_ready] = True
            if args.ready_telemetry_out:
                ready_gate_value = ready_metrics.get("v11_ready_deadline_gate")
                ready_gate_now = (
                    ready_gate_value > 0.5
                    if ready_gate_value is not None
                    else torch.zeros(int(args.num_envs), dtype=torch.bool, device=base_env.device)
                )
                for env_id, state in enumerate(ready_cycle_states):
                    if state is None:
                        continue
                    snapshot = _ready_snapshot(env_id)
                    if snapshot["ready_all_pass"] and state["first_ready_step"] is None:
                        state["first_ready_step"] = int(step_index)
                    latched = _ready_metric("ready_station_latched", env_id) > 0.5
                    if latched and state["stable_ready_step"] is None:
                        state["stable_ready_step"] = int(step_index)
                    if bool(ready_gate_now[env_id].item()):
                        state["deadline_step"] = int(step_index)
                        state["deadline_snapshot"] = snapshot
                    relative_step = int(step_index) - int(state["strike_step"])
                    if relative_step in ready_trace_offset_seconds:
                        trace_row = {
                            "offset_s": ready_trace_offset_seconds[relative_step],
                            "strict_ready_sample": snapshot["ready_all_pass"],
                            "stable_ready": bool(latched),
                            "component_pass": snapshot["component_pass"],
                            "support_pass": snapshot["support_pass"],
                            "values": snapshot["values"],
                            "support_values": snapshot["support_values"],
                        }
                        state["trace"].append(trace_row)
                    if residual_ratio_batch is not None:
                        ratio = residual_ratio_batch[env_id].detach().cpu().tolist()
                        peak = state.get("residual_peak_ratio_31d")
                        if peak is None:
                            state["residual_peak_ratio_31d"] = ratio
                        else:
                            state["residual_peak_ratio_31d"] = [
                                max(float(old), float(new))
                                for old, new in zip(peak, ratio)
                            ]
                        state["residual_samples"] += 1
                        state["residual_saturated_samples"] += int(
                            sum(float(value) >= 0.95 for value in ratio)
                        )
            exact_strike_mask = torch.zeros(
                int(args.num_envs), dtype=torch.bool, device=base_env.device
            )
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
                strike_interval_steps.fill_(-1)
                strike_ids = exact_strike_mask.nonzero(as_tuple=False).flatten()
                if strike_ids.numel() > 0:
                    # Resolve the previous strike-to-READY cycle before opening the new one.
                    preceding = exact_strike_mask & ready_pending
                    if bool(preceding.any()):
                        ready_cycles_completed += int(preceding.sum().item())
                        ready_cycles_passed += int((preceding & ready_cycle_passed).sum().item())
                        ready_cycles_failed += int((preceding & (~ready_cycle_passed)).sum().item())
                    if args.ready_telemetry_out:
                        for env_id in strike_ids.tolist():
                            _finalize_ready_cycle(env_id, terminal="next_strike")
                    ready_pending[strike_ids] = True
                    ready_cycle_passed[strike_ids] = False
                    ready_cycle_start_step[strike_ids] = int(step_index)
                    if args.ready_telemetry_out:
                        for env_id in strike_ids.tolist():
                            ready_next_cycle_id += 1
                            ready_cycle_states[env_id] = {
                                "cycle_id": int(ready_next_cycle_id),
                                "env_id": int(env_id),
                                "checkpoint": os.path.basename(checkpoint),
                                "strike_end_time_s": float(step_index * base_env.step_dt),
                                "strike_step": int(step_index),
                                "ready_deadline_time_s": None,
                                "first_ready_time_s": None,
                                "stable_ready_time_s": None,
                                "deadline_step": None,
                                "first_ready_step": None,
                                "stable_ready_step": None,
                                "deadline_snapshot": None,
                                "trace": [],
                                "terminal": None,
                                "classification": None,
                                "residual_peak_ratio_31d": None,
                                "residual_samples": 0,
                                "residual_saturated_samples": 0,
                            }
                    previous_steps = last_strike_global_step[strike_ids]
                    strike_interval_steps[strike_ids] = torch.where(
                        previous_steps >= 0,
                        int(step_index) - previous_steps,
                        torch.full_like(previous_steps, -1),
                    )
                    last_strike_global_step[strike_ids] = int(step_index)
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
                    robot_state = read_robot_state()
                    incoming_vx = getattr(cmd, "vb_vel_in_w", None)
                    recipe_index = getattr(cmd, "_paired_recipe_current_index", None)
                    recipe_is_wrap = getattr(cmd, "_paired_recipe_current_is_wrap", None)
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
                        fh_correction_applied = is_fh_correction_applied(
                            e, clip_id, venue_selected, incoming_v
                        )
                        if selected and venue_intended is not None:
                            intended = _tolist(venue_intended, e)
                        elif default_intended is not None:
                            intended = default_intended.detach().float().cpu().tolist()
                        else:
                            intended = None
                        virtual_telemetry_rows.append(
                            {
                                "env_id": int(e),
                                "snapshot_branch": (
                                    snapshot_branch_labels.get(int(e))
                                    if args.snapshot_branch_mode
                                    else None
                                ),
                                "global_step": int(step_index),
                                "episode_id": int(telemetry_episode_ids[e].item()),
                                "strike_index": int(telemetry_strike_indices[e].item()),
                                "strike_timestamp_s": float(step_index * base_env.step_dt),
                                "strike_interval_s": (
                                    float(strike_interval_steps[e].item() * base_env.step_dt)
                                    if int(strike_interval_steps[e].item()) >= 0
                                    else None
                                ),
                                "target_strike_interval_s": _scalar(
                                    getattr(motion_cmd, "metrics", {}).get(
                                        "target_strike_interval_s"
                                    ),
                                    e,
                                ),
                                "required_hold_s": _scalar(
                                    getattr(motion_cmd, "metrics", {}).get("required_hold_s"), e
                                ),
                                "scheduled_hold_s": _scalar(
                                    getattr(motion_cmd, "metrics", {}).get("scheduled_hold_s"), e
                                ),
                                "previous_clip_poststrike_s": _scalar(
                                    getattr(motion_cmd, "metrics", {}).get(
                                        "previous_clip_poststrike_s"
                                    ),
                                    e,
                                ),
                                "next_clip_prestrike_s": _scalar(
                                    getattr(motion_cmd, "metrics", {}).get(
                                        "next_clip_prestrike_s"
                                    ),
                                    e,
                                ),
                                "strike_interval_scheduler_unreachable": _bool(
                                    getattr(motion_cmd, "metrics", {})
                                    .get("strike_interval_scheduler_unreachable"),
                                    e,
                                ),
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
                                "fh_correction_applied": fh_correction_applied,
                                "paired_recipe_index": (
                                    int(recipe_index[e].item()) if recipe_index is not None else -1
                                ),
                                "paired_recipe_is_wrap": (
                                    bool(recipe_is_wrap[e].item()) if recipe_is_wrap is not None else None
                                ),
                                "robot_root_pos_env": _tolist(robot_state[0], e, local_position=True),
                                "robot_root_quat_w": _tolist(robot_state[1], e),
                                "robot_root_lin_vel_w": _tolist(robot_state[2], e),
                                "robot_root_ang_vel_w": _tolist(robot_state[3], e),
                                "robot_joint_pos": _tolist(robot_state[4], e),
                                "robot_joint_vel": _tolist(robot_state[5], e),
                            }
                        )
                        if (
                            args.snapshot_branch_mode
                            and recipe_index is not None
                            and int(recipe_index[e].item()) == 1
                        ):
                            snapshot_event1_logged[e] = True
                        if clip_id is not None and int(clip_id[e].detach().cpu().item()) == 0 and not selected:
                            post_pending[e] = True
                            post_source_step[e] = int(step_index)
                            post_source_recipe_index[e] = (
                                int(recipe_index[e].item()) if recipe_index is not None else -1
                            )
                            post_source_episode[e] = int(telemetry_episode_ids[e].item())
                            post_source_strike[e] = int(telemetry_strike_indices[e].item())
                            post_source_clip[e] = 0
                            post_source_corrected[e] = bool(fh_correction_applied)
                            post_captured_mask[e] = 0
                        telemetry_strike_indices[e] += 1
            # A strike happens when the reference clock crosses the strike frame (tts: >0 -> <=0).
            # Environments that RESET this step are excluded: a time-out/fall reset re-seeds the
            # clock, and counting it would contaminate the denominator with non-swings.
            reset_now = dones.reshape(-1).to(dtype=torch.bool, device=tts.device)
            reset_pending = ready_pending & reset_now
            if bool(reset_pending.any()):
                ready_cycles_failed_by_reset += int(reset_pending.sum().item())
                if args.ready_telemetry_out:
                    for env_id in reset_pending.nonzero(as_tuple=False).flatten().tolist():
                        _finalize_ready_cycle(env_id, terminal="reset")
                ready_pending[reset_pending] = False
                ready_cycle_passed[reset_pending] = False
                ready_cycle_start_step[reset_pending] = -1
            termination_manager = getattr(base_env, "termination_manager", None)
            reset_state = read_robot_state() if bool(reset_now.any()) else (None,) * 6
            if bool(reset_now.any()):
                active_terms = getattr(termination_manager, "_term_names", []) if termination_manager else []
                for env_id in reset_now.nonzero(as_tuple=False).flatten().tolist():
                    reasons = []
                    for term_name in active_terms:
                        try:
                            term_value = termination_manager.get_term(term_name)
                            if bool(term_value[env_id].detach().cpu().item()):
                                reasons.append(term_name)
                        except Exception:
                            pass
                    reset_events.append(
                        {
                            "env_id": int(env_id),
                            "global_step": int(step_index),
                            "episode_id": int(telemetry_episode_ids[env_id].item()),
                            "paired_recipe_index": int(
                                getattr(cmd, "_paired_recipe_current_index", torch.full_like(telemetry_episode_ids, -1))[env_id].item()
                            ),
                            "termination_reasons": reasons,
                            "time_out": "time_out" in reasons,
                            "robot_root_pos_after_reset_env": _tolist(
                                reset_state[0], env_id, local_position=True
                            ),
                            "robot_root_lin_vel_after_reset": _tolist(reset_state[2], env_id),
                            "robot_root_ang_vel_after_reset": _tolist(reset_state[3], env_id),
                        }
                    )
                    if bool(post_pending[env_id].item()):
                        post_strike_state_rows.append(
                            {
                                "env_id": int(env_id),
                                "reset_before_post_offset": True,
                                "source_global_step": int(post_source_step[env_id].item()),
                                "source_episode_id": int(post_source_episode[env_id].item()),
                                "source_strike_index": int(post_source_strike[env_id].item()),
                                "source_paired_recipe_index": int(post_source_recipe_index[env_id].item()),
                                "source_clip_id": int(post_source_clip[env_id].item()),
                                "source_fh_correction_applied": bool(post_source_corrected[env_id].item()),
                                "reset_global_step": int(step_index),
                                "reset_reasons": reasons,
                            }
                        )
                        post_pending[env_id] = False
                        post_captured_mask[env_id] = 0
            telemetry_episode_ids[reset_now] += 1
            telemetry_strike_indices[reset_now] = 0
            last_strike_global_step[reset_now] = -1
            strike_interval_steps[reset_now] = -1
            reset_count += int(reset_now.sum().item())
            if args.diagnostics:
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
            # The virtual telemetry contract normally records only exact-strike attempts.  The
            # in-process branch audit also needs the complementary outcome: a fixed next-shot
            # clock crossing where the intervention caused the racket to miss capture.  Record it
            # as an explicit MISS_CAPTURE row so branch comparisons remain paired instead of
            # silently dropping the branch.
            if args.snapshot_branch_mode:
                clip_id_live = getattr(motion_cmd, "clip_id", None)
                recipe_index_live = getattr(cmd, "_paired_recipe_current_index", None)
                incoming_live = getattr(cmd, "vb_vel_in_w", None)
                spin_live = getattr(cmd, "vb_spin_in_w", None)
                selected_live = getattr(cmd, "_venue_tuple_selected", None)
                intended_live = getattr(cmd, "_venue_intended_landing_xy", None)
                default_intended_live = getattr(cmd, "_vb_target_xy", None)
                robot_state_live = read_robot_state()
                # A corrected branch can cross the paired recipe's strike time
                # without producing the exact-strike edge (for example after a
                # capture/reset mismatch).  Treat that as an explicit MISS_CAPTURE
                # for event 1 rather than dropping the branch from the paired set.
                if recipe_index_live is not None:
                    event1_due = (
                        (recipe_index_live == 1)
                        & (tts <= 0.0)
                        & (~snapshot_event1_logged)
                        & (~reset_now)
                    )
                    due_ids = event1_due.nonzero(as_tuple=False).flatten().tolist()
                else:
                    due_ids = []
                fallback_ids = {
                    int(e) for e in idx if not bool(exact_strike_mask[e].item())
                }
                fallback_ids.update(int(e) for e in due_ids)
                for e in sorted(fallback_ids):
                    if bool(snapshot_event1_logged[e].item()):
                        continue
                    if bool(exact_strike_mask[e].item()):
                        continue
                    selected = _bool(selected_live, e) or False
                    if selected and intended_live is not None:
                        intended = _tolist(intended_live, e)
                    elif default_intended_live is not None:
                        intended = default_intended_live.detach().float().cpu().tolist()
                    else:
                        intended = None
                    virtual_telemetry_rows.append(
                        {
                            "env_id": int(e),
                            "snapshot_branch": snapshot_branch_labels.get(int(e)),
                            "global_step": int(step_index),
                            "episode_id": int(telemetry_episode_ids[e].item()),
                            "strike_index": int(telemetry_strike_indices[e].item()),
                            "strike_timestamp_s": float(step_index * base_env.step_dt),
                            "strike_interval_s": None,
                            "target_strike_interval_s": _scalar(
                                getattr(motion_cmd, "metrics", {}).get("target_strike_interval_s"), e
                            ),
                            "required_hold_s": _scalar(
                                getattr(motion_cmd, "metrics", {}).get("required_hold_s"), e
                            ),
                            "scheduled_hold_s": _scalar(
                                getattr(motion_cmd, "metrics", {}).get("scheduled_hold_s"), e
                            ),
                            "previous_clip_poststrike_s": _scalar(
                                getattr(motion_cmd, "metrics", {}).get(
                                    "previous_clip_poststrike_s"
                                ),
                                e,
                            ),
                            "next_clip_prestrike_s": _scalar(
                                getattr(motion_cmd, "metrics", {}).get("next_clip_prestrike_s"), e
                            ),
                            "strike_interval_scheduler_unreachable": _bool(
                                getattr(motion_cmd, "metrics", {})
                                .get("strike_interval_scheduler_unreachable"),
                                e,
                            ),
                            "clip_id": int(clip_id_live[e].detach().cpu().item()) if clip_id_live is not None else None,
                            "swing_sign": _scalar(getattr(cmd, "swing_sign", None), e),
                            "venue_tuple_selected": selected,
                            "incoming_velocity": _tolist(incoming_live, e),
                            "incoming_spin": _tolist(spin_live, e),
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
                            "capture_gate": False,
                            "net_crossed": False,
                            "net_clear": False,
                            "net_z_env": None,
                            "landing_valid": False,
                            "landing_xy_env": [0.0, 0.0],
                            "on_opponent": False,
                            "failure_code": "MISS_CAPTURE",
                            "fh_correction_applied": bool(
                                is_fh_correction_applied(
                                    e,
                                    clip_id_live,
                                    selected_live,
                                    incoming_live,
                                )
                            ),
                            "paired_recipe_index": (
                                int(recipe_index_live[e].item()) if recipe_index_live is not None else -1
                            ),
                            "paired_recipe_is_wrap": (
                                bool(getattr(cmd, "_paired_recipe_current_is_wrap")[e].item())
                                if getattr(cmd, "_paired_recipe_current_is_wrap", None) is not None
                                else None
                            ),
                            "robot_root_pos_env": _tolist(robot_state_live[0], e, local_position=True),
                            "robot_root_quat_w": _tolist(robot_state_live[1], e),
                            "robot_root_lin_vel_w": _tolist(robot_state_live[2], e),
                            "robot_root_ang_vel_w": _tolist(robot_state_live[3], e),
                            "robot_joint_pos": _tolist(robot_state_live[4], e),
                            "robot_joint_vel": _tolist(robot_state_live[5], e),
                        }
                    )
                    if recipe_index_live is not None and int(recipe_index_live[e].item()) == 1:
                        snapshot_event1_logged[e] = True
                    telemetry_strike_indices[e] += 1
            prev_tts = tts.clone()

        if args.ready_telemetry_out:
            for env_id in range(int(args.num_envs)):
                if ready_cycle_states[env_id] is not None:
                    _finalize_ready_cycle(env_id, terminal="end_of_rollout")
        result = accumulator.as_dict()
        strike_intervals = sorted(
            float(row["strike_interval_s"])
            for row in virtual_telemetry_rows
            if row.get("strike_interval_s") is not None
        )

        def _percentile(values, fraction):
            if not values:
                return None
            if len(values) == 1:
                return float(values[0])
            position = (len(values) - 1) * float(fraction)
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            weight = position - lower
            return float(values[lower] * (1.0 - weight) + values[upper] * weight)

        result["strike_interval_summary_s"] = {
            "count": len(strike_intervals),
            "mean": (
                float(sum(strike_intervals) / len(strike_intervals))
                if strike_intervals
                else None
            ),
            "p10": _percentile(strike_intervals, 0.10),
            "p50": _percentile(strike_intervals, 0.50),
            "p90": _percentile(strike_intervals, 0.90),
        }
        ready_recovery_times = sorted(
            float(value) * float(base_env.step_dt) for value in ready_recovery_times_steps
        )
        ready_recovery_summary = {
            "count": len(ready_recovery_times),
            "mean_s": (
                float(sum(ready_recovery_times) / len(ready_recovery_times))
                if ready_recovery_times
                else None
            ),
            "p10_s": _percentile(ready_recovery_times, 0.10),
            "p50_s": _percentile(ready_recovery_times, 0.50),
            "p90_s": _percentile(ready_recovery_times, 0.90),
            "max_s": _percentile(ready_recovery_times, 1.00),
        }
        ready_failure_denominator = ready_cycles_completed + ready_cycles_failed_by_reset
        ready_failure_numerator = ready_cycles_failed + ready_cycles_failed_by_reset
        result["ready_deadline_audit"] = {
            "deadline_gate_steps": int(ready_deadline_gate_steps),
            "strict_pass_steps": int(ready_deadline_strict_pass_steps),
            "strict_pass_step_rate": float(
                ready_deadline_strict_pass_steps / max(ready_deadline_gate_steps, 1)
            ),
            "completed_recovery_cycles": int(ready_cycles_completed),
            "passed_recovery_cycles": int(ready_cycles_passed),
            "failed_recovery_cycles_before_next_strike": int(ready_cycles_failed),
            "failed_recovery_cycles_by_reset": int(ready_cycles_failed_by_reset),
            "ready_failure_rate": float(
                ready_failure_numerator / max(ready_failure_denominator, 1)
            ),
            "unresolved_recovery_cycles_at_end": int(ready_pending.sum().item()),
            "recovery_time_s": ready_recovery_summary,
            "definition": (
                "A cycle starts at an exact strike. It passes when the phase-gated "
                "ready_deadline_strict_pass is observed before the next exact strike. "
                "Reset-ended cycles count as failures; normal timeout alone is not a failure."
            ),
        }
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
                    "reset_reason_counts": {
                        reason: sum(reason in event["termination_reasons"] for event in reset_events)
                        for reason in sorted(
                            {
                                reason
                                for event in reset_events
                                for reason in event["termination_reasons"]
                            }
                        )
                    },
                    "isaac_internal_metrics_mean": internal_metrics,
                    "paired_recipe_mismatch_count": int(
                        len(getattr(cmd, "_paired_recipe_mismatches", []))
                    ),
                    "paired_recipe_mismatches": getattr(
                        cmd, "_paired_recipe_mismatches", []
                    )[:100],
                }
            )
        if args.virtual_telemetry_out:
            telemetry_path = os.path.abspath(args.virtual_telemetry_out)
            os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
            telemetry_payload = {
                "schema_version": 2,
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
                    "fh_vx_max": (
                        None if args.condition_fh_vx_max is None else float(args.condition_fh_vx_max)
                    ),
                },
                "strike_interval_scheduler": {
                    "enabled": args.target_strike_interval_s is not None,
                    "target_range_s": (
                        None
                        if args.target_strike_interval_s is None
                        else [float(value) for value in args.target_strike_interval_s]
                    ),
                    "short_transition_env_fraction": args.short_transition_env_fraction,
                    "transition_clip_sequence": (
                        None
                        if args.transition_clip_sequence is None
                        else [
                            int(item.strip())
                            for item in args.transition_clip_sequence.split(",")
                            if item.strip()
                        ]
                    ),
                },
                "table_frame": {
                    "env_local_table_near_x": table_near_x,
                    "env_local_net_x": float(cmd._vb_net_x),
                    "env_local_far_x": float(cmd._vb_far_x),
                    "half_width": table_half_w,
                    "surface_z": table_surface_z,
                },
                "reset_events": reset_events,
                "post_strike_state_rows": post_strike_state_rows,
                "state_transplant": {
                    "source": args.state_transplant_source,
                    "offset_s": args.state_transplant_offset,
                    "fields": sorted(transplant_fields),
                    "applied_events": transplant_events,
                },
                "snapshot_branch": {
                    "enabled": bool(args.snapshot_branch_mode),
                    "source": args.snapshot_branch_source,
                    "source_env_id": args.snapshot_branch_source_env_id,
                    "offset_s": float(args.snapshot_branch_offset),
                    "blend_steps": int(args.snapshot_branch_blend_steps),
                    "action_replay_steps": int(args.snapshot_action_replay_steps),
                    "action_replay_sequence": bool(args.snapshot_action_replay_sequence),
                    "triggered": bool(snapshot_branch_triggered),
                    "branches": snapshot_branch_labels if args.snapshot_branch_mode else {},
                    "applied_events": snapshot_branch_events,
                },
                "rows": virtual_telemetry_rows,
                "strike_interval_summary_s": result["strike_interval_summary_s"],
            }
            with open(telemetry_path, "w", encoding="utf-8") as f:
                json.dump(telemetry_payload, f, ensure_ascii=False)
                f.write("\n")
            result["virtual_telemetry_out"] = telemetry_path
        if args.ready_telemetry_out:
            from collections import Counter

            class_counts = Counter(str(row.get("classification")) for row in ready_cycle_records)
            failed_condition_counts = Counter()
            failure_combinations = Counter()
            support_failure_combinations = Counter()
            for row in ready_cycle_records:
                component_pass = row.get("deadline_component_pass") or {}
                failed = tuple(sorted(name for name, passed in component_pass.items() if not passed))
                if failed:
                    for name in failed:
                        failed_condition_counts[name] += 1
                    failure_combinations["+".join(failed)] += 1
                support_pass = row.get("deadline_support_pass") or {}
                support_failed = tuple(sorted(name for name, passed in support_pass.items() if not passed))
                if support_failed:
                    support_failure_combinations["+".join(support_failed)] += 1
            ready_path = os.path.abspath(args.ready_telemetry_out)
            os.makedirs(os.path.dirname(ready_path) or ".", exist_ok=True)
            ready_payload = {
                "schema_version": 1,
                "checkpoint": checkpoint,
                "seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "num_steps": int(args.num_steps),
                "control_dt_s": float(base_env.step_dt),
                "ready_contract": {
                    "position_x_y_margin_m": 0.10,
                    "planar_speed_margin_mps": 0.20,
                    "heading_margin_deg": 15.0,
                    "yaw_rate_margin_radps": 0.35,
                    "tilt_margin_projected_gravity": 0.14,
                    "joint_velocity_rms_margin_radps": 0.80,
                    "dwell_s": 0.12,
                    "foot_slip_margin_mps": float(getattr(cmd.cfg, "step_settle_slip_thresh", 0.03)),
                    "classification": {
                        "on_time": "stable READY observed by the fixed deadline",
                        "late": "stable READY first observed after the fixed deadline",
                        "never_ready": "no strict READY sample observed in the cycle",
                        "unstable_ready": "strict READY sample observed but no stable READY dwell",
                        "deadline_not_observed": (
                            "the configured deadline gate did not apply to this cycle's station step class"
                        ),
                    },
                },
                "summary": {
                    "cycles": len(ready_cycle_records),
                    "deadline_observed_cycles": sum(
                        bool(row.get("deadline_observed")) for row in ready_cycle_records
                    ),
                    "classification_counts": dict(class_counts),
                    "deadline_failure_condition_counts": dict(failed_condition_counts),
                    "top_deadline_failure_combinations": failure_combinations.most_common(20),
                    "top_support_failure_combinations": support_failure_combinations.most_common(20),
                },
                "rows": ready_cycle_records,
            }
            with open(ready_path, "w", encoding="utf-8") as f:
                json.dump(ready_payload, f, ensure_ascii=False)
                f.write("\n")
            result["ready_telemetry_out"] = ready_path
        if args.paired_recipe_mode == "capture":
            recipe_path = cmd.write_paired_recipe(args.paired_recipe_path)
            result["paired_recipe_out"] = recipe_path
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

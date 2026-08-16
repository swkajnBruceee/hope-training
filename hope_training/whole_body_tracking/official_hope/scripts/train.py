"""Hydra training entry for the HOPE Agibot A3 policy.

Single task, single algo. Build the ``HOPE-HitterPingPong-AgibotA3-v0`` environment (110-D
``hitter_pure`` actor observation, privileged critic, 50 Hz control, ``wrap_teleport: false``), a
rsl_rl PPO runner, and train. Checkpoints are written locally (periodic every ``save_interval`` and
a final one). There is no Weights & Biases, no external logging service, no gate / lineage
machinery.

Usage:
    python scripts/train.py task=HOPEPingPong algo=ppo headless=true
    python scripts/train.py task=HOPEPingPong algo=ppo_residual \
        residual_warm_start_path=checkpoints/model_21800.pt headless=true

Override any field on the CLI, e.g.:
    python scripts/train.py task=HOPEPingPong num_envs=2048 max_iterations=20000 seed=1 \
        motion_file=/abs/hope_forehand.npz motion_file_2=/abs/hope_backhand.npz

Tune training by editing cfg/task/HOPEPingPong.yaml and the selected cfg/algo/*.yaml.
"""

import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf


def _repo_root() -> pathlib.Path:
    """Return the root of this training package, not the parent project."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cfg").is_dir() and (parent / "source" / "whole_body_tracking").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    """Resolve a clip path: absolute / cwd-relative first, then repo-root-relative."""
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    if rooted.is_file():
        return str(rooted.resolve())
    # Return the repo-root candidate so the error message points at a stable location.
    return str(rooted)


def _resolve_motion_sources(cfg) -> list[str]:
    """Return the list of local clip paths [forehand, backhand] (CLI overrides the task cfg)."""
    primary = cfg.motion_file if cfg.motion_file is not None else cfg.task.get("motion_file")
    secondary = cfg.motion_file_2 if cfg.motion_file_2 is not None else cfg.task.get("motion_file_2")
    clips = [primary]
    if secondary is not None:
        clips.append(secondary)
    resolved = [_resolve_motion_path(c) for c in clips if c is not None]
    if not resolved:
        raise RuntimeError(
            "No motion clip configured. Set motion_file (and motion_file_2) on the CLI or in "
            "cfg/task/HOPEPingPong.yaml."
        )
    for clip in resolved:
        if not pathlib.Path(clip).is_file():
            raise FileNotFoundError(
                f"motion clip not found: {clip}\nProvide your own clips or the bundled clips "
                "under motions/preprocessed/."
            )
    return resolved


def _set_dotted(obj, dotted: str, value, applied: list, where: str) -> None:
    """Set ``obj.<a>.<b>... = value`` if the attribute chain exists; else warn and skip."""
    parts = dotted.split(".")
    node = obj
    for attr in parts[:-1]:
        if not hasattr(node, attr):
            print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{attr}'; skipped.", flush=True)
            return
        node = getattr(node, attr)
    leaf = parts[-1]
    if not hasattr(node, leaf):
        print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{leaf}'; skipped.", flush=True)
        return
    setattr(node, leaf, value)
    applied.append(f"{dotted} = {value}")


def _resolve_racket_clip_ranges(key: str, value):
    """Convert the public YAML per-side range mapping to Isaac's tuple contract.

    The task file is intentionally readable as ``forehand: {x: [lo, hi], ...}``,
    while ``RacketTargetCommand`` consumes ``(clip, axis, bound)`` tuples.  Keep
    this conversion at the single YAML boundary so train/play/evaluate share it.
    """
    per_clip_keys = {
        "racket_pos_range_per_clip",
        "racket_vel_range_per_clip",
        "racket_vel_start_range_per_clip",
        "racket_vel_planner_range_per_clip",
    }
    if key not in per_clip_keys or value is None:
        return value
    if not isinstance(value, dict):
        return value
    axes = ("x", "y", "z")
    sides = ("forehand", "backhand")
    try:
        return tuple(
            tuple(tuple(float(v) for v in side_cfg[axis]) for axis in axes)
            for side_cfg in (value[side] for side in sides)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must contain forehand/backhand x/y/z [lo, hi] ranges"
        ) from exc


def _apply_domain_rand(env_cfg, dr, applied: list) -> None:
    """Apply the shared link-mass / PD-gain randomization knobs.

    The event terms are named ``events.randomize_link_mass`` and
    ``events.randomize_pd_gains`` in :class:`HOPEEventCfg` — the override MUST target
    those exact fields (see ``tests/test_domain_rand_overrides.py``). Semantics per
    range knob:
      * absent          -> keep the env-cfg default;
      * ``null``        -> disable the event entirely (set the term to None);
      * ``[lo, hi]``    -> override the distribution parameters.

    PD gains have two modes:
      * ``pd_mode`` absent (legacy generic scale DR): ``pd_gain_range`` drives the
        stiffness/damping distribution params of ``events.randomize_pd_gains``;
      * ``pd_mode: a3_message_passive_nominal_cohort_v1`` (the shipped
        HitterPingPong recipe, ``mdp.randomize_a3_message_pd_gains``):
        ``pd_gain_range`` MUST be null (it only retires the generic scale DR — the
        a3-message term stays), and ``pd_alpha_range`` / ``pd_beta_range`` /
        ``pd_nominal_fraction`` override the term's alpha/beta/nominal params.
    """
    if dr is None:
        return
    events = getattr(env_cfg, "events", None)
    if events is None:
        return

    def _apply(range_key: str, event_name: str, param_keys: tuple[str, ...]) -> None:
        if range_key not in dr:
            return
        if not hasattr(events, event_name):
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} does not "
                "exist on this env cfg; skipped.",
                flush=True,
            )
            return
        rng = dr.get(range_key)
        if rng is None:
            if getattr(events, event_name) is not None:
                setattr(events, event_name, None)
                applied.append(f"events.{event_name} = None (disabled)")
            return
        term = getattr(events, event_name)
        if term is None:
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} is already "
                "disabled in the env cfg; range ignored.",
                flush=True,
            )
            return
        lo, hi = float(rng[0]), float(rng[1])
        for key in param_keys:
            term.params[key] = (lo, hi)
        applied.append(f"events.{event_name} = {(lo, hi)}")

    _apply("link_mass_range", "randomize_link_mass", ("mass_distribution_params",))

    pd_mode = dr.get("pd_mode")
    if pd_mode is None:
        _apply(
            "pd_gain_range",
            "randomize_pd_gains",
            ("stiffness_distribution_params", "damping_distribution_params"),
        )
        return

    # a3-message PD cohort (the env cfg already installs randomize_a3_message_pd_gains
    # for this recipe): pd_gain_range null only documents that the generic scale DR is
    # off; the alpha/beta/nominal knobs refine the installed term.
    if dr.get("pd_gain_range") is not None:
        print(
            f"[train.py] WARNING: domain_rand.pd_mode={pd_mode!r} requires "
            "pd_gain_range: null (the a3-message term replaces the generic scale DR); "
            "pd_gain_range ignored.",
            flush=True,
        )
    term = getattr(events, "randomize_pd_gains", None)
    if term is None:
        print(
            f"[train.py] WARNING: domain_rand.pd_mode={pd_mode!r}: "
            "events.randomize_pd_gains is disabled in the env cfg; PD knobs ignored.",
            flush=True,
        )
        return
    for range_key, param_key in (
        ("pd_alpha_range", "alpha_range"),
        ("pd_beta_range", "beta_range"),
    ):
        rng = dr.get(range_key)
        if rng is not None:
            term.params[param_key] = (float(rng[0]), float(rng[1]))
            applied.append(f"events.randomize_pd_gains.{param_key} = {term.params[param_key]}")
    nominal = dr.get("pd_nominal_fraction")
    if nominal is not None:
        term.params["nominal_fraction"] = float(nominal)
        applied.append(f"events.randomize_pd_gains.nominal_fraction = {float(nominal)}")


def _apply_friction_curriculum(env_cfg, friction_cfg, applied: list) -> None:
    """Install the per-episode effective foot-floor friction contract.

    The normal HITTER env config keeps its historical startup material event for backwards
    compatibility.  The stance Curriculum-FT task explicitly opts into this reset-time event,
    scoped only to the two A3 feet and coupled to the action term's stance alpha.
    """
    if friction_cfg is None or not bool(friction_cfg.get("enabled", False)):
        return
    events = getattr(env_cfg, "events", None)
    if events is None or not hasattr(events, "physics_material"):
        raise RuntimeError("friction curriculum requires events.physics_material")
    from isaaclab.managers import SceneEntityCfg
    import whole_body_tracking.tasks.tracking.mdp as mdp

    term = getattr(events, "physics_material")
    if term is None:
        raise RuntimeError("friction curriculum cannot replace a disabled physics_material event")
    values = OmegaConf.to_container(friction_cfg, resolve=True)
    if not bool(values.get("sample_per_episode", True)):
        raise ValueError("friction_curriculum.sample_per_episode must be true")
    if not bool(values.get("randomize_per_env", True)):
        raise ValueError("friction_curriculum.randomize_per_env must be true")
    nominal = float(values.get("nominal", 1.0))
    minimum = float(values.get("min", 0.3))
    maximum = float(values.get("max", 1.5))
    start_alpha = float(values.get("start_stance_alpha", 0.25))
    if not (0.0 < minimum <= nominal <= maximum):
        raise ValueError("friction_curriculum requires 0 < min <= nominal <= max")
    if not 0.0 <= start_alpha < 1.0:
        raise ValueError("friction_curriculum.start_stance_alpha must lie in [0, 1)")
    edges = tuple(float(item) for item in values.get("bucket_edges", (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)))
    term.func = mdp.randomize_effective_ground_friction
    term.mode = "reset"
    term.params.clear()
    term.params.update(
        asset_cfg=SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_Link", "right_ankle_roll_Link"],
            preserve_order=True,
        ),
        mu_nominal=nominal,
        mu_min=minimum,
        mu_max=maximum,
        start_stance_alpha=start_alpha,
        restitution_range=tuple(float(item) for item in values.get("restitution_range", (0.0, 0.0))),
        bucket_edges=edges,
    )
    applied.append(
        "events.physics_material = effective_ground_friction(reset, feet-only, "
        f"mu=[{minimum}, {maximum}], nominal={nominal}, start_alpha={start_alpha})"
    )


def _apply_task_overrides(env_cfg, cfg, applied: list) -> None:
    """Apply the task YAML to the registered Isaac environment config.

    The official recipe keeps most command/action settings in the Hydra task YAML, while the
    registered Isaac task supplies the dataclass defaults.  Both training and playback must
    apply the same direct blocks; otherwise, for example, the YAML can enable the full-pose
    mocap ability gate while the env still has the default mocap-disabled settings.
    """
    from whole_body_tracking.utils.task_reward_overrides import apply_reward_overrides

    task = cfg.task
    # episode length (top-level on ManagerBasedRLEnvCfg).
    env_block = task.get("env")
    if env_block is not None and env_block.get("episode_length_s") is not None:
        _set_dotted(env_cfg, "episode_length_s", float(env_block.get("episode_length_s")), applied, "env")
    # continuous multi-rally lifecycle: no teleport on clip wrap.
    motion_block = task.get("motion")
    if motion_block is not None:
        motion_values = OmegaConf.to_container(motion_block, resolve=True)
        for key, value in motion_values.items():
            _set_dotted(env_cfg, f"commands.motion.{key}", value, applied, "motion")

    # These blocks are intentionally direct, one-to-one mappings to the registered env cfg.
    # Reward YAML is handled by the task's reward configuration and is not a flat dataclass
    # block, so it is deliberately excluded here.
    for block_name, target_path in (
        ("action", "actions.joint_pos"),
        ("racket", "commands.racket_target"),
    ):
        block = task.get(block_name)
        if block is None:
            continue
        values = OmegaConf.to_container(block, resolve=True)
        for key, value in values.items():
            # The task YAML intentionally uses the shorter public names used by the
            # deployment/venue contract.  RacketTargetCommand stores the same values with
            # the ``racket_`` prefix to distinguish them from the shared box ranges.
            # Resolve these aliases here so training, playback, and evaluation all apply
            # the YAML recipe instead of silently falling back to dataclass defaults.
            if block_name == "racket":
                key = {
                    "pos_range_per_clip": "racket_pos_range_per_clip",
                    "vel_range_per_clip": "racket_vel_range_per_clip",
                    "vel_start_range_per_clip": "racket_vel_start_range_per_clip",
                    "vel_planner_range_per_clip": "racket_vel_planner_range_per_clip",
                    "vel_planner_mix_prob": "racket_vel_planner_mix_prob",
                    "vel_range_ramp_steps": "racket_vel_range_ramp_steps",
                }.get(key, key)
            elif block_name == "action" and key == "contract":
                # Contract is a manifest/deploy metadata field, not an Isaac action-term
                # dataclass attribute.  It is validated by the action/export contract gates.
                applied.append(f"action.contract = {value} (metadata-only)")
                continue
            value = _resolve_racket_clip_ranges(key, value)
            _set_dotted(env_cfg, f"{target_path}.{key}", value, applied, block_name)
    # domain randomization.
    _apply_domain_rand(env_cfg, task.get("domain_rand"), applied)
    _apply_friction_curriculum(env_cfg, task.get("friction_curriculum"), applied)
    # Reward YAML is a separate recipe layer from the registered EnvCfg. Apply it
    # explicitly so this launcher matches the published HOPE training recipe.
    apply_reward_overrides(env_cfg.rewards, task.get("rewards"), applied)
    # generic overrides map (dotted attribute paths -> value).
    overrides = task.get("overrides")
    if overrides:
        for dotted, value in OmegaConf.to_container(overrides, resolve=True).items():
            _set_dotted(env_cfg, str(dotted), value, applied, "overrides")


def _run(cfg):
    import gymnasium as gym
    import torch
    from datetime import datetime

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import importlib

    importlib.import_module("whole_body_tracking.tasks")  # registers the gym task
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) environment cfg from the registered gym task + task-cfg overrides.
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    applied: list = []
    _apply_task_overrides(env_cfg, cfg, applied)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    print(f"[train.py] task={task_id} num_envs={num_envs} — applied {len(applied)} task override(s):", flush=True)
    for line in applied:
        print(f"[train.py]     {line}", flush=True)

    # 2) reference motion clips (local .npz; clip 0 = forehand, clip 1 = backhand).
    motion_files = _resolve_motion_sources(cfg)
    for i, mf in enumerate(motion_files):
        print(f"[train.py] motion clip {i}: {mf}", flush=True)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # 3) PPO runner cfg from cfg/algo/ppo.yaml.
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)

    # 4) local logging directory.
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[train.py] experiment={agent_cfg.experiment_name} | log_dir={log_dir}", flush=True)

    # 5) build env, (optionally) record video, wrap for rsl_rl.
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)

    # Validate the articulation joint set before training. Isaac/PhysX is allowed to enumerate
    # the same articulated tree in a different traversal order than the deploy wire order; the
    # action/manifest path carries the resolved policy order separately. A strict list equality
    # check here incorrectly rejects the official A3 URDF, whose 31 actuated names are a valid
    # bijection but are returned by PhysX in a leg-first traversal.
    from whole_body_tracking.utils.action_adapter_config import load_joint_order

    _joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
    _expected_order = list(load_joint_order())
    if len(_joint_names) != len(_expected_order) or set(_joint_names) != set(_expected_order):
        raise RuntimeError(
            "Articulation joint set does not match the canonical deploy joint set "
            "(mujoco_reference/config/joint_order_agibot_a3.yaml).\n"
            f"  articulation: {_joint_names}\n"
            f"  canonical:    {_expected_order}\n"
            "Fix the A3 URDF/USD before training."
        )
    if _joint_names == _expected_order:
        print("[train.py] joint-order gate: articulation matches the canonical order.", flush=True)
    else:
        print(
            "[train.py] joint-set gate: valid 31-joint bijection; PhysX articulation order "
            "differs from deploy order and will be recorded in export metadata.",
            flush=True,
        )

    # Validate the actor observation contract when the task declares one (guarded import).
    # The name (e.g. hitter_pure, 110-D) is resolved generically through the contract registry.
    expected_contract = cfg.task.get("actor_obs_contract")
    if expected_contract is not None:
        try:
            from whole_body_tracking.tasks.tracking.actor_observation_contract import (
                validate_actor_observation_contract,
            )

            contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
            print(
                f"[train.py] actor observation contract validated: {contract.name} "
                f"({contract.total_dim}D)",
                flush=True,
            )
        except ImportError:
            print("[train.py] NOTE: actor_observation_contract validator not available; skipping.", flush=True)

    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # 6) optional Residual MVP model-only warm-start, or ordinary continuation resume.
    residual_warm_start = getattr(cfg, "residual_warm_start_path", None)
    if residual_warm_start is not None and getattr(cfg, "checkpoint_path", None) is not None:
        raise ValueError(
            "set only one of residual_warm_start_path and checkpoint_path; "
            "Residual warm-start and ordinary continuation have different optimizer semantics"
        )
    if residual_warm_start is not None:
        residual_warm_start = os.path.abspath(str(residual_warm_start))
        if not os.path.isfile(residual_warm_start):
            raise FileNotFoundError(
                f"[train.py] residual_warm_start_path does not exist: {residual_warm_start}"
            )
        runner.load_residual_warm_start(residual_warm_start)
        print(
            f"[train.py] Residual model-only warm-start from: {residual_warm_start}",
            flush=True,
        )

    # Ordinary resume restores the checkpoint's actor/critic/std, optimizer and iteration state.
    ckpt = getattr(cfg, "checkpoint_path", None)
    if ckpt is not None:
        ckpt = os.path.abspath(str(ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {ckpt}")
        exact_resume = bool(getattr(cfg, "checkpoint_exact_resume", False))
        if exact_resume:
            runner.checkpoint_exact_resume = True
            runner.load_exact(ckpt)
            print(f"[train.py] exact-resumed from checkpoint: {ckpt}", flush=True)
        else:
            runner.load(ckpt)
            print(f"[train.py] resumed from checkpoint: {ckpt}", flush=True)

    # 7) dump the resolved configuration + train.
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # Emit the actual completed clip-to-clip transition counts.  The configured clip sequence
    # describes intent, but reset/fall events can change the realized denominator; checkpoint
    # selection and transition oversampling must use this audit rather than the config alone.
    try:
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        counts = getattr(motion_cmd, "transition_event_counts", None)
        if counts is not None:
            print(
                "[train.py] completed_transition_counts="
                + str(counts.detach().cpu().tolist()),
                flush=True,
            )
    except Exception as exc:
        print(f"[train.py] transition count audit unavailable: {exc}", flush=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so Kit does not try to parse
    # Hydra's task=.../algo=... overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video))
    simulation_app = app_launcher.app

    failed = False
    try:
        _run(cfg)
    except Exception:
        import traceback

        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Hydra training entry for HOPE Agibot A3 WBC.

Pick the task/algo YAML on the command line and override any field:

    python scripts/train.py task=TrackingFlat algo=ppo headless=true logger=tensorboard \
        motion_file=sample_motions/agibot_a3_smoke_stand.npz num_envs=32 max_iterations=3

    python scripts/train.py task=TrackingFlat algo=ppo num_envs=2048 max_iterations=20000 \
        registry_name=<org>/wandb-registry-motions/<motion_name>

Tune by editing cfg/task/*.yaml (env / reward / racket / DR) and cfg/algo/ppo.yaml (PPO). This
script reuses BeyondMimic's training mechanics (Isaac Lab + rsl_rl). A local `motion_file=...`
is preferred for public smoke runs; WandB registry loading is optional.
"""

import os
import sys

import hydra
from omegaconf import OmegaConf


# Make the ``training`` package importable regardless of how this script was
# invoked. Without this, the script silently relies on PYTHONPATH being set
# by an external wrapper (e.g. ``hope_isaac_py`` from setup_train_env.sh),
# and a forgotten ``source setup_train_env.sh`` makes ``import training``
# fail with ModuleNotFoundError deep inside _run(). Resolve paths relative
# to THIS FILE so the script works from any cwd and any checkout location.
_HERE = os.path.dirname(os.path.abspath(__file__))                # .../scripts
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))          # .../whole_body_tracking
for _p in (
    _REPO_ROOT,
    os.path.normpath(os.path.join(_REPO_ROOT, "show")),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _HERE, _REPO_ROOT, _p


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab builds that no longer expose dump_pickle."""
    import os
    import pickle

    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
# Task YAML -> Isaac Lab env cfg overrides (only keys present in the YAML are applied).
# --------------------------------------------------------------------------- #
def _get(node, key, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return default


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


class _OverrideError(AttributeError):
    """Raised when the task YAML asks to override an attribute the composed env cfg does not have."""


def _require(cond, target):
    # The YAML explicitly set a value, but the target attribute is missing on the composed env cfg.
    # That is NEVER a benign no-op: either a STALE/shadowed training was imported (so the
    # cfg classes differ from the working tree) or the Hydra base groups failed to compose. Fail loud
    # instead of silently dropping the override (the old behaviour that hid the std/curriculum edits).
    if not cond:
        raise _OverrideError(
            f"[train.py] task YAML overrides '{target}' but the composed env cfg has no such attribute. "
            f"Check the '[train.py] env cfg source:' line above — if it points into site-packages rather "
            f"than your working tree, a stale install is shadowing the source (fix PYTHONPATH ordering / "
            f"reinstall editable). Otherwise the Hydra base-group composition for this task failed."
        )


def _set_attr(obj, attr, val, cast, applied, where):
    if val is None:
        return  # key absent from YAML -> keep the code default (documented contract)
    _require(hasattr(obj, attr), f"{where}.{attr}")
    setattr(obj, attr, cast(val))
    applied.append(f"{where}.{attr}={cast(val)!r}")


def _set_range(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    rng = (float(val[0]), float(val[1]))
    setattr(obj, attr, rng)
    applied.append(f"{where}.{attr}={rng}")


def _set_vec3(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    vec = (float(val[0]), float(val[1]), float(val[2]))
    setattr(obj, attr, vec)
    applied.append(f"{where}.{attr}={vec}")


def _set_reward(rewards, name, weight, std, applied):
    if weight is None and std is None:
        return  # this reward term is not overridden by the YAML -> keep code defaults
    _require(hasattr(rewards, name), f"rewards.{name}")
    term = getattr(rewards, name)
    _require(term is not None, f"rewards.{name} (term is disabled/None)")
    if weight is not None:
        term.weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)}")
    if std is not None:
        _require("std" in term.params, f"rewards.{name}.params['std']")
        term.params["std"] = float(std)
        applied.append(f"rewards.{name}.params.std={float(std)}")


# YAML keys under `racket:` that target the RacketTargetCommandCfg (used to decide whether the task
# actually requested racket overrides before requiring the command to exist).
_RACKET_KEYS = (
    "strike_phase", "strike_window_s", "strike_time_std_s", "strike_success_pos_thresh",
    "strike_success_vel_thresh", "strike_success_normal_thresh_deg",
    "pos_x_range", "pos_y_range", "pos_z_range",
    "vel_x_range", "vel_y_range", "vel_z_range",
    "base_target_x_range", "base_target_y_range",
    "normal_mode", "forehand_on_negative_y", "mount_normal_axis", "mount_normal_sign",
    "target_mode", "ref_perturb_pos", "ref_perturb_vel", "ref_perturb_normal",
    "ref_perturb_curriculum_steps", "ref_perturb_curriculum_start",
    "ref_perturb_success_gated", "ref_perturb_advance_threshold", "ref_perturb_advance_rate",
    "exact_success_decay", "exact_success_min_count",
)


def _apply_task_overrides(env_cfg, task):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg.

    Returns the list of applied "attr=value" strings (logged by the caller). Keys absent from the
    YAML are left at the code default; keys present whose target attribute is missing RAISE (so a
    stale/shadowed cfg or a broken Hydra composition can never silently swallow an override).
    """
    applied = []

    # env base (num_envs is applied earlier via parse_env_cfg). Read every value through _get so the
    # logic works on both OmegaConf nodes (runtime) and plain dicts (unit tests).
    env = _get(task, "env")
    if env is not None:
        es = _get(env, "env_spacing")
        if es is not None:
            env_cfg.scene.env_spacing = float(es)
            applied.append(f"scene.env_spacing={float(es)}")
        els = _get(env, "episode_length_s")
        if els is not None:
            env_cfg.episode_length_s = float(els)
            applied.append(f"episode_length_s={float(els)}")

    # sim base (control frequency = 1 / (dt * decimation))
    sim = _get(task, "sim")
    if sim is not None:
        dt = _get(sim, "dt")
        if dt is not None:
            env_cfg.sim.dt = float(dt)
            applied.append(f"sim.dt={float(dt)}")
        dec = _get(sim, "decimation")
        if dec is not None:
            env_cfg.decimation = int(dec)
            env_cfg.sim.render_interval = env_cfg.decimation  # keep render in step with decimation
            applied.append(f"decimation={int(dec)}")

    actions = _get(task, "actions")
    if actions is not None and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        raw_clip = _get(actions, "raw_clip")
        if raw_clip is not None:
            _require(hasattr(env_cfg.actions.joint_pos, "raw_clip"), "actions.joint_pos.raw_clip")
            env_cfg.actions.joint_pos.raw_clip = float(raw_clip)
            applied.append(f"actions.joint_pos.raw_clip={float(raw_clip)}")
        soft_limit_margin = _get(actions, "soft_limit_margin_frac")
        if soft_limit_margin is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "soft_limit_margin_frac"),
                "actions.joint_pos.soft_limit_margin_frac",
            )
            env_cfg.actions.joint_pos.soft_limit_margin_frac = float(soft_limit_margin)
            applied.append(f"actions.joint_pos.soft_limit_margin_frac={float(soft_limit_margin)}")
        residual_scale = _get(actions, "native_residual_scale")
        if residual_scale is not None:
            scale = getattr(env_cfg.actions.joint_pos, "scale", None)
            _require(scale is not None, "actions.joint_pos.scale")
            residual_scale = float(residual_scale)
            if isinstance(scale, dict):
                env_cfg.actions.joint_pos.scale = {k: float(v) * residual_scale for k, v in scale.items()}
            else:
                env_cfg.actions.joint_pos.scale = float(scale) * residual_scale
            applied.append(f"actions.joint_pos.scale*=native_residual_scale({residual_scale})")
        scale_multipliers = _get(actions, "native_joint_scale_multipliers")
        if scale_multipliers is not None:
            scale = getattr(env_cfg.actions.joint_pos, "scale", None)
            _require(isinstance(scale, dict), "actions.joint_pos.scale dict for native_joint_scale_multipliers")
            for name, multiplier in scale_multipliers.items():
                _require(name in scale, f"actions.joint_pos.scale['{name}']")
                scale[name] = float(scale[name]) * float(multiplier)
                applied.append(f"actions.joint_pos.scale[{name}]*={float(multiplier)}")

    rw = _get(task, "rewards")
    if rw is not None:
        R = env_cfg.rewards
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"), applied)
        _set_reward(R, "racket_position_y", _get(rw, "racket_position_y_weight"), _get(rw, "racket_position_y_std"), applied)
        _set_reward(R, "racket_position_fine", _get(rw, "racket_position_fine_weight"), _get(rw, "racket_position_fine_std"), applied)
        _set_reward(R, "racket_position_y_fine", _get(rw, "racket_position_y_fine_weight"), _get(rw, "racket_position_y_fine_std"), applied)
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"), applied)
        _set_reward(R, "racket_normal", _get(rw, "racket_normal_weight"), _get(rw, "racket_normal_std"), applied)
        _set_reward(R, "racket_hit_coupled", _get(rw, "racket_hit_coupled_weight"), None, applied)
        if hasattr(R, "racket_hit_coupled") and R.racket_hit_coupled is not None:
            coupled = _get(rw, "racket_hit_coupled")
            if coupled is not None:
                for key in ("pos_std", "vel_std", "normal_std", "base", "vel_coeff", "normal_coeff"):
                    val = _get(coupled, key)
                    if val is not None:
                        R.racket_hit_coupled.params[key] = float(val)
                        applied.append(f"rewards.racket_hit_coupled.params.{key}={float(val)}")
        _set_reward(R, "base_position", _get(rw, "base_position_weight"), _get(rw, "base_position_std"), applied)
        jt = _get(rw, "joint_torques_weight")
        if jt is not None:
            _require(hasattr(R, "joint_torques"), "rewards.joint_torques")
            R.joint_torques.weight = float(jt)
            applied.append(f"rewards.joint_torques.weight={float(jt)}")

        # --- motion imitation prior (the 6 motion_* terms; base weights sum ~5.0) ---------------
        # `motion_scale` multiplies all six at once — the main lever to demote imitation to a soft
        # prior so the racket goal can dominate. Per-term weight/std overrides are also accepted
        # (e.g. motion_body_pos_weight / motion_body_pos_std) and are applied BEFORE the scale.
        _MOTION_TERMS = (
            "motion_global_anchor_pos", "motion_global_anchor_ori",
            "motion_body_pos", "motion_body_ori",
            "motion_torso_ori", "motion_native_joint_pos",
            "motion_body_lin_vel", "motion_body_ang_vel",
        )
        for _t in _MOTION_TERMS:
            _set_reward(R, _t, _get(rw, f"{_t}_weight"), _get(rw, f"{_t}_std"), applied)
        ms = _get(rw, "motion_scale")
        if ms is not None:
            ms = float(ms)
            for _t in _MOTION_TERMS:
                _require(hasattr(R, _t), f"rewards.{_t}")
                _term = getattr(R, _t)
                if _term is not None:
                    _term.weight *= ms
            applied.append(f"rewards.motion_scale={ms} (enabled motion weights only)")

        # --- penalties / regularization (negative weights: energy + smoothness + safety) --------
        for _name, _key in (
            ("action_rate_l2", "action_rate_weight"),
            ("action_residual_l2", "action_residual_weight"),
            ("joint_limit", "joint_limit_weight"),
            ("undesired_contacts", "undesired_contacts_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                getattr(R, _name).weight = float(_w)
                applied.append(f"rewards.{_name}.weight={float(_w)}")

    rk = _get(task, "racket")
    if rk is not None:
        # Only require the racket_target command when the YAML actually sets racket keys, so tasks
        # without a racket objective (e.g. TrackingFlat, which has no `racket:` block) never trip this.
        provided = [k for k in _RACKET_KEYS if _get(rk, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "racket_target"),
                     f"commands.racket_target (task YAML sets racket keys {provided})")
            C = env_cfg.commands.racket_target
            _set_attr(C, "strike_phase", _get(rk, "strike_phase"), float, applied, "racket_target")
            _set_attr(C, "strike_window_s", _get(rk, "strike_window_s"), float, applied, "racket_target")
            _set_attr(C, "strike_time_std_s", _get(rk, "strike_time_std_s"), float, applied, "racket_target")
            _set_attr(C, "strike_success_pos_thresh", _get(rk, "strike_success_pos_thresh"), float, applied, "racket_target")
            _set_attr(C, "strike_success_vel_thresh", _get(rk, "strike_success_vel_thresh"), float, applied, "racket_target")
            _set_attr(C, "strike_success_normal_thresh_deg", _get(rk, "strike_success_normal_thresh_deg"), float, applied, "racket_target")
            _set_range(C, "racket_pos_x_range", _get(rk, "pos_x_range"), applied, "racket_target")
            _set_range(C, "racket_pos_y_range", _get(rk, "pos_y_range"), applied, "racket_target")
            _set_range(C, "racket_pos_z_range", _get(rk, "pos_z_range"), applied, "racket_target")
            _set_range(C, "racket_vel_x_range", _get(rk, "vel_x_range"), applied, "racket_target")
            _set_range(C, "racket_vel_y_range", _get(rk, "vel_y_range"), applied, "racket_target")
            _set_range(C, "racket_vel_z_range", _get(rk, "vel_z_range"), applied, "racket_target")
            _set_range(C, "base_target_x_range", _get(rk, "base_target_x_range"), applied, "racket_target")
            _set_range(C, "base_target_y_range", _get(rk, "base_target_y_range"), applied, "racket_target")
            _set_attr(C, "normal_mode", _get(rk, "normal_mode"), str, applied, "racket_target")
            _set_attr(C, "forehand_on_negative_y", _get(rk, "forehand_on_negative_y"), _as_bool, applied, "racket_target")
            _set_attr(C, "mount_normal_axis", _get(rk, "mount_normal_axis"), int, applied, "racket_target")
            _set_attr(C, "mount_normal_sign", _get(rk, "mount_normal_sign"), float, applied, "racket_target")
            # reference_perturbed target sampling (rank 5): couple targets to the reference swing.
            _set_attr(C, "target_mode", _get(rk, "target_mode"), str, applied, "racket_target")
            _set_vec3(C, "ref_perturb_pos", _get(rk, "ref_perturb_pos"), applied, "racket_target")
            _set_vec3(C, "ref_perturb_vel", _get(rk, "ref_perturb_vel"), applied, "racket_target")
            _set_attr(C, "ref_perturb_normal", _get(rk, "ref_perturb_normal"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_steps", _get(rk, "ref_perturb_curriculum_steps"), int, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_start", _get(rk, "ref_perturb_curriculum_start"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_success_gated", _get(rk, "ref_perturb_success_gated"), _as_bool, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_threshold", _get(rk, "ref_perturb_advance_threshold"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_rate", _get(rk, "ref_perturb_advance_rate"), float, applied, "racket_target")
            _set_attr(C, "exact_success_decay", _get(rk, "exact_success_decay"), float, applied, "racket_target")
            _set_attr(C, "exact_success_min_count", _get(rk, "exact_success_min_count"), float, applied, "racket_target")

    # Domain randomization: behaviour preserved exactly (the pd_gain "absent/null -> disable" semantics
    # are intentional). Only logging is added; the hasattr guards stay so DR stays optional per task.
    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        if mr is not None and hasattr(E, "randomize_link_mass") and E.randomize_link_mass is not None:
            E.randomize_link_mass.params["mass_distribution_params"] = (float(mr[0]), float(mr[1]))
            applied.append(f"events.randomize_link_mass.mass_distribution_params=({float(mr[0])}, {float(mr[1])})")
        if hasattr(E, "randomize_pd_gains"):
            pr = _get(dr, "pd_gain_range")
            if pr is None:
                E.randomize_pd_gains = None  # disable
                applied.append("events.randomize_pd_gains=None(disabled)")
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))
                applied.append(f"events.randomize_pd_gains=({float(pr[0])}, {float(pr[1])})")

    motion = _get(task, "motion")
    if motion is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        C = env_cfg.commands.motion
        pose_range = _get(motion, "pose_range")
        if pose_range is not None:
            C.pose_range = {str(k): (float(v[0]), float(v[1])) for k, v in pose_range.items()}
            applied.append(f"commands.motion.pose_range={C.pose_range}")
        velocity_range = _get(motion, "velocity_range")
        if velocity_range is not None:
            C.velocity_range = {str(k): (float(v[0]), float(v[1])) for k, v in velocity_range.items()}
            applied.append(f"commands.motion.velocity_range={C.velocity_range}")
        joint_position_range = _get(motion, "joint_position_range")
        if joint_position_range is not None:
            C.joint_position_range = (float(joint_position_range[0]), float(joint_position_range[1]))
            applied.append(f"commands.motion.joint_position_range={C.joint_position_range}")
        sample_random_start_phase = _get(motion, "sample_random_start_phase")
        if sample_random_start_phase is not None:
            C.sample_random_start_phase = _as_bool(sample_random_start_phase)
            applied.append(f"commands.motion.sample_random_start_phase={C.sample_random_start_phase}")

    return applied


# --------------------------------------------------------------------------- #
# Training (runs after the simulator is launched).
# --------------------------------------------------------------------------- #
def _run(cfg):
    import os
    import pathlib
    from datetime import datetime

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import training  # noqa: F401
    import training.tasks  # noqa: F401  -- registers the gym tasks
    from training.utils.my_on_policy_runner import MotionOnPolicyRunner, MyOnPolicyRunner
    from training.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Provenance: confirm we imported the WORKING TREE, not a stale install. If this path points into
    # site-packages instead of .../training, a shadow copy is overriding your edits
    # (fix PYTHONPATH ordering in setup_train_env.sh / reinstall editable) and the YAML edits below are
    # being applied onto the wrong cfg classes.
    print(f"[train.py] training imported from: {training.__file__}", flush=True)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) env cfg (gym registry) + task YAML overrides
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _cfg_mod = sys.modules.get(type(env_cfg).__module__)
    print(f"[train.py] env cfg source: {type(env_cfg).__name__} <- {getattr(_cfg_mod, '__file__', '?')}", flush=True)
    applied = _apply_task_overrides(env_cfg, cfg.task)
    print(f"[train.py] applied {len(applied)} task override(s) from cfg/task/{_get(cfg.task, 'name', task_id)}.yaml:", flush=True)
    for _a in applied:
        print(f"[train.py]     {_a}", flush=True)
    if not applied:
        print("[train.py] WARNING: 0 task overrides applied -> the run is using CODE DEFAULTS, not the "
              "YAML (the rewards/racket/env blocks did not compose, or all keys were absent).", flush=True)
    # Human-readable confirmation of the strike-training knobs, straight from the post-override cfg, so
    # you can read the actual runtime values off the launch log without opening logs/.../params/env.yaml.
    R = env_cfg.rewards
    if hasattr(R, "racket_position"):
        fine = ""
        if hasattr(R, "racket_position_fine") and R.racket_position_fine is not None:
            fine = (
                f" pos_fine={R.racket_position_fine.params.get('std')}"
                f"/w={R.racket_position_fine.weight}"
            )
        print("[train.py] racket reward std (post-override): "
              f"pos={R.racket_position.params.get('std')}{fine} "
              f"vel={R.racket_velocity.params.get('std')} normal={R.racket_normal.params.get('std')}", flush=True)
    if hasattr(env_cfg.commands, "racket_target"):
        _C = env_cfg.commands.racket_target
        print("[train.py] racket target (post-override): "
              f"target_mode={_C.target_mode} ref_perturb_curriculum_start={_C.ref_perturb_curriculum_start} "
              f"strike_window_s={_C.strike_window_s} strike_time_std_s={_C.strike_time_std_s}", flush=True)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    has_motion_command = hasattr(env_cfg.commands, "motion")

    # 2) PPO runner cfg from cfg.algo
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)
    agent_cfg.resume = bool(cfg.get("resume", False))
    if cfg.get("load_run", None) is not None:
        agent_cfg.load_run = str(cfg.load_run)
    if cfg.get("checkpoint", None) is not None:
        agent_cfg.load_checkpoint = str(cfg.checkpoint)

    # 3) motion source. Motion-imitation tasks require a clip; pure table-tennis RL tasks do not.
    registry_name = None
    if has_motion_command:
        motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else _get(cfg.task, "motion_manifest")
        motion_file = cfg.motion_file if cfg.motion_file is not None else _get(cfg.task, "motion_file")
        if motion_manifest is not None:
            manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = pathlib.Path.cwd() / manifest_path
            if not manifest_path.is_file():
                raise FileNotFoundError(f"motion_manifest does not exist: {manifest_path}")
            env_cfg.commands.motion.motion_manifest = str(manifest_path)
            env_cfg.commands.motion.motion_file = None
            subset_size = cfg.manifest_subset_size if cfg.manifest_subset_size is not None else _get(cfg.task, "manifest_subset_size")
            if subset_size is not None:
                env_cfg.commands.motion.manifest_subset_size = int(subset_size)
            frame_z_offset = (
                cfg.manifest_frame_z_offset
                if cfg.manifest_frame_z_offset is not None
                else _get(cfg.task, "manifest_frame_z_offset")
            )
            if frame_z_offset is not None:
                env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
            registry_name = f"local:{manifest_path}"
            print(
                f"[train.py] using local motion_manifest: {manifest_path} "
                f"(subset_size={env_cfg.commands.motion.manifest_subset_size}, "
                f"frame_z_offset={env_cfg.commands.motion.manifest_frame_z_offset:.4f}m)",
                flush=True,
            )
        elif motion_file is not None:
            motion_path = pathlib.Path(str(motion_file)).expanduser()
            if not motion_path.is_absolute():
                motion_path = pathlib.Path.cwd() / motion_path
            if not motion_path.is_file():
                raise FileNotFoundError(
                    f"motion_file does not exist: {motion_path}. "
                    "Generate the public smoke clip with scripts/create_smoke_motion.py or pass a retargeted .npz."
                )
            env_cfg.commands.motion.motion_file = str(motion_path)
            registry_name = f"local:{motion_path}"
            print(f"[train.py] using local motion_file: {motion_path}", flush=True)
        else:
            registry_name = cfg.registry_name if cfg.registry_name is not None else cfg.task.registry_name
            registry_name = str(registry_name)
            if ":" not in registry_name:
                registry_name += ":latest"
            print(f"[train.py] loading motion from WandB registry: {registry_name}", flush=True)
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        print("[train.py] env has no motion command; running pure RL task without motion source.", flush=True)

    # 4) logging dir (same layout as scripts/rsl_rl/train.py so export/eval are unchanged)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Task: {task_id} | experiment: {agent_cfg.experiment_name} | log: {log_dir}")

    # 5) build env, wrap, run
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner_registry_name = None if registry_name and registry_name.startswith("local:") else registry_name
    if has_motion_command:
        runner = MotionOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=runner_registry_name
        )
    else:
        runner = MyOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so the kit app does not try to
    # parse Hydra's `task=...`/`algo=...` overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    try:
        _run(cfg)
    except Exception:
        import os
        import traceback

        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            simulation_app.close()
        finally:
            os._exit(1)
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()

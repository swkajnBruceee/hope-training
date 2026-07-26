#!/usr/bin/env python3
"""Paired F0 migration audit for the frozen fixed-base strike policy.

This is deliberately not a training entry point.  It runs the same composite
plant in three modes and only changes root fixation and the source of the
Base14 action:

* fixed_model900
* floating_model900_zero_leg
* floating_model900_stageA

The two actor checkpoints are loaded directly so each observation normalizer
and action head remains independent and auditable.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from train import _apply_task_overrides  # noqa: E402
from stage_a_compat import adapt_stage_a_observation_legacy_yaw_pi, validate_stage_a_legacy_layout  # noqa: E402


UPPER_DEFAULT = (
    "/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_native_strike_manifest/"
    "2026-07-24_22-00-32_backhand_strike_only_v1_shoulders_lead12_res025_clip050_3000it/model_900.pt"
)
STAGE_A_DEFAULT = (
    "/workspace/hopetmp/whole_body_tracking_logs/rsl_rl/agibot_a3_strike_stabilizer_a_unified_k8/"
    "2026-07-22_23-12-10_k17_96env_from_model2897_500it/model_3396.pt"
)

SHARED_STANCE = {
    "root_z_m": 1.0400,
    "hip_pitch_rad": -0.1600,
    "knee_rad": 0.3200,
    "ankle_pitch_rad": -0.1550,
    "left_hip_roll_rad": 0.0800,
    "right_hip_roll_rad": -0.0800,
}
SHARED_STANCE_JOINTS = {
    "left_hip_pitch_joint": -0.1600,
    "right_hip_pitch_joint": -0.1600,
    "left_knee_joint": 0.3200,
    "right_knee_joint": 0.3200,
    "left_ankle_pitch_joint": -0.1550,
    "right_ankle_pitch_joint": -0.1550,
    "left_hip_roll_joint": 0.0800,
    "right_hip_roll_joint": -0.0800,
}


class CheckpointPolicy:
    """Minimal deterministic actor loader with the checkpoint's frozen normalizer."""

    def __init__(self, path: str, device: torch.device):
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = state["model_state_dict"]
        linear_ids = sorted(
            int(key.split(".")[1])
            for key in model
            if key.startswith("actor.") and key.endswith(".weight")
        )
        if not linear_ids:
            raise RuntimeError(f"No actor layers found in {path}")
        layers: list[torch.nn.Module] = []
        for index, layer_id in enumerate(linear_ids):
            weight = model[f"actor.{layer_id}.weight"]
            bias = model[f"actor.{layer_id}.bias"]
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.append(layer)
            if index + 1 < len(linear_ids):
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers).to(device).eval()
        normalizer = state.get("obs_norm_state_dict")
        if normalizer is None:
            raise RuntimeError(f"Checkpoint has no observation normalizer: {path}")
        self.mean = normalizer["_mean"].to(device)
        self.std = normalizer["_std"].to(device).clamp_min(1.0e-6)
        self.path = str(path)
        self.obs_dim = int(self.mean.shape[-1])
        self.action_dim = int(model["std"].shape[-1])

    @torch.inference_mode()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"Observation width mismatch for {self.path}: "
                f"checkpoint={self.obs_dim}, runtime={obs.shape[-1]}"
            )
        normalized = torch.clamp((obs - self.mean) / self.std, -100.0, 100.0)
        return self.actor(normalized)


class LegacySupportReferenceProxy:
    """Feed model_3396 its historical upper-reference preview, read-only.

    The physical upper body remains driven by model_900.  Only the 9-DOF
    reference/velocity preview terms supplied to the frozen leg actor are
    replaced, per current motion, with the closest archived K17 motion.  This
    directly tests whether upper-reference timing is the missing contract.
    """

    _REFERENCE_START = 84
    _REFERENCE_END = 120

    def __init__(self, raw, mapping_report: pathlib.Path):
        from training.robots.agibot_a3 import AGIBOT_A3_JOINT_NAMES, A3_STRIKE_V2_REFERENCE_JOINTS

        report = json.loads(mapping_report.read_text())
        print(f"[F0] loading legacy support mapping {mapping_report}", flush=True)
        legacy_manifest = pathlib.Path(report["legacy_manifest"])
        # Isaac's process locale is ASCII on this workstation while the
        # archived manifest has Chinese absolute paths; never rely on locale.
        legacy_data = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        entries = {str(item["episode_id"]): item for item in legacy_data["motions"]}
        print(f"[F0] loaded {len(entries)} archived manifest entries", flush=True)
        nearest = {
            str(item["current_episode_id"]): str(item["nearest_legacy"][0]["episode_id"])
            for item in report["nearest_legacy_by_current"]
        }
        print(f"[F0] loaded {len(nearest)} current-to-legacy mappings", flush=True)
        current_ids = list(raw.command_manager.get_term("motion").motion.episode_ids)
        print(f"[F0] runtime current ids={current_ids}", flush=True)
        selected = []
        for current_id in current_ids:
            legacy_id = nearest.get(str(current_id))
            if legacy_id is None or legacy_id not in entries:
                raise RuntimeError(f"No legacy support-proxy mapping for current motion {current_id!r}")
            selected.append(entries[legacy_id])
        print(f"[F0] selected legacy support motions={[item['episode_id'] for item in selected]}", flush=True)
        joint_ids = [AGIBOT_A3_JOINT_NAMES.index(name) for name in A3_STRIKE_V2_REFERENCE_JOINTS]
        q, qd, lengths, legacy_ids = [], [], [], []
        for entry in selected:
            print(f"[F0] loading support NPZ {entry['motion_npz']}", flush=True)
            with np.load(entry["motion_npz"]) as data:
                q.append(np.asarray(data["joint_pos"], dtype=np.float32)[:, joint_ids])
                qd.append(np.asarray(data["joint_vel"], dtype=np.float32)[:, joint_ids])
            lengths.append(q[-1].shape[0])
            legacy_ids.append(str(entry["episode_id"]))
        max_length = max(lengths)
        def pad(rows):
            result = np.empty((len(rows), max_length, len(joint_ids)), dtype=np.float32)
            for index, row in enumerate(rows):
                result[index, :len(row)] = row
                result[index, len(row):] = row[-1]
            return torch.as_tensor(result, device=raw.device)
        self.q = pad(q)
        self.qd = pad(qd)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=raw.device)
        self.legacy_ids = legacy_ids

    def apply(self, observation: torch.Tensor, motion, robot) -> torch.Tensor:
        result = observation.clone()
        step = torch.minimum(motion.time_steps, self.lengths - 1)
        env_ids = torch.arange(observation.shape[0], device=observation.device)
        q = self.q[env_ids, step]
        # Match MotionCommand.joint_pos: only position is prelude blended;
        # velocity and lookahead are the original reference feed-forward.
        prelude_steps = max(int(motion.prelude_steps), 1)
        alpha = (motion.prelude_elapsed_steps.float() / prelude_steps).clamp(0.0, 1.0).unsqueeze(-1)
        default_q = robot.data.default_joint_pos[:, [
            robot.joint_names.index(name) for name in (
                "waist_yaw_joint", "waist_pitch_joint", "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            )
        ]]
        q = default_q + alpha * (q - default_q)
        qd = self.qd[env_ids, step]
        qd8 = self.qd[env_ids, torch.minimum(step + 8, self.lengths - 1)]
        qd16 = self.qd[env_ids, torch.minimum(step + 16, self.lengths - 1)]
        result[:, self._REFERENCE_START:self._REFERENCE_END] = torch.cat((q, qd, qd8, qd16), dim=-1)
        result[:, 120] = (step.float() / (self.lengths - 1).clamp(min=1).float()).clamp(0.0, 1.0)
        return result


def _group_obs(env, name: str) -> torch.Tensor:
    value = env.observation_manager.compute_group(name)
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get(name, next(iter(value.values())))
    return value


def _vec(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def _path(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _make_env(cfg, fixed: bool, cases: int, seed: int):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401 -- register gym task

    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=cases)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = seed
    root_frame = str(cfg.get("root_frame", "motion"))
    if root_frame == "asset_default":
        env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    elif root_frame == "motion":
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError("root_frame must be 'motion' or 'asset_default'")
    env_cfg.scene.robot.spawn.fix_base = bool(fixed)
    manifest = cfg.task.get("motion_manifest", None)
    if manifest is None:
        raise ValueError("F0 requires task.motion_manifest")
    manifest_path = _path(str(manifest), pathlib.Path.cwd())
    env_cfg.commands.motion.motion_manifest = str(manifest_path)
    env_cfg.commands.motion.motion_file = None
    subset = cfg.task.get("manifest_subset_size", None)
    env_cfg.commands.motion.manifest_subset_size = None if subset is None else int(subset)
    frame_offset = cfg.task.get("manifest_frame_z_offset", None)
    if frame_offset is not None:
        env_cfg.commands.motion.manifest_frame_z_offset = float(frame_offset)
    release_steps = int(cfg.get("upper_prelude_release_steps", 0))
    if release_steps < 0:
        raise ValueError("upper_prelude_release_steps must be non-negative")
    env_cfg.actions.joint_pos.upper_prelude_release_steps = release_steps
    return gym.make(task_id, cfg=env_cfg)


def _prepare_episode(env, cases: int, seed: int):
    torch.manual_seed(seed)
    env.reset()
    raw = env.unwrapped
    motion = raw.command_manager.get_term("motion")
    racket = raw.command_manager.get_term("racket_target")
    if motion.motion.num_motions < cases:
        raise ValueError(f"Manifest contains {motion.motion.num_motions} motions, need {cases}")
    ids = torch.arange(cases, dtype=torch.long, device=raw.device)
    motion.motion_ids[:] = ids
    motion.time_steps.zero_()
    motion.tail_steps.zero_()
    motion.prelude_elapsed_steps.zero_()
    racket._resample_command(torch.arange(cases, device=raw.device))
    # Manifest mode must be paired by motion id, not by any target-command
    # sampling state left over from env.reset().  Freeze the nominal target
    # tensors so every F0 mode evaluates the identical task point.
    if not getattr(racket.cfg, "manifest_base_aligned", False):
        racket.racket_target_pos_w[:] = raw.scene.env_origins + motion.motion.strike_pos_w[ids]
    racket.racket_target_vel_w[:] = motion.motion.strike_vel_w[ids]
    racket.racket_target_normal_w[:] = motion.motion.strike_normal_w[ids]
    target_pos = racket.racket_target_pos_w.clone()
    target_vel = racket.racket_target_vel_w.clone()
    target_normal = racket.racket_target_normal_w.clone()
    racket._compute_strike_timing()
    action_term = raw.action_manager.get_term("joint_pos")
    root0 = raw.scene["robot"].data.root_pos_w.clone()
    foot_ids, foot_names = raw.scene["robot"].find_bodies(
        ["left_ankle_roll_Link", "right_ankle_roll_Link"], preserve_order=True
    )
    foot0 = raw.scene["robot"].data.body_pos_w[:, foot_ids].clone()
    robot = raw.scene["robot"]
    joint_name_to_id = {name: index for index, name in enumerate(robot.joint_names)}
    stance_joint_names = tuple(
        name for name in SHARED_STANCE_JOINTS if name in joint_name_to_id
    )
    stance_joint_ids = [joint_name_to_id[name] for name in stance_joint_names]
    initial_stance_joint_pos = robot.data.joint_pos[:, stance_joint_ids].clone()
    raw.f0_upper_raw_action = torch.zeros_like(action_term.upper_raw_actions)
    return (
        motion,
        racket,
        action_term,
        root0,
        foot0,
        foot_ids,
        foot_names,
        target_pos,
        target_vel,
        target_normal,
        stance_joint_names,
        initial_stance_joint_pos,
    )


@torch.inference_mode()
def _run_mode(cfg, mode: str, upper: CheckpointPolicy, stage_a: CheckpointPolicy | None, cases: int, seed: int):
    fixed = mode == "fixed_model900"
    env = _make_env(cfg, fixed=fixed, cases=cases, seed=seed)
    try:
        raw = env.unwrapped
        (
            motion,
            racket,
            action_term,
            root0,
            foot0,
            foot_ids,
            foot_names,
            target_pos,
            target_vel,
            target_normal,
            stance_joint_names,
            initial_stance_joint_pos,
        ) = _prepare_episode(env, cases, seed)
        initial_motion_ids = motion.motion_ids.clone()
        if upper.action_dim != 10:
            raise RuntimeError(f"model_900 action width must be 10, got {upper.action_dim}")
        if mode.endswith("stageA"):
            if stage_a is None or stage_a.action_dim != 14:
                raise RuntimeError("Stage-A checkpoint must expose the 14-D Base14 action contract")
        print(
            f"[F0] mode={mode} cases={cases} upper_obs={upper.obs_dim} "
            f"stage_obs={stage_a.obs_dim if stage_a else '-'} "
            f"upper_action={upper.action_dim} stage_action={stage_a.action_dim if stage_a else '-'} "
            f"feet={foot_names}",
            flush=True,
        )
        device = raw.device
        robot = raw.scene["robot"]
        hit_frame = motion.motion.hit_frame[motion.motion_ids]
        prelude_steps = int(getattr(motion, "prelude_steps", 0))
        exact = [None] * cases
        window: list[list[dict[str, float]]] = [[] for _ in range(cases)]
        max_root_disp = torch.zeros(cases, device=device)
        max_foot_disp = torch.zeros(cases, device=device)
        upper_gap_sq = torch.zeros(cases, device=device)
        gap_count = torch.zeros(cases, device=device)
        action_term._env.f0_upper_last_action.zero_()
        layer_probe = bool(cfg.get("layer_probe", False))
        adapt_stage_a_obs_180 = bool(cfg.get("adapt_stage_a_obs_180", False))
        use_legacy_support_proxy = bool(cfg.get("legacy_support_reference_proxy", False))
        legacy_support_proxy = None
        if mode.endswith("stageA") and use_legacy_support_proxy:
            proxy_path = _path(
                str(cfg.get("legacy_support_proxy_report", "eval_outputs/upper_contract/motion_envelope_report.json")),
                pathlib.Path.cwd(),
            )
            if not proxy_path.is_file():
                raise FileNotFoundError(f"Legacy support-proxy report not found: {proxy_path}")
            print(f"[F0] initializing legacy support proxy from {proxy_path}", flush=True)
            legacy_support_proxy = LegacySupportReferenceProxy(raw, proxy_path)
            print(f"[F0] legacy support proxy={legacy_support_proxy.legacy_ids}", flush=True)
        capture_stage_a_obs = bool(cfg.get("capture_stage_a_obs", False))
        record_trace = bool(cfg.get("record_trace", False))
        traces: list[list[dict[str, Any]]] | None = ([[] for _ in range(cases)] if record_trace else None)
        if mode.endswith("stageA") and adapt_stage_a_obs_180:
            validate_stage_a_legacy_layout(
                raw.observation_manager._group_obs_term_names["stage_a"],
                [int(shape[0]) for shape in raw.observation_manager.group_obs_term_dim["stage_a"]],
            )
        requested_steps = cfg.get("max_steps", None)
        steps = int(raw.max_episode_length) + 1
        if requested_steps is not None:
            steps = min(steps, int(requested_steps))
        for step_index in range(steps):
            racket.racket_target_pos_w[:] = target_pos
            racket.racket_target_vel_w[:] = target_vel
            racket.racket_target_normal_w[:] = target_normal
            upper_obs = _group_obs(raw, "policy")
            upper_action = upper(upper_obs)
            if mode.endswith("stageA"):
                stage_obs_raw = _group_obs(raw, "stage_a")
                stage_obs = (
                    adapt_stage_a_observation_legacy_yaw_pi(stage_obs_raw)
                    if adapt_stage_a_obs_180
                    else stage_obs_raw
                )
                if legacy_support_proxy is not None:
                    stage_obs = legacy_support_proxy.apply(stage_obs, motion, robot)
                base_action = stage_a(stage_obs)
            else:
                base_action = torch.zeros((cases, 14), device=device)
            raw.f0_upper_raw_action[:] = upper_action
            if not torch.isfinite(raw.f0_upper_raw_action).all():
                raise RuntimeError(f"Non-finite upper action at step {step_index}")
            if not torch.isfinite(action_term.full_joint_targets).all():
                raise RuntimeError(f"Non-finite composed joint target at step {step_index}")

            root_disp = torch.linalg.vector_norm(robot.data.root_pos_w - root0, dim=-1)
            foot_disp = torch.linalg.vector_norm(robot.data.body_pos_w[:, foot_ids] - foot0, dim=-1).amax(dim=-1)
            max_root_disp = torch.maximum(max_root_disp, root_disp)
            max_foot_disp = torch.maximum(max_foot_disp, foot_disp)
            upper_gap = robot.data.joint_pos[:, action_term._upper_joint_ids_tensor] - action_term.upper_processed_actions
            upper_gap_sq += torch.sum(torch.square(upper_gap), dim=-1)
            gap_count += 1.0

            if traces is not None:
                upper_ids = action_term._upper_joint_ids_tensor
                for env_id in range(cases):
                    traces[env_id].append(
                        {
                            "step": step_index,
                            "motion_frame": int(motion.time_steps[env_id].item()),
                            "prelude_elapsed_steps": int(motion.prelude_elapsed_steps[env_id].item()),
                            "root_pos_w": _vec(robot.data.root_pos_w[env_id]),
                            "root_quat_wxyz": _vec(robot.data.root_quat_w[env_id]),
                            "root_lin_vel_b": _vec(robot.data.root_lin_vel_b[env_id]),
                            "root_ang_vel_b": _vec(robot.data.root_ang_vel_b[env_id]),
                            "upper_reference_q": _vec(motion.joint_pos[env_id, upper_ids]),
                            "upper_reference_qd": _vec(motion.joint_vel[env_id, upper_ids]),
                            "upper_processed_target_q": _vec(action_term.upper_processed_actions[env_id]),
                            "upper_q": _vec(robot.data.joint_pos[env_id, upper_ids]),
                            "upper_qd": _vec(robot.data.joint_vel[env_id, upper_ids]),
                            "upper_policy_action": _vec(upper_action[env_id]),
                            "base_policy_action": _vec(base_action[env_id]),
                            "racket_pos_w": _vec(racket.racket_pos_w[env_id]),
                            "racket_vel_w": _vec(racket.racket_lin_vel_w[env_id]),
                        }
                    )

            current = motion.time_steps
            # ``time_steps`` is the motion-library frame, not wall-clock time.
            # During the ready-pose prelude it remains at frame zero while the
            # reference is blended from the reset pose.  A frame number equal
            # to hit_frame is therefore not a valid strike observation until
            # the prelude has completed.
            strike_phase_active = motion.prelude_elapsed_steps >= prelude_steps
            is_near = strike_phase_active & (torch.abs(current - hit_frame) <= 2)
            for env_id in range(cases):
                if not bool(is_near[env_id]):
                    continue
                pos_err = racket.racket_target_pos_w[env_id] - racket.racket_pos_w[env_id]
                vel_err = racket.racket_target_vel_w[env_id] - racket.racket_lin_vel_w[env_id]
                dot = torch.clamp(torch.dot(racket.racket_target_normal_w[env_id], racket.racket_normal_w[env_id]), -1.0, 1.0)
                item = {
                    "step_offset": int(current[env_id].item() - hit_frame[env_id].item()),
                    "position_error_m": float(torch.linalg.vector_norm(pos_err).item()),
                    "position_error_x_m": float(pos_err[0].item()),
                    "position_error_y_m": float(pos_err[1].item()),
                    "position_error_z_m": float(pos_err[2].item()),
                    "velocity_error_m_s": float(torch.linalg.vector_norm(vel_err).item()),
                    "normal_error_deg": float(torch.rad2deg(torch.acos(dot)).item()),
                }
                window[env_id].append(item)
                if bool(strike_phase_active[env_id]) and int(current[env_id].item()) == int(hit_frame[env_id].item()):
                    origin = raw.scene.env_origins[env_id]
                    target_rel = racket.racket_target_pos_w[env_id] - origin
                    # Recompute from the current articulation cache.  The command
                    # term is normally refreshed by the manager, but this makes
                    # the diagnostic independent of manager update ordering.
                    racket._compute_racket_state()
                    actual_rel = racket.racket_pos_w[env_id] - origin
                    root_initial_rel = root0[env_id] - origin
                    upper_ids = action_term._upper_joint_ids_tensor
                    reference_q = motion.motion.joint_pos[
                        motion.motion_ids[env_id], hit_frame[env_id], upper_ids
                    ]
                    reference_full_q = motion.motion.joint_pos[
                        motion.motion_ids[env_id], hit_frame[env_id]
                    ]
                    reference_root_pos = motion.motion._body_pos_w[
                        motion.motion_ids[env_id], hit_frame[env_id], 0
                    ] + raw.scene.env_origins[env_id]
                    reference_root_quat = motion.motion._body_quat_w[
                        motion.motion_ids[env_id], hit_frame[env_id], 0
                    ]
                    command_q = action_term.upper_processed_actions[env_id]
                    actual_q = robot.data.joint_pos[env_id, upper_ids]
                    probe_fields: dict[str, Any] = {}
                    if layer_probe:
                        current_full_q = robot.data.joint_pos.clone()
                        current_root_state = robot.data.root_state_w.clone()
                        env_ids_all = torch.arange(cases, device=device)
                        zero_joint_vel = torch.zeros_like(robot.data.joint_vel)

                        def _probe_racket(q_probe: torch.Tensor, root_state_probe: torch.Tensor) -> torch.Tensor:
                            robot.write_joint_state_to_sim(q_probe, zero_joint_vel, env_ids=env_ids_all)
                            robot.write_root_state_to_sim(root_state_probe, env_ids=env_ids_all)
                            raw.scene.write_data_to_sim()
                            racket._compute_racket_state()
                            return racket.racket_pos_w.clone()

                        upper_reference_q = current_full_q.clone()
                        upper_reference_q[env_id, upper_ids] = reference_q
                        upper_reference_racket = _probe_racket(upper_reference_q, current_root_state)

                        full_reference_q = current_full_q.clone()
                        full_reference_q[env_id] = reference_full_q
                        full_reference_current_root_racket = _probe_racket(
                            full_reference_q, current_root_state
                        )

                        reference_root_state = current_root_state.clone()
                        reference_root_state[env_id, :3] = reference_root_pos
                        reference_root_state[env_id, 3:7] = reference_root_quat
                        reference_root_state[env_id, 7:] = 0.0
                        full_reference_reference_root_racket = _probe_racket(
                            full_reference_q, reference_root_state
                        )

                        # Restore the live state before continuing the rollout.
                        robot.write_joint_state_to_sim(current_full_q, zero_joint_vel, env_ids=env_ids_all)
                        robot.write_root_state_to_sim(current_root_state, env_ids=env_ids_all)
                        raw.scene.write_data_to_sim()
                        racket._compute_racket_state()
                        probe_fields = {
                            "probe_upper_reference_current_root_racket_m": _vec(
                                upper_reference_racket[env_id]
                            ),
                            "probe_full_reference_current_root_racket_m": _vec(
                                full_reference_current_root_racket[env_id]
                            ),
                            "probe_full_reference_reference_root_racket_m": _vec(
                                full_reference_reference_root_racket[env_id]
                            ),
                        }
                    item.update(
                        {
                            "target_position_x_m": float(target_rel[0].item()),
                            "target_position_y_m": float(target_rel[1].item()),
                            "target_position_z_m": float(target_rel[2].item()),
                            "actual_racket_position_x_m": float(actual_rel[0].item()),
                            "actual_racket_position_y_m": float(actual_rel[1].item()),
                            "actual_racket_position_z_m": float(actual_rel[2].item()),
                            "recomputed_racket_position_m": _vec(racket.racket_pos_w[env_id]),
                            **probe_fields,
                            "wrist_body_position_m": _vec(
                                robot.data.body_pos_w[env_id, racket._wrist_body_index]
                            ),
                            "wrist_body_quat_wxyz": _vec(
                                robot.data.body_quat_w[env_id, racket._wrist_body_index]
                            ),
                            "full_joint_names": list(robot.joint_names),
                            "full_actual_q_rad": _vec(robot.data.joint_pos[env_id]),
                            "full_command_q_rad": _vec(action_term.full_joint_targets[env_id]),
                            "initial_root_position_x_m": float(root_initial_rel[0].item()),
                            "initial_root_position_y_m": float(root_initial_rel[1].item()),
                            "initial_root_position_z_m": float(root_initial_rel[2].item()),
                            "prelude_steps": prelude_steps,
                            "prelude_elapsed_steps_at_hit": int(
                                motion.prelude_elapsed_steps[env_id].item()
                            ),
                            "wall_clock_control_step": int(step_index),
                            "current_root_position_m": _vec(robot.data.root_pos_w[env_id]),
                            "current_root_quat_wxyz": _vec(robot.data.root_quat_w[env_id]),
                            "reference_motion_root_position_m": _vec(reference_root_pos),
                            "reference_motion_root_quat_wxyz": _vec(reference_root_quat),
                            "reference_upper_joint_names": [
                                robot.joint_names[int(index)] for index in upper_ids.detach().cpu().tolist()
                            ],
                            "reference_upper_q_rad": _vec(reference_q),
                            "command_upper_q_rad": _vec(command_q),
                            "actual_upper_q_rad": _vec(actual_q),
                            "command_minus_reference_upper_max_rad": float(
                                torch.max(torch.abs(command_q - reference_q)).item()
                            ),
                            "actual_minus_command_upper_max_rad": float(
                                torch.max(torch.abs(actual_q - command_q)).item()
                            ),
                        }
                    )
                    if mode.endswith("stageA") and capture_stage_a_obs:
                        item["stage_a_observation_raw"] = _vec(stage_obs_raw[env_id])
                        item["stage_a_observation_for_actor"] = _vec(stage_obs[env_id])
                        item["stage_a_action"] = _vec(base_action[env_id])
                    exact[env_id] = item
            env.step(base_action)
            if step_index == 0 or (step_index + 1) % 10 == 0:
                print(f"[F0] mode={mode} step={step_index + 1}/{steps}", flush=True)

        if any(item is None for item in exact):
            missing = [i for i, item in enumerate(exact) if item is None]
            raise RuntimeError(f"Did not observe exact strike frame for envs {missing}")
        results = []
        for env_id in range(cases):
            item = dict(exact[env_id])
            item.update(
                {
                    "motion_id": int(initial_motion_ids[env_id].item()),
                    "max_root_displacement_m": float(max_root_disp[env_id].item()),
                    "max_foot_displacement_m": float(max_foot_disp[env_id].item()),
                    "upper_tracking_rmse_rad": float(torch.sqrt(upper_gap_sq[env_id] / gap_count[env_id]).item()),
                    "legacy_support_proxy_motion": (
                        legacy_support_proxy.legacy_ids[env_id] if legacy_support_proxy is not None else None
                    ),
                    "initial_stance_joint_pos_rad": {
                        name: float(initial_stance_joint_pos[env_id, index].item())
                        for index, name in enumerate(stance_joint_names)
                    },
                    "window_pm2": window[env_id],
                }
            )
            if traces is not None:
                item["trace"] = traces[env_id]
            results.append(item)
        return results
    finally:
        env.close()


@torch.inference_mode()
def _run_stage_a_observation_probe(cfg, stage_a: CheckpointPolicy, cases: int, seed: int):
    """Compare old/new yaw observations at one identical injected state."""
    env = _make_env(cfg, fixed=False, cases=cases, seed=seed)
    try:
        raw = env.unwrapped
        (
            motion,
            racket,
            _action_term,
            _root0,
            _foot0,
            _foot_ids,
            _foot_names,
            target_pos,
            target_vel,
            target_normal,
            _stance_names,
            _stance_pos,
        ) = _prepare_episode(env, cases, seed)
        robot = raw.scene["robot"]
        env_ids = torch.arange(cases, device=raw.device)
        hit_frames = motion.motion.hit_frame[motion.motion_ids]
        motion.time_steps[:] = hit_frames
        motion.prelude_elapsed_steps[:] = int(getattr(motion, "prelude_steps", 0))
        racket.racket_target_pos_w[:] = target_pos
        racket.racket_target_vel_w[:] = target_vel
        racket.racket_target_normal_w[:] = target_normal
        racket._compute_strike_timing()
        joint_pos = motion.motion.joint_pos[motion.motion_ids, hit_frames].clone()
        root_state = robot.data.root_state_w.clone()

        names = list(raw.observation_manager._group_obs_term_names["stage_a"])
        dims = [int(shape[0]) for shape in raw.observation_manager.group_obs_term_dim["stage_a"]]
        validate_stage_a_legacy_layout(names, dims)

        def set_state(quat):
            state = root_state.clone()
            state[:, 3:7] = torch.tensor(quat, device=raw.device, dtype=state.dtype)
            state[:, 7:] = 0.0
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.joint_vel), env_ids=env_ids)
            robot.write_root_state_to_sim(state, env_ids=env_ids)
            raw.scene.write_data_to_sim()
            # State writes update PhysX, not the cached body transforms read by
            # racket FK. Forward once without stepping dynamics before sampling.
            raw.sim.forward()
            raw.scene.update(0.0)
            racket._compute_racket_state()

        term_cfgs = raw.observation_manager._group_obs_term_cfgs["stage_a"]
        saved_noise = [term.noise for term in term_cfgs]
        for term in term_cfgs:
            term.noise = None
        try:
            set_state((1.0, 0.0, 0.0, 0.0))
            old_obs = _group_obs(raw, "stage_a").clone()
            set_state((0.0, 0.0, 0.0, 1.0))
            new_obs = _group_obs(raw, "stage_a").clone()
        finally:
            for term, noise in zip(term_cfgs, saved_noise):
                term.noise = noise

        adapted = adapt_stage_a_observation_legacy_yaw_pi(new_obs)
        old_action = stage_a(old_obs)
        adapted_action = stage_a(adapted)
        obs_diff = torch.abs(old_obs - adapted)
        action_diff = torch.abs(old_action - adapted_action)
        term_diffs = []
        cursor = 0
        for name, width in zip(names, dims):
            span = slice(cursor, cursor + width)
            values = obs_diff[:, span]
            term_diffs.append(
                {
                    "term": name,
                    "start": cursor,
                    "width": width,
                    "max_abs_diff": float(values.max().item()),
                    "mean_abs_diff": float(values.mean().item()),
                }
            )
            cursor += width
        per_motion = []
        for env_id in range(cases):
            per_motion.append(
                {
                    "motion_id": int(motion.motion_ids[env_id].item()),
                    "hit_frame": int(hit_frames[env_id].item()),
                    "obs_max_abs_diff": float(obs_diff[env_id].max().item()),
                    "obs_mean_abs_diff": float(obs_diff[env_id].mean().item()),
                    "action_max_abs_diff": float(action_diff[env_id].max().item()),
                    "action_mean_abs_diff": float(action_diff[env_id].mean().item()),
                    "old_action": _vec(old_action[env_id]),
                    "adapted_action": _vec(adapted_action[env_id]),
                }
            )
        return {
            "root_position_same": True,
            "old_root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "new_root_quat_wxyz": [0.0, 0.0, 0.0, 1.0],
            "motion_time_is_hit_frame": True,
            "observation_noise_disabled": True,
            "term_names": names,
            "term_dims": dims,
            "term_diffs": term_diffs,
            "per_motion": per_motion,
        }
    finally:
        env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg: Any):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=str(cfg.device))
    try:
        base = pathlib.Path.cwd()
        upper_value = cfg.get("upper_checkpoint", None) or UPPER_DEFAULT
        stage_value = cfg.get("stage_a_checkpoint", None) or STAGE_A_DEFAULT
        upper_path = _path(str(upper_value), base)
        stage_path = _path(str(stage_value), base)
        if not upper_path.is_file():
            raise FileNotFoundError(upper_path)
        if not stage_path.is_file():
            raise FileNotFoundError(stage_path)
        cases = int(cfg.get("cases", 6))
        seed = int(cfg.get("seed", 20260725))
        device = torch.device(str(cfg.device))
        upper = CheckpointPolicy(str(upper_path), device)
        stage_a = CheckpointPolicy(str(stage_path), device)
        if upper.obs_dim != 56 or upper.action_dim != 10:
            raise RuntimeError(f"Unexpected model_900 contract: obs={upper.obs_dim}, action={upper.action_dim}")
        if stage_a.obs_dim != 126 or stage_a.action_dim != 14:
            raise RuntimeError(f"Unexpected Stage-A contract: obs={stage_a.obs_dim}, action={stage_a.action_dim}")
        selected_mode = cfg.get("mode", None)
        if selected_mode is not None:
            selected_mode = str(selected_mode)
            if selected_mode == "stageA_observation_probe":
                report = {
                    "stage_a_checkpoint": str(stage_path),
                    "seed": seed,
                    "mode": selected_mode,
                    **_run_stage_a_observation_probe(cfg, stage_a, cases, seed),
                }
                output = _path(str(cfg.get("output", "eval_outputs/stagea_180_observation_probe.json")), base)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
                print(json.dumps({"output": str(output), "per_motion": report["per_motion"]}, indent=2), flush=True)
                return
            valid_modes = {
                "fixed_model900",
                "floating_model900_zero_leg",
                "floating_model900_stageA",
            }
            if selected_mode not in valid_modes:
                raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {selected_mode!r}")
            results = _run_mode(cfg, selected_mode, upper, stage_a, cases, seed)
            report = {
                "upper_checkpoint": str(upper_path),
                "stage_a_checkpoint": str(stage_path),
                "seed": seed,
                "mode": selected_mode,
                "root_frame": str(cfg.get("root_frame", "motion")),
                "adapt_stage_a_obs_180": bool(cfg.get("adapt_stage_a_obs_180", False)),
                "legacy_support_reference_proxy": bool(cfg.get("legacy_support_reference_proxy", False)),
                "shared_initial_stance": SHARED_STANCE,
                "shared_initial_stance_joints": SHARED_STANCE_JOINTS,
                "results": results,
            }
            output = _path(str(cfg.get("output", "eval_outputs/f0_migration/mode.json")), base)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
            print(json.dumps({"output": str(output), "mode": selected_mode}, indent=2), flush=True)
            return
        report: dict[str, Any] = {
            "upper_checkpoint": str(upper_path),
            "stage_a_checkpoint": str(stage_path),
            "seed": seed,
            "shared_initial_stance": SHARED_STANCE,
            "shared_initial_stance_joints": SHARED_STANCE_JOINTS,
            "modes": {},
        }
        for mode in ("fixed_model900", "floating_model900_zero_leg", "floating_model900_stageA"):
            report["modes"][mode] = _run_mode(cfg, mode, upper, stage_a, cases, seed)
        fixed = {row["motion_id"]: row for row in report["modes"]["fixed_model900"]}
        zero = {row["motion_id"]: row for row in report["modes"]["floating_model900_zero_leg"]}
        stage = {row["motion_id"]: row for row in report["modes"]["floating_model900_stageA"]}
        summary = []
        for motion_id in sorted(fixed):
            ef = fixed[motion_id]["position_error_m"]
            ez = zero[motion_id]["position_error_m"]
            es = stage[motion_id]["position_error_m"]
            summary.append(
                {
                    "motion_id": motion_id,
                    "fixed_pos_error_m": ef,
                    "floating_zero_pos_error_m": ez,
                    "floating_stageA_pos_error_m": es,
                    "floating_added_error_m": ez - ef,
                    "stageA_recovery_m": ez - es,
                }
            )
        report["summary"] = summary
        output = _path(str(cfg.get("output", "eval_outputs/f0_migration/f0_report.json")), base)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps({"output": str(output), "summary": summary}, indent=2), flush=True)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()

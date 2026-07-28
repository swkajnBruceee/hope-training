from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import matrix_from_quat, quat_apply, subtract_frame_transforms

from training.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def f0_upper_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the frozen upper actor's 10-D action history for F0."""
    value = getattr(env, "f0_upper_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 10), device=env.device)
    return value


def legacy_stage_a_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the frozen model_3396 14-D action history, not PPO's upper correction."""
    value = getattr(env, "legacy_stage_a_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 14), device=env.device)
    return value


def joint_coordinator_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the previous 22-D joint-coordinator correction action."""
    value = getattr(env, "joint_coordinator_last_action", None)
    if value is None:
        return torch.zeros((env.num_envs, 22), device=env.device)
    return value


def _observation_group_tensor(env: ManagerBasedEnv, name: str) -> torch.Tensor:
    """Read a private observation group without duplicating its term contract."""
    value = env.observation_manager.compute_group(name)
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get(name, next(iter(value.values())))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Observation group {name!r} did not return a tensor")
    return value


def joint_coordinator_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Combine the immutable Stage-A and model_900 contracts for joint PPO.

    The frozen actors keep consuming their own normalized 126-D and 56-D
    groups.  The new coordinator sees both unnormalized runtime contracts plus
    its own previous correction, so it can coordinate support, waist and arm
    corrections without altering either checkpoint's input semantics.
    """
    # ObservationManager probes term shapes while it is still constructing
    # itself. The private frozen-policy groups are unavailable at that point,
    # but their reviewed widths are immutable: 126 + 56 + 22.
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 204), device=env.device)
    return torch.cat(
        (
            _observation_group_tensor(env, "stage_a"),
            _observation_group_tensor(env, "upper"),
            joint_coordinator_last_action(env),
        ),
        dim=-1,
    )


_STAGGER_SUPPORT_OBSERVATION_DIM = 19
_WIDE_STAGGER_SUPPORT_OBSERVATION_DIM = 23
_WIDE_STAGGER_RECOVERY_OBSERVATION_DIM = 2
_STAGGER_FOOT_HALF_LENGTH_M = 0.10
_STAGGER_FOOT_HALF_WIDTH_M = 0.055


def stagger_support_state(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    """Build stance-aware support quantities in the pelvis-yaw frame."""
    from training.robots.agibot_a3 import A3_FEET_BODIES

    robot = env.scene["robot"]
    cache = getattr(env, "_stagger_support_state_cache", None)
    if cache is None:
        foot_body_ids, resolved = robot.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if resolved != A3_FEET_BODIES:
            raise RuntimeError(
                f"Stagger support robot-foot mapping mismatch: expected={A3_FEET_BODIES}, got={resolved}"
            )
        sensor = env.scene.sensors["contact_forces"]
        sensor_ids, sensor_resolved = sensor.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if sensor_resolved != A3_FEET_BODIES:
            raise RuntimeError(
                "Stagger support contact-foot mapping mismatch: "
                f"expected={A3_FEET_BODIES}, got={sensor_resolved}"
            )
        cache = {
            "foot_body_ids": foot_body_ids,
            "foot_sensor_ids": sensor_ids,
            "masses": robot.data.default_mass.to(device=env.device),
        }
        env._stagger_support_state_cache = cache

    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    local_forward = torch.zeros_like(root_pos)
    local_forward[:, 0] = 1.0
    forward_w = quat_apply(root_quat, local_forward)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)

    def project_xy(vector_xy: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sum(vector_xy * forward_xy.unsqueeze(1), dim=-1),
                torch.sum(vector_xy * lateral_xy.unsqueeze(1), dim=-1),
            ),
            dim=-1,
        )

    foot_pos_w = robot.data.body_pos_w[:, cache["foot_body_ids"]]
    foot_rel_root_b = project_xy(foot_pos_w[:, :, :2] - root_pos[:, None, :2])

    masses = cache["masses"]
    total_mass = masses.sum(dim=-1).clamp_min(1.0e-6)
    com_pos_w = torch.sum(robot.data.body_com_pos_w * masses.unsqueeze(-1), dim=1)
    com_pos_w = com_pos_w / total_mass.unsqueeze(-1)
    com_velocity_w = torch.sum(robot.data.body_com_lin_vel_w * masses.unsqueeze(-1), dim=1)
    com_velocity_w = com_velocity_w / total_mass.unsqueeze(-1)
    com_rel_root_b = project_xy((com_pos_w[:, :2] - root_pos[:, :2]).unsqueeze(1)).squeeze(1)
    com_velocity_b = project_xy(com_velocity_w[:, None, :2]).squeeze(1)
    support_center_b = 0.5 * (foot_rel_root_b[:, 0] + foot_rel_root_b[:, 1])
    com_rel_support_b = com_rel_root_b - support_center_b

    root_velocity_b = robot.data.root_lin_vel_b[:, :2]
    com_height = (com_pos_w[:, 2] - foot_pos_w[:, :, 2].mean(dim=-1)).clamp(min=0.45, max=1.50)
    capture_x_b = com_rel_root_b[:, 0] + com_velocity_b[:, 0] / torch.sqrt(9.81 / com_height)
    capture_y_b = com_rel_root_b[:, 1] + com_velocity_b[:, 1] / torch.sqrt(9.81 / com_height)
    rear_limit = foot_rel_root_b[:, :, 0].min(dim=-1).values - _STAGGER_FOOT_HALF_LENGTH_M
    front_limit = foot_rel_root_b[:, :, 0].max(dim=-1).values + _STAGGER_FOOT_HALF_LENGTH_M
    lateral_negative_limit = (
        foot_rel_root_b[:, :, 1].min(dim=-1).values - _STAGGER_FOOT_HALF_WIDTH_M
    )
    lateral_positive_limit = (
        foot_rel_root_b[:, :, 1].max(dim=-1).values + _STAGGER_FOOT_HALF_WIDTH_M
    )
    capture_front_margin = front_limit - capture_x_b
    capture_rear_margin = capture_x_b - rear_limit
    capture_lateral_positive_margin = lateral_positive_limit - capture_y_b
    capture_lateral_negative_margin = capture_y_b - lateral_negative_limit

    sensor = env.scene.sensors["contact_forces"]
    foot_forces_w = sensor.data.net_forces_w[:, cache["foot_sensor_ids"]]
    vertical_load = foot_forces_w[:, :, 2].clamp_min(0.0)
    force_norm = torch.linalg.vector_norm(foot_forces_w, dim=-1)
    contacts = force_norm > 10.0
    body_weight = total_mass * 9.81
    normalized_load = vertical_load / body_weight.unsqueeze(-1)
    total_load_ratio = normalized_load.sum(dim=-1)
    load_balance = (vertical_load[:, 0] - vertical_load[:, 1]) / vertical_load.sum(
        dim=-1
    ).clamp_min(1.0)

    return {
        "foot_rel_root_b": foot_rel_root_b,
        "com_rel_support_b": com_rel_support_b,
        "capture_x_b": capture_x_b,
        "capture_rel_support_x_b": capture_x_b - support_center_b[:, 0],
        "capture_y_b": capture_y_b,
        "capture_rel_support_y_b": capture_y_b - support_center_b[:, 1],
        "capture_front_margin": capture_front_margin,
        "capture_rear_margin": capture_rear_margin,
        "capture_lateral_positive_margin": capture_lateral_positive_margin,
        "capture_lateral_negative_margin": capture_lateral_negative_margin,
        "normalized_load": normalized_load,
        "total_load_ratio": total_load_ratio,
        "load_balance": load_balance,
        "contacts": contacts,
        "com_velocity_b": com_velocity_b,
        "root_velocity_b": root_velocity_b,
        "root_roll_pitch_rate_b": robot.data.root_ang_vel_b[:, :2],
        "sagittal_span": torch.abs(foot_rel_root_b[:, 0, 0] - foot_rel_root_b[:, 1, 0]),
        "lateral_span": torch.abs(foot_rel_root_b[:, 0, 1] - foot_rel_root_b[:, 1, 1]),
    }


def coordinator_stagger_support_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Return a compact physical description of the fore-aft support state."""
    if not hasattr(env, "scene"):
        return torch.zeros((env.num_envs, _STAGGER_SUPPORT_OBSERVATION_DIM), device=env.device)
    state = stagger_support_state(env)
    observation = torch.cat(
        (
            state["foot_rel_root_b"].reshape(env.num_envs, 4) / 0.25,
            state["com_rel_support_b"] / 0.20,
            state["capture_rel_support_x_b"].unsqueeze(-1) / 0.20,
            state["capture_front_margin"].unsqueeze(-1) / 0.20,
            state["capture_rear_margin"].unsqueeze(-1) / 0.20,
            state["normalized_load"],
            state["load_balance"].unsqueeze(-1),
            state["total_load_ratio"].unsqueeze(-1),
            state["contacts"].to(dtype=torch.float32),
            state["root_velocity_b"] / 0.50,
            state["root_roll_pitch_rate_b"],
        ),
        dim=-1,
    )
    if observation.shape[-1] != _STAGGER_SUPPORT_OBSERVATION_DIM:
        raise RuntimeError(
            "Stagger support observation width mismatch: "
            f"{observation.shape[-1]} != {_STAGGER_SUPPORT_OBSERVATION_DIM}"
        )
    return observation.clamp(min=-4.0, max=4.0)


def joint_coordinator_observation_with_stagger_support(env: ManagerBasedEnv) -> torch.Tensor:
    """Append stance-aware support geometry without changing the legacy 204-D prefix."""
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 223), device=env.device)
    return torch.cat(
        (joint_coordinator_observation(env), coordinator_stagger_support_observation(env)),
        dim=-1,
    )


def coordinator_wide_stagger_support_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Add explicit lateral capture geometry to the 19-D stagger contract."""
    if not hasattr(env, "scene"):
        return torch.zeros(
            (env.num_envs, _WIDE_STAGGER_SUPPORT_OBSERVATION_DIM),
            device=env.device,
        )
    state = stagger_support_state(env)
    base = coordinator_stagger_support_observation(env)
    lateral = torch.stack(
        (
            state["capture_rel_support_y_b"] / 0.20,
            state["capture_lateral_positive_margin"] / 0.20,
            state["capture_lateral_negative_margin"] / 0.20,
            state["lateral_span"] / 0.50,
        ),
        dim=-1,
    )
    observation = torch.cat((base, lateral), dim=-1)
    if observation.shape[-1] != _WIDE_STAGGER_SUPPORT_OBSERVATION_DIM:
        raise RuntimeError(
            "Wide stagger support observation width mismatch: "
            f"{observation.shape[-1]} != {_WIDE_STAGGER_SUPPORT_OBSERVATION_DIM}"
        )
    return observation.clamp(min=-4.0, max=4.0)


def joint_coordinator_observation_with_wide_stagger_support(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Append the complete 23-D two-dimensional support contract."""
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 227), device=env.device)
    return torch.cat(
        (
            joint_coordinator_observation(env),
            coordinator_wide_stagger_support_observation(env),
        ),
        dim=-1,
    )


def _post_hit_recovery_gate(
    env: ManagerBasedEnv,
    command_name: str,
    delay_steps: int,
    ramp_steps: int,
) -> torch.Tensor:
    """Return a smooth gate that is identically zero through exact impact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command._use_motion_library:
        hit = command.motion.hit_frame[command.motion_ids]
        final = command.motion.motion_lengths[command.motion_ids] - 1
    else:
        hit = torch.full_like(command.time_steps, int(command.motion.hit_frame[0]))
        final = torch.full_like(command.time_steps, int(command.motion.time_step_total) - 1)

    physical_phase = command.time_steps + 1
    steps_after_hit = torch.clamp(physical_phase - hit, min=0)
    tail_after_hit = torch.clamp(final - hit, min=0) + command.tail_steps
    steps_after_hit = torch.where(command.tail_steps > 0, tail_after_hit, steps_after_hit)
    steps_after_hit = torch.where(
        command.prelude_elapsed_steps >= int(command.prelude_steps),
        steps_after_hit,
        torch.zeros_like(steps_after_hit),
    )
    progress = (
        (steps_after_hit - int(delay_steps)).float()
        / float(max(int(ramp_steps), 1))
    ).clamp(0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


class JointCoordinatorWideStaggerRecoveryObservation(ManagerTermBase):
    """Append capture-point velocity and a post-hit recovery gate to V22."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_capture_x = torch.zeros(env.num_envs, device=env.device)
        self._capture_rate = torch.zeros(env.num_envs, device=env.device)
        self._last_episode_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._needs_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._needs_reset[:] = True
            self._last_episode_step[:] = -1
            self._capture_rate.zero_()
        else:
            self._needs_reset[env_ids] = True
            self._last_episode_step[env_ids] = -1
            self._capture_rate[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        command_name: str = "motion",
        gate_delay_steps: int = 2,
        gate_ramp_steps: int = 8,
        capture_rate_scale_mps: float = 1.0,
    ) -> torch.Tensor:
        if not hasattr(env, "scene") or not hasattr(env, "observation_manager"):
            return torch.zeros((env.num_envs, 229), device=env.device)

        state = stagger_support_state(env)
        capture_x = state["capture_rel_support_x_b"]
        episode_step = getattr(env, "episode_length_buf", None)
        if episode_step is None:
            episode_step = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )

        reset_ids = self._needs_reset | (episode_step < self._last_episode_step)
        if reset_ids.any():
            self._previous_capture_x[reset_ids] = capture_x[reset_ids]
            self._capture_rate[reset_ids] = 0.0
            self._needs_reset[reset_ids] = False

        advance_ids = (~reset_ids) & (episode_step != self._last_episode_step)
        if advance_ids.any():
            self._capture_rate[advance_ids] = (
                capture_x[advance_ids] - self._previous_capture_x[advance_ids]
            ) / float(env.step_dt)
            self._previous_capture_x[advance_ids] = capture_x[advance_ids]
        self._last_episode_step[:] = episode_step

        gate = _post_hit_recovery_gate(
            env,
            command_name=command_name,
            delay_steps=gate_delay_steps,
            ramp_steps=gate_ramp_steps,
        )
        recovery = torch.stack(
            (
                self._capture_rate / float(capture_rate_scale_mps),
                gate,
            ),
            dim=-1,
        ).clamp(min=-4.0, max=4.0)
        observation = torch.cat(
            (
                joint_coordinator_observation_with_wide_stagger_support(env),
                recovery,
            ),
            dim=-1,
        )
        if observation.shape[-1] != 229:
            raise RuntimeError(
                "Wide stagger recovery observation width mismatch: "
                f"{observation.shape[-1]} != 229"
            )
        return observation


_COORDINATOR_PREVIEW_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
_COORDINATOR_PREVIEW_OFFSETS = (4, 8, 12)
_GRAVITY = 9.81
# A 0.10 body-weight momentum-rate change is represented as one actor unit.
# The canonical six-motion library has 95th-percentile raw components around
# 0.04 after m*g normalization, so this keeps informative values at O(1)
# without relying on a running normalizer or motion-specific statistics.
_MOMENTUM_PREVIEW_ACTOR_SCALE = 10.0
_MOMENTUM_PREVIEW_CLIP = 4.0
# The old Stage-A prior has not seen the current high-acceleration strikes.
# Give the coordinator 0.6 s to redistribute support before swing launch;
# the smooth gate still leaves the first 20 ready steps untouched.
_MOMENTUM_PREVIEW_PRELUDE_RAMP_STEPS = 30


def coordinator_upper_velocity_preview(env: ManagerBasedEnv) -> torch.Tensor:
    """Return known future upper-reference velocities for preemptive support.

    The preview deliberately contains only finite-difference motion reference
    data in local joint coordinates.  It has no motion id, world target, or
    future closed-loop model_900 residual, so it remains available at runtime.
    """
    if not hasattr(env, "command_manager"):
        return torch.zeros((env.num_envs, 18), device=env.device)
    motion: MotionCommand = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    joint_ids, resolved = robot.find_joints(list(_COORDINATOR_PREVIEW_JOINTS), preserve_order=True)
    if tuple(resolved) != _COORDINATOR_PREVIEW_JOINTS:
        raise RuntimeError(f"Coordinator preview joint mapping mismatch: {resolved}")
    joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    if motion._use_motion_library:
        full = motion.motion.joint_pos[motion.motion_ids]
        final = motion.motion.motion_lengths[motion.motion_ids].long() - 1
    else:
        full = motion.motion.joint_pos.unsqueeze(0).expand(env.num_envs, -1, -1)
        final = torch.full_like(motion.time_steps, full.shape[1] - 1)
    control_dt = float(env.cfg.decimation * env.cfg.sim.dt)
    if control_dt <= 0.0:
        raise RuntimeError(f"Invalid coordinator preview control dt: {control_dt}")

    preview = []
    for offset in _COORDINATOR_PREVIEW_OFFSETS:
        phase = (motion.time_steps + offset).clamp(min=0)
        before = torch.minimum((phase - 1).clamp(min=0), final)
        after = torch.minimum(phase + 1, final)
        before_q = torch.gather(full, 1, before.view(-1, 1, 1).expand(-1, 1, full.shape[-1])).squeeze(1)
        after_q = torch.gather(full, 1, after.view(-1, 1, 1).expand(-1, 1, full.shape[-1])).squeeze(1)
        velocity = (after_q[:, joint_ids] - before_q[:, joint_ids]) / (2.0 * control_dt)
        # The split post-hit contract, rather than the finite strike motion,
        # owns recovery dynamics.  Do not leak a stale final-frame velocity.
        velocity = torch.where((motion.tail_steps > 0).unsqueeze(-1), torch.zeros_like(velocity), velocity)
        preview.append(velocity)
    preview = torch.cat(preview, dim=-1)
    # Evaluation-only causal ablations.  They alter only the appended preview
    # columns, leaving the legacy 204-D coordinator contract untouched.
    mode = str(getattr(env, "coordinator_preview_audit_mode", "normal"))
    if mode == "normal":
        return preview
    if mode == "zero":
        return torch.zeros_like(preview)
    if mode == "shuffle":
        # Fixed-motion audits assign one motion per environment, so a roll
        # deterministically supplies another motion's known preview.
        return torch.roll(preview, shifts=1, dims=0)
    if mode == "reverse":
        return preview.view(env.num_envs, len(_COORDINATOR_PREVIEW_OFFSETS), -1).flip(1).reshape_as(preview)
    raise ValueError(f"Unknown coordinator_preview_audit_mode={mode!r}")


def joint_coordinator_observation_with_upper_preview(env: ManagerBasedEnv) -> torch.Tensor:
    """Append the fixed 18-D anticipatory upper-dynamics preview to the 204-D contract."""
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 222), device=env.device)
    return torch.cat((joint_coordinator_observation(env), coordinator_upper_velocity_preview(env)), dim=-1)


def coordinator_upper_momentum_preview(env: ManagerBasedEnv) -> torch.Tensor:
    """Return normalized future upper momentum-change rates at 4/8/12 steps."""
    if not hasattr(env, "command_manager"):
        return torch.zeros((env.num_envs, 18), device=env.device)
    motion: MotionCommand = env.command_manager.get_term("motion")
    if not motion._use_motion_library or not motion.motion.has_canonical_upper_momentum:
        raise RuntimeError("Momentum preview requires a manifest with canonical upper momentum")

    full = motion.motion.upper_momentum_pelvis[motion.motion_ids]
    final = motion.motion.motion_lengths[motion.motion_ids].long() - 1
    current_phase = torch.minimum(motion.time_steps, final)
    current = torch.gather(full, 1, current_phase.view(-1, 1, 1).expand(-1, 1, 6)).squeeze(1)
    mass = motion.motion.upper_mass_kg[motion.motion_ids].unsqueeze(-1)
    length = motion.motion.upper_length_scale_m[motion.motion_ids].unsqueeze(-1)
    control_dt = float(env.cfg.decimation * env.cfg.sim.dt)
    if control_dt <= 0.0:
        raise RuntimeError(f"Invalid coordinator preview control dt: {control_dt}")

    preview = []
    for offset in _COORDINATOR_PREVIEW_OFFSETS:
        future_phase = torch.minimum(motion.time_steps + offset, final)
        future = torch.gather(full, 1, future_phase.view(-1, 1, 1).expand(-1, 1, 6)).squeeze(1)
        momentum_rate = (future - current) / (float(offset) * control_dt)
        linear = momentum_rate[:, :3] / (mass * _GRAVITY)
        angular = momentum_rate[:, 3:] / (mass * _GRAVITY * length)
        preview.append(torch.cat((linear, angular), dim=-1))
    preview = torch.cat(preview, dim=-1)
    preview = torch.clamp(
        preview * _MOMENTUM_PREVIEW_ACTOR_SCALE,
        min=-_MOMENTUM_PREVIEW_CLIP,
        max=_MOMENTUM_PREVIEW_CLIP,
    )

    # During reset-to-ready the reference phase is held at frame zero. Without
    # this gate the preview branch would apply the same correction throughout
    # all 50 prelude steps. Activate it only over the final preview horizon so
    # the policy can preload support without disturbing the settled ready pose.
    ramp_start = max(int(motion.prelude_steps) - _MOMENTUM_PREVIEW_PRELUDE_RAMP_STEPS, 0)
    prelude_u = (
        (motion.prelude_elapsed_steps - ramp_start).float()
        / float(max(_MOMENTUM_PREVIEW_PRELUDE_RAMP_STEPS, 1))
    ).clamp(0.0, 1.0)
    prelude_gate = prelude_u * prelude_u * (3.0 - 2.0 * prelude_u)
    in_swing = motion.prelude_elapsed_steps >= int(motion.prelude_steps)
    preview_gate = torch.where(in_swing, torch.ones_like(prelude_gate), prelude_gate)
    preview = preview * preview_gate.unsqueeze(-1)
    preview = torch.where((motion.tail_steps > 0).unsqueeze(-1), torch.zeros_like(preview), preview)

    mode = str(getattr(env, "coordinator_preview_audit_mode", "normal"))
    if mode == "normal":
        return preview
    if mode == "zero":
        return torch.zeros_like(preview)
    if mode == "shuffle":
        return torch.roll(preview, shifts=1, dims=0)
    if mode == "reverse":
        return preview.view(env.num_envs, len(_COORDINATOR_PREVIEW_OFFSETS), -1).flip(1).reshape_as(preview)
    if mode == "scale_080":
        return preview * 0.8
    if mode == "scale_120":
        return preview * 1.2
    raise ValueError(f"Unknown coordinator_preview_audit_mode={mode!r}")


def joint_coordinator_observation_with_momentum_preview(env: ManagerBasedEnv) -> torch.Tensor:
    """Append canonical 18-D upper momentum preview to the legacy 204-D state."""
    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 222), device=env.device)
    return torch.cat((joint_coordinator_observation(env), coordinator_upper_momentum_preview(env)), dim=-1)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

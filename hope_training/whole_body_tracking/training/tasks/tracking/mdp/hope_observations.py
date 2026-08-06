"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_b`  — desired racket velocity in base frame (3)
* :func:`racket_target_normal_b` — desired racket normal in base frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)

The actual racket state can be actor-visible when it is computed from deployable proprioception
(joint state + fixed racket mount FK). :func:`swing_type` is provided for a unified forehand+backhand
policy.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul, quat_rotate_inverse, yaw_quat

from training.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- actor (policy) observations: desired targets only ------------------------------------ #
def racket_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _cmd(env, command_name)
    # The floating target-conditioned replay deliberately keeps the frozen
    # coordinator on the nominal anchor path.  Its external target offset is
    # consumed only by the dedicated feedforward action term.  Other tasks
    # retain the historical behavior because the flag is absent by default.
    if getattr(env, "target_conditioned_anchor_observation", False):
        return command.racket_anchor_target_pos_b()
    return command.racket_target_pos_b()


def coordinator_racket_target_pos_b(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Target position for the trainable coordinator's private upper copy.

    The frozen model_900 observation group must remain on the manifest anchor.
    A target-conditioned coordinator may still consume the external position
    through this separate observation group without contaminating the frozen
    actor's input contract.
    """
    command = _cmd(env, command_name)
    if getattr(env, "target_conditioned_coordinator_external_observation", False):
        return command.racket_target_pos_b()
    return command.racket_anchor_target_pos_b()


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_vel_w


def racket_target_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _cmd(env, command_name)
    if getattr(env, "target_conditioned_anchor_observation", False):
        return command.racket_anchor_target_vel_b()
    return command.racket_target_vel_b()


def racket_target_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _cmd(env, command_name)
    if getattr(env, "target_conditioned_anchor_observation", False):
        return command.racket_anchor_target_normal_b()
    return command.racket_target_normal_b()


def racket_anchor_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Nominal manifest/anchor target, excluding any external position latch."""
    return _cmd(env, command_name).racket_anchor_target_pos_b()


def racket_anchor_target_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_anchor_target_vel_b()


def racket_anchor_target_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_anchor_target_normal_b()


def target_adapter_observation(
    env: ManagerBasedRLEnv, command_name: str = "racket_target", target_delta_scale: float = 0.01
) -> torch.Tensor:
    """Compact P0 target-adapter contract.

    Layout is ``delta_local(3), target_error(3), racket_velocity(3),
    time_to_hit(1), phase_sin/cos(2), motion_one_hot(6), prev_adapter(7)``.
    The last channel is maintained by the action term and is intentionally
    zero when the adapter is disabled.
    """
    command = _cmd(env, command_name)
    # P0's centimetre-scale command must occupy an O(1) feature range.
    delta = command.external_target_delta_local_b() / max(float(target_delta_scale), 1.0e-6)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_pos_w - command.base_pos_w)
    error = command.racket_target_pos_b() - actual
    velocity = quat_rotate_inverse(base_yaw, command.racket_lin_vel_w)
    motion = env.command_manager.get_term(command.cfg.motion_command_name)
    if getattr(motion, "_use_motion_library", False):
        lengths = motion.motion.motion_lengths[motion.motion_ids].clamp_min(2).to(torch.float32)
        phase = motion.time_steps.to(torch.float32) / (lengths - 1.0)
        motion_id = motion.motion_ids.to(torch.long)
    else:
        total = max(int(motion.motion.time_step_total) - 1, 1)
        phase = motion.time_steps.to(torch.float32) / float(total)
        motion_id = torch.zeros_like(motion.time_steps, dtype=torch.long)
    phase_sc = torch.stack((torch.sin(phase * torch.pi), torch.cos(phase * torch.pi)), dim=-1)
    one_hot = torch.nn.functional.one_hot(motion_id.clamp(0, 5), num_classes=6).to(delta.dtype)
    previous = getattr(env, "target_adapter_last_action", None)
    if previous is None:
        previous = torch.zeros((env.num_envs, 7), device=delta.device, dtype=delta.dtype)
    return torch.cat(
        (
            delta,
            error,
            velocity,
            command.time_to_strike.unsqueeze(-1),
            phase_sc,
            one_hot,
            previous,
        ),
        dim=-1,
    )


def racket_target_error_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket position in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_pos_w - command.base_pos_w)
    return command.racket_target_pos_b() - actual


def racket_target_error_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket velocity in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_lin_vel_w)
    return command.racket_target_vel_b() - actual


def racket_target_error_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired-minus-actual racket face normal in the current base yaw frame."""

    command = _cmd(env, command_name)
    base_yaw = yaw_quat(command.base_quat_w)
    actual = quat_rotate_inverse(base_yaw, command.racket_normal_w)
    return command.racket_target_normal_b() - actual


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def time_to_strike_with_prelude(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Time-to-strike in wall-clock time for actors that run through a prelude."""
    cmd = _cmd(env, command_name)
    motion_name = getattr(getattr(cmd, "cfg", None), "motion_command_name", None)
    if motion_name is None:
        return cmd.time_to_strike.unsqueeze(-1)
    motion_cmd = env.command_manager.get_term(motion_name)
    prelude_steps = getattr(motion_cmd, "prelude_steps", None)
    prelude_elapsed_steps = getattr(motion_cmd, "prelude_elapsed_steps", None)
    if prelude_steps is None or prelude_elapsed_steps is None:
        return cmd.time_to_strike.unsqueeze(-1)
    remaining_prelude = (
        int(prelude_steps) - prelude_elapsed_steps
    ).clamp_min(0).to(dtype=cmd.time_to_strike.dtype)
    return (cmd.time_to_strike + remaining_prelude * env.step_dt).unsqueeze(-1)


def base_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).base_target_pos_b()


def swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1). Only needed for a unified (single) policy."""
    return _cmd(env, command_name).swing_sign.unsqueeze(-1)


def fixed_swing_type(env: ManagerBasedRLEnv, value: float = 1.0) -> torch.Tensor:
    """Return a semantic swing label fixed by the task contract.

    A single-family policy must not infer forehand/backhand from target-side
    geometry: a valid forehand target can cross the base-Y centerline after a
    stance or reference-frame change.  Use this term for a forehand-only (or
    backhand-only) task; unified policies should instead provide an explicit
    semantic label from the motion manifest/planner.
    """
    command = _cmd(env, "racket_target")
    return torch.full_like(command.swing_sign, float(value)).unsqueeze(-1)


def manifest_swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the explicit forehand/backhand label stored in the motion manifest.

    Target geometry is intentionally not used: a target can cross the base-Y
    centerline after a stance or reference-frame change while its stroke family
    remains unchanged.  The legacy forehand checkpoint used ``-1`` for this
    channel, so the unified contract retains forehand ``-1`` and assigns
    backhand ``+1``.  Unknown entries get ``0`` rather than silently being
    classified as either family.
    """
    command = _cmd(env, command_name)
    motion_command = env.command_manager.get_term(command.cfg.motion_command_name)
    stroke_ids = motion_command.motion.stroke_ids[motion_command.motion_ids]
    label = torch.zeros_like(stroke_ids, dtype=torch.float)
    label = torch.where(stroke_ids == 0, -torch.ones_like(label), label)
    label = torch.where(stroke_ids == 1, torch.ones_like(label), label)
    return label.unsqueeze(-1)


def motion_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Normalized phase of the known strike reference, in ``[0, 1]``."""

    command = env.command_manager.get_term(command_name)
    lengths = command.motion.motion_lengths[command.motion_ids].clamp(min=2)
    phase = command.time_steps.float() / (lengths - 1).float()
    return phase.clamp(0.0, 1.0).unsqueeze(-1)


def motion_phase_sin(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    phase = motion_phase(env, command_name).squeeze(-1)
    return torch.sin(2.0 * torch.pi * phase).unsqueeze(-1)


def motion_phase_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    phase = motion_phase(env, command_name).squeeze(-1)
    return torch.cos(2.0 * torch.pi * phase).unsqueeze(-1)


def motion_hit_step_normalized(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    lengths = command.motion.motion_lengths[command.motion_ids].clamp(min=2).to(torch.float32)
    hit = command.motion.hit_frame[command.motion_ids].to(torch.float32)
    return (hit / (lengths - 1.0)).clamp(0.0, 1.0).unsqueeze(-1)


def motion_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lookahead_steps: int = 0,
) -> torch.Tensor:
    """Phase-indexed joint-position reference for deployable strike feed-forward."""

    command = env.command_manager.get_term(command_name)
    if int(lookahead_steps) == 0:
        return command.joint_pos[:, asset_cfg.joint_ids]
    steps = command.time_steps + int(lookahead_steps)
    if command._use_motion_library:
        steps = torch.minimum(steps, command.motion.motion_lengths[command.motion_ids] - 1)
        reference = command.motion.joint_pos[command.motion_ids, steps]
    else:
        steps = torch.clamp(steps, max=command.motion.time_step_total - 1)
        reference = command.motion.joint_pos[steps]
    return (reference + command._joint_position_offset)[:, asset_cfg.joint_ids]


def motion_joint_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lookahead_steps: int = 0,
) -> torch.Tensor:
    """Phase-indexed joint-velocity reference for deployable strike feed-forward."""

    command = env.command_manager.get_term(command_name)
    if int(lookahead_steps) == 0:
        return command.joint_vel[:, asset_cfg.joint_ids]
    steps = command.time_steps + int(lookahead_steps)
    if command._use_motion_library:
        steps = torch.minimum(steps, command.motion.motion_lengths[command.motion_ids] - 1)
        return command.motion.joint_vel[command.motion_ids, steps][:, asset_cfg.joint_ids]
    steps = torch.clamp(steps, max=command.motion.time_step_total - 1)
    return command.motion.joint_vel[steps][:, asset_cfg.joint_ids]


def motion_joint_position_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.robot_joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids]


def motion_joint_velocity_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.robot_joint_vel[:, asset_cfg.joint_ids] - command.joint_vel[:, asset_cfg.joint_ids]


def _reference_racket_state_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0):
    """Interpolated candidate TCP state using the same wrist mount as runtime FK."""
    motion = env.command_manager.get_term(command_name)
    target = env.command_manager.get_term("racket_target")
    names = list(getattr(motion.cfg, "body_names", ()))
    wrist_name = target.cfg.wrist_body_name
    if wrist_name not in names:
        raise RuntimeError(f"Reference body contract missing {wrist_name!r}; body_names={names}")
    wrist_index = names.index(wrist_name)
    query = motion.time_steps.to(torch.float32) + float(lookahead_steps)
    if motion._use_motion_library:
        lengths = motion.motion.motion_lengths[motion.motion_ids]
        pos = motion.motion.body_pos_w[motion.motion_ids]
        quat = motion.motion.body_quat_w[motion.motion_ids]
        vel = motion.motion.body_lin_vel_w[motion.motion_ids]
        ang = motion.motion.body_ang_vel_w[motion.motion_ids]
        max_t = (lengths - 1).to(torch.float32)
    else:
        pos = motion.motion.body_pos_w.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
        quat = motion.motion.body_quat_w.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
        vel = motion.motion.body_lin_vel_w.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
        ang = motion.motion.body_ang_vel_w.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
        max_t = torch.full_like(query, pos.shape[1] - 1)
    query = torch.minimum(query.clamp_min(0.0), max_t)
    t0 = query.floor().long()
    t1 = torch.minimum(t0 + 1, max_t.long())
    alpha = (query - t0.to(query.dtype)).unsqueeze(-1)
    batch = torch.arange(env.num_envs, device=query.device)
    def interp(x):
        return x[batch, t0] + alpha * (x[batch, t1] - x[batch, t0])
    wpos, wquat, wvel, wang = (interp(x[:, :, wrist_index]) for x in (pos, quat, vel, ang))
    wquat = wquat / torch.linalg.vector_norm(wquat, dim=-1, keepdim=True).clamp_min(1.0e-6)
    mount = torch.tensor(target.cfg.mount_offset, dtype=wpos.dtype, device=wpos.device).expand_as(wpos)
    mount_w = quat_apply(wquat, mount)
    tcp_pos_w = wpos + mount_w
    tcp_vel_w = wvel + torch.cross(wang, mount_w, dim=-1)
    mount_quat = torch.tensor(target.cfg.mount_quat, dtype=wquat.dtype, device=wquat.device).expand_as(wquat)
    tcp_quat = quat_mul(wquat, mount_quat)
    tcp_normal_w = matrix_from_quat(tcp_quat)[:, :, int(target.cfg.mount_normal_axis)] * float(target.cfg.mount_normal_sign)
    base_yaw = yaw_quat(target.base_quat_w)
    return (
        quat_rotate_inverse(base_yaw, tcp_pos_w - target.base_pos_w),
        quat_rotate_inverse(base_yaw, tcp_vel_w),
        quat_rotate_inverse(base_yaw, tcp_normal_w),
    )


def motion_racket_pos_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    return _reference_racket_state_b(env, command_name, lookahead_steps)[0]


def motion_racket_vel_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    return _reference_racket_state_b(env, command_name, lookahead_steps)[1]


def motion_racket_normal_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    return _reference_racket_state_b(env, command_name, lookahead_steps)[2]


def motion_racket_pos_error_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    ref = motion_racket_pos_b(env, command_name, lookahead_steps)
    return ref - racket_pos_b(env, "racket_target")


def motion_racket_vel_error_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    ref = motion_racket_vel_b(env, command_name, lookahead_steps)
    return ref - racket_lin_vel_b(env, "racket_target")


def motion_racket_normal_error_b(env: ManagerBasedRLEnv, command_name: str, lookahead_steps: int = 0) -> torch.Tensor:
    ref = motion_racket_normal_b(env, command_name, lookahead_steps)
    return ref - racket_normal_b(env, "racket_target")


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Per-foot load-bearing contact state for the deployable P5D tracker."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return (force > float(threshold)).to(dtype=torch.float32)


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_w


def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity in base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_lin_vel_w)


def racket_normal_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal in base yaw frame (FK from robot joint state)."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_normal_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal (FK), world frame."""
    return _cmd(env, command_name).racket_normal_w


def episode_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in the episode (seconds). HITTER critic privileged input."""
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        # Isaac Lab calls observation terms once while constructing the manager,
        # before runtime episode buffers are attached. Return the reset-state
        # shape so the observation manager can infer dimensions.
        left = torch.full((env.num_envs,), float(env.max_episode_length) * float(env.step_dt), device=env.device)
    else:
        left = (env.max_episode_length - episode_length_buf).float() * env.step_dt
    return left.unsqueeze(-1)

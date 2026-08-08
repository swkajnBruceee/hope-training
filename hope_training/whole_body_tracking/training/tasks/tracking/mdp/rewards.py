from __future__ import annotations

import json
from pathlib import Path
import torch
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude

from training.tasks.tracking.mdp.commands import MotionCommand
from training.tasks.tracking.mdp.fall_state import unified_fall_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_REALIZED_PHASE_REFERENCE_CACHE: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}


def strict_fall_risk_l2(
    env: ManagerBasedRLEnv,
    minimum_upright: float = 0.80,
    minimum_height: float = 0.90,
    minimum_torso_upright: float = 0.85,
    minimum_torso_height: float = 0.80,
) -> torch.Tensor:
    """Dense early warning cost for a visually developing fall.

    This is deliberately separate from the terminal ``fall`` reward.  It
    gives PPO a gradient while the robot is already leaning or dropping but
    before the strict fall predicate has persisted for its required steps.
    The historical threshold arguments remain accepted for config
    compatibility; the physical values come from ``UnifiedFallState`` so the
    reward and termination cannot disagree.
    """

    # All fall consumers share one cached physical state.  Keeping this term
    # on the same source as the termination prevents PPO from being rewarded
    # for a state that the done term classifies differently.
    if hasattr(env, "scene") and hasattr(env.scene, "__getitem__"):
        state = unified_fall_state(env)
        env.unified_fall_risk_components = state.risk_components
        return state.risk_score.square()

    robot = env.scene["robot"]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=-1.0, max=1.0)
    tilt_risk = torch.square(torch.relu(float(minimum_upright) - upright))
    height_risk = torch.square(
        torch.relu(float(minimum_height) - robot.data.root_pos_w[:, 2])
    )
    torso_risk = torch.zeros_like(tilt_risk)
    torso_height_risk = torch.zeros_like(height_risk)
    torso_id = getattr(env, "_strict_fall_torso_body_id", None)
    if torso_id is None:
        try:
            ids, names = robot.find_bodies(["torso_Link"], preserve_order=True)
            torso_id = int(ids[0]) if names else -1
        except Exception:
            torso_id = -1
        setattr(env, "_strict_fall_torso_body_id", torso_id)
    if int(torso_id) >= 0:
        torso_pos = robot.data.body_pos_w[:, int(torso_id)]
        torso_quat = robot.data.body_quat_w[:, int(torso_id)]
        gravity_w = torch.zeros_like(torso_pos)
        gravity_w[:, 2] = -1.0
        torso_gravity_b = math_utils.quat_rotate_inverse(torso_quat, gravity_w)
        torso_upright = torch.clamp(-torso_gravity_b[:, 2], min=-1.0, max=1.0)
        torso_risk = torch.square(torch.relu(float(minimum_torso_upright) - torso_upright))
        torso_height_risk = torch.square(
            torch.relu(float(minimum_torso_height) - torso_pos[:, 2])
        )
    return tilt_risk + height_risk + torso_risk + torso_height_risk


def recovery_completion_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus only after the unified recovery hold, never at ordinary timeout."""
    state = unified_fall_state(env)
    return state.recovery_ready.to(dtype=state.risk_score.dtype)


def terminal_remaining_horizon_penalty(env: ManagerBasedRLEnv, horizon_steps: int = 250) -> torch.Tensor:
    """Expose remaining-horizon cost so early termination cannot be profitable."""
    state = unified_fall_state(env)
    terminated = state.confirmed_fall | state.predicted_unrecoverable
    episode_steps = getattr(env, "episode_length_buf", torch.zeros_like(state.risk_score))
    max_steps = max(int(getattr(env, "max_episode_length", horizon_steps)), 1)
    remaining = (max_steps - episode_steps.to(dtype=state.risk_score.dtype)).clamp_min(0.0)
    return terminated.to(dtype=state.risk_score.dtype) * (remaining / float(max_steps))


def _realized_phase_reference(
    env: ManagerBasedRLEnv, command_name: str, reference_bank: str
) -> dict[str, torch.Tensor]:
    """Load continuous zero-residual traces as reward-only phase references.

    These traces deliberately never reset physics.  They are safe to compare
    against during a continuous prefix rollout, but are not simulator states
    that may be teleported into the environment.
    """

    cache_key = (id(env), command_name, str(reference_bank))
    cached = _REALIZED_PHASE_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    bank = Path(reference_bank).expanduser()
    manifest = json.loads((bank / "rsi_capture_manifest.json").read_text(encoding="utf-8"))
    by_episode = {str(entry["episode_id"]): entry for entry in manifest["entries"]}
    command: MotionCommand = env.command_manager.get_term(command_name)
    episode_ids = [str(item) for item in command.motion.episode_ids]
    fields = {"body_pos_w": [], "body_quat_w": [], "body_ang_vel_w": [], "motion_step": []}
    for episode_id in episode_ids:
        entry = by_episode.get(episode_id)
        if entry is None:
            raise ValueError(f"Realized phase bank has no episode {episode_id!r}")
        with np.load(bank / entry["state_file"], allow_pickle=False) as data:
            for field in fields:
                fields[field].append(torch.as_tensor(data[field], dtype=torch.float32, device=env.device))
    lengths = {tensor.shape[0] for tensor in fields["motion_step"]}
    if len(lengths) != 1:
        raise ValueError("Stage-A realized phase references must have equal lengths")
    cached = {name: torch.stack(values, dim=0) for name, values in fields.items()}
    _REALIZED_PHASE_REFERENCE_CACHE[cache_key] = cached
    return cached


def _realized_phase_indices(
    env: ManagerBasedRLEnv, command_name: str, reference_bank: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    reference = _realized_phase_reference(env, command_name, reference_bank)
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Capture frame zero is simulator motion step one.  Clamp the brief reset
    # interval to that first continuous state rather than inventing a teleport.
    indices = (command.time_steps - 1).clamp(min=0, max=reference["motion_step"].shape[1] - 1)
    if not getattr(command, "_use_motion_library", False):
        motion_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        motion_ids = command.motion_ids
    if motion_ids.numel() != env.num_envs:
        raise RuntimeError(
            "Stage-A realized reference motion-id batch does not match the environment batch: "
            f"motion_ids={motion_ids.numel()} envs={env.num_envs}"
        )
    if int(motion_ids.max().item()) >= reference["motion_step"].shape[0]:
        raise RuntimeError(
            "Stage-A realized reference bank lacks a sampled motion: "
            f"max_motion_id={int(motion_ids.max().item())} bank_size={reference['motion_step'].shape[0]}"
        )
    return reference, indices, motion_ids


def realized_torso_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    torso_body_name: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    body_id = env.scene["robot"].body_names.index(torso_body_name)
    expected = reference["body_quat_w"][motion_ids, indices, body_id]
    actual = env.scene["robot"].data.body_quat_w[:, body_id]
    error = quat_error_magnitude(expected, actual) ** 2
    return torch.exp(-error / std**2)


def realized_torso_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    torso_body_name: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_id = env.scene["robot"].body_names.index(torso_body_name)
    expected = reference["body_ang_vel_w"][motion_ids, indices, body_id]
    # In a finite strike's settling tail the arm/waist target is held at the
    # last frame.  The desired torso velocity is therefore zero: the policy is
    # rewarded for dissipating residual swing momentum rather than reproducing
    # the last nonzero sampled motion velocity.
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is not None:
        expected = torch.where(tail_steps.unsqueeze(-1) > 0, torch.zeros_like(expected), expected)
    actual = env.scene["robot"].data.body_ang_vel_w[:, body_id]
    error = torch.sum(torch.square(expected - actual), dim=-1)
    return torch.exp(-error / std**2)


def realized_root_height_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_bank: str,
    std: float,
) -> torch.Tensor:
    reference, indices, motion_ids = _realized_phase_indices(env, command_name, reference_bank)
    expected_z = reference["body_pos_w"][motion_ids, indices, 0, 2]
    actual_z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(expected_z - actual_z) / std**2)


def post_strike_root_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward translational settling only after the finite strike reaches its tail.

    During the swing the torso/COM may move deliberately.  Once the final
    arm/waist target is held, persistent root translation is instead evidence
    of an unresolved fall.  The term is zero before the tail so it cannot
    dominate the swing phase with a constant reward.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b), dim=-1)
    settling_reward = torch.exp(-error / std**2)
    return torch.where(tail_steps > 0, settling_reward, torch.zeros_like(settling_reward))


def post_strike_ready_score(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    linear_velocity_std: float,
    angular_velocity_std: float,
    minimum_height: float,
    height_std: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Score a task-level, post-strike ready state without prescribing joints.

    The score is enabled only after the final swing frame is held.  It rewards
    the *outcome* needed for a next strike: low body-frame translation and
    rotation, a non-collapsing root height, and both feet bearing load.  It
    deliberately contains no knee/hip/ankle target or directional action cue.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    linear_error = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1)
    angular_error = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1)
    height_deficit = torch.relu(minimum_height - robot.data.root_pos_w[:, 2])

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    both_feet = (forces > contact_threshold).all(dim=-1).to(dtype=torch.float32)

    score = (
        torch.exp(-linear_error / linear_velocity_std**2)
        * torch.exp(-angular_error / angular_velocity_std**2)
        * torch.exp(-torch.square(height_deficit) / height_std**2)
        * both_feet
    )
    return torch.where(tail_steps > 0, score, torch.zeros_like(score))


def _return_or_ready_mask(command: MotionCommand) -> torch.Tensor:
    """Select the smooth return and READY hold, never the follow-through hold."""

    hold_steps = int(command.cfg.hold_last_frame_steps)
    return command.tail_steps > hold_steps


def post_strike_bent_ready_arm_score(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.15,
    velocity_std: float = 0.50,
) -> torch.Tensor:
    """Reward a quiet right arm at the task's configured READY manifold."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    joint_ids = getattr(env, "_bent_ready_reward_right_arm_joint_ids", None)
    if joint_ids is None:
        from training.robots.agibot_a3 import A3_RIGHT_ARM_JOINTS

        joint_ids, resolved = robot.find_joints(
            A3_RIGHT_ARM_JOINTS, preserve_order=True
        )
        if resolved != A3_RIGHT_ARM_JOINTS:
            raise RuntimeError(
                "Bent-ready reward right-arm mapping mismatch: "
                f"expected={A3_RIGHT_ARM_JOINTS}, got={resolved}"
            )
        joint_ids = tuple(int(index) for index in joint_ids)
        env._bent_ready_reward_right_arm_joint_ids = joint_ids
    position_error = torch.mean(
        torch.square(
            robot.data.joint_pos[:, joint_ids]
            - command.ready_joint_pos[:, joint_ids]
        ),
        dim=-1,
    )
    velocity_error = torch.mean(
        torch.square(robot.data.joint_vel[:, joint_ids]), dim=-1
    )
    score = torch.exp(-position_error / position_std**2) * torch.exp(
        -velocity_error / velocity_std**2
    )
    return torch.where(_return_or_ready_mask(command), score, torch.zeros_like(score))


class PostStrikeBentReadyProgress(ManagerTermBase):
    """Reward return-phase reduction of body and bent-READY arm error."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        from training.robots.agibot_a3 import A3_RIGHT_ARM_JOINTS

        robot = env.scene["robot"]
        joint_ids, resolved = robot.find_joints(A3_RIGHT_ARM_JOINTS, preserve_order=True)
        if resolved != A3_RIGHT_ARM_JOINTS:
            raise RuntimeError(
                "Bent-ready progress right-arm mapping mismatch: "
                f"expected={A3_RIGHT_ARM_JOINTS}, got={resolved}"
            )
        self._right_arm_joint_ids = tuple(int(index) for index in joint_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        capture_scale_m: float = 0.10,
        linear_velocity_scale: float = 0.50,
        angular_velocity_scale: float = 0.80,
        arm_position_scale: float = 0.20,
        arm_velocity_scale: float = 0.60,
    ) -> torch.Tensor:
        from training.tasks.tracking.mdp.observations import stagger_support_state

        command: MotionCommand = env.command_manager.get_term(command_name)
        robot = env.scene["robot"]
        arm_ids = self._right_arm_joint_ids
        arm_position_error = torch.mean(
            torch.square(
                robot.data.joint_pos[:, arm_ids]
                - command.ready_joint_pos[:, arm_ids]
            ),
            dim=-1,
        )
        arm_velocity_error = torch.mean(
            torch.square(robot.data.joint_vel[:, arm_ids]), dim=-1
        )
        capture_error = torch.square(
            stagger_support_state(env)["capture_rel_support_x_b"] / capture_scale_m
        )
        linear_error = torch.sum(
            torch.square(robot.data.root_lin_vel_b / linear_velocity_scale), dim=-1
        )
        angular_error = torch.sum(
            torch.square(robot.data.root_ang_vel_b / angular_velocity_scale), dim=-1
        )
        potential = (
            capture_error
            + linear_error
            + angular_error
            + arm_position_error / arm_position_scale**2
            + arm_velocity_error / arm_velocity_scale**2
        )
        progress = (self._previous - potential) / float(env.step_dt)
        active = _return_or_ready_mask(command)
        progress = torch.where(self._initialized & active, progress, torch.zeros_like(progress))
        self._previous[:] = potential
        self._initialized[:] = True
        return torch.where(active, torch.clamp(progress, -25.0, 25.0), torch.zeros_like(progress))


def post_strike_root_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    angular: bool = False,
) -> torch.Tensor:
    """Dense tail-only body-frame velocity cost for result-driven recovery.

    Unlike an exponential reward with a narrow standard deviation, this stays
    informative while the robot is still moving quickly.  It says only that
    the floating base must stop translating/rotating after the strike; it does
    not encode any preferred leg motion.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    velocity = robot.data.root_ang_vel_b if angular else robot.data.root_lin_vel_b
    cost = torch.sum(torch.square(velocity), dim=-1)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def post_strike_root_height_deficit_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_height: float,
) -> torch.Tensor:
    """Dense tail-only penalty for a base that keeps dropping after a swing."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    deficit = torch.relu(minimum_height - env.scene["robot"].data.root_pos_w[:, 2])
    cost = torch.square(deficit)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def post_strike_root_tilt_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalize developing roll/pitch throughout the finite recovery tail.

    Root-height termination only fires after a substantial fall has already
    developed.  This term supplies a dense, outcome-only signal from the
    first observable lean without constraining a particular leg or waist
    posture during the strike itself.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    gravity_b = env.scene["robot"].data.projected_gravity_b
    tilt_cost = torch.sum(torch.square(gravity_b[:, :2]), dim=-1)
    return torch.where(tail_steps > 0, tilt_cost, torch.zeros_like(tilt_cost))


def post_strike_torso_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torso_body_name: str = "torso_Link",
    deadband: float = 0.06,
) -> torch.Tensor:
    """Tail-only damping for visible torso roll/pitch sway.

    The strike and follow-through are intentionally left untouched.  Once the
    finite reference reaches its held final frame (``tail_steps > 0``), the
    torso's body-frame roll/pitch angular velocity is penalized outside a
    small sensor/noise deadband.  Yaw is excluded so the robot can retain the
    intended racket-facing heading while settling.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    cache_name = "_post_strike_torso_body_id"
    torso_id = getattr(env, cache_name, None)
    if torso_id is None:
        ids, names = robot.find_bodies([torso_body_name], preserve_order=True)
        if not names:
            raise ValueError(f"Unable to resolve torso body {torso_body_name!r}")
        torso_id = int(ids[0])
        setattr(env, cache_name, torso_id)
    torso_quat = robot.data.body_quat_w[:, torso_id]
    torso_ang_vel_w = robot.data.body_ang_vel_w[:, torso_id]
    torso_ang_vel_b = math_utils.quat_rotate_inverse(torso_quat, torso_ang_vel_w)
    excess = torch.relu(torch.abs(torso_ang_vel_b[:, :2]) - float(deadband))
    cost = torch.sum(torch.square(excess), dim=-1)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def post_strike_torso_tilt_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torso_body_name: str = "torso_Link",
) -> torch.Tensor:
    """Tail-only torso upright cost, independent of the yaw heading."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    cache_name = "_post_strike_torso_body_id"
    torso_id = getattr(env, cache_name, None)
    if torso_id is None:
        ids, names = robot.find_bodies([torso_body_name], preserve_order=True)
        if not names:
            raise ValueError(f"Unable to resolve torso body {torso_body_name!r}")
        torso_id = int(ids[0])
        setattr(env, cache_name, torso_id)
    torso_quat = robot.data.body_quat_w[:, torso_id]
    gravity_w = torch.zeros_like(robot.data.body_pos_w[:, torso_id])
    gravity_w[:, 2] = -1.0
    gravity_b = math_utils.quat_rotate_inverse(torso_quat, gravity_w)
    cost = torch.sum(torch.square(gravity_b[:, :2]), dim=-1)
    return torch.where(tail_steps > 0, cost, torch.zeros_like(cost))


def _reference_free_post_hit_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    follow_through_s: float = 0.15,
    ramp_s: float = 0.35,
) -> torch.Tensor:
    """Smooth recovery-only gate from a signed public time-to-hit command.

    V1.3B has exactly one target and does not expose a motion phase to its
    actor.  The command's signed ``time_to_strike`` is nevertheless a valid
    *environment-internal* event clock: it is positive before impact and
    stays negative afterwards.  This leaves the first 150 ms of natural
    follow-through unpenalized, then ramps recovery shaping to full strength
    over the next 350 ms.  No private motion/reference state is consulted.
    """

    command = env.command_manager.get_term(command_name)
    tau = getattr(command, "time_to_strike", None)
    if tau is None:
        raise ValueError(
            f"post-hit reference-free reward requires signed time_to_strike on {command_name!r}"
        )
    elapsed = torch.relu(-tau)
    u = ((elapsed - float(follow_through_s)) / max(float(ramp_s), 1.0e-6)).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def post_hit_goal_torso_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torso_body_name: str = "torso_Link",
    deadband: float = 0.06,
    follow_through_s: float = 0.15,
    ramp_s: float = 0.35,
) -> torch.Tensor:
    """After follow-through, damp torso roll/pitch motion without braking yaw."""

    robot = env.scene["robot"]
    cache_name = "_v13b_post_hit_torso_body_id"
    torso_id = getattr(env, cache_name, None)
    if torso_id is None:
        ids, names = robot.find_bodies([torso_body_name], preserve_order=True)
        if names != [torso_body_name]:
            raise ValueError(f"Unable to resolve torso body {torso_body_name!r}")
        torso_id = int(ids[0])
        setattr(env, cache_name, torso_id)
    torso_ang_vel_b = math_utils.quat_rotate_inverse(
        robot.data.body_quat_w[:, torso_id], robot.data.body_ang_vel_w[:, torso_id]
    )
    excess = torch.relu(torch.abs(torso_ang_vel_b[:, :2]) - float(deadband))
    return _reference_free_post_hit_gate(
        env, command_name, follow_through_s, ramp_s
    ) * torch.sum(torch.square(excess), dim=-1)


def post_hit_goal_torso_tilt_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torso_body_name: str = "torso_Link",
    follow_through_s: float = 0.15,
    ramp_s: float = 0.35,
) -> torch.Tensor:
    """After follow-through, restore torso roll/pitch uprightness, not yaw."""

    robot = env.scene["robot"]
    cache_name = "_v13b_post_hit_torso_body_id"
    torso_id = getattr(env, cache_name, None)
    if torso_id is None:
        ids, names = robot.find_bodies([torso_body_name], preserve_order=True)
        if names != [torso_body_name]:
            raise ValueError(f"Unable to resolve torso body {torso_body_name!r}")
        torso_id = int(ids[0])
        setattr(env, cache_name, torso_id)
    gravity_w = torch.zeros_like(robot.data.body_pos_w[:, torso_id])
    gravity_w[:, 2] = -1.0
    gravity_b = math_utils.quat_rotate_inverse(robot.data.body_quat_w[:, torso_id], gravity_w)
    return _reference_free_post_hit_gate(
        env, command_name, follow_through_s, ramp_s
    ) * torch.sum(torch.square(gravity_b[:, :2]), dim=-1)


def post_hit_goal_forward_velocity_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.08,
    follow_through_s: float = 0.15,
    ramp_s: float = 0.35,
) -> torch.Tensor:
    """After follow-through, discourage sustained fore-aft body drift only."""

    speed = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0])
    excess = torch.relu(speed - float(deadband))
    return _reference_free_post_hit_gate(
        env, command_name, follow_through_s, ramp_s
    ) * torch.square(excess)


def _through_hit_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Select prelude and strike states through the exact hit frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command._use_motion_library:
        hit = command.motion.hit_frame[command.motion_ids]
    else:
        hit = torch.full_like(command.time_steps, int(command.motion.hit_frame[0]))
    in_prelude = command.prelude_elapsed_steps < int(command.prelude_steps)
    return in_prelude | ((command.tail_steps == 0) & (command.time_steps <= hit))


def pre_hit_root_tilt_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Dense prelude-to-impact roll/pitch cost for anticipatory stabilization."""
    gravity_b = env.scene["robot"].data.projected_gravity_b
    cost = torch.sum(torch.square(gravity_b[:, :2]), dim=-1)
    return torch.where(_through_hit_mask(env, command_name), cost, torch.zeros_like(cost))


def pre_hit_root_angular_velocity_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize roll/pitch angular momentum before it becomes a recovery problem."""
    angular_velocity_b = env.scene["robot"].data.root_ang_vel_b[:, :2]
    cost = torch.sum(torch.square(angular_velocity_b), dim=-1)
    return torch.where(_through_hit_mask(env, command_name), cost, torch.zeros_like(cost))


def pre_hit_root_forward_velocity_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Keep body-frame fore-aft drift small through impact without forcing a lean direction."""
    forward_velocity = env.scene["robot"].data.root_lin_vel_b[:, 0]
    cost = torch.square(forward_velocity)
    return torch.where(_through_hit_mask(env, command_name), cost, torch.zeros_like(cost))


def _strike_approach_weight(
    env: ManagerBasedRLEnv, command_name: str, window_steps: int
) -> torch.Tensor:
    """Smooth 0..1 weight over the final reference steps before exact hit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command._use_motion_library:
        hit = command.motion.hit_frame[command.motion_ids]
    else:
        hit = torch.full_like(command.time_steps, int(command.motion.hit_frame[0]))
    in_swing = command.prelude_elapsed_steps >= int(command.prelude_steps)
    # Rewards observe the post-physics state one command update before
    # MotionCommand advances its index.
    physical_phase = command.time_steps + 1
    active = in_swing & (command.tail_steps == 0) & (physical_phase <= hit)
    start = hit - int(window_steps)
    u = ((physical_phase - start).float() / float(max(window_steps, 1))).clamp(0.0, 1.0)
    smooth = u * u * (3.0 - 2.0 * u)
    return torch.where(active, smooth, torch.zeros_like(smooth))


def strike_approach_pitch_rate_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    window_steps: int = 30,
    deadband: float = 0.06,
) -> torch.Tensor:
    """Symmetrically drive body pitch rate toward zero during swing approach."""
    pitch_rate = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 1])
    excess = torch.clamp(pitch_rate - float(deadband), min=0.0)
    return _strike_approach_weight(env, command_name, window_steps) * torch.square(excess)


def strike_approach_forward_velocity_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    window_steps: int = 30,
    deadband: float = 0.05,
) -> torch.Tensor:
    """Symmetrically drive body-frame fore-aft velocity toward zero before hit."""
    forward_speed = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0])
    excess = torch.clamp(forward_speed - float(deadband), min=0.0)
    return _strike_approach_weight(env, command_name, window_steps) * torch.square(excess)


def _exact_strike_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command._use_motion_library:
        hit = command.motion.hit_frame[command.motion_ids]
    else:
        hit = torch.full_like(command.time_steps, int(command.motion.hit_frame[0]))
    # ManagerBasedRLEnv computes rewards after physics but advances commands
    # afterward. Thus the physical state reported as motion step `hit` was
    # produced while command.time_steps still held `hit - 1`.
    return (
        (command.prelude_elapsed_steps >= int(command.prelude_steps))
        & (command.tail_steps == 0)
        & (command.time_steps + 1 == hit)
    )


def exact_strike_pitch_rate_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.06,
) -> torch.Tensor:
    """One-step impact cost whose integrated magnitude is timestep invariant."""
    pitch_rate = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 1])
    excess = torch.clamp(pitch_rate - float(deadband), min=0.0)
    cost = torch.square(excess) / float(env.step_dt)
    return torch.where(_exact_strike_mask(env, command_name), cost, torch.zeros_like(cost))


def exact_strike_forward_velocity_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.05,
) -> torch.Tensor:
    """Directly constrain body-frame forward speed at the exact impact frame."""
    forward_speed = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0])
    excess = torch.clamp(forward_speed - float(deadband), min=0.0)
    cost = torch.square(excess) / float(env.step_dt)
    return torch.where(_exact_strike_mask(env, command_name), cost, torch.zeros_like(cost))


def _post_hit_weight(
    env: ManagerBasedRLEnv,
    command_name: str,
    delay_steps: int = 2,
    ramp_steps: int = 8,
) -> torch.Tensor:
    """Smoothly enable recovery shaping after, never before, exact impact."""
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
    u = (
        (steps_after_hit - int(delay_steps)).float()
        / float(max(int(ramp_steps), 1))
    ).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def post_hit_forward_velocity_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.06,
    delay_steps: int = 2,
    ramp_steps: int = 8,
) -> torch.Tensor:
    """Symmetrically brake fore-aft motion without rewarding a sign reversal."""
    speed = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0])
    excess = torch.relu(speed - float(deadband))
    return _post_hit_weight(
        env, command_name, delay_steps, ramp_steps
    ) * torch.square(excess)


def post_hit_pitch_rate_deadband_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.08,
    delay_steps: int = 2,
    ramp_steps: int = 8,
) -> torch.Tensor:
    """Symmetrically brake pitch rotation after the strike."""
    rate = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 1])
    excess = torch.relu(rate - float(deadband))
    return _post_hit_weight(
        env, command_name, delay_steps, ramp_steps
    ) * torch.square(excess)


def post_hit_capture_point_center_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.04,
    delay_steps: int = 2,
    ramp_steps: int = 8,
) -> torch.Tensor:
    """Penalize sagittal capture-point distance from the support center."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    distance = torch.abs(stagger_support_state(env)["capture_rel_support_x_b"])
    excess = torch.relu(distance - float(deadband))
    return _post_hit_weight(
        env, command_name, delay_steps, ramp_steps
    ) * torch.square(excess)


def post_hit_capture_point_barrier_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_margin: float = 0.06,
    delay_steps: int = 2,
    ramp_steps: int = 8,
) -> torch.Tensor:
    """Increase braking pressure smoothly near either sagittal support edge."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    state = stagger_support_state(env)
    front = torch.relu(float(target_margin) - state["capture_front_margin"])
    rear = torch.relu(float(target_margin) - state["capture_rear_margin"])
    return _post_hit_weight(
        env, command_name, delay_steps, ramp_steps
    ) * (torch.square(front) + torch.square(rear))


class PostHitCapturePointCenterProgress(ManagerTermBase):
    """Reward capture-point motion toward center and penalize motion away."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(env.num_envs, device=env.device)
        self._active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._active[:] = False
        else:
            self._active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        delay_steps: int = 2,
        ramp_steps: int = 8,
    ) -> torch.Tensor:
        from training.tasks.tracking.mdp.observations import stagger_support_state

        weight = _post_hit_weight(env, command_name, delay_steps, ramp_steps)
        active = weight > 0.0
        distance = torch.abs(stagger_support_state(env)["capture_rel_support_x_b"])
        progress = (self._previous_distance - distance) / float(env.step_dt)
        progress = torch.where(self._active & active, progress, torch.zeros_like(progress))
        self._previous_distance[:] = distance
        self._active[:] = active
        return weight * torch.clamp(progress, min=-2.0, max=2.0)


class PostStrikeRootRecoveryProgress(ManagerTermBase):
    """Reward tail-only reduction of an observable floating-root instability potential.

    The term is outcome-based: it uses root tilt, body-frame linear/angular
    velocity, and height loss, but never encodes a preferred leg or waist
    posture.  Returning a rate makes the integrated reward equal the actual
    potential decrease after RewardManager applies the policy timestep.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tilt_scale: float = 0.35,
        linear_velocity_scale: float = 0.75,
        angular_velocity_scale: float = 1.25,
        minimum_height: float = 0.70,
        height_scale: float = 0.12,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        tail_steps = getattr(command, "tail_steps", None)
        if tail_steps is None:
            return torch.zeros(env.num_envs, device=env.device)

        robot = env.scene["robot"]
        gravity = robot.data.projected_gravity_b[:, :2]
        potential = torch.sum(torch.square(gravity / tilt_scale), dim=-1)
        potential += torch.sum(torch.square(robot.data.root_lin_vel_b / linear_velocity_scale), dim=-1)
        potential += torch.sum(torch.square(robot.data.root_ang_vel_b / angular_velocity_scale), dim=-1)
        height_deficit = torch.relu(minimum_height - robot.data.root_pos_w[:, 2])
        potential += torch.square(height_deficit / height_scale)

        progress = (self._previous - potential) / float(env.step_dt)
        progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
        self._previous[:] = potential
        self._initialized[:] = True
        # Bound a single contact transient without removing the direction of
        # the recovery signal.
        progress = torch.clamp(progress, min=-25.0, max=25.0)
        return torch.where(tail_steps > 0, progress, torch.zeros_like(progress))


def post_strike_both_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Tail-only result reward for both feet carrying contact load."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    tail_steps = getattr(command, "tail_steps", None)
    if tail_steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    both_feet = (force > threshold).all(dim=-1).to(dtype=torch.float32)
    return torch.where(tail_steps > 0, both_feet, torch.zeros_like(both_feet))


def feet_contact_fraction(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Fraction of designated feet with current contact force above threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return (force > threshold).to(dtype=torch.float32).mean(dim=-1)


def stagger_capture_point_margin_l2(
    env: ManagerBasedRLEnv,
    target_margin: float = 0.04,
) -> torch.Tensor:
    """Penalize a sagittal capture point that approaches either support edge."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    state = stagger_support_state(env)
    front_deficit = torch.relu(float(target_margin) - state["capture_front_margin"])
    rear_deficit = torch.relu(float(target_margin) - state["capture_rear_margin"])
    return torch.square(front_deficit) + torch.square(rear_deficit)


def stagger_lateral_capture_point_margin_l2(
    env: ManagerBasedRLEnv,
    target_margin: float = 0.035,
) -> torch.Tensor:
    """Penalize a lateral capture point approaching either widened support edge."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    state = stagger_support_state(env)
    positive_deficit = torch.relu(
        float(target_margin) - state["capture_lateral_positive_margin"]
    )
    negative_deficit = torch.relu(
        float(target_margin) - state["capture_lateral_negative_margin"]
    )
    return torch.square(positive_deficit) + torch.square(negative_deficit)


def stagger_minimum_foot_load(
    env: ManagerBasedRLEnv,
    minimum_body_weight_fraction: float = 0.08,
) -> torch.Tensor:
    """Reward keeping both feet meaningfully loaded without prescribing 50/50 load."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    state = stagger_support_state(env)
    minimum_load = state["normalized_load"].min(dim=-1).values
    load_score = torch.clamp(
        minimum_load / float(minimum_body_weight_fraction),
        min=0.0,
        max=1.0,
    )
    return load_score * state["contacts"].all(dim=-1).to(dtype=load_score.dtype)


def stagger_sagittal_span_l2(
    env: ManagerBasedRLEnv,
    target_span: float = 0.08,
    deadband: float = 0.015,
) -> torch.Tensor:
    """Keep the fore-aft stance from collapsing back toward parallel feet."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    span_error = torch.abs(stagger_support_state(env)["sagittal_span"] - float(target_span))
    return torch.square(torch.relu(span_error - float(deadband)))


def stagger_lateral_span_l2(
    env: ManagerBasedRLEnv,
    target_span: float = 0.42,
    deadband: float = 0.03,
) -> torch.Tensor:
    """Keep the widened stance from collapsing laterally during recovery."""
    from training.tasks.tracking.mdp.observations import stagger_support_state

    span_error = torch.abs(stagger_support_state(env)["lateral_span"] - float(target_span))
    return torch.square(torch.relu(span_error - float(deadband)))


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track reference joint positions for a selected joint subset."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.robot_joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids])
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense actual-vs-reference joint-velocity tracking score for P5D.

    This is reference-relative rather than goal-relative, so it supplies a
    useful signal across the full swing instead of only at the hit window.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.robot_joint_vel[:, asset_cfg.joint_ids] - command.joint_vel[:, asset_cfg.joint_ids])
    return torch.exp(-error.mean(-1) / std**2)


def action_raw_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Penalize raw policy residual magnitude for custom action terms."""
    action_term = env.action_manager.get_term(action_name)
    raw_actions = getattr(action_term, "_raw_actions", None)
    if raw_actions is None:
        raw_actions = env.action_manager.action
    return torch.mean(torch.square(raw_actions), dim=-1)


def action_subset_raw_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    action_indices: tuple[int, ...] = (),
) -> torch.Tensor:
    """Penalize a named coordinator action group without coupling its peers."""
    action_term = env.action_manager.get_term(action_name)
    raw_actions = getattr(action_term, "raw_actions", None)
    if raw_actions is None:
        raw_actions = env.action_manager.action
    if not action_indices:
        raise ValueError("action_subset_raw_l2 requires at least one action index")
    indices = torch.tensor(action_indices, dtype=torch.long, device=raw_actions.device)
    return torch.mean(torch.square(raw_actions.index_select(dim=-1, index=indices)), dim=-1)


def action_unbounded_excess_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    raw_limit: float = 0.20,
    action_indices: tuple[int, ...] = (),
) -> torch.Tensor:
    """Penalize latent actor output beyond a pre-saturation trust band.

    The action term applies masks and phase gates first, then exposes the
    resulting unbounded residual.  Penalizing that value prevents PPO from
    treating the bounded actuator envelope as a flat, free control region.
    ``action_indices`` keeps structurally masked channels (e.g. waist during
    Stage-A) out of the leg-stabilizer objective.
    """
    action_term = env.action_manager.get_term(action_name)
    raw_actions = getattr(action_term, "unbounded_actions", None)
    if raw_actions is None:
        raw_actions = env.action_manager.action
    if action_indices:
        ids = torch.tensor(action_indices, dtype=torch.long, device=raw_actions.device)
        raw_actions = raw_actions.index_select(dim=-1, index=ids)
    excess = torch.relu(torch.abs(raw_actions) - raw_limit)
    return torch.mean(torch.square(excess), dim=-1)


def action_execution_gap_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    action_indices: tuple[int, ...] = (),
    deadband: float = 0.02,
) -> torch.Tensor:
    """Penalize only the meaningful latent-to-executed action mismatch.

    The action term may structurally mask channels or phase-gate authority.
    This uses the post-gate latent command, then ignores a small smooth-bound
    deadband so normal near-zero compression is free while persistent
    saturation dependence remains visible to PPO.
    """
    action_term = env.action_manager.get_term(action_name)
    latent = getattr(action_term, "unbounded_actions", None)
    executed = getattr(action_term, "raw_actions", None)
    if latent is None or executed is None:
        return torch.zeros(env.num_envs, device=env.device)
    if action_indices:
        ids = torch.tensor(action_indices, dtype=torch.long, device=latent.device)
        latent = latent.index_select(dim=-1, index=ids)
        executed = executed.index_select(dim=-1, index=ids)
    excess = torch.relu(torch.abs(latent - executed) - deadband)
    return torch.mean(torch.square(excess), dim=-1)


def upper_execution_gap_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    deadband: float = 0.02,
) -> torch.Tensor:
    """Penalize floating-base dynamics that pull the frozen upper chain off target."""
    action_term = env.action_manager.get_term(action_name)
    target = getattr(action_term, "upper_processed_actions", None)
    joint_ids = getattr(action_term, "_upper_joint_ids_tensor", None)
    if target is None or joint_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    actual = env.scene["robot"].data.joint_pos[:, joint_ids]
    excess = torch.relu(torch.abs(actual - target) - float(deadband))
    return torch.mean(torch.square(excess), dim=-1)


def root_position_drift_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Softly discourage whole-body translation away from the in-place stance."""
    robot = env.scene["robot"]
    root_origin = robot.data.default_root_state[:, :3] + env.scene.env_origins
    drift = robot.data.root_pos_w - root_origin
    return torch.sum(torch.square(drift), dim=-1)


def feet_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize tangential foot motion only while the foot is load-bearing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    in_contact = (force > threshold).to(dtype=torch.float32)
    foot_velocity = env.scene["robot"].data.body_lin_vel_w[:, sensor_cfg.body_ids]
    tangential_speed_sq = torch.sum(torch.square(foot_velocity[..., :2]), dim=-1)
    return (tangential_speed_sq * in_contact).sum(dim=-1) / in_contact.sum(dim=-1).clamp_min(1.0)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

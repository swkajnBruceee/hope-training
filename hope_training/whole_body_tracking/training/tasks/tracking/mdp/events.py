from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_progressive_fall_assist(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_torque_nm: float = 55.0,
    kp_nm_per_rad: float = 75.0,
    kd_nms_per_rad: float = 8.0,
    tilt_deadband_rad: float = 0.05,
    anneal_steps: int = 12000,
    minimum_scale: float = 0.10,
    adaptive_enabled: bool = True,
    adapt_interval_steps: int = 24,
    failure_rate_high: float = 0.0025,
    failure_rate_low: float = 0.0008,
    adaptive_increase: float = 0.03,
    adaptive_decrease: float = 0.01,
    torso_max_torque_nm: float = 90.0,
    torso_kp_nm_per_rad: float = 140.0,
    torso_kd_nms_per_rad: float = 3.0,
    emergency_tilt_rad: float = 0.35,
    emergency_gain: float = 5.0,
    torso_emergency_tilt_rad: float = 0.35,
    torso_emergency_gain: float = 7.0,
    max_force_n: float = 260.0,
    force_kp_n_per_rad: float = 420.0,
    force_kd_ns_per_mps: float = 35.0,
    torso_max_force_n: float = 160.0,
    asset_cfg: SceneEntityCfg | None = None,
):
    """Apply a bounded, opposite-direction anti-fall wrench during bootstrap PPO.

    ``projected_gravity_b`` is the gravity direction in the pelvis frame.  The
    restoring moment ``[g_y, -g_x, 0]`` is the shortest body-frame moment that
    brings the body-up vector back toward world-up; angular-rate damping avoids
    injecting an oscillation.  The wrench is held by IsaacLab's external-wrench
    buffer and therefore acts at every physics step until the next update.

    The assist is deliberately a curriculum aid, not a replacement controller:
    its scale decreases from one to ``minimum_scale`` over ``anneal_steps``
    control steps.  It never writes pose or velocity and does not disable the
    physical fall termination.
    """
    robot = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    gravity_b = robot.data.projected_gravity_b
    gravity_norm = torch.linalg.vector_norm(gravity_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
    gravity_dir = gravity_b / gravity_norm
    tilt = torch.atan2(
        torch.linalg.vector_norm(gravity_dir[:, :2], dim=-1),
        (-gravity_dir[:, 2]).clamp_min(1.0e-5),
    )
    active = tilt > float(tilt_deadband_rad)

    # Use the environment-global control counter when available.  This keeps
    # the assistance schedule monotonic across per-env resets.
    step_counter = getattr(env, "common_step_counter", 0)
    if torch.is_tensor(step_counter):
        step_counter = int(step_counter.detach().item())
    else:
        step_counter = int(step_counter)
    if anneal_steps > 0:
        progress = min(max(step_counter / float(anneal_steps), 0.0), 1.0)
        schedule_scale = max(float(minimum_scale), 1.0 - progress * (1.0 - float(minimum_scale)))
    else:
        schedule_scale = 1.0

    # Performance-aware curriculum: the scheduled scale is a lower bound, so
    # the assist cannot disappear before the configured horizon.  A recent
    # strict-fall rate above the high threshold raises the global scale; only
    # a genuinely low failure rate permits it to decay toward the schedule.
    scale = schedule_scale
    if adaptive_enabled:
        adaptive_scale = float(getattr(env, "fall_assist_adaptive_scale", 1.0))
        last_adapt_step = int(getattr(env, "fall_assist_last_adapt_step", -1))
        interval = max(int(adapt_interval_steps), 1)
        if step_counter == 0 and last_adapt_step < 0:
            adaptive_scale = 1.0
        if step_counter - last_adapt_step >= interval:
            failure_rate = None
            try:
                tm = getattr(env, "termination_manager", None)
                if tm is not None and "strict_fall" in tm.active_terms:
                    failure_rate = float(tm.get_term("strict_fall").float().mean().item())
            except Exception:
                failure_rate = None
            if failure_rate is not None:
                if failure_rate > float(failure_rate_high):
                    adaptive_scale = min(1.0, adaptive_scale + float(adaptive_increase))
                elif failure_rate < float(failure_rate_low):
                    adaptive_scale = max(schedule_scale, adaptive_scale - float(adaptive_decrease))
                else:
                    adaptive_scale = max(schedule_scale, adaptive_scale)
            else:
                adaptive_scale = max(schedule_scale, adaptive_scale)
            setattr(env, "fall_assist_adaptive_scale", adaptive_scale)
            setattr(env, "fall_assist_last_adapt_step", step_counter)
        scale = max(schedule_scale, adaptive_scale)
        setattr(env, "fall_assist_scale", scale)

    # Restoring body-frame moment plus angular-rate damping at the pelvis.
    emergency_factor = 1.0 + float(emergency_gain) * torch.relu(tilt - float(emergency_tilt_rad))
    kp_term = float(kp_nm_per_rad) * emergency_factor[:, None] * gravity_dir[:, :2]
    rate_term = -float(kd_nms_per_rad) * robot.data.root_ang_vel_b[:, :2]
    torque_xy = (kp_term + rate_term) * scale
    torque_norm = torch.linalg.vector_norm(torque_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    torque_xy = torque_xy * (float(max_torque_nm) / torque_norm).clamp_max(1.0)
    torque_xy = torch.where(active[:, None], torque_xy, torch.zeros_like(torque_xy))

    # The strict fall contract also monitors torso_Link.  A pelvis-only wrench
    # cannot arrest a rapidly folding upper body, so apply a separate bounded
    # restoring moment to the torso.  Both mappings are explicit and fail
    # loudly if the A3 asset changes.
    body_ids, body_names = robot.find_bodies(["pelvis_link", "torso_Link"], preserve_order=True)
    if body_names != ["pelvis_link", "torso_Link"]:
        raise RuntimeError(f"fall-assist body mapping mismatch: {body_names}")
    torso_id = int(body_ids[1])
    torso_quat_w = robot.data.body_quat_w[:, torso_id]
    gravity_w = robot.data.GRAVITY_VEC_W.to(device=robot.device, dtype=gravity_b.dtype).expand(robot.num_instances, -1)
    torso_gravity_b = math_utils.quat_rotate_inverse(torso_quat_w, gravity_w)
    torso_gravity_b = torso_gravity_b / torch.linalg.vector_norm(torso_gravity_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
    torso_ang_vel_w = robot.data.body_ang_vel_w[:, torso_id]
    torso_ang_vel_b = math_utils.quat_rotate_inverse(torso_quat_w, torso_ang_vel_w)
    torso_tilt = torch.atan2(
        torch.linalg.vector_norm(torso_gravity_b[:, :2], dim=-1),
        (-torso_gravity_b[:, 2]).clamp_min(1.0e-5),
    )
    torso_emergency_factor = 1.0 + float(torso_emergency_gain) * torch.relu(torso_tilt - float(torso_emergency_tilt_rad))
    torso_torque_xy = (
        float(torso_kp_nm_per_rad) * torso_emergency_factor[:, None] * torso_gravity_b[:, :2]
        - float(torso_kd_nms_per_rad) * torso_ang_vel_b[:, :2]
    ) * scale
    torso_norm = torch.linalg.vector_norm(torso_torque_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    torso_torque_xy = torso_torque_xy * (float(torso_max_torque_nm) / torso_norm).clamp_max(1.0)
    torso_torque_xy = torch.where(torso_tilt[:, None] > float(tilt_deadband_rad), torso_torque_xy, torch.zeros_like(torso_torque_xy))

    # A bounded horizontal catch force removes developing fall momentum.  It
    # is expressed in each body's local frame (the IsaacLab contract) and is
    # proportional to the projected gravity direction plus velocity damping.
    # The force is zero while upright, so it does not act as a continuous pose
    # clamp during the initial standing phase.
    root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", torch.zeros_like(gravity_b))
    force_xy = (
        -float(force_kp_n_per_rad) * gravity_dir[:, :2]
        - float(force_kd_ns_per_mps) * root_lin_vel_b[:, :2]
    ) * scale
    force_norm = torch.linalg.vector_norm(force_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    force_xy = force_xy * (float(max_force_n) / force_norm).clamp_max(1.0)
    force_xy = torch.where(active[:, None], force_xy, torch.zeros_like(force_xy))
    torso_force_xy = (
        -float(force_kp_n_per_rad) * torso_gravity_b[:, :2]
        - float(force_kd_ns_per_mps) * torso_ang_vel_b[:, :2]
    ) * scale
    torso_force_norm = torch.linalg.vector_norm(torso_force_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    torso_force_xy = torso_force_xy * (float(torso_max_force_n) / torso_force_norm).clamp_max(1.0)
    torso_force_xy = torch.where(torso_tilt[:, None] > float(tilt_deadband_rad), torso_force_xy, torch.zeros_like(torso_force_xy))

    forces = torch.zeros((robot.num_instances, 2, 3), device=robot.device, dtype=gravity_b.dtype)
    forces[:, 0, :2] = force_xy
    forces[:, 1, :2] = torso_force_xy
    torques = torch.zeros((robot.num_instances, 2, 3), device=robot.device, dtype=gravity_b.dtype)
    torques[:, 0, :2] = torque_xy
    torques[:, 1, :2] = torso_torque_xy
    robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)


def sample_strike_stabilizer_handoff_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    full_swing_probability: float,
    candidate_steps: tuple[int, ...],
):
    """Sample a policy handoff phase without teleporting simulator state.

    The Stage-A action term holds leg residuals at zero before this step. The
    physics still runs from the start of the swing, so the policy never learns
    an artificial reset/contact impulse.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "strike_stabilizer_handoff_steps"):
        env.strike_stabilizer_handoff_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    draw_full = torch.rand(len(env_ids), device=env.device) < float(full_swing_probability)
    candidates = torch.as_tensor(candidate_steps, dtype=torch.long, device=env.device)
    sampled = candidates[torch.randint(len(candidates), (len(env_ids),), device=env.device)]
    env.strike_stabilizer_handoff_steps[env_ids] = torch.where(
        draw_full, torch.zeros_like(sampled), sampled
    )


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        env_index = env_ids
        if isinstance(env_ids, torch.Tensor) and isinstance(joint_ids, torch.Tensor):
            env_index = env_ids[:, None]
        asset.data.default_joint_pos[env_index, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_index, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

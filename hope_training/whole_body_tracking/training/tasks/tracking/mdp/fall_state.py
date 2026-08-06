"""Unified physical fall/recovery state for A3 strike cycles.

This module is intentionally the single source of physical stability state.
It keeps the semantic distinction between a developing risk, a prediction that
the remaining recovery time is insufficient, a physically confirmed fall, and
being ready to start another strike.  The state is cached per environment and
updated at most once per simulator control step so a reward and a termination
term cannot advance different debounce counters.

The horizontal frame is frozen on the first valid observation after reset.  Its
forward axis is the initial root-heading projection and its lateral axis is
the left-handed perpendicular.  It is never recomputed from the current yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

try:  # keep the signed-frame helpers dependency-free for unit tests
    from isaaclab.utils.math import quat_apply
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI
    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        q_xyz = quat[..., 1:]
        q_w = quat[..., :1]
        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


class FallLevel(IntEnum):
    STABLE = 0
    AT_RISK = 1
    RECOVERABLE_UNSTABLE = 2
    EMERGENCY = 3
    PREDICTED_UNRECOVERABLE = 4
    FALL_CONFIRMED = 5
    RECOVERING = 6
    RECOVERY_READY = 7


class FallReason(IntEnum):
    NONE = 0
    FORWARD_FALL = 1
    BACKWARD_FALL = 2
    LEFT_FALL = 3
    RIGHT_FALL = 4
    FOOT_SLIP_FALL = 5
    CONTACT_LOSS_FALL = 6
    ACTUATOR_LIMITED_FALL = 7
    ILLEGAL_BODY_CONTACT = 8
    RECOVERY_TIMEOUT = 9


@dataclass(frozen=True)
class FallStateConfig:
    control_dt_s: float = 0.02
    debounce_steps: int = 3
    recovery_hold_steps: int = 15
    root_tilt_at_risk_rad: float = 0.30
    root_tilt_emergency_rad: float = 0.785398
    torso_tilt_at_risk_rad: float = 0.30
    torso_tilt_confirm_rad: float = 0.785398
    root_height_drop_confirm_m: float = 0.30
    relative_height_min_m: float = 0.82
    angular_rate_emergency_radps: float = 1.8
    com_speed_at_risk_mps: float = 0.35
    contact_force_threshold_n: float = 10.0
    illegal_contact_threshold_n: float = 1.0
    slip_speed_threshold_mps: float = 0.08
    safety_projection_threshold: float = 0.05
    min_capture_margin_m: float = 0.0
    predicted_margin_threshold_m: float = -0.04
    predicted_tilt_threshold_rad: float = 0.75
    recovery_relative_height_m: float = 0.88
    recovery_tilt_rad: float = 0.20
    recovery_rate_radps: float = 0.45
    recovery_com_speed_mps: float = 0.18
    recovery_margin_m: float = 0.015
    prediction_horizons_s: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)


@dataclass
class FallStateOutput:
    risk_score: torch.Tensor
    risk_level: torch.Tensor
    fall_direction: torch.Tensor
    fall_reason: torch.Tensor
    recoverability: torch.Tensor
    predicted_unrecoverable: torch.Tensor
    confirmed_fall: torch.Tensor
    recovery_ready: torch.Tensor
    recovery_progress: torch.Tensor
    recovery_stable_steps: torch.Tensor
    forward_tilt_rad: torch.Tensor
    lateral_tilt_rad: torch.Tensor
    torso_forward_tilt_rad: torch.Tensor
    torso_lateral_tilt_rad: torch.Tensor
    forward_tilt_rate_radps: torch.Tensor
    lateral_tilt_rate_radps: torch.Tensor
    relative_root_height_m: torch.Tensor
    relative_torso_height_m: torch.Tensor
    root_linear_velocity_b: torch.Tensor
    root_angular_velocity_b: torch.Tensor
    torso_linear_velocity_b: torch.Tensor
    torso_angular_velocity_b: torch.Tensor
    com_position_b: torch.Tensor
    com_velocity_b: torch.Tensor
    capture_point_b: torch.Tensor
    cop_position_b: torch.Tensor
    support_margins: torch.Tensor
    predicted_support_margins: torch.Tensor
    foot_contact: torch.Tensor
    foot_slip_mps: torch.Tensor
    illegal_body_contact: torch.Tensor
    actuator_saturation: torch.Tensor
    safety_projection: torch.Tensor
    risk_components: dict[str, torch.Tensor]
    prediction: dict[str, torch.Tensor]
    cycle_state: torch.Tensor


def _safe_unit_xy(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _tilt_in_heading(up_w: torch.Tensor, forward_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed forward/left tilt; positive means forward/left lean."""
    world_up = torch.zeros_like(up_w)
    world_up[:, 2] = 1.0
    lateral_w = torch.stack((-forward_w[:, 1], forward_w[:, 0], torch.zeros_like(forward_w[:, 0])), dim=-1)
    forward = torch.atan2((up_w * forward_w).sum(-1), (up_w * world_up).sum(-1).clamp_min(1.0e-5))
    lateral = torch.atan2((up_w * lateral_w).sum(-1), (up_w * world_up).sum(-1).clamp_min(1.0e-5))
    return forward, lateral


def signed_tilt_from_up(up_w: torch.Tensor, initial_forward_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure helper used by tests and runtime: signed tilt in an immutable frame."""
    if up_w.shape != initial_forward_w.shape or up_w.shape[-1] != 3:
        raise ValueError("up_w and initial_forward_w must have shape [N, 3]")
    return _tilt_in_heading(up_w, _safe_unit_xy(initial_forward_w))


def debounce_counter(counter: torch.Tensor, condition: torch.Tensor, required_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure 3--5-step debounce primitive used by the runtime and tests."""
    if counter.shape != condition.shape:
        raise ValueError("counter and condition shapes must match")
    if required_steps < 1:
        raise ValueError("required_steps must be positive")
    next_counter = torch.where(condition, counter + 1, torch.zeros_like(counter))
    return next_counter, next_counter >= required_steps


def recovery_ready_gate(
    *,
    double_foot_contact: torch.Tensor,
    illegal_contact: torch.Tensor,
    relative_height: torch.Tensor,
    tilt_rad: torch.Tensor,
    rate_radps: torch.Tensor,
    com_speed_mps: torch.Tensor,
    capture_min_margin_m: torch.Tensor,
    foot_slip_mps: torch.Tensor,
    height_min_m: float,
    tilt_max_rad: float,
    rate_max_radps: float,
    com_speed_max_mps: float,
    margin_min_m: float,
    slip_max_mps: float,
) -> torch.Tensor:
    """Pure physical READY predicate; a caller must still apply hold steps."""
    return (
        double_foot_contact
        & (~illegal_contact)
        & (relative_height >= height_min_m)
        & (torch.abs(tilt_rad) <= tilt_max_rad)
        & (torch.abs(rate_radps) <= rate_max_radps)
        & (com_speed_mps <= com_speed_max_mps)
        & (capture_min_margin_m >= margin_min_m)
        & (foot_slip_mps <= slip_max_mps)
    )


class UnifiedFallState:
    """Stateful physical fall detector/recovery gate shared by all consumers."""

    def __init__(self, env: Any, cfg: FallStateConfig = FallStateConfig()):
        self.cfg = cfg
        self.num_envs = int(env.num_envs)
        self.device = env.device
        n = self.num_envs
        self.initial_forward_w = torch.zeros((n, 3), device=self.device)
        self.initial_forward_w[:, 0] = 1.0
        self.initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.previous_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.previous_forward_tilt = torch.zeros(n, device=self.device)
        self.previous_lateral_tilt = torch.zeros(n, device=self.device)
        self.relative_height_baseline = torch.full((n,), 1.0, device=self.device)
        self.confirm_counter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.recovery_counter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.recovery_timeout_counter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last: FallStateOutput | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.initialized[ids] = False
        self.previous_step[ids] = -1
        self.previous_forward_tilt[ids] = 0.0
        self.previous_lateral_tilt[ids] = 0.0
        self.relative_height_baseline[ids] = 1.0
        self.confirm_counter[ids] = 0
        self.recovery_counter[ids] = 0
        self.recovery_timeout_counter[ids] = 0
        self.last = None

    def _cache_bodies(self, env: Any) -> dict[str, Any]:
        cache = getattr(env, "_unified_fall_state_body_cache", None)
        if cache is not None:
            return cache
        robot = env.scene["robot"]
        from training.robots.agibot_a3 import A3_FEET_BODIES

        feet, feet_names = robot.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if feet_names != A3_FEET_BODIES:
            raise RuntimeError(f"Unified fall state foot mapping mismatch: {feet_names}")
        torso, torso_names = robot.find_bodies(["torso_Link"], preserve_order=True)
        if not torso_names:
            raise RuntimeError("Unified fall state requires torso_Link")
        sensor = env.scene.sensors["contact_forces"]
        sensor_feet, sensor_names = sensor.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if sensor_names != A3_FEET_BODIES:
            raise RuntimeError(f"Unified fall state contact-foot mapping mismatch: {sensor_names}")
        sensor_names_all = tuple(getattr(sensor, "body_names", ()))
        if not sensor_names_all:
            sensor_names_all = tuple(getattr(robot, "body_names", ()))
        foot_set = set(A3_FEET_BODIES)
        illegal_ids = [i for i, name in enumerate(sensor_names_all) if name not in foot_set]
        cache = {
            "feet": feet, "torso": int(torso[0]), "sensor_feet": sensor_feet,
            "sensor_illegal": torch.as_tensor(illegal_ids, dtype=torch.long, device=self.device),
            "masses": robot.data.default_mass.to(device=self.device),
        }
        env._unified_fall_state_body_cache = cache
        return cache

    def update(self, env: Any, *, max_tilt_rad: float | None = None,
               minimum_height: float | None = None, max_torso_tilt_rad: float | None = None,
               minimum_torso_height: float | None = None, required_steps: int | None = None) -> FallStateOutput:
        episode_step = getattr(env, "episode_length_buf", None)
        if episode_step is None:
            episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        rewound = (self.previous_step >= 0) & (episode_step < self.previous_step)
        if rewound.any():
            # Some tasks expose the risk term without the strict termination
            # term; detect their vector-env reset here so the immutable frame,
            # debounce and recovery counters cannot leak across episodes.
            self.reset(torch.where(rewound)[0])
        # Reward and termination terms may both call us at one simulator step.
        if self.last is not None and torch.equal(episode_step, self.previous_step):
            return self.last
        robot = env.scene["robot"]
        cache = self._cache_bodies(env)
        root_quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w
        heading = quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1))
        heading[:, 2] = 0.0
        heading = _safe_unit_xy(heading)
        new = ~self.initialized
        self.initial_forward_w[new] = heading[new]
        self.initialized[new] = True
        forward_w = self.initial_forward_w
        root_up = quat_apply(root_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
        root_forward, root_lateral = _tilt_in_heading(root_up, forward_w)
        torso_quat = robot.data.body_quat_w[:, cache["torso"]]
        torso_pos = robot.data.body_pos_w[:, cache["torso"]]
        torso_up = quat_apply(torso_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
        torso_forward, torso_lateral = _tilt_in_heading(torso_up, forward_w)
        feet_pos = robot.data.body_pos_w[:, cache["feet"]]
        feet_xy = feet_pos[:, :, :2]
        root_xy = root_pos[:, :2]
        rel_xy = feet_xy - root_xy.unsqueeze(1)
        lateral = torch.stack((-forward_w[:, 1], forward_w[:, 0]), dim=-1)
        feet_b = torch.stack(((rel_xy * forward_w[:, :2].unsqueeze(1)).sum(-1), (rel_xy * lateral.unsqueeze(1)).sum(-1)), dim=-1)
        rear = feet_b[:, :, 0].min(-1).values - 0.10
        front = feet_b[:, :, 0].max(-1).values + 0.10
        left = feet_b[:, :, 1].max(-1).values + 0.055
        right = -feet_b[:, :, 1].min(-1).values + 0.055
        masses = cache["masses"]
        total_mass = masses.sum(-1).clamp_min(1e-6)
        com_pos = (robot.data.body_com_pos_w * masses.unsqueeze(-1)).sum(1) / total_mass.unsqueeze(-1)
        com_vel = (robot.data.body_com_lin_vel_w * masses.unsqueeze(-1)).sum(1) / total_mass.unsqueeze(-1)
        com_rel = com_pos[:, :2] - root_xy
        com_b = torch.stack(((com_rel * forward_w[:, :2]).sum(-1), (com_rel * lateral).sum(-1)), dim=-1)
        com_vel_b = torch.stack(((com_vel[:, :2] * forward_w[:, :2]).sum(-1), (com_vel[:, :2] * lateral).sum(-1)), dim=-1)
        support_z = feet_pos[:, :, 2].mean(-1)
        rel_height = root_pos[:, 2] - support_z
        torso_rel_height = torso_pos[:, 2] - support_z
        self.relative_height_baseline[new] = rel_height[new].detach()
        height_drop = self.relative_height_baseline - rel_height
        com_height = (com_pos[:, 2] - support_z).clamp(0.45, 1.50)
        omega = torch.sqrt(9.81 / com_height)
        capture = com_b + com_vel_b / omega.unsqueeze(-1)
        margins = torch.stack((front - capture[:, 0], capture[:, 0] - rear, left - capture[:, 1], right + capture[:, 1]), dim=-1)
        dt = float(getattr(env, "step_dt", self.cfg.control_dt_s))
        step_delta = (episode_step - self.previous_step).clamp_min(1).to(torch.float32) * dt
        f_rate = (root_forward - self.previous_forward_tilt) / step_delta
        l_rate = (root_lateral - self.previous_lateral_tilt) / step_delta
        # The first sample after reset establishes the derivative reference;
        # it is not a physical one-step tilt impulse.
        f_rate = torch.where(new, torch.zeros_like(f_rate), f_rate)
        l_rate = torch.where(new, torch.zeros_like(l_rate), l_rate)
        self.previous_forward_tilt = root_forward.detach()
        self.previous_lateral_tilt = root_lateral.detach()
        self.previous_step = episode_step.detach().clone()
        sensor = env.scene.sensors["contact_forces"]
        foot_force = sensor.data.net_forces_w[:, cache["sensor_feet"]]
        foot_contact = torch.linalg.vector_norm(foot_force, dim=-1) > self.cfg.contact_force_threshold_n
        vertical_load = foot_force[..., 2].clamp_min(0.0)
        cop_position_b = (feet_b * vertical_load.unsqueeze(-1)).sum(1) / vertical_load.sum(-1, keepdim=True).clamp_min(1.0)
        foot_vel = robot.data.body_lin_vel_w[:, cache["feet"], :2]
        foot_slip = torch.linalg.vector_norm(foot_vel, dim=-1) * foot_contact.to(foot_vel.dtype)
        if cache["sensor_illegal"].numel():
            illegal_force = torch.linalg.vector_norm(sensor.data.net_forces_w[:, cache["sensor_illegal"]], dim=-1)
            illegal_contact = illegal_force.max(-1).values > self.cfg.illegal_contact_threshold_n
        else:
            illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        joint_pos = robot.data.joint_pos
        limits = robot.data.soft_joint_pos_limits
        if limits.ndim == 2:
            limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)
        if limits.ndim != 3 or limits.shape[-1] != 2:
            raise RuntimeError("Unified fall state requires soft_joint_pos_limits [N, J, 2]")
        margin = torch.minimum(joint_pos - limits[..., 0], limits[..., 1] - joint_pos)
        position_sat = (margin < 0.03).to(torch.float32).mean(-1)
        # Do not divide joint velocity by an effort limit: those are different
        # physical units and used to produce meaningless saturation values.
        # Prefer the simulator's velocity limits when available; otherwise the
        # velocity component remains zero and torque saturation is left to the
        # existing action/safety diagnostics.
        vel_ratio = torch.zeros_like(position_sat)
        velocity_limits = getattr(robot.data, "soft_joint_vel_limits", None)
        if velocity_limits is None:
            velocity_limits = getattr(robot.data, "joint_vel_limits", None)
        if velocity_limits is not None:
            if velocity_limits.ndim == 1:
                velocity_limits = velocity_limits.unsqueeze(0).expand(self.num_envs, -1)
            vel_ratio = (torch.abs(robot.data.joint_vel) / velocity_limits.clamp_min(1e-3)).mean(-1).clamp(0, 1)
        actuator_sat = torch.maximum(position_sat, vel_ratio)
        safety_projection = torch.zeros(self.num_envs, device=self.device)
        try:
            action_term = env.action_manager.get_term("joint_pos")
            overrides = []
            for name in ("safety_override", "upper_safety_override", "upper_velocity_safety_override", "upper_dynamic_safety_override", "upper_dynamic_velocity_safety_override"):
                value = getattr(action_term, name, None)
                if isinstance(value, torch.Tensor):
                    overrides.append(torch.linalg.vector_norm(value, dim=-1))
            if overrides:
                safety_projection = torch.stack(overrides, dim=-1).max(dim=-1).values
        except Exception:
            pass
        min_margin = margins.min(-1).values
        tilt_risk = torch.relu(torch.abs(root_forward) - self.cfg.root_tilt_at_risk_rad) / max(self.cfg.root_tilt_emergency_rad - self.cfg.root_tilt_at_risk_rad, 1e-3)
        lateral_risk = torch.relu(torch.abs(root_lateral) - self.cfg.root_tilt_at_risk_rad) / max(self.cfg.root_tilt_emergency_rad - self.cfg.root_tilt_at_risk_rad, 1e-3)
        rate_risk = torch.relu(torch.abs(f_rate) + torch.abs(l_rate) - 0.5) / 2.0
        capture_risk = torch.relu(-min_margin) / 0.15
        velocity_risk = torch.relu(torch.linalg.vector_norm(com_vel_b, dim=-1) - self.cfg.com_speed_at_risk_mps) / 1.0
        contact_risk = (~foot_contact.all(-1)).to(torch.float32)
        slip_risk = torch.relu(foot_slip.max(-1).values - self.cfg.slip_speed_threshold_mps) / 0.20
        height_risk = torch.relu(height_drop - 0.08) / max(self.cfg.root_height_drop_confirm_m, 1e-3)
        risk_components = {
            "forward_tilt": tilt_risk.clamp(0, 1), "lateral_tilt": lateral_risk.clamp(0, 1),
            "torso_tilt": torch.relu(torch.maximum(torch.abs(torso_forward), torch.abs(torso_lateral)) - self.cfg.torso_tilt_at_risk_rad).clamp(0, 1),
            "tilt_rate": rate_risk.clamp(0, 1), "capture_margin": capture_risk.clamp(0, 1),
            "com_velocity": velocity_risk.clamp(0, 1), "support_contact": contact_risk,
            "foot_slip": slip_risk.clamp(0, 1), "actuator_saturation": actuator_sat.clamp(0, 1),
            "root_height_drop": height_risk.clamp(0, 1),
            "safety_projection": (safety_projection / max(self.cfg.safety_projection_threshold, 1e-6)).clamp(0, 1),
        }
        torso_risk = risk_components["torso_tilt"]
        risk_score = (0.16 * tilt_risk + 0.07 * lateral_risk + 0.07 * torso_risk + 0.11 * rate_risk + 0.17 * capture_risk + 0.10 * velocity_risk + 0.10 * contact_risk + 0.08 * slip_risk + 0.05 * actuator_sat + 0.04 * height_risk + 0.05 * risk_components["safety_projection"]).clamp(0, 1)
        horizons = torch.as_tensor(self.cfg.prediction_horizons_s, device=self.device, dtype=torch.float32)
        predicted_tilt = torch.stack([torch.abs(root_forward + f_rate * h) for h in horizons], dim=-1)
        predicted_height = torch.stack([rel_height - robot.data.root_lin_vel_w[:, 2] * h for h in horizons], dim=-1)
        # Margin signs follow the support polygon definition above: moving
        # forward consumes the front margin but increases the rear margin;
        # moving left consumes the left margin but increases the right one.
        margin_rate = torch.stack((f_rate, -f_rate, l_rate, -l_rate), dim=-1)
        predicted_margins = torch.stack([margins - margin_rate * h for h in horizons], dim=-1)
        predicted_min_margin = predicted_margins.min(dim=1).values.min(dim=-1).values
        predicted_unrecoverable = (predicted_tilt.max(-1).values > self.cfg.predicted_tilt_threshold_rad) | (predicted_min_margin < self.cfg.predicted_margin_threshold_m)
        torso_tilt = torch.maximum(torch.abs(torso_forward), torch.abs(torso_lateral))
        root_linear_velocity_b = getattr(robot.data, "root_lin_vel_b", torch.zeros((self.num_envs, 3), device=self.device))
        root_angular_velocity_b = getattr(robot.data, "root_ang_vel_b", torch.zeros((self.num_envs, 3), device=self.device))
        torso_linear_velocity_w = getattr(robot.data, "body_lin_vel_w", torch.zeros_like(robot.data.body_pos_w))[:, cache["torso"]]
        torso_angular_velocity_w = getattr(robot.data, "body_ang_vel_w", torch.zeros_like(robot.data.body_pos_w))[:, cache["torso"]]
        # Project torso world velocities into the same immutable initial frame
        # used for signed tilt; z remains world-up and is not affected by yaw.
        torso_linear_velocity_b = torch.stack(
            ((torso_linear_velocity_w[:, :2] * forward_w[:, :2]).sum(-1),
             (torso_linear_velocity_w[:, :2] * lateral).sum(-1),
             torso_linear_velocity_w[:, 2]), dim=-1)
        torso_angular_velocity_b = torch.stack(
            ((torso_angular_velocity_w[:, :2] * forward_w[:, :2]).sum(-1),
             (torso_angular_velocity_w[:, :2] * lateral).sum(-1),
             torso_angular_velocity_w[:, 2]), dim=-1)
        hard = illegal_contact | (torso_tilt > (max_torso_tilt_rad if max_torso_tilt_rad is not None else self.cfg.torso_tilt_confirm_rad)) | (height_drop > self.cfg.root_height_drop_confirm_m) | (rel_height < (minimum_height if minimum_height is not None else self.cfg.relative_height_min_m)) | (torso_rel_height < (minimum_torso_height if minimum_torso_height is not None else self.cfg.relative_height_min_m))
        composite = (torch.abs(root_forward) > (max_tilt_rad if max_tilt_rad is not None else self.cfg.root_tilt_emergency_rad)) & ((torch.abs(f_rate) + torch.abs(l_rate)) > 0.4)
        composite |= (min_margin < -0.05) & (~foot_contact.all(-1))
        composite |= (torch.abs(torso_forward) > (max_torso_tilt_rad if max_torso_tilt_rad is not None else self.cfg.torso_tilt_confirm_rad))
        self.confirm_counter, debounced_confirmed = debounce_counter(
            self.confirm_counter,
            hard | composite,
            int(required_steps if required_steps is not None else self.cfg.debounce_steps),
        )
        confirmed = hard | debounced_confirmed
        recoverable = (1.0 - risk_score).clamp(0, 1)
        ready_now = recovery_ready_gate(
            double_foot_contact=foot_contact.all(-1), illegal_contact=illegal_contact,
            relative_height=rel_height,
            tilt_rad=torch.maximum(torch.abs(root_forward), torch.abs(root_lateral)),
            rate_radps=torch.sqrt(f_rate.square() + l_rate.square()),
            com_speed_mps=torch.linalg.vector_norm(com_vel_b, dim=-1),
            capture_min_margin_m=margins.min(-1).values,
            foot_slip_mps=foot_slip.max(-1).values,
            # The termination's hard minimum (often 0.82 m) is not the
            # recovery-ready minimum.  Readiness keeps its stricter calibrated
            # height regardless of which consumer updates this state first.
            height_min_m=self.cfg.recovery_relative_height_m,
            tilt_max_rad=self.cfg.recovery_tilt_rad, rate_max_radps=self.cfg.recovery_rate_radps,
            com_speed_max_mps=self.cfg.recovery_com_speed_mps, margin_min_m=self.cfg.recovery_margin_m,
            slip_max_mps=self.cfg.slip_speed_threshold_mps,
        )
        self.recovery_counter = torch.where(ready_now, self.recovery_counter + 1, torch.zeros_like(self.recovery_counter))
        # Recovery readiness is an action-admission state, not merely a pose
        # predicate. A future-prediction veto or confirmed fall must keep it
        # false even if instantaneous support/contact checks look good.
        recovery_ready = (
            (self.recovery_counter >= self.cfg.recovery_hold_steps)
            & (~predicted_unrecoverable)
            & (~confirmed)
        )
        direction = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        tilted = (torch.abs(root_forward) > 0.05) | (torch.abs(root_lateral) > 0.05)
        direction = torch.where(
            tilted & (torch.abs(root_forward) >= torch.abs(root_lateral)),
            torch.where(root_forward >= 0, int(FallReason.FORWARD_FALL), int(FallReason.BACKWARD_FALL)),
            direction,
        )
        direction = torch.where(
            tilted & (torch.abs(root_lateral) > torch.abs(root_forward)),
            torch.where(root_lateral >= 0, int(FallReason.LEFT_FALL), int(FallReason.RIGHT_FALL)),
            direction,
        )
        reason = direction.clone()
        reason = torch.where(illegal_contact, int(FallReason.ILLEGAL_BODY_CONTACT), reason)
        reason = torch.where(~foot_contact.any(-1) & (~illegal_contact), int(FallReason.CONTACT_LOSS_FALL), reason)
        reason = torch.where((foot_slip.max(-1).values > self.cfg.slip_speed_threshold_mps) & (~confirmed), int(FallReason.FOOT_SLIP_FALL), reason)
        reason = torch.where((actuator_sat > 0.95) & (~confirmed), int(FallReason.ACTUATOR_LIMITED_FALL), reason)
        level = torch.full((self.num_envs,), int(FallLevel.STABLE), dtype=torch.long, device=self.device)
        level = torch.where(risk_score >= 0.25, int(FallLevel.AT_RISK), level)
        level = torch.where((risk_score >= 0.45) & (~predicted_unrecoverable), int(FallLevel.RECOVERABLE_UNSTABLE), level)
        level = torch.where(predicted_unrecoverable & (~confirmed), int(FallLevel.PREDICTED_UNRECOVERABLE), level)
        level = torch.where((risk_score >= 0.70) & (~confirmed), int(FallLevel.EMERGENCY), level)
        level = torch.where(confirmed, int(FallLevel.FALL_CONFIRMED), level)
        level = torch.where((~confirmed) & (~recovery_ready) & (risk_score < 0.7) & (self.recovery_counter > 0), int(FallLevel.RECOVERING), level)
        level = torch.where(recovery_ready, int(FallLevel.RECOVERY_READY), level)
        cycle_state = getattr(env, "fall_cycle_phase", torch.zeros_like(level))
        if not isinstance(cycle_state, torch.Tensor) or cycle_state.shape != level.shape:
            cycle_state = torch.zeros_like(level)
        predicted_capture = torch.stack([capture + com_vel_b * h for h in horizons], dim=-1)
        prediction = {"horizons_s": horizons, "tilt_rad": predicted_tilt, "relative_height_m": predicted_height, "capture_point_b": predicted_capture, "support_margins": predicted_margins}
        try:
            motion = env.command_manager.get_term("motion")
            if motion._use_motion_library:
                lengths = motion.motion.motion_lengths[motion.motion_ids]
                hit_steps = motion.motion.hit_frame[motion.motion_ids]
            else:
                lengths = torch.full_like(motion.time_steps, motion.motion.time_step_total)
                hit_steps = torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))
            future_indices = (motion.time_steps.unsqueeze(-1) + (horizons / max(dt, 1.0e-6)).round().to(torch.long)).clamp_min(0)
            future_indices = torch.minimum(future_indices, (lengths - 1).unsqueeze(-1))
            prediction["time_to_hit_s"] = (hit_steps.unsqueeze(-1) - future_indices).to(torch.float32) * dt
            prediction["future_reference_phase"] = future_indices.to(torch.float32) / (lengths.unsqueeze(-1).clamp_min(2) - 1.0)
            prediction["future_reference_available"] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            # Preserve actual future joint reference samples for the audit.
            # They are diagnostics only; no future reference is fed back into
            # the fall detector or controller from this dictionary.
            if hasattr(motion.motion, "joint_pos"):
                if motion._use_motion_library:
                    prediction["future_reference_joint_pos"] = motion.motion.joint_pos[motion.motion_ids.unsqueeze(-1), future_indices]
                else:
                    prediction["future_reference_joint_pos"] = motion.motion.joint_pos[future_indices]
        except Exception:
            prediction["time_to_hit_s"] = torch.full((self.num_envs, horizons.numel()), float("nan"), device=self.device)
            prediction["future_reference_phase"] = torch.full((self.num_envs, horizons.numel()), float("nan"), device=self.device)
            prediction["future_reference_available"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        output = FallStateOutput(
            risk_score, level, direction, reason, recoverable, predicted_unrecoverable,
            confirmed, recovery_ready,
            (self.recovery_counter.float() / float(self.cfg.recovery_hold_steps)).clamp(0, 1),
            self.recovery_counter.clone(), root_forward, root_lateral, torso_forward,
            torso_lateral, f_rate, l_rate, rel_height, torso_rel_height,
            root_linear_velocity_b, root_angular_velocity_b, torso_linear_velocity_b,
            torso_angular_velocity_b, com_b, com_vel_b, capture, cop_position_b,
            margins, predicted_margins, foot_contact, foot_slip, illegal_contact,
            actuator_sat, safety_projection, risk_components, prediction, cycle_state,
        )
        self.last = output
        return output


def unified_fall_state(env: Any, **kwargs: Any) -> FallStateOutput:
    manager = getattr(env, "_unified_fall_state", None)
    if manager is None:
        manager = UnifiedFallState(env)
        env._unified_fall_state = manager
    return manager.update(env, **kwargs)


def reset_unified_fall_state(env: Any, env_ids: torch.Tensor | None = None) -> None:
    manager = getattr(env, "_unified_fall_state", None)
    if manager is not None:
        manager.reset(env_ids)

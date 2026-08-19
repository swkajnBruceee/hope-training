"""AMP motion-prior regularizer for the first HOPE experiment.

The discriminator receives one motion sample at a time: either a policy transition or an
expert/reference transition.  A reference transition is never concatenated to a policy
transition.  Motion features are root-relative and use continuous 6-D local rotations so the
discriminator learns a motion distribution instead of a task-tracking distance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1.0
    return result / torch.sum(quat.square(), dim=-1, keepdim=True).clamp_min(1.0e-12)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_mul(_quat_mul(quat, pure), _quat_inv(quat))[..., 1:]


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw-only heading while preserving world-up roll/pitch in the local frame."""
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    return torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ),
        dim=-1,
    )


def _activation(name: str) -> nn.Module:
    table = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"unsupported AMP activation {name!r}")
    return table[key]()


def _mlp(input_dim: int, hidden_dims: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        width = int(width)
        if width <= 0:
            raise ValueError(f"AMP discriminator hidden dimensions must be positive, got {width}")
        layers.extend((nn.Linear(previous, width), _activation(activation)))
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class AMPMotionFeature:
    """Build the same private motion feature for robot and reference samples."""

    def __init__(
        self,
        joint_velocity_scale: float = 0.1,
        body_linear_velocity_scale: float = 0.1,
        body_angular_velocity_scale: float = 0.1,
        excluded_joint_names: Sequence[str] = ("head_yaw_joint", "head_pitch_joint"),
        lower_body_body_names: Sequence[str] = (
            "left_hip_roll_Link",
            "left_knee_Link",
            "left_ankle_roll_Link",
            "right_hip_roll_Link",
            "right_knee_Link",
            "right_ankle_roll_Link",
        ),
        lower_body_joint_names: Sequence[str] = (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
        lower_body_feature_scale: float = 0.3,
    ) -> None:
        for name, value in (
            ("joint_velocity_scale", joint_velocity_scale),
            ("body_linear_velocity_scale", body_linear_velocity_scale),
            ("body_angular_velocity_scale", body_angular_velocity_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"AMP {name} must be finite and positive")
        self.joint_velocity_scale = float(joint_velocity_scale)
        self.body_linear_velocity_scale = float(body_linear_velocity_scale)
        self.body_angular_velocity_scale = float(body_angular_velocity_scale)
        self.excluded_joint_names = tuple(str(name) for name in excluded_joint_names)
        self.lower_body_body_names = tuple(str(name) for name in lower_body_body_names)
        self.lower_body_joint_names = tuple(str(name) for name in lower_body_joint_names)
        self.lower_body_feature_scale = float(lower_body_feature_scale)
        if len(self.excluded_joint_names) != len(set(self.excluded_joint_names)):
            raise ValueError("AMP excluded_joint_names must be unique")
        if not math.isfinite(self.lower_body_feature_scale) or not 0.0 <= self.lower_body_feature_scale <= 1.0:
            raise ValueError("AMP lower_body_feature_scale must lie in [0, 1]")
        self._joint_indices: torch.Tensor | None = None
        self._joint_names: tuple[str, ...] = ()
        self._body_names: tuple[str, ...] = ()
        self._lower_body_mask: torch.Tensor | None = None
        self._lower_joint_mask: torch.Tensor | None = None

    @staticmethod
    def _motion_term(env):
        try:
            return env.command_manager.get_term("motion")
        except (AttributeError, KeyError, ValueError) as exc:
            raise RuntimeError("AMP requires the registered MotionCommand term") from exc

    @staticmethod
    def _root_index(command) -> int:
        names = tuple(str(name) for name in getattr(command.cfg, "body_names", ()))
        if not names:
            raise RuntimeError("AMP MotionCommand has no tracked body names")
        # The A3 prepared motion schema starts with pelvis_link.  Keep a safe fallback for
        # compact compatible clips, while refusing a silently different body ordering.
        if "pelvis_link" in names:
            return names.index("pelvis_link")
        return 0

    @staticmethod
    def _root_relative(
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        root_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos = body_pos_w[:, root_index]
        root_quat = body_quat_w[:, root_index]
        # Remove global position and yaw only. Keeping roll/pitch relative to world-up lets AMP
        # distinguish an upright body from the same pose rigidly tilted in space.
        root_inv = _quat_inv(_heading_quat(root_quat))
        root_inv_b = root_inv[:, None, :]
        local_pos = _quat_apply(root_inv_b, body_pos_w - root_pos[:, None, :])
        local_quat = _quat_mul(root_inv_b, body_quat_w)
        local_lin_vel = _quat_apply(root_inv_b, body_lin_vel_w)
        local_ang_vel = _quat_apply(root_inv_b, body_ang_vel_w)
        return local_pos, local_quat, local_lin_vel, local_ang_vel

    @staticmethod
    def _rotation_6d(local_quat: torch.Tensor) -> torch.Tensor:
        # The first two rotation-matrix columns are continuous and avoid q == -q ambiguity.
        matrix = _matrix_from_quat(local_quat)
        return matrix[..., :, :2].reshape(local_quat.shape[0], local_quat.shape[1], 6)

    def _build(
        self,
        command,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        self._resolve_layout(command, joint_pos)
        root_index = self._root_index(command)
        local_pos, local_quat, local_lin, local_ang = self._root_relative(
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            root_index,
        )
        body_feature = torch.cat(
            (
                local_pos,
                self._rotation_6d(local_quat),
                local_lin * self.body_linear_velocity_scale,
                local_ang * self.body_angular_velocity_scale,
            ),
            dim=-1,
        ).flatten(start_dim=1)
        if self._lower_body_mask is not None and self.lower_body_feature_scale != 1.0:
            body_feature = body_feature.view(body_feature.shape[0], len(self._body_names), 15)
            body_feature = torch.where(
                self._lower_body_mask.view(1, -1, 1),
                body_feature * self.lower_body_feature_scale,
                body_feature,
            ).flatten(start_dim=1)
        selected_joint_pos = joint_pos.index_select(-1, self._joint_indices)
        selected_joint_vel = joint_vel.index_select(-1, self._joint_indices)
        if self._lower_joint_mask is not None and self.lower_body_feature_scale != 1.0:
            selected_joint_pos = torch.where(
                self._lower_joint_mask.view(1, -1),
                selected_joint_pos * self.lower_body_feature_scale,
                selected_joint_pos,
            )
            selected_joint_vel = torch.where(
                self._lower_joint_mask.view(1, -1),
                selected_joint_vel * self.lower_body_feature_scale,
                selected_joint_vel,
            )
        return torch.cat(
            (
                selected_joint_pos,
                selected_joint_vel * self.joint_velocity_scale,
                body_feature,
            ),
            dim=-1,
        )

    def _resolve_layout(self, command, joint_pos: torch.Tensor) -> None:
        if self._joint_indices is not None:
            return
        robot = getattr(command, "robot", None)
        names = tuple(
            str(name)
            for name in getattr(robot, "joint_names", getattr(getattr(robot, "data", None), "joint_names", ()))
        )
        joint_count = int(joint_pos.shape[-1])
        if names and len(names) != joint_count:
            raise RuntimeError(
                "AMP joint layout mismatch: MotionCommand joint feature has "
                f"{joint_count} values but robot exposes {len(names)} joint names"
            )
        if names:
            missing = [name for name in self.excluded_joint_names if name not in names]
            if missing:
                raise RuntimeError(
                    "AMP passive-joint exclusion names are missing from the A3 articulation: "
                    f"{missing}; available={list(names)}"
                )
            active = [index for index, name in enumerate(names) if name not in self.excluded_joint_names]
            self._joint_names = tuple(names[index] for index in active)
        else:
            # Host-only unit tests may provide a compact command stub without articulation names.
            # Real Isaac startup always takes the named path above, where passive-head exclusion is
            # fail-closed.
            active = list(range(joint_count))
            self._joint_names = tuple(f"joint_{index}" for index in active)
        self._joint_indices = torch.as_tensor(active, dtype=torch.long, device=joint_pos.device)
        self._lower_joint_mask = torch.tensor(
            [name in self.lower_body_joint_names for name in self._joint_names],
            dtype=torch.bool,
            device=joint_pos.device,
        )
        self._body_names = tuple(str(name) for name in getattr(command.cfg, "body_names", ()))
        if not self._body_names:
            raise RuntimeError("AMP MotionCommand has no tracked body names")
        lower = torch.tensor(
            [name in self.lower_body_body_names for name in self._body_names],
            dtype=torch.bool,
            device=joint_pos.device,
        )
        self._lower_body_mask = lower

    def signature(self, command=None, joint_pos: torch.Tensor | None = None) -> dict:
        """Return the semantic AMP input contract used for checkpoint compatibility."""
        if self._joint_indices is None:
            if command is None or joint_pos is None:
                raise RuntimeError("AMP feature layout has not been resolved")
            self._resolve_layout(command, joint_pos)
        feature_dim = len(self._joint_names) * 2 + len(self._body_names) * 15
        return {
            "joint_names": list(self._joint_names),
            "excluded_joint_names": list(self.excluded_joint_names),
            "body_names": list(self._body_names),
            "feature_dim": feature_dim,
            "rotation_representation": "local_rotation_6d",
            "root_frame": "pelvis_heading_yaw_only_world_up",
            "joint_velocity_scale": self.joint_velocity_scale,
            "body_linear_velocity_scale": self.body_linear_velocity_scale,
            "body_angular_velocity_scale": self.body_angular_velocity_scale,
            "lower_body_body_names": list(self.lower_body_body_names),
            "lower_body_joint_names": list(self.lower_body_joint_names),
            "lower_body_feature_scale": self.lower_body_feature_scale,
        }

    def robot(self, env) -> torch.Tensor:
        command = self._motion_term(env)
        return self._build(
            command,
            command.robot_joint_pos,
            command.robot_joint_vel,
            command.robot_body_pos_w,
            command.robot_body_quat_w,
            command.robot_body_lin_vel_w,
            command.robot_body_ang_vel_w,
        )

    def expert(self, env) -> torch.Tensor:
        command = self._motion_term(env)
        return self._build(
            command,
            command.joint_pos,
            command.joint_vel,
            command.body_pos_w,
            command.body_quat_w,
            command.body_lin_vel_w,
            command.body_ang_vel_w,
        )


class AMPDiscriminator(nn.Module):
    """Expert-vs-policy discriminator with a standard softplus AMP reward."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        learning_rate: float = 1.0e-4,
        batch_size: int = 1024,
        updates_per_rollout: int = 2,
        reward_clip: float = 5.0,
        gradient_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if int(state_dim) <= 0:
            raise ValueError("AMP transition state_dim must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("AMP learning_rate must be positive")
        if int(batch_size) <= 0 or int(updates_per_rollout) <= 0:
            raise ValueError("AMP batch_size and updates_per_rollout must be positive")
        if float(reward_clip) <= 0.0:
            raise ValueError("AMP reward_clip must be positive")
        if not math.isfinite(float(gradient_penalty_weight)) or float(gradient_penalty_weight) < 0.0:
            raise ValueError("AMP gradient_penalty_weight must be finite and non-negative")
        self.state_dim = int(state_dim)
        self.batch_size = int(batch_size)
        self.updates_per_rollout = int(updates_per_rollout)
        self.reward_clip = float(reward_clip)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        self.discriminator = _mlp(self.state_dim, hidden_dims, activation)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.last_stats: dict[str, float] = {
            "disc_loss": 0.0,
            "expert_prob": 0.0,
            "policy_prob": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "weighted_reward_mean": 0.0,
            "sample_count": 0.0,
            "expert_logit_mean": 0.0,
            "policy_logit_mean": 0.0,
            "valid_transition_fraction": 0.0,
            "gradient_penalty": 0.0,
        }

    def logits(self, transition: torch.Tensor) -> torch.Tensor:
        if transition.ndim != 2 or transition.shape[-1] != self.state_dim:
            raise ValueError(
                f"AMP transition must have shape [N, {self.state_dim}], got {tuple(transition.shape)}"
            )
        return self.discriminator(transition).squeeze(-1)

    @staticmethod
    def transition(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if previous.ndim != 2 or current.ndim != 2 or previous.shape != current.shape:
            raise ValueError("AMP transition endpoints must have identical rank-2 shapes")
        return torch.cat((previous, current), dim=-1)

    @torch.no_grad()
    def reward(self, policy_transition: torch.Tensor) -> torch.Tensor:
        """Return ``-log(1-D(policy_transition))`` without policy gradients."""
        return F.softplus(self.logits(policy_transition)).clamp(min=0.0, max=self.reward_clip)

    def update(
        self,
        policy_transitions: torch.Tensor,
        expert_transitions: torch.Tensor,
        lambda_amp: float = 0.0,
    ) -> dict[str, float]:
        if policy_transitions.ndim != 2 or expert_transitions.ndim != 2:
            raise ValueError("AMP transition batches must be rank-2 tensors")
        if (
            policy_transitions.shape[-1] != self.state_dim
            or expert_transitions.shape[-1] != self.state_dim
        ):
            raise ValueError("AMP transition dimension does not match the discriminator")
        count = min(int(policy_transitions.shape[0]), int(expert_transitions.shape[0]))
        if count == 0:
            return dict(self.last_stats)
        policy_transitions = policy_transitions[:count].detach()
        expert_transitions = expert_transitions[:count].detach()
        loss_sum = expert_prob = policy_prob = gradient_penalty_sum = 0.0
        steps = 0
        for _ in range(self.updates_per_rollout):
            batch = min(self.batch_size, count)
            policy_idx = torch.randint(count, (batch,), device=policy_transitions.device)
            expert_idx = torch.randint(count, (batch,), device=expert_transitions.device)
            policy_logits = self.logits(policy_transitions.index_select(0, policy_idx))
            expert_logits = self.logits(expert_transitions.index_select(0, expert_idx))
            labels_loss = F.binary_cross_entropy_with_logits(
                torch.cat((expert_logits, policy_logits)),
                torch.cat((torch.ones_like(expert_logits), torch.zeros_like(policy_logits))),
            )
            if self.gradient_penalty_weight > 0.0:
                alpha = torch.rand((batch, 1), device=policy_transitions.device)
                interpolated = (
                    alpha * expert_transitions.index_select(0, expert_idx)
                    + (1.0 - alpha) * policy_transitions.index_select(0, policy_idx)
                ).detach().requires_grad_(True)
                interpolated_logits = self.logits(interpolated)
                gradients = torch.autograd.grad(
                    outputs=interpolated_logits,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(interpolated_logits),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                gradient_penalty = (gradients.flatten(start_dim=1).norm(2, dim=1) - 1.0).square().mean()
            else:
                gradient_penalty = labels_loss.new_zeros(())
            loss = labels_loss + self.gradient_penalty_weight * gradient_penalty
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            self.optimizer.step()
            with torch.no_grad():
                loss_sum += float(loss.detach().cpu())
                gradient_penalty_sum += float(gradient_penalty.detach().cpu())
                expert_prob += float(torch.sigmoid(expert_logits).mean().cpu())
                policy_prob += float(torch.sigmoid(policy_logits).mean().cpu())
            steps += 1
        with torch.no_grad():
            rollout_reward = self.reward(policy_transitions)
            expert_logits = self.logits(expert_transitions)
            policy_logits = self.logits(policy_transitions)
        reward_mean = float(rollout_reward.mean().cpu())
        reward_std = float(rollout_reward.std(unbiased=False).cpu())
        self.last_stats = {
            "disc_loss": loss_sum / max(steps, 1),
            "expert_prob": expert_prob / max(steps, 1),
            "policy_prob": policy_prob / max(steps, 1),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "weighted_reward_mean": float(lambda_amp) * reward_mean,
            "sample_count": float(count),
            "expert_logit_mean": float(expert_logits.mean().cpu()),
            "policy_logit_mean": float(policy_logits.mean().cpu()),
            "valid_transition_fraction": 1.0,
            "gradient_penalty": gradient_penalty_sum / max(steps, 1),
        }
        return dict(self.last_stats)


def robot_motion_feature(env, feature: AMPMotionFeature) -> torch.Tensor:
    return feature.robot(env)


def expert_motion_feature(env, feature: AMPMotionFeature) -> torch.Tensor:
    return feature.expert(env)


def amp_transition_valid_mask(
    done: torch.Tensor,
    current_hold: torch.Tensor,
    previous_hold: torch.Tensor,
    resampled: torch.Tensor,
    *,
    ignore_hold: bool = True,
    ignore_terminal: bool = True,
    ignore_resample: bool = True,
    ignore_hold_exit: bool = True,
) -> torch.Tensor:
    """Return the fail-closed mask for physical AMP transitions.

    Hold, reset, clip-wrap and resample transitions are bookkeeping boundaries rather than
    learned robot motion. ``previous_hold`` also removes the first released swing transition,
    which otherwise joins the synthetic hold state to the first live swing frame.
    """
    masks = (done, current_hold, previous_hold, resampled)
    if any(mask.ndim != 1 for mask in masks):
        raise ValueError("AMP transition masks must be rank-1")
    if any(mask.shape != done.shape for mask in masks):
        raise ValueError("AMP transition masks must have identical shapes")
    valid = torch.ones_like(done, dtype=torch.bool)
    if ignore_terminal:
        valid &= ~done.bool()
    if ignore_hold:
        valid &= ~(current_hold.bool() | previous_hold.bool())
    if ignore_hold_exit:
        valid &= ~(previous_hold.bool() & ~current_hold.bool())
    if ignore_resample:
        valid &= ~resampled.bool()
    return valid


def build_amp_from_config(state_dim: int, cfg: Mapping) -> AMPDiscriminator:
    return AMPDiscriminator(
        state_dim=state_dim,
        hidden_dims=tuple(int(x) for x in cfg.get("hidden_dims", (256, 256))),
        activation=str(cfg.get("activation", "elu")),
        learning_rate=float(cfg.get("learning_rate", 1.0e-4)),
        batch_size=int(cfg.get("batch_size", 1024)),
        updates_per_rollout=int(cfg.get("updates_per_rollout", 2)),
        reward_clip=float(cfg.get("reward_clip", 5.0)),
        gradient_penalty_weight=float(cfg.get("gradient_penalty_weight", 0.0)),
    )

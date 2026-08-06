"""Predictive lower-body recovery adapter that preserves P3's arm actor.

The P3 coordinator has a 204-D actor contract. This module keeps that actor
intact and attaches a zero-initialized lower-body residual in the
READY/swing/recovery window. Small motion-specific brace priors are kept
separate from the learned residual. Motion 3's prior was support-audited;
motion 1 receives the same conservative posture only as a recovery-training
bootstrap after a tail probe showed it delays (but does not remove) the fall.
Motion 1 remains unavailable to the external-target executor until its own
full-tail validation passes.
Target-position control stays in the frozen P3 actor and its calibrated
feedforward path.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


class TargetConditionedRecoveryActorCritic(ActorCritic):
    """Frozen P3 policy plus a gated lower-body recovery residual."""

    BASE_OBS_DIM = 204
    RECOVERY_OBS_DIM = 9
    # All leg DOFs plus waist roll/pitch. The support audit showed that the
    # motion-3 fall cannot be corrected using sagittal joints alone: left hip
    # roll and waist roll are causal stabilizing directions. Waist yaw and
    # every arm joint remain owned by the frozen strike path.
    RECOVERY_ACTION_INDICES = tuple(range(12)) + (13, 14)
    MOTION3_BRACE_ACTION = {
        1: -0.50,  # left hip roll
        3: +0.50,  # left knee
        13: +0.50,  # waist roll
    }
    # This is intentionally a bootstrap copy rather than a claim that motion
    # 1 has passed the motion-3 support audit. P11 trains a bounded residual
    # around it; P10 continues to reject motion 1 for external execution.
    MOTION1_BOOTSTRAP_BRACE_ACTION = MOTION3_BRACE_ACTION

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128, 64],
        critic_hidden_dims: list[int] = [256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 0.03,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        self.recovery_total_obs_dim = self.BASE_OBS_DIM + self.RECOVERY_OBS_DIM
        if num_actor_obs != self.recovery_total_obs_dim:
            raise ValueError(
                "TargetConditionedRecoveryActorCritic requires "
                f"{self.recovery_total_obs_dim} actor observations, got {num_actor_obs}"
            )
        if num_actions != 22:
            raise ValueError(
                "TargetConditionedRecoveryActorCritic requires the 22-D P3 "
                f"coordinator action, got {num_actions}"
            )

        # Construct the base actor at P3's original input width so model_202
        # loads without any actor-weight migration.  The critic intentionally
        # receives the widened recovery observation and is trained afresh.
        super().__init__(
            self.BASE_OBS_DIM,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.recovery_action_dim = len(self.RECOVERY_ACTION_INDICES)
        self.recovery_encoder = nn.Sequential(
            nn.Linear(self.recovery_total_obs_dim, 96),
            nn.ELU(),
            nn.Linear(96, 64),
            nn.ELU(),
        )
        self.recovery_adapter = nn.Linear(64, self.recovery_action_dim)
        nn.init.zeros_(self.recovery_adapter.weight)
        nn.init.zeros_(self.recovery_adapter.bias)

        indices = torch.tensor(self.RECOVERY_ACTION_INDICES, dtype=torch.long)
        # The learned branch remains bounded. It can use every lower-body
        # correction direction except strike-owned waist yaw, with a smaller
        # trust region on waist pitch because it directly affects racket pose.
        gains = torch.tensor(
            (0.80,) * 12 + (0.70, 0.40),
            dtype=torch.float32,
        )
        std_mask = torch.zeros(num_actions, dtype=torch.float32)
        std_mask[indices] = 1.0
        self.register_buffer("recovery_action_indices", indices, persistent=False)
        self.register_buffer("recovery_action_gain", gains, persistent=False)
        self.register_buffer("recovery_std_mask", std_mask, persistent=False)
        brace = torch.zeros(num_actions, dtype=torch.float32)
        for action_index, value in self.MOTION3_BRACE_ACTION.items():
            brace[action_index] = float(value)
        self.register_buffer("audited_brace_action", brace, persistent=False)

        # Training is allowed to update only the recovery branch, critic and
        # exploration scale.  This makes the pre-hit P3 actor immutable.
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        Normal.set_default_validate_args(False)

    def base_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate the immutable P3 actor on its original 204-D contract."""
        if observations.shape[-1] != self.BASE_OBS_DIM:
            raise ValueError(
                f"Expected {self.BASE_OBS_DIM}-D P3 observation, got "
                f"{observations.shape[-1]}"
            )
        return self.actor(observations)

    def _action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.recovery_total_obs_dim:
            raise ValueError(
                f"Expected {self.recovery_total_obs_dim} actor observations, got "
                f"{observations.shape[-1]}"
            )
        base_observation = observations[..., : self.BASE_OBS_DIM]
        recovery_observation = observations[..., self.BASE_OBS_DIM :]
        base = self.base_action_mean(base_observation)
        selected_delta = torch.tanh(
            self.recovery_adapter(self.recovery_encoder(observations))
        )
        selected_delta = selected_delta * self.recovery_action_gain
        full_delta = torch.zeros_like(base)
        full_delta[..., self.recovery_action_indices] = selected_delta
        gate = recovery_observation[..., -1:].clamp(0.0, 1.0)
        # The penultimate suffix channel is the raw (unnormalized) motion id.
        # Motion 3 receives the audited brace. Motion 1 receives its exact
        # conservative copy only as P11's recovery-training bootstrap; it is
        # deliberately not admitted by the P10 external-target selector.
        motion_id = recovery_observation[..., -2:-1]
        motion1_or_3 = (
            torch.isclose(
                motion_id,
                torch.tensor(1.0, dtype=observations.dtype, device=observations.device),
                atol=1.0e-4,
                rtol=0.0,
            )
            | torch.isclose(
                motion_id,
                torch.tensor(3.0, dtype=observations.dtype, device=observations.device),
                atol=1.0e-4,
                rtol=0.0,
            )
        ).to(dtype=base.dtype)
        brace = motion1_or_3 * self.audited_brace_action.to(dtype=base.dtype)
        return base + gate * (brace + full_delta)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._action_mean(observations)
        if self.noise_std_type == "scalar":
            learned_std = self.std.clamp_min(1.0e-6)
        elif self.noise_std_type == "log":
            learned_std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        gate = observations[..., -1:].clamp(0.0, 1.0)
        std = 1.0e-4 + gate * (
            learned_std.unsqueeze(0) * self.recovery_std_mask.unsqueeze(0)
        )
        self.distribution = Normal(mean, std.expand_as(mean))

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._action_mean(observations)

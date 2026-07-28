"""Frozen legacy coordinator with a stagger-aware leg/waist adapter."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


class StaggerSupportActorCritic(ActorCritic):
    """Preserve the qualified arm policy while learning stance-aware support."""

    SUPPORT_OBS_DIM = 19

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128, 64],
        critic_hidden_dims: list[int] = [256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 0.10,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        self.legacy_obs_dim = 204
        self.support_obs_dim = self.SUPPORT_OBS_DIM
        self.total_obs_dim = self.legacy_obs_dim + self.support_obs_dim
        self.support_action_dim = 15
        self.arm_action_dim = 7
        self.zero_legacy_support_action = False
        if num_actor_obs != self.total_obs_dim:
            raise ValueError(
                f"StaggerSupportActorCritic requires {self.total_obs_dim} actor observations, "
                f"got {num_actor_obs}"
            )
        if num_actions != self.support_action_dim + self.arm_action_dim:
            raise ValueError(f"StaggerSupportActorCritic requires 22 actions, got {num_actions}")

        # The legacy actor intentionally remains 204-D so its checkpoint loads
        # exactly. The critic receives the complete stance-aware observation.
        super().__init__(
            self.legacy_obs_dim,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.support_state_encoder = nn.Sequential(
            nn.Linear(self.legacy_obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.stagger_encoder = nn.Sequential(
            nn.Linear(self.support_obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )
        self.support_fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ELU(),
        )
        self.support_adapter = nn.Linear(64, self.support_action_dim)
        nn.init.zeros_(self.support_adapter.weight)
        nn.init.zeros_(self.support_adapter.bias)

        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.fixed_arm_std = 1.0e-4
        Normal.set_default_validate_args(False)

    def _action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.total_obs_dim:
            raise ValueError(
                f"Expected {self.total_obs_dim} actor observations, got {observations.shape[-1]}"
            )
        legacy = observations[..., : self.legacy_obs_dim]
        support_observation = observations[..., self.legacy_obs_dim :]
        base = self.actor(legacy)
        features = torch.cat(
            (
                self.support_state_encoder(legacy),
                self.stagger_encoder(support_observation),
            ),
            dim=-1,
        )
        support_delta = self.support_adapter(self.support_fusion(features))
        legacy_support = base[..., : self.support_action_dim]
        if self.zero_legacy_support_action:
            legacy_support = torch.zeros_like(legacy_support)
        support = legacy_support + support_delta
        return torch.cat((support, base[..., self.support_action_dim :]), dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._action_mean(observations)
        if self.noise_std_type == "scalar":
            learned_std = self.std[: self.support_action_dim].clamp_min(1.0e-6)
        elif self.noise_std_type == "log":
            learned_std = torch.exp(self.log_std[: self.support_action_dim])
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        arm_std = torch.full(
            (self.arm_action_dim,),
            self.fixed_arm_std,
            dtype=mean.dtype,
            device=mean.device,
        )
        std = torch.cat((learned_std, arm_std), dim=0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._action_mean(observations)


class WideStaggerSupportActorCritic(StaggerSupportActorCritic):
    """Stagger adapter with explicit sagittal and lateral capture state."""

    SUPPORT_OBS_DIM = 23


class WideStaggerRecoveryActorCritic(WideStaggerSupportActorCritic):
    """Freeze V22 and learn a gated, post-hit whole-body braking correction."""

    BASE_OBS_DIM = 227
    RECOVERY_OBS_DIM = 2

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
        recovery_arm_std_scale: float = 0.25,
        **kwargs,
    ):
        expected_obs = self.BASE_OBS_DIM + self.RECOVERY_OBS_DIM
        if num_actor_obs != expected_obs:
            raise ValueError(
                f"WideStaggerRecoveryActorCritic requires {expected_obs} actor "
                f"observations, got {num_actor_obs}"
            )
        super().__init__(
            num_actor_obs=self.BASE_OBS_DIM,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.recovery_total_obs_dim = expected_obs
        self.recovery_action_dim = num_actions
        self.recovery_arm_std_scale = float(recovery_arm_std_scale)
        if not 0.0 < self.recovery_arm_std_scale <= 1.0:
            raise ValueError("recovery_arm_std_scale must be in (0, 1]")

        # Reuse V22's frozen full-state and support representations. Capture
        # point state alone is ambiguous: the same margin can require opposite
        # corrections at different joint configurations or swing phases.
        recovery_input_dim = 64 + 32 + self.RECOVERY_OBS_DIM
        self.recovery_encoder = nn.Sequential(
            nn.Linear(recovery_input_dim, 96),
            nn.ELU(),
            nn.Linear(96, 64),
            nn.ELU(),
        )
        self.recovery_adapter = nn.Linear(64, self.recovery_action_dim)
        nn.init.zeros_(self.recovery_adapter.weight)
        nn.init.zeros_(self.recovery_adapter.bias)

        # model_1499 is the immutable V22 capability prior. Only this new
        # branch, the critic, and exploration scale are optimized.
        for module in (
            self.actor,
            self.support_state_encoder,
            self.stagger_encoder,
            self.support_fusion,
            self.support_adapter,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def base_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate the frozen V22 policy on its unchanged 227-D contract."""
        return super()._action_mean(observations)

    def _action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.recovery_total_obs_dim:
            raise ValueError(
                f"Expected {self.recovery_total_obs_dim} actor observations, "
                f"got {observations.shape[-1]}"
            )
        base_observation = observations[..., : self.BASE_OBS_DIM]
        recovery_observation = observations[..., self.BASE_OBS_DIM :]
        base = self.base_action_mean(base_observation)
        recovery_features = torch.cat(
            (
                self.support_state_encoder(
                    base_observation[..., : self.legacy_obs_dim]
                ),
                self.stagger_encoder(
                    base_observation[..., self.legacy_obs_dim :]
                ),
                recovery_observation,
            ),
            dim=-1,
        )
        recovery_delta = self.recovery_adapter(
            self.recovery_encoder(recovery_features)
        )
        gate = recovery_observation[..., -1:].clamp(0.0, 1.0)
        return base + gate * recovery_delta

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._action_mean(observations)
        if self.noise_std_type == "scalar":
            learned_std = self.std.clamp_min(1.0e-6)
        elif self.noise_std_type == "log":
            learned_std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        gate = observations[..., -1:].clamp(0.0, 1.0)
        group_scale = torch.ones(
            (self.recovery_action_dim,), dtype=mean.dtype, device=mean.device
        )
        group_scale[self.support_action_dim :] = self.recovery_arm_std_scale
        active_std = learned_std * group_scale
        # Exploration is also gated, so the loaded strike policy remains
        # deterministic through exact impact instead of relying on a reward to
        # approximately preserve it.
        std = 1.0e-4 + gate * active_std.unsqueeze(0)
        self.distribution = Normal(mean, std.expand_as(mean))

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._action_mean(observations)

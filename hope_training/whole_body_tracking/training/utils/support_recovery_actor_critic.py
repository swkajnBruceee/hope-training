"""Frozen coordinator prior with a trainable leg/waist recovery branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


class SupportRecoveryActorCritic(ActorCritic):
    """Train reactive support without modifying the qualified arm policy."""

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
        self.state_obs_dim = 204
        self.support_action_dim = 15
        self.arm_action_dim = 7
        if num_actor_obs != self.state_obs_dim:
            raise ValueError(
                f"SupportRecoveryActorCritic requires 204 actor observations, got {num_actor_obs}"
            )
        if num_actions != self.support_action_dim + self.arm_action_dim:
            raise ValueError(f"SupportRecoveryActorCritic requires 22 actions, got {num_actions}")

        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.support_encoder = nn.Sequential(
            nn.Linear(self.state_obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
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
        if observations.shape[-1] != self.state_obs_dim:
            raise ValueError(f"Expected 204 actor observations, got {observations.shape[-1]}")
        base = self.actor(observations)
        support_delta = self.support_adapter(self.support_encoder(observations))
        support = base[..., : self.support_action_dim] + support_delta
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

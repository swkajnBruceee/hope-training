"""Actor-critic with a zero-initialized momentum-preview correction branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


class MomentumPreviewActorCritic(ActorCritic):
    """Preserve a legacy actor and add state/preview support corrections."""

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
        self.state_obs_dim = 204
        self.preview_obs_dim = 18
        self.support_action_dim = 15
        self.arm_action_dim = 7
        if num_actor_obs != self.state_obs_dim + self.preview_obs_dim:
            raise ValueError(
                f"MomentumPreviewActorCritic requires 222 actor observations, got {num_actor_obs}"
            )
        if num_actions != self.support_action_dim + self.arm_action_dim:
            raise ValueError(f"MomentumPreviewActorCritic requires 22 actions, got {num_actions}")

        # The inherited actor is deliberately built with the legacy 204-D
        # input so its state_dict is exactly compatible with model_0.
        super().__init__(
            self.state_obs_dim,
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
            nn.Linear(self.state_obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.preview_encoder = nn.Sequential(
            nn.Linear(self.preview_obs_dim, 64, bias=False),
            nn.ELU(),
            nn.Linear(64, 64, bias=False),
            nn.ELU(),
        )
        self.preview_state_gate = nn.Linear(64, 64, bias=False)
        self.preview_adapter = nn.Linear(64, self.support_action_dim, bias=False)
        nn.init.zeros_(self.preview_state_gate.weight)
        nn.init.zeros_(self.preview_adapter.weight)

        # P0 learns only predictive support. The old state actor remains an
        # immutable capability prior; arm exploration is suppressed below.
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.fixed_arm_std = 1.0e-4
        Normal.set_default_validate_args(False)

    def _action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.state_obs_dim + self.preview_obs_dim:
            raise ValueError(f"Expected 222 actor observations, got {observations.shape[-1]}")
        state = observations[..., : self.state_obs_dim]
        base = self.actor(state)
        # The dedicated state encoder retains direct balance feedback that may
        # have been discarded by the frozen task actor. It can only modulate a
        # bias-free preview path, preserving the hard zero-preview -> zero
        # correction invariant and preventing a reactive-only shortcut.
        state_features = self.support_state_encoder(state)
        preview_features = self.preview_encoder(observations[..., self.state_obs_dim :])
        state_modulation = 1.0 + torch.tanh(self.preview_state_gate(state_features))
        support_delta = self.preview_adapter(preview_features * state_modulation)
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

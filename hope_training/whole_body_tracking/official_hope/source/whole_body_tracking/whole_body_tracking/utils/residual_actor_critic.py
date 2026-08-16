"""Frozen base policy plus bounded residual-mean Actor-Critic.

This module implements the first model-optimization milestone only.  The base actor remains
the deployable 110-D -> 31-D policy prior; a zero-initialized residual head modifies its policy
mean on active action channels only.  The base action distribution is retained as a diagonal
Normal distribution, matching the baseline checkpoint contract.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from rsl_rl.modules import ActorCritic

from whole_body_tracking.tasks.tracking.actor_observation_contract import (
    resolve_actor_observation_contract,
)


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
    }
    if key not in table:
        raise ValueError(f"unsupported residual activation {name!r}")
    return table[key]()


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Iterable[int],
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    hidden = [int(width) for width in hidden_dims]
    for width in hidden:
        if width <= 0:
            raise ValueError(f"residual hidden dimensions must be positive, got {hidden}")
        layers.append(nn.Linear(previous, width))
        layers.append(_activation(activation))
        previous = width
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class _StructuredResidualActor(nn.Module):
    """Separate proprioception/goal/time encoders followed by a fusion head."""

    def __init__(
        self,
        proprio_dim: int,
        goal_dim: int,
        time_dim: int,
        output_dim: int,
        proprio_hidden_dims: Iterable[int],
        goal_hidden_dims: Iterable[int],
        time_hidden_dims: Iterable[int],
        fusion_hidden_dims: Iterable[int],
        activation: str,
    ) -> None:
        super().__init__()
        proprio_hidden_dims = tuple(int(x) for x in proprio_hidden_dims)
        goal_hidden_dims = tuple(int(x) for x in goal_hidden_dims)
        time_hidden_dims = tuple(int(x) for x in time_hidden_dims)
        if not proprio_hidden_dims or not goal_hidden_dims or not time_hidden_dims:
            raise ValueError("structured encoder hidden dimensions must be non-empty")
        self.proprio_encoder = _build_mlp(
            proprio_dim, proprio_hidden_dims[-1], proprio_hidden_dims[:-1], activation
        )
        self.goal_encoder = _build_mlp(
            goal_dim, goal_hidden_dims[-1], goal_hidden_dims[:-1], activation
        )
        self.time_encoder = _build_mlp(
            time_dim, time_hidden_dims[-1], time_hidden_dims[:-1], activation
        )
        fusion_input_dim = (
            proprio_hidden_dims[-1]
            + goal_hidden_dims[-1]
            + time_hidden_dims[-1]
        )
        self.fusion = _build_mlp(
            fusion_input_dim, output_dim, fusion_hidden_dims, activation
        )

    def forward(self, proprio: torch.Tensor, goal: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.fusion(
            torch.cat(
                (
                    self.proprio_encoder(proprio),
                    self.goal_encoder(goal),
                    self.time_encoder(time),
                ),
                dim=-1,
            )
        )


class _StructuredFiLMResidualActor(nn.Module):
    """Structured residual trunk with time-conditioned FiLM modulation.

    The trunk first encodes proprioception, goal, and time independently.  The
    fused feature is then modulated by a second time-conditioned generator:

        h' = (1 + delta_gamma(time)) * h + delta_beta(time)

    Both the FiLM generator and the residual head are zero initialized by the
    parent policy, so the complete policy is exactly the frozen HOPE actor at
    initialization.
    """

    def __init__(
        self,
        proprio_dim: int,
        goal_dim: int,
        time_dim: int,
        output_dim: int,
        proprio_hidden_dims: Iterable[int],
        goal_hidden_dims: Iterable[int],
        time_hidden_dims: Iterable[int],
        fusion_hidden_dims: Iterable[int],
        film_hidden_dims: Iterable[int],
        activation: str,
    ) -> None:
        super().__init__()
        proprio_hidden_dims = tuple(int(x) for x in proprio_hidden_dims)
        goal_hidden_dims = tuple(int(x) for x in goal_hidden_dims)
        time_hidden_dims = tuple(int(x) for x in time_hidden_dims)
        fusion_hidden_dims = tuple(int(x) for x in fusion_hidden_dims)
        if not proprio_hidden_dims or not goal_hidden_dims or not time_hidden_dims:
            raise ValueError("structured FiLM encoder hidden dimensions must be non-empty")
        if not fusion_hidden_dims:
            raise ValueError("structured FiLM fusion hidden dimensions must be non-empty")

        self.proprio_encoder = _build_mlp(
            proprio_dim, proprio_hidden_dims[-1], proprio_hidden_dims[:-1], activation
        )
        self.goal_encoder = _build_mlp(
            goal_dim, goal_hidden_dims[-1], goal_hidden_dims[:-1], activation
        )
        self.time_encoder = _build_mlp(
            time_dim, time_hidden_dims[-1], time_hidden_dims[:-1], activation
        )
        fusion_input_dim = (
            proprio_hidden_dims[-1]
            + goal_hidden_dims[-1]
            + time_hidden_dims[-1]
        )
        self.feature_dim = int(fusion_hidden_dims[-1])
        self.fusion = _build_mlp(
            fusion_input_dim,
            self.feature_dim,
            fusion_hidden_dims[:-1],
            activation,
        )
        self.film_generator = _build_mlp(
            time_dim,
            2 * self.feature_dim,
            tuple(int(x) for x in film_hidden_dims),
            activation,
        )
        self.residual_head = nn.Linear(self.feature_dim, int(output_dim))

    def forward(
        self,
        proprio: torch.Tensor,
        goal: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        h = self.fusion(
            torch.cat(
                (
                    self.proprio_encoder(proprio),
                    self.goal_encoder(goal),
                    self.time_encoder(time),
                ),
                dim=-1,
            )
        )
        film = self.film_generator(time)
        delta_gamma, delta_beta = film.chunk(2, dim=-1)
        h = (1.0 + delta_gamma) * h + delta_beta
        return self.residual_head(h)


class ResidualMeanActorCritic(ActorCritic):
    """Base actor prior plus a zero-initialized bounded residual mean.

    ``self.actor`` and ``self.critic`` retain the baseline checkpoint key names.  The new
    ``self.residual_actor`` is intentionally separate, so the baseline checkpoint can be loaded
    as a model-only warm start while the PPO optimizer is rebuilt for the new parameter set.
    """

    def __init__(
        self,
        *args,
        residual_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        residual_delta_q_max_rad: float = 0.05,
        residual_time_scale: float = 1.0,
        residual_train_std: bool = False,
        residual_architecture: str = "plain",
        residual_active_joint_names: list[str] | tuple[str, ...] = (),
        structured_proprio_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        structured_goal_hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        structured_time_hidden_dims: list[int] | tuple[int, ...] = (16, 8),
        structured_fusion_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        structured_film_hidden_dims: list[int] | tuple[int, ...] = (32, 32),
        **kwargs,
    ):
        self.residual_delta_q_max_rad = float(residual_delta_q_max_rad)
        self.residual_time_scale = float(residual_time_scale)
        self.residual_train_std = bool(residual_train_std)
        self.residual_architecture = str(residual_architecture).lower()
        self.residual_active_joint_names = tuple(str(name) for name in residual_active_joint_names)
        if self.residual_architecture not in ("plain", "structured", "structured_film"):
            raise ValueError(
                "residual_architecture must be 'plain', 'structured', or 'structured_film', "
                f"got {residual_architecture!r}"
            )
        self.structured_proprio_hidden_dims = tuple(int(x) for x in structured_proprio_hidden_dims)
        self.structured_goal_hidden_dims = tuple(int(x) for x in structured_goal_hidden_dims)
        self.structured_time_hidden_dims = tuple(int(x) for x in structured_time_hidden_dims)
        self.structured_fusion_hidden_dims = tuple(int(x) for x in structured_fusion_hidden_dims)
        self.structured_film_hidden_dims = tuple(int(x) for x in structured_film_hidden_dims)
        if self.residual_delta_q_max_rad < 0.0:
            raise ValueError("residual_delta_q_max_rad must be non-negative")
        if self.residual_time_scale <= 0.0:
            raise ValueError("residual_time_scale must be positive")

        super().__init__(*args, **kwargs)

        if not hasattr(self, "std"):
            raise ValueError(
                "ResidualMeanActorCritic MVP requires the scalar-std ActorCritic "
                "distribution (noise_std_type='scalar')"
            )

        actor_obs_dim = int(args[0]) if args else int(kwargs["num_actor_obs"])
        action_dim = int(args[2]) if len(args) > 2 else int(kwargs["num_actions"])
        activation = str(kwargs.get("activation", "elu"))

        contract = resolve_actor_observation_contract("hitter_pure")
        if contract is None or contract.total_dim != actor_obs_dim:
            raise ValueError(
                "ResidualMeanActorCritic requires the 110-D hitter_pure actor observation contract"
            )
        self._term_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for term in contract.terms:
            next_cursor = cursor + int(term.dim)
            self._term_slices[term.name] = (cursor, next_cursor)
            cursor = next_cursor

        if self.residual_architecture == "plain":
            self.residual_actor = _build_mlp(
                input_dim=actor_obs_dim,
                output_dim=action_dim,
                hidden_dims=residual_hidden_dims,
                activation=activation,
            )
        else:
            required_terms = (
                "base_ang_vel", "joint_pos", "joint_vel", "actions",
                "projected_gravity", "base_forward_xy",
                "base_target_delta_xy", "racket_target_rel_base", "racket_target_vel_w",
                "time_to_strike",
            )
            missing = [name for name in required_terms if name not in self._term_slices]
            if missing:
                raise ValueError(f"structured residual contract is missing terms: {missing}")
            proprio_dim = sum(
                self._term_slices[name][1] - self._term_slices[name][0]
                for name in (
                    "base_ang_vel", "joint_pos", "joint_vel", "actions",
                    "projected_gravity", "base_forward_xy",
                )
            )
            goal_dim = sum(
                self._term_slices[name][1] - self._term_slices[name][0]
                for name in (
                    "base_target_delta_xy", "racket_target_rel_base", "racket_target_vel_w",
                )
            )
            time_dim = self._term_slices["time_to_strike"][1] - self._term_slices["time_to_strike"][0]
            self._structured_proprio_terms = (
                "base_ang_vel", "joint_pos", "joint_vel", "actions",
                "projected_gravity", "base_forward_xy",
            )
            self._structured_goal_terms = (
                "base_target_delta_xy", "racket_target_rel_base", "racket_target_vel_w",
            )
            actor_cls = (
                _StructuredFiLMResidualActor
                if self.residual_architecture == "structured_film"
                else _StructuredResidualActor
            )
            actor_kwargs = dict(
                proprio_dim=proprio_dim,
                goal_dim=goal_dim,
                time_dim=time_dim,
                output_dim=action_dim,
                proprio_hidden_dims=self.structured_proprio_hidden_dims,
                goal_hidden_dims=self.structured_goal_hidden_dims,
                time_hidden_dims=self.structured_time_hidden_dims,
                fusion_hidden_dims=self.structured_fusion_hidden_dims,
                activation=activation,
            )
            if self.residual_architecture == "structured_film":
                actor_kwargs["film_hidden_dims"] = self.structured_film_hidden_dims
            self.residual_actor = actor_cls(**actor_kwargs)
        if self.residual_architecture == "structured_film":
            final = self.residual_actor.residual_head
            film_final = self.residual_actor.film_generator[-1]
            if not isinstance(film_final, nn.Linear):
                raise RuntimeError("FiLM generator must end in a Linear layer")
            nn.init.zeros_(film_final.weight)
            nn.init.zeros_(film_final.bias)
        else:
            final = self.residual_actor.fusion[-1] if self.residual_architecture == "structured" else self.residual_actor[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("residual actor must end in a Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        # Residual MVP is deliberately not Direct Fine-tuning.  The released HOPE actor is a
        # frozen prior; only the residual head (and the critic used for PPO targets) may update.
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        for parameter in self.residual_actor.parameters():
            parameter.requires_grad_(True)

        action_dim = int(self.std.numel())
        self.register_buffer(
            "residual_active_mask",
            torch.zeros(action_dim, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "residual_bound_raw",
            torch.zeros(action_dim),
            persistent=True,
        )
        self.register_buffer(
            "resolved_action_scale",
            torch.ones(action_dim),
            persistent=True,
        )
        self._residual_contract_bound = False

        self._time_slice = self._term_slices["time_to_strike"]

        # The released checkpoint has a trainable std parameter, but Residual MVP explicitly freezes
        # it so zero residual means the complete policy distribution remains the baseline one.
        self.std.requires_grad_(self.residual_train_std)
        self.base_checkpoint_sha256: str | None = None
        if any(parameter.requires_grad for parameter in self.actor.parameters()):
            raise RuntimeError("HOPE actor must be frozen in Residual MVP")

    @property
    def residual_active_count(self) -> int:
        return int(self.residual_active_mask.sum().item())

    @torch.no_grad()
    def bind_residual_action_contract(
        self,
        action_scale: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Bind resolved ActionManager scale and active columns after env construction."""
        scale = torch.as_tensor(action_scale, device=self.std.device, dtype=self.std.dtype).reshape(-1)
        mask = torch.as_tensor(active_mask, device=self.std.device, dtype=torch.bool).reshape(-1)
        expected = self.std.numel()
        if scale.numel() != expected or mask.numel() != expected:
            raise ValueError(
                f"Residual action contract expects {expected} columns, "
                f"got scale={scale.numel()} mask={mask.numel()}"
            )
        if bool((torch.abs(scale[mask]) <= 1.0e-12).any()):
            raise ValueError("active Residual action channels must have non-zero action_scale")
        self.residual_active_mask.copy_(mask)
        bound = torch.zeros_like(scale)
        bound[mask] = self.residual_delta_q_max_rad / torch.abs(scale[mask])
        self.residual_bound_raw.copy_(bound)
        self.resolved_action_scale.copy_(scale)
        self._residual_contract_bound = True
        print(
            "[ResidualMeanActorCritic] action contract bound: "
            f"active={int(mask.sum())}/{expected} "
            f"delta_q_max_rad={self.residual_delta_q_max_rad:.6g}",
            flush=True,
        )

    def _residual_features(self, obs: torch.Tensor) -> torch.Tensor:
        start, end = self._time_slice  # type: ignore[misc]
        features = obs.clone()
        features[..., start:end] = torch.clamp(
            features[..., start:end] / self.residual_time_scale,
            min=-1.0,
            max=1.0,
        )
        return features

    def _mean_components(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._residual_contract_bound:
            raise RuntimeError(
                "Residual action contract is not bound; construct the runner with the configured "
                "ActionManager before requesting policy actions"
            )
        hope_mean = self.actor(obs)
        residual_features = self._residual_features(obs)
        if self.residual_architecture == "plain":
            residual_raw = self.residual_actor(residual_features)
        else:
            proprio = torch.cat(
                [residual_features[..., self._term_slices[name][0]:self._term_slices[name][1]]
                 for name in self._structured_proprio_terms],
                dim=-1,
            )
            goal = torch.cat(
                [residual_features[..., self._term_slices[name][0]:self._term_slices[name][1]]
                 for name in self._structured_goal_terms],
                dim=-1,
            )
            start, end = self._time_slice
            residual_raw = self.residual_actor(proprio, goal, residual_features[..., start:end])
        residual_mean = (
            self.residual_active_mask.to(dtype=residual_raw.dtype)
            * self.residual_bound_raw.to(dtype=residual_raw.dtype)
            * torch.tanh(residual_raw)
        )
        return hope_mean, residual_mean, hope_mean + residual_mean

    def _set_distribution(self, obs: torch.Tensor) -> None:
        _, _, mean = self._mean_components(obs)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def update_distribution(self, observations) -> None:
        """Public rsl_rl entrypoint used by PPO rollout and log-prob collection."""
        self._set_distribution(observations)

    def _update_distribution(self, observations) -> None:
        """Compatibility entrypoint used by repository diagnostics and older wrappers."""
        self._set_distribution(observations)

    def act_inference(self, obs) -> torch.Tensor:
        """Return the deterministic combined raw action, preserving the 31-D export contract."""
        _, _, mean = self._mean_components(obs)
        return mean

    @torch.no_grad()
    def residual_diagnostics(self, obs) -> dict[str, torch.Tensor]:
        """Return active-only residual diagnostics in raw and joint-angle units.

        The clipped quantity is a deterministic raw-action clipping estimate. Exact sampled
        execution remains an ActionManager-side diagnostic because the environment may apply
        affine decoding and safety clamps after PPO sampling.
        """
        hope_mean, residual_mean, combined_mean = self._mean_components(obs)
        active = self.residual_active_mask
        delta_raw = residual_mean[..., active]
        delta_q = delta_raw * self.resolved_action_scale[active]
        combined_raw = combined_mean[..., active]
        hope_raw = hope_mean[..., active]
        clipped_delta_q = (
            torch.clamp(combined_raw, -1.0, 1.0)
            - torch.clamp(hope_raw, -1.0, 1.0)
        ) * self.resolved_action_scale[active]
        return {
            "residual_mean_l2_active": torch.linalg.vector_norm(delta_raw, dim=-1),
            "residual_mean_abs_active": delta_raw.abs().mean(dim=-1),
            "residual_mean_max_abs_active": delta_raw.abs().amax(dim=-1),
            "residual_q_nom_abs_active": delta_q.abs().mean(dim=-1),
            "residual_q_nom_max_abs_active": delta_q.abs().amax(dim=-1),
            "residual_q_raw_clip_estimate_abs_active": clipped_delta_q.abs().mean(dim=-1),
            "residual_mean_saturation_rate_active": (combined_raw.abs() > 1.0).float().mean(dim=-1),
        }

    @torch.no_grad()
    def get_model_metadata(self) -> dict:
        """Return JSON/checkpoint-safe metadata for reproducing this policy contract."""
        contract = resolve_actor_observation_contract("hitter_pure")
        cursor = 0
        layout = {}
        for term in contract.terms:
            layout[term.name] = [cursor, cursor + int(term.dim)]
            cursor += int(term.dim)
        return {
            "model_variant": (
                "Frozen HOPE + Structured FiLM Bounded Residual Mean"
                if self.residual_architecture == "structured_film"
                else "Frozen HOPE + Structured Bounded Residual Mean"
                if self.residual_architecture == "structured"
                else "Frozen HOPE + Bounded Residual Mean MVP"
            ),
            "policy_class": self.__class__.__name__,
            "residual_architecture": self.residual_architecture,
            "observation_contract": contract.name,
            "observation_contract_dim": int(contract.total_dim),
            "observation_contract_layout": layout,
            "observation_normalization": "none",
            "residual_delta_q_max_rad": float(self.residual_delta_q_max_rad),
            "residual_time_scale": float(self.residual_time_scale),
            "residual_train_std": bool(self.residual_train_std),
            "residual_active_joint_names": list(self.residual_active_joint_names),
            "std_trainable": bool(self.std.requires_grad),
            "resolved_action_scale_31d": self.resolved_action_scale.detach().cpu().tolist(),
            "residual_bound_raw_31d": self.residual_bound_raw.detach().cpu().tolist(),
            "residual_active_mask_31d": self.residual_active_mask.detach().cpu().tolist(),
            "residual_active_count": self.residual_active_count,
            "residual_contract_bound": bool(self._residual_contract_bound),
            "residual_actor_parameter_count": sum(
                parameter.numel() for parameter in self.residual_actor.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "structured_split_indices": {
                name: [int(start), int(end)]
                for name, (start, end) in self._term_slices.items()
            },
            "structured_proprio_terms": list(
                getattr(self, "_structured_proprio_terms", ())
            ),
            "structured_goal_terms": list(
                getattr(self, "_structured_goal_terms", ())
            ),
            "structured_proprio_hidden_dims": list(self.structured_proprio_hidden_dims),
            "structured_goal_hidden_dims": list(self.structured_goal_hidden_dims),
            "structured_time_hidden_dims": list(self.structured_time_hidden_dims),
            "structured_fusion_hidden_dims": list(self.structured_fusion_hidden_dims),
            "structured_film_hidden_dims": list(self.structured_film_hidden_dims),
            "film_enabled": self.residual_architecture == "structured_film",
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
        }

    @torch.no_grad()
    def load_official_model_state(self, model_state: dict[str, torch.Tensor]) -> list[str]:
        """Load base actor/critic/std weights without touching the new optimizer."""
        # rsl_rl's ActorCritic.load_state_dict intentionally returns a boolean, so call the
        # nn.Module implementation directly to retain missing/unexpected-key diagnostics.
        incompatible = nn.Module.load_state_dict(self, model_state, strict=False)
        allowed_missing = {
            key
            for key in incompatible.missing_keys
            if key.startswith("residual_actor.")
            or key in {"residual_active_mask", "residual_bound_raw", "resolved_action_scale"}
        }
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
        if unexpected or missing:
            raise RuntimeError(
                "base checkpoint does not match ResidualMeanActorCritic contract: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return sorted(allowed_missing)


@configclass
class RslRlResidualMeanActorCriticCfg(RslRlPpoActorCriticCfg):
    """Runner configuration for the first Residual MVP."""

    class_name: str = "ResidualMeanActorCritic"
    residual_hidden_dims: list[int] = [256, 128]
    residual_delta_q_max_rad: float = 0.05
    residual_time_scale: float = 1.0
    residual_train_std: bool = False
    residual_architecture: str = "plain"
    residual_active_joint_names: list[str] = []
    structured_proprio_hidden_dims: list[int] = [128, 64]
    structured_goal_hidden_dims: list[int] = [64, 32]
    structured_time_hidden_dims: list[int] = [16, 8]
    structured_fusion_hidden_dims: list[int] = [256, 128]
    structured_film_hidden_dims: list[int] = [32, 32]


def register_with_rsl_rl_runner() -> None:
    """Register the repository policy with rsl_rl's class-name factory."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.ResidualMeanActorCritic = ResidualMeanActorCritic

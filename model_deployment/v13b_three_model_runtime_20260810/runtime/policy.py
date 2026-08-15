"""Inference-only loader for the exported IsaacLab PPO actor checkpoints.

The checkpoints contain ``model_state_dict`` and ``obs_norm_state_dict``.  The
actor is a Linear -> ELU -> Linear -> ELU -> Linear -> ELU -> Linear MLP.
This file deliberately has no IsaacLab dependency; it is usable on a clean
Python + PyTorch installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointPolicy:
    """Load one actor and its frozen empirical observation normalizer."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        expected_obs_dim: int | None = None,
        expected_action_dim: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(checkpoint).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.path}")
        self.device = torch.device(device)

        try:
            payload: Any = torch.load(self.path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only existed.
            payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint payload is not a mapping: {self.path}")
        model = payload.get("model_state_dict", payload)
        if not isinstance(model, dict):
            raise RuntimeError(f"checkpoint has no model_state_dict: {self.path}")

        layer_ids = sorted(
            int(key.split(".")[1])
            for key in model
            if key.startswith("actor.") and key.endswith(".weight")
        )
        if not layer_ids:
            raise RuntimeError(f"checkpoint has no actor linear layers: {self.path}")

        layers: list[torch.nn.Module] = []
        for index, layer_id in enumerate(layer_ids):
            weight_key = f"actor.{layer_id}.weight"
            bias_key = f"actor.{layer_id}.bias"
            if bias_key not in model:
                raise RuntimeError(f"missing {bias_key} in {self.path}")
            weight = model[weight_key]
            bias = model[bias_key]
            layer = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.append(layer)
            if index + 1 < len(layer_ids):
                layers.append(torch.nn.ELU())

        normalizer = payload.get("obs_norm_state_dict")
        if not isinstance(normalizer, dict):
            raise RuntimeError(f"observation normalizer is missing: {self.path}")
        if "_mean" not in normalizer or "_std" not in normalizer:
            raise RuntimeError(f"normalizer must contain _mean and _std: {self.path}")

        self.actor = torch.nn.Sequential(*layers).to(self.device).eval()
        self.mean = normalizer["_mean"].to(self.device, dtype=torch.float32)
        self.std = normalizer["_std"].to(self.device, dtype=torch.float32).clamp_min(1.0e-6)
        self.obs_dim = int(self.mean.shape[-1])

        action_std = model.get("std")
        if action_std is None:
            action_std = torch.empty(int(model[f"actor.{layer_ids[-1]}.bias"].shape[0]))
        self.action_dim = int(action_std.shape[-1])

        if expected_obs_dim is not None and self.obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"{self.path.name}: expected obs={expected_obs_dim}, got {self.obs_dim}"
            )
        if expected_action_dim is not None and self.action_dim != expected_action_dim:
            raise RuntimeError(
                f"{self.path.name}: expected action={expected_action_dim}, got {self.action_dim}"
            )

    @torch.inference_mode()
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"{self.path.name}: expected [N,{self.obs_dim}], got {tuple(observation.shape)}"
            )
        if not torch.isfinite(observation).all():
            raise ValueError(f"{self.path.name}: observation contains NaN/Inf")
        normalized = torch.clamp((observation - self.mean) / self.std, -100.0, 100.0)
        action = self.actor(normalized)
        if not torch.isfinite(action).all():
            raise RuntimeError(f"{self.path.name}: actor output contains NaN/Inf")
        return action

    def describe(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "normalizer": "obs_norm_state_dict._mean/_std",
            "device": str(self.device),
        }

"""Three-model v13b inference coordinator.

This module owns model loading, normalizers, action bounding and the final
prior/student blend.  It intentionally does not pretend to know a MuJoCo
state estimator or a racket tracker.  The adapter must provide the three
exact observations and the already reconstructed physical prior targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .policy import CheckpointPolicy


LOWER_SCALE_RAD = torch.tensor(
    [0.192, 0.048, 0.192, 0.192, 0.144, 0.192,
     0.192, 0.048, 0.096, 0.072, 0.144, 0.192],
    dtype=torch.float32,
)
UPPER_SCALE_RAD = torch.tensor(
    [0.440, 0.022, 0.110, 0.440, 0.0132, 0.110,
     0.440, 0.440, 0.440, 0.440],
    dtype=torch.float32,
)


@dataclass
class ThreeModelOutput:
    student_action: torch.Tensor
    lower_prior_action: torch.Tensor
    upper_prior_action: torch.Tensor


class ThreeModelRuntime:
    """Load and run the v13b student plus both frozen prior actors."""

    def __init__(self, package_root: str | Path, *, device: str | torch.device = "cpu") -> None:
        root = Path(package_root).expanduser().resolve()
        weights = root / "weights"
        self.device = torch.device(device)
        self.student = CheckpointPolicy(
            weights / "model_5000_student.pt",
            expected_obs_dim=98,
            expected_action_dim=26,
            device=self.device,
        )
        self.lower = CheckpointPolicy(
            weights / "model_3396_lower_prior.pt",
            expected_obs_dim=126,
            expected_action_dim=14,
            device=self.device,
        )
        self.upper = CheckpointPolicy(
            weights / "model_900_upper_prior.pt",
            expected_obs_dim=56,
            expected_action_dim=10,
            device=self.device,
        )

    @torch.inference_mode()
    def infer(
        self,
        student_observation: torch.Tensor,
        lower_observation: torch.Tensor,
        upper_observation: torch.Tensor,
    ) -> ThreeModelOutput:
        """Run all actors; observations must already follow their contracts."""
        return ThreeModelOutput(
            student_action=self.student(student_observation),
            lower_prior_action=self.lower(lower_observation),
            upper_prior_action=self.upper(upper_observation),
        )

    @staticmethod
    def bound_student_action(action: torch.Tensor) -> torch.Tensor:
        """Match the training contract's smooth raw bound: tanh(action)."""
        return torch.tanh(torch.as_tensor(action, dtype=torch.float32))

    @torch.inference_mode()
    def blend_targets(
        self,
        student_action: torch.Tensor,
        *,
        ready_lower: torch.Tensor,
        lower_prior_target: torch.Tensor,
        ready_upper: torch.Tensor,
        upper_prior_target: torch.Tensor,
        microstep_delta: torch.Tensor | None = None,
        alpha_lower: float = 1.0,
        alpha_upper: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend physical prior targets with the 26-D student residual.

        ``lower_prior_target`` must already be the reconstructed physical
        target from model3396 around its historical READY pose.  Likewise,
        ``upper_prior_target`` must already include the model900 reference,
        12-frame shoulder lead, prelude release and velocity-feedforward
        policy.  This prevents callers from accidentally adding raw prior
        actions directly to the student action.
        """
        action = self.bound_student_action(student_action).to(self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != 26:
            raise ValueError(f"student action must be [N,26], got {tuple(action.shape)}")

        ready_lower = torch.as_tensor(ready_lower, dtype=torch.float32, device=self.device)
        lower_prior_target = torch.as_tensor(lower_prior_target, dtype=torch.float32, device=self.device)
        ready_upper = torch.as_tensor(ready_upper, dtype=torch.float32, device=self.device)
        upper_prior_target = torch.as_tensor(upper_prior_target, dtype=torch.float32, device=self.device)
        if ready_lower.shape[-1] != 12 or lower_prior_target.shape[-1] != 12:
            raise ValueError("lower ready/prior targets must have 12 channels")
        if ready_upper.shape[-1] != 10 or upper_prior_target.shape[-1] != 10:
            raise ValueError("upper ready/prior targets must have 10 channels")

        lower_scale = LOWER_SCALE_RAD.to(self.device)
        upper_scale = UPPER_SCALE_RAD.to(self.device)
        lower_student = action[..., :12] * lower_scale
        upper_student = action[..., 12:22] * upper_scale
        if microstep_delta is None:
            microstep_delta = torch.zeros_like(lower_student)
        microstep_delta = torch.as_tensor(microstep_delta, dtype=torch.float32, device=self.device)
        if microstep_delta.shape[-1] != 12:
            raise ValueError("microstep_delta must have 12 channels")

        lower_target = ready_lower + float(alpha_lower) * (lower_prior_target - ready_lower)
        lower_target = lower_target + lower_student + microstep_delta
        upper_target = ready_upper + float(alpha_upper) * (upper_prior_target - ready_upper)
        upper_target = upper_target + upper_student
        if not torch.isfinite(lower_target).all() or not torch.isfinite(upper_target).all():
            raise RuntimeError("blended target contains NaN/Inf")
        return lower_target, upper_target

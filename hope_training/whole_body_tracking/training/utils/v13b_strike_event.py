"""Small, testable primitives for the V1.3B one-strike event contract."""
from __future__ import annotations

import torch


def rephase_teacher_start_frames(
    teacher_hit_frames: torch.Tensor, public_time_to_hit_s: torch.Tensor, fps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return start frames and resulting physical hit times.

    No prelude is included here.  The V1.3B private teacher starts directly
    at the returned motion frame, so this formula is the complete mapping.
    Quantization error is bounded by half a motion frame.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if teacher_hit_frames.shape != public_time_to_hit_s.shape:
        raise ValueError("teacher_hit_frames and public_time_to_hit_s must have the same shape")
    steps = torch.round(public_time_to_hit_s * float(fps)).to(dtype=torch.long)
    start = torch.clamp(teacher_hit_frames.to(dtype=torch.long) - steps, min=0)
    physical_time = (teacher_hit_frames.to(dtype=public_time_to_hit_s.dtype) - start.to(dtype=public_time_to_hit_s.dtype)) / float(fps)
    return start, physical_time

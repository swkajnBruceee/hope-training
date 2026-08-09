#!/usr/bin/env python3
"""Pure unit tests for V1.3B public-time/teacher-frame alignment."""
from __future__ import annotations

import importlib.util
import pathlib

import torch

MODULE = pathlib.Path(__file__).resolve().parents[1] / "training/utils/v13b_strike_event.py"
SPEC = importlib.util.spec_from_file_location("v13b_strike_event_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
rephase_teacher_start_frames = MOD.rephase_teacher_start_frames


def main() -> None:
    fps = 50.0
    hit = torch.tensor([32, 90, 110], dtype=torch.long)
    tau = torch.tensor([0.20, 0.40, 0.60])
    start, physical = rephase_teacher_start_frames(hit, tau, fps)
    assert torch.equal(start, torch.tensor([22, 70, 80]))
    assert torch.allclose(physical, tau, atol=1.0e-6)

    # Non-grid timing is allowed one half-frame of quantization, with no
    # hidden +1 s prelude.  0.41 s at 50 Hz maps to exactly 20 frames = 0.40.
    start, physical = rephase_teacher_start_frames(
        torch.tensor([90]), torch.tensor([0.41]), fps
    )
    assert int(start[0]) == 70
    assert abs(float(physical[0]) - 0.41) <= 1.0 / fps
    print("V1.3B strike-event rephase unit test: PASS")


if __name__ == "__main__":
    main()

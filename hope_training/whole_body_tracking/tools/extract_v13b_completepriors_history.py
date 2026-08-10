#!/usr/bin/env python3
"""Extract immutable scalar history used to shortlist Rescue checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/rsl_rl/agibot_a3_target_conditioned_reference_free_v13b_complete_priors_rightfront_v1/2026-08-09_18-10-06_v13b_resetfixed_model18900_clean_23118_rightfront_16384x50000_resume_from2300_exact"
EVENT = next(RUN.glob("events.out.tfevents.*"))
OUT = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection/history_scalars.json"
WANTED = {
    "position_error_m": "Metrics/racket_target/racket_pos_error_exact_strike",
    "velocity_error_mps": "Metrics/racket_target/racket_vel_error_exact_strike",
    "normal_error_deg": "Metrics/racket_target/racket_normal_error_deg_exact_strike",
    "combined_success": "Metrics/racket_target/strike_composite_success_exact",
    "position_pass": "Metrics/racket_target/strike_pos_pass_exact",
    "velocity_pass": "Metrics/racket_target/strike_vel_pass_exact",
    "normal_pass": "Metrics/racket_target/strike_normal_pass_exact",
    "strict_fall_risk": "Episode_Reward/strict_fall_risk",
}


def main() -> None:
    acc = EventAccumulator(str(EVENT), size_guidance={"scalars": 0, "tensors": 0, "histograms": 0, "images": 0, "audio": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", ()))
    result = {"event": str(EVENT), "tags": {}, "missing": []}
    for name, tag in WANTED.items():
        if tag not in tags:
            result["missing"].append({"name": name, "tag": tag})
            continue
        result["tags"][name] = [{"step": item.step, "value": item.value} for item in acc.Scalars(tag)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

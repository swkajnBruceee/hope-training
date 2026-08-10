#!/usr/bin/env python3
"""Read-only CompletePriors checkpoint inventory for PrecisionRescue.

This deliberately does *not* select a source without the required four
deterministic prior ablations.  It produces the historical candidate table and
marks every candidate pending until those physics evaluations are recorded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from training.utils.v13b_contract import lower_prior_alpha, upper_prior_alpha


RUN = Path("logs/rsl_rl/agibot_a3_target_conditioned_reference_free_v13b_complete_priors_rightfront_v1/2026-08-09_18-10-06_v13b_resetfixed_model18900_clean_23118_rightfront_16384x50000_resume_from2300_exact")
OUT = Path("eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection")
TOTAL_UPDATES = 50000
TAGS = {
    "position_error_m": "Metrics/racket_target/racket_pos_error_exact_strike",
    "velocity_error_mps": "Metrics/racket_target/racket_vel_error_exact_strike",
    "normal_error_deg": "Metrics/racket_target/racket_normal_error_deg_exact_strike",
    "position_pass": "Metrics/racket_target/strike_pos_pass_exact",
    "velocity_pass": "Metrics/racket_target/strike_vel_pass_exact",
    "normal_pass": "Metrics/racket_target/strike_normal_pass_exact",
    "combined_success": "Metrics/racket_target/strike_composite_success_exact",
    "strict_fall": "Episode_Reward/strict_fall_risk",
}


def main() -> None:
    checkpoints = sorted(
        ((int(match.group(1)), path) for path in RUN.glob("model_*.pt")
         if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))),
        key=lambda item: item[0],
    )
    if not checkpoints:
        raise SystemExit(f"no checkpoints found under {RUN}")
    candidates = []
    for iteration, path in checkpoints:
        progress = min(1.0, iteration / max(TOTAL_UPDATES - 1, 1))
        row = {
            "checkpoint": str(path.resolve()),
            "model": path.name,
            "iteration": iteration,
            "historical_progress": progress,
            "historical_lower_alpha": lower_prior_alpha(progress),
            "historical_upper_alpha": upper_prior_alpha(progress),
            "pending_required_physics_ablations": True,
        }
        # The active event file is intentionally not scanned here: it is a
        # large append-only file owned by the live 16k-env training process.
        # Deterministic physics ablations are the required source-selection
        # evidence anyway, so historical values remain explicitly pending.
        row.update({key: None for key in TAGS})
        candidates.append(row)
    # The onset is a historical warning only.  It does not make a selection.
    onset = None
    payload = {
        "status": "pending_physics_prior_ablations",
        "run": str(RUN.resolve()),
        "checkpoint_count": len(candidates),
        "schedule_total_updates": TOTAL_UPDATES,
        "candidate_pool": candidates,
        "degradation_onset_candidate": onset,
        "model7000": next((row for row in candidates if row["iteration"] == 7000), None),
        "selection": None,
        "reason": "Selection is prohibited until Historical/HalfUpper/UpperOff/AllOff deterministic evaluations and Pareto scoring are complete. The live TensorBoard file is deliberately not scanned while its owner is training.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoint_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT / "checkpoint_metrics.json")


if __name__ == "__main__":
    main()

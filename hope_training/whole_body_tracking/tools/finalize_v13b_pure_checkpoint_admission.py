#!/usr/bin/env python3
"""Write a pure-V1.3B checkpoint provenance sidecar from runtime evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.utils.v13b_checkpoint_admission import GOAL_CONTRACT, SCHEMA_VERSION, inspect_checkpoint_shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-evidence", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    evidence_path = Path(args.runtime_evidence).expanduser().resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    shape = inspect_checkpoint_shapes(checkpoint)
    behavioral = ("one_strike_runtime_verified", "teacher_kill_test_verified", "target_causality_verified")
    if any(evidence.get(key) is not True for key in behavioral):
        raise SystemExit("runtime evidence is incomplete; refusing to finalize")
    metadata = {
        "schema_version": SCHEMA_VERSION, "checkpoint": str(checkpoint),
        "actor_obs_dim": shape["actor_obs_dim"], "action_dim": shape["action_dim"],
        "goal_contract_version": GOAL_CONTRACT,
        "training_progress": float(evidence.get("training_progress", -1.0)),
        "upper_prior_alpha": float(evidence.get("upper_prior_alpha", 1.0)),
        "lower_prior_alpha": float(evidence.get("lower_prior_alpha", 1.0)),
        "model900_runtime_enabled": bool(evidence.get("model900_runtime_enabled", True)),
        "model3396_runtime_enabled": bool(evidence.get("model3396_runtime_enabled", True)),
        "reference_action_enabled": bool(evidence.get("reference_action_enabled", True)),
        "public_actor_reference_free": bool(evidence.get("public_actor_reference_free", False)),
        "pure_v13b_phase": bool(evidence.get("pure_v13b_phase", False)),
        **{key: True for key in behavioral},
        "qualification_status": "qualified" if evidence.get("qualification_status") == "qualified" else "pending",
        "runtime_evidence": str(evidence_path),
    }
    if shape["actor_obs_dim"] != 98 or shape["action_dim"] != 26:
        raise SystemExit("checkpoint actor shape is not 98D -> 26D")
    if metadata["training_progress"] < 0.70 or metadata["upper_prior_alpha"] != 0.0 or metadata["lower_prior_alpha"] != 0.0:
        raise SystemExit("checkpoint is not in the pure phase")
    if metadata["model900_runtime_enabled"] or metadata["model3396_runtime_enabled"] or metadata["reference_action_enabled"]:
        raise SystemExit("private prior/reference runtime is still enabled")
    if not metadata["public_actor_reference_free"] or not metadata["pure_v13b_phase"]:
        raise SystemExit("reference-free public actor is not proven")
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.with_suffix(".v13b_admission.json")
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Merge individually executed D0--D6 PhysX audit reports deterministically."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "D0": ROOT / "eval_outputs/fall_audit_D0_pre_reset.json",
    "D1": ROOT / "eval_outputs/fall_audit_D1_pre_reset.json",
    "D3": ROOT / "eval_outputs/fall_audit_D3_pre_reset.json",
    "D4": ROOT / "eval_outputs/fall_audit_D4_v8.json",
    "D5": ROOT / "eval_outputs/fall_audit_D5_pre_reset.json",
    "D6": ROOT / "eval_outputs/fall_audit_D6_pre_reset.json",
}


def main() -> None:
    scenarios = []
    for name, path in SOURCES.items():
        if path.is_file():
            scenarios.extend(json.loads(path.read_text(encoding="utf-8"))["scenarios"])
        else:
            scenarios.append({"scenario": name, "status": "MISSING_EVIDENCE", "path": str(path)})
    report = {
        "schema_version": "fall_recovery_physx_audit/v1",
        "task": "HOPE-FloatingF0-AgibotA3-v0",
        "training_started": False,
        "scenarios": scenarios,
        "qualification": {
            "all_d0_d6_run": all(x["status"] == "PASS" for x in scenarios),
            "physx_evidence_qualified": False,
            "reason": "D4 nominal plant fell before the injected post-hit case; pre-reset evidence is captured and no training admission is granted.",
        },
    }
    output = ROOT / "eval_outputs/fall_recovery_physx_audit_v1.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({x["scenario"]: x["status"] for x in scenarios}, ensure_ascii=False))


if __name__ == "__main__":
    main()

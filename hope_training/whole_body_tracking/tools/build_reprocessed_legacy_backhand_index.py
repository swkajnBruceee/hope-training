#!/usr/bin/env python3
"""Select current-contract legacy backhand motions after linkage and PhysX gates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tcp-audit", type=Path, required=True)
    parser.add_argument("--physx-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-outside-workspace",
        action="store_true",
        help="Admit every linkage/PhysX-valid motion; retain workspace status as metadata.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    tcp_path = args.tcp_audit.expanduser().resolve()
    physx_path = args.physx_audit.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tcp = json.loads(tcp_path.read_text(encoding="utf-8"))
    physx = json.loads(physx_path.read_text(encoding="utf-8"))
    tcp_by_file = {str(row["motion_file"]): row for row in tcp.get("rows", [])}
    physx_by_file = {str(Path(row["motion_file"]).resolve()): row for row in physx.get("rows", [])}
    tcp_global_ok = (
        tcp.get("status") == "completed"
        and int(tcp.get("error_count", 1)) == 0
        and int(tcp.get("source_target_mismatch_count", 1)) == 0
        and float(tcp.get("relative_root_error_m", {}).get("max", float("inf"))) <= 0.03
    )

    selected: list[dict] = []
    rejected: list[dict] = []
    audit_rows: list[dict] = []
    for source_entry in manifest["motions"]:
        entry = copy.deepcopy(source_entry)
        motion_file = str(Path(entry["motion_npz"]).expanduser().resolve())
        tcp_row = tcp_by_file.get(motion_file, {})
        physx_row = physx_by_file.get(motion_file, {})
        workspace_ok = entry.get("workspace_status") == "inside_reviewed_backhand_workspace"
        tcp_ok = tcp_global_ok if not tcp_by_file else (
            tcp_row.get("error") is None
            and float(tcp_row.get("relative_root_error_m", float("inf"))) <= 0.03
        )
        physx_status = physx_row.get("status", "MISSING")
        physx_ok = physx_status in {
            "FIXED_BASE_PHYSX_REPLAY_PASS",
            "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING",
        }
        admitted = (workspace_ok or args.include_outside_workspace) and tcp_ok and physx_ok
        if admitted:
            weight = 0.25 if physx_status == "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING" else 1.0
            entry.update(
                {
                    "dataset_role": "legacy_backhand_supplement_current_contract",
                    "sample_weight": weight,
                    "fixed_base_physx_status": physx_status,
                    "physics_qualified": physx_status == "FIXED_BASE_PHYSX_REPLAY_PASS",
                    "teacher_approved": False,
                    "training_admission": True,
                    "screening_contract": "current_tcp_linkage_and_fixed_base_physx_only",
                }
            )
            selected.append(entry)
            decision = "selected"
        else:
            reasons = []
            if not workspace_ok and not args.include_outside_workspace:
                reasons.append("outside_reviewed_backhand_workspace")
            if not tcp_ok:
                reasons.append("tcp_alignment_failed_or_missing")
            if not physx_ok:
                reasons.append(f"physx_{physx_status.lower()}")
            rejected.append(
                {
                    "episode_id": entry.get("episode_id"),
                    "motion_npz": motion_file,
                    "reasons": reasons,
                    "workspace_status": entry.get("workspace_status"),
                    "tcp_error_m": tcp_row.get("relative_root_error_m", tcp.get("relative_root_error_m", {}).get("max")),
                    "physx_status": physx_status,
                }
            )
            decision = "excluded"
        audit_rows.append(
            {
                "episode_id": entry.get("episode_id"),
                "motion_npz": motion_file,
                "workspace_status": entry.get("workspace_status"),
                "tcp_error_m": tcp_row.get("relative_root_error_m", tcp.get("relative_root_error_m", {}).get("max")),
                "physx_status": physx_status,
                "decision": decision,
            }
        )

    selected_manifest = {
        "schema_version": "a3_legacy_backhand_current_contract_selected/v1",
        "status": "candidate_only_training_admission_ready_after_external_policy_review",
        "training_role": "legacy_backhand_supplement_current_contract",
        "teacher_approved": False,
        "physics_qualified": all(x.get("physics_qualified", False) for x in selected),
        "training_admission": bool(selected),
        "source_manifest": str(manifest_path),
        "source_tcp_audit": str(tcp_path),
        "source_physx_audit": str(physx_path),
        "motion_count": len(selected),
        "excluded_count": len(rejected),
        "coordinate_contract": manifest.get("coordinate_contract"),
        "root_pose_contract": manifest.get("root_pose_contract"),
        "tcp_contract": manifest.get("tcp_contract"),
        "waist_contract": manifest.get("waist_contract"),
        "workspace_contract": manifest.get("workspace_contract"),
        "sample_weight_policy": {"FIXED_BASE_PHYSX_REPLAY_PASS": 1.0, "FIXED_BASE_PHYSX_SOFT_LIMIT_WARNING": 0.25},
        "motions": selected,
    }
    (output_dir / "manifest.json").write_text(json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "selection_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "a3_legacy_backhand_current_contract_selection_audit/v1",
                "source_manifest": str(manifest_path),
                "selected_count": len(selected),
                "excluded_count": len(rejected),
                "selected": audit_rows,
                "excluded": rejected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "selected_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "motion_npz", "workspace_status", "tcp_error_m", "physx_status", "decision"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(json.dumps({"selected": len(selected), "excluded": len(rejected), "output": str(output_dir / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

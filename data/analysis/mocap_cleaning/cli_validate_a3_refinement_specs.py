#!/usr/bin/env python3
"""Dry-run validator for A3 constrained refinement specs."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_COORD_KEYS = [
    "position_frame",
    "orientation_frame",
    "position_unit",
    "angle_unit",
    "time_unit",
    "quat_order",
    "fps",
    "dt",
    "hit_index",
    "hit_timestamp_rel_s",
    "sequence_length_frames",
]

REQUIRED_HIT_KEYS = [
    "hit_index",
    "time_rel_s",
    "racket_position_m",
    "racket_quat_xyzw",
    "racket_normal_w",
    "racket_tangent_w",
    "racket_up_w",
    "racket_velocity_mps",
    "racket_velocity_direction_w",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _check_vec(name: str, value: Any, expected_len: int, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != expected_len:
        errors.append(f"{name}: expected list len {expected_len}")


def _validate_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if spec.get("spec_version") != "1.1.0":
        errors.append("spec_version: expected 1.1.0")
    if spec.get("contract_version") != "a3_refinement_contract_v1":
        errors.append("contract_version: expected a3_refinement_contract_v1")
    coord = spec.get("coordinate_contract", {})
    for key in REQUIRED_COORD_KEYS:
        if key not in coord:
            errors.append(f"coordinate_contract.{key}: missing")
    if coord.get("quat_order") != "xyzw":
        errors.append("coordinate_contract.quat_order: expected xyzw")
    if coord.get("position_unit") != "m":
        errors.append("coordinate_contract.position_unit: expected m")
    if coord.get("time_unit") != "s":
        errors.append("coordinate_contract.time_unit: expected s")

    hit = spec.get("hit_target", {})
    for key in REQUIRED_HIT_KEYS:
        if key not in hit:
            errors.append(f"hit_target.{key}: missing")
    _check_vec("hit_target.racket_position_m", hit.get("racket_position_m"), 3, errors)
    _check_vec("hit_target.racket_quat_xyzw", hit.get("racket_quat_xyzw"), 4, errors)
    _check_vec("hit_target.racket_normal_w", hit.get("racket_normal_w"), 3, errors)
    _check_vec("hit_target.racket_tangent_w", hit.get("racket_tangent_w"), 3, errors)
    _check_vec("hit_target.racket_up_w", hit.get("racket_up_w"), 3, errors)
    _check_vec("hit_target.racket_velocity_direction_w", hit.get("racket_velocity_direction_w"), 3, errors)

    masks = spec.get("joint_masks", {})
    all_joints = set(spec.get("a3_joint_order", []))
    for key in ("active_joints_first_pass", "locked_joints_first_pass", "weak_track_joints_first_pass"):
        if key not in masks or not isinstance(masks[key], list) or not masks[key]:
            errors.append(f"joint_masks.{key}: missing or empty")
        elif not set(masks[key]).issubset(all_joints):
            errors.append(f"joint_masks.{key}: contains joints outside a3_joint_order")

    windows = spec.get("windows", {})
    sequence_length_frames = int(coord.get("sequence_length_frames", 0) or 0)
    for name in ("pre_hit", "hit", "post_hit"):
        block = windows.get(name, {})
        for key in ("frame_start", "frame_end", "time_rel_start_s", "time_rel_end_s"):
            if key not in block:
                errors.append(f"windows.{name}.{key}: missing")
        if block and block.get("frame_start", 0) > block.get("frame_end", 0):
            errors.append(f"windows.{name}: frame_start > frame_end")
        if sequence_length_frames > 0 and block:
            if int(block.get("frame_start", -1)) < 0 or int(block.get("frame_end", -1)) >= sequence_length_frames:
                errors.append(f"windows.{name}: frame out of range")

    thresholds = spec.get("quality_thresholds", {})
    if "warning" not in thresholds or "reject" not in thresholds:
        errors.append("quality_thresholds: expected warning and reject")
    else:
        for key, reject_value in thresholds["reject"].items():
            warning_value = thresholds["warning"].get(key)
            if warning_value is None:
                errors.append(f"quality_thresholds.warning.{key}: missing")
            elif warning_value > reject_value:
                errors.append(f"quality_thresholds.{key}: warning > reject")

    artifacts = spec.get("artifacts", {})
    for key in ("generic_retarget_csv", "retarget_csv", "refined_retarget_csv", "motion_npz", "quality_report_json", "refinement_spec_json"):
        if key not in artifacts:
            errors.append(f"artifacts.{key}: missing")

    fingerprints = spec.get("input_fingerprints", {})
    for key in ("source_sample_npz", "source_clean_npz", "source_debug_npz", "source_bvh", "generic_retarget_csv"):
        if key not in fingerprints:
            errors.append(f"input_fingerprints.{key}: missing")

    a3 = spec.get("a3_bodies", {})
    for key in (
        "wrist_body",
        "racket_body",
        "wrist_to_racket_pos_m",
        "wrist_to_racket_quat_xyzw",
        "racket_center_body",
        "racket_normal_axis",
        "racket_tangent_axis",
        "racket_up_axis",
    ):
        if key not in a3:
            errors.append(f"a3_bodies.{key}: missing")

    vel_dir = hit.get("racket_velocity_direction_w")
    if isinstance(vel_dir, list) and len(vel_dir) == 3 and all(abs(float(x)) < 1e-9 for x in vel_dir):
        errors.append("hit_target.racket_velocity_direction_w: zero vector")

    return errors


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# A3 Refinement Spec Validation",
        "",
        f"- manifest: `{report['manifest']}`",
        f"- checked: `{report['checked']}`",
        f"- passed: `{report['passed']}`",
        f"- failed: `{report['failed']}`",
        "",
        "## Failures",
        "",
    ]
    if not report["failures"]:
        lines.append("- none")
    else:
        for item in report["failures"]:
            lines.append(f"- `{item['job_id']}`: {'; '.join(item['errors'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3/refinement_spec_manifest.json"),
    )
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/mocap_cleaning_outputs/DATA260703_combined/retarget_jobs/agibot_a3"),
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    specs = manifest["specs"]
    requested = int(args.sample_count)
    if requested <= 0:
        selected = specs
    else:
        sample_count = min(requested, len(specs))
        if sample_count <= 0:
            raise ValueError("no specs available to validate")
        stride = max(1, len(specs) // sample_count)
        selected = [specs[i] for i in range(0, len(specs), stride)][:sample_count]

    failures = []
    passed = 0
    for item in selected:
        spec = _load_json(Path(item["spec_path"]))
        errors = _validate_spec(spec)
        if errors:
            failures.append({"job_id": item["job_id"], "errors": errors})
        else:
            passed += 1

    report = {
        "manifest": str(args.manifest),
        "checked": len(selected),
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "refinement_spec_validation.json"
    md_path = args.output_dir / "refinement_spec_validation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(md_path, report)
    print(f"Checked {report['checked']} specs")
    print(f"Passed {report['passed']}")
    print(f"Failed {report['failed']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

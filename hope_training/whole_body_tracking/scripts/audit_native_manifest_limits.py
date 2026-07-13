#!/usr/bin/env python3
"""Audit native-strike manifest joint-limit feasibility.

This is a static data QA pass. It does not start Isaac Sim. It checks the
reference motion NPZ at the hit frame against the A3 URDF joint limits for the
native strike action joints: waist + right arm.

The NPZ joint_pos array is saved in Isaac articulation order. Do not use
AGIBOT_A3_JOINT_NAMES as the NPZ order; that list is the CSV/GMR input column
order for scripts/csv_to_npz.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _load_a3_constants(path: Path) -> dict[str, object]:
    module = ast.parse(path.read_text())
    values: dict[str, object] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        try:
            values[name] = ast.literal_eval(node.value)
        except Exception:
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                left = values.get(node.value.left.id) if isinstance(node.value.left, ast.Name) else None
                right = values.get(node.value.right.id) if isinstance(node.value.right, ast.Name) else None
                if left is not None and right is not None:
                    values[name] = left + right
    return values


def _load_urdf_limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def _manifest_motions(obj: object) -> list[dict]:
    if isinstance(obj, dict) and isinstance(obj.get("motions"), list):
        return obj["motions"]
    if isinstance(obj, list):
        return obj
    raise ValueError("manifest must be a list or contain a 'motions' list")


def _motion_hit_frame(motion: dict) -> int:
    hit = motion.get("hit_event") or {}
    if "motion_hit_frame" in hit:
        return int(hit["motion_hit_frame"])
    if "hit_frame" in motion:
        return int(motion["hit_frame"])
    raise KeyError(f"missing hit frame for {motion.get('episode_id', '<unknown>')}")


def _resolve_motion_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    manifest_relative = (manifest_path.parent / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return manifest_relative


def _load_articulation_joint_names(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    names = obj.get("joint_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{path}: missing string list field 'joint_names'")
    return names


def audit_manifest(
    manifest_path: Path,
    robot_constants_path: Path,
    urdf_path: Path,
    articulation_metadata_path: Path,
    near_margin: float,
) -> tuple[list[dict], dict]:
    constants = _load_a3_constants(robot_constants_path)
    native_joints = constants["A3_NATIVE_STRIKE_JOINTS"]
    joint_names = _load_articulation_joint_names(articulation_metadata_path)
    limits = _load_urdf_limits(urdf_path)
    native_ids = [joint_names.index(name) for name in native_joints]

    obj = json.loads(manifest_path.read_text())
    motions = _manifest_motions(obj)
    rows: list[dict] = []
    for rank, motion in enumerate(motions, start=1):
        hit_frame = _motion_hit_frame(motion)
        motion_npz = _resolve_motion_path(str(motion.get("library_motion_npz") or motion["motion_npz"]), manifest_path)
        joint_pos = np.load(motion_npz)["joint_pos"]
        q = joint_pos[hit_frame, native_ids]

        margins: dict[str, float] = {}
        violating: list[str] = []
        near: list[str] = []
        for name, value in zip(native_joints, q):
            lower, upper = limits[name]
            span = max(upper - lower, 1.0e-9)
            margin = min(float(value - lower), float(upper - value)) / span
            margins[name] = margin
            if margin < 0.0:
                violating.append(name)
            if margin < near_margin:
                near.append(name)

        rows.append(
            {
                "rank": rank,
                "episode_id": motion.get("episode_id", str(rank)),
                "stroke_type": motion.get("stroke_type", "unknown"),
                "motion_npz": str(motion_npz),
                "hit_frame": hit_frame,
                "min_margin": min(margins.values()),
                "near_fraction": len(near) / len(native_joints),
                "violation_count": len(violating),
                "near_count": len(near),
                "violating_joints": violating,
                "near_joints": near,
                "native_joint_margins": margins,
                "native_joint_values": {name: float(value) for name, value in zip(native_joints, q)},
            }
        )
    return rows, obj if isinstance(obj, dict) else {"motions": motions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--robot-constants", type=Path, default=Path("training/robots/agibot_a3.py"))
    parser.add_argument("--urdf", type=Path, default=Path("training/assets/agibot_a3/urdf/model.urdf"))
    parser.add_argument(
        "--articulation-metadata",
        type=Path,
        default=Path("docs/a3_articulation_metadata.json"),
        help="JSON produced by scripts/export_a3_articulation_metadata.py.",
    )
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--filtered-manifest-out", type=Path)
    parser.add_argument("--min-margin", type=float, default=0.05)
    args = parser.parse_args()

    rows, manifest_obj = audit_manifest(
        args.manifest.resolve(),
        args.robot_constants,
        args.urdf,
        args.articulation_metadata,
        args.near_margin,
    )
    valid = [row for row in rows if row["min_margin"] >= args.min_margin]
    strokes = sorted({row["stroke_type"] for row in rows})
    print(
        f"total={len(rows)} valid_min_margin>={args.min_margin:g}={len(valid)} "
        + " ".join(f"{stroke}={sum(1 for row in valid if row['stroke_type'] == stroke)}" for stroke in strokes)
    )
    for row in rows:
        status = "OK" if row["min_margin"] >= args.min_margin else ("NEAR" if row["min_margin"] >= 0 else "VIOL")
        print(
            f"{row['rank']:03d} {status:4s} {row['stroke_type']:8s} {row['episode_id']} "
            f"min_margin={row['min_margin']:.4f} near_fraction={row['near_fraction']:.2f} "
            f"viol={','.join(row['violating_joints']) or '-'} near={','.join(row['near_joints']) or '-'}"
        )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "episode_id",
                    "stroke_type",
                    "motion_npz",
                    "hit_frame",
                    "min_margin",
                    "near_fraction",
                    "violation_count",
                    "near_count",
                    "violating_joints",
                    "near_joints",
                ],
            )
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["violating_joints"] = "|".join(row["violating_joints"])
                out["near_joints"] = "|".join(row["near_joints"])
                out.pop("native_joint_margins", None)
                out.pop("native_joint_values", None)
                writer.writerow(out)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
    if args.filtered_manifest_out:
        valid_ids = {row["episode_id"] for row in valid}
        motions = [motion for motion in _manifest_motions(manifest_obj) if motion.get("episode_id") in valid_ids]
        filtered = dict(manifest_obj)
        filtered["motions"] = motions
        filtered["replay_ready_count"] = len(motions)
        filtered["native_joint_limit_audit"] = {
            "source_manifest": str(args.manifest),
            "articulation_metadata": str(args.articulation_metadata),
            "near_margin": args.near_margin,
            "min_margin": args.min_margin,
            "valid_count": len(motions),
        }
        args.filtered_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.filtered_manifest_out.write_text(json.dumps(filtered, indent=2))


if __name__ == "__main__":
    main()

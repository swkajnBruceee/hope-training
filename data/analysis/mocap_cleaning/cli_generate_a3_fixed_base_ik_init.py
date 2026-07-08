#!/usr/bin/env python3
"""Generate fixed-base A3 IK initialization CSVs for competition targets."""

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from analysis.mocap_cleaning.a3_metadata import A3_DEFAULT_JOINT_POS, A3_POLICY_JOINT_ORDER
from analysis.mocap_cleaning.a3_refinement_solver import _fk_racket_state, load_a3_joint_limits, write_retarget_csv
from analysis.mocap_cleaning.config import load_config


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _default_joint_vector() -> np.ndarray:
    return np.asarray([float(A3_DEFAULT_JOINT_POS.get(name, 0.0)) for name in A3_POLICY_JOINT_ORDER], dtype=np.float64)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norm, 1e-9))


def _target_normals(quat_xyzw: np.ndarray) -> np.ndarray:
    normals = []
    for quat in quat_xyzw:
        normals.append(_quat_xyzw_to_matrix(quat)[:, 1])
    return np.asarray(normals, dtype=np.float64)


def _solve_frame(
    *,
    x0: np.ndarray,
    q_full_default: np.ndarray,
    active_idx: list[int],
    lower: np.ndarray,
    upper: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    target_pos: np.ndarray,
    target_normal: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    target_normal = target_normal / max(float(np.linalg.norm(target_normal)), 1e-9)

    def residual(q_active: np.ndarray) -> np.ndarray:
        q = q_full_default.copy()
        q[active_idx] = q_active
        pos, rot = _fk_racket_state(base_pos, base_quat, q, joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        normal = rot[:, 1]
        return np.concatenate(
            [
                float(weights["position_weight"]) * (pos - target_pos),
                float(weights["normal_weight"]) * (normal - target_normal),
                float(weights["regularization_weight"]) * (q_active - x0),
            ]
        )

    result = least_squares(
        residual,
        x0=np.clip(x0, lower, upper),
        bounds=(lower, upper),
        max_nfev=int(weights["max_nfev_per_frame"]),
        verbose=0,
    )
    return result.x.astype(np.float64), {"cost": float(result.cost), "nfev": int(result.nfev), "success": bool(result.success)}


def _evaluate(csv_data: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray, hit_index: int, base_pos: np.ndarray, base_quat: np.ndarray) -> dict[str, Any]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    pos = []
    normal = []
    for row in csv_data:
        p, r = _fk_racket_state(base_pos, base_quat, row[7:], joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        pos.append(p)
        normal.append(r[:, 1])
    pos = np.asarray(pos)
    normal = np.asarray(normal)
    target_normal = _target_normals(target_quat)
    pos_err = np.linalg.norm(pos - target_pos, axis=1)
    normal_cos = np.sum(_normalize_rows(normal) * _normalize_rows(target_normal), axis=1)
    normal_err_deg = np.degrees(np.arccos(np.clip(normal_cos, -1.0, 1.0)))
    return {
        "racket_position_error_at_hit_m": float(pos_err[hit_index]),
        "racket_position_error_p50_m": float(np.nanpercentile(pos_err, 50)),
        "racket_position_error_p90_m": float(np.nanpercentile(pos_err, 90)),
        "racket_orientation_error_at_hit_deg": float(normal_err_deg[hit_index]),
        "racket_orientation_error_p50_deg": float(np.nanpercentile(normal_err_deg, 50)),
        "racket_orientation_error_p90_deg": float(np.nanpercentile(normal_err_deg, 90)),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# A3 P2 Fixed-Base IK Initialization",
        "",
        f"- processed: `{report['processed']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(report["status_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- manifest: `{report['manifest']}`")
    lines.append(f"- quality dir: `{report['quality_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(str(config["output_root"]))
    manifest_path = args.manifest or output_root / "retarget_ready" / "retarget_target_manifest.json"
    manifest = json.loads(Path(manifest_path).read_text())
    samples = manifest["samples"]
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    ik_dir = output_root / "ik_init_csv"
    quality_dir = output_root / "ik_quality_reports"
    ik_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    limits = load_a3_joint_limits()
    active_names = [str(x) for x in config["ik"]["active_joints"]]
    active_idx = [A3_POLICY_JOINT_ORDER.index(name) for name in active_names]
    lower = np.asarray([limits[name][0] for name in active_names], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in active_names], dtype=np.float64)
    base_pos = np.asarray(config["robot_base"]["position_m"], dtype=np.float64)
    base_quat = np.asarray(config["robot_base"]["quat_xyzw"], dtype=np.float64)
    weights = dict(config["ik"])

    entries = []
    status_counts = Counter()
    for item in samples:
        target = np.load(item["target_npz"], allow_pickle=True)
        target_pos = target["racket_pos"].astype(np.float64)
        target_quat = target["racket_quat"].astype(np.float64)
        target_normal = _target_normals(target_quat)
        hit_index = int(target["hit_index"])
        q_default = _default_joint_vector()
        q_active = q_default[active_idx].copy()
        rows = []
        frame_reports = []
        for frame in range(target_pos.shape[0]):
            q_active, frame_report = _solve_frame(
                x0=q_active,
                q_full_default=q_default,
                active_idx=active_idx,
                lower=lower,
                upper=upper,
                base_pos=base_pos,
                base_quat=base_quat,
                target_pos=target_pos[frame],
                target_normal=target_normal[frame],
                weights=weights,
            )
            q_full = q_default.copy()
            q_full[active_idx] = q_active
            rows.append(np.concatenate([base_pos, base_quat, q_full]))
            frame_reports.append(frame_report)
        csv_data = np.asarray(rows, dtype=np.float64)
        episode_id = str(item["episode_id"])
        csv_path = ik_dir / f"{episode_id}.csv"
        write_retarget_csv(csv_path, csv_data)
        metrics = _evaluate(csv_data, target_pos, target_quat, hit_index, base_pos, base_quat)
        metrics.update(
            {
                "episode_id": episode_id,
                "target_spec_json": item["target_spec_json"],
                "ik_init_csv": str(csv_path),
                "active_joints": active_names,
                "frame_solver_success_count": int(sum(r["success"] for r in frame_reports)),
                "frames": int(target_pos.shape[0]),
            }
        )
        status = "pass" if metrics["racket_position_error_at_hit_m"] <= float(config["quality_thresholds"]["hit_position_reject_m"]) else "reject"
        metrics["status"] = status
        status_counts[status] += 1
        quality_path = quality_dir / f"{episode_id}.json"
        quality_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        entries.append({**item, "ik_init_csv": str(csv_path), "ik_quality_report": str(quality_path), "ik_status": status})

    out_manifest = {
        **manifest,
        "stage": "a3_fixed_base_ik_init",
        "processed_count": len(entries),
        "samples": entries,
    }
    out_manifest_path = output_root / "ik_init_manifest.json"
    out_manifest_path.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "processed": len(entries),
        "status_counts": dict(status_counts),
        "manifest": str(out_manifest_path),
        "quality_dir": str(quality_dir),
    }
    report_path = output_root / "ik_init_summary.md"
    _write_markdown(report, report_path)
    print(f"Processed {len(entries)} IK targets")
    print(dict(status_counts))
    print(f"Wrote {out_manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

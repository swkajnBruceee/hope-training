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


WAIST_YAW_ABS_LIMIT_RAD = 1.00
WAIST_YAW_REGULARIZATION_SCALE = 5.0
WAIST_YAW_NOMINAL_WEIGHT = 0.20
RIGHT_WRIST_JOINTS = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


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


def _fill_nonfinite_rows(values: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.asarray(values, dtype=np.float64).copy()
    bad = ~np.isfinite(out).all(axis=1)
    bad_count = int(np.sum(bad))
    if bad_count == 0:
        return out, 0
    fallback = np.asarray(fallback, dtype=np.float64)
    last = fallback.copy()
    for idx in range(out.shape[0]):
        if np.isfinite(out[idx]).all():
            last = out[idx].copy()
        else:
            out[idx] = last
    next_valid = fallback.copy()
    for idx in range(out.shape[0] - 1, -1, -1):
        if np.isfinite(values[idx]).all():
            next_valid = out[idx].copy()
        elif not np.isfinite(out[idx]).all():
            out[idx] = next_valid
    out[~np.isfinite(out)] = np.broadcast_to(fallback, out.shape)[~np.isfinite(out)]
    return out, bad_count


def _target_normals(quat_xyzw: np.ndarray) -> np.ndarray:
    normals = []
    for quat in quat_xyzw:
        normals.append(_quat_xyzw_to_matrix(quat)[:, 1])
    return np.asarray(normals, dtype=np.float64)


def _target_axes(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = []
    tangents = []
    for quat in quat_xyzw:
        rot = _quat_xyzw_to_matrix(quat)
        normals.append(rot[:, 1])
        tangents.append(rot[:, 0])
    return np.asarray(normals, dtype=np.float64), np.asarray(tangents, dtype=np.float64)


def _config_float(mapping: dict[str, Any], key: str, default: float) -> float:
    return float(mapping.get(key, default))


def _angle_delta(values: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    return (values - neutral + np.pi) % (2.0 * np.pi) - np.pi


def _joint_weight_vector(weights: dict[str, Any], key: str, active_names: list[str]) -> np.ndarray:
    configured = weights.get(key, {})
    out = np.zeros(len(active_names), dtype=np.float64)
    if isinstance(configured, dict):
        for idx, name in enumerate(active_names):
            out[idx] = float(configured.get(name, 0.0))
    return out


def _deadband_vector(weights: dict[str, Any], active_names: list[str]) -> np.ndarray:
    configured = weights.get("joint_neutral_deadband_rad", {})
    out = np.zeros(len(active_names), dtype=np.float64)
    if isinstance(configured, dict):
        for idx, name in enumerate(active_names):
            out[idx] = max(float(configured.get(name, 0.0)), 0.0)
    return out


def _deadband_delta(delta: np.ndarray, deadband: np.ndarray) -> np.ndarray:
    return np.sign(delta) * np.maximum(np.abs(delta) - deadband, 0.0)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


def _comfort_range_penalty(q_active: np.ndarray, weights: dict[str, Any], active_names: list[str]) -> np.ndarray:
    ranges = weights.get("joint_comfort_ranges_rad", {})
    comfort_weight = _config_float(weights, "joint_comfort_weight", 0.0)
    if comfort_weight <= 0.0 or not isinstance(ranges, dict):
        return np.zeros(0, dtype=np.float64)
    softness = max(_config_float(weights, "joint_comfort_softness_rad", 0.0), 0.0)
    residuals = []
    for idx, name in enumerate(active_names):
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        value = float(q_active[idx])
        lower_excess = float(lo) - value
        upper_excess = value - float(hi)
        if softness > 0.0:
            residuals.append(softness * _softplus(np.asarray(lower_excess / softness)))
            residuals.append(softness * _softplus(np.asarray(upper_excess / softness)))
        else:
            residuals.append(max(lower_excess, 0.0))
            residuals.append(max(upper_excess, 0.0))
    if not residuals:
        return np.zeros(0, dtype=np.float64)
    return comfort_weight * np.asarray(residuals, dtype=np.float64)


def _wrist_naturalness_metrics(joint_pos: np.ndarray, hit_index: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in RIGHT_WRIST_JOINTS:
        idx = A3_POLICY_JOINT_ORDER.index(name)
        neutral = float(A3_DEFAULT_JOINT_POS.get(name, 0.0))
        delta_deg = np.degrees(np.abs(_angle_delta(joint_pos[:, idx], neutral)))
        prefix = name.replace("_joint", "")
        metrics[f"{prefix}_neutral_delta_hit_deg"] = float(delta_deg[hit_index])
        metrics[f"{prefix}_neutral_delta_p95_deg"] = float(np.nanpercentile(delta_deg, 95))
        metrics[f"{prefix}_neutral_delta_max_deg"] = float(np.nanmax(delta_deg))
    metrics["right_wrist_bend_pitch_yaw_p95_deg"] = float(
        max(
            metrics["right_wrist_pitch_neutral_delta_p95_deg"],
            metrics["right_wrist_yaw_neutral_delta_p95_deg"],
        )
    )
    metrics["right_wrist_bend_pitch_yaw_max_deg"] = float(
        max(
            metrics["right_wrist_pitch_neutral_delta_max_deg"],
            metrics["right_wrist_yaw_neutral_delta_max_deg"],
        )
    )
    return metrics


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
    target_tangent: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    active_names = [A3_POLICY_JOINT_ORDER[i] for i in active_idx]
    waist_yaw_local_idx = active_names.index("waist_yaw_joint") if "waist_yaw_joint" in active_names else None
    neutral_active = q_full_default[active_idx]
    naturalness_weight = _joint_weight_vector(weights, "joint_neutral_weights", active_names)
    naturalness_deadband = _deadband_vector(weights, active_names)
    target_normal = target_normal / max(float(np.linalg.norm(target_normal)), 1e-9)
    target_tangent = target_tangent / max(float(np.linalg.norm(target_tangent)), 1e-9)
    reg_weight = np.ones(len(active_idx), dtype=np.float64) * float(weights["regularization_weight"])
    waist_yaw_abs_limit = _config_float(weights, "waist_yaw_abs_limit_rad", WAIST_YAW_ABS_LIMIT_RAD)
    if waist_yaw_local_idx is not None:
        reg_weight[waist_yaw_local_idx] *= WAIST_YAW_REGULARIZATION_SCALE

    def residual(q_active: np.ndarray) -> np.ndarray:
        q = q_full_default.copy()
        q[active_idx] = q_active
        pos, rot = _fk_racket_state(base_pos, base_quat, q, joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        normal = rot[:, 1]
        tangent = rot[:, 0]
        terms = [
            float(weights["position_weight"]) * (pos - target_pos),
            float(weights["normal_weight"]) * (normal - target_normal),
            _config_float(weights, "tangent_weight", 0.0) * (tangent - target_tangent),
            reg_weight * (q_active - x0),
        ]
        if np.any(naturalness_weight > 0.0):
            terms.append(naturalness_weight * _deadband_delta(_angle_delta(q_active, neutral_active), naturalness_deadband))
        comfort = _comfort_range_penalty(q_active, weights, active_names)
        if comfort.size:
            terms.append(comfort)
        if waist_yaw_local_idx is not None:
            waist_default = q_full_default[active_idx[waist_yaw_local_idx]]
            terms.append(np.asarray([WAIST_YAW_NOMINAL_WEIGHT * (q_active[waist_yaw_local_idx] - waist_default)], dtype=np.float64))
        return np.concatenate(terms)

    solve_lower = lower.copy()
    solve_upper = upper.copy()
    if waist_yaw_local_idx is not None:
        waist_default = q_full_default[active_idx[waist_yaw_local_idx]]
        solve_lower[waist_yaw_local_idx] = max(solve_lower[waist_yaw_local_idx], waist_default - waist_yaw_abs_limit)
        solve_upper[waist_yaw_local_idx] = min(solve_upper[waist_yaw_local_idx], waist_default + waist_yaw_abs_limit)

    result = least_squares(
        residual,
        x0=np.clip(x0, solve_lower, solve_upper),
        bounds=(solve_lower, solve_upper),
        max_nfev=int(weights["max_nfev_per_frame"]),
        verbose=0,
    )
    return result.x.astype(np.float64), {"cost": float(result.cost), "nfev": int(result.nfev), "success": bool(result.success)}


def _evaluate(csv_data: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray, hit_index: int, base_pos: np.ndarray, base_quat: np.ndarray) -> dict[str, Any]:
    joint_index_by_name = {name: i for i, name in enumerate(A3_POLICY_JOINT_ORDER)}
    pos = []
    normal = []
    tangent = []
    for row in csv_data:
        p, r = _fk_racket_state(base_pos, base_quat, row[7:], joint_index_by_name, {"a3_joint_order": A3_POLICY_JOINT_ORDER})
        pos.append(p)
        normal.append(r[:, 1])
        tangent.append(r[:, 0])
    pos = np.asarray(pos)
    normal = np.asarray(normal)
    tangent = np.asarray(tangent)
    target_normal, target_tangent = _target_axes(target_quat)
    pos_err = np.linalg.norm(pos - target_pos, axis=1)
    normal_cos = np.sum(_normalize_rows(normal) * _normalize_rows(target_normal), axis=1)
    normal_err_deg = np.degrees(np.arccos(np.clip(normal_cos, -1.0, 1.0)))
    tangent_cos = np.sum(_normalize_rows(tangent) * _normalize_rows(target_tangent), axis=1)
    tangent_err_deg = np.degrees(np.arccos(np.clip(tangent_cos, -1.0, 1.0)))
    return {
        "racket_position_error_at_hit_m": float(pos_err[hit_index]),
        "racket_position_error_p50_m": float(np.nanpercentile(pos_err, 50)),
        "racket_position_error_p90_m": float(np.nanpercentile(pos_err, 90)),
        "racket_orientation_error_at_hit_deg": float(normal_err_deg[hit_index]),
        "racket_orientation_error_p50_deg": float(np.nanpercentile(normal_err_deg, 50)),
        "racket_orientation_error_p90_deg": float(np.nanpercentile(normal_err_deg, 90)),
        "racket_tangent_error_at_hit_deg": float(tangent_err_deg[hit_index]),
        "racket_tangent_error_p50_deg": float(np.nanpercentile(tangent_err_deg, 50)),
        "racket_tangent_error_p90_deg": float(np.nanpercentile(tangent_err_deg, 90)),
        **_wrist_naturalness_metrics(csv_data[:, 7:], hit_index),
    }


def _ik_pose_status(metrics: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    position_ok = metrics["racket_position_error_at_hit_m"] <= float(thresholds["hit_position_reject_m"])
    normal_ok = metrics["racket_orientation_error_at_hit_deg"] <= float(thresholds["hit_orientation_reject_deg"])
    tangent_gate = bool(thresholds.get("hit_tangent_gate", False))
    tangent_ok = (not tangent_gate) or metrics["racket_tangent_error_at_hit_deg"] <= _config_float(thresholds, "hit_tangent_reject_deg", float("inf"))
    if not position_ok:
        reasons.append("hit_position_error")
        return "unreachable", reasons
    if not normal_ok:
        reasons.append("hit_orientation_error")
        return "position_reachable", reasons
    if not tangent_ok:
        reasons.append("hit_tangent_error")
        return "pose_reachable", reasons
    return "seed_ready", reasons


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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the output root from the config so an experiment cannot overwrite another dataset.",
    )
    parser.add_argument(
        "--require-tangent-gate",
        action="store_true",
        help="Require the configured tangent reject threshold during formal IK admission.",
    )
    parser.add_argument(
        "--input-fps",
        type=int,
        default=None,
        help="Override config time.fps for a source dataset with a different sampling rate.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse an existing per-episode IK CSV and quality report under the output root. "
            "This makes a long batch resumable without changing its input manifest."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input_fps is not None:
        config["time"]["fps"] = int(args.input_fps)
    output_root = args.output_root or Path(str(config["output_root"]))
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
        episode_id = str(item["episode_id"])
        csv_path = ik_dir / f"{episode_id}.csv"
        quality_path = quality_dir / f"{episode_id}.json"
        if args.skip_existing and csv_path.is_file() and quality_path.is_file():
            metrics = json.loads(quality_path.read_text(encoding="utf-8"))
            status = str(metrics.get("status", "reject"))
            status_counts[status] += 1
            entries.append(
                {
                    **item,
                    "ik_init_csv": str(csv_path),
                    "ik_quality_report": str(quality_path),
                    "ik_status": status,
                    "ik_pose_status": str(metrics.get("ik_pose_status", "unknown")),
                }
            )
            continue
        target = np.load(item["target_npz"], allow_pickle=True)
        target_pos, nonfinite_target_pos_frames = _fill_nonfinite_rows(target["racket_pos"].astype(np.float64), np.zeros(3, dtype=np.float64))
        target_quat, nonfinite_target_quat_frames = _fill_nonfinite_rows(
            target["racket_quat"].astype(np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        target_normal, target_tangent = _target_axes(target_quat)
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
                target_tangent=target_tangent[frame],
                weights=weights,
            )
            q_full = q_default.copy()
            q_full[active_idx] = q_active
            rows.append(np.concatenate([base_pos, base_quat, q_full]))
            frame_reports.append(frame_report)
        csv_data = np.asarray(rows, dtype=np.float64)
        write_retarget_csv(csv_path, csv_data)
        metrics = _evaluate(csv_data, target_pos, target_quat, hit_index, base_pos, base_quat)
        quality_thresholds = dict(config["quality_thresholds"])
        if args.require_tangent_gate:
            quality_thresholds["hit_tangent_gate"] = True
        ik_pose_status, reject_reasons = _ik_pose_status(metrics, quality_thresholds)
        metrics.update(
            {
                "episode_id": episode_id,
                "target_spec_json": item["target_spec_json"],
                "ik_init_csv": str(csv_path),
                "active_joints": active_names,
                "frame_solver_success_count": int(sum(r["success"] for r in frame_reports)),
                "frames": int(target_pos.shape[0]),
                "ik_pose_status": ik_pose_status,
                "reject_reasons": reject_reasons,
                "tangent_gate_required": bool(args.require_tangent_gate),
                "nonfinite_target_pos_frames_filled": nonfinite_target_pos_frames,
                "nonfinite_target_quat_frames_filled": nonfinite_target_quat_frames,
            }
        )
        status = "pass" if ik_pose_status == "seed_ready" else "reject"
        metrics["status"] = status
        status_counts[status] += 1
        quality_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        entries.append(
            {
                **item,
                "ik_init_csv": str(csv_path),
                "ik_quality_report": str(quality_path),
                "ik_status": status,
                "ik_pose_status": ik_pose_status,
            }
        )

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

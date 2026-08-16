#!/usr/bin/env python3
"""Offline BH planner sweep and FH CF0/CF1/CF2 counterfactuals.

The script replays the same venue-fitted contact and coarse landing model used by the virtual-ball
command.  It never starts Isaac or changes a checkpoint.

BH sweep:
    target_vz_new = target_vz_old + clip(k * (v_ref - incoming_vz), +/- delta_max)

The default ``preserve_execution`` mode keeps the measured A5 tracking residual fixed, i.e.
actual_v_new = actual_v_old + target_v_new - target_v_old.  This is the cleanest offline estimate
of a Planner-only change.  ``ideal_target`` is also reported as an optimistic upper-bound mode.

FH counterfactuals:
    CF0 = actual velocity + actual normal
    CF1 = target velocity + actual normal
    CF2 = target velocity + target normal

FH horizontal sweep:
    target_v_xy_new = target_v_xy_old + (delta_vx, delta_vy)
    actual_v_new = actual_v_old + target_v_new - target_v_old
    This preserves each sample's measured execution residual while screening horizontal
    target corrections. It is an offline candidate search, not an Actor rerun.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from analyze_a5_failure_chain import _load_rows, _pct  # noqa: E402


TABLE_NEAR_X = 0.5
TABLE_SURFACE_Z = 0.76
TABLE_LENGTH = 2.74
TABLE_HALF_W = 1.525 / 2.0
NET_X = TABLE_NEAR_X + 1.37
FAR_X = TABLE_NEAR_X + TABLE_LENGTH
NET_CLEAR_Z = TABLE_SURFACE_Z + 0.1525 + 0.020
FAILURE_ORDER = [
    "MISS_CAPTURE",
    "NO_NET_CROSS",
    "NET_TOO_LOW",
    "NO_LANDING_WITHIN_HORIZON",
    "LAND_OWN_HALF",
    "LAND_OUT_FAR",
    "LAND_OUT_SIDE",
    "LAND_OUT_OTHER",
    "LEGAL",
]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--telemetry", nargs="+", required=True)
    p.add_argument("--venue-config", default="configs/ball_physics_venue.yaml")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--k-grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        help="Comma-separated BH compensation gains.",
    )
    p.add_argument(
        "--v-ref-grid",
        default="-0.25,0,0.25",
        help="Comma-separated BH reference incoming vertical velocities.",
    )
    p.add_argument("--delta-max", type=float, default=0.20)
    p.add_argument(
        "--fh-dvx-grid",
        default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4",
        help="Comma-separated FH horizontal target velocity x offsets (m/s).",
    )
    p.add_argument(
        "--fh-dvy-grid",
        default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4",
        help="Comma-separated FH horizontal target velocity y offsets (m/s).",
    )
    return p.parse_args()


def _vec(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def _params(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    pad = raw["contact"]["paddle"]
    return {
        "ball_radius": float(raw["ball"]["radius"]),
        "inertia_coeff": float(raw["ball"]["inertia_coeff"]),
        "paddle_a_t": float(pad["a_t"]),
        "paddle_b_t": float(pad["b_t"]),
        "paddle_mu": float(pad["mu_safety"]),
        "paddle_e_g1": float(pad["e_exp_g1"]),
        "paddle_e_g2": float(pad["e_exp_g2"]),
        "k_d": float(raw["flight"]["k_d"]),
        "k_m": float(raw["flight"]["k_m"]),
        "g": float(raw["flight"]["g"]),
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1.0e-9)


def _contact_batch(
    incoming_v: np.ndarray,
    racket_v: np.ndarray,
    racket_n: np.ndarray,
    incoming_w: np.ndarray,
    prm: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    r = prm["ball_radius"]
    c = prm["inertia_coeff"]
    n = _normalize(racket_n)
    approaching = np.sum((incoming_v - racket_v) * n, axis=1) > 0.0
    n = np.where(approaching[:, None], -n, n)
    lever = -r * n
    u = incoming_v + np.cross(incoming_w, lever) - racket_v
    u_n = np.sum(u * n, axis=1, keepdims=True)
    u_t = u - u_n * n
    u_t_mag = np.linalg.norm(u_t, axis=1, keepdims=True)
    e = np.clip(
        prm["paddle_e_g1"] * np.exp(prm["paddle_e_g2"] * np.abs(u_n)), 0.05, 0.95
    )
    cos_theta = np.abs(u_n) / np.maximum(np.hypot(u_t_mag, u_n), 1.0e-9)
    raw = (prm["paddle_a_t"] + prm["paddle_b_t"] * cos_theta) * u_t_mag
    cap = prm["paddle_mu"] * (1.0 + e) * np.abs(u_n)
    impulse = np.minimum(np.maximum(raw, 0.0), cap)
    delta_t = np.where(
        u_t_mag > 1.0e-9,
        -impulse * u_t / np.maximum(u_t_mag, 1.0e-9),
        0.0,
    )
    delta_n = -(1.0 + e) * u_n * n
    delta_w = -(1.0 / (c * r)) * np.cross(n, delta_t)
    return incoming_v + delta_n + delta_t, incoming_w + delta_w


def _accel(v: np.ndarray, omega: np.ndarray, prm: dict[str, float]) -> np.ndarray:
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    a = -prm["k_d"] * speed * v + prm["k_m"] * np.cross(omega, v)
    a[:, 2] -= prm["g"]
    return a


def _rk4(p: np.ndarray, v: np.ndarray, omega: np.ndarray, prm: dict[str, float], h: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = _accel(v, omega, prm)
    a2 = _accel(v + 0.5 * h * a1, omega, prm)
    a3 = _accel(v + 0.5 * h * a2, omega, prm)
    a4 = _accel(v + h * a3, omega, prm)
    v_new = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    p_new = p + (h / 6.0) * (
        v + 2.0 * (v + 0.5 * h * a1) + 2.0 * (v + 0.5 * h * a2) + (v + h * a3)
    )
    return p_new, v_new


def _rollout(
    rows: list[dict[str, Any]],
    racket_v: np.ndarray,
    racket_n: np.ndarray,
    prm: dict[str, float],
) -> dict[str, np.ndarray]:
    incoming_v = _vec(rows, "incoming_velocity")
    incoming_w = _vec(rows, "incoming_spin")
    p = _vec(rows, "achieved_racket_pos_env").copy()
    v, omega = _contact_batch(incoming_v, racket_v, racket_n, incoming_w, prm)
    landed = np.zeros(len(rows), dtype=bool)
    net_crossed = np.zeros(len(rows), dtype=bool)
    land_xy = np.zeros((len(rows), 2), dtype=float)
    net_z = np.zeros(len(rows), dtype=float)
    h = 0.01
    for _ in range(100):
        p_new, v_new = _rk4(p, v, omega, prm, h)
        ncross = (~net_crossed) & (~landed) & (p[:, 0] < NET_X) & (p_new[:, 0] >= NET_X)
        fn = np.clip((NET_X - p[:, 0]) / np.maximum(p_new[:, 0] - p[:, 0], 1.0e-9), 0.0, 1.0)
        z_at = p[:, 2] + (p_new[:, 2] - p[:, 2]) * fn
        net_z = np.where(ncross, z_at, net_z)
        net_crossed |= ncross
        cross = (~landed) & (p[:, 2] > TABLE_SURFACE_Z) & (p_new[:, 2] <= TABLE_SURFACE_Z)
        fz = np.clip(
            (p[:, 2] - TABLE_SURFACE_Z) / np.maximum(p[:, 2] - p_new[:, 2], 1.0e-9),
            0.0,
            1.0,
        )
        xy = p[:, :2] + (p_new[:, :2] - p[:, :2]) * fz[:, None]
        land_xy = np.where(cross[:, None], xy, land_xy)
        landed |= cross
        p, v = p_new, v_new
    net_clear = net_crossed & (net_z > NET_CLEAR_Z)
    return {
        "outgoing_velocity": v,  # post-rollout v is not used as contact output; kept for shape/debug only
        "net_crossed": net_crossed,
        "net_clear": net_clear,
        "land_valid": landed,
        "net_z": net_z,
        "landing_xy": land_xy,
    }


def _codes(outcome: dict[str, np.ndarray]) -> list[str]:
    codes = []
    for crossed, clear, valid, xy in zip(
        outcome["net_crossed"], outcome["net_clear"], outcome["land_valid"], outcome["landing_xy"]
    ):
        if not crossed:
            codes.append("NO_NET_CROSS")
        elif not clear:
            codes.append("NET_TOO_LOW")
        elif not valid:
            codes.append("NO_LANDING_WITHIN_HORIZON")
        elif xy[0] <= NET_X:
            codes.append("LAND_OWN_HALF")
        elif xy[0] > FAR_X:
            codes.append("LAND_OUT_FAR")
        elif abs(xy[1]) > TABLE_HALF_W:
            codes.append("LAND_OUT_SIDE")
        else:
            codes.append("LEGAL")
    return codes


def _counterfactual_rates(outcome: dict[str, np.ndarray], n: int) -> dict[str, Any]:
    codes = _codes(outcome)
    count = collections.Counter(codes)
    return {
        "n_captured_denominator": n,
        "net_cross_pct": _pct(sum(outcome["net_crossed"]), n),
        "net_clear_pct": _pct(sum(outcome["net_clear"]), n),
        "legal_pct": _pct(count["LEGAL"], n),
        "failure_codes": {key: int(count[key]) for key in FAILURE_ORDER if count[key]},
    }


def _failure_rate_summary(outcome: dict[str, np.ndarray], n: int) -> dict[str, float]:
    """Return the FH sweep failure buckets as percentages of captured rows."""
    codes = collections.Counter(_codes(outcome))
    return {
        "legal_pct": _pct(codes["LEGAL"], n),
        "land_out_side_pct": _pct(codes["LAND_OUT_SIDE"], n),
        "land_out_far_pct": _pct(codes["LAND_OUT_FAR"], n),
        "no_landing_pct": _pct(codes["NO_LANDING_WITHIN_HORIZON"], n),
        "net_failure_pct": _pct(codes["NO_NET_CROSS"] + codes["NET_TOO_LOW"], n),
        "no_net_cross_pct": _pct(codes["NO_NET_CROSS"], n),
        "net_too_low_pct": _pct(codes["NET_TOO_LOW"], n),
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _shift_stats(actual: dict[str, np.ndarray], cf: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    valid = mask & actual["land_valid"] & cf["land_valid"]
    delta = cf["landing_xy"][valid] - actual["landing_xy"][valid]
    return {
        "paired_landing_n": int(np.sum(valid)),
        "delta_x_m": _stats(delta[:, 0] if len(delta) else np.array([])),
        "delta_y_m": _stats(delta[:, 1] if len(delta) else np.array([])),
    }


def _fh_counterfactual(rows: list[dict[str, Any]], prm: dict[str, float]) -> dict[str, Any]:
    captured = [row for row in rows if bool(row.get("capture_gate"))]
    incoming_n = len(captured)
    actual_v = _vec(captured, "achieved_racket_velocity")
    actual_n = _vec(captured, "achieved_racket_normal")
    target_v = _vec(captured, "planner_racket_velocity")
    target_n = _vec(captured, "planner_racket_normal")
    cf0 = _rollout(captured, actual_v, actual_n, prm)
    cf1 = _rollout(captured, target_v, actual_n, prm)
    cf2 = _rollout(captured, target_v, target_n, prm)
    codes0 = np.asarray(_codes(cf0), dtype=object)
    recorded = np.asarray([str(row.get("failure_code")) for row in captured], dtype=object)
    result: dict[str, Any] = {
        "captured_rows": incoming_n,
        "recorded_rates": {
            "legal_pct": _pct(np.sum(recorded == "LEGAL"), incoming_n),
            "failure_codes": dict(collections.Counter(recorded.tolist())),
        },
        "simulated_cf0_rates": _counterfactual_rates(cf0, incoming_n),
        "CF0_actual_velocity_actual_normal": _counterfactual_rates(cf0, incoming_n),
        "CF1_target_velocity_actual_normal": _counterfactual_rates(cf1, incoming_n),
        "CF2_target_velocity_target_normal": _counterfactual_rates(cf2, incoming_n),
        "simulation_vs_recorded": {
            "legal_agreement_pct": _pct(np.sum((codes0 == "LEGAL") == (recorded == "LEGAL")), incoming_n),
            "recorded_legal_but_sim_not_legal": int(np.sum((recorded == "LEGAL") & (codes0 != "LEGAL"))),
            "recorded_not_legal_but_sim_legal": int(np.sum((recorded != "LEGAL") & (codes0 == "LEGAL"))),
        },
        "landing_shift": {},
    }
    category_masks = {
        "all_captured": np.ones(incoming_n, dtype=bool),
        "recorded_legal": recorded == "LEGAL",
        "recorded_out_side": recorded == "LAND_OUT_SIDE",
        "recorded_out_far": recorded == "LAND_OUT_FAR",
    }
    for label, mask in category_masks.items():
        result["landing_shift"][label] = {
            "CF1_minus_CF0": _shift_stats(cf0, cf1, mask),
            "CF2_minus_CF0": _shift_stats(cf0, cf2, mask),
            "CF1_outcome_counts": dict(collections.Counter(np.asarray(_codes(cf1), dtype=object)[mask].tolist())),
            "CF2_outcome_counts": dict(collections.Counter(np.asarray(_codes(cf2), dtype=object)[mask].tolist())),
        }
    return result


def _fh_horizontal_sweep(
    rows: list[dict[str, Any]],
    prm: dict[str, float],
    dvx_grid: list[float],
    dvy_grid: list[float],
) -> dict[str, Any]:
    """Screen FH target-velocity xy offsets while preserving baseline execution residuals."""
    captured = [row for row in rows if bool(row.get("capture_gate"))]
    n = len(captured)
    actual_v = _vec(captured, "achieved_racket_velocity")
    actual_n = _vec(captured, "achieved_racket_normal")
    target_v = _vec(captured, "planner_racket_velocity")
    baseline = _rollout(captured, actual_v, actual_n, prm)
    baseline_rates = _counterfactual_rates(baseline, n)
    baseline_rates["failure_rates_pct"] = _failure_rate_summary(baseline, n)

    entries: list[dict[str, Any]] = []
    for dvx in dvx_grid:
        for dvy in dvy_grid:
            delta = np.zeros_like(target_v)
            delta[:, 0] = float(dvx)
            delta[:, 1] = float(dvy)
            new_target = target_v + delta
            # Preserve each captured sample's measured execution residual:
            # actual_v_new - target_v_new == actual_v_old - target_v_old.
            new_actual = actual_v + (new_target - target_v)
            outcome = _rollout(captured, new_actual, actual_n, prm)
            rates = _counterfactual_rates(outcome, n)
            rates["failure_rates_pct"] = _failure_rate_summary(outcome, n)
            entries.append(
                {
                    "delta_vx_mps": float(dvx),
                    "delta_vy_mps": float(dvy),
                    "mode": "preserve_execution_residual",
                    "rates": rates,
                    "delta_vs_cf0_pp": {
                        "legal": rates["legal_pct"] - baseline_rates["legal_pct"],
                        "net_clear": rates["net_clear_pct"] - baseline_rates["net_clear_pct"],
                        "side_out": (
                            rates["failure_rates_pct"]["land_out_side_pct"]
                            - baseline_rates["failure_rates_pct"]["land_out_side_pct"]
                        ),
                        "far_out": (
                            rates["failure_rates_pct"]["land_out_far_pct"]
                            - baseline_rates["failure_rates_pct"]["land_out_far_pct"]
                        ),
                    },
                }
            )

    # Candidate ranking is diagnostic only. The first list excludes candidates with a material
    # (>1 pp) net-clear regression; the second keeps the raw legal-rate ordering.
    eligible = [
        entry
        for entry in entries
        if entry["delta_vs_cf0_pp"]["net_clear"] >= -1.0
    ]
    best_key = lambda entry: (
        entry["rates"]["legal_pct"],
        entry["rates"]["net_clear_pct"],
        -entry["rates"]["failure_rates_pct"]["net_failure_pct"],
    )
    eligible_sorted = sorted(eligible, key=best_key, reverse=True)
    all_sorted = sorted(entries, key=best_key, reverse=True)
    return {
        "captured_rows": n,
        "grid": {
            "dvx_mps": dvx_grid,
            "dvy_mps": dvy_grid,
            "points": len(entries),
        },
        "baseline_CF0": baseline_rates,
        "surface": entries,
        "top_candidates_net_clear_guard": eligible_sorted[:10],
        "top_candidates_unfiltered": all_sorted[:10],
        "selection_rule": "rank by legal_pct, retain candidates with net_clear no worse than CF0 by 1 pp",
        "assumptions": {
            "normal": "measured actual normal held fixed",
            "vertical_velocity": "unchanged",
            "execution": "actual_v_new = actual_v_old + target_v_new - target_v_old",
            "capture_gate": "original capture_gate held fixed; no actor rerun",
        },
    }


def _fh_conditional_benefit(
    rows: list[dict[str, Any]],
    prm: dict[str, float],
    delta_vx: float = -0.10,
    delta_vy: float = 0.0,
) -> dict[str, Any]:
    """Measure per-shot benefit/harm of one FH correction across interpretable state bins."""
    captured = [row for row in rows if bool(row.get("capture_gate"))]
    n = len(captured)
    actual_v = _vec(captured, "achieved_racket_velocity")
    actual_n = _vec(captured, "achieved_racket_normal")
    target_v = _vec(captured, "planner_racket_velocity")
    delta = np.zeros_like(target_v)
    delta[:, 0] = float(delta_vx)
    delta[:, 1] = float(delta_vy)
    baseline = _rollout(captured, actual_v, actual_n, prm)
    corrected = _rollout(captured, actual_v + delta, actual_n, prm)
    base_codes = np.asarray(_codes(baseline), dtype=object)
    corrected_codes = np.asarray(_codes(corrected), dtype=object)
    base_legal = base_codes == "LEGAL"
    corrected_legal = corrected_codes == "LEGAL"
    incoming = np.asarray([row["incoming_velocity"] for row in captured], dtype=float)
    contact_y = np.asarray([row["achieved_racket_pos_env"][1] for row in captured], dtype=float)

    def _group_summary(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0:
            return {"n": 0}

        def rate(values: np.ndarray) -> float:
            return _pct(int(np.sum(values[mask])), count)

        return {
            "n": count,
            "baseline_legal_pct": rate(base_legal),
            "corrected_legal_pct": rate(corrected_legal),
            "delta_legal_pp": rate(corrected_legal) - rate(base_legal),
            "rescued_pct": _pct(int(np.sum((~base_legal & corrected_legal)[mask])), count),
            "broken_pct": _pct(int(np.sum((base_legal & ~corrected_legal)[mask])), count),
            "baseline_far_pct": rate(base_codes == "LAND_OUT_FAR"),
            "corrected_far_pct": rate(corrected_codes == "LAND_OUT_FAR"),
            "delta_far_pp": rate(corrected_codes == "LAND_OUT_FAR") - rate(base_codes == "LAND_OUT_FAR"),
            "baseline_side_pct": rate(base_codes == "LAND_OUT_SIDE"),
            "corrected_side_pct": rate(corrected_codes == "LAND_OUT_SIDE"),
            "delta_side_pp": rate(corrected_codes == "LAND_OUT_SIDE") - rate(base_codes == "LAND_OUT_SIDE"),
            "baseline_net_failure_pct": rate((base_codes == "NO_NET_CROSS") | (base_codes == "NET_TOO_LOW")),
            "corrected_net_failure_pct": rate(
                (corrected_codes == "NO_NET_CROSS") | (corrected_codes == "NET_TOO_LOW")
            ),
            "delta_net_failure_pp": rate(
                (corrected_codes == "NO_NET_CROSS") | (corrected_codes == "NET_TOO_LOW")
            ) - rate((base_codes == "NO_NET_CROSS") | (base_codes == "NET_TOO_LOW")),
        }

    def quantile_groups(values: np.ndarray, count: int = 4) -> tuple[np.ndarray, list[float]]:
        edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1)).astype(float)
        edges = np.maximum.accumulate(edges)
        labels = np.searchsorted(edges[1:-1], values, side="right")
        return labels, [float(edge) for edge in edges]

    vx_group, vx_edges = quantile_groups(incoming[:, 0])
    vy_group, vy_edges = quantile_groups(incoming[:, 1])
    cy_group, cy_edges = quantile_groups(contact_y)

    def one_dimensional(name: str, groups: np.ndarray, edges: list[float]) -> dict[str, Any]:
        return {
            "edges": edges,
            "groups": {
                f"q{group}": _group_summary(groups == group)
                for group in range(4)
            },
        }

    vx_vy = {}
    for vx_bin in range(4):
        for vy_bin in range(4):
            vx_vy[f"vx_q{vx_bin}_vy_q{vy_bin}"] = _group_summary(
                (vx_group == vx_bin) & (vy_group == vy_bin)
            )

    return {
        "candidate": {"delta_vx_mps": float(delta_vx), "delta_vy_mps": float(delta_vy)},
        "captured_rows": n,
        "overall": _group_summary(np.ones(n, dtype=bool)),
        "by_incoming_vx": one_dimensional("incoming_vx", vx_group, vx_edges),
        "by_incoming_vy": one_dimensional("incoming_vy", vy_group, vy_edges),
        "by_contact_y": one_dimensional("contact_y", cy_group, cy_edges),
        "by_incoming_vx_x_incoming_vy": {
            "vx_edges": vx_edges,
            "vy_edges": vy_edges,
            "groups": vx_vy,
        },
        "definitions": {
            "delta_L": "I(corrected is LEGAL) - I(baseline is LEGAL)",
            "rescued": "baseline non-LEGAL and corrected LEGAL",
            "broken": "baseline LEGAL and corrected non-LEGAL",
            "execution": "actual_v_new = actual_v_old + (delta_vx, delta_vy, 0); normal held fixed",
            "binning": "quartile bins computed on captured FH core rows",
        },
    }


def _sweep_entry(
    rows: list[dict[str, Any]],
    outcome: dict[str, np.ndarray],
    low: np.ndarray,
    high: np.ndarray,
    k: float,
    v_ref: float,
    delta_max: float,
    mode: str,
) -> dict[str, Any]:
    codes = np.asarray(_codes(outcome), dtype=object)
    result = {
        "k_z": k,
        "v_ref": v_ref,
        "delta_max": delta_max,
        "mode": mode,
        "all": _counterfactual_rates(outcome, len(rows)),
        "low_incoming_vz_lt_-0.5": _counterfactual_rates({key: value[low] for key, value in outcome.items()}, int(np.sum(low))),
        "high_incoming_vz_ge_0": _counterfactual_rates({key: value[high] for key, value in outcome.items()}, int(np.sum(high))),
        "low_high_legal_gap_pp": float(
            _pct(np.sum(codes[low] == "LEGAL"), np.sum(low))
            - _pct(np.sum(codes[high] == "LEGAL"), np.sum(high))
        ),
    }
    return result


def _bh_sweep(rows: list[dict[str, Any]], prm: dict[str, float], k_grid: list[float], v_ref_grid: list[float], delta_max: float) -> dict[str, Any]:
    captured = [row for row in rows if bool(row.get("capture_gate"))]
    incoming_v = _vec(captured, "incoming_velocity")
    old_target = _vec(captured, "planner_racket_velocity")
    old_actual = _vec(captured, "achieved_racket_velocity")
    actual_n = _vec(captured, "achieved_racket_normal")
    vz = incoming_v[:, 2]
    low = vz < -0.5
    high = vz >= 0.0
    entries = []
    for mode in ("preserve_execution", "ideal_target"):
        for v_ref in v_ref_grid:
            for k in k_grid:
                delta = np.clip(k * (v_ref - vz), -delta_max, delta_max)
                new_target = old_target.copy()
                new_target[:, 2] += delta
                if mode == "preserve_execution":
                    new_racket_v = old_actual + (new_target - old_target)
                else:
                    new_racket_v = new_target
                outcome = _rollout(captured, new_racket_v, actual_n, prm)
                entries.append(_sweep_entry(captured, outcome, low, high, k, v_ref, delta_max, mode))

    baseline = {
        "captured_rows": len(captured),
        "recorded_low": {
            "net_clear_pct": _pct(sum(row.get("net_clear") for row in captured if float(row["incoming_velocity"][2]) < -0.5), np.sum(low)),
            "legal_pct": _pct(sum(row.get("failure_code") == "LEGAL" for row in captured if float(row["incoming_velocity"][2]) < -0.5), np.sum(low)),
        },
        "recorded_high": {
            "net_clear_pct": _pct(sum(row.get("net_clear") for row in captured if float(row["incoming_velocity"][2]) >= 0.0), np.sum(high)),
            "legal_pct": _pct(sum(row.get("failure_code") == "LEGAL" for row in captured if float(row["incoming_velocity"][2]) >= 0.0), np.sum(high)),
        },
        "sweep_definition": {
            "formula": "new_target_vz = old_target_vz + clip(k_z * (v_ref - incoming_vz), +/- delta_max)",
            "low_bin": "incoming_vz < -0.5",
            "high_bin": "incoming_vz >= 0",
            "normal": "measured actual normal held fixed",
            "capture_gate": "original capture_gate held fixed; no actor rerun",
        },
    }
    baseline_high_legal = baseline["recorded_high"]["legal_pct"]
    for entry in entries:
        entry["high_legal_delta_vs_recorded_pp"] = entry["high_incoming_vz_ge_0"]["legal_pct"] - baseline_high_legal
    eligible = [entry for entry in entries if entry["high_legal_delta_vs_recorded_pp"] >= -1.0]
    eligible.sort(key=lambda item: item["low_incoming_vz_lt_-0.5"]["net_clear_pct"], reverse=True)
    return {
        "baseline": baseline,
        "grid": entries,
        "best_preserve_execution": next((entry for entry in eligible if entry["mode"] == "preserve_execution"), None),
        "best_ideal_target": next((entry for entry in eligible if entry["mode"] == "ideal_target"), None),
        "selection_rule": "max low-bin net-clear among entries whose high-bin legal rate is no worse than baseline by 1 percentage point",
    }


def main() -> int:
    args = _args()
    rows = _load_rows(args.telemetry)
    prm = _params(args.venue_config)
    core = [row for row in rows if not row.get("venue_tuple_selected")]
    fh = [row for row in core if row.get("clip_id") is not None and int(row["clip_id"]) == 0]
    bh = [row for row in core if row.get("clip_id") is not None and int(row["clip_id"]) == 1]
    k_grid = [float(value) for value in args.k_grid.split(",") if value.strip()]
    v_ref_grid = [float(value) for value in args.v_ref_grid.split(",") if value.strip()]
    fh_dvx_grid = [float(value) for value in args.fh_dvx_grid.split(",") if value.strip()]
    fh_dvy_grid = [float(value) for value in args.fh_dvy_grid.split(",") if value.strip()]
    report = {
        "schema": "a5_counterfactual_v1",
        "inputs": [str(pathlib.Path(path).resolve()) for path in args.telemetry],
        "rows": {"all": len(rows), "core": len(core), "fh_core": len(fh), "bh_core": len(bh)},
        "fh_CF0_CF1_CF2": _fh_counterfactual(fh, prm),
        "fh_horizontal_velocity_sweep": _fh_horizontal_sweep(fh, prm, fh_dvx_grid, fh_dvy_grid),
        "fh_conditional_benefit": _fh_conditional_benefit(fh, prm),
        "bh_planner_vz_sweep": _bh_sweep(bh, prm, k_grid, v_ref_grid, args.delta_max),
        "notes": [
            "All counterfactuals hold the original capture_gate fixed; they isolate post-contact landing physics and do not predict a new actor trajectory.",
            "CF0/CF1/CF2 use the same incoming velocity/spin and achieved contact position for each paired sample.",
            "BH preserve_execution keeps each sample's measured velocity tracking residual while shifting the Planner target; ideal_target is an optimistic upper bound.",
            "The output is an offline sweep, not a recommendation to apply the best k_z before checking the curve.",
        ],
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": str(out.resolve()), "rows": report["rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

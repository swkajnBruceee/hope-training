#!/usr/bin/env python3
"""Analyze A5 virtual-ball failures along the racket/contact/landing chain.

This is deliberately an offline diagnostic.  It consumes the exact-strike telemetry emitted by
``scripts/evaluate.py`` and does not start Isaac, load PPO, or modify a checkpoint.  Outgoing ball
velocity/spin are reconstructed with the same venue-fitted contact equation used by
``tasks.tracking.mdp.virtual_ball``; landing and net values remain the authoritative values saved
by the evaluator.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
from typing import Any, Iterable

import numpy as np
import yaml


FH = 0
BH = 1
FH_OUT_CODES = {"LAND_OUT_SIDE", "LAND_OUT_FAR", "LAND_OUT_OTHER", "LAND_OWN_HALF"}
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
    p.add_argument("--telemetry", nargs="+", required=True, help="Telemetry JSON files to pool.")
    p.add_argument(
        "--venue-config",
        default="configs/ball_physics_venue.yaml",
        help="Venue YAML used by the virtual-ball contact model.",
    )
    p.add_argument("--out", required=True, help="Output JSON report path.")
    return p.parse_args()


def _pct(n: int | float, d: int | float) -> float:
    return 100.0 * float(n) / max(float(d), 1.0)


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return np.asarray(result, dtype=float)


def _stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    a = _finite(values)
    if not len(a):
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
    }


def _vec(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    return a if a.shape == (3,) and np.all(np.isfinite(a)) else None


def _normal(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1.0e-9)


def _contact(
    v_minus: np.ndarray,
    v_r: np.ndarray,
    normal: np.ndarray,
    omega_minus: np.ndarray,
    prm: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of virtual_ball.predict_paddle_contact."""
    radius = prm["ball_radius"]
    inertia = prm["inertia_coeff"]
    n = _normal(normal)
    if float(np.dot(v_minus - v_r, n)) > 0.0:
        n = -n
    r = -radius * n
    u = v_minus + np.cross(omega_minus, r) - v_r
    u_n = float(np.dot(u, n))
    u_t = u - u_n * n
    u_t_mag = float(np.linalg.norm(u_t))
    e = float(np.clip(prm["paddle_e_g1"] * math.exp(prm["paddle_e_g2"] * abs(u_n)), 0.05, 0.95))
    cos_theta = abs(u_n) / max(math.hypot(u_t_mag, u_n), 1.0e-9)
    raw = (prm["paddle_a_t"] + prm["paddle_b_t"] * cos_theta) * u_t_mag
    cap = prm["paddle_mu"] * (1.0 + e) * abs(u_n)
    impulse = min(max(raw, 0.0), cap)
    delta_t = -impulse * u_t / max(u_t_mag, 1.0e-9) if u_t_mag > 1.0e-9 else np.zeros(3)
    delta_n = -(1.0 + e) * u_n * n
    delta_omega = -(1.0 / (inertia * radius)) * np.cross(n, delta_t)
    return v_minus + delta_n + delta_t, omega_minus + delta_omega


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for index, row in enumerate(payload.get("rows", [])):
            item = dict(row)
            item["_source_file"] = str(pathlib.Path(path).resolve())
            item["_source_row"] = index
            rows.append(item)
    return rows


def _derive(rows: list[dict[str, Any]], prm: dict[str, float]) -> None:
    for row in rows:
        target_v = _vec(row, "planner_racket_velocity")
        actual_v = _vec(row, "achieved_racket_velocity")
        target_n = _vec(row, "planner_racket_normal")
        actual_n = _vec(row, "achieved_racket_normal")
        target_p = _vec(row, "planner_racket_pos_env")
        actual_p = _vec(row, "achieved_racket_pos_env")
        incoming_v = _vec(row, "incoming_velocity")
        incoming_w = _vec(row, "incoming_spin")
        if target_v is not None and actual_v is not None:
            ev = actual_v - target_v
            row["e_v_xyz"] = ev.tolist()
            row["e_v_norm"] = float(np.linalg.norm(ev))
            row["e_v_x"] = float(ev[0])
            row["e_v_y"] = float(ev[1])
            row["e_v_z"] = float(ev[2])
        if target_n is not None and actual_n is not None:
            row["e_n_deg"] = float(
                math.degrees(math.acos(float(np.clip(np.dot(_normal(target_n), _normal(actual_n)), -1.0, 1.0))))
            )
        if target_p is not None and actual_p is not None:
            row["e_pos_norm"] = float(np.linalg.norm(actual_p - target_p))
        if incoming_v is not None:
            row["incoming_vz"] = float(incoming_v[2])
            row["incoming_vy"] = float(incoming_v[1])
        if target_v is not None:
            row["target_vz"] = float(target_v[2])
        if actual_v is not None:
            row["actual_vz"] = float(actual_v[2])
        if target_n is not None:
            row["target_nz"] = float(_normal(target_n)[2])
        if actual_n is not None:
            row["actual_nz"] = float(_normal(actual_n)[2])
        if incoming_v is not None and incoming_w is not None and actual_v is not None and actual_n is not None:
            outgoing_v, outgoing_w = _contact(incoming_v, actual_v, actual_n, incoming_w, prm)
            row["outgoing_velocity"] = outgoing_v.tolist()
            row["outgoing_spin"] = outgoing_w.tolist()
            row["outgoing_vz"] = float(outgoing_v[2])
            row["outgoing_spin_norm"] = float(np.linalg.norm(outgoing_w))


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = collections.Counter(str(row.get("failure_code", "UNKNOWN")) for row in rows)
    return {key: int(counts[key]) for key in FAILURE_ORDER if counts[key]} | {
        key: int(value) for key, value in sorted(counts.items()) if key not in FAILURE_ORDER
    }


def _chain_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "failure_codes": _failure_counts(rows),
        "e_pos_norm_m": _stats(row.get("e_pos_norm") for row in rows),
        "e_v_norm_mps": _stats(row.get("e_v_norm") for row in rows),
        "e_v_x_mps": _stats(row.get("e_v_x") for row in rows),
        "e_v_y_mps": _stats(row.get("e_v_y") for row in rows),
        "e_v_z_mps": _stats(row.get("e_v_z") for row in rows),
        "e_n_deg": _stats(row.get("e_n_deg") for row in rows),
        "incoming_vz_mps": _stats(row.get("incoming_vz") for row in rows),
        "target_vz_mps": _stats(row.get("target_vz") for row in rows),
        "actual_vz_mps": _stats(row.get("actual_vz") for row in rows),
        "target_nz": _stats(row.get("target_nz") for row in rows),
        "actual_nz": _stats(row.get("actual_nz") for row in rows),
        "outgoing_vz_mps": _stats(row.get("outgoing_vz") for row in rows),
        "net_z_env_m": _stats(row.get("net_z_env") for row in rows if row.get("net_crossed")),
        "landing_x_env_m": _stats((row.get("landing_xy_env") or [None, None])[0] for row in rows if row.get("landing_valid")),
        "landing_y_env_m": _stats((row.get("landing_xy_env") or [None, None])[1] for row in rows if row.get("landing_valid")),
    }


def _bin_label(lo: float, hi: float, last: bool = False) -> str:
    return f"[{lo:g},{hi:g}{']' if last else ')'}"


def _rate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    hits = sum(bool(row.get("capture_gate")) for row in rows)
    net_cross = sum(bool(row.get("net_crossed")) for row in rows)
    net_clear = sum(bool(row.get("net_clear")) for row in rows)
    legal = sum(row.get("failure_code") == "LEGAL" for row in rows)
    return {
        "n": total,
        "capture_rate_pct": _pct(hits, total),
        "net_cross_rate_pct_attempt": _pct(net_cross, total),
        "net_clear_rate_pct_attempt": _pct(net_clear, total),
        "net_clear_per_hit_pct": _pct(net_clear, hits),
        "legal_rate_pct_attempt": _pct(legal, total),
        "legal_per_net_clear_pct": _pct(legal, net_clear),
        "failure_codes": _failure_counts(rows),
    }


def _fh_error_bins(rows: list[dict[str, Any]], field: str, edges: list[float]) -> dict[str, Any]:
    result = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            row for row in rows
            if row.get(field) is not None and (lo <= float(row[field]) < hi or (i == len(edges) - 2 and float(row[field]) == hi))
        ]
        out = sum(row.get("failure_code") in FH_OUT_CODES for row in selected)
        legal = sum(row.get("failure_code") == "LEGAL" for row in selected)
        result[_bin_label(lo, hi, i == len(edges) - 2)] = {
            "n": len(selected),
            "out_rate_pct": _pct(out, len(selected)),
            "legal_rate_pct": _pct(legal, len(selected)),
            "failure_codes": _failure_counts(selected),
        }
    return result


def _fh_landing_error_bins(rows: list[dict[str, Any]], field: str, edges: list[float]) -> dict[str, Any]:
    """FH landing attribution after the ball has crossed and landed.

    This removes net/no-landing failures from P(out | tracking error), so the result specifically
    tests the horizontal landing-control branch requested by the diagnosis.
    """
    return _fh_error_bins(
        [row for row in rows if row.get("net_clear") and row.get("landing_valid")], field, edges
    )


def _bh_vz_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edges = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.5]
    result = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            row for row in rows
            if row.get("incoming_vz") is not None
            and (lo <= float(row["incoming_vz"]) < hi or (i == len(edges) - 2 and float(row["incoming_vz"]) == hi))
        ]
        result[_bin_label(lo, hi, i == len(edges) - 2)] = {
            **_rate_rows(selected),
            "target_racket_vz_mps": _stats(row.get("target_vz") for row in selected),
            "actual_racket_vz_mps": _stats(row.get("actual_vz") for row in selected),
            "target_normal_z": _stats(row.get("target_nz") for row in selected),
            "actual_normal_z": _stats(row.get("actual_nz") for row in selected),
            "outgoing_vz_mps": _stats(row.get("outgoing_vz") for row in selected),
            "net_z_env_m": _stats(row.get("net_z_env") for row in selected if row.get("net_crossed")),
        }
    return result


def _goal_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def q(values: Any, step: float) -> tuple[int, ...] | None:
        if values is None:
            return None
        return tuple(round(float(v) / step) for v in values)

    return (
        q(row.get("planner_racket_pos_env"), 0.05),
        q(row.get("planner_racket_velocity"), 0.10),
        round(float(row.get("time_to_strike") or 0.0) / 0.02),
        int(row.get("clip_id") or 0),
        int(round(float(row.get("swing_sign") or 0.0))),
    )


def _aliasing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[_goal_key(row)].append(row)
    candidates = []
    for key, group in groups.items():
        if len(group) < 4:
            continue
        vv = np.asarray([row.get("incoming_velocity") for row in group if row.get("incoming_velocity") is not None], dtype=float)
        ww = np.asarray([row.get("incoming_spin") for row in group if row.get("incoming_spin") is not None], dtype=float)
        if len(vv) < 4 or len(ww) < 4:
            continue
        v_span = np.ptp(vv, axis=0)
        w_span = np.ptp(ww, axis=0)
        v_norm_span = float(np.ptp(np.linalg.norm(vv, axis=1)))
        w_norm_span = float(np.ptp(np.linalg.norm(ww, axis=1)))
        # Match the coarse probe in audit_virtual_protocol.py: either a material
        # velocity span or a material spin span is enough to nominate a group.
        if float(np.max(v_span)) < 0.5 and float(np.max(w_span)) < 10.0:
            continue
        legal = sum(row.get("failure_code") == "LEGAL" for row in group)
        candidates.append({
            "goal_key": [list(x) if isinstance(x, tuple) else x for x in key],
            "n": len(group),
            "legal_rate_pct": _pct(legal, len(group)),
            "incoming_velocity_span": v_span.tolist(),
            "incoming_velocity_norm_span_mps": v_norm_span,
            "incoming_spin_span": w_span.tolist(),
            "incoming_spin_norm_span": w_norm_span,
            "failure_codes": _failure_counts(group),
            "examples": [
                {
                    "source_file": row["_source_file"],
                    "source_row": row["_source_row"],
                    "incoming_velocity": row.get("incoming_velocity"),
                    "incoming_spin": row.get("incoming_spin"),
                    "failure_code": row.get("failure_code"),
                }
                for row in group[:6]
            ],
        })
    candidates.sort(key=lambda item: (item["legal_rate_pct"], -item["n"]))
    rates = [item["legal_rate_pct"] for item in candidates]
    return {
        "quantization": {"position_m": 0.05, "velocity_mps": 0.10, "time_s": 0.02},
        "candidate_groups": len(candidates),
        "candidate_rows": int(sum(item["n"] for item in candidates)),
        "candidate_legal_rate_pct_min": float(min(rates)) if rates else None,
        "candidate_legal_rate_pct_median": float(np.median(rates)) if rates else None,
        "candidate_legal_rate_pct_max": float(max(rates)) if rates else None,
        "low_legal_examples": candidates[:10],
        "interpretation": "Candidate aliasing only; same quantized planner goal does not prove identical observation or causal insufficiency.",
    }


def main() -> int:
    args = _args()
    with open(args.venue_config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    pad = raw["contact"]["paddle"]
    prm = {
        "ball_radius": float(raw["ball"]["radius"]),
        "inertia_coeff": float(raw["ball"]["inertia_coeff"]),
        "paddle_a_t": float(pad["a_t"]),
        "paddle_b_t": float(pad["b_t"]),
        "paddle_mu": float(pad["mu_safety"]),
        "paddle_e_g1": float(pad["e_exp_g1"]),
        "paddle_e_g2": float(pad["e_exp_g2"]),
    }
    rows = _load_rows(args.telemetry)
    _derive(rows, prm)
    core = [row for row in rows if not row.get("venue_tuple_selected")]
    fh = [row for row in core if (int(row["clip_id"]) if row.get("clip_id") is not None else -1) == FH]
    bh = [row for row in core if (int(row["clip_id"]) if row.get("clip_id") is not None else -1) == BH]
    fh_out = [row for row in fh if row.get("failure_code") in FH_OUT_CODES]

    by_failure = {
        code: _chain_group([row for row in core if row.get("failure_code") == code])
        for code in FAILURE_ORDER
        if any(row.get("failure_code") == code for row in core)
    }
    report = {
        "schema": "a5_failure_chain_v1",
        "inputs": [str(pathlib.Path(path).resolve()) for path in args.telemetry],
        "rows": {"all": len(rows), "core": len(core), "fh_core": len(fh), "bh_core": len(bh)},
        "contact_model": {"venue_config": str(pathlib.Path(args.venue_config).resolve()), "parameters": prm},
        "overall_core": _rate_rows(core),
        "by_failure_core": by_failure,
        "forehand_core": {
            "rates": _rate_rows(fh),
            "outcome_chain": {
                "legal": _chain_group([row for row in fh if row.get("failure_code") == "LEGAL"]),
                "out_side_or_far": _chain_group(fh_out),
                "no_landing": _chain_group([row for row in fh if row.get("failure_code") == "NO_LANDING_WITHIN_HORIZON"]),
            },
            "outcome_by_e_v_norm_mps": _fh_error_bins(fh, "e_v_norm", [0.0, 0.10, 0.25, 0.50, 1.0, 10.0]),
            "outcome_by_e_v_y_mps": _fh_error_bins(fh, "e_v_y", [-1.0, 0.0, 0.10, 0.20, 0.30, 0.50, 1.0]),
            "outcome_by_e_n_deg": _fh_error_bins(fh, "e_n_deg", [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 180.0]),
            "landing_only_outcome_by_e_v_norm_mps": _fh_landing_error_bins(
                fh, "e_v_norm", [0.0, 0.10, 0.25, 0.50, 1.0, 10.0]
            ),
            "landing_only_outcome_by_e_n_deg": _fh_landing_error_bins(
                fh, "e_n_deg", [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 180.0]
            ),
        },
        "backhand_core": {
            "rates": _rate_rows(bh),
            "incoming_vz_bins": _bh_vz_bins(bh),
        },
        "planner_goal_aliasing": {
            "all_rows": _aliasing(rows),
            "core_rows": _aliasing(core),
        },
        "notes": [
            "outgoing_velocity/outgoing_spin are reconstructed from the same venue-fitted contact equation; telemetry landing_xy/net_z are retained as evaluator outputs.",
            "FH out-of-bounds means LAND_OUT_SIDE/FAR/OTHER/OWN_HALF; exact failure codes remain available in each group.",
            "Core means venue_tuple_selected=false, so tuple samples do not dilute the failure diagnosis.",
        ],
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "out": str(out.resolve()),
                "rows": report["rows"],
                "alias_groups_all": report["planner_goal_aliasing"]["all_rows"]["candidate_groups"],
                "alias_groups_core": report["planner_goal_aliasing"]["core_rows"]["candidate_groups"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

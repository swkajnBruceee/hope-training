#!/usr/bin/env python3
"""Audit virtual-ball protocol data without starting Isaac or PPO.

The auditor deliberately keeps the virtual-ball coordinates in the env-local frame used by
RacketTargetCommand: the near table edge is ``vb_table_near_x`` (normally 0.5 m), so the
opponent half is ``(near_x + 1.37, near_x + 2.74]``.

Examples:
    python scripts/audit_virtual_protocol.py --bank assets/venue_tuple/v17_r12_physical_tuple_bank.npz
    python scripts/audit_virtual_protocol.py --telemetry logs/.../virtual_telemetry.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
from typing import Any


TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_X = TABLE_LENGTH / 2.0


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", default=None, help="Optional venue tuple .npz to audit.")
    p.add_argument(
        "--telemetry",
        nargs="+",
        default=None,
        help="One or more --virtual-telemetry-out JSON files; multiple files are pooled.",
    )
    p.add_argument("--near-x", type=float, default=0.5, help="Env-local near table edge.")
    p.add_argument("--out", default=None, help="Optional JSON report path.")
    return p.parse_args()


def _pct(n: int | float, d: int | float) -> float:
    return 100.0 * float(n) / max(float(d), 1.0)


def _range_stats(values):
    import numpy as np

    values = np.asarray(values)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def audit_bank(path: str, near_x: float) -> dict[str, Any]:
    import numpy as np

    data = np.load(path)
    intended = np.asarray(data["intended_landing_xy"])
    predicted = np.asarray(data["predicted_landing_xy"])
    clip = np.asarray(data["clip"])
    net_x = near_x + NET_X
    far_x = near_x + TABLE_LENGTH
    half_w = TABLE_WIDTH / 2.0

    def masks(xy):
        return {
            "legal_opponent": (xy[:, 0] > net_x) & (xy[:, 0] <= far_x) & (np.abs(xy[:, 1]) <= half_w),
            "out_far": xy[:, 0] > far_x,
            "own_half_or_before_net": xy[:, 0] <= net_x,
            "out_side": np.abs(xy[:, 1]) > half_w,
        }

    result: dict[str, Any] = {
        "path": str(pathlib.Path(path).resolve()),
        "coordinate_contract": {
            "near_x": near_x,
            "net_x": net_x,
            "far_x": far_x,
            "half_width": half_w,
        },
        "rows": int(len(intended)),
        "intended_x": _range_stats(intended[:, 0]),
        "intended_y": _range_stats(intended[:, 1]),
        "predicted_x": _range_stats(predicted[:, 0]),
        "predicted_y": _range_stats(predicted[:, 1]),
        "intended_predicted_abs_error_mean": np.mean(np.abs(predicted - intended), axis=0).tolist(),
        "intended_predicted_abs_error_p95": np.quantile(np.abs(predicted - intended), 0.95, axis=0).tolist(),
    }
    for label, xy in (("intended", intended), ("predicted", predicted)):
        mm = masks(xy)
        result[label] = {
            "legal_opponent_count": int(mm["legal_opponent"].sum()),
            "legal_opponent_rate": _pct(mm["legal_opponent"].sum(), len(xy)),
            "out_far_count": int(mm["out_far"].sum()),
            "out_far_rate": _pct(mm["out_far"].sum(), len(xy)),
            "own_half_or_before_net_count": int(mm["own_half_or_before_net"].sum()),
            "out_side_count": int(mm["out_side"].sum()),
        }
        by_side = {}
        for side, name in ((0, "forehand"), (1, "backhand")):
            selected = clip == side
            by_side[name] = {
                "rows": int(selected.sum()),
                "legal_opponent_rate": _pct((mm["legal_opponent"] & selected).sum(), selected.sum()),
                "out_far_rate": _pct((mm["out_far"] & selected).sum(), selected.sum()),
            }
        result[label]["by_side"] = by_side
    return result


def _goal_key(row: dict[str, Any]):
    def q(values, step):
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


def audit_telemetry(path: str | list[str]) -> dict[str, Any]:
    if isinstance(path, (list, tuple)):
        paths = [str(item) for item in path]
        payloads = []
        for item in paths:
            with open(item, "r", encoding="utf-8") as f:
                payloads.append(json.load(f))
        payload = dict(payloads[0])
        payload["rows"] = [row for item in payloads for row in item.get("rows", [])]
        path_label = ",".join(str(pathlib.Path(item).resolve()) for item in paths)
    else:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        path_label = str(pathlib.Path(path).resolve())
    rows = list(payload.get("rows", []))
    codes = collections.Counter(str(row.get("failure_code", "UNKNOWN")) for row in rows)
    by_clip: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_tuple: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_clip_tuple: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        clip = str(row.get("clip_id", "unknown"))
        tuple_key = "tuple" if row.get("venue_tuple_selected") else "core"
        code = str(row.get("failure_code", "UNKNOWN"))
        by_clip[clip][code] += 1
        by_tuple[tuple_key][code] += 1
        by_clip_tuple[f"clip_{clip}_{tuple_key}"][code] += 1

    def group_rates(counter: collections.Counter[str]) -> dict[str, float | int]:
        total = sum(counter.values())
        hits = total - counter.get("MISS_CAPTURE", 0)
        net_crossed = total - counter.get("MISS_CAPTURE", 0) - counter.get("NO_NET_CROSS", 0)
        net_clear = net_crossed - counter.get("NET_TOO_LOW", 0)
        legal = counter.get("LEGAL", 0)
        return {
            "attempts": total,
            "hits": hits,
            "net_clear": net_clear,
            "legal": legal,
            "hit_per_attempt_pct": _pct(hits, total),
            "net_clear_per_hit_pct": _pct(net_clear, hits),
            "legal_per_attempt_pct": _pct(legal, total),
            "legal_per_net_clear_pct": _pct(legal, net_clear),
        }

    def counter_dict(value):
        return dict(sorted(value.items()))

    def binned_rates(field: str, index: int, edges: list[float]):
        buckets: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for row in rows:
            if row.get("venue_tuple_selected"):
                continue
            values = row.get(field)
            if values is None or len(values) <= index:
                continue
            value = float(values[index])
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo <= value < hi or (hi == edges[-1] and value == hi):
                    key = f"[{lo:g},{hi:g}{']' if hi == edges[-1] else ')'}"
                    buckets[key][str(row.get("failure_code", "UNKNOWN"))] += 1
                    break
        return {
            key: {"rates": group_rates(value), "failure_codes": counter_dict(value)}
            for key, value in sorted(buckets.items())
        }

    attempts = len(rows)
    hits = sum(codes[k] for k in codes if k != "MISS_CAPTURE")
    net_crossed = sum(codes[k] for k in codes if k not in {"MISS_CAPTURE", "NO_NET_CROSS"})
    net_clear = sum(
        codes[k]
        for k in codes
        if k not in {"MISS_CAPTURE", "NO_NET_CROSS", "NET_TOO_LOW"}
    )
    legal = codes.get("LEGAL", 0)

    # A coarse aliasing probe: group by the actor-visible planner goal and look for groups whose
    # hidden incoming velocity/spin varies materially. This is a diagnostic candidate count, not
    # a causal proof; the follow-up should compare outcomes within these groups.
    groups: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[_goal_key(row)].append(row)
    alias_groups = 0
    alias_rows = 0
    alias_success_rates = []
    for group in groups.values():
        incoming_v = [r.get("incoming_velocity") for r in group if r.get("incoming_velocity") is not None]
        incoming_w = [r.get("incoming_spin") for r in group if r.get("incoming_spin") is not None]
        if len(group) < 4 or not incoming_v or not incoming_w:
            continue
        v_span = max(max(v[i] for v in incoming_v) - min(v[i] for v in incoming_v) for i in range(3))
        w_span = max(max(v[i] for v in incoming_w) - min(v[i] for v in incoming_w) for i in range(3))
        if v_span < 0.5 and w_span < 10.0:
            continue
        alias_groups += 1
        alias_rows += len(group)
        alias_success_rates.append(sum(r.get("failure_code") == "LEGAL" for r in group) / len(group))

    spin_buckets: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    spin_edges = [0.0, 20.0, 40.0, 60.0, 100.0]
    for row in rows:
        if row.get("venue_tuple_selected") or row.get("incoming_spin") is None:
            continue
        norm = math.sqrt(sum(float(v) * float(v) for v in row["incoming_spin"]))
        for lo, hi in zip(spin_edges[:-1], spin_edges[1:]):
            if lo <= norm < hi or (hi == spin_edges[-1] and norm <= hi):
                key = f"[{lo:g},{hi:g}{']' if hi == spin_edges[-1] else ')'}"
                spin_buckets[key][str(row.get("failure_code", "UNKNOWN"))] += 1
                break

    return {
        "schema_version": 1,
        "path": path_label,
        "source_metadata": {k: payload.get(k) for k in ("checkpoint", "seed", "num_envs", "num_steps")},
        "table_frame": payload.get("table_frame"),
        "counts": {
            "attempts": attempts,
            "hits": hits,
            "net_crossed": net_crossed,
            "net_clear": net_clear,
            "legal": legal,
        },
        "rates": {
            "hit_per_attempt": _pct(hits, attempts),
            "net_cross_per_hit": _pct(net_crossed, hits),
            "net_clear_per_hit": _pct(net_clear, hits),
            "legal_per_attempt": _pct(legal, attempts),
            "legal_per_net_clear": _pct(legal, net_clear),
        },
        "failure_code_counts": counter_dict(codes),
        "failure_code_by_clip": {k: counter_dict(v) for k, v in sorted(by_clip.items())},
        "rates_by_clip": {k: group_rates(v) for k, v in sorted(by_clip.items())},
        "failure_code_by_tuple_status": {k: counter_dict(v) for k, v in sorted(by_tuple.items())},
        "rates_by_tuple_status": {k: group_rates(v) for k, v in sorted(by_tuple.items())},
        "failure_code_by_clip_and_tuple_status": {
            k: counter_dict(v) for k, v in sorted(by_clip_tuple.items())
        },
        "rates_by_clip_and_tuple_status": {
            k: group_rates(v) for k, v in sorted(by_clip_tuple.items())
        },
        "goal_aliasing_probe": {
            "quantization": {"planner_pos_m": 0.05, "planner_velocity_mps": 0.10, "time_s": 0.02},
            "candidate_groups": alias_groups,
            "candidate_rows": alias_rows,
            "candidate_legal_rate_mean": (
                100.0 * sum(alias_success_rates) / len(alias_success_rates) if alias_success_rates else None
            ),
        },
        "core_failure_slices": {
            "incoming_velocity_z": binned_rates("incoming_velocity", 2, [-1.0, -0.5, 0.0, 0.5]),
            "incoming_velocity_y": binned_rates("incoming_velocity", 1, [-0.6, -0.2, 0.2, 0.6]),
            "incoming_spin_norm": {
                key: {"rates": group_rates(value), "failure_codes": counter_dict(value)}
                for key, value in sorted(spin_buckets.items())
            },
        },
    }


def main() -> int:
    args = _args()
    report: dict[str, Any] = {"schema_version": 1}
    if args.bank:
        report["bank"] = audit_bank(args.bank, args.near_x)
    if args.telemetry:
        report["telemetry"] = audit_telemetry(args.telemetry)
    if not report.keys() - {"schema_version"}:
        raise SystemExit("provide --bank and/or --telemetry")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

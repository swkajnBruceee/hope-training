#!/usr/bin/env python3
"""Label landing and success for competition-table-frame packed samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
TABLE_Z_M = 0.0
NET_X_M = TABLE_LENGTH_M / 2.0
Y_MIN_M = -TABLE_WIDTH_M
Y_MAX_M = 0.0


def _as_json_list(values: np.ndarray) -> list[dict[str, Any]]:
    return [json.loads(str(v)) for v in values]


def _detect_landing(ball_pos: np.ndarray, ball_vel: np.ndarray, hit_index: int, z_tolerance_m: float) -> tuple[int | None, np.ndarray]:
    start = max(1, int(hit_index) + 1)
    for i in range(start, len(ball_pos)):
        if not (np.isfinite(ball_pos[i]).all() and np.isfinite(ball_vel[i]).all() and np.isfinite(ball_vel[i - 1]).all()):
            continue
        near_table = abs(float(ball_pos[i, 2]) - TABLE_Z_M) <= z_tolerance_m
        vertical_bounce = ball_vel[i - 1, 2] < 0.0 and ball_vel[i, 2] > 0.0
        if near_table and vertical_bounce:
            return i, ball_pos[i].copy()
    return None, np.full(3, np.nan)


def _expected_opponent_half(racket: str) -> tuple[float, float, str]:
    if racket == "liang01":
        return NET_X_M, TABLE_LENGTH_M, "p2_half"
    if racket == "gao01":
        return 0.0, NET_X_M, "p1_half"
    return float("nan"), float("nan"), "unknown"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Competition Success Label Report",
        "",
        f"Input: `{report['input_dataset']}`",
        f"Output: `{report['output_dataset']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {report['samples']} |",
        f"| Landing detected | {report['landing_detected']} |",
        f"| Reliable success labels | {report['success_label_reliable']} |",
        f"| Success | {report['success_counts'].get('1', 0)} |",
        f"| Failure | {report['success_counts'].get('0', 0)} |",
        f"| Unknown | {report['success_counts'].get('-1', 0)} |",
        "",
        "## Rule",
        "",
        "- Landing is the first post-hit frame near table height with vertical velocity changing from downward to upward.",
        "- `liang01` success requires landing in P2 half: `1.37 <= x <= 2.74` and `-1.525 <= y <= 0`.",
        "- `gao01` success requires landing in P1 half: `0 <= x <= 1.37` and `-1.525 <= y <= 0`.",
        "- Samples without a detected landing remain `success=-1` instead of being forced to failure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--z-tolerance-m", type=float, default=0.05)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    attrs = json.loads(str(arrays["dataset_attrs_json"]))
    if attrs.get("coordinate_frame") != "competition_table_m":
        raise ValueError(f"expected competition_table_m, got {attrs.get('coordinate_frame')!r}")

    n = int(arrays["ball_pos"].shape[0])
    hit_index = arrays["hit_index"].astype(int)
    sources = _as_json_list(arrays["source_json"])
    quality = _as_json_list(arrays["quality_flags_json"])

    success = np.full(n, -1, dtype=np.int8)
    landing_pos = np.full((n, 3), np.nan, dtype=np.float64)
    landing_index = np.full(n, -1, dtype=np.int64)
    landing_side = np.full(n, "unknown", dtype="<U16")
    success_reason = np.full(n, "landing_not_detected_in_episode", dtype="<U96")

    for i in range(n):
        idx, pos = _detect_landing(arrays["ball_pos"][i], arrays["ball_vel"][i], int(hit_index[i]), args.z_tolerance_m)
        racket = str(sources[i].get("racket", "unknown"))
        q = dict(quality[i])
        q["table_config_available"] = True
        q["table_z_m"] = TABLE_Z_M
        q["table_y_min_m"] = Y_MIN_M
        q["table_y_max_m"] = Y_MAX_M
        q["table_net_x_m"] = NET_X_M
        q["landing_detected"] = idx is not None
        q["success_label_reliable"] = idx is not None

        if idx is None:
            q["success_label_reason"] = "landing_not_detected_in_episode"
            quality[i] = q
            continue

        landing_index[i] = int(idx)
        landing_pos[i] = pos
        x_min, x_max, expected_half = _expected_opponent_half(racket)
        landing_side[i] = "p1_half" if pos[0] < NET_X_M else "p2_half"
        in_expected_x = np.isfinite(x_min) and x_min <= float(pos[0]) <= x_max
        in_table_y = Y_MIN_M <= float(pos[1]) <= Y_MAX_M
        is_success = bool(in_expected_x and in_table_y)
        success[i] = int(is_success)
        success_reason[i] = "opponent_table_landing_rule" if is_success else "landing_outside_expected_opponent_half"

        q.update(
            {
                "landing_index": int(idx),
                "landing_time_rel_s": float(arrays["time_rel"][i, idx]),
                "landing_side": str(landing_side[i]),
                "expected_landing_side": expected_half,
                "landing_in_table_y": bool(in_table_y),
                "landing_in_expected_x": bool(in_expected_x),
                "success_label_reason": str(success_reason[i]),
            }
        )
        quality[i] = q

    arrays["success"] = success
    arrays["landing_pos"] = landing_pos
    arrays["landing_index"] = landing_index
    arrays["landing_side"] = landing_side
    arrays["success_label_reason"] = success_reason
    arrays["quality_flags_json"] = np.asarray([json.dumps(q, ensure_ascii=False) for q in quality])
    attrs["success_labels"] = {
        "frame": "competition_table_m",
        "table_length_m": TABLE_LENGTH_M,
        "table_width_m": TABLE_WIDTH_M,
        "table_z_m": TABLE_Z_M,
        "z_tolerance_m": float(args.z_tolerance_m),
        "unknown_encoding": -1,
    }
    arrays["dataset_attrs_json"] = np.asarray(json.dumps(attrs, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    report = {
        "input_dataset": str(args.input),
        "output_dataset": str(args.output),
        "samples": n,
        "z_tolerance_m": float(args.z_tolerance_m),
        "landing_detected": int(np.sum(landing_index >= 0)),
        "success_label_reliable": int(np.sum(landing_index >= 0)),
        "success_counts": dict(Counter(str(int(x)) for x in success)),
        "landing_side_counts": dict(Counter(str(x) for x in landing_side)),
        "success_reason_counts": dict(Counter(str(x) for x in success_reason)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.report.with_suffix(".md"))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.report.with_suffix('.md')}")


if __name__ == "__main__":
    main()

"""Associate natural post-FH state with the next-shot outcome.

This deliberately does not alter simulator state.  It is an on-manifold association
audit for selecting recovery variables before designing a PPO objective; it is not a
causal intervention.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


OFFSETS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def tilt_angle(quat) -> float:
    # Quaternion is [w, x, y, z].  The body-up z component is 1 - 2(x^2+y^2).
    w, x, y, z = (float(value) for value in quat)
    del w, z
    up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.acos(up_z)


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": sorted(values)[max(0, math.ceil(0.90 * len(values)) - 1)],
    }


def main() -> int:
    args = parse_args()
    telemetry = json.loads(Path(args.telemetry).read_text(encoding="utf-8"))
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    rows = telemetry.get("rows", [])
    post_rows = telemetry.get("post_strike_state_rows", [])

    first_mismatch = {}
    for mismatch in result.get("paired_recipe_mismatches", []):
        if "env_id" in mismatch and "event_index" in mismatch:
            first_mismatch.setdefault(
                int(mismatch["env_id"]), int(mismatch["event_index"])
            )

    outcome_rows = {
        (int(row["env_id"]), int(row["paired_recipe_index"])): row
        for row in rows
        if row.get("paired_recipe_index") is not None
    }
    post = {
        (
            int(row["env_id"]),
            int(row["source_paired_recipe_index"]),
            round(float(row["offset_s"]), 6),
        ): row
        for row in post_rows
        if "source_paired_recipe_index" in row and "offset_s" in row
    }
    resets = defaultdict(list)
    for event in telemetry.get("reset_events", []):
        resets[(int(event["env_id"]), int(event.get("paired_recipe_index", -1)))].append(event)

    samples = []
    excluded = Counter()
    for row in rows:
        if not row.get("fh_correction_applied", False):
            continue
        env_id = int(row["env_id"])
        source_index = int(row["paired_recipe_index"])
        next_index = source_index + 1
        if env_id in first_mismatch and next_index >= first_mismatch[env_id]:
            excluded["after_first_recipe_mismatch"] += 1
            continue
        next_row = outcome_rows.get((env_id, next_index))
        if next_row is None:
            excluded["missing_next_outcome"] += 1
            continue
        reset_before = [
            event
            for event in resets.get((env_id, next_index), [])
            if int(event.get("global_step", 10**18)) <= int(next_row.get("global_step", 10**18))
        ]
        if reset_before:
            outcome = "RESET_BEFORE_NEXT_SHOT"
            reasons = sorted(
                {
                    reason
                    for event in reset_before
                    for reason in event.get("termination_reasons", [])
                }
            )
        else:
            outcome = next_row.get("failure_code", "UNKNOWN")
            reasons = []
        samples.append(
            {
                "env_id": env_id,
                "source_recipe_index": source_index,
                "next_recipe_index": next_index,
                "outcome": outcome,
                "reset_reasons": reasons,
            }
        )

    by_offset = {}
    feature_names = (
        "root_lin_speed",
        "root_ang_speed",
        "joint_speed",
        "racket_speed",
        "root_tilt_rad",
        "policy_action_norm",
    )
    for offset in OFFSETS:
        grouped = defaultdict(lambda: defaultdict(list))
        group_counts = Counter()
        usable = 0
        for sample in samples:
            key = (
                sample["env_id"],
                sample["source_recipe_index"],
                round(float(offset), 6),
            )
            state = post.get(key)
            if state is None:
                continue
            features = {
                "root_lin_speed": norm(state["robot_root_lin_vel_w"]),
                "root_ang_speed": norm(state["robot_root_ang_vel_w"]),
                "joint_speed": norm(state["robot_joint_vel"]),
                "racket_speed": norm(state["racket_velocity"]),
                "root_tilt_rad": tilt_angle(state["robot_root_quat_w"]),
                "policy_action_norm": float(state["policy_action_norm"])
                if state.get("policy_action_norm") is not None
                else None,
            }
            outcome_group = "LEGAL" if sample["outcome"] == "LEGAL" else "FAILURE"
            if sample["outcome"] == "RESET_BEFORE_NEXT_SHOT":
                outcome_group = "RESET_BEFORE_NEXT_SHOT"
            group_counts[outcome_group] += 1
            for name, value in features.items():
                if value is None:
                    continue
                grouped[outcome_group][name].append(value)
            usable += 1
        by_offset[str(offset)] = {
            "n": usable,
            "outcome_counts": dict(group_counts),
            "features": {
                group: {
                    name: summary(values)
                    for name, values in feature_map.items()
                }
                for group, feature_map in grouped.items()
            },
        }

    report = {
        "schema_version": 1,
        "telemetry": str(Path(args.telemetry)),
        "result": str(Path(args.result)),
        "sample_selection": {
            "fh_conditioned_samples": len(samples),
            "excluded": dict(excluded),
            "strict_prefix_only": True,
            "association_not_causal": True,
        },
        "outcome_counts": dict(Counter(sample["outcome"] for sample in samples)),
        "reset_reason_counts": dict(
            Counter(
                reason
                for sample in samples
                for reason in sample["reset_reasons"]
            )
        ),
        "by_offset": by_offset,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

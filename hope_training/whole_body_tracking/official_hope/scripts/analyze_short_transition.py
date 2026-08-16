"""Summarize strike intervals and FH/BH transition success from evaluator telemetry.

The transition denominator is an executed previous shot (``capture_gate`` true),
not a previous legal return.  This directly measures:

    P(next shot legal | previous shot completed)

Rows are paired within one ``env_id`` and ``episode_id`` only; reset boundaries
cannot create a false transition.
"""

import argparse
import json
import pathlib


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _clip_name(clip_id):
    return {0: "FH", 1: "BH"}.get(clip_id)


def _completed(row):
    return bool(row.get("capture_gate", False))


def _legal(row):
    return all(
        bool(row.get(field, False))
        for field in ("capture_gate", "net_clear", "landing_valid", "on_opponent")
    )


def summarize(payload):
    rows = list(payload.get("rows", []))
    intervals = [
        float(row["strike_interval_s"])
        for row in rows
        if row.get("strike_interval_s") is not None
    ]

    grouped = {}
    for row in rows:
        clip = _clip_name(row.get("clip_id"))
        if clip is None:
            continue
        key = (int(row.get("env_id", -1)), int(row.get("episode_id", -1)))
        grouped.setdefault(key, []).append(row)

    transitions = {
        f"{source}->{target}": {"attempts": 0, "next_legal": 0, "rate": None}
        for source in ("FH", "BH")
        for target in ("FH", "BH")
    }
    previous_completed = 0
    previous_legal = 0
    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda row: (
                int(row.get("global_step", -1)),
                int(row.get("strike_index", -1)),
            )
        )
        for previous, current in zip(group_rows, group_rows[1:]):
            previous_clip = _clip_name(previous.get("clip_id"))
            current_clip = _clip_name(current.get("clip_id"))
            if previous_clip is None or current_clip is None:
                continue
            previous_index = previous.get("strike_index")
            current_index = current.get("strike_index")
            if previous_index is not None and current_index is not None:
                if int(current_index) != int(previous_index) + 1:
                    continue
            key = f"{previous_clip}->{current_clip}"
            if _completed(previous):
                previous_completed += 1
                transitions[key]["attempts"] += 1
                if _legal(current):
                    previous_legal += 1
                    transitions[key]["next_legal"] += 1

    for stats in transitions.values():
        if stats["attempts"]:
            stats["rate"] = stats["next_legal"] / stats["attempts"]

    return {
        "schema_version": 1,
        "source_schema_version": payload.get("schema_version"),
        "rows": len(rows),
        "strike_interval_s": {
            "count": len(intervals),
            "mean": sum(intervals) / len(intervals) if intervals else None,
            "p10": _percentile(intervals, 0.10),
            "p50": _percentile(intervals, 0.50),
            "p90": _percentile(intervals, 0.90),
        },
        "next_legal_given_previous_completed": {
            "attempts": previous_completed,
            "next_legal": previous_legal,
            "rate": previous_legal / previous_completed if previous_completed else None,
        },
        "transitions": transitions,
        "definition": {
            "previous_completed": "capture_gate == true",
            "next_legal": "capture_gate && net_clear && landing_valid && on_opponent",
            "pairing": "same env_id and episode_id, consecutive strike_index",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", help="evaluate.py --virtual-telemetry-out JSON")
    parser.add_argument("--json-out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    path = pathlib.Path(args.telemetry).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = summarize(payload)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        pathlib.Path(args.json_out).expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

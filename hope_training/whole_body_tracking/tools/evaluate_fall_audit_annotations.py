"""Evaluate fall-risk/confirmed/recovery labels against annotated traces.

The script deliberately consumes JSON produced by the replay audit rather than
recomputing physics.  It reports separate physical-confirmed, predicted-
unrecoverable, recovery-ready and timeout metrics, including lead-time
quantiles.  It is usable before IsaacLab is installed so annotation/calibration
work can proceed independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _first(frames, key, predicate):
    for frame in frames:
        value = frame.get("state", frame).get("fall_state", {})
        if predicate(value):
            return int(frame.get("control_step", 0))
    return None


def evaluate(paths: list[Path]) -> dict:
    confirmed_tp = confirmed_fp = confirmed_fn = 0
    ready_fp = 0
    risk_lead = []
    predicted_count = timeout_count = 0
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        frames = trace.get("frames") or trace.get("trace") or trace.get("pre_reset_trace_last_100") or []
        annotation = trace.get("annotation", {})
        label = annotation.get("label", "STABLE")
        actual_fall = label in {"PHYSICAL_CONTACT", "UNRECOVERABLE", "LATE_POST_HIT_FALL"}
        predicted_step = _first(frames, "predicted_unrecoverable", lambda state: bool(state.get("predicted_unrecoverable", False)))
        confirmed_step = _first(frames, "confirmed_fall", lambda state: bool(state.get("confirmed_fall", False)))
        ready_step = _first(frames, "recovery_ready", lambda state: bool(state.get("recovery_ready", False)))
        if predicted_step is not None:
            predicted_count += 1
        if trace.get("failure", {}).get("reason") == "recovery_timeout":
            timeout_count += 1
        if confirmed_step is not None and actual_fall:
            confirmed_tp += 1
        elif confirmed_step is not None and not actual_fall:
            confirmed_fp += 1
        elif actual_fall:
            confirmed_fn += 1
        if ready_step is not None and label not in {"STABLE", "RECOVERABLE", "NEAR_MISS"}:
            ready_fp += 1
        annotated_risk = annotation.get("first_visually_at_risk_step")
        if annotated_risk is not None and predicted_step is not None:
            risk_lead.append(int(annotated_risk) - predicted_step)
    precision = confirmed_tp / max(confirmed_tp + confirmed_fp, 1)
    recall = confirmed_tp / max(confirmed_tp + confirmed_fn, 1)
    ordered = sorted(risk_lead)
    quantile = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else None
    return {
        "trace_count": len(paths),
        "confirmed_precision": precision,
        "confirmed_recall": recall,
        "predicted_unrecoverable_trace_count": predicted_count,
        "false_recovery_ready_count": ready_fp,
        "recovery_timeout_count": timeout_count,
        "risk_lead_time_steps": {"P10": quantile(0.10), "P50": quantile(0.50), "P90": quantile(0.90)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.traces)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

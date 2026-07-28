#!/usr/bin/env python3
"""Generate and rank batched leg/waist support trajectories.

The simulator consumes the generated NPZ directly through
``scripts/train.py +audit_support_candidates=...``.  Candidates are additive
raw coordinator actions over the 12 leg and 3 waist dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SUPPORT_DIMS = 15


def _save_candidates(
    output: Path,
    motion_id: int,
    knots: np.ndarray,
    *,
    seed: int,
    source: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motion_ids=np.full(knots.shape[0], motion_id, dtype=np.int64),
        support_knots=knots.astype(np.float32),
        seed=np.asarray(seed, dtype=np.int64),
        source=np.asarray(source),
    )


def _antithetic_population(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    count: int,
    clip: float,
) -> np.ndarray:
    if count < 2:
        raise ValueError("num-candidates must be at least 2")
    population = [mean.copy()]
    pair_count = (count - 1) // 2
    noise = rng.standard_normal((pair_count, *mean.shape))
    for index in range(pair_count):
        delta = noise[index] * std
        population.extend((mean + delta, mean - delta))
    if len(population) < count:
        population.append(mean + rng.standard_normal(mean.shape) * std)
    return np.clip(np.stack(population[:count]), -clip, clip)


def generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    mean = np.zeros((args.num_knots, SUPPORT_DIMS), dtype=np.float64)
    std = np.full_like(mean, args.std)
    knots = _antithetic_population(rng, mean, std, args.num_candidates, args.clip)
    _save_candidates(
        args.output,
        args.motion_id,
        knots,
        seed=args.seed,
        source=f"zero_mean_antithetic_std_{args.std:g}",
    )


def _candidate_score(row: dict, knots: np.ndarray) -> float:
    if "position_error_m" not in row:
        return 1.0e9
    position = float(row["position_error_m"])
    forward = abs(float(row["root_forward_velocity_mps_at_hit"]))
    pitch_rate = abs(float(row["root_pitch_rate_radps_at_hit"]))
    velocity_error = float(row["velocity_error_mps"])
    normal_error = float(row["normal_error_deg"])

    score = (forward / 0.10) ** 2 + (pitch_rate / 0.15) ** 2
    score += 0.15 * (position / 0.10) ** 2
    score += 16.0 * (max(position - 0.10, 0.0) / 0.02) ** 2
    score += 0.05 * (velocity_error / 2.0) ** 2
    score += 4.0 * (max(normal_error - 20.0, 0.0) / 10.0) ** 2
    score += 0.02 * float(np.mean((knots / 0.25) ** 2))
    if knots.shape[0] > 1:
        score += 0.02 * float(np.mean((np.diff(knots, axis=0) / 0.25) ** 2))
    return score


def rank_and_refine(args: argparse.Namespace) -> None:
    with np.load(args.candidates, allow_pickle=False) as data:
        motion_ids = np.asarray(data["motion_ids"], dtype=np.int64)
        knots = np.asarray(data["support_knots"], dtype=np.float64)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows_by_index = {
        int(row["support_candidate_index"]): row
        for row in report["results"]
        if row.get("support_candidate_index") is not None
    }
    if len(rows_by_index) != knots.shape[0]:
        raise RuntimeError(
            f"report/candidate count mismatch: rows={len(rows_by_index)} candidates={knots.shape[0]}"
        )
    if not np.all(motion_ids == motion_ids[0]):
        raise ValueError("refinement expects a single motion per candidate batch")

    scores = np.asarray(
        [_candidate_score(rows_by_index[index], knots[index]) for index in range(knots.shape[0])]
    )
    order = np.argsort(scores)
    summary = []
    for index in order[: args.print_top]:
        row = rows_by_index[int(index)]
        summary.append(
            {
                "candidate_index": int(index),
                "score": float(scores[index]),
                "position_error_cm": 100.0 * float(row["position_error_m"]),
                "forward_velocity_mps": float(row["root_forward_velocity_mps_at_hit"]),
                "pitch_rate_radps": float(row["root_pitch_rate_radps_at_hit"]),
                "velocity_error_mps": float(row["velocity_error_mps"]),
                "normal_error_deg": float(row["normal_error_deg"]),
            }
        )
    print(json.dumps(summary, indent=2))

    best_index = int(order[0])
    _save_candidates(
        args.best_output,
        int(motion_ids[0]),
        knots[best_index : best_index + 1],
        seed=args.seed,
        source=f"best_from_{args.candidates}",
    )
    if args.refine_output is None:
        return

    elite_count = max(2, min(args.elite_count, knots.shape[0]))
    elite = knots[order[:elite_count]]
    elite_mean = elite.mean(axis=0)
    elite_std = np.maximum(elite.std(axis=0), args.min_std)
    rng = np.random.default_rng(args.seed)
    refined = _antithetic_population(
        rng,
        elite_mean,
        elite_std,
        args.num_candidates,
        args.clip,
    )
    _save_candidates(
        args.refine_output,
        int(motion_ids[0]),
        refined,
        seed=args.seed,
        source=f"cem_refine_elite_{elite_count}_from_{args.candidates}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--motion-id", type=int, required=True)
    generate_parser.add_argument("--num-candidates", type=int, default=256)
    generate_parser.add_argument("--num-knots", type=int, default=1)
    generate_parser.add_argument("--std", type=float, default=0.20)
    generate_parser.add_argument("--clip", type=float, default=0.50)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.set_defaults(func=generate)

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--candidates", type=Path, required=True)
    rank_parser.add_argument("--report", type=Path, required=True)
    rank_parser.add_argument("--best-output", type=Path, required=True)
    rank_parser.add_argument("--refine-output", type=Path)
    rank_parser.add_argument("--num-candidates", type=int, default=256)
    rank_parser.add_argument("--elite-count", type=int, default=24)
    rank_parser.add_argument("--min-std", type=float, default=0.04)
    rank_parser.add_argument("--clip", type=float, default=0.50)
    rank_parser.add_argument("--seed", type=int, default=1)
    rank_parser.add_argument("--print-top", type=int, default=10)
    rank_parser.set_defaults(func=rank_and_refine)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

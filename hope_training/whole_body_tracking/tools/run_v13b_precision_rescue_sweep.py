#!/usr/bin/env python3
"""Run the two-stage, read-only V1.3B PrecisionRescue checkpoint sweep.

No PPO update is performed.  The only child process is
``evaluate_v13b_precision_rescue_candidate.py`` on the requested evaluation
GPU.  The active CompletePriors training process is neither inspected for
mutable state nor signalled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue/checkpoint_selection"
EVALUATOR = ROOT / "tools/evaluate_v13b_precision_rescue_suite.py"
# The Common-set must remain teacher-aligned while it evaluates historical
# upper-prior conditions.  The evaluator freezes the companion private motion
# episode (motion id + rephased start frame) as part of this suite.
COMMON_PROGRESS = 0.20


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--python", default=None,
        help="Isaac-enabled Python executable. Defaults to the active interpreter, or the local hope-isaac env when launched from another shell.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--common-progress", type=float, default=COMMON_PROGRESS)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--finalists", type=int, default=None)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(base: Path, stage: str, row: dict, test_set: str, condition: str) -> Path:
    return base / f"physics_stage_{stage}" / f"model_{row['iteration']}" / f"{test_set}_{condition}.json"


def _load_valid(path: Path, row: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("status") != "pass":
        return None
    if int(value.get("iteration", -1)) != int(row["iteration"]):
        return None
    if Path(str(value.get("checkpoint", ""))).resolve() != Path(row["checkpoint"]).resolve():
        return None
    return value


def _run_suite(
    args: argparse.Namespace,
    stage: str,
    row: dict,
    test_sets: tuple[str, ...],
    conditions: list[str] | tuple[str, ...],
) -> None:
    outputs = [_result_path(args.base, stage, row, test_set, condition) for test_set in test_sets for condition in conditions]
    if not args.overwrite and all(_load_valid(path, row) is not None for path in outputs):
        print(f"[skip] stage={stage} model={row['iteration']} (all suite reports exist)", flush=True)
        return
    command = [
        args.python, str(EVALUATOR),
        "--checkpoint", str(row["checkpoint"]),
        "--iteration", str(row["iteration"]),
        "--historical-progress", f"{float(row['historical_progress']):.12g}",
        "--source-lower-alpha", f"{float(row['historical_lower_alpha']):.12g}",
        "--source-upper-alpha", f"{float(row['historical_upper_alpha']):.12g}",
        "--sets", ",".join(test_sets),
        "--conditions", ",".join(conditions),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--common-progress", f"{float(args.common_progress):.12g}",
        "--device", args.device,
        "--max-steps", str(args.max_steps),
        "--common-goal-suite", str(
            args.base / (
                f"common_set_teacher_aligned_v2_p{args.common_progress:.3f}_"
                f"seed{args.seed}_n{args.episodes}.json"
            )
        ),
        "--output-dir", str(_result_path(args.base, stage, row, test_sets[0], conditions[0]).parent),
    ]
    log = _result_path(args.base, stage, row, test_sets[0], conditions[0]).parent / "suite.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] stage={stage} model={row['iteration']} suite={','.join(test_sets)}×{','.join(conditions)}", flush=True)
    if args.dry_run:
        print(" ".join(command), flush=True)
        return
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed ({completed.returncode}): model={row['iteration']}; inspect {log}")
    missing = [path for path in outputs if _load_valid(path, row) is None]
    if missing:
        raise RuntimeError(f"suite exited without valid reports: {missing}")


def _pareto_front(rows: list[dict]) -> list[dict]:
    """Return non-dominated Common-set Upper-off actors.

    Smaller normal/velocity error and smaller reliance gap are better; larger
    Upper-off combined success is better.  This is intentionally not a single
    arbitrary scalar score.
    """
    front: list[dict] = []
    for candidate in rows:
        c = candidate["selection_metrics"]
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            o = other["selection_metrics"]
            no_worse = (
                o["upper_off_normal_error_deg"] <= c["upper_off_normal_error_deg"]
                and o["upper_off_velocity_error_mps"] <= c["upper_off_velocity_error_mps"]
                and o["prior_reliance_gap"] <= c["prior_reliance_gap"]
                and o["upper_off_combined_success"] >= c["upper_off_combined_success"]
            )
            strictly_better = (
                o["upper_off_normal_error_deg"] < c["upper_off_normal_error_deg"]
                or o["upper_off_velocity_error_mps"] < c["upper_off_velocity_error_mps"]
                or o["prior_reliance_gap"] < c["prior_reliance_gap"]
                or o["upper_off_combined_success"] > c["upper_off_combined_success"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["iteration"])


def _stage1_summary(base: Path, candidates: list[dict], finalists: int) -> dict:
    rows = []
    hashes: set[str] = set()
    for candidate in candidates:
        reports = {}
        complete = True
        for test_set in ("native", "common"):
            for condition in ("historical", "upper_off", "all_off"):
                report = _load_valid(_result_path(base, "1", candidate, test_set, condition), candidate)
                reports[f"{test_set}_{condition}"] = report
                complete &= report is not None
                if report is not None and test_set == "common":
                    hashes.add(report["common_episode_sha256"])
        if not complete:
            continue
        common_historical = reports["common_historical"]["metrics"]
        common_upper = reports["common_upper_off"]["metrics"]
        common_all = reports["common_all_off"]["metrics"]
        # Never silently rank a candidate whose strike event did not occur.
        numeric = (
            common_upper["normal_error_deg"], common_upper["velocity_error_mps"],
            common_historical["normal_error_deg"], common_historical["velocity_error_mps"],
        )
        if any(value is None for value in numeric):
            continue
        gap = max(0.0, common_upper["normal_error_deg"] - common_historical["normal_error_deg"])
        gap += max(0.0, common_upper["velocity_error_mps"] - common_historical["velocity_error_mps"])
        entry = {
            "iteration": candidate["iteration"], "checkpoint": candidate["checkpoint"],
            "historical_progress": candidate["historical_progress"],
            "historical_lower_alpha": candidate["historical_lower_alpha"],
            "historical_upper_alpha": candidate["historical_upper_alpha"],
            "reports": {key: str(_result_path(base, "1", candidate, *key.split("_", 1))) for key in reports},
            "hard_filter_pass": (
                reports["native_historical"]["metrics"]["survival_10s"] >= 0.95
                and reports["native_historical"]["metrics"]["position_error_m"] is not None
                and reports["native_historical"]["metrics"]["position_error_m"] <= 0.03
                and common_historical["survival_10s"] >= 0.95
            ),
            "selection_metrics": {
                "upper_off_normal_error_deg": common_upper["normal_error_deg"],
                "upper_off_velocity_error_mps": common_upper["velocity_error_mps"],
                "upper_off_combined_success": common_upper["combined_success"],
                "all_off_normal_error_deg": common_all["normal_error_deg"],
                "all_off_velocity_error_mps": common_all["velocity_error_mps"],
                "prior_reliance_gap": gap,
            },
        }
        rows.append(entry)
    eligible = [row for row in rows if row["hard_filter_pass"]]
    front = _pareto_front(eligible)
    # A deterministic tie order is only a shortlist mechanism.  It is NOT a
    # final source selection; stage 2 keeps the whole Pareto set where small.
    ordered = sorted(
        front,
        key=lambda row: (
            row["selection_metrics"]["upper_off_normal_error_deg"],
            row["selection_metrics"]["upper_off_velocity_error_mps"],
            row["selection_metrics"]["prior_reliance_gap"],
            -row["selection_metrics"]["upper_off_combined_success"],
        ),
    )
    shortlist = ordered[:finalists]
    return {
        "status": "pass" if len(hashes) == 1 and shortlist else "no_selection",
        "stage": 1,
        "common_set_progress": COMMON_PROGRESS,
        "common_episode_hashes": sorted(hashes),
        "common_set_identical": len(hashes) == 1,
        "hard_filter": {
            "native_historical_survival_10s_min": 0.95,
            "native_historical_position_error_max_m": 0.03,
            "common_historical_survival_10s_min": 0.95,
        },
        "evaluated": rows,
        "pareto_front": front,
        "stage_2_shortlist": shortlist,
        "selection_prohibited": True,
        "reason": "Stage 1 is a coarse filter only; final source selection requires Stage 2 Common+Native four-condition evidence.",
    }


def main() -> None:
    args = _args()
    if args.python is None:
        local_isaac_python = Path("/home/bistu/anaconda3/envs/hope-isaac/bin/python")
        # This makes an accidental invocation from the base shell fail less
        # surprisingly, while a different correctly activated Isaac env still
        # takes precedence through sys.executable.
        args.python = (
            str(local_isaac_python)
            if "hope-isaac" not in sys.executable and local_isaac_python.is_file()
            else sys.executable
        )
    plan = _read(args.base / "sweep_plan.json")
    if args.stage == "1":
        episodes = plan["stage_1"]["episodes_per_condition"] if args.episodes is None else args.episodes
        candidates = plan["stage_1"]["candidates"]
        conditions = plan["stage_1"]["conditions"]
        test_sets = ("native", "common")
        finalists = plan["stage_2"]["select_top_pareto"] if args.finalists is None else args.finalists
    else:
        source = _read(args.base / "stage_1_summary.json")
        if not source.get("common_set_identical", False):
            raise SystemExit("Stage 2 refused: Common-set hash contract did not pass Stage 1")
        candidates = source.get("stage_2_shortlist", [])
        if not candidates:
            raise SystemExit("Stage 2 refused: Stage 1 produced no eligible Pareto candidates")
        episodes = plan["stage_2"]["episodes_per_condition"] if args.episodes is None else args.episodes
        conditions = plan["stage_2"]["conditions"]
        test_sets = tuple(plan["stage_2"]["sets"])
        finalists = len(candidates)
    args.episodes = int(episodes)
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    for row in candidates:
        _run_suite(args, args.stage, row, test_sets, conditions)
    if args.dry_run:
        return
    if args.stage == "1":
        summary = _stage1_summary(args.base, candidates, int(finalists))
        out = args.base / "stage_1_summary.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[done] {out} status={summary['status']}", flush=True)
        if summary["status"] != "pass":
            raise SystemExit("Stage 1 did not produce an admissible shortlist")
    else:
        # Stage-2 source selection and Rescue no-learning replay are separate
        # evidence steps by contract; do not auto-select a source here.
        reports = [
            _load_valid(_result_path(args.base, "2", row, test_set, condition), row)
            for row in candidates for test_set in test_sets for condition in conditions
        ]
        hashes = {report["common_episode_sha256"] for report in reports if report and report["set"] == "common"}
        summary = {
            "status": "pass" if len(hashes) == 1 else "fail_common_episode_contract",
            "stage": 2, "common_episode_hashes": sorted(hashes),
            "finalists": candidates,
            "selection_prohibited": True,
            "next_required_step": "Pareto selection report, then no-learning PrecisionRescue wide-reward replay for 3-5 finalists.",
        }
        out = args.base / "stage_2_summary.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[done] {out} status={summary['status']}", flush=True)
        if summary["status"] != "pass":
            raise SystemExit("Stage 2 Common-set identity failed")


if __name__ == "__main__":
    main()

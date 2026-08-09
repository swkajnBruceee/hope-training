#!/usr/bin/env python3
"""Create the fail-closed V1.3B CompletePriors long-training gate.

The two PPO JSON summaries are intentionally explicit inputs: the training
runner must not infer success from a checkpoint filename or a scalar reward.
"""
from __future__ import annotations

import argparse
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_outputs/v13b_complete_priors_contract"


def read_pass(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("pass", False))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-20", type=pathlib.Path, required=True)
    parser.add_argument("--preflight-100", type=pathlib.Path, required=True)
    args = parser.parse_args()
    gates = {
        "one_strike_10s": read_pass(OUT / "one_strike_10s.json"),
        "alignment_100": read_pass(OUT / "alignment_100.json"),
        "ppo_preflight_20": read_pass(args.preflight_20),
        "ppo_preflight_100": read_pass(args.preflight_100),
    }
    result = {
        "contract": "v13b_complete_priors_aligned_one_strike_v1",
        "gates": gates,
        "pass": all(gates.values()),
        "preflight_20": str(args.preflight_20),
        "preflight_100": str(args.preflight_100),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("V1.3B CompletePriors long-training gate remains closed")


if __name__ == "__main__":
    main()

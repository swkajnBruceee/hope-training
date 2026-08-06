#!/usr/bin/env python3
"""Offline P1 TCP/time contract analyser.

The input is JSON either as ``[{sample}, ...]`` or ``{"samples": [...]}``.
Use one invocation per probe type:

  python tools/audit_strike_goal_contract.py tcp tcp_samples.json
  python tools/audit_strike_goal_contract.py time time_samples.json \
      --source-to-control-offset-s 0.0

The tool only analyses recorded values.  It does not connect a goal to an
actor and does not manufacture a clock-domain conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.strike_goal_contract_probe import (  # noqa: E402
    TcpProbeSample,
    TimeProbeSample,
    analyze_tcp_samples,
    analyze_time_samples,
)


def _load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError("JSON must be a sample list or an object with a 'samples' list")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("each sample must be a JSON object")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("tcp", "time"))
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--source-to-control-offset-s",
        type=float,
        default=None,
        help="explicitly measured mapping: control_time = source_time + offset; never guessed",
    )
    args = parser.parse_args()
    raw_samples = _load_samples(args.input)
    if args.probe == "tcp":
        report = analyze_tcp_samples([TcpProbeSample.from_mapping(item) for item in raw_samples])
    else:
        report = analyze_time_samples(
            [TimeProbeSample.from_mapping(item) for item in raw_samples],
            source_to_control_offset_s=args.source_to_control_offset_s,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate TTMD6 source-normalized artifacts without A3 assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = []
    errors = []
    for record in manifest["records"]:
        path = Path(record["normalized_npz"])
        item = {"source_id": record["source_id"], "normalized_npz": str(path)}
        try:
            data = np.load(path)
            human = data["human_local_raw"]
            paddle = data["paddle_local_raw"]
            basis = data["basis_local_columns_raw"]
            finite = bool(np.isfinite(human).all() and np.isfinite(paddle).all() and np.isfinite(basis).all())
            ortho = np.einsum("tji,tjk->tik", basis, basis)
            identity_error = float(np.max(np.abs(ortho - np.eye(3))))
            consecutive_dots = np.sum(basis[1:] * basis[:-1], axis=2)
            min_axis_dot = float(np.min(consecutive_dots)) if len(consecutive_dots) else 1.0
            item.update(
                {
                    "finite": finite,
                    "human_shape": list(human.shape),
                    "paddle_shape": list(paddle.shape),
                    "basis_orthonormal_max_error": identity_error,
                    "basis_min_consecutive_axis_dot": min_axis_dot,
                    "basis_frames_below_dot_0p95": int(np.sum(consecutive_dots < 0.95)),
                    "status": "pass" if finite and identity_error < 1e-5 and min_axis_dot >= 0.95 else "fail",
                }
            )
        except Exception as exc:  # noqa: BLE001 - report all clip failures together
            item.update({"status": "fail", "error": str(exc)})
        if item["status"] != "pass":
            errors.append(item)
        reports.append(item)

    output = {
        "input_manifest": str(args.manifest),
        "stage": "source_normalized_validation",
        "a3_retarget_started": False,
        "training_eligible": False,
        "record_count": len(reports),
        "pass_count": sum(item["status"] == "pass" for item in reports),
        "fail_count": len(errors),
        "errors": errors,
        "records": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(reports)} source-normalized clips")
    print(f"pass={output['pass_count']} fail={output['fail_count']}")
    print(args.output)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

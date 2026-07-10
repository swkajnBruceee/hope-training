#!/usr/bin/env python3
"""Build csv_to_npz commands for A3 retarget outputs."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    del _ROOT

import argparse
import json
from pathlib import Path

from analysis.mocap_cleaning.config import load_config


def _csv_for_stage(item: dict, stage: str) -> tuple[str | None, str | None]:
    if stage == "optimized":
        status = "pass" if bool(item.get("replay_ready", item.get("optimized_status") == "pass")) else "reject"
        return item.get("optimized_csv"), status
    if stage == "ik":
        return item.get("ik_init_csv"), item.get("ik_status")
    raise ValueError(f"unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/analysis/mocap_cleaning/configs/retarget_DATA260708_p2_a3_fixed.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=["optimized", "ik"], default="optimized")
    parser.add_argument("--include-reject", action="store_true", help="Generate commands for reject samples too.")
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--launcher",
        choices=["hope_isaac_py", "python"],
        default="hope_isaac_py",
        help="Interpreter command written into the generated shell script.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(str(config["output_root"]))
    if args.manifest is None:
        args.manifest = output_root / ("optimized_manifest.json" if args.stage == "optimized" else "ik_init_manifest.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    npz_dir = output_root / f"{args.stage}_motion_npz"
    jobs_path = output_root / f"csv_to_npz_{args.stage}_jobs.json"
    commands_path = output_root / f"csv_to_npz_{args.stage}_commands.sh"
    jobs = []
    for item in manifest.get("samples", []):
        csv_path, status = _csv_for_stage(item, args.stage)
        if not csv_path:
            continue
        if status != "pass" and not args.include_reject:
            continue
        episode_id = str(item["episode_id"])
        out_path = npz_dir / f"{episode_id}.npz"
        jobs.append(
            {
                "episode_id": episode_id,
                "input_file": str(csv_path),
                "output_file": str(out_path),
                "output_name": episode_id,
                "input_fps": int(config["time"]["fps"]),
                "output_fps": int(args.output_fps),
                "target_npz": str(item.get("target_npz", "")),
                "target_spec_json": str(item.get("target_spec_json", "")),
            }
        )

    jobs_path.write_text(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    if args.launcher == "hope_isaac_py":
        header.extend(
            [
                "source hope_training/whole_body_tracking/setup_train_env.sh",
                "",
            ]
        )
    cmd = [
        args.launcher,
        "hope_training/whole_body_tracking/scripts/csv_to_npz.py",
        "--batch_jobs_json",
        str(jobs_path),
        "--input_fps",
        str(int(config["time"]["fps"])),
        "--output_fps",
        str(args.output_fps),
        "--robot",
        "agibot_a3",
    ]
    if args.headless:
        cmd.append("--headless")
    commands_path.write_text("\n".join(header) + " ".join(cmd) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    report = {
        "stage": args.stage,
        "manifest": str(args.manifest),
        "commands": str(commands_path),
        "jobs_json": str(jobs_path),
        "motion_npz_dir": str(npz_dir),
        "command_count": len(jobs),
        "include_reject": bool(args.include_reject),
        "launcher": args.launcher,
    }
    report_path = output_root / f"csv_to_npz_{args.stage}_commands.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {commands_path}")
    print(f"Wrote {report_path}")
    print(f"command_count={len(jobs)}")


if __name__ == "__main__":
    main()

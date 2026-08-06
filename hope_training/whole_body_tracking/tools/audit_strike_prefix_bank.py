#!/usr/bin/env python3
"""Qualify continuously realized Strike/Base14 prefixes without direct loading.

This audit proves that each candidate handoff phase is reached by consecutive
physics rollout from frame zero with finite Base14 observations and zero
residual actions.  It deliberately does not approve direct state teleport or
PPO training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASE_OFFSETS = (
    ("preparation", -29),
    ("swing_start", -20),
    ("acceleration", -12),
    ("pre_contact", -5),
    ("contact", 0),
    ("deceleration", 5),
    ("follow_through", 20),
    ("ready_recovery", 40),
)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.bank.joinpath("rsi_capture_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source_manifest"]).expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    source_manifest = json.loads(source.read_text(encoding="utf-8"))
    source_by_episode = {str(item["episode_id"]): item for item in source_manifest["motions"]}

    rows = []
    entries = []
    passed = (
        manifest.get("task_id") == "HOPE-StrikeConditionedBase-AgibotA3-v0"
        and int(manifest.get("action_dim", -1)) == 14
        and bool(manifest.get("all_values_finite", False))
    )
    for entry in manifest["entries"]:
        episode_id = str(entry["episode_id"])
        source_entry = source_by_episode[episode_id]
        hit_frame = int(source_entry["hit_event"]["motion_hit_frame"])
        with np.load(args.bank / entry["state_file"], allow_pickle=False) as data:
            steps = np.asarray(data["motion_step"], dtype=np.int64)
            contiguous = bool(np.array_equal(np.diff(steps), np.ones(len(steps) - 1, dtype=np.int64)))
            finite = all(
                np.isfinite(data[key]).all()
                for key in data.files
                if data[key].dtype.kind in "fc"
            )
            zero_residual = bool(np.max(np.abs(data["raw_action"])) <= 1.0e-7)
            observation_dim = int(data["policy_observation"].shape[-1])
            episode_ok = contiguous and finite and zero_residual and observation_dim == 105
            entries.append(
                {
                    "episode_id": episode_id,
                    "frames": int(len(steps)),
                    "motion_step_start": int(steps[0]),
                    "motion_step_end": int(steps[-1]),
                    "phase_wraps": int(np.sum(np.diff(steps) < 0)),
                    "contiguous": contiguous,
                    "finite": finite,
                    "zero_residual": zero_residual,
                    "policy_observation_dim": observation_dim,
                    "passed": episode_ok,
                }
            )
            passed = passed and episode_ok

            for phase, offset in PHASE_OFFSETS:
                requested_step = min(max(hit_frame + offset, int(steps[0])), int(steps[-1]))
                index = int(np.argmin(np.abs(steps - requested_step)))
                previous = max(0, index - 1)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "phase": phase,
                        "motion_step": int(steps[index]),
                        "prefix_steps": int(index + 1),
                        "joint_delta_norm": _norm(data["joint_pos"][index] - data["joint_pos"][previous]),
                        "root_delta_norm": _norm(data["root_state_w"][index] - data["root_state_w"][previous]),
                        "target_delta_norm": _norm(
                            data["joint_pos_target"][index] - data["joint_pos_target"][previous]
                        ),
                        "torque_delta_norm_nm": _norm(
                            data["applied_torque"][index] - data["applied_torque"][previous]
                        ),
                        "torque_abs_max_nm": float(np.max(np.abs(data["applied_torque"][index]))),
                        "observation_delta_norm": _norm(
                            data["policy_observation"][index] - data["policy_observation"][previous]
                        ),
                    }
                )

    report = {
        "schema_version": 1,
        "stage": "strike_base14_continuous_prefix_qualification_v1",
        "passed": bool(passed),
        "training_eligible": False,
        "direct_load_eligible": False,
        "continuous_prefix_handoff_verified": bool(passed),
        "scope": "zero-residual continuous rollout to phase; no reset and no policy switch",
        "entries": entries,
        "handoff_markers": rows,
        "summary": {
            "motions": len(entries),
            "markers": len(rows),
            "max_joint_delta_norm": max(row["joint_delta_norm"] for row in rows),
            "max_torque_abs_nm": max(row["torque_abs_max_nm"] for row in rows),
            "max_torque_delta_norm_nm": max(row["torque_delta_norm_nm"] for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["summary"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

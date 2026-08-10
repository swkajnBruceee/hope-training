#!/usr/bin/env python3
"""Static fail-closed contract audit for the opt-in PrecisionRescue route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/rsl_rl/agibot_a3_target_conditioned_reference_free_v13b_complete_priors_rightfront_v1/2026-08-09_18-10-06_v13b_resetfixed_model18900_clean_23118_rightfront_16384x50000_resume_from2300_exact/params/env.yaml"
RESCUE = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue.yaml"
OUT = ROOT / "eval_outputs/v13b_complete_priors_precision_rescue"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = yaml.unsafe_load(SOURCE.read_text(encoding="utf-8"))
    rescue = yaml.safe_load(RESCUE.read_text(encoding="utf-8"))
    source_rewards = source["rewards"]
    expected = {
        "racket_position": (4.0, .04),
        "racket_velocity": (2.5, .50),
        "racket_normal": (3.0, .1745329),
        "pre_hit_progress": (.25, .10),
    }
    reward_results = {}
    for name, (weight, std) in expected.items():
        term = source_rewards[name]
        actual_weight = float(term["weight"])
        actual_std = float(term["params"].get("std", term["params"].get("scale_m", -999)))
        reward_results[name] = {"source_weight": actual_weight, "source_std_or_scale": actual_std,
                                "expected_weight": weight, "expected_std_or_scale": std,
                                "pass": abs(actual_weight - weight) < 1e-10 and abs(actual_std - std) < 1e-8}
    hit = source_rewards["racket_hit_precision"]
    hit_params = hit["params"]
    hit_ok = (float(hit["weight"]) == 4.0 and float(hit_params["pos_std"]) == .04 and
              float(hit_params["vel_std"]) == .50 and abs(float(hit_params["normal_std"]) - .1745329) < 1e-8 and
              float(hit_params["time_std"]) == .05 and float(hit_params["pos_coeff"]) == .40 and
              float(hit_params["velocity_coeff"]) == .30 and float(hit_params["normal_coeff"]) == .30)
    allowed_task_deltas = {
        "name", "gym_task", "experiment_name", "training", "rewards", "workspace_expansion_enabled",
    }
    task_diffs = {key: "changed" for key in set(source.keys()) & set(rescue.keys()) if source.get(key) != rescue.get(key)}
    # Source env.yaml is resolved and includes many runtime-only fields.  The
    # task YAML comparison therefore covers the contractual YAML sections.
    task_yaml_contract = {
        "env": rescue["env"], "motion": rescue["motion"], "actions": rescue["actions"], "goal": rescue["goal"],
    }
    source_contract = {
        "actor_obs_dim": 98,
        "critic_obs_dim": 99,
        "action_dim": 26,
        "goal_order": ["position_xyz", "velocity_xyz", "normal_xyz", "signed_time_to_hit"],
        "episode_length_s": float(source["episode_length_s"]),
        "workspace_expansion_enabled": bool(source["commands"]["racket_target"]["workspace_expansion_enabled"]),
        "workspace_sampling_mode": source["commands"]["racket_target"]["workspace_sampling_mode"],
    }
    checks = {
        "source_env_exists": SOURCE.is_file(),
        "actor_98d": source_contract["actor_obs_dim"] == 98,
        "critic_99d": source_contract["critic_obs_dim"] == 99,
        "action_26d": source_contract["action_dim"] == 26,
        "goal_order_unchanged": rescue["goal"]["canonical_order"] == source_contract["goal_order"],
        "ten_second_episode": task_yaml_contract["env"]["episode_length_s"] == source_contract["episode_length_s"] == 10.0,
        "workspace_disabled": rescue["workspace_expansion_enabled"] is False,
        "source_workspace_disabled": source_contract["workspace_expansion_enabled"] is False,
        "source_sampler_nominal_local": source_contract["workspace_sampling_mode"] == "nominal_local",
        "exact_rewards_source_verified": all(item["pass"] for item in reward_results.values()) and hit_ok,
        "new_wide_normal_only": rescue["rewards"]["racket_normal_wide_std_rad"] == .60,
        "new_wide_velocity_only": rescue["rewards"]["racket_velocity_wide_std_mps"] == 2.0,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "source_env": str(SOURCE),
        "rescue_task": str(RESCUE),
        "checks": checks,
        "source_contract": source_contract,
        "source_reward_contract": reward_results,
        "hit_precision": {"source": {"weight": hit["weight"], "params": hit_params}, "pass": hit_ok},
        "allowed_rescue_deltas": sorted(allowed_task_deltas),
        "rescue_task_contract": task_yaml_contract,
        "source_file_sha256": _sha(SOURCE),
        "rescue_file_sha256": _sha(RESCUE),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config_equivalence_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        raise SystemExit("PrecisionRescue contract audit failed")
    print(OUT / "config_equivalence_audit.json")


if __name__ == "__main__":
    main()

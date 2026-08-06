"""Hydra training entry for HOPE Agibot A3 WBC.

Pick the task/algo YAML on the command line and override any field:

    python scripts/train.py task=TrackingFlat algo=ppo headless=true logger=tensorboard \
        motion_file=sample_motions/agibot_a3_smoke_stand.npz num_envs=32 max_iterations=3

    python scripts/train.py task=TrackingFlat algo=ppo num_envs=2048 max_iterations=20000 \
        registry_name=<org>/wandb-registry-motions/<motion_name>

Tune by editing cfg/task/*.yaml (env / reward / racket / DR) and cfg/algo/ppo.yaml (PPO). This
script reuses BeyondMimic's training mechanics (Isaac Lab + rsl_rl). A local `motion_file=...`
is preferred for public smoke runs; WandB registry loading is optional.
"""

import os
import sys
import hashlib
import json
import math
from pathlib import Path

import hydra
from omegaconf import OmegaConf


# Make the ``training`` package importable regardless of how this script was
# invoked. Without this, the script silently relies on PYTHONPATH being set
# by an external wrapper (e.g. ``hope_isaac_py`` from setup_train_env.sh),
# and a forgotten ``source setup_train_env.sh`` makes ``import training``
# fail with ModuleNotFoundError deep inside _run(). Resolve paths relative
# to THIS FILE so the script works from any cwd and any checkout location.
_HERE = os.path.dirname(os.path.abspath(__file__))                # .../scripts
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))          # .../whole_body_tracking
for _p in (
    _REPO_ROOT,
    os.path.normpath(os.path.join(_REPO_ROOT, "show")),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _HERE, _REPO_ROOT, _p

from tools.a3_strike_contract import assert_training_manifest


def _assert_a3_base_stand_smoke_gate(
    task_id: str,
    max_iterations: int,
    num_envs: int | None = None,
    init_noise_std: float | None = None,
    recovery_v23: bool = False,
) -> None:
    """Fail closed unless this is an explicitly bounded, audited Stand smoke."""
    stand_v0 = "A3BaseStand-v0"
    authority_candidate = "A3BaseStandAuthorityCandidate-v0"
    clip_candidate = "A3BaseStandClipCandidate-v0"
    authority_clip_candidate = "A3BaseStandAuthorityClipCandidate-v0"
    passive_stable_candidate = "A3BaseStandPassiveStableCandidate-v0"
    recovery_a = "A3BaseStandRecoveryA-v0"
    recovery_v2_tasks = {
        "A3BaseStandRecoveryAV2-v0",
        "A3BaseStandRecoveryAV2WaistMask-v0",
    }
    recovery_v21_tasks = {"A3BaseStandRecoveryAV21WaistMask-v0"}
    if task_id not in (
        stand_v0,
        authority_candidate,
        clip_candidate,
        authority_clip_candidate,
        passive_stable_candidate,
        recovery_a,
        *recovery_v2_tasks,
        *recovery_v21_tasks,
    ):
        return

    root = Path(__file__).resolve().parents[1]
    if task_id == recovery_a or task_id in recovery_v2_tasks or task_id in recovery_v21_tasks:
        passive_decision_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_passive_stable_decision_v1.json"
        )
        recovery_gate_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_recovery_a_gate_v1.json"
        )
        passive_decision = json.loads(passive_decision_path.read_text(encoding="utf-8"))
        recovery_gate = json.loads(recovery_gate_path.read_text(encoding="utf-8"))
        passive_status = passive_decision["qualification_status"]
        recovery_status = recovery_gate["qualification_status"]
        if not (
            passive_status.get("passive_stand_plant_approved") is True
            and passive_status.get("stand_recovery_task_development_approved") is True
            and recovery_status.get("recovery_a_environment_runtime_qualified") is True
            and recovery_status.get("recovery_reward_v3_semantics_approved") is True
            and recovery_status.get("recovery_disturbance_contract_approved") is True
            and recovery_status.get("recovery_envelope_approved") is True
            and recovery_status.get("zero_actor_initialization_runtime_verified") is True
            and recovery_status.get("untrained_stochastic_policy_safety_verified") is True
            and recovery_status.get("bounded_recovery_smoke_approved") is True
            and (
                task_id == recovery_a
                or recovery_status.get("recovery_a_v2_training_approved") is True
                or recovery_status.get("recovery_a_v21_training_approved") is True
            )
        ):
            raise RuntimeError(
                "A3BaseStandRecoveryA-v0 PPO is closed until its disturbance, passive-baseline, "
                "reward-v3, action-bound, and zero-actor gates pass."
            )
        for evidence in recovery_gate["evidence"].values():
            evidence_path = root / evidence["path"]
            if not evidence_path.is_file():
                raise RuntimeError(f"A3 Base Recovery evidence is missing: {evidence_path}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence["sha256"]:
                raise RuntimeError(f"A3 Base Recovery evidence hash mismatch: {evidence_path}")
        budget = (
            recovery_gate["v23_smoke_budget"]
            if recovery_v23
            else
            recovery_gate["v21_smoke_budget"]
            if task_id in recovery_v21_tasks
            else recovery_gate["v2_smoke_budget"]
            if task_id in recovery_v2_tasks
            else recovery_gate["bounded_smoke_budget"]
        )
        if budget["max_iterations"] is None or budget["max_num_envs"] is None:
            raise RuntimeError("A3 Base Recovery bounded smoke budget is not frozen")
        if not (
            int(max_iterations) == int(budget["max_iterations"])
            and num_envs is not None
            and 1 <= int(num_envs) <= int(budget["max_num_envs"])
            and init_noise_std is not None
            and abs(float(init_noise_std) - float(budget["required_init_noise_std"])) <= 1.0e-9
        ):
            raise RuntimeError(
                "A3 Base Recovery request does not match its bounded smoke budget: "
                f"iterations={max_iterations}, envs={num_envs}, noise={init_noise_std}"
            )
        return
    final_decision_path = (
        root
        / "contracts"
        / "a3_base_locomotion_v1"
        / "stand_causal_audit_decision_v1.json"
    )
    if final_decision_path.is_file() and task_id != passive_stable_candidate:
        final_decision = json.loads(final_decision_path.read_text(encoding="utf-8"))
        final_status = final_decision.get("qualification_status", {})
        if final_status.get("additional_ppo_smoke_approved") is not True:
            raise RuntimeError(
                "All additional A3 Base Stand PPO is closed by "
                "stand_causal_audit_decision_v1.json until static working-point, "
                "reward-v2, contact, and low-noise exploration gates pass."
            )
    if task_id == passive_stable_candidate:
        if int(max_iterations) != 100 or num_envs is None or not 1 <= int(num_envs) <= 64:
            raise RuntimeError(
                "A3BaseStandPassiveStableCandidate-v0 is approved only for exactly "
                f"100 iterations and at most 64 environments; requested {max_iterations}/{num_envs}."
            )
        if init_noise_std is None or abs(float(init_noise_std) - 0.15) > 1.0e-9:
            raise RuntimeError(
                "A3BaseStandPassiveStableCandidate-v0 requires init_noise_std=0.15; "
                f"requested {init_noise_std}."
            )
        gate_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_passive_stable_candidate_gate_v3.json"
        )
        if not gate_path.is_file():
            raise RuntimeError("A3 Base passive-stable candidate gate is missing")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        status = gate["qualification_status"]
        if not (
            status.get("bounded_100_iteration_smoke_approved") is True
            and status.get("candidate_gain_contract_approved") is False
            and status.get("stand_long_training_approved") is False
            and status.get("deployment_approved") is False
        ):
            raise RuntimeError("A3 Base passive-stable candidate gate status mismatch")
        for evidence in gate["local_evidence"].values():
            evidence_path = root / evidence["path"]
            if not evidence_path.is_file():
                raise RuntimeError(f"A3 Base passive-stable evidence is missing: {evidence_path}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence["sha256"]:
                raise RuntimeError(f"A3 Base passive-stable evidence hash mismatch: {evidence_path}")
        audit = json.loads(
            (root / gate["local_evidence"]["deterministic_audit"]["path"]).read_text(encoding="utf-8")
        )
        reward = json.loads(
            (root / gate["local_evidence"]["reward_v2_audit"]["path"]).read_text(encoding="utf-8")
        )
        ablation = json.loads(
            (root / gate["local_evidence"]["gain_ablation"]["path"]).read_text(encoding="utf-8")
        )
        if not (
            audit.get("task") == passive_stable_candidate
            and audit.get("passed") is True
            and audit.get("zero_action_baseline_stable_for_requested_window") is True
            and reward.get("passed") is True
            and reward.get("termination_penalty_equivalent_alive_seconds") == 2
            and ablation["passive_gain_ablation"]["pd_base14"]["recorded_steps"] == 500
            and ablation["passive_gain_ablation"]["pd_base14"]["non_timeout_failure"] is False
        ):
            raise RuntimeError("A3 Base passive-stable evidence payload mismatch")
        print(
            "[train.py] A3 Base passive-stable 100-iteration gate passed: "
            f"gate={gate['gate_id']}",
            flush=True,
        )
        return
    if task_id == authority_candidate:
        if int(max_iterations) != 100:
            raise RuntimeError(
                "A3BaseStandAuthorityCandidate-v0 is approved for exactly one "
                f"100-iteration diagnostic smoke; requested {max_iterations}."
            )
        if num_envs is None or not 1 <= int(num_envs) <= 64:
            raise RuntimeError(
                "A3BaseStandAuthorityCandidate-v0 is capped at 64 environments; "
                f"requested {num_envs}."
            )
        gate_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_authority_candidate_gate_v1.json"
        )
        if not gate_path.is_file():
            raise RuntimeError("A3 Base authority candidate gate is missing")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        status = gate["qualification_status"]
        if not (
            status.get("stand_authority_candidate_smoke_approved") is True
            and status.get("candidate_gain_contract_approved") is False
            and status.get("stand_phase1_qualified") is False
            and status.get("stand_long_training_approved") is False
            and status.get("deployment_approved") is False
        ):
            raise RuntimeError("A3 Base authority candidate gate status mismatch")
        for path_key, hash_key in (
            ("authority_audit_path", "authority_audit_sha256"),
            ("waist_scan_path", "waist_scan_sha256"),
        ):
            evidence_path = root / gate["local_evidence"][path_key]
            if not evidence_path.is_file():
                raise RuntimeError(f"A3 Base authority evidence is missing: {evidence_path}")
            actual_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if actual_hash != gate["local_evidence"][hash_key]:
                raise RuntimeError(
                    f"A3 Base authority evidence hash mismatch: {evidence_path}"
                )
        audit = json.loads(
            (root / gate["local_evidence"]["authority_audit_path"]).read_text(encoding="utf-8")
        )
        scan = json.loads(
            (root / gate["local_evidence"]["waist_scan_path"]).read_text(encoding="utf-8")
        )
        if not (
            audit.get("task") == authority_candidate
            and audit.get("passed") is True
            and scan.get("runtime_integrity_passed") is True
            and scan.get("waist_pitch_kp_nm_per_rad") == 350.0
            and scan.get("waist_pitch_kd_nms_per_rad") == 7.0
        ):
            raise RuntimeError("A3 Base authority candidate evidence payload mismatch")
        print(
            "[train.py] A3 Base Stand authority candidate gate passed: "
            f"iterations={max_iterations}, gate={gate['gate_id']}",
            flush=True,
        )
        return

    if task_id == authority_clip_candidate:
        if int(max_iterations) != 100 or num_envs is None or not 1 <= int(num_envs) <= 64:
            raise RuntimeError(
                "A3BaseStandAuthorityClipCandidate-v0 is approved only for exactly "
                f"100 iterations and at most 64 environments; requested {max_iterations}/{num_envs}."
            )
        gate_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_authority_clip_candidate_gate_v1.json"
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        status = gate["qualification_status"]
        if not (
            status.get("factorial_final_smoke_approved") is True
            and status.get("candidate_action_contract_approved") is False
            and status.get("candidate_gain_contract_approved") is False
            and status.get("stand_phase1_qualified") is False
            and status.get("stand_long_training_approved") is False
            and status.get("deployment_approved") is False
        ):
            raise RuntimeError("A3 Base authority+clip gate status mismatch")
        for path_key, hash_key in (
            ("candidate_audit_path", "candidate_audit_sha256"),
            ("support_audit_path", "support_audit_sha256"),
        ):
            evidence_path = root / gate["local_evidence"][path_key]
            if not evidence_path.is_file():
                raise RuntimeError(f"A3 Base authority+clip evidence is missing: {evidence_path}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != gate["local_evidence"][hash_key]:
                raise RuntimeError(f"A3 Base authority+clip evidence hash mismatch: {evidence_path}")
        audit = json.loads(
            (root / gate["local_evidence"]["candidate_audit_path"]).read_text(encoding="utf-8")
        )
        if not (
            audit.get("task") == authority_clip_candidate
            and audit.get("passed") is True
            and audit.get("reset_contract", {}).get("raw_action_clip_abs") == 0.5
        ):
            raise RuntimeError("A3 Base authority+clip audit payload mismatch")
        print(
            "[train.py] A3 Base Stand authority+clip factorial gate passed: "
            f"iterations={max_iterations}, gate={gate['gate_id']}",
            flush=True,
        )
        return

    if task_id == clip_candidate:
        if int(max_iterations) != 100:
            raise RuntimeError(
                "A3BaseStandClipCandidate-v0 is approved for exactly one "
                f"100-iteration diagnostic smoke; requested {max_iterations}."
            )
        if num_envs is None or not 1 <= int(num_envs) <= 64:
            raise RuntimeError(
                "A3BaseStandClipCandidate-v0 is capped at 64 environments; "
                f"requested {num_envs}."
            )
        gate_path = (
            root
            / "contracts"
            / "a3_base_locomotion_v1"
            / "stand_clip_candidate_gate_v1.json"
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        status = gate["qualification_status"]
        if not (
            status.get("stand_clip_candidate_smoke_approved") is True
            and status.get("candidate_action_contract_approved") is False
            and status.get("stand_phase1_qualified") is False
            and status.get("stand_long_training_approved") is False
            and status.get("deployment_approved") is False
        ):
            raise RuntimeError("A3 Base clip candidate gate status mismatch")
        for path_key, hash_key in (
            ("clip_audit_path", "clip_audit_sha256"),
            ("v1_checkpoint_eval_path", "v1_checkpoint_eval_sha256"),
        ):
            evidence_path = root / gate["local_evidence"][path_key]
            if not evidence_path.is_file():
                raise RuntimeError(f"A3 Base clip evidence is missing: {evidence_path}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != gate["local_evidence"][hash_key]:
                raise RuntimeError(f"A3 Base clip evidence hash mismatch: {evidence_path}")
        audit = json.loads(
            (root / gate["local_evidence"]["clip_audit_path"]).read_text(encoding="utf-8")
        )
        if not (
            audit.get("task") == clip_candidate
            and audit.get("passed") is True
            and audit.get("reset_contract", {}).get("raw_action_clip_abs") == 0.5
        ):
            raise RuntimeError("A3 Base clip candidate audit payload mismatch")
        print(
            "[train.py] A3 Base Stand clip candidate gate passed: "
            f"iterations={max_iterations}, gate={gate['gate_id']}",
            flush=True,
        )
        return

    if not 100 <= int(max_iterations) <= 500:
        raise RuntimeError(
            "A3BaseStand-v0 is approved only for a 100--500 iteration PPO smoke; "
            f"requested {max_iterations}. Long training remains closed."
        )

    gate_path = root / "contracts" / "a3_base_locomotion_v1" / "stand_fixture_gate_v1.json"
    audit_path = root / "artifacts" / "a3_base_stand" / "stand_audit_v1.json"
    if not gate_path.is_file() or not audit_path.is_file():
        raise RuntimeError(
            "A3 Base Stand smoke requires both stand_fixture_gate_v1.json and the local "
            "artifacts/a3_base_stand/stand_audit_v1.json evidence."
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    status = gate["qualification_status"]
    required_status = {
        "fixture_runner_qualified": True,
        "fixture_matrix_approved": True,
        "stand_task_approved": True,
        "stand_smoke_approved": True,
        "stand_long_training_approved": False,
        "locomotion_command_approved": False,
        "deployment_approved": False,
    }
    mismatched = {
        key: status.get(key) for key, expected in required_status.items() if status.get(key) is not expected
    }
    if mismatched:
        raise RuntimeError(f"A3 Base Stand gate status mismatch: {mismatched}")

    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if not (
        audit.get("task") == task_id
        and audit.get("passed") is True
        and audit.get("reset_contract", {}).get("passed") is True
        and audit.get("composer_full_target_passed") is True
        and all(stage.get("passed_runtime_integrity") is True for stage in audit.get("stages", []))
        and audit.get("stand_long_training_approved") is False
        and audit.get("deployment_approved") is False
    ):
        raise RuntimeError("A3 Base Stand deterministic audit does not satisfy the bounded smoke preconditions")
    print(
        "[train.py] A3 Base Stand bounded smoke gate passed: "
        f"iterations={max_iterations}, audit_sha256={hashlib.sha256(audit_bytes).hexdigest()}",
        flush=True,
    )


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab builds that no longer expose dump_pickle."""
    import os
    import pickle

    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
# Task YAML -> Isaac Lab env cfg overrides (only keys present in the YAML are applied).
# --------------------------------------------------------------------------- #
def _get(node, key, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return default


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


class _OverrideError(AttributeError):
    """Raised when the task YAML asks to override an attribute the composed env cfg does not have."""


def _require(cond, target):
    # The YAML explicitly set a value, but the target attribute is missing on the composed env cfg.
    # That is NEVER a benign no-op: either a STALE/shadowed training was imported (so the
    # cfg classes differ from the working tree) or the Hydra base groups failed to compose. Fail loud
    # instead of silently dropping the override (the old behaviour that hid the std/curriculum edits).
    if not cond:
        raise _OverrideError(
            f"[train.py] task YAML overrides '{target}' but the composed env cfg has no such attribute. "
            f"Check the '[train.py] env cfg source:' line above — if it points into site-packages rather "
            f"than your working tree, a stale install is shadowing the source (fix PYTHONPATH ordering / "
            f"reinstall editable). Otherwise the Hydra base-group composition for this task failed."
        )


def _set_attr(obj, attr, val, cast, applied, where):
    if val is None:
        return  # key absent from YAML -> keep the code default (documented contract)
    _require(hasattr(obj, attr), f"{where}.{attr}")
    setattr(obj, attr, cast(val))
    applied.append(f"{where}.{attr}={cast(val)!r}")


def _set_range(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    rng = (float(val[0]), float(val[1]))
    setattr(obj, attr, rng)
    applied.append(f"{where}.{attr}={rng}")


def _set_vec3(obj, attr, val, applied, where):
    if val is None:
        return
    _require(hasattr(obj, attr), f"{where}.{attr}")
    vec = (float(val[0]), float(val[1]), float(val[2]))
    setattr(obj, attr, vec)
    applied.append(f"{where}.{attr}={vec}")


def _assert_strict_fall_contract(env_cfg, task_id: str) -> None:
    """Fail fast if P5U training is composed without the strict fall contract.

    The old root-height/recovery terms are intentionally retained for legacy
    diagnostics, but they are too permissive to protect a P5U run.  A future
    Hydra composition or stale installed package must therefore never be
    allowed to launch a tracker run silently without both the terminal gate
    and its dense early-warning reward.
    """
    if "UnifiedUpperReferenceTracker" not in str(task_id):
        return
    term = getattr(getattr(env_cfg, "terminations", None), "strict_fall", None)
    risk = getattr(getattr(env_cfg, "rewards", None), "strict_fall_risk", None)
    if term is None or risk is None:
        raise RuntimeError(
            "P5U strict-fall contract missing: expected terminations.strict_fall "
            "and rewards.strict_fall_risk in the composed working-tree config."
        )
    term_params = dict(getattr(term, "params", {}) or {})
    required = {
        "max_tilt_rad": 0.785398,
        "minimum_height": 0.82,
        "max_torso_tilt_rad": 0.785398,
        "minimum_torso_height": 0.70,
        "required_steps": 2,
    }
    for key, expected in required.items():
        actual = term_params.get(key)
        if actual is None or abs(float(actual) - float(expected)) > 1.0e-5:
            raise RuntimeError(
                f"P5U strict-fall contract mismatch for {key}: "
                f"expected {expected}, got {actual!r}"
            )
    risk_weight = float(getattr(risk, "weight", 0.0))
    if risk_weight >= 0.0:
        raise RuntimeError(
            f"P5U strict_fall_risk must be a negative penalty, got weight={risk_weight}"
        )
    print(
        "[train.py] strict-fall contract verified: "
        f"tilt>{required['max_tilt_rad']:.6f} rad or height<{required['minimum_height']:.2f} m "
        f"for {required['required_steps']} steps; dense risk weight={risk_weight:.3f}",
        flush=True,
    )


def _assert_fall_recovery_admission(
    task_id: str,
    task_cfg_name: str | None = None,
    *,
    legacy_fall_strategy: bool = False,
) -> None:
    """Keep floating-base fall/recovery PPO closed until the full audit is signed.

    A config-level strict-fall term is not sufficient evidence for the DOCX
    admission contract.  The gate is deliberately fail-closed and requires a
    checked-in JSON decision containing the D0--D6 and precision/recall
    evidence flags.  This function only guards training; replay/audit tools
    remain available while the gate is closed.
    """
    task_text = f"{task_id} {task_cfg_name or ''}"
    import re
    # Every floating-base A3 strike/tracker variant shares the same physical
    # fall and next-action contract, not only the historical F0--F8 names.
    # Keep fixed-base/legacy table-tennis training unaffected, but fail closed
    # for P5D/P5U and later Floating* registrations until the signed audit is
    # complete.
    if not (
        "AgibotA3" in task_text
        and "Floating" in task_text
    ):
        return
    if legacy_fall_strategy:
        print(
            "[train.py] legacy fall strategy enabled: using the established "
            "strict_fall/base_height/recovery_tilt termination terms; the newer "
            "signed D0-D6 admission gate is intentionally bypassed for this run.",
            flush=True,
        )
        return
    gate_path = Path(__file__).resolve().parents[1] / "contracts" / "fall_detection_recovery_admission_v1.json"
    if not gate_path.is_file():
        raise RuntimeError(
            "Fall/recovery training admission is closed: missing "
            f"{gate_path}. Complete the DOCX audit and sign the gate before "
            "starting floating-base fall/recovery PPO."
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    status = gate.get("qualification_status", {})
    required = ("unified_state_source", "cycle_guard", "recovery_gate", "d0_d6_physx", "metrics_precision_recall")
    if gate.get("schema_version") != "fall_detection_recovery_admission/v1" or not all(status.get(key) is True for key in required) or status.get("training_admission") is not True:
        raise RuntimeError(
            "Fall/recovery training admission is closed: the signed gate does "
            f"not prove all required audit items ({', '.join(required)})."
        )


def _set_reward(rewards, name, weight, std, applied):
    if weight is None and std is None:
        return  # this reward term is not overridden by the YAML -> keep code defaults
    _require(hasattr(rewards, name), f"rewards.{name}")
    term = getattr(rewards, name)
    _require(term is not None, f"rewards.{name} (term is disabled/None)")
    if weight is not None:
        term.weight = float(weight)
        applied.append(f"rewards.{name}.weight={float(weight)}")
    if std is not None:
        _require("std" in term.params, f"rewards.{name}.params['std']")
        term.params["std"] = float(std)
        applied.append(f"rewards.{name}.params.std={float(std)}")


# YAML keys under `racket:` that target the RacketTargetCommandCfg (used to decide whether the task
# actually requested racket overrides before requiring the command to exist).
_RACKET_KEYS = (
    "strike_phase", "strike_window_s", "strike_time_std_s", "strike_success_pos_thresh",
    "strike_success_vel_thresh", "strike_success_normal_thresh_deg",
    "pos_x_range", "pos_y_range", "pos_z_range",
    "vel_x_range", "vel_y_range", "vel_z_range",
    "base_target_x_range", "base_target_y_range",
    "normal_mode", "forehand_on_negative_y", "mount_normal_axis", "mount_normal_sign",
    "target_mode", "ref_perturb_pos", "ref_perturb_vel", "ref_perturb_normal",
    "ref_perturb_curriculum_steps", "ref_perturb_curriculum_start",
    "ref_perturb_success_gated", "ref_perturb_advance_threshold", "ref_perturb_advance_rate",
    "exact_success_decay", "exact_success_min_count",
    "adapter_external_offset_half_range", "adapter_external_zero_probability", "adapter_external_paired",
)


def _apply_task_overrides(env_cfg, task):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg.

    Returns the list of applied "attr=value" strings (logged by the caller). Keys absent from the
    YAML are left at the code default; keys present whose target attribute is missing RAISE (so a
    stale/shadowed cfg or a broken Hydra composition can never silently swallow an override).
    """
    applied = []

    # This is a post-construction env-config switch.  The A3 env class applies
    # the default profile during __post_init__, so merely setting the string in
    # Hydra would leave the actual actuator gains unchanged.
    native_profile = _get(task, "native_actuator_profile")
    if native_profile is not None:
        _require(hasattr(env_cfg, "apply_native_actuator_profile"), "native_actuator_profile")
        env_cfg.apply_native_actuator_profile(str(native_profile))
        applied.append(f"native_actuator_profile={str(native_profile)!r}")

    # env base (num_envs is applied earlier via parse_env_cfg). Read every value through _get so the
    # logic works on both OmegaConf nodes (runtime) and plain dicts (unit tests).
    env = _get(task, "env")
    if env is not None:
        es = _get(env, "env_spacing")
        if es is not None:
            env_cfg.scene.env_spacing = float(es)
            applied.append(f"scene.env_spacing={float(es)}")
        els = _get(env, "episode_length_s")
        if els is not None:
            env_cfg.episode_length_s = float(els)
            applied.append(f"episode_length_s={float(els)}")
        staggered_half_span = _get(env, "staggered_stance_half_span_m")
        staggered_front_foot = _get(env, "staggered_front_foot")
        left_forward_offset = _get(env, "left_foot_forward_offset_m")
        lateral_widen_per_foot = _get(env, "stance_lateral_widen_per_foot_m")
        knee_flexion = _get(env, "stance_knee_flexion_rad")
        if staggered_half_span is not None and left_forward_offset is not None:
            raise ValueError(
                "Choose one stance transform: staggered_stance_half_span_m "
                "or left_foot_forward_offset_m"
            )
        if (lateral_widen_per_foot is not None or knee_flexion is not None) and staggered_half_span is None:
            raise ValueError(
                "stance_lateral_widen_per_foot_m and stance_knee_flexion_rad "
                "currently require staggered_stance_half_span_m"
            )
        if staggered_half_span is not None:
            if staggered_front_foot is None:
                staggered_front_foot = "left"
            staggered_front_foot = str(staggered_front_foot).strip().lower()
            _require(
                staggered_front_foot in {"left", "right"},
                "env.staggered_front_foot must be 'left' or 'right'",
            )
            front_sign = 1.0 if staggered_front_foot == "left" else -1.0
            custom_lateral_stance = lateral_widen_per_foot is not None
            custom_knee_flexion = knee_flexion is not None
            staggered_half_span = float(staggered_half_span)
            _require(
                0.0 <= staggered_half_span <= 0.10,
                "0 <= env.staggered_stance_half_span_m <= 0.10",
            )
            lateral_widen_per_foot = (
                0.0 if lateral_widen_per_foot is None else float(lateral_widen_per_foot)
            )
            _require(
                0.0 <= lateral_widen_per_foot <= 0.08,
                "0 <= env.stance_lateral_widen_per_foot_m <= 0.08",
            )
            base_hip_pitch = -0.160
            base_knee = 0.320
            base_ankle_pitch = -0.155
            base_foot_pitch = base_hip_pitch + base_knee + base_ankle_pitch
            nominal_knee = base_knee if knee_flexion is None else float(knee_flexion)
            _require(
                base_knee <= nominal_knee <= 0.60,
                "0.32 <= env.stance_knee_flexion_rad <= 0.60",
            )
            # Split extra flexion equally across hip and ankle while retaining
            # the validated nearly-flat sagittal foot orientation.
            nominal_hip_pitch = -0.5 * nominal_knee
            nominal_ankle_pitch = base_foot_pitch - nominal_hip_pitch - nominal_knee

            # Shift the feet fore/aft by equal amounts. A separate lateral
            # transform below widens the support without moving its center.
            thigh_length = 0.370
            shank_length = 0.415
            nominal_foot_pitch = base_foot_pitch
            a = thigh_length + shank_length * math.cos(nominal_knee)
            b = shank_length * math.sin(nominal_knee)
            radius = math.hypot(a, b)
            phase = math.atan2(b, a)

            def _foot_x(hip_pitch: float) -> float:
                return (
                    -thigh_length * math.sin(hip_pitch)
                    - shank_length * math.sin(hip_pitch + nominal_knee)
                )

            def _foot_z(hip_pitch: float) -> float:
                return (
                    -thigh_length * math.cos(hip_pitch)
                    - shank_length * math.cos(hip_pitch + nominal_knee)
                )

            def _hip_for_x(target_x: float) -> float:
                ratio = max(-1.0, min(1.0, -target_x / radius))
                primary = math.asin(ratio) - phase
                alternate = math.pi - math.asin(ratio) - phase
                candidates = (
                    primary,
                    alternate,
                    primary + 2.0 * math.pi,
                    primary - 2.0 * math.pi,
                    alternate + 2.0 * math.pi,
                    alternate - 2.0 * math.pi,
                )
                return min(candidates, key=lambda value: abs(value - nominal_hip_pitch))

            nominal_x = _foot_x(nominal_hip_pitch)
            left_hip_pitch = _hip_for_x(nominal_x + front_sign * staggered_half_span)
            right_hip_pitch = _hip_for_x(nominal_x - front_sign * staggered_half_span)
            left_ankle_pitch = nominal_foot_pitch - left_hip_pitch - nominal_knee
            right_ankle_pitch = nominal_foot_pitch - right_hip_pitch - nominal_knee
            baseline_z = (
                -thigh_length * math.cos(base_hip_pitch)
                - shank_length * math.cos(base_hip_pitch + base_knee)
            )
            mean_staggered_z = 0.5 * (
                _foot_z(left_hip_pitch) + _foot_z(right_hip_pitch)
            )

            # Increase hip roll symmetrically to add the requested lateral
            # distance per foot. Counter-rotate each ankle by the same delta
            # so the validated sole-roll orientation is preserved.
            base_hip_roll_abs = 0.080
            base_left_ankle_roll = -0.0078
            base_right_ankle_roll = 0.0078
            leg_height = max(0.40, -mean_staggered_z)
            widened_sin = math.sin(base_hip_roll_abs) + lateral_widen_per_foot / leg_height
            _require(widened_sin < 0.95, "requested lateral stance is reachable")
            hip_roll_abs = math.asin(widened_sin)
            hip_roll_delta = hip_roll_abs - base_hip_roll_abs
            left_ankle_roll = base_left_ankle_roll - hip_roll_delta
            right_ankle_roll = base_right_ankle_roll + hip_roll_delta

            # Lower the pelvis so deeper flexion and the wider roll transform
            # keep the feet on the same ground plane.
            baseline_vertical = baseline_z * math.cos(base_hip_roll_abs)
            staggered_vertical = mean_staggered_z * math.cos(hip_roll_abs)
            pelvis_height_delta = (
                baseline_vertical - staggered_vertical
                if custom_lateral_stance or custom_knee_flexion
                else baseline_z - mean_staggered_z
            )

            joint_pos = env_cfg.scene.robot.init_state.joint_pos
            joint_pos.pop(".*_hip_pitch_joint", None)
            joint_pos.pop(".*_knee_joint", None)
            joint_pos.pop(".*_ankle_pitch_joint", None)
            joint_pos.update(
                {
                    "left_hip_pitch_joint": left_hip_pitch,
                    "right_hip_pitch_joint": right_hip_pitch,
                    "left_knee_joint": nominal_knee,
                    "right_knee_joint": nominal_knee,
                    "left_ankle_pitch_joint": left_ankle_pitch,
                    "right_ankle_pitch_joint": right_ankle_pitch,
                    "left_hip_roll_joint": hip_roll_abs,
                    "right_hip_roll_joint": -hip_roll_abs,
                    "left_ankle_roll_joint": left_ankle_roll,
                    "right_ankle_roll_joint": right_ankle_roll,
                }
            )
            root_x, root_y, root_z = env_cfg.scene.robot.init_state.pos
            env_cfg.scene.robot.init_state.pos = (
                root_x,
                root_y,
                root_z + pelvis_height_delta,
            )
            applied.append(
                "staggered_stance("
                f"half_span={staggered_half_span:.3f}m,"
                f"front_foot={staggered_front_foot},"
                f"lateral_widen_per_foot={lateral_widen_per_foot:.3f}m,"
                f"knee={nominal_knee:.6f},"
                f"left_hip={left_hip_pitch:.6f},"
                f"right_hip={right_hip_pitch:.6f},"
                f"left_ankle={left_ankle_pitch:.6f},"
                f"right_ankle={right_ankle_pitch:.6f},"
                f"hip_roll_abs={hip_roll_abs:.6f},"
                f"left_ankle_roll={left_ankle_roll:.6f},"
                f"right_ankle_roll={right_ankle_roll:.6f},"
                f"pelvis_dz={pelvis_height_delta:.6f}m)"
            )
        if left_forward_offset is not None:
            left_forward_offset = float(left_forward_offset)
            _require(
                0.0 <= left_forward_offset <= 0.10,
                "0 <= env.left_foot_forward_offset_m <= 0.10",
            )
            # Move only the left foot forward while keeping its height and
            # pitch exact.  The right leg and pelvis remain at the validated
            # Stage-A ready state, so the support center moves forward without
            # introducing a right-leg out-of-distribution posture.
            thigh_length = 0.370
            shank_length = 0.415
            nominal_hip_pitch = -0.160
            nominal_knee = 0.320
            nominal_ankle_pitch = -0.155
            nominal_foot_pitch = (
                nominal_hip_pitch + nominal_knee + nominal_ankle_pitch
            )
            nominal_x = (
                -thigh_length * math.sin(nominal_hip_pitch)
                - shank_length * math.sin(nominal_hip_pitch + nominal_knee)
            )
            nominal_z = (
                -thigh_length * math.cos(nominal_hip_pitch)
                - shank_length * math.cos(nominal_hip_pitch + nominal_knee)
            )
            target_x = nominal_x + left_forward_offset
            distance_sq = target_x * target_x + nominal_z * nominal_z
            knee_cos = (
                distance_sq - thigh_length * thigh_length - shank_length * shank_length
            ) / (2.0 * thigh_length * shank_length)
            _require(abs(knee_cos) <= 1.0, "left-forward stance is reachable")
            left_knee = math.acos(max(-1.0, min(1.0, knee_cos)))
            a = thigh_length + shank_length * math.cos(left_knee)
            b = shank_length * math.sin(left_knee)
            theta = math.atan2(-target_x, -nominal_z)
            left_hip_pitch = theta - math.atan2(b, a)
            left_ankle_pitch = nominal_foot_pitch - left_hip_pitch - left_knee

            joint_pos = env_cfg.scene.robot.init_state.joint_pos
            joint_pos.pop(".*_hip_pitch_joint", None)
            joint_pos.pop(".*_knee_joint", None)
            joint_pos.pop(".*_ankle_pitch_joint", None)
            joint_pos.update(
                {
                    "left_hip_pitch_joint": left_hip_pitch,
                    "right_hip_pitch_joint": nominal_hip_pitch,
                    "left_knee_joint": left_knee,
                    "right_knee_joint": nominal_knee,
                    "left_ankle_pitch_joint": left_ankle_pitch,
                    "right_ankle_pitch_joint": nominal_ankle_pitch,
                }
            )
            applied.append(
                "left_forward_stance("
                f"offset={left_forward_offset:.3f}m,"
                f"left_hip={left_hip_pitch:.6f},"
                f"left_knee={left_knee:.6f},"
                f"left_ankle={left_ankle_pitch:.6f},"
                "right_leg=validated_ready)"
            )

    # sim base (control frequency = 1 / (dt * decimation))
    sim = _get(task, "sim")
    if sim is not None:
        dt = _get(sim, "dt")
        if dt is not None:
            env_cfg.sim.dt = float(dt)
            applied.append(f"sim.dt={float(dt)}")
        dec = _get(sim, "decimation")
        if dec is not None:
            env_cfg.decimation = int(dec)
            env_cfg.sim.render_interval = env_cfg.decimation  # keep render in step with decimation
            applied.append(f"decimation={int(dec)}")

    actions = _get(task, "actions")
    if actions is not None and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        raw_clip = _get(actions, "raw_clip")
        if raw_clip is not None:
            _require(hasattr(env_cfg.actions.joint_pos, "raw_clip"), "actions.joint_pos.raw_clip")
            env_cfg.actions.joint_pos.raw_clip = float(raw_clip)
            applied.append(f"actions.joint_pos.raw_clip={float(raw_clip)}")
        soft_limit_margin = _get(actions, "soft_limit_margin_frac")
        if soft_limit_margin is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "soft_limit_margin_frac"),
                "actions.joint_pos.soft_limit_margin_frac",
            )
            env_cfg.actions.joint_pos.soft_limit_margin_frac = float(soft_limit_margin)
            applied.append(f"actions.joint_pos.soft_limit_margin_frac={float(soft_limit_margin)}")
        residual_scale = _get(actions, "native_residual_scale")
        if residual_scale is not None:
            scale = getattr(env_cfg.actions.joint_pos, "scale", None)
            _require(scale is not None, "actions.joint_pos.scale")
            residual_scale = float(residual_scale)
            if isinstance(scale, dict):
                env_cfg.actions.joint_pos.scale = {k: float(v) * residual_scale for k, v in scale.items()}
            else:
                env_cfg.actions.joint_pos.scale = float(scale) * residual_scale
            applied.append(f"actions.joint_pos.scale*=native_residual_scale({residual_scale})")
        interpolate_reference = _get(actions, "interpolate_reference")
        if interpolate_reference is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "interpolate_reference"),
                "actions.joint_pos.interpolate_reference",
            )
            env_cfg.actions.joint_pos.interpolate_reference = bool(interpolate_reference)
            applied.append(
                f"actions.joint_pos.interpolate_reference={bool(interpolate_reference)}"
            )
        reference_lookahead_steps = _get(actions, "reference_lookahead_steps")
        if reference_lookahead_steps is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "reference_lookahead_steps"),
                "actions.joint_pos.reference_lookahead_steps",
            )
            reference_lookahead_steps = int(reference_lookahead_steps)
            _require(
                reference_lookahead_steps >= 0,
                "actions.joint_pos.reference_lookahead_steps >= 0",
            )
            env_cfg.actions.joint_pos.reference_lookahead_steps = reference_lookahead_steps
            applied.append(
                "actions.joint_pos.reference_lookahead_steps="
                f"{reference_lookahead_steps}"
            )
        joint_reference_lookahead_steps = _get(actions, "joint_reference_lookahead_steps")
        if joint_reference_lookahead_steps is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "joint_reference_lookahead_steps"),
                "actions.joint_pos.joint_reference_lookahead_steps",
            )
            env_cfg.actions.joint_pos.joint_reference_lookahead_steps = {
                str(name): float(value) for name, value in dict(joint_reference_lookahead_steps).items()
            }
            applied.append(
                "actions.joint_pos.joint_reference_lookahead_steps="
                f"{env_cfg.actions.joint_pos.joint_reference_lookahead_steps!r}"
            )
        velocity_ff_mode = _get(actions, "joint_velocity_feedforward_mode")
        if velocity_ff_mode is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "joint_velocity_feedforward_mode"),
                "actions.joint_pos.joint_velocity_feedforward_mode",
            )
            velocity_ff_mode = str(velocity_ff_mode)
            _require(
                velocity_ff_mode in {"none", "position_lead", "task_phase"},
                "actions.joint_velocity_feedforward_mode",
            )
            env_cfg.actions.joint_pos.joint_velocity_feedforward_mode = velocity_ff_mode
            applied.append(
                "actions.joint_pos.joint_velocity_feedforward_mode="
                f"{velocity_ff_mode!r}"
            )
        velocity_ff_beta = _get(actions, "joint_velocity_feedforward_beta")
        if velocity_ff_beta is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "joint_velocity_feedforward_beta"),
                "actions.joint_pos.joint_velocity_feedforward_beta",
            )
            velocity_ff_beta = float(velocity_ff_beta)
            _require(0.0 <= velocity_ff_beta <= 1.0, "0 <= actions.joint_velocity_feedforward_beta <= 1")
            env_cfg.actions.joint_pos.joint_velocity_feedforward_beta = velocity_ff_beta
            applied.append(
                "actions.joint_pos.joint_velocity_feedforward_beta="
                f"{velocity_ff_beta}"
            )
        velocity_ff_joints = _get(actions, "joint_velocity_feedforward_joint_names")
        if velocity_ff_joints is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "joint_velocity_feedforward_joint_names"),
                "actions.joint_pos.joint_velocity_feedforward_joint_names",
            )
            env_cfg.actions.joint_pos.joint_velocity_feedforward_joint_names = tuple(
                str(name) for name in velocity_ff_joints
            )
            applied.append(
                "actions.joint_pos.joint_velocity_feedforward_joint_names="
                f"{env_cfg.actions.joint_pos.joint_velocity_feedforward_joint_names!r}"
            )
        velocity_ff_decay_steps = _get(actions, "joint_velocity_feedforward_post_hit_decay_steps")
        if velocity_ff_decay_steps is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "joint_velocity_feedforward_post_hit_decay_steps"),
                "actions.joint_pos.joint_velocity_feedforward_post_hit_decay_steps",
            )
            velocity_ff_decay_steps = int(velocity_ff_decay_steps)
            _require(
                velocity_ff_decay_steps >= 1,
                "actions.joint_velocity_feedforward_post_hit_decay_steps >= 1",
            )
            env_cfg.actions.joint_pos.joint_velocity_feedforward_post_hit_decay_steps = velocity_ff_decay_steps
            applied.append(
                "actions.joint_pos.joint_velocity_feedforward_post_hit_decay_steps="
                f"{velocity_ff_decay_steps}"
            )
        microstep_enabled = _get(actions, "microstep_enabled")
        if microstep_enabled is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "microstep_enabled"),
                "actions.joint_pos.microstep_enabled",
            )
            env_cfg.actions.joint_pos.microstep_enabled = bool(microstep_enabled)
            applied.append(
                f"actions.joint_pos.microstep_enabled={bool(microstep_enabled)}"
            )
        microstep_limit = _get(actions, "microstep_step_limit_m")
        if microstep_limit is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "microstep_step_limit_m"),
                "actions.joint_pos.microstep_step_limit_m",
            )
            microstep_limit = float(microstep_limit)
            _require(
                0.005 <= microstep_limit <= 0.02,
                "0.005 <= actions.joint_pos.microstep_step_limit_m <= 0.02",
            )
            env_cfg.actions.joint_pos.microstep_step_limit_m = microstep_limit
            applied.append(
                f"actions.joint_pos.microstep_step_limit_m={microstep_limit}"
            )
        microstep_alpha = _get(actions, "microstep_lowpass_alpha")
        if microstep_alpha is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "microstep_lowpass_alpha"),
                "actions.joint_pos.microstep_lowpass_alpha",
            )
            microstep_alpha = float(microstep_alpha)
            _require(
                0.0 < microstep_alpha <= 1.0,
                "0 < actions.joint_pos.microstep_lowpass_alpha <= 1",
            )
            env_cfg.actions.joint_pos.microstep_lowpass_alpha = microstep_alpha
            applied.append(
                f"actions.joint_pos.microstep_lowpass_alpha={microstep_alpha}"
            )
        upper_prelude_release_steps = _get(actions, "upper_prelude_release_steps")
        if upper_prelude_release_steps is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "upper_prelude_release_steps"),
                "actions.joint_pos.upper_prelude_release_steps",
            )
            upper_prelude_release_steps = int(upper_prelude_release_steps)
            _require(
                upper_prelude_release_steps >= 0,
                "actions.joint_pos.upper_prelude_release_steps >= 0",
            )
            env_cfg.actions.joint_pos.upper_prelude_release_steps = upper_prelude_release_steps
            applied.append(
                "actions.joint_pos.upper_prelude_release_steps="
                f"{upper_prelude_release_steps}"
            )
        for key in (
            "phase_gate_tail_release_steps",
            "ready_hold_residual_release_steps",
            "upper_policy_tail_release_steps",
            "upper_policy_waist_post_hit_release_steps",
            "coordinator_arm_tail_release_steps",
            "waist_post_hit_settle_steps",
            "waist_post_hit_return_steps",
            "arm_tail_hold_steps",
            "arm_tail_return_steps",
            "waist_soft_limit_brake_lead_steps",
            "waist_soft_limit_prediction_horizon_steps",
            "upper_dynamic_soft_limit_prediction_horizon_steps",
            "stage_a_sagittal_exit_positive_confirm_steps",
            "stage_a_sagittal_exit_neutral_confirm_steps",
            "stage_a_sagittal_exit_decay_steps",
            "stage_a_sagittal_rearm_stable_steps",
            "stage_a_sagittal_rearm_ramp_steps",
        ):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            value = int(value)
            _require(value >= 0, f"actions.{key} >= 0")
            setattr(env_cfg.actions.joint_pos, key, value)
            applied.append(f"actions.joint_pos.{key}={value}")
        for key in (
            "stage_a_sagittal_exit_enabled",
            "stage_a_sagittal_exit_require_both_feet",
            "stage_a_sagittal_rearm_enabled",
            "stage_a_sagittal_rearm_require_ready_reference",
            "anchor_observation",
            "coordinator_external_observation",
            "coordinator_target_feedforward_enabled",
        ):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            setattr(env_cfg.actions.joint_pos, key, _as_bool(value))
            applied.append(f"actions.joint_pos.{key}={_as_bool(value)}")
        for key in (
            "stage_a_sagittal_exit_center_half_width_m",
            "stage_a_sagittal_exit_velocity_deadband_mps",
            "stage_a_sagittal_front_gain",
            "stage_a_sagittal_front_margin_m",
            "stage_a_sagittal_front_velocity_mps",
            "stage_a_sagittal_rearm_center_half_width_m",
            "stage_a_sagittal_rearm_velocity_max_mps",
            "stage_a_sagittal_rearm_pitch_rate_max_radps",
            "stage_a_sagittal_rearm_tilt_max_rad",
            "stage_a_sagittal_rearm_arm_error_max_rad",
            "stage_a_sagittal_rearm_arm_velocity_max_radps",
        ):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            value = float(value)
            _require(value >= 0.0, f"actions.{key} >= 0")
            setattr(env_cfg.actions.joint_pos, key, value)
            applied.append(f"actions.joint_pos.{key}={value}")
        waist_soft_limit_margin = _get(actions, "waist_soft_limit_margin_rad")
        if waist_soft_limit_margin is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "waist_soft_limit_margin_rad"),
                "actions.joint_pos.waist_soft_limit_margin_rad",
            )
            waist_soft_limit_margin = float(waist_soft_limit_margin)
            _require(waist_soft_limit_margin >= 0.0, "actions.waist_soft_limit_margin_rad >= 0")
            env_cfg.actions.joint_pos.waist_soft_limit_margin_rad = waist_soft_limit_margin
            applied.append(
                "actions.joint_pos.waist_soft_limit_margin_rad="
                f"{waist_soft_limit_margin}"
            )
        waist_margins_by_motion = _get(
            actions, "waist_soft_limit_margin_rad_by_motion"
        )
        if waist_margins_by_motion is not None:
            _require(
                hasattr(
                    env_cfg.actions.joint_pos,
                    "waist_soft_limit_margin_rad_by_motion",
                ),
                "actions.joint_pos.waist_soft_limit_margin_rad_by_motion",
            )
            values = tuple(float(item) for item in waist_margins_by_motion)
            _require(values and all(item > 0.0 for item in values),
                     "actions.waist_soft_limit_margin_rad_by_motion values > 0")
            env_cfg.actions.joint_pos.waist_soft_limit_margin_rad_by_motion = values
            applied.append(
                "actions.joint_pos.waist_soft_limit_margin_rad_by_motion="
                f"{values}"
            )
        waist_soft_limit_velocity_brake_gain = _get(actions, "waist_soft_limit_velocity_brake_gain")
        if waist_soft_limit_velocity_brake_gain is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "waist_soft_limit_velocity_brake_gain"),
                "actions.joint_pos.waist_soft_limit_velocity_brake_gain",
            )
            waist_soft_limit_velocity_brake_gain = float(waist_soft_limit_velocity_brake_gain)
            _require(
                waist_soft_limit_velocity_brake_gain >= 0.0,
                "actions.waist_soft_limit_velocity_brake_gain >= 0",
            )
            env_cfg.actions.joint_pos.waist_soft_limit_velocity_brake_gain = waist_soft_limit_velocity_brake_gain
            applied.append(
                "actions.joint_pos.waist_soft_limit_velocity_brake_gain="
                f"{waist_soft_limit_velocity_brake_gain}"
            )
        for key in ("waist_soft_limit_guard_in_prelude", "waist_soft_limit_enforce_inner_limit"):
            value = _get(actions, key)
            if value is not None:
                _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
                setattr(env_cfg.actions.joint_pos, key, bool(value))
                applied.append(f"actions.joint_pos.{key}={bool(value)}")
        dynamic_joint_names = _get(actions, "upper_dynamic_soft_limit_joint_names")
        dynamic_margins = _get(actions, "upper_dynamic_soft_limit_margin_rad")
        if dynamic_joint_names is not None or dynamic_margins is not None:
            _require(
                dynamic_joint_names is not None and dynamic_margins is not None,
                "actions.upper_dynamic_soft_limit_joint_names and "
                "actions.upper_dynamic_soft_limit_margin_rad must be set together",
            )
            _require(
                hasattr(env_cfg.actions.joint_pos, "upper_dynamic_soft_limit_joint_names"),
                "actions.joint_pos.upper_dynamic_soft_limit_joint_names",
            )
            names = tuple(str(item) for item in dynamic_joint_names)
            margins = tuple(float(item) for item in dynamic_margins)
            _require(len(names) == len(margins), "actions.dynamic guard lengths match")
            _require(all(item > 0.0 for item in margins), "actions.dynamic margins > 0")
            env_cfg.actions.joint_pos.upper_dynamic_soft_limit_joint_names = names
            env_cfg.actions.joint_pos.upper_dynamic_soft_limit_margin_rad = margins
            applied.append(
                "actions.joint_pos.upper_dynamic_soft_limit_joint_names=" f"{names}"
            )
            applied.append(
                "actions.joint_pos.upper_dynamic_soft_limit_margin_rad=" f"{margins}"
            )
        recovery_ready_offsets = _get(
            actions, "upper_joint_recovery_ready_offset_rad"
        )
        if recovery_ready_offsets is not None:
            _require(
                hasattr(
                    env_cfg.actions.joint_pos,
                    "upper_joint_recovery_ready_offset_rad",
                ),
                "actions.joint_pos.upper_joint_recovery_ready_offset_rad",
            )
            offsets = {
                str(name): float(value)
                for name, value in dict(recovery_ready_offsets).items()
            }
            env_cfg.actions.joint_pos.upper_joint_recovery_ready_offset_rad = offsets
            applied.append(
                "actions.joint_pos.upper_joint_recovery_ready_offset_rad="
                f"{offsets!r}"
            )
        recovery_ready_offsets_by_motion = _get(
            actions, "upper_joint_recovery_ready_offset_rad_by_motion"
        )
        if recovery_ready_offsets_by_motion is not None:
            _require(
                hasattr(
                    env_cfg.actions.joint_pos,
                    "upper_joint_recovery_ready_offset_rad_by_motion",
                ),
                "actions.joint_pos.upper_joint_recovery_ready_offset_rad_by_motion",
            )
            offsets_by_motion = {
                str(name): tuple(float(item) for item in values)
                for name, values in dict(recovery_ready_offsets_by_motion).items()
            }
            _require(
                all(values for values in offsets_by_motion.values()),
                "actions.motion-conditioned recovery offsets must be non-empty",
            )
            env_cfg.actions.joint_pos.upper_joint_recovery_ready_offset_rad_by_motion = (
                offsets_by_motion
            )
            applied.append(
                "actions.joint_pos.upper_joint_recovery_ready_offset_rad_by_motion="
                f"{offsets_by_motion!r}"
            )
        passive_joint_margins = _get(
            actions, "passive_joint_soft_limit_margin_rad"
        )
        if passive_joint_margins is not None:
            _require(
                hasattr(
                    env_cfg.actions.joint_pos,
                    "passive_joint_soft_limit_margin_rad",
                ),
                "actions.joint_pos.passive_joint_soft_limit_margin_rad",
            )
            passive_margins = {
                str(name): float(value)
                for name, value in dict(passive_joint_margins).items()
            }
            _require(
                passive_margins
                and all(value > 0.0 for value in passive_margins.values()),
                "actions.passive_joint_soft_limit_margin_rad values > 0",
            )
            env_cfg.actions.joint_pos.passive_joint_soft_limit_margin_rad = (
                passive_margins
            )
            applied.append(
                "actions.joint_pos.passive_joint_soft_limit_margin_rad="
                f"{passive_margins!r}"
            )
        for key in (
            "upper_dynamic_soft_limit_velocity_brake_gain",
            "upper_dynamic_soft_limit_position_brake_gain",
            "upper_dynamic_soft_limit_max_position_correction_rad",
            "upper_dynamic_soft_limit_max_velocity_correction_radps",
        ):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            value = float(value)
            if key.startswith("upper_dynamic_soft_limit_max_"):
                _require(value > 0.0, f"actions.{key} > 0")
            else:
                _require(value >= 0.0, f"actions.{key} >= 0")
            setattr(env_cfg.actions.joint_pos, key, value)
            applied.append(f"actions.joint_pos.{key}={value}")
        for key, expected_length in (
            ("coordinator_leg_correction_scale_rad", 12),
            ("coordinator_waist_correction_scale_rad", 3),
            ("coordinator_arm_correction_scale_rad", 7),
        ):
            value = _get(actions, key)
            if value is None:
                continue
            attr = key.removeprefix("coordinator_")
            _require(hasattr(env_cfg.actions.joint_pos, attr), f"actions.joint_pos.{attr}")
            values = tuple(float(item) for item in value)
            _require(len(values) == expected_length, f"actions.{key} length == {expected_length}")
            _require(all(item > 0.0 for item in values), f"actions.{key} values > 0")
            setattr(env_cfg.actions.joint_pos, attr, values)
            applied.append(f"actions.joint_pos.{attr}={values}")
        for key, expected_length in (("adapter_scale_rad", 7),):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            values = tuple(float(item) for item in value)
            _require(len(values) == expected_length, f"actions.{key} length == {expected_length}")
            _require(all(item > 0.0 for item in values), f"actions.{key} values > 0")
            setattr(env_cfg.actions.joint_pos, key, values)
            applied.append(f"actions.joint_pos.{key}={values}")
        for key in (
            "coordinator_target_feedforward_raw_clip",
            "adapter_raw_clip",
            "adapter_ramp_in_steps",
            "adapter_ramp_out_steps",
            "adapter_policy_residual_gain",
        ):
            value = _get(actions, key)
            if value is None:
                continue
            _require(hasattr(env_cfg.actions.joint_pos, key), f"actions.joint_pos.{key}")
            value = (
                float(value)
                if key in {
                    "coordinator_target_feedforward_raw_clip",
                    "adapter_raw_clip",
                    "adapter_policy_residual_gain",
                }
                else int(value)
            )
            _require(value >= 0.0, f"actions.{key} >= 0")
            setattr(env_cfg.actions.joint_pos, key, value)
            applied.append(f"actions.joint_pos.{key}={value}")
        coordinator_target_feedforward = _get(
            actions, "coordinator_target_feedforward_by_motion"
        )
        if coordinator_target_feedforward is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "coordinator_target_feedforward_by_motion"),
                "actions.joint_pos.coordinator_target_feedforward_by_motion",
            )
            matrices = tuple(
                tuple(tuple(float(value) for value in row) for row in matrix)
                for matrix in coordinator_target_feedforward
            )
            _require(len(matrices) > 0, "actions.coordinator_target_feedforward_by_motion nonempty")
            _require(
                all(len(matrix) == 22 and all(len(row) == 3 for row in matrix) for matrix in matrices),
                "actions.coordinator_target_feedforward_by_motion shape == (M, 22, 3)",
            )
            setattr(
                env_cfg.actions.joint_pos,
                "coordinator_target_feedforward_by_motion",
                matrices,
            )
            applied.append(
                "actions.joint_pos.coordinator_target_feedforward_by_motion="
                f"{len(matrices)}x22x3"
            )
        upper_checkpoint = _get(actions, "upper_checkpoint")
        if upper_checkpoint is not None:
            _require(hasattr(env_cfg.actions.joint_pos, "upper_checkpoint"), "actions.joint_pos.upper_checkpoint")
            env_cfg.actions.joint_pos.upper_checkpoint = str(upper_checkpoint)
            applied.append(f"actions.joint_pos.upper_checkpoint={str(upper_checkpoint)!r}")
        legacy_stage_a_checkpoint = _get(actions, "legacy_stage_a_checkpoint")
        if legacy_stage_a_checkpoint is not None:
            _require(
                hasattr(env_cfg.actions.joint_pos, "legacy_stage_a_checkpoint"),
                "actions.joint_pos.legacy_stage_a_checkpoint",
            )
            env_cfg.actions.joint_pos.legacy_stage_a_checkpoint = str(legacy_stage_a_checkpoint)
            applied.append(
                "actions.joint_pos.legacy_stage_a_checkpoint="
                f"{str(legacy_stage_a_checkpoint)!r}"
            )
        scale_multipliers = _get(actions, "native_joint_scale_multipliers")
        if scale_multipliers is not None:
            scale = getattr(env_cfg.actions.joint_pos, "scale", None)
            _require(isinstance(scale, dict), "actions.joint_pos.scale dict for native_joint_scale_multipliers")
            for name, multiplier in scale_multipliers.items():
                _require(name in scale, f"actions.joint_pos.scale['{name}']")
                scale[name] = float(scale[name]) * float(multiplier)
                applied.append(f"actions.joint_pos.scale[{name}]*={float(multiplier)}")

    rw = _get(task, "rewards")
    if rw is not None:
        R = env_cfg.rewards
        _set_reward(
            R,
            "racket_incremental_position",
            _get(rw, "racket_incremental_position_weight"),
            _get(rw, "racket_incremental_position_std"),
            applied,
        )
        _set_reward(
            R,
            "racket_incremental_direction",
            _get(rw, "racket_incremental_direction_weight"),
            None,
            applied,
        )
        _set_reward(
            R,
            "racket_incremental_dense_huber",
            _get(rw, "racket_incremental_dense_huber_weight"),
            None,
            applied,
        )
        _set_reward(
            R,
            "racket_incremental_gain",
            _get(rw, "racket_incremental_gain_weight"),
            None,
            applied,
        )
        _set_reward(
            R,
            "racket_incremental_cross_axis",
            _get(rw, "racket_incremental_cross_axis_weight"),
            None,
            applied,
        )
        _set_reward(
            R,
            "target_adapter_zero_hold",
            _get(rw, "target_adapter_zero_hold_weight"),
            None,
            applied,
        )
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"), applied)
        _set_reward(R, "racket_position_y", _get(rw, "racket_position_y_weight"), _get(rw, "racket_position_y_std"), applied)
        _set_reward(R, "racket_position_fine", _get(rw, "racket_position_fine_weight"), _get(rw, "racket_position_fine_std"), applied)
        _set_reward(R, "racket_position_y_fine", _get(rw, "racket_position_y_fine_weight"), _get(rw, "racket_position_y_fine_std"), applied)
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"), applied)
        # P5U causal reward ablation terms.  These are deliberately explicit
        # YAML overrides so R0/R1/R2/R3 can be replayed without editing the
        # environment code or silently changing the reward contract.
        for _name, _key in (
            ("racket_velocity_magnitude", "racket_velocity_magnitude_weight"),
            ("racket_velocity_direction", "racket_velocity_direction_weight"),
            ("racket_signed_velocity", "racket_signed_velocity_weight"),
            ("racket_pass_through", "racket_pass_through_weight"),
            ("racket_stop_at_target", "racket_stop_at_target_weight"),
            ("racket_reverse_motion", "racket_reverse_motion_weight"),
            ("racket_hit_timing", "racket_hit_timing_weight"),
            ("phase_magnitude", "phase_magnitude_weight"),
            ("phase_rate", "phase_rate_weight"),
            ("phase_group_consistency", "phase_group_consistency_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                _require(getattr(R, _name) is not None, f"rewards.{_name} (term is disabled/None)")
                getattr(R, _name).weight = float(_w)
                applied.append(f"rewards.{_name}.weight={float(_w)}")
        gated_velocity_weight = _get(rw, "racket_velocity_position_gated_weight")
        if gated_velocity_weight is not None:
            _require(
                hasattr(R, "racket_velocity_position_gated"),
                "rewards.racket_velocity_position_gated",
            )
            R.racket_velocity_position_gated.weight = float(gated_velocity_weight)
            applied.append(
                "rewards.racket_velocity_position_gated.weight="
                f"{float(gated_velocity_weight)}"
            )
        gated_velocity = _get(rw, "racket_velocity_position_gated")
        if gated_velocity is not None:
            _require(
                hasattr(R, "racket_velocity_position_gated"),
                "rewards.racket_velocity_position_gated",
            )
            for key in ("velocity_std", "position_threshold", "position_excess_std"):
                val = _get(gated_velocity, key)
                if val is not None:
                    R.racket_velocity_position_gated.params[key] = float(val)
                    applied.append(
                        "rewards.racket_velocity_position_gated.params."
                        f"{key}={float(val)}"
                    )
        _set_reward(R, "racket_normal", _get(rw, "racket_normal_weight"), _get(rw, "racket_normal_std"), applied)
        _set_reward(R, "racket_hit_coupled", _get(rw, "racket_hit_coupled_weight"), None, applied)
        if hasattr(R, "racket_hit_coupled") and R.racket_hit_coupled is not None:
            coupled = _get(rw, "racket_hit_coupled")
            if coupled is not None:
                for key in ("pos_std", "vel_std", "normal_std", "base", "vel_coeff", "normal_coeff"):
                    val = _get(coupled, key)
                    if val is not None:
                        R.racket_hit_coupled.params[key] = float(val)
                        applied.append(f"rewards.racket_hit_coupled.params.{key}={float(val)}")
        _set_reward(R, "racket_hit_precision", _get(rw, "racket_hit_precision_weight"), None, applied)
        if hasattr(R, "racket_hit_precision") and R.racket_hit_precision is not None:
            precision = _get(rw, "racket_hit_precision")
            if precision is not None:
                for key in ("pos_std", "vel_std", "normal_std", "time_std"):
                    val = _get(precision, key)
                    if val is not None:
                        R.racket_hit_precision.params[key] = float(val)
                        applied.append(f"rewards.racket_hit_precision.params.{key}={float(val)}")
        _set_reward(R, "base_position", _get(rw, "base_position_weight"), _get(rw, "base_position_std"), applied)
        jt = _get(rw, "joint_torques_weight")
        if jt is not None:
            _require(hasattr(R, "joint_torques"), "rewards.joint_torques")
            R.joint_torques.weight = float(jt)
            applied.append(f"rewards.joint_torques.weight={float(jt)}")

        # --- motion imitation prior (the 6 motion_* terms; base weights sum ~5.0) ---------------
        # `motion_scale` multiplies all six at once — the main lever to demote imitation to a soft
        # prior so the racket goal can dominate. Per-term weight/std overrides are also accepted
        # (e.g. motion_body_pos_weight / motion_body_pos_std) and are applied BEFORE the scale.
        _MOTION_TERMS = (
            "motion_global_anchor_pos", "motion_global_anchor_ori",
            "motion_body_pos", "motion_body_ori",
            "motion_torso_ori", "motion_native_joint_pos",
            "motion_body_lin_vel", "motion_body_ang_vel",
        )
        for _t in _MOTION_TERMS:
            _set_reward(R, _t, _get(rw, f"{_t}_weight"), _get(rw, f"{_t}_std"), applied)
        ms = _get(rw, "motion_scale")
        if ms is not None:
            ms = float(ms)
            for _t in _MOTION_TERMS:
                _require(hasattr(R, _t), f"rewards.{_t}")
                _term = getattr(R, _t)
                if _term is not None:
                    _term.weight *= ms
            applied.append(f"rewards.motion_scale={ms} (enabled motion weights only)")

        # --- penalties / regularization (negative weights: energy + smoothness + safety) --------
        for _name, _key in (
            ("action_rate_l2", "action_rate_weight"),
            ("action_residual_l2", "action_residual_weight"),
            ("action_execution_gap", "action_execution_gap_weight"),
            ("joint_limit", "joint_limit_weight"),
            ("undesired_contacts", "undesired_contacts_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                getattr(R, _name).weight = float(_w)
                applied.append(f"rewards.{_name}.weight={float(_w)}")

        for _name, _key in (
            ("post_strike_root_tilt_l2", "post_strike_root_tilt_weight"),
            ("post_strike_recovery_progress", "post_strike_recovery_progress_weight"),
            ("pre_hit_root_tilt_l2", "pre_hit_root_tilt_weight"),
            ("pre_hit_root_angular_velocity_l2", "pre_hit_root_angular_velocity_l2_weight"),
            ("pre_hit_root_forward_velocity_l2", "pre_hit_root_forward_velocity_l2_weight"),
            ("strike_approach_pitch_rate_deadband_l2", "strike_approach_pitch_rate_deadband_l2_weight"),
            (
                "strike_approach_forward_velocity_deadband_l2",
                "strike_approach_forward_velocity_deadband_l2_weight",
            ),
            ("exact_strike_pitch_rate_deadband_l2", "exact_strike_pitch_rate_deadband_l2_weight"),
            (
                "exact_strike_forward_velocity_deadband_l2",
                "exact_strike_forward_velocity_deadband_l2_weight",
            ),
            (
                "post_hit_forward_velocity_deadband_l2",
                "post_hit_forward_velocity_deadband_l2_weight",
            ),
            (
                "post_hit_pitch_rate_deadband_l2",
                "post_hit_pitch_rate_deadband_l2_weight",
            ),
            (
                "post_hit_capture_point_center_l2",
                "post_hit_capture_point_center_l2_weight",
            ),
            (
                "post_hit_capture_point_barrier_l2",
                "post_hit_capture_point_barrier_l2_weight",
            ),
            (
                "post_hit_capture_point_center_progress",
                "post_hit_capture_point_center_progress_weight",
            ),
            ("post_strike_root_linear_velocity", "post_strike_root_linear_velocity_weight"),
            ("post_strike_root_linear_velocity_l2", "post_strike_root_linear_velocity_l2_weight"),
            ("post_strike_root_angular_velocity_l2", "post_strike_root_angular_velocity_l2_weight"),
            ("post_strike_root_height_deficit", "post_strike_root_height_deficit_weight"),
            ("post_strike_both_feet_contact", "post_strike_both_feet_contact_weight"),
            ("post_strike_ready", "post_strike_ready_weight"),
            ("fall", "fall_weight"),
            ("alive", "alive_weight"),
            ("root_position_drift", "root_position_drift_weight"),
            ("feet_slip", "feet_slip_weight"),
            ("stagger_capture_point_margin_l2", "stagger_capture_point_margin_l2_weight"),
            (
                "stagger_lateral_capture_point_margin_l2",
                "stagger_lateral_capture_point_margin_l2_weight",
            ),
            ("stagger_minimum_foot_load", "stagger_minimum_foot_load_weight"),
            ("stagger_sagittal_span_l2", "stagger_sagittal_span_l2_weight"),
            ("stagger_lateral_span_l2", "stagger_lateral_span_l2_weight"),
            ("coordinator_leg_l2", "coordinator_leg_l2_weight"),
            ("coordinator_waist_l2", "coordinator_waist_l2_weight"),
            ("coordinator_arm_l2", "coordinator_arm_l2_weight"),
        ):
            _w = _get(rw, _key)
            if _w is not None:
                _require(hasattr(R, _name), f"rewards.{_name}")
                _require(getattr(R, _name) is not None, f"rewards.{_name} (term is disabled/None)")
                getattr(R, _name).weight = float(_w)
                applied.append(f"rewards.{_name}.weight={float(_w)}")

    rk = _get(task, "racket")
    if rk is not None:
        # Only require the racket_target command when the YAML actually sets racket keys, so tasks
        # without a racket objective (e.g. TrackingFlat, which has no `racket:` block) never trip this.
        provided = [k for k in _RACKET_KEYS if _get(rk, k) is not None]
        if provided:
            _require(hasattr(env_cfg.commands, "racket_target"),
                     f"commands.racket_target (task YAML sets racket keys {provided})")
            C = env_cfg.commands.racket_target
            _set_attr(C, "strike_phase", _get(rk, "strike_phase"), float, applied, "racket_target")
            _set_attr(C, "strike_window_s", _get(rk, "strike_window_s"), float, applied, "racket_target")
            _set_attr(C, "strike_time_std_s", _get(rk, "strike_time_std_s"), float, applied, "racket_target")
            _set_attr(C, "strike_success_pos_thresh", _get(rk, "strike_success_pos_thresh"), float, applied, "racket_target")
            _set_attr(C, "strike_success_vel_thresh", _get(rk, "strike_success_vel_thresh"), float, applied, "racket_target")
            _set_attr(C, "strike_success_normal_thresh_deg", _get(rk, "strike_success_normal_thresh_deg"), float, applied, "racket_target")

            _set_range(C, "racket_pos_x_range", _get(rk, "pos_x_range"), applied, "racket_target")
            _set_range(C, "racket_pos_y_range", _get(rk, "pos_y_range"), applied, "racket_target")
            _set_range(C, "racket_pos_z_range", _get(rk, "pos_z_range"), applied, "racket_target")
            _set_range(C, "racket_vel_x_range", _get(rk, "vel_x_range"), applied, "racket_target")
            _set_range(C, "racket_vel_y_range", _get(rk, "vel_y_range"), applied, "racket_target")
            _set_range(C, "racket_vel_z_range", _get(rk, "vel_z_range"), applied, "racket_target")
            _set_range(C, "base_target_x_range", _get(rk, "base_target_x_range"), applied, "racket_target")
            _set_range(C, "base_target_y_range", _get(rk, "base_target_y_range"), applied, "racket_target")
            _set_attr(C, "normal_mode", _get(rk, "normal_mode"), str, applied, "racket_target")
            _set_attr(C, "forehand_on_negative_y", _get(rk, "forehand_on_negative_y"), _as_bool, applied, "racket_target")
            _set_attr(C, "mount_normal_axis", _get(rk, "mount_normal_axis"), int, applied, "racket_target")
            _set_attr(C, "mount_normal_sign", _get(rk, "mount_normal_sign"), float, applied, "racket_target")
            # reference_perturbed target sampling (rank 5): couple targets to the reference swing.
            _set_attr(C, "target_mode", _get(rk, "target_mode"), str, applied, "racket_target")
            _set_vec3(C, "ref_perturb_pos", _get(rk, "ref_perturb_pos"), applied, "racket_target")
            _set_vec3(C, "ref_perturb_vel", _get(rk, "ref_perturb_vel"), applied, "racket_target")
            _set_attr(C, "ref_perturb_normal", _get(rk, "ref_perturb_normal"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_steps", _get(rk, "ref_perturb_curriculum_steps"), int, applied, "racket_target")
            _set_attr(C, "ref_perturb_curriculum_start", _get(rk, "ref_perturb_curriculum_start"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_success_gated", _get(rk, "ref_perturb_success_gated"), _as_bool, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_threshold", _get(rk, "ref_perturb_advance_threshold"), float, applied, "racket_target")
            _set_attr(C, "ref_perturb_advance_rate", _get(rk, "ref_perturb_advance_rate"), float, applied, "racket_target")
            _set_attr(C, "exact_success_decay", _get(rk, "exact_success_decay"), float, applied, "racket_target")
            _set_attr(C, "exact_success_min_count", _get(rk, "exact_success_min_count"), float, applied, "racket_target")
            _set_vec3(
                C,
                "adapter_external_offset_half_range",
                _get(rk, "adapter_external_offset_half_range"),
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "adapter_external_zero_probability",
                _get(rk, "adapter_external_zero_probability"),
                float,
                applied,
                "racket_target",
            )
            _set_attr(
                C,
                "adapter_external_paired",
                _get(rk, "adapter_external_paired"),
                _as_bool,
                applied,
                "racket_target",
            )

    tracker = _get(task, "tracker")
    if tracker is not None and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        A = env_cfg.actions.joint_pos
        scale = _get(tracker, "upper_correction_scale_rad")
        if scale is not None and hasattr(A, "upper_correction_scale_rad"):
            value = float(scale)
            _require(value > 0.0, "tracker.upper_correction_scale_rad > 0")
            A.upper_correction_scale_rad = (value,) * len(A.upper_correction_scale_rad)
            applied.append(f"actions.joint_pos.upper_correction_scale_rad={value}")

    terminations = _get(task, "terminations")
    if terminations is not None:
        recovery_tilt_max_deg = _get(terminations, "recovery_tilt_max_deg")
        recovery_tilt_required_steps = _get(terminations, "recovery_tilt_required_steps")
        if recovery_tilt_max_deg is not None or recovery_tilt_required_steps is not None:
            _require(hasattr(env_cfg.terminations, "recovery_tilt"), "terminations.recovery_tilt")
            recovery_tilt = env_cfg.terminations.recovery_tilt
            _require(recovery_tilt is not None, "terminations.recovery_tilt (term is disabled/None)")
            if recovery_tilt_max_deg is not None:
                recovery_tilt_max_deg = float(recovery_tilt_max_deg)
                _require(0.0 < recovery_tilt_max_deg < 90.0, "terminations.recovery_tilt_max_deg in (0, 90)")
                recovery_tilt.params["max_tilt_rad"] = recovery_tilt_max_deg * 3.141592653589793 / 180.0
                applied.append(
                    "terminations.recovery_tilt.max_tilt_rad="
                    f"{recovery_tilt.params['max_tilt_rad']}"
                )
            if recovery_tilt_required_steps is not None:
                recovery_tilt_required_steps = int(recovery_tilt_required_steps)
                _require(recovery_tilt_required_steps >= 1, "terminations.recovery_tilt_required_steps >= 1")
                recovery_tilt.params["required_steps"] = recovery_tilt_required_steps
                applied.append(
                    "terminations.recovery_tilt.required_steps="
                    f"{recovery_tilt_required_steps}"
                )

    # Domain randomization: behaviour preserved exactly (the pd_gain "absent/null -> disable" semantics
    # are intentional). Only logging is added; the hasattr guards stay so DR stays optional per task.
    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        if mr is not None and hasattr(E, "randomize_link_mass") and E.randomize_link_mass is not None:
            E.randomize_link_mass.params["mass_distribution_params"] = (float(mr[0]), float(mr[1]))
            applied.append(f"events.randomize_link_mass.mass_distribution_params=({float(mr[0])}, {float(mr[1])})")
        if hasattr(E, "randomize_pd_gains"):
            pr = _get(dr, "pd_gain_range")
            if pr is None:
                E.randomize_pd_gains = None  # disable
                applied.append("events.randomize_pd_gains=None(disabled)")
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))
                applied.append(f"events.randomize_pd_gains=({float(pr[0])}, {float(pr[1])})")

    motion = _get(task, "motion")
    if motion is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        C = env_cfg.commands.motion
        fixed_motion_id = _get(motion, "fixed_motion_id")
        if fixed_motion_id is not None:
            _require(hasattr(C, "fixed_motion_id"), "commands.motion.fixed_motion_id")
            C.fixed_motion_id = int(fixed_motion_id)
            _require(C.fixed_motion_id >= 0, "motion.fixed_motion_id >= 0")
            applied.append(f"commands.motion.fixed_motion_id={C.fixed_motion_id}")
        manifest_balance_strokes = _get(motion, "manifest_balance_strokes")
        if manifest_balance_strokes is not None:
            _require(hasattr(C, "manifest_balance_strokes"), "commands.motion.manifest_balance_strokes")
            C.manifest_balance_strokes = _as_bool(manifest_balance_strokes)
            applied.append(
                f"commands.motion.manifest_balance_strokes={C.manifest_balance_strokes}"
            )
        reference_sampling_mode = _get(motion, "reference_sampling_mode")
        if reference_sampling_mode is not None:
            _require(hasattr(C, "reference_sampling_mode"), "commands.motion.reference_sampling_mode")
            mode = str(reference_sampling_mode).strip().lower()
            _require(mode in {"uniform", "balanced_by_region", "difficulty_weighted", "curriculum"}, "motion.reference_sampling_mode is valid")
            C.reference_sampling_mode = mode
            applied.append(f"commands.motion.reference_sampling_mode={mode}")
        reference_curriculum_stage = _get(motion, "reference_curriculum_stage")
        if reference_curriculum_stage is not None:
            _require(hasattr(C, "reference_curriculum_stage"), "commands.motion.reference_curriculum_stage")
            C.reference_curriculum_stage = int(reference_curriculum_stage)
            _require(C.reference_curriculum_stage >= 0, "motion.reference_curriculum_stage >= 0")
            applied.append(f"commands.motion.reference_curriculum_stage={C.reference_curriculum_stage}")
        reference_curriculum_sizes = _get(motion, "reference_curriculum_sizes")
        if reference_curriculum_sizes is not None:
            _require(hasattr(C, "reference_curriculum_sizes"), "commands.motion.reference_curriculum_sizes")
            C.reference_curriculum_sizes = tuple(int(x) for x in reference_curriculum_sizes)
            _require(all(x > 0 for x in C.reference_curriculum_sizes), "motion.reference_curriculum_sizes > 0")
            applied.append(f"commands.motion.reference_curriculum_sizes={C.reference_curriculum_sizes}")
        for key in ("prelude_steps", "prelude_settle_steps", "prelude_launch_steps"):
            value = _get(motion, key)
            if value is not None:
                _require(hasattr(C, key), f"commands.motion.{key}")
                value = int(value)
                _require(value >= 0, f"motion.{key} >= 0")
                setattr(C, key, value)
                applied.append(f"commands.motion.{key}={value}")
        prelude_minimum_jerk = _get(motion, "prelude_minimum_jerk")
        if prelude_minimum_jerk is not None:
            _require(hasattr(C, "prelude_minimum_jerk"), "commands.motion.prelude_minimum_jerk")
            C.prelude_minimum_jerk = bool(prelude_minimum_jerk)
            applied.append(f"commands.motion.prelude_minimum_jerk={C.prelude_minimum_jerk}")
        prelude_quintic_hermite = _get(motion, "prelude_quintic_hermite")
        if prelude_quintic_hermite is not None:
            _require(
                hasattr(C, "prelude_quintic_hermite"),
                "commands.motion.prelude_quintic_hermite",
            )
            C.prelude_quintic_hermite = _as_bool(prelude_quintic_hermite)
            applied.append(
                "commands.motion.prelude_quintic_hermite="
                f"{C.prelude_quintic_hermite}"
            )
        ready_joint_positions = _get(motion, "ready_joint_positions")
        if ready_joint_positions is not None:
            _require(
                hasattr(C, "ready_joint_positions"),
                "commands.motion.ready_joint_positions",
            )
            C.ready_joint_positions = {
                str(name): float(value)
                for name, value in ready_joint_positions.items()
            }
            _require(
                bool(C.ready_joint_positions),
                "motion.ready_joint_positions must not be empty when provided",
            )
            applied.append(
                "commands.motion.ready_joint_positions="
                f"{C.ready_joint_positions}"
            )
        prelude_continuous_velocity = _get(motion, "prelude_continuous_velocity_reference")
        if prelude_continuous_velocity is not None:
            _require(
                hasattr(C, "prelude_continuous_velocity_reference"),
                "commands.motion.prelude_continuous_velocity_reference",
            )
            C.prelude_continuous_velocity_reference = bool(prelude_continuous_velocity)
            applied.append(
                "commands.motion.prelude_continuous_velocity_reference="
                f"{C.prelude_continuous_velocity_reference}"
            )
        prelude_waist_anchor = _get(motion, "prelude_waist_pitch_anchor_rad")
        if prelude_waist_anchor is not None:
            _require(
                hasattr(C, "prelude_waist_pitch_anchor_rad"),
                "commands.motion.prelude_waist_pitch_anchor_rad",
            )
            C.prelude_waist_pitch_anchor_rad = float(prelude_waist_anchor)
            applied.append(
                "commands.motion.prelude_waist_pitch_anchor_rad="
                f"{C.prelude_waist_pitch_anchor_rad}"
            )
        pose_range = _get(motion, "pose_range")
        if pose_range is not None:
            C.pose_range = {str(k): (float(v[0]), float(v[1])) for k, v in pose_range.items()}
            applied.append(f"commands.motion.pose_range={C.pose_range}")
        velocity_range = _get(motion, "velocity_range")
        if velocity_range is not None:
            C.velocity_range = {str(k): (float(v[0]), float(v[1])) for k, v in velocity_range.items()}
            applied.append(f"commands.motion.velocity_range={C.velocity_range}")
        joint_position_range = _get(motion, "joint_position_range")
        if joint_position_range is not None:
            C.joint_position_range = (float(joint_position_range[0]), float(joint_position_range[1]))
            applied.append(f"commands.motion.joint_position_range={C.joint_position_range}")
        sample_random_start_phase = _get(motion, "sample_random_start_phase")
        if sample_random_start_phase is not None:
            C.sample_random_start_phase = _as_bool(sample_random_start_phase)
            applied.append(f"commands.motion.sample_random_start_phase={C.sample_random_start_phase}")
        for key, cast in (
            ("prelude_steps", int),
            ("hold_last_frame_steps", int),
            ("return_to_default_steps", int),
        ):
            value = _get(motion, key)
            if value is not None:
                _require(hasattr(C, key), f"commands.motion.{key}")
                setattr(C, key, cast(value))
                applied.append(f"commands.motion.{key}={getattr(C, key)}")
        reset_to_default_pose = _get(motion, "reset_to_default_pose")
        if reset_to_default_pose is not None:
            _require(hasattr(C, "reset_to_default_pose"), "commands.motion.reset_to_default_pose")
            C.reset_to_default_pose = _as_bool(reset_to_default_pose)
            applied.append(f"commands.motion.reset_to_default_pose={C.reset_to_default_pose}")
        for key in ("reset_perturbation_probability", "hard_case_probability"):
            value = _get(motion, key)
            if value is not None:
                _require(hasattr(C, key), f"commands.motion.{key}")
                setattr(C, key, float(value))
                applied.append(f"commands.motion.{key}={getattr(C, key)}")
        hard_case_motion_ids = _get(motion, "hard_case_motion_ids")
        if hard_case_motion_ids is not None:
            _require(hasattr(C, "hard_case_motion_ids"), "commands.motion.hard_case_motion_ids")
            C.hard_case_motion_ids = tuple(int(v) for v in hard_case_motion_ids)
            applied.append(f"commands.motion.hard_case_motion_ids={C.hard_case_motion_ids}")
        hard_case_velocity_range = _get(motion, "hard_case_velocity_range")
        if hard_case_velocity_range is not None:
            _require(hasattr(C, "hard_case_velocity_range"), "commands.motion.hard_case_velocity_range")
            C.hard_case_velocity_range = {
                str(k): (float(v[0]), float(v[1])) for k, v in hard_case_velocity_range.items()
            }
            applied.append(f"commands.motion.hard_case_velocity_range={C.hard_case_velocity_range}")
        joint_position_offset = _get(motion, "joint_position_offset")
        if joint_position_offset is not None:
            _require(hasattr(C, "joint_position_offset"), "commands.motion.joint_position_offset")
            C.joint_position_offset = {str(k): float(v) for k, v in joint_position_offset.items()}
            applied.append(f"commands.motion.joint_position_offset={C.joint_position_offset}")
        root_position_offset = _get(motion, "root_position_offset")
        if root_position_offset is not None:
            _require(hasattr(C, "root_position_offset"), "commands.motion.root_position_offset")
            _require(len(root_position_offset) == 3, "commands.motion.root_position_offset length == 3")
            C.root_position_offset = tuple(float(v) for v in root_position_offset)
            applied.append(f"commands.motion.root_position_offset={C.root_position_offset}")

    return applied


# --------------------------------------------------------------------------- #
# Training (runs after the simulator is launched).
# --------------------------------------------------------------------------- #
def _run(cfg):
    import os
    import pathlib
    from datetime import datetime

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import training  # noqa: F401
    import training.tasks  # noqa: F401  -- registers the gym tasks
    from training.utils.my_on_policy_runner import MotionOnPolicyRunner, MyOnPolicyRunner
    from training.utils.a3_base_actor_init import initialize_zero_residual_actor_mean
    from training.utils.momentum_preview_actor_critic import MomentumPreviewActorCritic
    from training.utils.stagger_support_actor_critic import (
        BentReadyRecoveryActorCritic,
        StaggerSupportActorCritic,
        WideStaggerRecoveryActorCritic,
        WideStaggerSupportActorCritic,
    )
    from training.utils.target_conditioned_recovery_actor_critic import (
        TargetConditionedRecoveryActorCritic,
    )
    from training.utils.support_recovery_actor_critic import SupportRecoveryActorCritic
    from training.utils.natural_prefix_rollout import (
        NaturalPrefixPPO,
        NaturalPrefixRolloutWrapper,
    )
    from training.utils.ppo_cfg import runner_kwargs
    import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Provenance: confirm we imported the WORKING TREE, not a stale install. If this path points into
    # site-packages instead of .../training, a shadow copy is overriding your edits
    # (fix PYTHONPATH ordering in setup_train_env.sh / reinstall editable) and the YAML edits below are
    # being applied onto the wrong cfg classes.
    print(f"[train.py] training imported from: {training.__file__}", flush=True)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) env cfg (gym registry) + task YAML overrides
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _cfg_mod = sys.modules.get(type(env_cfg).__module__)
    print(f"[train.py] env cfg source: {type(env_cfg).__name__} <- {getattr(_cfg_mod, '__file__', '?')}", flush=True)
    applied = _apply_task_overrides(env_cfg, cfg.task)
    _assert_strict_fall_contract(env_cfg, task_id)
    _assert_fall_recovery_admission(
        task_id,
        _get(cfg.task, "name", task_id),
        legacy_fall_strategy=bool(cfg.get("legacy_fall_strategy", False)),
    )
    print(f"[train.py] applied {len(applied)} task override(s) from cfg/task/{_get(cfg.task, 'name', task_id)}.yaml:", flush=True)
    for _a in applied:
        print(f"[train.py]     {_a}", flush=True)
    if not applied:
        print("[train.py] WARNING: 0 task overrides applied -> the run is using CODE DEFAULTS, not the "
              "YAML (the rewards/racket/env blocks did not compose, or all keys were absent).", flush=True)
    # Human-readable confirmation of the strike-training knobs, straight from the post-override cfg, so
    # you can read the actual runtime values off the launch log without opening logs/.../params/env.yaml.
    R = env_cfg.rewards
    if hasattr(R, "racket_position") and R.racket_position is not None:
        fine = ""
        if hasattr(R, "racket_position_fine") and R.racket_position_fine is not None:
            fine = (
                f" pos_fine={R.racket_position_fine.params.get('std')}"
                f"/w={R.racket_position_fine.weight}"
            )
        print("[train.py] racket reward std (post-override): "
              f"pos={R.racket_position.params.get('std')}{fine} "
              f"vel={R.racket_velocity.params.get('std')} normal={R.racket_normal.params.get('std')}", flush=True)
    if hasattr(env_cfg.commands, "racket_target"):
        _C = env_cfg.commands.racket_target
        print("[train.py] racket target (post-override): "
              f"target_mode={_C.target_mode} ref_perturb_curriculum_start={_C.ref_perturb_curriculum_start} "
              f"strike_window_s={_C.strike_window_s} strike_time_std_s={_C.strike_time_std_s}", flush=True)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    has_motion_command = hasattr(env_cfg.commands, "motion")

    # 2) PPO runner cfg from cfg.algo
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)
    agent_cfg.resume = bool(cfg.get("resume", False))
    natural_prefix_recovery = bool(cfg.get("natural_prefix_recovery", False))
    if natural_prefix_recovery:
        natural_prefix_tasks = {
            "HOPE-FloatingJointCoordinatorV10WideStaggerRecovery-AgibotA3-v0",
            "HOPE-FloatingJointCoordinatorV11BentReadyRecovery-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecovery-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryYComp-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion0Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion2Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion4Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion1Train-AgibotA3-v0",
        }
        if task_id not in natural_prefix_tasks:
            raise ValueError(
                "natural_prefix_recovery requires a registered post-hit recovery task"
            )
        rsl_on_policy_runner.NaturalPrefixPPO = NaturalPrefixPPO
        # rsl_rl's runner uses the literal class name "PPO" to select the RL
        # training type before evaluating the algorithm class.  Alias that
        # resolver to the compact natural-prefix implementation so it keeps
        # the normal PPO lifecycle while excluding masked prefix transitions.
        rsl_on_policy_runner.PPO = NaturalPrefixPPO
        agent_cfg.algorithm.class_name = "PPO"
        print(
            "[train.py] Natural-Prefix Recovery enabled: masked prefix transitions "
            "are excluded from PPO storage",
            flush=True,
        )
    momentum_preview_task = task_id == "HOPE-FloatingJointCoordinatorV6MomentumPreview-AgibotA3-v0"
    support_recovery_task = (
        task_id == "HOPE-FloatingJointCoordinatorV7StaggeredRecovery-AgibotA3-v0"
    )
    legacy_stagger_support_task = (
        task_id == "HOPE-FloatingJointCoordinatorV8StaggerSupport-AgibotA3-v0"
    )
    wide_stagger_support_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV9WideStaggerSupport-AgibotA3-v0"
    )
    wide_stagger_recovery_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV10WideStaggerRecovery-AgibotA3-v0"
    )
    bent_ready_recovery_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV11BentReadyRecovery-AgibotA3-v0"
    )
    target_conditioned_recovery_task = (
        task_id
        in {
            "HOPE-FloatingTargetConditionedRecovery-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryYComp-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion0Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion2Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion4Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion1Train-AgibotA3-v0",
        }
    )
    stagger_support_task = (
        legacy_stagger_support_task
        or wide_stagger_support_task
        or wide_stagger_recovery_task
        or bent_ready_recovery_task
    )
    gated_recovery_task = wide_stagger_recovery_task or bent_ready_recovery_task
    frozen_support_task = momentum_preview_task or support_recovery_task or stagger_support_task
    if momentum_preview_task:
        # OnPolicyRunner resolves policy classes in its own module globals.
        rsl_on_policy_runner.MomentumPreviewActorCritic = MomentumPreviewActorCritic
        agent_cfg.policy.class_name = "MomentumPreviewActorCritic"
        print(
            "[train.py] V19 policy=MomentumPreviewActorCritic "
            "(legacy state actor frozen; preview adapter trains leg+waist only)",
            flush=True,
        )
    elif support_recovery_task:
        rsl_on_policy_runner.SupportRecoveryActorCritic = SupportRecoveryActorCritic
        agent_cfg.policy.class_name = "SupportRecoveryActorCritic"
        print(
            "[train.py] V20 policy=SupportRecoveryActorCritic "
            "(legacy coordinator frozen; state adapter trains leg+waist only)",
            flush=True,
        )
    elif bent_ready_recovery_task:
        rsl_on_policy_runner.BentReadyRecoveryActorCritic = (
            BentReadyRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "BentReadyRecoveryActorCritic"
        print(
            "[train.py] V28 policy=BentReadyRecoveryActorCritic "
            "(frozen V22; bounded post-hit bent-READY settling adapter)",
            flush=True,
        )
    elif target_conditioned_recovery_task:
        rsl_on_policy_runner.TargetConditionedRecoveryActorCritic = (
            TargetConditionedRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "TargetConditionedRecoveryActorCritic"
        print(
            "[train.py] P4 policy=TargetConditionedRecoveryActorCritic "
            "(frozen P3 target policy; gated lower-body brace/residual adapter)",
            flush=True,
        )
    elif wide_stagger_recovery_task:
        rsl_on_policy_runner.WideStaggerRecoveryActorCritic = (
            WideStaggerRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "WideStaggerRecoveryActorCritic"
        print(
            "[train.py] V23 policy=WideStaggerRecoveryActorCritic "
            "(frozen V22 model_1499; gated 22-D post-hit recovery adapter)",
            flush=True,
        )
    elif wide_stagger_support_task:
        rsl_on_policy_runner.WideStaggerSupportActorCritic = (
            WideStaggerSupportActorCritic
        )
        agent_cfg.policy.class_name = "WideStaggerSupportActorCritic"
        print(
            "[train.py] V22 policy=WideStaggerSupportActorCritic "
            "(legacy coordinator frozen; 2-D support adapter trains leg+waist only)",
            flush=True,
        )
    elif legacy_stagger_support_task:
        rsl_on_policy_runner.StaggerSupportActorCritic = StaggerSupportActorCritic
        agent_cfg.policy.class_name = "StaggerSupportActorCritic"
        print(
            "[train.py] V21 policy=StaggerSupportActorCritic "
            "(legacy coordinator frozen; explicit stance branch trains leg+waist only)",
            flush=True,
        )
    warm_start_actor_only = bool(cfg.get("warm_start_actor_only", False))
    warm_start_support_actor_only = bool(cfg.get("warm_start_support_actor_only", False))
    warm_start_append_zero_policy_obs = bool(cfg.get("warm_start_append_zero_policy_obs", False))
    actor_only_warm_start = warm_start_actor_only or warm_start_support_actor_only
    if warm_start_actor_only and warm_start_support_actor_only:
        raise ValueError(
            "warm_start_actor_only=true and warm_start_support_actor_only=true are mutually exclusive"
        )
    if agent_cfg.resume and actor_only_warm_start:
        raise ValueError("resume=true and actor-only warm starts are mutually exclusive")
    if actor_only_warm_start and cfg.get("checkpoint", None) is None:
        raise ValueError("actor-only warm starts require an explicit checkpoint=<model_*.pt>")
    if warm_start_append_zero_policy_obs and not actor_only_warm_start:
        raise ValueError(
            "warm_start_append_zero_policy_obs=true requires an actor-only warm start"
        )
    if warm_start_support_actor_only and not stagger_support_task:
        raise ValueError(
            "warm_start_support_actor_only=true is only valid for the stagger-support policy"
        )
    if cfg.get("load_run", None) is not None:
        agent_cfg.load_run = str(cfg.load_run)
    if cfg.get("checkpoint", None) is not None:
        agent_cfg.load_checkpoint = str(cfg.checkpoint)
    if (
        cfg.get("checkpoint", None) is not None
        and not agent_cfg.resume
        and not actor_only_warm_start
    ):
        raise ValueError(
            "checkpoint=<model_*.pt> was provided but would be ignored because "
            "resume=false. Use resume=true for an exact checkpoint load, or an "
            "explicit warm_start_actor_only/warm_start_support_actor_only mode."
        )

    _assert_a3_base_stand_smoke_gate(
        task_id,
        agent_cfg.max_iterations,
        num_envs,
        getattr(agent_cfg.policy, "init_noise_std", None),
        bool(cfg.get("recovery_v23", False)),
    )

    # 3) motion source. Motion-imitation tasks require a clip; pure table-tennis RL tasks do not.
    registry_name = None
    if has_motion_command:
        motion_manifest = cfg.motion_manifest if cfg.motion_manifest is not None else _get(cfg.task, "motion_manifest")
        motion_file = cfg.motion_file if cfg.motion_file is not None else _get(cfg.task, "motion_file")
        if motion_manifest is not None:
            manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = pathlib.Path.cwd() / manifest_path
            if not manifest_path.is_file():
                raise FileNotFoundError(f"motion_manifest does not exist: {manifest_path}")
            # A3 strike PPO must never silently consume an old K8/K12 manifest,
            # a target-relabel result, or a diagnostic archive.  This happens
            # before the Isaac environment is constructed, so it protects both
            # expensive training and any code path that would otherwise load
            # the motion library.
            if str(getattr(cfg.task, "gym_task", "")).startswith("HOPE-NativeStrike"):
                assert_training_manifest(manifest_path)
            env_cfg.commands.motion.motion_manifest = str(manifest_path)
            env_cfg.commands.motion.motion_file = None
            subset_size = cfg.manifest_subset_size if cfg.manifest_subset_size is not None else _get(cfg.task, "manifest_subset_size")
            if subset_size is not None:
                env_cfg.commands.motion.manifest_subset_size = int(subset_size)
            frame_z_offset = (
                cfg.manifest_frame_z_offset
                if cfg.manifest_frame_z_offset is not None
                else _get(cfg.task, "manifest_frame_z_offset")
            )
            if (
                task_id == "HOPE-FloatingUnifiedUpperReferenceTracker-AgibotA3-v0"
                and frame_z_offset is not None
                and abs(float(frame_z_offset)) > 1.0e-8
            ):
                raise ValueError(
                    "P5U unified upper tracker requires manifest_frame_z_offset=0.0: "
                    "P5D scene-placed NPZ files already contain the world z anchor; "
                    f"received {float(frame_z_offset):.6f} m (would double-apply the lift)."
                )
            if frame_z_offset is not None:
                env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
            if _as_bool(cfg.get("validate_stance_contract", False)):
                env_cfg.commands.motion.validate_stance_contract = True
                stance_mode = cfg.get("stance_contract_mode", None)
                if stance_mode is not None:
                    env_cfg.commands.motion.stance_contract_mode = str(stance_mode)
                print(
                    "[train.py] stance contract validation enabled for this manifest "
                    "(prepositioned metadata + NPZ root consistency)",
                    flush=True,
                )
            registry_name = f"local:{manifest_path}"
            print(
                f"[train.py] using local motion_manifest: {manifest_path} "
                f"(subset_size={env_cfg.commands.motion.manifest_subset_size}, "
                f"frame_z_offset={env_cfg.commands.motion.manifest_frame_z_offset:.4f}m)",
                flush=True,
            )
        elif motion_file is not None:
            motion_path = pathlib.Path(str(motion_file)).expanduser()
            if not motion_path.is_absolute():
                motion_path = pathlib.Path.cwd() / motion_path
            if not motion_path.is_file():
                raise FileNotFoundError(
                    f"motion_file does not exist: {motion_path}. "
                    "Generate the public smoke clip with scripts/create_smoke_motion.py or pass a retargeted .npz."
                )
            env_cfg.commands.motion.motion_file = str(motion_path)
            registry_name = f"local:{motion_path}"
            print(f"[train.py] using local motion_file: {motion_path}", flush=True)
        else:
            registry_name = cfg.registry_name if cfg.registry_name is not None else cfg.task.registry_name
            registry_name = str(registry_name)
            if ":" not in registry_name:
                registry_name += ":latest"
            print(f"[train.py] loading motion from WandB registry: {registry_name}", flush=True)
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        print("[train.py] env has no motion command; running pure RL task without motion source.", flush=True)

    # 4) logging dir (same layout as scripts/rsl_rl/train.py so export/eval are unchanged)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Task: {task_id} | experiment: {agent_cfg.experiment_name} | log: {log_dir}")

    # 5) build env, wrap, run
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    if natural_prefix_recovery:
        env = NaturalPrefixRolloutWrapper(
            env,
            phase_mode=str(cfg.get("natural_recovery_phase_mode", "phase_balanced")),
            min_post_hit_steps=int(cfg.get("natural_recovery_min_post_hit_steps", 10)),
            early_max_post_hit_steps=int(
                cfg.get("natural_recovery_early_max_post_hit_steps", 30)
            ),
            mid_max_post_hit_steps=int(
                cfg.get("natural_recovery_mid_max_post_hit_steps", 80)
            ),
        )
    preview_audit_mode = str(cfg.get("preview_audit_mode", "normal"))
    if preview_audit_mode not in {"normal", "zero", "shuffle", "reverse", "scale_080", "scale_120"}:
        raise ValueError(f"Unknown preview_audit_mode={preview_audit_mode!r}")
    if preview_audit_mode != "normal" and not bool(cfg.get("audit_policy_action", False)):
        raise ValueError("preview_audit_mode other than 'normal' is allowed only with +audit_policy_action=true")
    env.unwrapped.coordinator_preview_audit_mode = preview_audit_mode
    if preview_audit_mode != "normal":
        print(f"[train.py] coordinator preview causal audit mode={preview_audit_mode}", flush=True)
    prelude_audit_mode = str(cfg.get("coordinator_prelude_audit_mode", "none"))
    if prelude_audit_mode not in {"none", "all", "leg", "waist", "arm"}:
        raise ValueError(f"Unknown coordinator_prelude_audit_mode={prelude_audit_mode!r}")
    if prelude_audit_mode != "none" and not bool(cfg.get("audit_policy_action", False)):
        raise ValueError("coordinator_prelude_audit_mode other than 'none' requires +audit_policy_action=true")
    env.unwrapped.coordinator_prelude_audit_mode = prelude_audit_mode
    if prelude_audit_mode != "none":
        print(f"[train.py] coordinator prelude causal audit mode={prelude_audit_mode}", flush=True)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner_registry_name = None if registry_name and registry_name.startswith("local:") else registry_name
    if has_motion_command:
        runner = MotionOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=runner_registry_name
        )
    else:
        runner = MyOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if momentum_preview_task:
        # The final 18 observations are already physically normalized by
        # m*g and m*g*L. Preserve them exactly instead of allowing the generic
        # running normalizer to erase magnitude semantics or map physical zero
        # to a nonzero feature vector.
        original_obs_normalizer_forward = runner.obs_normalizer.forward

        def v19_actor_observation_normalizer(observation):
            normalized = original_obs_normalizer_forward(observation)
            normalized[..., 204:] = observation[..., 204:]
            return normalized

        runner.obs_normalizer.forward = v19_actor_observation_normalizer
        print(
            "[train.py] V19 actor normalizer preserves canonical preview columns 204:222",
            flush=True,
        )
    elif stagger_support_task or target_conditioned_recovery_task:
        original_obs_normalizer_forward = runner.obs_normalizer.forward

        def v21_actor_observation_normalizer(observation):
            normalized = original_obs_normalizer_forward(observation)
            normalized[..., 204:] = observation[..., 204:]
            return normalized

        runner.obs_normalizer.forward = v21_actor_observation_normalizer
        support_end = (
            213
            if target_conditioned_recovery_task
            else 235
            if bent_ready_recovery_task
            else 227
            if wide_stagger_support_task or wide_stagger_recovery_task
            else 223
        )
        print(
            "[train.py] recovery/support actor normalizer preserves physical columns "
            f"204:{support_end}",
            flush=True,
        )
        if gated_recovery_task or target_conditioned_recovery_task:
            original_train_mode = runner.train_mode

            def v23_train_mode_with_frozen_actor_normalizer():
                original_train_mode()
                # The inherited strike actor is an immutable behavior prior.
                # Updating its legacy observation statistics would change the
                # frozen actor even when every network parameter stays fixed.
                runner.obs_normalizer.eval()

            runner.train_mode = v23_train_mode_with_frozen_actor_normalizer
            print(
                "[train.py] recovery freezes the loaded strike actor observation "
                "normalizer; critic normalization remains trainable",
                flush=True,
            )
    zero_legacy_support_action = bool(cfg.get("zero_legacy_support_action", False))
    if zero_legacy_support_action:
        if not wide_stagger_support_task:
            raise ValueError(
                "zero_legacy_support_action=true is only valid for the V22 wide-stagger policy"
            )
        runner.alg.policy.zero_legacy_support_action = True
        print(
            "[train.py] V22 audit contract: legacy leg/waist coordinator output is zero; "
            "legacy arm output and frozen base priors remain active",
            flush=True,
        )
    zero_residual_tasks = {
        "A3BaseStandPassiveStableCandidate-v0",
        "A3CatchReadyStand-v0",
        "A3BaseStandRecoveryA-v0",
        "A3BaseStandRecoveryAV2-v0",
        "A3BaseStandRecoveryAV2WaistMask-v0",
        "A3BaseStandRecoveryAV21WaistMask-v0",
        "HOPE-StrikeStabilizerA-AgibotA3-v0",
        "HOPE-RetrainStrikeStabilizerA-AgibotA3-v0",
        "HOPE-FloatingF1-AgibotA3-v0",
        "HOPE-FloatingUpperCorrection-AgibotA3-v0",
        "HOPE-FloatingJointCoordinator-AgibotA3-v0",
        "HOPE-FloatingJointCoordinatorV2-AgibotA3-v0",
        "HOPE-FloatingJointCoordinatorV3-AgibotA3-v0",
        "HOPE-FloatingJointCoordinatorV4-AgibotA3-v0",
        "HOPE-FixedBaseReferenceStrike-AgibotA3-v0",
        "HOPE-FixedBaseBackhandReferenceStrike-AgibotA3-v0",
        "HOPE-FixedBaseTargetAdapter-AgibotA3-v0",
        "HOPE-FloatingPriorGuidedReferenceTracker-AgibotA3-v0",
        "HOPE-FloatingUnifiedUpperReferenceTracker-AgibotA3-v0",
        "HOPE-FloatingUnifiedUpperReferenceTrackerB-AgibotA3-v0",
        "HOPE-FloatingUnifiedUpperReferenceTrackerC-AgibotA3-v0",
    }
    if task_id in zero_residual_tasks and not agent_cfg.resume and not warm_start_actor_only:
        # This task controls a non-integrating residual around a passively
        # stable nominal posture.  A random output layer can create a large
        # deterministic residual before PPO sees one transition (observed as
        # 52% raw-action clipping in the v2 model_0 audit).  Keep exploration
        # in the Gaussian std, but make the initial mean policy exactly zero.
        initialize_zero_residual_actor_mean(
            runner,
            action_dim=(
                22 if task_id in {
                    "HOPE-FloatingJointCoordinator-AgibotA3-v0",
                    "HOPE-FloatingJointCoordinatorV2-AgibotA3-v0",
                    "HOPE-FloatingJointCoordinatorV3-AgibotA3-v0",
                    "HOPE-FloatingJointCoordinatorV4-AgibotA3-v0",
                    "HOPE-FloatingJointCoordinatorV5Preview-AgibotA3-v0",
                }
                else 7 if task_id == "HOPE-FixedBaseTargetAdapter-AgibotA3-v0"
                else 11 if task_id == "HOPE-FloatingUnifiedUpperReferenceTrackerB-AgibotA3-v0"
                else 14 if str(_get(cfg.task, "name", "")).endswith("Microstep")
                else 10 if task_id.startswith("HOPE-FixedBase") or task_id in {
                    "HOPE-FloatingUpperCorrection-AgibotA3-v0",
                    "HOPE-FloatingPriorGuidedReferenceTracker-AgibotA3-v0",
                    "HOPE-FloatingUnifiedUpperReferenceTracker-AgibotA3-v0",
                }
                else 14
            ),
        )
        print(
            "[train.py] initialized A3 Base/Recovery actor mean to exact zero residual; "
            f"exploration remains init_noise_std={agent_cfg.policy.init_noise_std}",
            flush=True,
        )
    runner.add_git_repo_to_log(__file__)
    if actor_only_warm_start:
        # A changed episode horizon/recovery objective invalidates the old
        # critic, optimizer moments, iteration count and critic normalizer.
        # Preserve only the compatible coordinator actor and its 204-D input
        # normalization so the new task starts with V2's hit behavior.
        warm_path = Path(str(agent_cfg.load_checkpoint)).expanduser()
        if not warm_path.is_file():
            raise FileNotFoundError(f"actor-only warm-start checkpoint does not exist: {warm_path}")
        warm_state = torch.load(warm_path, map_location="cpu", weights_only=False)
        model_state = warm_state.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise RuntimeError(f"checkpoint has no model_state_dict: {warm_path}")
        if warm_start_support_actor_only:
            support_prefixes = (
                "support_state_encoder.",
                "stagger_encoder.",
                "support_fusion.",
                "support_adapter.",
            )
            actor_state = {
                name: value
                for name, value in model_state.items()
                if name == "std"
                or name.startswith("actor.")
                or name.startswith(support_prefixes)
            }
        else:
            actor_state = {
                name: value
                for name, value in model_state.items()
                if name == "std" or name.startswith("actor.")
            }
        if not actor_state or "std" not in actor_state:
            raise RuntimeError(f"checkpoint has no compatible actor/std tensors: {warm_path}")
        runtime_state_keys = set(runner.alg.policy.state_dict())
        runtime_state = runner.alg.policy.state_dict()
        appended_obs_features = 0
        if target_conditioned_recovery_task:
            if not warm_start_actor_only or not warm_start_append_zero_policy_obs:
                raise RuntimeError(
                    "P4 target-conditioned recovery requires "
                    "+warm_start_actor_only=true and "
                    "+warm_start_append_zero_policy_obs=true"
                )
            # P3's actor remains 204-D; only the runtime normalizer widens
            # for the private recovery suffix.  Keep P4's configured
            # exploration scale rather than importing P3's all-action noise.
            actor_state["std"] = runtime_state["std"].clone()
            mismatched = [
                name
                for name, value in actor_state.items()
                if name in runtime_state
                and torch.is_tensor(value)
                and value.shape != runtime_state[name].shape
            ]
            if mismatched:
                raise RuntimeError(
                    "P4's frozen P3 actor must retain the exact 204-D actor shape; "
                    f"mismatched={mismatched}"
                )
        elif frozen_support_task:
            if support_recovery_task and warm_start_append_zero_policy_obs:
                raise RuntimeError(
                    "V20 support recovery keeps the legacy 204-D observation contract; "
                    "do not set +warm_start_append_zero_policy_obs=true"
                )
            if gated_recovery_task and (
                not warm_start_support_actor_only
                or not warm_start_append_zero_policy_obs
            ):
                raise RuntimeError(
                    "gated recovery requires +warm_start_support_actor_only=true and "
                    "+warm_start_append_zero_policy_obs=true"
                )
            if not warm_start_append_zero_policy_obs and not warm_start_support_actor_only:
                if momentum_preview_task:
                    raise RuntimeError("V19 warm start requires +warm_start_append_zero_policy_obs=true")
                if stagger_support_task:
                    raise RuntimeError("V21 warm start requires +warm_start_append_zero_policy_obs=true")
            # Preserve V2's deterministic actor mean, but retain the new
            # support branch's configured exploration scale.
            actor_state["std"] = runtime_state["std"].clone()
            mismatched = [
                name
                for name, value in actor_state.items()
                if name in runtime_state and torch.is_tensor(value) and value.shape != runtime_state[name].shape
            ]
            if mismatched:
                raise RuntimeError(
                    "frozen legacy actor must retain the exact 204-D model_0 shape; "
                    f"mismatched={mismatched}"
                )
        elif warm_start_append_zero_policy_obs:
            mismatched = [
                name
                for name, value in actor_state.items()
                if name in runtime_state and torch.is_tensor(value) and value.shape != runtime_state[name].shape
            ]
            if len(mismatched) != 1:
                raise RuntimeError(
                    "append-zero actor migration requires exactly one mismatched actor tensor, "
                    f"got {mismatched}"
                )
            input_weight_name = mismatched[0]
            old_weight = actor_state[input_weight_name]
            new_weight = runtime_state[input_weight_name]
            if (
                not input_weight_name.startswith("actor.")
                or old_weight.ndim != 2
                or new_weight.ndim != 2
                or old_weight.shape[0] != new_weight.shape[0]
                or old_weight.shape[1] >= new_weight.shape[1]
            ):
                raise RuntimeError(
                    "append-zero actor migration only permits widening the first actor input layer: "
                    f"{input_weight_name} {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}"
                )
            appended_obs_features = int(new_weight.shape[1] - old_weight.shape[1])
            migrated_weight = torch.zeros_like(new_weight)
            migrated_weight[:, : old_weight.shape[1]] = old_weight
            actor_state[input_weight_name] = migrated_weight
        expected_missing = {name for name in runtime_state_keys if name.startswith("critic.")}
        if target_conditioned_recovery_task:
            expected_missing.update(
                name
                for name in runtime_state_keys
                if name.startswith("recovery_encoder.")
                or name.startswith("recovery_adapter.")
            )
        elif warm_start_support_actor_only:
            expected_missing = {
                name for name in runtime_state_keys if name.startswith("critic.")
            }
            if gated_recovery_task:
                expected_missing.update(
                    name
                    for name in runtime_state_keys
                    if name.startswith("recovery_encoder.")
                    or name.startswith("recovery_adapter.")
                )
        elif momentum_preview_task:
            expected_missing.update(
                name
                for name in runtime_state_keys
                if name.startswith("preview_encoder.")
                or name.startswith("support_state_encoder.")
                or name.startswith("preview_state_gate.")
                or name.startswith("preview_adapter.")
            )
        elif stagger_support_task:
            expected_missing.update(
                name
                for name in runtime_state_keys
                if name.startswith("support_state_encoder.")
                or name.startswith("stagger_encoder.")
                or name.startswith("support_fusion.")
                or name.startswith("support_adapter.")
            )
        elif support_recovery_task:
            expected_missing.update(
                name
                for name in runtime_state_keys
                if name.startswith("support_encoder.")
                or name.startswith("support_adapter.")
            )
        missing_keys = runtime_state_keys - set(actor_state)
        unexpected_keys = set(actor_state) - runtime_state_keys
        if missing_keys != expected_missing or unexpected_keys:
            raise RuntimeError(
                "actor-only warm-start state mismatch: "
                f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
            )
        # rsl_rl's ActorCritic intentionally returns a boolean here rather
        # than PyTorch's IncompatibleKeys object, so validate keys above.
        runner.alg.policy.load_state_dict(actor_state, strict=False)
        if not getattr(runner, "empirical_normalization", False):
            raise RuntimeError("actor-only warm-start requires empirical observation normalization")
        actor_norm_state = warm_state.get("obs_norm_state_dict")
        if not isinstance(actor_norm_state, dict):
            raise RuntimeError(f"checkpoint has no actor observation normalizer: {warm_path}")
        runtime_norm_state = runner.obs_normalizer.state_dict()
        migrated_normalizer = False
        for key in ("_mean", "_var", "_std"):
            if key not in actor_norm_state or key not in runtime_norm_state:
                raise RuntimeError(f"actor-only warm-start normalizer missing {key!r}")
            if actor_norm_state[key].shape != runtime_norm_state[key].shape:
                if not warm_start_append_zero_policy_obs:
                    raise RuntimeError(
                        "actor-only warm-start observation width mismatch: "
                        f"checkpoint {key}={tuple(actor_norm_state[key].shape)}, "
                        f"runtime={tuple(runtime_norm_state[key].shape)}"
                    )
                old_value = actor_norm_state[key]
                new_value = runtime_norm_state[key]
                if old_value.ndim != new_value.ndim or old_value.shape[:-1] != new_value.shape[:-1] or old_value.shape[-1] >= new_value.shape[-1]:
                    raise RuntimeError(
                        "append-zero normalizer migration requires an appended observation width: "
                        f"{key} {tuple(old_value.shape)} -> {tuple(new_value.shape)}"
                    )
                migrated = new_value.clone()
                migrated[..., : old_value.shape[-1]] = old_value
                actor_norm_state[key] = migrated
                appended_obs_features = max(
                    appended_obs_features,
                    int(new_value.shape[-1] - old_value.shape[-1]),
                )
                migrated_normalizer = True
        if warm_start_append_zero_policy_obs and (appended_obs_features <= 0 or not migrated_normalizer):
            raise RuntimeError("append-zero actor migration did not widen both actor and observation normalizer")
        runner.obs_normalizer.load_state_dict(actor_norm_state)
        equivalence_max_abs = None
        if (
            (gated_recovery_task and warm_start_support_actor_only)
            or target_conditioned_recovery_task
        ):
            policy = runner.alg.policy
            if any(
                torch.count_nonzero(value).item() != 0
                for value in policy.recovery_adapter.state_dict().values()
            ):
                raise RuntimeError("gated recovery adapter is not exactly zero after warm start")
            with torch.no_grad():
                probe = torch.linspace(
                    -1.0,
                    1.0,
                    steps=4 * policy.recovery_total_obs_dim,
                    device=agent_cfg.device,
                ).reshape(4, policy.recovery_total_obs_dim)
                if target_conditioned_recovery_task:
                    # The motion-3 brace and P11 motion-1 bootstrap are
                    # intentional baseline changes. Verify zero learned-
                    # residual equivalence on an untouched P3 motion instead
                    # of falsely treating either support prior as drift.
                    probe[:, -2] = 0.0
                expected_action = policy.base_action_mean(
                    probe[:, : policy.BASE_OBS_DIM]
                )
                expanded_action = policy.act_inference(probe)
                equivalence_max_abs = float(
                    torch.max(torch.abs(expected_action - expanded_action)).item()
                )
            if equivalence_max_abs >= 1.0e-6:
                raise RuntimeError(
                    "gated recovery zero-adapter equivalence failed: "
                    f"max_abs={equivalence_max_abs:.9g}"
                )
            print(
                "[train.py] gated recovery zero-residual equivalence passed "
                "outside motion-1/motion-3 braces: "
                f"max_abs={equivalence_max_abs:.3e}",
                flush=True,
            )
        elif frozen_support_task and not warm_start_support_actor_only:
            policy = runner.alg.policy
            if momentum_preview_task:
                if any(torch.count_nonzero(value).item() != 0 for value in policy.preview_adapter.state_dict().values()):
                    raise RuntimeError("V19 preview adapter is not exactly zero after warm start")
                if any(torch.count_nonzero(value).item() != 0 for value in policy.preview_state_gate.state_dict().values()):
                    raise RuntimeError("V19 preview state gate is not exactly zero after warm start")
            else:
                if any(torch.count_nonzero(value).item() != 0 for value in policy.support_adapter.state_dict().values()):
                    raise RuntimeError("Support adapter is not exactly zero after warm start")
            with torch.no_grad():
                probe_state = torch.linspace(
                    -1.0, 1.0, steps=4 * 204, device=agent_cfg.device
                ).reshape(4, 204)
                legacy_action = policy.actor(probe_state)
                if momentum_preview_task:
                    probe_preview = torch.linspace(
                        0.75, -0.75, steps=4 * 18, device=agent_cfg.device
                    ).reshape(4, 18)
                    expanded_input = torch.cat((probe_state, probe_preview), dim=-1)
                elif stagger_support_task:
                    probe_support = torch.linspace(
                        -0.5,
                        0.5,
                        steps=4 * policy.support_obs_dim,
                        device=agent_cfg.device,
                    ).reshape(4, policy.support_obs_dim)
                    expanded_input = torch.cat((probe_state, probe_support), dim=-1)
                else:
                    expanded_input = probe_state
                expanded_action = policy.act_inference(expanded_input)
                expected_action = legacy_action
                if zero_legacy_support_action:
                    expected_action = legacy_action.clone()
                    expected_action[..., : policy.support_action_dim] = 0.0
                equivalence_max_abs = float(
                    torch.max(torch.abs(expected_action - expanded_action)).item()
                )
            if equivalence_max_abs >= 1.0e-6:
                raise RuntimeError(
                    f"frozen-support model_0 action equivalence failed: max_abs={equivalence_max_abs:.9g}"
                )
            print(
                "[train.py] frozen-support model_0 action equivalence passed: "
                f"max_abs={equivalence_max_abs:.3e}; arm exploration std={policy.fixed_arm_std:.1e}",
                flush=True,
            )
        # The optimizer and critic normalizer are intentionally untouched.
        # Explicitly reset the visible iteration counter as a guard against
        # accidentally treating this new contract as a continuation.
        runner.current_learning_iteration = 0
        warm_start_record = {
            "mode": (
                "support_actor_only"
                if warm_start_support_actor_only
                else "actor_only"
            ),
            "checkpoint": str(warm_path.resolve()),
            "loaded": (
                [
                    "actor",
                    "stagger_support_encoder_and_adapter",
                    "actor_observation_normalizer",
                ]
                if warm_start_support_actor_only
                else ["actor", "actor_observation_normalizer"]
                if frozen_support_task
                else ["actor", "std", "actor_observation_normalizer"]
            ),
            "reset_support_exploration_std": (
                float(runner.alg.policy.std[: runner.alg.policy.support_action_dim].mean().item())
                if frozen_support_task
                else None
            ),
            "appended_zero_policy_observation_features": appended_obs_features,
            "model_0_action_equivalence_max_abs": equivalence_max_abs,
            "frozen_legacy_state_actor": frozen_support_task,
            "loaded_support_adapter": warm_start_support_actor_only,
            "support_corrected_action_indices": list(range(15)) if frozen_support_task else None,
            "fixed_arm_exploration_std": (
                runner.alg.policy.fixed_arm_std if frozen_support_task else None
            ),
            "frozen_actor_observation_normalizer": (
                gated_recovery_task or target_conditioned_recovery_task
            ),
            "reset": ["critic", "critic_observation_normalizer", "optimizer", "iteration"],
        }
        Path(log_dir, "params", "warm_start.json").parent.mkdir(parents=True, exist_ok=True)
        Path(log_dir, "params", "warm_start.json").write_text(
            json.dumps(warm_start_record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[train.py] actor-only warm start: loaded "
            + (
                "legacy actor + learned stagger support adapter"
                if warm_start_support_actor_only
                else "V2 actor"
            )
            + "/actor normalizer"
            + (" with support std reset; " if frozen_support_task else "/std; ")
            +
            "critic, critic normalizer, optimizer, and iteration reset",
            flush=True,
        )
    elif agent_cfg.resume:
        # A new task contract may intentionally reuse a compatible checkpoint
        # from another experiment directory (for example final-pose hold ->
        # return-to-ready).  Accept an explicit file path before falling back
        # to RSL-RL's experiment-relative lookup.
        direct_checkpoint = Path(str(agent_cfg.load_checkpoint)).expanduser()
        if direct_checkpoint.is_file():
            resume_path = str(direct_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        runner.load(resume_path)
        if bool(cfg.get("reset_manifest_swing_type_input", False)):
            # A forehand-only checkpoint observed a constant swing-type value,
            # so its empirical-normalizer slot has effectively zero variance.
            # Feeding the first semantic backhand label through that old slot
            # would create an artificial O(10^2) feature.  Migrate *only* this
            # one input: normalize the explicit manifest label around 0 with
            # unit scale, and start the corresponding actor input weights at
            # zero.  The critic has a different privileged-observation schema,
            # so its input column 70 must not be altered.  All learned
            # dynamics/control features remain untouched.
            swing_idx = int(cfg.get("manifest_swing_type_obs_index", 70))
            policy = runner.alg.policy
            normalized_terms = []
            normalizer = getattr(runner, "obs_normalizer", None)
            if normalizer is None:
                raise RuntimeError("runner has no obs_normalizer; cannot safely migrate manifest swing_type input")
            state = normalizer.state_dict()
            if not all(key in state for key in ("_mean", "_var", "_std")):
                raise RuntimeError("obs_normalizer does not expose empirical-normalizer state")
            if swing_idx >= state["_mean"].numel():
                raise RuntimeError(
                    f"manifest_swing_type_obs_index={swing_idx} outside obs_normalizer "
                    f"dimension {state['_mean'].numel()}"
                )
            state["_mean"][..., swing_idx] = 0.0
            state["_var"][..., swing_idx] = 1.0
            state["_std"][..., swing_idx] = 1.0
            normalizer.load_state_dict(state)
            normalized_terms.append("runner.obs_normalizer")

            first_linear = next(
                (module for module in policy.actor.modules() if isinstance(module, torch.nn.Linear)), None
            )
            if first_linear is None or swing_idx >= first_linear.weight.shape[1]:
                raise RuntimeError("actor input does not contain the manifest swing_type observation")
            with torch.no_grad():
                first_linear.weight[:, swing_idx].zero_()
            optimizer_state = runner.alg.optimizer.state.get(first_linear.weight, {})
            for value in optimizer_state.values():
                if torch.is_tensor(value) and value.shape == first_linear.weight.shape:
                    value[:, swing_idx].zero_()
            reset_layers = ["actor"]

            print(
                "[train.py] migrated manifest swing_type input contract: "
                f"index={swing_idx} normalizers={normalized_terms} "
                f"zeroed_first_layer_columns={reset_layers}",
                flush=True,
            )
        std_scale = float(cfg.get("policy_std_scale", 1.0))
        if std_scale <= 0.0:
            raise ValueError("policy_std_scale must be positive")
        if abs(std_scale - 1.0) > 1.0e-9:
            with torch.no_grad():
                before = runner.alg.policy.std.detach().clone()
                runner.alg.policy.std.mul_(std_scale)
                after = runner.alg.policy.std.detach().clone()
            print(
                "[train.py] continuation policy std scaled after checkpoint load: "
                f"scale={std_scale} before_mean={before.mean().item():.6f} "
                f"after_mean={after.mean().item():.6f}",
                flush=True,
            )

        std_max_cfg = cfg.get("policy_std_max", None)
        std_min_cfg = cfg.get("policy_std_min", None)
        std_max = None if std_max_cfg is None else float(std_max_cfg)
        std_min = None if std_min_cfg is None else float(std_min_cfg)
        if std_max is not None and std_max <= 0.0:
            raise ValueError("policy_std_max must be positive")
        if std_min is not None and std_min <= 0.0:
            raise ValueError("policy_std_min must be positive")
        if std_min is not None and std_max is not None and std_min > std_max:
            raise ValueError("policy_std_min cannot exceed policy_std_max")
        if std_min is not None or std_max is not None:
            with torch.no_grad():
                before = runner.alg.policy.std.detach().clone()
                runner.alg.policy.std.clamp_(min=std_min, max=std_max)
                after = runner.alg.policy.std.detach().clone()
            print(
                "[train.py] recovery policy std bounded after checkpoint load: "
                f"min={std_min} max={std_max} before_mean={before.mean().item():.6f} "
                f"after_mean={after.mean().item():.6f}",
                flush=True,
            )

            original_update = runner.alg.update

            def bounded_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                with torch.no_grad():
                    runner.alg.policy.std.clamp_(min=std_min, max=std_max)
                return result

            runner.alg.update = bounded_update

        if bool(cfg.get("freeze_actor_mean", False)):
            for parameter in runner.alg.policy.actor.parameters():
                parameter.requires_grad_(False)
            print("[train.py] frozen Recovery actor mean parameters after checkpoint load", flush=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    audit_zero_action = bool(cfg.get("audit_zero_action", False))
    audit_policy_action = bool(cfg.get("audit_policy_action", False))
    if audit_zero_action and audit_policy_action:
        raise ValueError("Choose exactly one audit action source: zero or deterministic policy")
    if audit_zero_action or audit_policy_action:
        # Deterministic integration audit: exercise the exact environment
        # construction used for training without invoking PPO collection or an
        # optimizer update.  This is used to qualify frozen composite policies
        # and learned coordinator checkpoints.
        #
        # Do not infer safety from the final root pose.  Isaac can reset an
        # environment immediately after a termination, which would otherwise
        # make a fallen robot look like it recovered.  Record every active
        # termination term and independent physical safety signals before the
        # hit frame and through a short post-hit settling window.
        raw = env.unwrapped
        recovery_adapter_scale = float(
            cfg.get("audit_recovery_adapter_scale", 1.0)
        )
        recovery_center_decay_steps = int(
            cfg.get("audit_recovery_center_decay_steps", 0)
        )
        recovery_center_half_width_m = float(
            cfg.get("audit_recovery_center_half_width_m", 0.04)
        )
        recovery_center_decay_source = str(
            cfg.get(
                "audit_recovery_center_decay_source",
                "recovery_adapter",
            )
        )
        stage_a_front_gain = float(cfg.get("audit_stage_a_front_gain", 1.0))
        stage_a_front_margin_m = float(
            cfg.get("audit_stage_a_front_margin_m", 0.04)
        )
        stage_a_front_velocity_mps = float(
            cfg.get("audit_stage_a_front_velocity_mps", 0.02)
        )
        stage_a_front_lead_steps = int(
            cfg.get("audit_stage_a_front_lead_steps", 0)
        )
        if recovery_adapter_scale <= 0.0:
            raise ValueError("audit_recovery_adapter_scale must be positive")
        if recovery_center_decay_steps < 0:
            raise ValueError("audit_recovery_center_decay_steps must be non-negative")
        if recovery_center_half_width_m <= 0.0:
            raise ValueError("audit_recovery_center_half_width_m must be positive")
        if stage_a_front_gain < 1.0:
            raise ValueError("audit_stage_a_front_gain must be at least 1.0")
        if stage_a_front_margin_m <= 0.0:
            raise ValueError("audit_stage_a_front_margin_m must be positive")
        if stage_a_front_velocity_mps < 0.0:
            raise ValueError("audit_stage_a_front_velocity_mps must be non-negative")
        if stage_a_front_lead_steps < 0:
            raise ValueError("audit_stage_a_front_lead_steps must be non-negative")
        if recovery_center_decay_source not in {
            "recovery_adapter",
            "v22_support_adapter",
            "all_coordinator",
            "stage_a",
        }:
            raise ValueError(
                "audit_recovery_center_decay_source must be one of "
                "recovery_adapter, v22_support_adapter, all_coordinator, stage_a"
            )
        if recovery_center_decay_steps > 0 and (
            not audit_policy_action or not gated_recovery_task
        ):
            raise ValueError(
                "audit_recovery_center_decay_steps requires a gated recovery policy audit"
            )
        if abs(stage_a_front_gain - 1.0) > 1.0e-9 and (
            not audit_policy_action or not stagger_support_task
        ):
            raise ValueError(
                "audit_stage_a_front_gain requires a stagger-support policy audit"
            )
        if (
            abs(stage_a_front_gain - 1.0) > 1.0e-9
            and recovery_center_decay_steps > 0
        ):
            raise ValueError(
                "Do not combine audit_stage_a_front_gain with center-decay audits"
            )
        if abs(recovery_adapter_scale - 1.0) > 1.0e-9:
            if not audit_policy_action or not gated_recovery_task:
                raise ValueError(
                    "audit_recovery_adapter_scale requires a gated recovery policy audit"
                )
            with torch.no_grad():
                for parameter in runner.alg.policy.recovery_adapter.parameters():
                    parameter.mul_(recovery_adapter_scale)
            print(
                "[train.py] gated recovery audit-only adapter scale="
                f"{recovery_adapter_scale:g}",
                flush=True,
            )
        motion = raw.command_manager.get_term("motion")
        racket = raw.command_manager.get_term("racket_target")
        cases = raw.num_envs
        support_fd_epsilon = float(cfg.get("audit_support_fd_epsilon", 0.0))
        support_fd_dims = 15
        support_fd_variants = 1 + 2 * support_fd_dims
        support_candidates_value = cfg.get("audit_support_candidates", None)
        support_candidates_path = (
            Path(str(support_candidates_value)).expanduser().resolve()
            if support_candidates_value
            else None
        )
        support_candidate_knots = None
        support_candidate_ids = None
        support_candidate_prelude_steps = int(cfg.get("audit_support_candidate_prelude_steps", 0))
        support_candidate_swing_steps = int(cfg.get("audit_support_candidate_swing_steps", 30))
        if support_fd_epsilon < 0.0:
            raise ValueError("audit_support_fd_epsilon must be non-negative")
        if support_candidates_path is not None and support_fd_epsilon > 0.0:
            raise ValueError("support candidates and finite-difference audit are mutually exclusive")
        if support_candidate_prelude_steps < 0 or support_candidate_swing_steps <= 0:
            raise ValueError(
                "support candidate windows require prelude_steps >= 0 and swing_steps > 0"
            )
        if support_fd_epsilon > 0.0:
            expected_cases = motion.motion.num_motions * support_fd_variants
            if not audit_policy_action:
                raise ValueError("support finite-difference audit requires +audit_policy_action=true")
            if cases != expected_cases:
                raise ValueError(
                    "support finite-difference audit requires "
                    f"num_envs={expected_cases}, got {cases}"
                )
            print(
                "[train.py] support finite-difference audit: "
                f"{motion.motion.num_motions} motions x {support_fd_variants} variants, "
                f"epsilon={support_fd_epsilon:.4f}",
                flush=True,
            )
        if support_candidates_path is not None:
            if not audit_policy_action:
                raise ValueError("support candidate audit requires +audit_policy_action=true")
            if not support_candidates_path.is_file():
                raise FileNotFoundError(f"support candidate file does not exist: {support_candidates_path}")
            import numpy as np

            with np.load(support_candidates_path, allow_pickle=False) as candidate_data:
                if "motion_ids" not in candidate_data or "support_knots" not in candidate_data:
                    raise ValueError(
                        "support candidate NPZ requires motion_ids[N] and support_knots[N,K,15]"
                    )
                support_candidate_ids = torch.as_tensor(
                    candidate_data["motion_ids"], dtype=torch.long, device=raw.device
                )
                support_candidate_knots = torch.as_tensor(
                    candidate_data["support_knots"], dtype=torch.float, device=raw.device
                )
            if support_candidate_ids.shape != (cases,):
                raise ValueError(
                    f"candidate motion_ids must have shape ({cases},), got {tuple(support_candidate_ids.shape)}"
                )
            if (
                support_candidate_knots.ndim != 3
                or support_candidate_knots.shape[0] != cases
                or support_candidate_knots.shape[2] != support_fd_dims
            ):
                raise ValueError(
                    "candidate support_knots must have shape "
                    f"({cases}, K, {support_fd_dims}), got {tuple(support_candidate_knots.shape)}"
                )
            if support_candidate_knots.shape[1] < 1:
                raise ValueError("candidate support_knots requires at least one temporal knot")
            if not torch.isfinite(support_candidate_knots).all():
                raise ValueError("candidate support_knots contains non-finite values")
            if (
                support_candidate_ids.min().item() < 0
                or support_candidate_ids.max().item() >= motion.motion.num_motions
            ):
                raise ValueError("candidate motion_ids contains an invalid motion index")
            print(
                "[train.py] support candidate audit: "
                f"source={support_candidates_path} cases={cases} "
                f"knots={support_candidate_knots.shape[1]} "
                f"prelude_window={support_candidate_prelude_steps} "
                f"swing_window={support_candidate_swing_steps}",
                flush=True,
            )
        env.reset()
        if support_candidate_ids is not None:
            ids = support_candidate_ids
        elif motion.cfg.fixed_motion_id is not None:
            # A focused recovery task must audit the same route it trains.
            # The historical audit fan-out remains the default for ordinary
            # multi-motion tasks.
            ids = torch.full(
                (cases,), int(motion.cfg.fixed_motion_id), dtype=torch.long, device=raw.device
            )
        else:
            ids = torch.arange(cases, device=raw.device) % motion.motion.num_motions
        fd_variant = torch.arange(cases, device=raw.device) // motion.motion.num_motions
        fd_action_index = torch.where(
            fd_variant > 0,
            (fd_variant - 1) // 2,
            torch.full_like(fd_variant, -1),
        )
        fd_sign = torch.where(
            fd_variant > 0,
            torch.where((fd_variant - 1) % 2 == 0, 1.0, -1.0),
            torch.zeros_like(fd_variant, dtype=torch.float),
        )
        motion.motion_ids[:] = ids
        motion.time_steps.zero_()
        motion.tail_steps.zero_()
        motion.prelude_elapsed_steps.zero_()
        racket._resample_command(torch.arange(cases, device=raw.device))
        if not getattr(racket.cfg, "manifest_base_aligned", False):
            racket.racket_target_pos_w[:] = raw.scene.env_origins + motion.motion.strike_pos_w[ids]
        racket.racket_target_vel_w[:] = motion.motion.strike_vel_w[ids]
        racket.racket_target_normal_w[:] = motion.motion.strike_normal_w[ids]
        target_pos = racket.racket_target_pos_w.clone()
        target_vel = racket.racket_target_vel_w.clone()
        target_normal = racket.racket_target_normal_w.clone()
        hit = motion.motion.hit_frame[motion.motion_ids]
        robot = raw.scene["robot"]
        root0 = robot.data.root_pos_w.clone()
        root_max = torch.zeros(cases, device=raw.device)
        root_xy_max = torch.zeros(cases, device=raw.device)
        min_root_height = torch.full((cases,), float("inf"), device=raw.device)
        max_root_lin_vel = torch.zeros(cases, device=raw.device)
        max_root_ang_vel = torch.zeros(cases, device=raw.device)
        max_root_tilt_deg = torch.zeros(cases, device=raw.device)
        max_torque = torch.zeros(cases, device=raw.device)
        min_joint_margin = torch.full((cases,), float("inf"), device=raw.device)
        min_joint_margin_index = torch.full((cases,), -1, dtype=torch.long, device=raw.device)
        min_joint_margin_step = torch.full((cases,), -1, dtype=torch.long, device=raw.device)
        min_joint_value = torch.zeros(cases, device=raw.device)
        min_joint_lower_limit = torch.zeros(cases, device=raw.device)
        min_joint_upper_limit = torch.zeros(cases, device=raw.device)
        min_joint_target = torch.zeros(cases, device=raw.device)
        torque_saturation_count = torch.zeros(cases, dtype=torch.long, device=raw.device)
        velocity_saturation_count = torch.zeros(cases, dtype=torch.long, device=raw.device)
        max_foot_slip = torch.zeros(cases, device=raw.device)
        foot_contact_sum = torch.zeros(cases, device=raw.device)
        observed_steps = torch.zeros(cases, dtype=torch.long, device=raw.device)
        post_hit_steps = torch.zeros(cases, dtype=torch.long, device=raw.device)
        min_capture_front_margin = torch.full(
            (cases,), float("inf"), device=raw.device
        )
        min_capture_rear_margin = torch.full(
            (cases,), float("inf"), device=raw.device
        )
        max_capture_center_distance = torch.zeros(cases, device=raw.device)
        front_outside_steps = torch.zeros(cases, dtype=torch.long, device=raw.device)
        rear_outside_steps = torch.zeros(cases, dtype=torch.long, device=raw.device)
        first_front_outside_step = torch.full(
            (cases,), -1, dtype=torch.long, device=raw.device
        )
        first_rear_outside_step = torch.full(
            (cases,), -1, dtype=torch.long, device=raw.device
        )
        front_recentered = torch.zeros(cases, dtype=torch.bool, device=raw.device)
        rear_recentered = torch.zeros(cases, dtype=torch.bool, device=raw.device)
        max_forward_velocity = torch.zeros(cases, device=raw.device)
        max_backward_velocity = torch.zeros(cases, device=raw.device)
        max_forward_pitch_rate = torch.zeros(cases, device=raw.device)
        max_backward_pitch_rate = torch.zeros(cases, device=raw.device)
        max_recovery_group_l2 = torch.zeros((cases, 3), device=raw.device)
        max_recovery_group_abs = torch.zeros((cases, 3), device=raw.device)
        active = torch.ones(cases, dtype=torch.bool, device=raw.device)
        finite = torch.ones(cases, dtype=torch.bool, device=raw.device)
        first_failure_step = torch.full((cases,), -1, dtype=torch.long, device=raw.device)
        termination_labels: list[list[str]] = [[] for _ in range(cases)]
        termination_counts = {
            name: torch.zeros(cases, dtype=torch.long, device=raw.device)
            for name in raw.termination_manager.active_terms
        }
        from training.robots.agibot_a3 import A3_FEET_BODIES

        contact_sensor = raw.scene.sensors["contact_forces"]
        foot_ids, resolved_feet = contact_sensor.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if resolved_feet != A3_FEET_BODIES:
            raise RuntimeError(
                f"zero-action audit foot sensor mismatch: expected={A3_FEET_BODIES}, got={resolved_feet}"
            )
        foot_initial_xy = robot.data.body_pos_w[:, foot_ids, :2].clone()
        action_term = raw.action_manager.get_term("joint_pos")
        backend_ids = torch.as_tensor(
            action_term._backend_joint_ids,
            dtype=torch.long,
            device=raw.device,
        )
        backend_joint_names = list(action_term.cfg.backend_joint_names)
        audit_trace_value = cfg.get("audit_trace_output", None)
        audit_trace_path = Path(str(audit_trace_value)) if audit_trace_value else None
        audit_trace: list[dict[str, Any]] = []
        audit_trace_after_hit_steps = int(cfg.get("audit_trace_after_hit_steps", 2))
        if audit_trace_after_hit_steps < 0:
            raise ValueError("audit_trace_after_hit_steps must be non-negative")
        audit_trace_full_episode = bool(cfg.get("audit_trace_full_episode", False))
        trace_joint_names = (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hip_pitch_joint",
            # The P11 bootstrap brace and recovery adapter have direct
            # authority over both hip-roll joints.  Keep them in every full
            # recovery trace: leaving them out makes a tilt failure impossible
            # to attribute to the causal lateral-support commands.
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        )
        trace_joint_backend_indices: dict[str, int] = {}
        if audit_trace_path is not None:
            missing_trace_joints = [name for name in trace_joint_names if name not in backend_joint_names]
            if missing_trace_joints:
                raise RuntimeError(f"audit trace joint mapping missing: {missing_trace_joints}")
            trace_joint_backend_indices = {
                name: backend_joint_names.index(name) for name in trace_joint_names
            }
        post_hit_required = int(cfg.get("audit_post_hit_steps", 20))
        if post_hit_required < 0:
            raise ValueError("audit_post_hit_steps must be non-negative")
        # A short post-hit window is sufficient for strike-only diagnostics,
        # but it can hide a fall that develops during an explicit hold/return
        # cycle.  Full-cycle qualification must run until the task's natural
        # timeout and treats every other termination as a failure.
        audit_full_episode = bool(cfg.get("audit_full_episode", False))
        if audit_full_episode:
            max_audit_steps = int(raw.max_episode_length)
        else:
            hit_deadline = int(motion.prelude_steps) + int(hit.max().item()) + post_hit_required + 2
            max_audit_steps = min(int(raw.max_episode_length), hit_deadline)
        from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
        exact = [None] * cases
        policy = None
        observation = None
        if audit_policy_action:
            # The reset above is deliberately followed by fixed motion IDs and
            # phase.  Build observations only after that synchronization so the
            # actor sees the same state that is about to be audited.
            policy = runner.get_inference_policy(device=raw.device)
            observation, _ = env.get_observations()
        zero = torch.zeros((cases, raw.action_manager.total_action_dim), device=raw.device)
        clean_timeout = torch.zeros(cases, dtype=torch.bool, device=raw.device)
        preview_adapter_action = torch.zeros_like(zero)
        recovery_adapter_action = torch.zeros_like(zero)
        recovery_decay_source_action = torch.zeros_like(zero)
        recovery_decay_trigger_step = torch.full(
            (cases,), -1, dtype=torch.long, device=raw.device
        )
        recovery_positive_forward_seen = torch.zeros(
            cases, dtype=torch.bool, device=raw.device
        )
        recovery_decay_factor = torch.ones(cases, device=raw.device)
        # Policy order is left leg, right leg, waist, right arm. Only retire
        # the learned V23 sagittal recovery pulse; frozen priors and lateral
        # stabilization remain untouched.
        recovery_decay_action_indices = torch.tensor(
            (0, 3, 4, 6, 9, 10, 14),
            dtype=torch.long,
            device=raw.device,
        )
        if (
            recovery_center_decay_steps > 0
            and recovery_center_decay_source == "stage_a"
        ):
            raw.stage_a_sagittal_audit_scale = torch.ones(
                cases, device=raw.device
            )
        elif abs(stage_a_front_gain - 1.0) > 1.0e-9:
            raw.stage_a_sagittal_audit_scale = torch.ones(
                cases, device=raw.device
            )
        for step in range(max_audit_steps):
            racket.racket_target_pos_w[:] = target_pos
            racket.racket_target_vel_w[:] = target_vel
            racket.racket_target_normal_w[:] = target_normal
            active_before_step = active.clone()
            if policy is None:
                action = zero
            else:
                with torch.inference_mode():
                    action = policy(observation)
                    if momentum_preview_task:
                        normalized = runner.obs_normalizer(observation)
                        v19_policy = runner.alg.policy
                        state_features = v19_policy.support_state_encoder(normalized[:, :204])
                        preview_features = v19_policy.preview_encoder(normalized[:, 204:])
                        preview_modulation = 1.0 + torch.tanh(
                            v19_policy.preview_state_gate(state_features)
                        )
                        preview_delta = v19_policy.preview_adapter(
                            preview_features * preview_modulation
                        )
                        preview_adapter_action.zero_()
                        preview_adapter_action[:, :15] = preview_delta
                    if gated_recovery_task:
                        normalized = runner.obs_normalizer(observation)
                        recovery_policy = runner.alg.policy
                        recovery_base_action = recovery_policy.base_action_mean(
                            normalized[:, : recovery_policy.BASE_OBS_DIM]
                        )
                        recovery_adapter_action = (
                            recovery_policy.act_inference(normalized)
                            - recovery_base_action
                        )
                        if recovery_center_decay_steps > 0:
                            from training.tasks.tracking.mdp.observations import (
                                stagger_support_state,
                            )

                            recovery_state = stagger_support_state(raw)
                            post_hit = torch.tensor(
                                [row is not None for row in exact],
                                dtype=torch.bool,
                                device=raw.device,
                            )
                            forward_velocity = robot.data.root_lin_vel_b[:, 0]
                            recovery_positive_forward_seen |= post_hit & (
                                forward_velocity > 0.0
                            )
                            centered = (
                                torch.abs(
                                    recovery_state["capture_rel_support_x_b"]
                                )
                                <= recovery_center_half_width_m
                            )
                            trigger = (
                                active_before_step
                                & post_hit
                                & recovery_positive_forward_seen
                                & centered
                                & (forward_velocity <= 0.0)
                                & (recovery_decay_trigger_step < 0)
                            )
                            recovery_decay_trigger_step = torch.where(
                                trigger,
                                torch.full_like(
                                    recovery_decay_trigger_step, step + 1
                                ),
                                recovery_decay_trigger_step,
                            )
                            triggered = recovery_decay_trigger_step >= 0
                            elapsed = (
                                (step + 1) - recovery_decay_trigger_step
                            ).clamp(min=0).to(dtype=action.dtype)
                            u = (
                                elapsed / float(recovery_center_decay_steps)
                            ).clamp(0.0, 1.0)
                            smooth = u * u * (3.0 - 2.0 * u)
                            recovery_decay_factor = torch.where(
                                triggered, 1.0 - smooth, torch.ones_like(smooth)
                            )
                            if (
                                recovery_center_decay_source
                                == "recovery_adapter"
                            ):
                                recovery_decay_source_action = (
                                    recovery_adapter_action
                                )
                            elif (
                                recovery_center_decay_source
                                == "v22_support_adapter"
                            ):
                                base_observation = normalized[
                                    :, : recovery_policy.BASE_OBS_DIM
                                ]
                                support_features = torch.cat(
                                    (
                                        recovery_policy.support_state_encoder(
                                            base_observation[
                                                :, : recovery_policy.legacy_obs_dim
                                            ]
                                        ),
                                        recovery_policy.stagger_encoder(
                                            base_observation[
                                                :, recovery_policy.legacy_obs_dim :
                                            ]
                                        ),
                                    ),
                                    dim=-1,
                                )
                                support_delta = (
                                    recovery_policy.support_adapter(
                                        recovery_policy.support_fusion(
                                            support_features
                                        )
                                    )
                                )
                                recovery_decay_source_action = (
                                    torch.zeros_like(action)
                                )
                                recovery_decay_source_action[:, :15] = (
                                    support_delta
                                )
                            elif recovery_center_decay_source == "stage_a":
                                recovery_decay_source_action = torch.zeros_like(
                                    action
                                )
                                raw.stage_a_sagittal_audit_scale[:] = (
                                    recovery_decay_factor
                                )
                            else:
                                recovery_decay_source_action = action
                            action = action.clone()
                            selected_delta = (
                                recovery_decay_source_action.index_select(
                                    1, recovery_decay_action_indices
                                )
                            )
                            action[:, recovery_decay_action_indices] -= (
                                (1.0 - recovery_decay_factor).unsqueeze(-1)
                                * selected_delta
                            )
                        for group_index, (start, end) in enumerate(
                            ((0, 12), (12, 15), (15, 22))
                        ):
                            group = recovery_adapter_action[:, start:end]
                            max_recovery_group_l2[:, group_index] = torch.maximum(
                                max_recovery_group_l2[:, group_index],
                                torch.linalg.vector_norm(group, dim=-1),
                            )
                            max_recovery_group_abs[:, group_index] = torch.maximum(
                                max_recovery_group_abs[:, group_index],
                                torch.abs(group).max(dim=-1).values,
                            )
                    if abs(stage_a_front_gain - 1.0) > 1.0e-9:
                        from training.tasks.tracking.mdp.observations import (
                            stagger_support_state,
                        )

                        front_support = stagger_support_state(raw)
                        # The lead is phase-based rather than motion-ID based.
                        # It answers whether front-side support is arriving too
                        # late, while keeping prelude behavior untouched.
                        in_swing = (
                            motion.prelude_elapsed_steps
                            >= int(motion.prelude_steps)
                        )
                        front_window = in_swing & (
                            motion.time_steps
                            >= (hit - stage_a_front_lead_steps).clamp(min=0)
                        )
                        both_feet = front_support["contacts"].all(dim=-1)
                        front_risk = (
                            active_before_step
                            & front_window
                            & both_feet
                            & (
                                front_support["capture_front_margin"]
                                <= stage_a_front_margin_m
                            )
                            & (
                                robot.data.root_lin_vel_b[:, 0]
                                >= stage_a_front_velocity_mps
                            )
                        )
                        raw.stage_a_sagittal_audit_scale[:] = torch.where(
                            front_risk,
                            torch.full_like(
                                raw.stage_a_sagittal_audit_scale,
                                stage_a_front_gain,
                            ),
                            torch.ones_like(raw.stage_a_sagittal_audit_scale),
                        )
            if support_fd_epsilon > 0.0:
                # The environment updates command phase after physics. Apply the
                # perturbation only after prelude has completed and when the
                # upcoming physical sample lies in the hit-30 through exact-hit
                # window.  During prelude motion.time_steps is held at zero, so
                # omitting the explicit prelude gate would silently apply a
                # hit-30 perturbation throughout the ready-pose transition.
                action = action.clone()
                upcoming_phase = motion.time_steps + 1
                in_swing = motion.prelude_elapsed_steps >= int(motion.prelude_steps)
                fd_window = (
                    in_swing
                    & (upcoming_phase >= torch.clamp(hit - 30, min=0))
                    & (upcoming_phase <= hit)
                )
                fd_active = fd_window & (fd_variant > 0)
                fd_env_ids = torch.nonzero(fd_active, as_tuple=False).flatten()
                if fd_env_ids.numel() > 0:
                    action[fd_env_ids, fd_action_index[fd_env_ids]] += (
                        support_fd_epsilon * fd_sign[fd_env_ids]
                    )
                    action.clamp_(-1.0, 1.0)
            if support_candidate_knots is not None:
                # Candidate trajectories are additive raw support corrections.
                # Prelude and swing windows are represented explicitly so a
                # candidate cannot enter prelude merely because motion phase is
                # held at zero there.
                action = action.clone()
                prelude_elapsed = motion.prelude_elapsed_steps
                prelude_total = int(motion.prelude_steps)
                upcoming_phase = motion.time_steps + 1
                swing_start = torch.clamp(hit - support_candidate_swing_steps, min=0)
                effective_swing_steps = (hit - swing_start).clamp_min(1)
                in_prelude_window = (
                    (support_candidate_prelude_steps > 0)
                    & (prelude_elapsed < prelude_total)
                    & (
                        prelude_elapsed + 1
                        > prelude_total - support_candidate_prelude_steps
                    )
                )
                in_swing_window = (
                    (prelude_elapsed >= prelude_total)
                    & (upcoming_phase >= swing_start)
                    & (upcoming_phase <= hit)
                )
                candidate_active = in_prelude_window | in_swing_window
                prelude_progress = (
                    prelude_elapsed + 1 - (prelude_total - support_candidate_prelude_steps)
                ).clamp(min=0, max=max(support_candidate_prelude_steps, 1))
                swing_progress = support_candidate_prelude_steps + (
                    upcoming_phase - swing_start
                ).clamp(min=0)
                combined_progress = torch.where(
                    in_prelude_window,
                    prelude_progress,
                    swing_progress,
                ).float()
                total_progress = (
                    support_candidate_prelude_steps + effective_swing_steps
                ).float().clamp_min(1.0)
                knot_position = (
                    combined_progress
                    / total_progress
                    * (support_candidate_knots.shape[1] - 1)
                )
                left_index = torch.floor(knot_position).to(torch.long).clamp(
                    0, support_candidate_knots.shape[1] - 1
                )
                right_index = (left_index + 1).clamp(
                    0, support_candidate_knots.shape[1] - 1
                )
                blend = (knot_position - left_index.float()).unsqueeze(-1)
                row_index = torch.arange(cases, device=raw.device)
                candidate_correction = (
                    support_candidate_knots[row_index, left_index] * (1.0 - blend)
                    + support_candidate_knots[row_index, right_index] * blend
                )
                action[:, :support_fd_dims] += (
                    candidate_correction * candidate_active.unsqueeze(-1)
                )
                action.clamp_(-1.0, 1.0)
            observation, _, _, _ = env.step(action)
            root_delta = robot.data.root_pos_w - root0
            root_max = torch.where(
                active_before_step,
                torch.maximum(root_max, torch.linalg.vector_norm(root_delta, dim=-1)),
                root_max,
            )
            root_xy_max = torch.where(
                active_before_step,
                torch.maximum(root_xy_max, torch.linalg.vector_norm(root_delta[:, :2], dim=-1)),
                root_xy_max,
            )
            min_root_height = torch.where(
                active_before_step,
                torch.minimum(min_root_height, robot.data.root_pos_w[:, 2]),
                min_root_height,
            )
            max_root_lin_vel = torch.where(
                active_before_step,
                torch.maximum(max_root_lin_vel, torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1)),
                max_root_lin_vel,
            )
            max_root_ang_vel = torch.where(
                active_before_step,
                torch.maximum(max_root_ang_vel, torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1)),
                max_root_ang_vel,
            )
            root_tilt_deg = torch.rad2deg(torch.arccos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)))
            max_root_tilt_deg = torch.where(
                active_before_step,
                torch.maximum(max_root_tilt_deg, root_tilt_deg),
                max_root_tilt_deg,
            )
            max_torque = torch.where(
                active_before_step,
                torch.maximum(
                    max_torque,
                    torch.abs(robot.data.applied_torque[:, backend_ids]).max(dim=-1).values,
                ),
                max_torque,
            )
            soft_limits = robot.data.soft_joint_pos_limits[:, backend_ids]
            joint_margin_by_joint = torch.minimum(
                robot.data.joint_pos[:, backend_ids] - soft_limits[..., 0],
                soft_limits[..., 1] - robot.data.joint_pos[:, backend_ids],
            )
            joint_margin, joint_margin_index = joint_margin_by_joint.min(dim=-1)
            new_min_margin = active_before_step & (joint_margin < min_joint_margin)
            min_joint_margin = torch.where(
                active_before_step, torch.minimum(min_joint_margin, joint_margin), min_joint_margin
            )
            min_joint_margin_index = torch.where(new_min_margin, joint_margin_index, min_joint_margin_index)
            min_joint_margin_step = torch.where(
                new_min_margin, torch.full_like(min_joint_margin_step, step + 1), min_joint_margin_step
            )
            selected_index = joint_margin_index.unsqueeze(-1)
            min_joint_value = torch.where(
                new_min_margin,
                robot.data.joint_pos[:, backend_ids].gather(1, selected_index).squeeze(-1),
                min_joint_value,
            )
            min_joint_lower_limit = torch.where(
                new_min_margin, soft_limits[..., 0].gather(1, selected_index).squeeze(-1), min_joint_lower_limit
            )
            min_joint_upper_limit = torch.where(
                new_min_margin, soft_limits[..., 1].gather(1, selected_index).squeeze(-1), min_joint_upper_limit
            )
            min_joint_target = torch.where(
                new_min_margin,
                action_term.full_joint_targets[:, backend_ids].gather(1, selected_index).squeeze(-1),
                min_joint_target,
            )
            torque_saturation_count += (
                (torch.abs(robot.data.applied_torque[:, backend_ids]) >= 0.95 * robot.data.joint_effort_limits[:, backend_ids])
                .sum(dim=-1)
                * active_before_step
            )
            velocity_saturation_count += (
                (torch.abs(robot.data.joint_vel[:, backend_ids]) >= 0.95 * robot.data.joint_vel_limits[:, backend_ids])
                .sum(dim=-1)
                * active_before_step
            )
            foot_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, foot_ids], dim=-1)
            foot_contact = foot_force > 10.0
            foot_tangential_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, foot_ids, :2], dim=-1
            )
            max_foot_slip = torch.where(
                active_before_step,
                torch.maximum(
                    max_foot_slip,
                    torch.where(foot_contact, foot_tangential_speed, torch.zeros_like(foot_tangential_speed))
                    .max(dim=-1)
                    .values,
                ),
                max_foot_slip,
            )
            foot_contact_sum += foot_contact.float().mean(dim=-1) * active_before_step
            observed_steps += active_before_step.to(torch.long)
            termination_masks = {
                name: raw.termination_manager.get_term(name).clone()
                for name in raw.termination_manager.active_terms
            }
            manager_done = torch.zeros(cases, dtype=torch.bool, device=raw.device)
            for mask in termination_masks.values():
                manager_done |= mask.to(torch.bool) & active_before_step
            if stagger_support_task:
                from training.tasks.tracking.mdp.observations import stagger_support_state

                support_state = stagger_support_state(raw)
                exact_before_metrics = torch.tensor(
                    [row is not None for row in exact],
                    dtype=torch.bool,
                    device=raw.device,
                )
                # Isaac resets terminated environments inside env.step(). Excluding
                # those rows prevents the reset pose from looking like a recovery.
                recovery_active = active_before_step & exact_before_metrics & ~manager_done
                front_margin = support_state["capture_front_margin"]
                rear_margin = support_state["capture_rear_margin"]
                capture_distance = torch.abs(
                    support_state["capture_rel_support_x_b"]
                )
                min_capture_front_margin = torch.where(
                    recovery_active,
                    torch.minimum(min_capture_front_margin, front_margin),
                    min_capture_front_margin,
                )
                min_capture_rear_margin = torch.where(
                    recovery_active,
                    torch.minimum(min_capture_rear_margin, rear_margin),
                    min_capture_rear_margin,
                )
                max_capture_center_distance = torch.where(
                    recovery_active,
                    torch.maximum(max_capture_center_distance, capture_distance),
                    max_capture_center_distance,
                )
                front_outside = recovery_active & (front_margin < 0.0)
                rear_outside = recovery_active & (rear_margin < 0.0)
                front_outside_steps += front_outside.to(torch.long)
                rear_outside_steps += rear_outside.to(torch.long)
                first_front = front_outside & (first_front_outside_step < 0)
                first_rear = rear_outside & (first_rear_outside_step < 0)
                first_front_outside_step = torch.where(
                    first_front,
                    torch.full_like(first_front_outside_step, step + 1),
                    first_front_outside_step,
                )
                first_rear_outside_step = torch.where(
                    first_rear,
                    torch.full_like(first_rear_outside_step, step + 1),
                    first_rear_outside_step,
                )
                centered = recovery_active & (capture_distance <= 0.04)
                front_recentered |= centered & (first_front_outside_step >= 0)
                rear_recentered |= centered & (first_rear_outside_step >= 0)
                forward_velocity = robot.data.root_lin_vel_b[:, 0]
                pitch_rate = robot.data.root_ang_vel_b[:, 1]
                max_forward_velocity = torch.where(
                    recovery_active,
                    torch.maximum(max_forward_velocity, torch.relu(forward_velocity)),
                    max_forward_velocity,
                )
                max_backward_velocity = torch.where(
                    recovery_active,
                    torch.maximum(max_backward_velocity, torch.relu(-forward_velocity)),
                    max_backward_velocity,
                )
                max_forward_pitch_rate = torch.where(
                    recovery_active,
                    torch.maximum(max_forward_pitch_rate, torch.relu(pitch_rate)),
                    max_forward_pitch_rate,
                )
                max_backward_pitch_rate = torch.where(
                    recovery_active,
                    torch.maximum(max_backward_pitch_rate, torch.relu(-pitch_rate)),
                    max_backward_pitch_rate,
                )
            finite &= (
                torch.isfinite(robot.data.root_state_w).all(dim=-1)
                & torch.isfinite(robot.data.joint_pos).all(dim=-1)
                & torch.isfinite(robot.data.joint_vel).all(dim=-1)
            )

            done = torch.zeros(cases, dtype=torch.bool, device=raw.device)
            failed = torch.zeros(cases, dtype=torch.bool, device=raw.device)
            for name, mask in termination_masks.items():
                mask = mask.to(torch.bool) & active_before_step
                termination_counts[name] += mask.to(torch.long)
                done |= mask
                for env_id in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                    termination_labels[env_id].append(name)
                if name == "time_out" and audit_full_episode:
                    clean_timeout |= mask
                else:
                    failed |= mask
            failed |= ~finite & active_before_step
            done |= ~finite & active_before_step
            for env_id in torch.nonzero(failed & (first_failure_step < 0), as_tuple=False).flatten().tolist():
                first_failure_step[env_id] = step + 1
                if not finite[env_id]:
                    termination_labels[env_id].append("non_finite_state")

            exact_before_step = torch.tensor(
                [row is not None for row in exact], dtype=torch.bool, device=raw.device
            )
            at_hit = (
                active_before_step
                & ~done
                & (motion.prelude_elapsed_steps >= int(motion.prelude_steps))
                & (motion.time_steps == hit)
            )
            racket._compute_racket_state()
            if audit_trace_path is not None:
                stagger_trace_state = None
                if stagger_support_task:
                    from training.tasks.tracking.mdp.observations import stagger_support_state

                    stagger_trace_state = stagger_support_state(raw)
                # Default to the compact pre-hit diagnostic window. Full-cycle
                # traces are opt-in because they are used to diagnose delayed
                # falls during the hold/return/ready segments.
                current_steps = motion.time_steps.clone()
                trace_window = (
                    active_before_step
                    & (
                        torch.ones_like(active_before_step)
                        if audit_trace_full_episode
                        else (
                            (motion.prelude_elapsed_steps >= int(motion.prelude_steps))
                            & (current_steps >= hit - 15)
                            & (current_steps <= hit + audit_trace_after_hit_steps)
                        )
                    )
                )
                for env_id in torch.nonzero(trace_window, as_tuple=False).flatten().tolist():
                    motion_id = int(ids[env_id].item())
                    motion_step = int(current_steps[env_id].item())
                    joints = {}
                    for name, backend_index in trace_joint_backend_indices.items():
                        sim_joint_index = int(backend_ids[backend_index].item())
                        joints[name] = {
                            "reference_pos_rad": float(
                                motion.motion.joint_pos[motion_id, motion_step, sim_joint_index].item()
                            ),
                            "target_pos_rad": float(
                                action_term.full_joint_targets[env_id, sim_joint_index].item()
                            ),
                            "target_vel_radps": float(
                                action_term._full_joint_velocity_targets[env_id, sim_joint_index].item()
                            ),
                            "actual_pos_rad": float(robot.data.joint_pos[env_id, sim_joint_index].item()),
                            "actual_vel_radps": float(robot.data.joint_vel[env_id, sim_joint_index].item()),
                            "applied_torque_nm": float(robot.data.applied_torque[env_id, sim_joint_index].item()),
                            "torque_limit_nm": float(robot.data.joint_effort_limits[env_id, sim_joint_index].item()),
                            "velocity_limit_radps": float(robot.data.joint_vel_limits[env_id, sim_joint_index].item()),
                        }
                    coordinator_groups = {}
                    # The coordinator has one policy but three physically different
                    # correction groups.  Keep them separate in full-cycle traces so
                    # a delayed fall can be attributed to its initiating channel.
                    for group_name, start, end in (
                        ("leg", 0, 12),
                        ("waist", 12, 15),
                        ("arm", 15, 22),
                    ):
                        raw_group = action_term.raw_actions[env_id, start:end]
                        processed_group = action_term.processed_actions[env_id, start:end]
                        coordinator_groups[group_name] = {
                            "raw_l2": float(torch.linalg.vector_norm(raw_group).item()),
                            "raw_max_abs": float(raw_group.abs().max().item()),
                            "physical_l2_rad": float(torch.linalg.vector_norm(processed_group).item()),
                            "physical_max_abs_rad": float(processed_group.abs().max().item()),
                        }
                    waist_command = {}
                    upper_names = tuple(getattr(action_term.cfg, "upper_joint_names", ()) or ())
                    for joint_name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
                        if joint_name not in upper_names:
                            continue
                        upper_index = upper_names.index(joint_name)
                        waist_command[joint_name] = {
                            "reference_rad": float(action_term.upper_reference_actions[env_id, upper_index].item()),
                            "frozen_primary_residual_rad": float(
                                action_term.upper_primary_contribution[env_id, upper_index].item()
                            ),
                            "coordinator_residual_rad": float(
                                action_term.upper_coordinator_contribution[env_id, upper_index].item()
                            ),
                            "safety_override_rad": float(
                                action_term.upper_safety_override[env_id, upper_index].item()
                            ),
                            "velocity_safety_override_radps": float(
                                action_term.upper_velocity_safety_override[env_id, upper_index].item()
                            ),
                            "final_target_rad": float(
                                action_term.upper_processed_actions[env_id, upper_index].item()
                            ),
                            "actual_rad": float(robot.data.joint_pos[env_id, backend_ids[upper_index]].item()),
                        }
                    foot_xy = robot.data.body_pos_w[env_id, foot_ids, :2]
                    foot_displacement = torch.linalg.vector_norm(foot_xy - foot_initial_xy[env_id], dim=-1)
                    foot_force_row = foot_force[env_id]
                    foot_contact_row = foot_contact[env_id]
                    foot_speed_row = foot_tangential_speed[env_id]
                    # Two feet define a center-line support proxy rather than a
                    # fictitious point-support polygon.  Positive values mean
                    # the root projection is within a conservative 10 cm tube
                    # around loaded-foot centers; the raw foot data is also
                    # retained for a full support-polygon analysis offline.
                    root_xy = robot.data.root_pos_w[env_id, :2]
                    support_radius = 0.10
                    if bool(foot_contact_row.all()):
                        segment = foot_xy[1] - foot_xy[0]
                        denom = torch.dot(segment, segment).clamp_min(1.0e-8)
                        alpha = (torch.dot(root_xy - foot_xy[0], segment) / denom).clamp(0.0, 1.0)
                        support_distance = torch.linalg.vector_norm(root_xy - (foot_xy[0] + alpha * segment))
                    elif bool(foot_contact_row.any()):
                        support_distance = torch.linalg.vector_norm(root_xy - foot_xy[foot_contact_row][0])
                    else:
                        support_distance = torch.tensor(float("inf"), device=raw.device)
                    feet_trace = [
                        {
                            "name": resolved_feet[index],
                            "contact": bool(foot_contact_row[index].item()),
                            "force_norm_n": float(foot_force_row[index].item()),
                            "tangential_speed_mps": float(foot_speed_row[index].item()),
                            "displacement_from_reset_m": float(foot_displacement[index].item()),
                            "position_xy_w_m": [float(value) for value in foot_xy[index].tolist()],
                        }
                        for index in range(len(resolved_feet))
                    ]
                    root_roll, root_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                    root_roll = wrap_to_pi(root_roll)
                    root_pitch = wrap_to_pi(root_pitch)
                    stagger_support = None
                    if stagger_trace_state is not None:
                        stagger_support = {
                            "foot_rel_root_b_m": [
                                [float(value) for value in row]
                                for row in stagger_trace_state["foot_rel_root_b"][env_id].tolist()
                            ],
                            "com_rel_support_b_m": [
                                float(value)
                                for value in stagger_trace_state["com_rel_support_b"][env_id].tolist()
                            ],
                            "capture_rel_support_x_m": float(
                                stagger_trace_state["capture_rel_support_x_b"][env_id].item()
                            ),
                            "capture_front_margin_m": float(
                                stagger_trace_state["capture_front_margin"][env_id].item()
                            ),
                            "capture_rear_margin_m": float(
                                stagger_trace_state["capture_rear_margin"][env_id].item()
                            ),
                            "normalized_foot_load": [
                                float(value)
                                for value in stagger_trace_state["normalized_load"][env_id].tolist()
                            ],
                            "load_balance_left_minus_right": float(
                                stagger_trace_state["load_balance"][env_id].item()
                            ),
                            "total_load_body_weight_ratio": float(
                                stagger_trace_state["total_load_ratio"][env_id].item()
                            ),
                            "sagittal_span_m": float(
                                stagger_trace_state["sagittal_span"][env_id].item()
                            ),
                        }
                    audit_trace.append(
                        {
                            "motion_id": motion_id,
                            "control_step": step + 1,
                            "motion_step": motion_step,
                            "tail_steps": int(motion.tail_steps[env_id].item()),
                            "prelude_elapsed_steps": int(motion.prelude_elapsed_steps[env_id].item()),
                            "relative_to_hit_steps": motion_step - int(hit[env_id].item()),
                            "root_height_m": float(robot.data.root_pos_w[env_id, 2].item()),
                            "root_linear_speed_mps": float(
                                torch.linalg.vector_norm(robot.data.root_lin_vel_b[env_id]).item()
                            ),
                            "root_angular_speed_radps": float(
                                torch.linalg.vector_norm(robot.data.root_ang_vel_b[env_id]).item()
                            ),
                            "root_roll_rad": float(root_roll[env_id].item()),
                            "root_pitch_rad": float(root_pitch[env_id].item()),
                            "root_forward_velocity_mps": float(robot.data.root_lin_vel_b[env_id, 0].item()),
                            "root_lateral_velocity_mps": float(robot.data.root_lin_vel_b[env_id, 1].item()),
                            "root_pitch_rate_radps": float(robot.data.root_ang_vel_b[env_id, 1].item()),
                            "root_roll_rate_radps": float(robot.data.root_ang_vel_b[env_id, 0].item()),
                            "root_tilt_deg": float(root_tilt_deg[env_id].item()),
                            "root_support_centerline_margin_m": float((support_radius - support_distance).item()),
                            "stagger_support": stagger_support,
                            "feet": feet_trace,
                            "coordinator_raw_l2": float(
                                torch.linalg.vector_norm(action_term.raw_actions[env_id]).item()
                            ),
                            "coordinator_raw_max_abs": float(
                                action_term.raw_actions[env_id].abs().max().item()
                            ),
                            "coordinator_groups": coordinator_groups,
                            "stage_a_sagittal_exit": {
                                "scale": float(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_exit_scale",
                                        torch.ones(cases, device=raw.device),
                                    )[env_id].item()
                                ),
                                "state": int(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_exit_state",
                                        torch.zeros(
                                            cases,
                                            dtype=torch.long,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "trigger_step": int(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_exit_trigger_step",
                                        torch.full(
                                            (cases,),
                                            -1,
                                            dtype=torch.long,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "front_gain": float(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_front_gain",
                                        torch.ones(cases, device=raw.device),
                                    )[env_id].item()
                                ),
                                "rearm_ready": bool(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_rearm_ready",
                                        torch.zeros(
                                            cases,
                                            dtype=torch.bool,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "rearm_stable": bool(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_rearm_stable",
                                        torch.zeros(
                                            cases,
                                            dtype=torch.bool,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "rearm_stable_steps": int(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_rearm_stable_steps",
                                        torch.zeros(
                                            cases,
                                            dtype=torch.long,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "rearm_rejected": bool(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_rearm_rejected",
                                        torch.zeros(
                                            cases,
                                            dtype=torch.bool,
                                            device=raw.device,
                                        ),
                                    )[env_id].item()
                                ),
                                "audit_scale": float(
                                    getattr(
                                        raw,
                                        "stage_a_sagittal_audit_scale",
                                        torch.ones(cases, device=raw.device),
                                    )[env_id].item()
                                ),
                                "raw_action": [
                                    float(value)
                                    for value in raw.legacy_stage_a_last_action[
                                        env_id, :12
                                    ].tolist()
                                ],
                            },
                            "waist_command": waist_command,
                            "racket_actual_velocity_mps": [
                                float(value) for value in racket.racket_lin_vel_w[env_id].tolist()
                            ],
                            "racket_target_velocity_mps": [
                                float(value) for value in racket.racket_target_vel_w[env_id].tolist()
                            ],
                            "joints": joints,
                        }
                    )
            for env_id in torch.nonzero(at_hit, as_tuple=False).flatten().tolist():
                if exact[env_id] is None:
                    position_error = racket.racket_target_pos_w[env_id] - racket.racket_pos_w[env_id]
                    velocity_error = racket.racket_target_vel_w[env_id] - racket.racket_lin_vel_w[env_id]
                    _, exact_root_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[env_id : env_id + 1])
                    exact_root_pitch = wrap_to_pi(exact_root_pitch)[0]
                    dot = torch.clamp(
                        torch.sum(racket.racket_target_normal_w[env_id] * racket.racket_normal_w[env_id]),
                        -1.0,
                        1.0,
                    )
                    exact[env_id] = {
                        "motion_id": int(ids[env_id].item()),
                        "position_error_m": float(torch.linalg.vector_norm(position_error).item()),
                        "position_error_x_m": float(position_error[0].item()),
                        "position_error_y_m": float(position_error[1].item()),
                        "position_error_z_m": float(position_error[2].item()),
                        "velocity_error_mps": float(torch.linalg.vector_norm(velocity_error).item()),
                        "velocity_error_x_mps": float(velocity_error[0].item()),
                        "velocity_error_y_mps": float(velocity_error[1].item()),
                        "velocity_error_z_mps": float(velocity_error[2].item()),
                        "racket_velocity_x_mps": float(racket.racket_lin_vel_w[env_id, 0].item()),
                        "racket_velocity_y_mps": float(racket.racket_lin_vel_w[env_id, 1].item()),
                        "racket_velocity_z_mps": float(racket.racket_lin_vel_w[env_id, 2].item()),
                        "target_velocity_x_mps": float(racket.racket_target_vel_w[env_id, 0].item()),
                        "target_velocity_y_mps": float(racket.racket_target_vel_w[env_id, 1].item()),
                        "target_velocity_z_mps": float(racket.racket_target_vel_w[env_id, 2].item()),
                        "normal_error_deg": float(torch.rad2deg(torch.arccos(dot)).item()),
                        "root_displacement_m": float(root_max[env_id].item()),
                        "root_forward_velocity_mps_at_hit": float(
                            robot.data.root_lin_vel_b[env_id, 0].item()
                        ),
                        "root_pitch_rate_radps_at_hit": float(
                            robot.data.root_ang_vel_b[env_id, 1].item()
                        ),
                        "root_pitch_rad_at_hit": float(exact_root_pitch.item()),
                        "hit_control_step": step + 1,
                    }
                    if momentum_preview_task:
                        adapter_row = preview_adapter_action[env_id]
                        exact[env_id]["preview_adapter_support_l2"] = float(
                            torch.linalg.vector_norm(adapter_row[:15]).item()
                        )
                        exact[env_id]["preview_adapter_support_max_abs"] = float(
                            adapter_row[:15].abs().max().item()
                        )
                        exact[env_id]["preview_adapter_arm_max_abs"] = float(
                            adapter_row[15:].abs().max().item()
                        )
                        exact[env_id]["preview_adapter_support_action"] = [
                            float(value) for value in adapter_row[:15].tolist()
                        ]
                    # For the joint coordinator, record what the learned actor
                    # actually contributed in physical radians.  This separates
                    # a reward/authority failure (no useful correction) from a
                    # downstream PD tracking failure (correction issued but not
                    # realized by the mechanism).
                    raw_action = action_term.raw_actions[env_id]
                    processed_action = action_term.processed_actions[env_id]
                    if raw_action.numel() == 22 and processed_action.numel() == 22:
                        exact[env_id]["coordinator_support_raw_action"] = [
                            float(value) for value in raw_action[:15].tolist()
                        ]
                        for group, start, end in (("leg", 0, 12), ("waist", 12, 15), ("arm", 15, 22)):
                            raw_group = raw_action[start:end]
                            physical_group = processed_action[start:end]
                            exact[env_id][f"coordinator_{group}_raw_l2"] = float(
                                torch.linalg.vector_norm(raw_group).item()
                            )
                            exact[env_id][f"coordinator_{group}_raw_max_abs"] = float(
                                raw_group.abs().max().item()
                            )
                            exact[env_id][f"coordinator_{group}_physical_l2_rad"] = float(
                                torch.linalg.vector_norm(physical_group).item()
                            )
                            exact[env_id][f"coordinator_{group}_physical_max_abs_rad"] = float(
                                physical_group.abs().max().item()
                            )

                    # The final target includes both frozen priors and the new
                    # correction.  Rank the observed target-tracking errors at
                    # hit so a stable whole-body rollout cannot hide a single
                    # shoulder or waist joint that missed its command.
                    tracking_error = (
                        robot.data.joint_pos[env_id, backend_ids]
                        - action_term.full_joint_targets[env_id, backend_ids]
                    )
                    top_count = min(5, len(backend_joint_names))
                    top_values, top_indices = tracking_error.abs().topk(top_count)
                    exact[env_id]["largest_joint_target_errors"] = [
                        {
                            "joint": backend_joint_names[int(index.item())],
                            "actual_minus_target_rad": float(tracking_error[int(index.item())].item()),
                            "abs_error_rad": float(value.item()),
                        }
                        for value, index in zip(top_values, top_indices)
                    ]
            # Count steps strictly *after* exact hit; the hit frame itself is
            # a measurement event, not part of the settling window.
            post_hit_steps += (exact_before_step & active_before_step & ~done).to(torch.long)
            exact_row = torch.tensor(
                [row is not None for row in exact], dtype=torch.bool, device=raw.device
            )
            active &= ~done
            if audit_full_episode:
                completed = ~active
            else:
                completed = (exact_row & (post_hit_steps >= post_hit_required)) | (~active & (first_failure_step >= 0))
            if bool(completed.all()):
                break
        rows = []
        for env_id in range(cases):
            row = exact[env_id] or {"motion_id": int(ids[env_id].item())}
            hard_safety_pass = bool(
                exact[env_id] is not None
                and first_failure_step[env_id].item() < 0
                and (
                    clean_timeout[env_id].item()
                    if audit_full_episode
                    else post_hit_steps[env_id].item() >= post_hit_required
                )
                and min_root_height[env_id].item() >= 0.65
                and finite[env_id].item()
            )
            # A simulator reset can hide a fall, while a task's termination
            # thresholds can be more permissive than a reviewer expects.  Keep
            # this independent physical screen separate from hard termination:
            # it flags near-falls even if the episode never reset.
            stability_pass = bool(
                hard_safety_pass
                and max_root_tilt_deg[env_id].item() <= 30.0
                and foot_contact_sum[env_id].item() / max(observed_steps[env_id].item(), 1) >= 0.50
            )
            capture_recenter_pass = True
            if stagger_support_task:
                capture_recenter_pass = not (
                    (
                        front_outside_steps[env_id].item() > 0
                        and not front_recentered[env_id].item()
                    )
                    or (
                        rear_outside_steps[env_id].item() > 0
                        and not rear_recentered[env_id].item()
                    )
                )
                stability_pass = stability_pass and capture_recenter_pass
            row.update({
                "support_candidate_index": env_id if support_candidate_knots is not None else None,
                "support_fd_action_index": (
                    int(fd_action_index[env_id].item()) if fd_variant[env_id].item() > 0 else None
                ),
                "support_fd_sign": float(fd_sign[env_id].item()),
                "support_fd_epsilon": support_fd_epsilon,
                "safety_pass": hard_safety_pass,
                "stability_pass": stability_pass,
                "capture_recenter_pass": capture_recenter_pass,
                "first_failure_step": None if first_failure_step[env_id].item() < 0 else int(first_failure_step[env_id].item()),
                "termination_reasons": termination_labels[env_id],
                "termination_count_by_reason": {
                    name: int(count[env_id].item()) for name, count in termination_counts.items()
                },
                "clean_timeout": bool(clean_timeout[env_id].item()),
                "observed_steps": int(observed_steps[env_id].item()),
                "post_hit_steps_observed": int(post_hit_steps[env_id].item()),
                "minimum_root_height_m": float(min_root_height[env_id].item()),
                "max_root_displacement_m": float(root_max[env_id].item()),
                "max_root_xy_displacement_m": float(root_xy_max[env_id].item()),
                "max_root_linear_velocity_mps": float(max_root_lin_vel[env_id].item()),
                "max_root_angular_velocity_radps": float(max_root_ang_vel[env_id].item()),
                "max_root_tilt_deg": float(max_root_tilt_deg[env_id].item()),
                "foot_contact_fraction": float(foot_contact_sum[env_id].item() / max(observed_steps[env_id].item(), 1)),
                "max_loaded_foot_tangential_speed_mps": float(max_foot_slip[env_id].item()),
                "max_applied_torque_nm": float(max_torque[env_id].item()),
                "minimum_soft_joint_margin_rad": float(min_joint_margin[env_id].item()),
                "minimum_soft_joint_margin_joint": (
                    backend_joint_names[min_joint_margin_index[env_id].item()]
                    if min_joint_margin_index[env_id].item() >= 0 else None
                ),
                "minimum_soft_joint_margin_step": int(min_joint_margin_step[env_id].item()),
                "minimum_margin_joint_position_rad": float(min_joint_value[env_id].item()),
                "minimum_margin_joint_lower_limit_rad": float(min_joint_lower_limit[env_id].item()),
                "minimum_margin_joint_upper_limit_rad": float(min_joint_upper_limit[env_id].item()),
                "minimum_margin_joint_target_rad": float(min_joint_target[env_id].item()),
                "torque_saturation_fraction": float(
                    torque_saturation_count[env_id].item() / max(observed_steps[env_id].item() * len(backend_joint_names), 1)
                ),
                "velocity_saturation_fraction": float(
                    velocity_saturation_count[env_id].item() / max(observed_steps[env_id].item() * len(backend_joint_names), 1)
                ),
                "finite_state": bool(finite[env_id].item()),
            })
            if stagger_support_task:
                row.update(
                    {
                        "post_hit_min_capture_front_margin_m": float(
                            min_capture_front_margin[env_id].item()
                        ),
                        "post_hit_min_capture_rear_margin_m": float(
                            min_capture_rear_margin[env_id].item()
                        ),
                        "post_hit_max_capture_center_distance_m": float(
                            max_capture_center_distance[env_id].item()
                        ),
                        "post_hit_front_outside_steps": int(
                            front_outside_steps[env_id].item()
                        ),
                        "post_hit_rear_outside_steps": int(
                            rear_outside_steps[env_id].item()
                        ),
                        "post_hit_first_front_outside_step": (
                            None
                            if first_front_outside_step[env_id].item() < 0
                            else int(first_front_outside_step[env_id].item())
                        ),
                        "post_hit_first_rear_outside_step": (
                            None
                            if first_rear_outside_step[env_id].item() < 0
                            else int(first_rear_outside_step[env_id].item())
                        ),
                        "post_hit_recentered_after_front_exit": bool(
                            front_recentered[env_id].item()
                        ),
                        "post_hit_recentered_after_rear_exit": bool(
                            rear_recentered[env_id].item()
                        ),
                        "post_hit_max_forward_velocity_mps": float(
                            max_forward_velocity[env_id].item()
                        ),
                        "post_hit_max_backward_velocity_mps": float(
                            max_backward_velocity[env_id].item()
                        ),
                        "post_hit_max_forward_pitch_rate_radps": float(
                            max_forward_pitch_rate[env_id].item()
                        ),
                        "post_hit_max_backward_pitch_rate_radps": float(
                            max_backward_pitch_rate[env_id].item()
                        ),
                    }
                )
            if gated_recovery_task:
                row.update(
                    {
                        "recovery_center_decay_steps": recovery_center_decay_steps,
                        "recovery_center_decay_source": (
                            recovery_center_decay_source
                        ),
                        "recovery_center_decay_trigger_step": (
                            None
                            if recovery_decay_trigger_step[env_id].item() < 0
                            else int(recovery_decay_trigger_step[env_id].item())
                        ),
                        "recovery_center_decay_final_factor": float(
                            recovery_decay_factor[env_id].item()
                        ),
                        "recovery_adapter_max_leg_raw_l2": float(
                            max_recovery_group_l2[env_id, 0].item()
                        ),
                        "recovery_adapter_max_waist_raw_l2": float(
                            max_recovery_group_l2[env_id, 1].item()
                        ),
                        "recovery_adapter_max_arm_raw_l2": float(
                            max_recovery_group_l2[env_id, 2].item()
                        ),
                        "recovery_adapter_max_leg_raw_abs": float(
                            max_recovery_group_abs[env_id, 0].item()
                        ),
                        "recovery_adapter_max_waist_raw_abs": float(
                            max_recovery_group_abs[env_id, 1].item()
                        ),
                        "recovery_adapter_max_arm_raw_abs": float(
                            max_recovery_group_abs[env_id, 2].item()
                        ),
                    }
                )
            rows.append(row)
        if any(row["position_error_m"] is None for row in rows if "position_error_m" in row):
            raise RuntimeError("zero-action audit produced an invalid exact-hit metric")
        safety_pass_count = sum(row["safety_pass"] for row in rows)
        stability_pass_count = sum(row["stability_pass"] for row in rows)
        exact_rows = [row for row in rows if "position_error_m" in row]
        report = {
            "task": task_id,
            "action": "all_zero" if policy is None else "deterministic_checkpoint_actor",
            "audit_scope": {
                "motions": int(motion.motion.num_motions),
                "cases": int(cases),
                "support_fd_variants": support_fd_variants if support_fd_epsilon > 0.0 else 1,
                "support_fd_epsilon": support_fd_epsilon,
                "recovery_adapter_scale": recovery_adapter_scale,
                "recovery_center_decay": {
                    "steps": recovery_center_decay_steps,
                    "source": recovery_center_decay_source,
                    "capture_center_half_width_m": recovery_center_half_width_m,
                    "trigger": (
                        "post-hit, positive forward velocity previously observed, "
                        "current forward velocity <= 0, capture point within center band"
                    ),
                    "action_indices": [
                        int(value) for value in recovery_decay_action_indices.tolist()
                    ],
                },
                "support_candidates": (
                    {
                        "source": str(support_candidates_path),
                        "temporal_knots": int(support_candidate_knots.shape[1]),
                        "support_dimensions": support_fd_dims,
                        "prelude_window_steps": support_candidate_prelude_steps,
                        "swing_window_steps": support_candidate_swing_steps,
                    }
                    if support_candidate_knots is not None
                    else None
                ),
                "full_episode": audit_full_episode,
                "post_hit_steps_required": post_hit_required,
                "hard_safety_rule": (
                    "natural timeout only, finite state, root height >= 0.65 m through the full episode"
                    if audit_full_episode
                    else "no termination, finite state, root height >= 0.65 m through exact hit plus post-hit window"
                ),
                "stability_screen": "hard safety plus root tilt <= 30 deg and loaded-foot contact fraction >= 0.50",
            },
            "results": rows,
            "safety_pass_count": safety_pass_count,
            "safety_pass_fraction": safety_pass_count / len(rows),
            "stability_pass_count": stability_pass_count,
            "stability_pass_fraction": stability_pass_count / len(rows),
            "mean_position_error_m": (
                sum(row["position_error_m"] for row in exact_rows) / len(exact_rows)
                if exact_rows else None
            ),
            "mean_root_displacement_m": sum(row["max_root_displacement_m"] for row in rows) / len(rows),
        }
        audit_path = Path(str(cfg.get("audit_output", "eval_outputs/upper_contract/zero_action_audit.json")))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if audit_trace_path is not None:
            audit_trace_path.parent.mkdir(parents=True, exist_ok=True)
            audit_trace_report = {
                "purpose": (
                    "deterministic full-cycle actuator and recovery trace"
                    if audit_trace_full_episode
                    else "deterministic V2 pre-hit actuator trace"
                ),
                "checkpoint": str(agent_cfg.load_checkpoint),
                "task": task_id,
                "joint_names": trace_joint_names,
                "window": (
                    "all active control steps through timeout or physical termination"
                    if audit_trace_full_episode
                    else f"hit-15 through hit+{audit_trace_after_hit_steps} control steps"
                ),
                "rows": audit_trace,
            }
            audit_trace_path.write_text(json.dumps(audit_trace_report, indent=2), encoding="utf-8")
            print(f"[train.py] deterministic actuator trace: {audit_trace_path}", flush=True)
        print(f"[train.py] deterministic rollout audit: {json.dumps(report)}", flush=True)
        env.close()
        return

    # Natural-prefix P0 deliberately starts every environment from the formal
    # fresh reset.  The same restriction is mandatory for paired target
    # identification: siblings must traverse the identical motion phase, or
    # their difference is dominated by reference-swing timing rather than the
    # external target delta.
    paired_target_identification = False
    fixed_motion_recovery = False
    if hasattr(env.unwrapped, "command_manager"):
        try:
            raw_command_manager = env.unwrapped.command_manager
            paired_target_identification = bool(
                raw_command_manager.get_term("racket_target").cfg.adapter_external_paired
            )
            # A fixed recovery route is a single full-trajectory contract.
            # It must begin at the formal READY state; sampling a random
            # internal phase would remove the very pre-hit brace that makes
            # the tail physically reachable.
            fixed_motion_recovery = (
                raw_command_manager.get_term("motion").cfg.fixed_motion_id
                is not None
            )
        except (KeyError, ValueError):
            pass
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=(
            not natural_prefix_recovery
            and not paired_target_identification
            and not fixed_motion_recovery
        ),
    )
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so the kit app does not try to
    # parse Hydra's `task=...`/`algo=...` overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    try:
        _run(cfg)
    except Exception:
        import os
        import traceback

        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            simulation_app.close()
        finally:
            os._exit(1)
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()

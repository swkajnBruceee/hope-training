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
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"), applied)
        _set_reward(R, "racket_position_y", _get(rw, "racket_position_y_weight"), _get(rw, "racket_position_y_std"), applied)
        _set_reward(R, "racket_position_fine", _get(rw, "racket_position_fine_weight"), _get(rw, "racket_position_fine_std"), applied)
        _set_reward(R, "racket_position_y_fine", _get(rw, "racket_position_y_fine_weight"), _get(rw, "racket_position_y_fine_std"), applied)
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"), applied)
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
    from training.utils.ppo_cfg import runner_kwargs

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
    if cfg.get("load_run", None) is not None:
        agent_cfg.load_run = str(cfg.load_run)
    if cfg.get("checkpoint", None) is not None:
        agent_cfg.load_checkpoint = str(cfg.checkpoint)

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
    zero_residual_tasks = {
        "A3BaseStandPassiveStableCandidate-v0",
        "A3CatchReadyStand-v0",
        "A3BaseStandRecoveryA-v0",
        "A3BaseStandRecoveryAV2-v0",
        "A3BaseStandRecoveryAV2WaistMask-v0",
        "A3BaseStandRecoveryAV21WaistMask-v0",
        "HOPE-StrikeStabilizerA-AgibotA3-v0",
        "HOPE-FixedBaseReferenceStrike-AgibotA3-v0",
        "HOPE-FixedBaseBackhandReferenceStrike-AgibotA3-v0",
    }
    if task_id in zero_residual_tasks and not agent_cfg.resume:
        # This task controls a non-integrating residual around a passively
        # stable nominal posture.  A random output layer can create a large
        # deterministic residual before PPO sees one transition (observed as
        # 52% raw-action clipping in the v2 model_0 audit).  Keep exploration
        # in the Gaussian std, but make the initial mean policy exactly zero.
        initialize_zero_residual_actor_mean(
            runner,
            action_dim=10 if task_id.startswith("HOPE-FixedBase") else 14,
        )
        print(
            "[train.py] initialized A3 Base/Recovery actor mean to exact zero residual; "
            f"exploration remains init_noise_std={agent_cfg.policy.init_noise_std}",
            flush=True,
        )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
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

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
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

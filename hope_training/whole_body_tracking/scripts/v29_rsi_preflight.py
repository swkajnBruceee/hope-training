#!/usr/bin/env python3
"""V29 recovery-only RSI snapshot preflight.

This runner is deliberately separate from ``play.py`` and training.  It
captures one motion-0 settled anchor, then verifies a restore-paired
observation/actor/target trace for twenty control steps.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hydra
import torch
from omegaconf import OmegaConf

from train import _apply_task_overrides


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "reproducibility_archives/v28_bent_ready_recovery_baseline_20260729/weights/coordinator_v28_model_100.pt"
DEFAULT_OUTPUT = ROOT / "eval_outputs/v29_rsi_preflight/motion0_settled"


def _observation_layout() -> list[dict[str, Any]]:
    """Return the reviewed 235-D coordinator observation contract.

    Keep this explicit and versioned in the preflight: the diagnostic must
    identify a changed field even when the observation manager only exposes a
    concatenated policy tensor.
    """
    layout: list[dict[str, Any]] = []

    def add(name: str, width: int, source: str) -> None:
        start = sum(int(item["width"]) for item in layout)
        layout.append({"name": name, "start": start, "end": start + width, "width": width, "source": source})

    for name, width in (
        ("base_lin_vel", 3), ("base_ang_vel", 3), ("joint_pos", 14),
        ("joint_vel", 14), ("actions", 14), ("projected_gravity", 3),
        ("racket_target_pos_b", 3), ("racket_target_vel_b", 3),
        ("racket_target_normal_b", 3), ("racket_pos_b", 3),
        ("racket_lin_vel_b", 3), ("racket_normal_b", 3),
        ("time_to_strike", 1), ("swing_type", 1), ("strike_joint_pos", 9),
        ("strike_joint_vel", 9), ("strike_reference_joint_pos", 9),
        ("strike_reference_joint_vel", 9), ("strike_reference_joint_vel_8", 9),
        ("strike_reference_joint_vel_16", 9), ("strike_phase", 1),
    ):
        source = (
            "physx_derived_kinematic"
            if name in {"racket_pos_b", "racket_lin_vel_b", "racket_normal_b"}
            else "non_contact_direct"
        )
        add(f"stage_a.{name}", width, source)
    for name, width in (
        ("base_ang_vel", 3), ("joint_pos", 10), ("joint_vel", 10),
        ("actions", 10), ("projected_gravity", 3),
        ("racket_target_pos_b", 3), ("racket_target_vel_b", 3),
        ("racket_target_normal_b", 3), ("racket_pos_b", 3),
        ("racket_lin_vel_b", 3), ("racket_normal_b", 3),
        ("time_to_strike", 1), ("swing_type", 1),
    ):
        source = (
            "physx_derived_kinematic"
            if name in {"racket_pos_b", "racket_lin_vel_b", "racket_normal_b"}
            else "non_contact_direct"
        )
        add(f"upper.{name}", width, source)
    add("coordinator.previous_action", 22, "non_contact_direct")
    for name, width, source in (
        ("foot_rel_root_b", 4, "contact_derived_support"),
        ("com_rel_support", 2, "contact_derived_support"),
        ("capture_rel_support_x_b", 1, "contact_derived_support"),
        ("capture_front_margin", 1, "contact_derived_support"),
        ("capture_rear_margin", 1, "contact_derived_support"),
        ("normalized_load", 2, "contact_direct"),
        ("load_balance", 1, "contact_direct"),
        ("total_load_ratio", 1, "contact_direct"),
        ("contacts", 2, "contact_direct"),
        ("root_velocity_b", 2, "contact_derived_support"),
        ("root_roll_pitch_rate_b", 2, "contact_derived_support"),
        ("capture_rel_support_y_b", 1, "contact_derived_support"),
        ("capture_lateral_positive_margin", 1, "contact_derived_support"),
        ("capture_lateral_negative_margin", 1, "contact_derived_support"),
        ("lateral_span", 1, "contact_derived_support"),
    ):
        add(f"wide_stagger_support.{name}", width, source)
    for name, source in (
        ("capture_rate", "contact_derived_support"),
        ("capture_rel_support_x_b", "contact_derived_support"),
        ("root_lin_vel_b_x", "non_contact_direct"),
        ("root_ang_vel_b_pitch", "non_contact_direct"),
        ("right_arm_position_error", "non_contact_direct"),
        ("right_arm_velocity", "non_contact_direct"),
        ("rearm_stable_fraction", "non_contact_direct"),
        ("post_hit_gate", "non_contact_direct"),
    ):
        add(f"bent_ready_recovery.{name}", 1, source)
    total = sum(int(item["width"]) for item in layout)
    if total != 235:
        raise RuntimeError(f"V29 observation layout width mismatch: {total} != 235")
    return layout


def _locate_observation_index(index: int, layout: list[dict[str, Any]]) -> dict[str, Any]:
    for item in layout:
        if int(item["start"]) <= index < int(item["end"]):
            offset = index - int(item["start"])
            return {
                "slice": f"[{item['start']}:{item['end']}]",
                "field": item["name"],
                "field_index": offset,
                "source": item["source"],
            }
    return {"slice": "unknown", "field": "unknown", "field_index": -1, "source": "unknown"}


def _observation_diff_diagnostics(
    golden: torch.Tensor,
    restored: torch.Tensor,
    *,
    step: int,
    layout: list[dict[str, Any]],
    limit: int = 20,
) -> dict[str, Any]:
    left = golden[0].detach().float().cpu()
    right = restored[0].detach().float().cpu()
    delta = (left - right).abs()
    order = torch.argsort(delta, descending=True)[:limit].tolist()
    top = []
    for index in order:
        location = _locate_observation_index(int(index), layout)
        top.append({
            "step": step,
            "index": int(index),
            **location,
            "golden_value": float(left[index].item()),
            "restore_value": float(right[index].item()),
            "abs_diff": float(delta[index].item()),
        })
    direct = [item for item in top if item["source"] == "non_contact_direct"]
    contact = [item for item in top if item["source"].startswith("contact_")]
    physx_derived = [item for item in top if item["source"] == "physx_derived_kinematic"]
    direct_max = max((item["abs_diff"] for item in direct), default=0.0)
    contact_max = max((item["abs_diff"] for item in contact), default=0.0)
    physx_max = max((item["abs_diff"] for item in physx_derived), default=0.0)
    return {
        "step": step,
        "top20": top,
        "non_contact_top20_count": len(direct),
        "non_contact_direct_top20_count": len(direct),
        "contact_related_top20_count": len(contact),
        "physx_derived_kinematic_top20_count": len(physx_derived),
        "non_contact_max_abs": direct_max,
        "non_contact_direct_max_abs": direct_max,
        "contact_related_max_abs": contact_max,
        "physx_derived_kinematic_max_abs": physx_max,
        "contact_concentrated": direct_max <= 1.0e-5 and (contact_max > 1.0e-5 or physx_max > 1.0e-5),
        "physx_derived_concentrated": direct_max <= 1.0e-5 and physx_max > 1.0e-5,
        "layout_width": sum(int(item["width"]) for item in layout),
    }


def _obs_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("policy", next(iter(value.values())))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"policy observation is not a tensor: {type(value)!r}")
    return value.to(device)


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return copy.deepcopy(value)


def _compare(a: Any, b: Any, *, atol: float, rtol: float, path: str = "") -> tuple[bool, float, str]:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, float("inf"), f"{path}: shape/dtype {a.shape}/{a.dtype} != {b.shape}/{b.dtype}"
        if a.dtype.is_floating_point:
            delta = (a - b).abs()
            max_delta = float(delta.max().item()) if delta.numel() else 0.0
            ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
            return ok, max_delta, f"{path}: max_abs={max_delta:.9g}"
        ok = bool(torch.equal(a, b))
        return ok, 0.0, f"{path}: integer/bool mismatch"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, float("inf"), f"{path}: keys differ"
        worst = (True, 0.0, "")
        for key in a:
            result = _compare(a[key], b[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
            if not result[0] and (worst[0] or result[1] > worst[1]):
                worst = result
            elif result[1] > worst[1]:
                worst = (worst[0] and result[0], result[1], result[2])
        return worst
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False, float("inf"), f"{path}: sequence lengths differ"
        worst = (True, 0.0, "")
        for index, (left, right) in enumerate(zip(a, b)):
            result = _compare(left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]")
            if not result[0] and (worst[0] or result[1] > worst[1]):
                worst = result
            elif result[1] > worst[1]:
                worst = (worst[0] and result[0], result[1], result[2])
        return worst
    if a != b:
        return False, float("inf"), f"{path}: {a!r} != {b!r}"
    return True, 0.0, ""


def _tensor_dict_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_dict_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_dict_cpu(item) for item in value)
    return value


def _state_signature(raw: Any) -> dict[str, torch.Tensor]:
    action = raw.action_manager.get_term("joint_pos")
    motion = raw.command_manager.get_term("motion")
    robot = raw.scene["robot"]
    return {
        "root_state_w": robot.data.root_state_w[:1].detach().clone(),
        "joint_pos": robot.data.joint_pos[:1].detach().clone(),
        "joint_vel": robot.data.joint_vel[:1].detach().clone(),
        "motion_ids": motion.motion_ids[:1].detach().clone(),
        "time_steps": motion.time_steps[:1].detach().clone(),
        "shot_cycle": motion.shot_cycle[:1].detach().clone(),
        "tail_steps": motion.tail_steps[:1].detach().clone(),
        "stage_a_exit_state": action._stage_a_exit_state[:1].detach().clone(),
        "stage_a_exit_scale": action._stage_a_exit_scale[:1].detach().clone(),
        "stage_a_rearm_ready": action._stage_a_rearm_ready[:1].detach().clone(),
        "stage_a_rearm_stable_steps": action._stage_a_rearm_stable_steps[:1].detach().clone(),
    }


def _target_metadata(raw: Any) -> dict[str, float | None]:
    from training.tasks.tracking.mdp.observations import stagger_support_state

    action = raw.action_manager.get_term("joint_pos")
    motion = raw.command_manager.get_term("motion")
    robot = raw.scene["robot"]
    arm_ids = action._upper_joint_ids_tensor[action._upper_arm_indices]
    ready_error = torch.max(
        torch.abs(robot.data.joint_pos[:, arm_ids] - motion.ready_joint_pos[:, arm_ids]),
        dim=-1,
    ).values
    support = stagger_support_state(raw)
    capture_margin = torch.minimum(
        support["capture_front_margin"], support["capture_rear_margin"]
    )
    return {
        "target_ready_distance": float(ready_error[0].item()),
        "target_capture_margin": float(capture_margin[0].item()),
    }


def _layer_trace(raw: Any, coordinator: torch.Tensor, adapter: torch.Tensor) -> dict[str, torch.Tensor]:
    action = raw.action_manager.get_term("joint_pos")
    return {
        "model_3396_action": action._legacy_raw[:1].detach().clone(),
        "model_900_action": action._upper_raw_actions[:1].detach().clone(),
        "v28_adapter_action": adapter[:1].detach().clone(),
        "coordinator_action": coordinator[:1].detach().clone(),
        "final_joint_target": action._full_joint_targets[:1].detach().clone(),
        "final_joint_velocity_target": action._full_joint_velocity_targets[:1].detach().clone(),
        "legacy_bounded": action._legacy_bounded[:1].detach().clone(),
        "upper_reference": action._upper_reference_actions[:1].detach().clone(),
        "upper_primary_contribution": action._upper_primary_contribution[:1].detach().clone(),
        "upper_coordinator_contribution": action._upper_coordinator_contribution[:1].detach().clone(),
    }


def _predict_parent_layers(raw: Any, coordinator: torch.Tensor, adapter: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate the two frozen parents without changing action/controller state."""
    action = raw.action_manager.get_term("joint_pos")
    stage_obs = action._compute_observation_group(action._legacy_stage_a_group)
    if action.cfg.legacy_stage_a_yaw_adapter:
        from training.tasks.base_locomotion.mdp.actions import adapt_stage_a_observation_legacy_yaw_pi
        stage_obs = adapt_stage_a_observation_legacy_yaw_pi(stage_obs)
    model_3396 = action._legacy_stage_a(stage_obs)
    upper_obs = action._compute_observation_group(action._upper_observation_group)
    model_900 = action._upper_policy(upper_obs).clamp(-action.cfg.upper_raw_clip, action.cfg.upper_raw_clip)
    return {
        "model_3396_action": model_3396[:1].detach().clone(),
        "model_900_action": model_900[:1].detach().clone(),
        "v28_adapter_action": adapter[:1].detach().clone(),
        "coordinator_action": coordinator[:1].detach().clone(),
    }


def _sync_motion_zero(raw: Any) -> None:
    """Select motion 0 without teleporting the already-settled reset state."""
    device = raw.device
    env_ids = torch.arange(raw.num_envs, device=device, dtype=torch.long)
    motion = raw.command_manager.get_term("motion")
    motion.motion_ids.fill_(0)
    motion.time_steps.zero_()
    motion.tail_steps.zero_()
    motion.prelude_elapsed_steps.zero_()
    motion.shot_cycle.zero_()
    motion._prev_motion_steps = motion.time_steps.clone()
    racket = raw.command_manager.get_term("racket_target")
    racket._resample_command(env_ids)
    racket._compute_strike_timing()
    motion._update_command()


def _exact_step(raw: Any, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one ManagerBasedRLEnv control step, stopping before obs compute."""
    raw.action_manager.process_action(action.to(raw.device))
    raw.recorder_manager.record_pre_step()
    is_rendering = raw.sim.has_gui() or raw.sim.has_rtx_sensors()
    for _ in range(raw.cfg.decimation):
        raw._sim_step_counter += 1
        raw.action_manager.apply_action()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        if raw._sim_step_counter % raw.cfg.sim.render_interval == 0 and is_rendering:
            raw.sim.render()
        raw.scene.update(dt=raw.physics_dt)
    raw.episode_length_buf += 1
    raw.common_step_counter += 1
    raw.reset_buf = raw.termination_manager.compute()
    raw.reset_terminated = raw.termination_manager.terminated
    raw.reset_time_outs = raw.termination_manager.time_outs
    raw.reward_buf = raw.reward_manager.compute(dt=raw.step_dt)
    if len(raw.recorder_manager.active_terms) > 0:
        raw.obs_buf = raw.observation_manager.compute()
        raw.recorder_manager.record_post_step()
    reset_ids = raw.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        raise RuntimeError(f"physical termination during V29 preflight: ids={reset_ids.tolist()}")
    raw.command_manager.compute(dt=raw.step_dt)
    if "interval" in raw.event_manager.available_modes:
        raw.event_manager.apply(mode="interval", dt=raw.step_dt)
    # This is the audited post-physics/pre-observation boundary.
    return raw.reset_terminated.detach().clone(), raw.reset_time_outs.detach().clone()


def _make_runner(cfg: Any, simulation_app: Any):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner
    import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
    import training.tasks  # noqa: F401
    from training.utils.ppo_cfg import runner_kwargs
    from training.utils.stagger_support_actor_critic import BentReadyRecoveryActorCritic

    task_id = str(cfg.task.gym_task)
    if task_id != "HOPE-FloatingJointCoordinatorV11BentReadyRecovery-AgibotA3-v0":
        raise ValueError(f"V29 preflight requires the V28/V11 task, got {task_id}")
    num_envs = 1
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.seed = int(cfg.seed)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 60.0)
    manifest = pathlib.Path(str(cfg.task.motion_manifest)).expanduser()
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    env_cfg.commands.motion.motion_manifest = str(manifest)
    env_cfg.commands.motion.motion_file = None

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)
    rsl_on_policy_runner.BentReadyRecoveryActorCritic = BentReadyRecoveryActorCritic
    agent_cfg.policy.class_name = "BentReadyRecoveryActorCritic"

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint_value = cfg.get("v29_checkpoint", None)
    checkpoint = pathlib.Path(
        str(checkpoint_value) if checkpoint_value else str(DEFAULT_CHECKPOINT)
    ).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    runner.load(str(checkpoint))
    runner.alg.policy.eval()

    # Match play.py's V21+ observation-normalizer contract.
    original_forward = runner.obs_normalizer.forward

    def preserve_recovery_columns(observation):
        normalized = original_forward(observation)
        normalized[..., 204:] = observation[..., 204:]
        return normalized

    runner.obs_normalizer.forward = preserve_recovery_columns
    return env, runner, checkpoint


def _run_trace(raw: Any, runner: Any, *, steps: int, snapshot: dict[str, Any], restore_first: bool, snapshotter) -> dict[str, Any]:
    if restore_first:
        from training.utils.v29_rsi_snapshot import restore
        restore(raw, snapshot, env_ids=torch.tensor([0], device=raw.device))
    obs = _obs_tensor(raw.observation_manager.compute(), raw.device)
    policy = runner.alg.policy
    normalized = runner.obs_normalizer(obs)
    coordinator = policy.act_inference(normalized)
    base = policy.base_action_mean(normalized[..., : policy.BASE_OBS_DIM])
    adapter = coordinator - base
    trace = []
    for index in range(steps):
        coordinator = policy.act_inference(normalized)
        base = policy.base_action_mean(normalized[..., : policy.BASE_OBS_DIM])
        adapter = coordinator - base
        raw.action_manager.process_action(coordinator.to(raw.device))
        layers = _layer_trace(raw, coordinator, adapter)
        terminated, truncated = _exact_step_after_process(raw)
        next_snapshot = snapshotter(raw)
        next_obs = _obs_tensor(raw.observation_manager.compute(), raw.device)
        trace.append({
            "step": index,
            "observation": obs[:1].detach().clone(),
            "layers": layers,
            "state_after": _state_signature(raw),
            "next_snapshot": next_snapshot,
            "terminated": terminated,
            "truncated": truncated,
        })
        obs = next_obs
        normalized = runner.obs_normalizer(obs)
    return {"trace": trace, "final_snapshot": snapshotter(raw)}


def _exact_step_after_process(raw: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The second half of _exact_step, used after actor/target capture."""
    raw.recorder_manager.record_pre_step()
    is_rendering = raw.sim.has_gui() or raw.sim.has_rtx_sensors()
    for _ in range(raw.cfg.decimation):
        raw._sim_step_counter += 1
        raw.action_manager.apply_action()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        if raw._sim_step_counter % raw.cfg.sim.render_interval == 0 and is_rendering:
            raw.sim.render()
        raw.scene.update(dt=raw.physics_dt)
    raw.episode_length_buf += 1
    raw.common_step_counter += 1
    raw.reset_buf = raw.termination_manager.compute()
    raw.reset_terminated = raw.termination_manager.terminated
    raw.reset_time_outs = raw.termination_manager.time_outs
    raw.reward_buf = raw.reward_manager.compute(dt=raw.step_dt)
    if len(raw.recorder_manager.active_terms) > 0:
        raw.obs_buf = raw.observation_manager.compute()
        raw.recorder_manager.record_post_step()
    reset_ids = raw.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        raise RuntimeError(f"physical termination during V29 preflight: ids={reset_ids.tolist()}")
    raw.command_manager.compute(dt=raw.step_dt)
    if "interval" in raw.event_manager.available_modes:
        raw.event_manager.apply(mode="interval", dt=raw.step_dt)
    return raw.reset_terminated.detach().clone(), raw.reset_time_outs.detach().clone()


def _warm_to_settled(env: Any, raw: Any, runner: Any, snapshotter) -> dict[str, Any]:
    """Run V28 to SETTLED and retain the exact prefix replay material."""
    obs = _obs_tensor(env.get_observations(), raw.device)
    policy = runner.alg.policy
    records: list[dict[str, Any]] = []
    for step in range(500):
        normalized = runner.obs_normalizer(obs)
        action = policy.act_inference(normalized)
        base = policy.base_action_mean(normalized[..., : policy.BASE_OBS_DIM])
        adapter = action - base
        result = env.step(action.to(raw.device))
        obs = _obs_tensor(result[0], raw.device)
        terminated = torch.as_tensor(result[2], device=raw.device, dtype=torch.bool)
        if torch.is_tensor(result[3]):
            truncated = torch.as_tensor(result[3], device=raw.device, dtype=torch.bool)
        else:
            truncated = torch.as_tensor(
                result[3].get("time_outs", torch.zeros_like(terminated)),
                device=raw.device,
                dtype=torch.bool,
            )
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            reasons = sorted(
                key.removeprefix("Episode_Termination/")
                for key, value in (raw.extras.get("log", {}) or {}).items()
                if key.startswith("Episode_Termination/")
                and float(torch.as_tensor(value).sum().item()) > 0.0
            )
            raise RuntimeError(
                f"physical termination while seeking SETTLED at step {step + 1}; "
                f"terminated={terminated.tolist()} truncated={truncated.tolist()} reasons={reasons}"
            )
        action_term = raw.action_manager.get_term("joint_pos")
        records.append({
            "control_step": step + 1,
            "observation_after": obs[:1].detach().clone(),
            "action_state": action_term.export_v29_rsi_state(torch.tensor([0], device=raw.device)),
            "layers": _layer_trace(raw, action, adapter),
            "snapshot": snapshotter(raw),
            "state_after": _state_signature(raw),
        })
        if bool(action_term._stage_a_rearm_ready[0].item()):
            return {
                "warmup_steps": step + 1,
                "records": records,
                "target_observation": obs[:1].detach().clone(),
            }
    raise RuntimeError("V28 did not reach fail-closed SETTLED/READY within 500 steps")


def _run_free_trace_from_current(
    raw: Any,
    runner: Any,
    *,
    steps: int,
    snapshotter,
    initial_obs: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Record a normal actor-driven continuation from the current boundary."""
    obs = (
        _obs_tensor(initial_obs, raw.device)
        if initial_obs is not None
        else _obs_tensor(raw.observation_manager.compute(), raw.device)
    )
    policy = runner.alg.policy
    trace = []
    for index in range(steps):
        normalized = runner.obs_normalizer(obs)
        coordinator = policy.act_inference(normalized)
        base = policy.base_action_mean(normalized[..., : policy.BASE_OBS_DIM])
        adapter = coordinator - base
        raw.action_manager.process_action(coordinator.to(raw.device))
        layers = _layer_trace(raw, coordinator, adapter)
        terminated, truncated = _exact_step_after_process(raw)
        next_snapshot = snapshotter(raw)
        next_obs = _obs_tensor(raw.observation_manager.compute(), raw.device)
        trace.append({
            "step": index,
            "observation": obs[:1].detach().clone(),
            "layers": layers,
            "state_after": _state_signature(raw),
            "next_snapshot": next_snapshot,
            "terminated": terminated,
            "truncated": truncated,
        })
        obs = next_obs
    return {"trace": trace, "final_snapshot": snapshotter(raw)}


def _replay_prefix(
    raw: Any,
    runner: Any,
    *,
    records: list[dict[str, Any]],
    target_step: int,
    prefix_steps: int,
    target_snapshot: dict[str, Any],
    golden_trace: dict[str, Any],
    snapshotter,
) -> dict[str, Any]:
    """Restore an earlier anchor and replay frozen actuator/controller state."""
    from training.utils.v29_rsi_snapshot import restore

    env_ids = torch.tensor([0], device=raw.device)
    anchor_step = target_step - prefix_steps
    anchor = records[anchor_step - 1]
    restore(raw, anchor["snapshot"], env_ids=env_ids)
    action_term = raw.action_manager.get_term("joint_pos")
    prefix_trace = []
    branch_mismatch = None
    for record in records[anchor_step:target_step]:
        # This restores the state produced by the archived actor for this
        # control step, including action history, FSM latches, and both final
        # actuator target caches.  No actor is called during this prefix.
        action_term.restore_v29_rsi_state(record["action_state"], env_ids)
        terminated, truncated = _exact_step_after_process(raw)
        current_state = _state_signature(raw)
        for key in ("motion_ids", "time_steps", "shot_cycle", "tail_steps",
                    "stage_a_exit_state", "stage_a_rearm_ready",
                    "stage_a_rearm_stable_steps"):
            if not torch.equal(current_state[key], record["state_after"][key]):
                branch_mismatch = {
                    "control_step": record["control_step"],
                    "field": key,
                }
                break
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            branch_mismatch = {
                "control_step": record["control_step"],
                "field": "termination",
            }
            break
        next_snapshot = snapshotter(raw)
        next_obs = _obs_tensor(raw.observation_manager.compute(), raw.device)
        prefix_trace.append({
            "control_step": record["control_step"],
            "next_snapshot": next_snapshot,
            "state_after": current_state,
            "observation_after": next_obs[:1].detach().clone(),
        })

    reconstructed = snapshotter(raw)
    ignored = {"torch_rng_state", "python_rng_state", "numpy_rng_state", "cuda_rng_state"}
    gate_ar = _compare(
        {key: value for key, value in target_snapshot.items() if key not in ignored},
        {key: value for key, value in reconstructed.items() if key not in ignored},
        atol=2.0e-3,
        rtol=2.0e-3,
        path="target_snapshot",
    )
    if branch_mismatch is not None:
        gate_ar = (False, float("inf"), f"prefix branch mismatch: {branch_mismatch}")

    if not prefix_trace:
        raise RuntimeError("V29 reconstruction prefix produced no target state")
    target_obs = prefix_trace[-1]["observation_after"]
    policy = runner.alg.policy
    normalized = runner.obs_normalizer(target_obs)
    coordinator = policy.act_inference(normalized)
    base = policy.base_action_mean(normalized[..., : policy.BASE_OBS_DIM])
    adapter = coordinator - base
    predicted = _predict_parent_layers(raw, coordinator, adapter)
    target_golden = golden_trace["trace"][0]
    gate_br = (True, 0.0, "")
    for name, left, right in (
        ("observation", target_obs[:1], target_golden["observation"]),
        ("layers.model_3396_action", predicted["model_3396_action"], target_golden["layers"]["model_3396_action"]),
        ("layers.model_900_action", predicted["model_900_action"], target_golden["layers"]["model_900_action"]),
        ("layers.v28_adapter_action", predicted["v28_adapter_action"], target_golden["layers"]["v28_adapter_action"]),
        ("layers.coordinator_action", predicted["coordinator_action"], target_golden["layers"]["coordinator_action"]),
    ):
        result = _compare(left, right, atol=1.0e-6, rtol=1.0e-6, path=f"target.{name}")
        if not result[0] and (gate_br[0] or result[1] > gate_br[1]):
            gate_br = result

    # Gate C-R begins at the reconstructed target boundary and is fully
    # actor-driven again.  The archived target continuation is the reference.
    continuation = _run_free_trace_from_current(
        raw, runner, steps=20, snapshotter=snapshotter, initial_obs=target_obs
    )
    gate_cr = (True, 0.0, "")
    first_divergence = None
    for index, (left, right) in enumerate(zip(golden_trace["trace"], continuation["trace"])):
        for name in ("observation",):
            result = _compare(left[name], right[name], atol=1.0e-6, rtol=1.0e-6, path=f"trace[{index}].{name}")
            if not result[0] and (gate_cr[0] or result[1] > gate_cr[1]):
                gate_cr = result
                first_divergence = index if first_divergence is None else first_divergence
        result = _compare(left["layers"]["coordinator_action"], right["layers"]["coordinator_action"],
                          atol=1.0e-6, rtol=1.0e-6, path=f"trace[{index}].layers.coordinator_action")
        if not result[0] and (gate_cr[0] or result[1] > gate_cr[1]):
            gate_cr = result
            first_divergence = index if first_divergence is None else first_divergence
        result = _compare(left["state_after"], right["state_after"], atol=1.0e-5, rtol=1.0e-5,
                          path=f"trace[{index}].state_after")
        if not result[0] and (gate_cr[0] or result[1] > gate_cr[1]):
            gate_cr = result
            first_divergence = index if first_divergence is None else first_divergence

    return {
        "prefix_steps": prefix_steps,
        "anchor_control_step": anchor_step,
        "gate_a_r": {"passed": gate_ar[0], "max_abs": gate_ar[1], "detail": gate_ar[2]},
        "gate_b_r": {"passed": gate_br[0], "max_abs": gate_br[1], "detail": gate_br[2]},
        "gate_c_r": {"passed": gate_cr[0], "max_abs": gate_cr[1], "detail": gate_cr[2], "steps": 20},
        "first_divergence_step": first_divergence,
        "branch_mismatch": branch_mismatch,
        "anchor_snapshot": anchor["snapshot"],
        "golden_position_targets": [
            record["action_state"]["_full_joint_targets"]
            for record in records[anchor_step:target_step]
        ],
        "golden_velocity_targets": [
            record["action_state"]["_full_joint_velocity_targets"]
            for record in records[anchor_step:target_step]
        ],
        "golden_action_history": [
            record["action_state"]["env_joint_coordinator_last_action"]
            for record in records[anchor_step:target_step]
        ],
        "golden_fsm_trace": [
            record["state_after"]
            for record in records[anchor_step:target_step]
        ],
        "prefix_trace": prefix_trace,
        "continuation": continuation,
        "reconstructed_snapshot": reconstructed,
    }


@hydra.main(version_base=None, config_path="../cfg", config_name="v29_rsi_preflight")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    env = None
    try:
        from training.utils.v29_rsi_snapshot import capture

        env, runner, checkpoint = _make_runner(cfg, simulation_app)
        raw = env.unwrapped
        # gym.make has already completed the task reset.  Keep this identical
        # to play.py's forced-motion path; an additional reset here changes
        # the startup handoff/physics anchor before the motion sync.
        _sync_motion_zero(raw)
        env_ids = torch.tensor([0], device=raw.device)
        snapshotter = lambda item: capture(item, env_ids=env_ids)
        warmup = _warm_to_settled(env, raw, runner, snapshotter)
        warmup_steps = int(warmup["warmup_steps"])
        records = warmup["records"]
        snapshot = records[-1]["snapshot"]

        # Preserve the original target's natural actor-driven continuation
        # before any restore or prefix attempt changes the simulator.
        golden_continuation = _run_free_trace_from_current(
            raw, runner, steps=20, snapshotter=snapshotter,
            initial_obs=warmup["target_observation"],
        )

        # Gate A: restoring the exact physical/controller snapshot must be
        # observable before any next-observation computation.
        from training.utils.v29_rsi_snapshot import restore

        restore(raw, snapshot, env_ids=env_ids)
        restored = capture(raw, env_ids=env_ids)
        gate_a = _compare(
            {key: value for key, value in snapshot.items() if key not in {"torch_rng_state", "python_rng_state", "numpy_rng_state", "cuda_rng_state"}},
            {key: value for key, value in restored.items() if key not in {"torch_rng_state", "python_rng_state", "numpy_rng_state", "cuda_rng_state"}},
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        target_metadata = _target_metadata(raw)

        golden = _run_trace(raw, runner, steps=20, snapshot=snapshot, restore_first=True, snapshotter=snapshotter)
        replay = _run_trace(raw, runner, steps=20, snapshot=snapshot, restore_first=True, snapshotter=snapshotter)

        gate_b_worst = (True, 0.0, "")
        gate_c_worst = (True, 0.0, "")
        for index, (left, right) in enumerate(zip(golden["trace"], replay["trace"])):
            result = _compare(left["observation"], right["observation"], atol=1.0e-6, rtol=1.0e-6, path=f"trace[{index}].observation")
            if not result[0] and (gate_b_worst[0] or result[1] > gate_b_worst[1]): gate_b_worst = result
            for name in left["layers"]:
                result = _compare(left["layers"][name], right["layers"][name], atol=1.0e-6, rtol=1.0e-6, path=f"trace[{index}].layers.{name}")
                if not result[0] and (gate_b_worst[0] or result[1] > gate_b_worst[1]): gate_b_worst = result
            result = _compare(left["state_after"], right["state_after"], atol=1.0e-5, rtol=1.0e-5, path=f"trace[{index}].state_after")
            if not result[0] and (gate_c_worst[0] or result[1] > gate_c_worst[1]): gate_c_worst = result

        layout = _observation_layout()
        gate_b_diagnostic = None
        for index, (left, right) in enumerate(zip(golden["trace"], replay["trace"])):
            result = _compare(left["observation"], right["observation"], atol=1.0e-6, rtol=1.0e-6)
            if not result[0]:
                gate_b_diagnostic = _observation_diff_diagnostics(
                    left["observation"], right["observation"],
                    step=index, layout=layout,
                )
                break
        if gate_b_diagnostic is None:
            gate_b_diagnostic = {
                "step": None, "top20": [], "non_contact_top20_count": 0,
                "contact_related_top20_count": 0, "non_contact_max_abs": 0.0,
                "contact_related_max_abs": 0.0, "contact_concentrated": False,
                "layout_width": 235,
            }

        # The minimum-prefix scan is deliberately limited to this one
        # motion-0 SETTLED target.  A bank entry is created only after all
        # reconstruction gates pass.
        prefix_results = []
        for prefix_steps in (10, 20, 40):
            if prefix_steps >= warmup_steps:
                continue
            prefix_results.append(_replay_prefix(
                raw, runner, records=records, target_step=warmup_steps,
                prefix_steps=prefix_steps, target_snapshot=snapshot,
                golden_trace=golden_continuation, snapshotter=snapshotter,
            ))
        selected_prefix = next(
            (
                item for item in prefix_results
                if item["gate_a_r"]["passed"]
                and item["gate_b_r"]["passed"]
                and item["gate_c_r"]["passed"]
            ),
            None,
        )

        all_passed = gate_a[0] and gate_b_worst[0] and gate_c_worst[0]
        failure_category = None
        if not all_passed:
            # A physical continuation mismatch takes precedence over the
            # downstream actor mismatch: observation/action drift after a
            # restore is not evidence of a bad checkpoint when PhysX contact
            # warm-start state was not restorable.
            if gate_b_diagnostic["non_contact_max_abs"] > 1.0e-5:
                failure_category = "missing_snapshot_field"
            elif not gate_c_worst[0]:
                failure_category = "contact_divergence"
            elif not gate_a[0]:
                failure_category = "step0_physics_mismatch"
            else:
                failure_category = "actor_action_mismatch"
        prefix_bank_entry = {
            "sample_id": "motion0_settled_single_sample",
            "motion_id": 0,
            "shot_index": 0,
            "target_control_step": warmup_steps,
            "anchor_control_step": (
                selected_prefix["anchor_control_step"] if selected_prefix is not None else None
            ),
            "prefix_steps": (
                selected_prefix["prefix_steps"] if selected_prefix is not None else None
            ),
            "target_recovery_stage": "SETTLED",
            "target_ready_distance": target_metadata["target_ready_distance"],
            "target_capture_margin": target_metadata["target_capture_margin"],
            "gate_a_r_pass": bool(selected_prefix and selected_prefix["gate_a_r"]["passed"]),
            "gate_b_r_pass": bool(selected_prefix and selected_prefix["gate_b_r"]["passed"]),
            "gate_c_r_pass": bool(selected_prefix and selected_prefix["gate_c_r"]["passed"]),
            "max_observation_error": (
                selected_prefix["gate_b_r"]["max_abs"] if selected_prefix is not None else None
            ),
            "max_action_error": (
                selected_prefix["gate_b_r"]["max_abs"] if selected_prefix is not None else None
            ),
            "max_joint_velocity_error": None,
            "first_divergence_step": (
                selected_prefix["first_divergence_step"] if selected_prefix is not None else None
            ),
            "restore_mode": "reconstruction_prefix",
            "replay_verified": selected_prefix is not None,
            "reject_reason": None if selected_prefix is not None else (
                "no_prefix_passed_10_20_40"
            ),
        }
        report = {
            "status": "replay_verified" if selected_prefix is not None else "rejected",
            "classification": "candidate",
            "training_eligible": False,
            "rejected_only_regression_evidence": True,
            "motion_id": 0,
            "physical_anchor": "SETTLED",
            "warmup_steps_to_settled": warmup_steps,
            "checkpoint": str(checkpoint),
            "snapshot_phase": snapshot["snapshot_phase"],
            "gate_a": {"passed": gate_a[0], "max_abs": gate_a[1], "detail": gate_a[2]},
            "gate_b": {"passed": gate_b_worst[0], "max_abs": gate_b_worst[1], "detail": gate_b_worst[2]},
            "gate_c": {"passed": gate_c_worst[0], "max_abs": gate_c_worst[1], "detail": gate_c_worst[2], "steps": 20},
            "first_failure_category": failure_category,
            "direct_restore": "accepted" if all_passed else "rejected",
            "reconstruction_prefix_required": failure_category == "contact_divergence",
            "gate_b_observation_diagnostic": gate_b_diagnostic,
            "reconstruction_prefix_scan": [
                {
                    key: value for key, value in item.items()
                    if key in {
                        "prefix_steps", "anchor_control_step", "gate_a_r",
                        "gate_b_r", "gate_c_r", "first_divergence_step",
                        "branch_mismatch",
                    }
                }
                for item in prefix_results
            ],
            "selected_prefix": (
                selected_prefix["prefix_steps"] if selected_prefix is not None else None
            ),
            "prefix_bank_entry": prefix_bank_entry,
        }
        output = pathlib.Path(str(cfg.get("v29_output", DEFAULT_OUTPUT))).expanduser()
        if not output.is_absolute(): output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        torch.save({
            "snapshot": _tensor_dict_cpu(snapshot),
            "golden_trace_direct_restore": _tensor_dict_cpu(golden),
            "golden_trace_reconstruction_target": _tensor_dict_cpu(golden_continuation),
            "prefix_bank_entry": prefix_bank_entry,
            "prefix_scan": _tensor_dict_cpu(prefix_results),
        }, output.with_suffix(".pt"))
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"V29_PREFLIGHT_REJECTED: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

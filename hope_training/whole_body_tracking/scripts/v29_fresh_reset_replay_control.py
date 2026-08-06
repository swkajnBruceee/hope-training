#!/usr/bin/env python3
"""Fresh-reset causal control for the rejected V29 contact-active RSI route.

This is not a recovery sampler and does not write rollouts.  It compares:

  A: fresh reset -> normal frozen model_100 actor
  B: the same fresh reset -> replay A's recorded actuator/controller state

The experiment distinguishes missing PhysX state at a restored anchor from a
golden-command timing or actuator-state recording defect.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hydra
import torch
from omegaconf import OmegaConf

from train import _apply_task_overrides
from v29_rsi_preflight import (
    DEFAULT_CHECKPOINT,
    _compare,
    _exact_step_after_process,
    _make_runner,
    _observation_diff_diagnostics,
    _observation_layout,
    _obs_tensor,
    _state_signature,
    _sync_motion_zero,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eval_outputs/v29_rsi_preflight/motion0_fresh_reset_control"


def _fresh_reset(env: Any, raw: Any, seed: int) -> torch.Tensor:
    """Perform the task's actual reset, then select motion 0 bookkeeping."""
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    _sync_motion_zero(raw)
    return _obs_tensor(result[0] if isinstance(result, tuple) else result, raw.device)


def _capture_torch_rng() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_torch_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _normal_run(env: Any, raw: Any, runner: Any, *, target_steps: int) -> dict[str, Any]:
    obs = _obs_tensor(env.get_observations(), raw.device)
    policy = runner.alg.policy
    records = []
    for step in range(target_steps):
        normalized = runner.obs_normalizer(obs)
        action = policy.act_inference(normalized)
        result = env.step(action.to(raw.device))
        obs = _obs_tensor(result[0], raw.device)
        terminated = torch.as_tensor(result[2], device=raw.device, dtype=torch.bool)
        truncated_value = result[3]
        if torch.is_tensor(truncated_value):
            truncated = torch.as_tensor(truncated_value, device=raw.device, dtype=torch.bool)
        else:
            truncated = torch.as_tensor(
                truncated_value.get("time_outs", torch.zeros_like(terminated)),
                device=raw.device,
                dtype=torch.bool,
            )
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            raise RuntimeError(f"A terminated during fresh-reset control at step {step + 1}")
        action_term = raw.action_manager.get_term("joint_pos")
        records.append({
            "control_step": step + 1,
            "observation_after": obs[:1].detach().clone(),
            "action_state": action_term.export_v29_rsi_state(
                torch.tensor([0], device=raw.device)
            ),
            "state_after": _state_signature(raw),
        })
    if not bool(action_term._stage_a_rearm_ready[0].item()):
        raise RuntimeError(f"A did not reach SETTLED by fixed target step {target_steps}")
    return {
        "target_step": target_steps,
        "records": records,
        "target_observation": obs[:1].detach().clone(),
        "target_state": _state_signature(raw),
    }


def _refresh_parent_observation_groups(raw: Any) -> None:
    """Mirror action processing's group sampling without invoking either actor."""
    action = raw.action_manager.get_term("joint_pos")
    action._compute_observation_group(action._legacy_stage_a_group)
    action._compute_observation_group(action._upper_observation_group)


def _replay_run(raw: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    env_ids = torch.tensor([0], device=raw.device)
    action_term = raw.action_manager.get_term("joint_pos")
    replay_records = []
    first_divergence = None
    for record in records:
        # A normal action step samples these groups before computing the
        # frozen parents.  Reproduce only that observation-manager timing;
        # the recorded action/FSM/target state remains authoritative.
        _refresh_parent_observation_groups(raw)
        action_term.restore_v29_rsi_state(record["action_state"], env_ids)
        terminated, truncated = _exact_step_after_process(raw)
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            raise RuntimeError(
                f"B terminated during golden-target replay at step {record['control_step']}"
            )
        obs = _obs_tensor(raw.observation_manager.compute(), raw.device)
        state = _state_signature(raw)
        replay_records.append({
            "control_step": record["control_step"],
            "observation_after": obs[:1].detach().clone(),
            "state_after": state,
        })
        if first_divergence is None:
            result = _compare(
                record["state_after"], state, atol=1.0e-5, rtol=1.0e-5,
                path=f"step[{record['control_step']}].state_after",
            )
            if not result[0]:
                first_divergence = {
                    "control_step": record["control_step"],
                    "detail": result[2],
                    "max_abs": result[1],
                }
    return {
        "records": replay_records,
        "target_observation": replay_records[-1]["observation_after"],
        "target_state": replay_records[-1]["state_after"],
        "first_divergence": first_divergence,
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
        env, runner, checkpoint = _make_runner(cfg, simulation_app)
        raw = env.unwrapped
        seed = int(cfg.seed)
        _fresh_reset(env, raw, seed)
        rng_after_a_reset = _capture_torch_rng()
        # Keep the causal control aligned with the original preflight target,
        # rather than stopping at the first earlier READY latch.
        normal = _normal_run(env, raw, runner, target_steps=328)

        # Reset the same task with the same seed.  No snapshot or physical
        # state writeback is used for B.
        _fresh_reset(env, raw, seed)
        _restore_torch_rng(rng_after_a_reset)
        # A calls env.get_observations() once before its first actor action;
        # consume the identical noisy policy observation for B so the
        # observation RNG stream has the same causal alignment.
        env.get_observations()
        replay = _replay_run(raw, normal["records"])

        target_state = _compare(
            normal["target_state"], replay["target_state"],
            atol=1.0e-5, rtol=1.0e-5, path="target_state",
        )
        target_obs = _compare(
            normal["target_observation"], replay["target_observation"],
            atol=1.0e-6, rtol=1.0e-6, path="target_observation",
        )
        state_worst = (True, 0.0, "")
        obs_worst = (True, 0.0, "")
        first_observation_divergence = None
        for left, right in zip(normal["records"], replay["records"]):
            result = _compare(
                left["state_after"], right["state_after"],
                atol=1.0e-5, rtol=1.0e-5,
                path=f"step[{left['control_step']}].state_after",
            )
            if not result[0] and (state_worst[0] or result[1] > state_worst[1]):
                state_worst = result
            result = _compare(
                left["observation_after"], right["observation_after"],
                atol=1.0e-6, rtol=1.0e-6,
                path=f"step[{left['control_step']}].observation_after",
            )
            if not result[0] and (obs_worst[0] or result[1] > obs_worst[1]):
                obs_worst = result
            if not result[0] and first_observation_divergence is None:
                first_observation_divergence = {
                    "control_step": left["control_step"],
                    "diagnostic": _observation_diff_diagnostics(
                        left["observation_after"], right["observation_after"],
                        step=left["control_step"], layout=_observation_layout(),
                    ),
                }

        if state_worst[0] and obs_worst[0]:
            interpretation = "A_and_B_reproduce"
        elif state_worst[0] and not obs_worst[0]:
            interpretation = "explicit_state_reproduces_observation_history_or_sampling_mismatch"
        elif obs_worst[0] and not state_worst[0]:
            interpretation = "A_reproduces_B_physics_diverges"
        elif not obs_worst[0] and not state_worst[0]:
            interpretation = "golden_target_or_execution_timing_mismatch"
        else:
            interpretation = "mixed_observation_state_result"

        report = {
            "status": "completed",
            "motion_id": 0,
            "checkpoint": str(checkpoint),
            "target_stage": "SETTLED",
            "normal_actor": "A_fresh_reset_model_100",
            "golden_target_replay": "B_fresh_reset_recorded_actuator_state",
            "target_step": normal["target_step"],
            "interpretation": interpretation,
            "target_state": {
                "passed": target_state[0],
                "max_abs": target_state[1],
                "detail": target_state[2],
            },
            "target_observation": {
                "passed": target_obs[0],
                "max_abs": target_obs[1],
                "detail": target_obs[2],
            },
            "state_trace": {
                "passed": state_worst[0],
                "max_abs": state_worst[1],
                "detail": state_worst[2],
            },
            "observation_trace": {
                "passed": obs_worst[0],
                "max_abs": obs_worst[1],
                "detail": obs_worst[2],
            },
            "first_divergence": replay["first_divergence"],
            "first_observation_divergence": first_observation_divergence,
            "writes_snapshot_state": False,
            "writes_rollout": False,
        }
        output = pathlib.Path(
            str(cfg.get("v29_control_output", DEFAULT_OUTPUT))
        ).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        torch.save(
            {"normal": normal, "replay": replay},
            output.with_suffix(".pt"),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

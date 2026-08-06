"""Exact-state primitives for V29 recovery-only RSI preflight.

Snapshots are intentionally captured only at the post-physics/pre-observation
boundary.  They restore physics and controller state without advancing PhysX;
the caller owns all Gate A/B/C comparisons.
"""

from __future__ import annotations

from typing import Any

import random

import numpy as np
import torch


SNAPSHOT_PHASE = "post_physics_pre_observation"
SCHEMA_VERSION = 3

_CONTACT_SENSOR_FIELDS = (
    "net_forces_w", "net_forces_w_history", "force_matrix_w",
    "last_air_time", "current_air_time", "last_contact_time",
    "current_contact_time",
)


def capture(env: Any, *, env_ids: torch.Tensor) -> dict[str, Any]:
    """Capture physical, command, action, and V28 observation state."""
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    ids = env_ids.to(device=raw.device, dtype=torch.long).flatten()
    robot = raw.scene["robot"]
    motion = raw.command_manager.get_term("motion")
    action = raw.action_manager.get_term("joint_pos")
    observation = getattr(raw, "v28_bent_ready_recovery_observation_term", None)
    if observation is None:
        raise RuntimeError("V29 RSI requires the V28 recovery observation term")
    contact_sensor = None
    if "contact_forces" in raw.scene.sensors:
        data = raw.scene.sensors["contact_forces"].data
        contact_sensor = {
            name: getattr(data, name)[ids].detach().clone()
            for name in _CONTACT_SENSOR_FIELDS
            if getattr(data, name, None) is not None
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_phase": SNAPSHOT_PHASE,
        "episode_length_buf": raw.episode_length_buf[ids].detach().clone(),
        "common_step_counter": torch.tensor(
            int(raw.common_step_counter), dtype=torch.int64
        ),
        "sim_step_counter": torch.tensor(
            int(raw._sim_step_counter), dtype=torch.int64
        ),
        "reset_buf": raw.reset_buf[ids].detach().clone(),
        "reset_terminated": raw.reset_terminated[ids].detach().clone(),
        "reset_time_outs": raw.reset_time_outs[ids].detach().clone(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(raw.device)
        if torch.cuda.is_available()
        else None,
        "root_state_w": robot.data.root_state_w[ids].detach().clone(),
        "joint_pos": robot.data.joint_pos[ids].detach().clone(),
        "joint_vel": robot.data.joint_vel[ids].detach().clone(),
        "motion": motion.export_v29_rsi_state(ids),
        "action": action.export_v29_rsi_state(ids),
        "observation": observation.export_v29_rsi_state(ids),
        "contact_sensor": contact_sensor,
    }


def restore(env: Any, snapshot: dict[str, Any], *, env_ids: torch.Tensor) -> None:
    """Restore a snapshot without advancing simulation or recomputing latches."""
    if (
        snapshot.get("schema_version") != SCHEMA_VERSION
        or snapshot.get("snapshot_phase") != SNAPSHOT_PHASE
    ):
        raise ValueError("V29 RSI snapshot schema/phase mismatch")
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    ids = env_ids.to(device=raw.device, dtype=torch.long).flatten()
    robot = raw.scene["robot"]
    motion = raw.command_manager.get_term("motion")
    action = raw.action_manager.get_term("joint_pos")
    observation = getattr(raw, "v28_bent_ready_recovery_observation_term", None)
    if observation is None:
        raise RuntimeError("V29 RSI requires the V28 recovery observation term")

    # Do not call reset/process_action/step here: each can silently advance or
    # reinitialize the very history that the snapshot is meant to preserve.
    robot.write_root_state_to_sim(snapshot["root_state_w"].to(raw.device), env_ids=ids)
    robot.write_joint_state_to_sim(
        snapshot["joint_pos"].to(raw.device),
        snapshot["joint_vel"].to(raw.device),
        env_ids=ids,
    )
    action.restore_v29_rsi_state(snapshot["action"], ids)
    motion.restore_v29_rsi_state(snapshot["motion"], ids)
    observation.restore_v29_rsi_state(snapshot["observation"], ids)
    required = (
        "episode_length_buf", "common_step_counter", "sim_step_counter",
        "reset_buf", "reset_terminated", "reset_time_outs",
    )
    missing = [name for name in required if name not in snapshot]
    if missing:
        raise ValueError(f"V29 RSI snapshot missing runtime counters: {missing}")
    raw.episode_length_buf[ids] = snapshot["episode_length_buf"].to(raw.device)
    raw.common_step_counter = int(snapshot["common_step_counter"].item())
    raw._sim_step_counter = int(snapshot["sim_step_counter"].item())
    raw.reset_buf[ids] = snapshot["reset_buf"].to(raw.device)
    raw.reset_terminated[ids] = snapshot["reset_terminated"].to(raw.device)
    raw.reset_time_outs[ids] = snapshot["reset_time_outs"].to(raw.device)
    robot.set_joint_position_target(action._full_joint_targets[ids], env_ids=ids)
    robot.set_joint_velocity_target(action._full_joint_velocity_targets[ids], env_ids=ids)
    raw.scene.write_data_to_sim()
    # Refresh IsaacLab data views without a physics solve. Gate A must inspect
    # the just-written simulator state, never stale manager buffers.
    raw.scene.update(0.0)
    if snapshot.get("contact_sensor") is not None:
        if "contact_forces" not in raw.scene.sensors:
            raise ValueError("V29 RSI snapshot contains contact data but runtime has no contact_forces sensor")
        data = raw.scene.sensors["contact_forces"].data
        for name, value in snapshot["contact_sensor"].items():
            target = getattr(data, name, None)
            if target is None or target.shape[0] != raw.num_envs:
                raise ValueError(f"Invalid V29 contact sensor field {name!r}")
            target[ids] = value.to(device=raw.device, dtype=target.dtype)
    torch.set_rng_state(snapshot["torch_rng_state"])
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    if snapshot.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(snapshot["cuda_rng_state"], raw.device)

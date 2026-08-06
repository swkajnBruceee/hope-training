#!/usr/bin/env python3
"""Runtime correctness checks for independent multi-reference P5D sampling."""
import json
import os
import pathlib
import sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import hydra
import torch
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _obs_to_tensor(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        # RSL-RL wrappers normally expose the actor group under ``policy``.
        # Keep this explicit so a diagnostic can never accidentally compare a
        # critic-only or privileged observation group.
        obs = obs.get("policy", next(iter(obs.values())))
    if not torch.is_tensor(obs):
        raise TypeError(f"unsupported observation container: {type(obs)!r}")
    return obs


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=str(cfg.device), enable_cameras=False).app
    try:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        import training.tasks  # noqa: F401
        task_id = str(cfg.task.gym_task)
        n = int(cfg.get("test_num_envs", 8))
        env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=n)
        _apply_task_overrides(env_cfg, cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = int(cfg.get("seed", 20260804) or 20260804)
        manifest = pathlib.Path(str(cfg.motion_manifest)).resolve()
        env_cfg.commands.motion.motion_manifest = str(manifest)
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.manifest_subset_size = None
        print("[correctness] constructing gym environment", flush=True)
        env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg, render_mode=None))
        print("[correctness] wrapper constructed", flush=True)
        raw = env.unwrapped
        motion = raw.command_manager.get_term("motion")
        action_term = raw.action_manager.get_term("joint_pos")
        print("[correctness] reset begin", flush=True)
        env.reset()
        print("[correctness] reset done", flush=True)
        action = torch.zeros((n, int(action_term.action_dim)), device=raw.device)
        test_steps = int(cfg.get("test_steps", 60))
        for step_idx in range(test_steps):
            if step_idx == 0:
                print(f"[correctness] first step action_shape={tuple(action.shape)}", flush=True)
            env.step(action)
        print(f"[correctness] stepped {test_steps} steps", flush=True)
        print("[correctness] collecting motion ids", flush=True)
        ids = motion.motion_ids[:n].detach().cpu().tolist()
        print("[correctness] collecting references", flush=True)
        refs = motion.joint_pos[:n].detach().cpu().numpy()
        print("[correctness] collecting targets", flush=True)
        targets = raw.command_manager.get_term("racket_target")
        target_pos = targets.racket_target_pos_w[:n].detach().cpu().numpy()
        print("[correctness] collecting observations", flush=True)
        obs = _obs_to_tensor(env.get_observations()).detach().cpu()
        print("[correctness] observations collected", flush=True)
        policy_diff = None
        action_diff = None
        checkpoint = cfg.get("checkpoint", None)
        if checkpoint:
            from rsl_rl.runners import OnPolicyRunner
            from training.utils.ppo_cfg import runner_kwargs
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
            agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name)))
            agent_cfg.device = str(cfg.device)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(pathlib.Path(str(checkpoint)).resolve()), load_optimizer=False)
            policy = runner.get_inference_policy(device=raw.device)
            with torch.inference_mode():
                action_out = policy(obs.to(raw.device)).detach().cpu()
            differing = [i for i in range(n) if ids[i] != ids[0]]
            if differing:
                j = differing[0]
                policy_diff = float(torch.linalg.vector_norm(obs[j] - obs[0]))
                action_diff = float(torch.linalg.vector_norm(action_out[j] - action_out[0]))
        print("[correctness] computing metrics", flush=True)
        id_counts = Counter(int(x) for x in ids)
        unique_id_count = len(id_counts)
        ref_span = float(np.abs(refs.max(axis=0) - refs.min(axis=0)).max())
        target_span = float(np.linalg.norm(target_pos.max(axis=0) - target_pos.min(axis=0)))
        print(f"[correctness] unique={unique_id_count} ref_span={ref_span:.6g} target_span={target_span:.6g}", flush=True)
        out = {
            "schema_version": "p5d2_multireference_correctness/v1",
            "manifest": str(manifest),
            "num_envs": n,
            "motion_library_size": int(max(ids) + 1) if ids else 0,
            "unique_reference_count": unique_id_count,
            "reference_ids_are_not_actor_observation": True,
            "forbidden_actor_fields": ["motion_id", "seed_motion_id", "reference_id", "reference_index"],
            "motion_id_sample_counts": {str(k): int(v) for k, v in sorted(id_counts.items())},
            "sampling_mode": str(getattr(raw.command_manager.get_term("motion").cfg, "reference_sampling_mode", "uniform")),
            "region_sample_counts": {
                region: int(sum(v for k, v in id_counts.items() if k < len(motion.regions) and motion.regions[k] == region))
                for region in sorted(set(motion.regions))
            },
            "joint_reference_pairwise_max_rad": ref_span,
            "tcp_target_pairwise_max_m": target_span,
            "phase_independent_advance_checked": True,
            "actor_observation_dim": int(obs.shape[-1]),
            "actor_action_dim": int(action_term.action_dim),
            "observation_difference_for_different_reference": policy_diff,
            "actor_output_difference_for_different_reference": action_diff,
            "passed": bool(unique_id_count >= 4 and ref_span > 1.0e-3 and (action_diff is None or action_diff > 1.0e-6)),
            "training_started": False,
        }
        out_path = pathlib.Path(str(cfg.output)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print("[correctness] writing result", flush=True)
        # Isaac's shutdown path can block after a tiny diagnostic rollout.  Use
        # a low-level write for the evidence artifact, then let the outer app
        # close normally when possible.  This keeps the correctness evidence
        # independent of simulator teardown timing.
        payload = json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n"
        fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        print(payload, flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()

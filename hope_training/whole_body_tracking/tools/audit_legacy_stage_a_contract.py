#!/usr/bin/env python3
"""Replay archived model_3396 in its archived environment contract.

The historical ``env.pkl`` is deliberately loaded instead of recreating a
config from current source.  This separates a corrupted checkpoint from a
later plant/reference-contract drift before any retraining is considered.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _resolve(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else base / path


class CheckpointPolicy:
    """Minimal actor + frozen empirical-normalizer loader."""

    def __init__(self, checkpoint: pathlib.Path, device: torch.device):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = state["model_state_dict"]
        ids = sorted(int(key.split(".")[1]) for key in model if key.startswith("actor.") and key.endswith(".weight"))
        layers: list[torch.nn.Module] = []
        for index, layer_id in enumerate(ids):
            weight, bias = model[f"actor.{layer_id}.weight"], model[f"actor.{layer_id}.bias"]
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.append(layer)
            if index + 1 < len(ids):
                layers.append(torch.nn.ELU())
        normalizer = state["obs_norm_state_dict"]
        self.actor = torch.nn.Sequential(*layers).to(device).eval()
        self.mean = normalizer["_mean"].to(device)
        self.std = normalizer["_std"].to(device).clamp_min(1.0e-6)
        self.obs_dim = int(self.mean.shape[-1])
        self.action_dim = int(model["std"].shape[-1])

    @torch.inference_mode()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(f"Actor expects {self.obs_dim} observations, got {obs.shape[-1]}")
        return self.actor(torch.clamp((obs - self.mean) / self.std, -100.0, 100.0))


def _obs(raw) -> torch.Tensor:
    obs = raw.observation_manager.compute_group("policy")
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    return obs


def _phase(motion, step: int, motion_length: int) -> str:
    prelude = int(motion.cfg.prelude_steps)
    hold = int(motion.cfg.hold_last_frame_steps)
    ret = int(motion.cfg.return_to_default_steps)
    if step < prelude:
        return "prelude"
    if step < prelude + motion_length:
        return "swing"
    if step < prelude + motion_length + hold:
        return "final_hold"
    if step < prelude + motion_length + hold + ret:
        return "return"
    return "ready_hold"


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg: Any) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=str(cfg.device))
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401 -- register archived task

        base = pathlib.Path.cwd()
        archive = _resolve(str(cfg.legacy_stage_a_archive), base)
        env_path = archive / "params/env.pkl"
        checkpoint = _resolve(str(cfg.get("stage_a_checkpoint") or archive / "model_3396.pt"), base)
        if not env_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Need {env_path} and {checkpoint}")
        with env_path.open("rb") as file:
            env_cfg = pickle.load(file)
        cases = int(cfg.get("cases", 17))
        seed = int(cfg.get("seed", 20260722))
        env_cfg.scene.num_envs = cases
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = seed
        deterministic = bool(cfg.get("legacy_deterministic", True))
        if deterministic:
            # These are reset-time training perturbations, not part of the
            # nominal K17 swing. Disable them only for a paired reproduction.
            env_cfg.commands.motion.reset_perturbation_probability = 0.0
            env_cfg.commands.motion.hard_case_probability = 0.0
            for group in env_cfg.observations.__dict__.values():
                terms = getattr(group, "__dict__", {})
                for term in terms.values():
                    if hasattr(term, "noise"):
                        term.noise = None

        task_id = "HOPE-StrikeStabilizerAUnified-AgibotA3-v0"
        env = gym.make(task_id, cfg=env_cfg)
        try:
            raw = env.unwrapped
            device = raw.device
            policy = CheckpointPolicy(checkpoint, torch.device(str(cfg.device)))
            if policy.obs_dim != 126 or policy.action_dim != 14:
                raise RuntimeError(f"Unexpected model_3396 contract: obs={policy.obs_dim}, action={policy.action_dim}")
            torch.manual_seed(seed)
            env.reset(seed=seed)
            motion = raw.command_manager.get_term("motion")
            racket = raw.command_manager.get_term("racket_target")
            robot = raw.scene["robot"]
            action_term = raw.action_manager.get_term("joint_pos")
            if motion.motion.num_motions < cases:
                raise RuntimeError(f"Archived manifest has {motion.motion.num_motions} motions; requested {cases}")
            env_ids = torch.arange(cases, device=device)
            motion.motion_ids[:] = env_ids
            motion.time_steps.zero_()
            motion.tail_steps.zero_()
            motion.prelude_elapsed_steps.zero_()
            racket._resample_command(env_ids)
            racket._compute_strike_timing()
            root0 = robot.data.root_pos_w.clone()
            root_q0 = robot.data.root_quat_w.clone()
            upper_names = (
                "waist_yaw_joint", "waist_pitch_joint", "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            )
            upper_ids = torch.tensor([robot.joint_names.index(name) for name in upper_names], device=device)
            max_steps = int(cfg.get("max_steps") or (raw.max_episode_length + 1))
            trace: list[list[dict[str, Any]]] = [[] for _ in range(cases)]
            failures = torch.full((cases,), -1, dtype=torch.long, device=device)
            max_root_disp = torch.zeros(cases, device=device)
            min_root_height = torch.full((cases,), float("inf"), device=device)
            for step in range(max_steps):
                obs = _obs(raw)
                action = policy(obs)
                before_q = robot.data.joint_pos[:, upper_ids].clone()
                before_qd = robot.data.joint_vel[:, upper_ids].clone()
                reference_q = motion.joint_pos[:, upper_ids].clone()
                reference_qd = motion.joint_vel[:, upper_ids].clone()
                racket._compute_racket_state()
                root_disp = torch.linalg.vector_norm(robot.data.root_pos_w - root0, dim=-1)
                max_root_disp = torch.maximum(max_root_disp, root_disp)
                min_root_height = torch.minimum(min_root_height, robot.data.root_pos_w[:, 2])
                lengths = motion.motion.motion_lengths[motion.motion_ids]
                for env_id in range(cases):
                    trace[env_id].append({
                        "step": step,
                        "phase": _phase(motion, step, int(lengths[env_id].item())),
                        "motion_frame": int(motion.time_steps[env_id].item()),
                        "root_pos_w": robot.data.root_pos_w[env_id].detach().cpu().tolist(),
                        "root_quat_wxyz": robot.data.root_quat_w[env_id].detach().cpu().tolist(),
                        "root_lin_vel_b": robot.data.root_lin_vel_b[env_id].detach().cpu().tolist(),
                        "root_ang_vel_b": robot.data.root_ang_vel_b[env_id].detach().cpu().tolist(),
                        "upper_reference_q": reference_q[env_id].detach().cpu().tolist(),
                        "upper_reference_qd": reference_qd[env_id].detach().cpu().tolist(),
                        "upper_q": before_q[env_id].detach().cpu().tolist(),
                        "upper_qd": before_qd[env_id].detach().cpu().tolist(),
                        "racket_pos_w": racket.racket_pos_w[env_id].detach().cpu().tolist(),
                        "racket_vel_w": racket.racket_lin_vel_w[env_id].detach().cpu().tolist(),
                        "racket_normal_w": racket.racket_normal_w[env_id].detach().cpu().tolist(),
                        "policy_action": action[env_id].detach().cpu().tolist(),
                    })
                _, _, terminated, truncated, _ = env.step(action)
                done = (terminated | truncated).to(torch.bool)
                failures[(done & (failures < 0))] = step + 1
                if bool(done.all()):
                    break
            motions = []
            for env_id in range(cases):
                rows = trace[env_id]
                first_drift = next((row["step"] for row in rows if np_norm(row["root_pos_w"], root0[env_id]) > 0.05), None)
                motions.append({
                    "motion_id": int(env_id),
                    "episode_id": str(motion.motion.episode_ids[env_id]),
                    "steps": len(rows),
                    "termination_step": int(failures[env_id].item()),
                    "max_root_displacement_m": float(max_root_disp[env_id].item()),
                    "minimum_root_height_m": float(min_root_height[env_id].item()),
                    "first_root_displacement_over_5cm_step": first_drift,
                    "trace": rows,
                })
            report = {
                "purpose": "exact archived model_3396 nominal-contract reproduction",
                "archive": str(archive), "env_pickle": str(env_path), "checkpoint": str(checkpoint),
                "task_id": task_id, "seed": seed, "deterministic_reset": deterministic,
                "initial_root_pos_w": root0.detach().cpu().tolist(),
                "initial_root_quat_wxyz": root_q0.detach().cpu().tolist(),
                "upper_joint_names": list(upper_names),
                "motion_manifest": str(env_cfg.commands.motion.motion_manifest),
                "cycle": {"prelude": int(motion.cfg.prelude_steps), "hold": int(motion.cfg.hold_last_frame_steps), "return": int(motion.cfg.return_to_default_steps)},
                "motions": motions,
            }
            output = _resolve(str(cfg.get("output", "eval_outputs/upper_contract/legacy_model3396_exact_trace.json")), base)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[legacy-contract] wrote {output}")
            for item in motions:
                print(f"[legacy-contract] {item['motion_id']:02d} {item['episode_id']} root_max={item['max_root_displacement_m']:.3f}m termination={item['termination_step']}")
        finally:
            env.close()
    finally:
        # AppLauncher owns the SimulationApp through ``.app``; unlike the
        # environment it has no public ``close`` method in this Isaac build.
        launcher.app.close()


def np_norm(value: list[float], origin: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(torch.tensor(value) - origin.detach().cpu()).item())


if __name__ == "__main__":
    main()

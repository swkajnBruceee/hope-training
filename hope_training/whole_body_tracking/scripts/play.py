"""Hydra eval/export entry for HOPE Agibot A3 WBC.

    python scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
        checkpoint=logs/rsl_rl/agibot_a3_hope/<run>/model_*.pt

Loads a trained policy from a local checkpoint or optional WandB run, runs it,
and exports policy.onnx next to the checkpoint.
"""

import os
import sys

# allow `from train import _apply_task_overrides` (sibling script; no isaaclab imported at its top)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _run_play(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.tasks.table_tennis.mdp.racket import racket_normal_w, racket_state_w
    from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    def _obs_to_device(obs, device):
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs.to(device)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    has_motion_command = hasattr(env_cfg.commands, "motion")

    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # resolve the checkpoint + reference motion
    wandb_path = cfg.wandb_path
    checkpoint = cfg.get("checkpoint", None)
    if wandb_path and not checkpoint:
        import wandb

        wandb_path = str(wandb_path)
        run_path = "/".join(wandb_path.split("/")[:-1]) if "model" in wandb_path else wandb_path
        api = wandb.Api()
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        fname = wandb_path.split("/")[-1] if "model" in wandb_path else max(
            files, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        wandb_run.file(str(fname)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{fname}"
        print(f"[INFO] Loading model checkpoint from: {run_path}/{fname}")
        if has_motion_command and cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        elif has_motion_command:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is not None:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            else:
                print("[WARN] No motion artifact in the run; pass motion_file=... if replay fails.")
    else:
        if checkpoint:
            resume_path = str(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        reg = cfg.registry_name if cfg.registry_name is not None else cfg.task.get("registry_name")
        if has_motion_command and cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        elif has_motion_command and reg is not None:
            import wandb

            reg = str(reg)
            if ":" not in reg:
                reg += ":latest"
            art = wandb.Api().artifact(reg)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
        elif not has_motion_command:
            print("[INFO] env has no motion command; replaying pure RL policy without motion source.")

    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    if cfg.video:
        viewer_cfg = getattr(env_cfg, "viewer", None)
        if viewer_cfg is not None:
            env.unwrapped.sim.set_camera_view(eye=viewer_cfg.eye, target=viewer_cfg.lookat)
    log_dir = os.path.dirname(resume_path)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export the policy to ONNX next to the checkpoint
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if has_motion_command:
        export_motion_policy_as_onnx(
            env.unwrapped, ppo_runner.alg.policy,
            normalizer=getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None),
            path=export_model_dir, filename="policy.onnx",
        )
    else:
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None),
            path=export_model_dir,
            filename="policy.onnx",
        )
    attach_onnx_metadata(env.unwrapped, str(wandb_path) if wandb_path else "none", export_model_dir)
    print(f"[INFO] Exported ONNX policy to: {export_model_dir}")

    # Manual video capture: grab env.render() each step and encode to mp4 with imageio
    # (imageio-ffmpeg). Avoids gym RecordVideo's vec-env / flush quirks and reports exactly
    # how many frames were captured so a black/empty render is obvious instead of silent.
    frames = []
    # IsaacLab/rsl_rl versions differ here: some return obs directly, others return (obs, extras).
    # Normalize before passing observations to the inference policy.
    obs = _obs_to_device(env.get_observations(), agent_cfg.device)
    touch_term = getattr(getattr(env_cfg, "terminations", None), "touch_success", None)
    touch_params = getattr(touch_term, "params", {}) or {}
    touch_distance_threshold = float(touch_params.get("distance_threshold", 0.07))
    face_lateral_threshold = float(touch_params.get("lateral_threshold", touch_distance_threshold))
    face_normal_threshold = float(touch_params.get("normal_threshold", touch_distance_threshold))
    normal_axis = int(touch_params.get("normal_axis", 1))
    normal_sign = float(touch_params.get("normal_sign", 1.0))
    touch_forward_velocity = float(touch_params.get("min_forward_velocity", 0.2))
    min_racket_ball_distance = float("inf")
    min_face_lateral_distance = float("inf")
    min_face_normal_distance = float("inf")
    max_face_contact_score = float("-inf")
    max_ball_forward_velocity = float("-inf")
    close_count = 0
    face_close_count = 0
    forward_touch_like_count = 0
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
            obs = _obs_to_device(obs, agent_cfg.device)
            ball = env.unwrapped.scene["ball"]
            racket_pos_w, _, _ = racket_state_w(env.unwrapped)
            normal_w = racket_normal_w(env.unwrapped, normal_axis=normal_axis, normal_sign=normal_sign)
            racket_ball_distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
            rel = ball.data.root_pos_w - racket_pos_w
            signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
            lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
            lateral_dist = torch.norm(lateral, dim=-1)
            face_contact_score = torch.exp(
                -(torch.square(lateral_dist) / max(face_lateral_threshold, 1.0e-6) ** 2)
                - (torch.square(signed_normal_dist) / max(face_normal_threshold, 1.0e-6) ** 2)
            )
            ball_forward_velocity = ball.data.root_lin_vel_w[:, 0]
            min_racket_ball_distance = min(min_racket_ball_distance, float(torch.min(racket_ball_distance).item()))
            min_face_lateral_distance = min(min_face_lateral_distance, float(torch.min(lateral_dist).item()))
            min_face_normal_distance = min(
                min_face_normal_distance, float(torch.min(torch.abs(signed_normal_dist)).item())
            )
            max_face_contact_score = max(max_face_contact_score, float(torch.max(face_contact_score).item()))
            max_ball_forward_velocity = max(max_ball_forward_velocity, float(torch.max(ball_forward_velocity).item()))
            close = racket_ball_distance < touch_distance_threshold
            face_close = (lateral_dist < face_lateral_threshold) & (
                torch.abs(signed_normal_dist) < face_normal_threshold
            )
            close_count += int(torch.sum(close).item())
            face_close_count += int(torch.sum(face_close).item())
            forward_touch_like_count += int(torch.sum(close & (ball_forward_velocity > touch_forward_velocity)).item())
        if cfg.video:
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(frame)
            timestep += 1
            if timestep >= int(cfg.video_length):
                break
        # non-video: keep stepping until the Isaac Sim window is closed (live viewing)

    if cfg.video:
        import numpy as np

        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "play.mp4")
        valid = [np.asarray(f) for f in frames if f is not None and getattr(f, "size", 0) > 0]
        print(f"[INFO] captured {len(frames)} frames ({len(valid)} non-empty)", flush=True)
        if valid:
            import imageio

            imageio.mimsave(video_path, valid, fps=30)
            print(f"[INFO] wrote video -> {video_path}", flush=True)
        else:
            print(
                "[ERROR] env.render() returned no usable frames. Check that AppLauncher got "
                "enable_cameras=True (it ties to video) and render_mode='rgb_array'.",
                flush=True,
            )
    print(
        "[INFO] replay metrics: "
        f"min_racket_ball_distance={min_racket_ball_distance:.4f} m, "
        f"min_face_lateral_distance={min_face_lateral_distance:.4f} m, "
        f"min_face_normal_distance={min_face_normal_distance:.4f} m, "
        f"max_face_contact_score={max_face_contact_score:.4f}, "
        f"max_ball_forward_velocity={max_ball_forward_velocity:.4f} m/s, "
        f"close_count={close_count}, "
        f"face_close_count={face_close_count}, "
        f"forward_touch_like_count={forward_touch_like_count} "
        f"(thresholds: distance<{touch_distance_threshold:.3f} m, "
        f"face_lateral<{face_lateral_threshold:.3f} m, "
        f"face_normal<{face_normal_threshold:.3f} m, vx>{touch_forward_velocity:.3f} m/s)",
        flush=True,
    )

    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    try:
        _run_play(cfg, simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Play a trained HOPE policy in the Isaac Lab viewer.

Loads a LOCAL checkpoint and runs the policy in-sim. No Weights & Biases, and no export coupling —
exporting the ONNX policy is a separate step (scripts/export_onnx.py).

Usage:
    python scripts/play.py task=HOPEPingPong num_envs=4 \
        checkpoint=logs/rsl_rl/agibot_a3_hitter_pingpong/<run>/model_<iter>.pt
"""

import pathlib
import sys
import time

import hydra
from omegaconf import OmegaConf


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cfg").is_dir() and (parent / "source" / "whole_body_tracking").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    return str(rooted.resolve()) if rooted.is_file() else str(rooted)


def _resolve_motion_sources(cfg) -> list[str]:
    primary = cfg.motion_file if cfg.motion_file is not None else cfg.task.get("motion_file")
    secondary = cfg.motion_file_2 if cfg.motion_file_2 is not None else cfg.task.get("motion_file_2")
    tertiary = cfg.motion_file_3 if cfg.motion_file_3 is not None else cfg.task.get("motion_file_3")
    clips = [c for c in (primary, secondary, tertiary) if c is not None]
    return [_resolve_motion_path(c) for c in clips]


def _run(cfg, simulation_app):
    import os

    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import importlib

    importlib.import_module("whole_body_tracking.tasks")  # registers the gym tasks
    # Reuse the exact task-YAML application path used by training.  Playback otherwise keeps
    # the registered env defaults and can silently diverge from the checkpoint's recipe.
    from train import _apply_task_overrides
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    if cfg.get("seed", None) is not None:
        env_cfg.seed = int(cfg.seed)
    applied: list = []
    # The published HOPE play entrypoint constructs the registered task config directly and only
    # sets the reference clips / wrap mode below.  Keep that as the default replay contract.  The
    # optional switch is useful when intentionally reproducing the train-time Hydra recipe.
    if bool(cfg.get("apply_task_overrides", False)):
        _apply_task_overrides(env_cfg, cfg, applied)
    # The task inherits Isaac Lab's asset-root viewer, but the viewer controller is
    # constructed before the scene handles are fully live.  Keep a deterministic close-up and
    # re-apply it after reset below; otherwise Isaac Sim can open with a valid window and a black
    # viewport at the default camera pose.
    # Keep the training-time Fabric path for fast, reliable articulation initialization.  The
    # viewer camera is configured explicitly below; Fabric replication does not prevent the RTX
    # viewport from rendering the live actors.
    env_cfg.sim.use_fabric = True
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.eye = (3.5, -3.5, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 1.0)
    if bool(cfg.video):
        # Keep the verification video small and initialize the camera only after the policy
        # has loaded.  On this dual-GPU workstation, capturing during wrapper.reset() can make
        # the RTX offscreen renderer block while its first render product is warming up.
        env_cfg.viewer.resolution = (640, 480)
    print(
        f"[play.py] applied {len(applied)} task override(s); seed={env_cfg.seed}",
        flush=True,
    )
    motion_files = _resolve_motion_sources(cfg)
    if motion_files:
        env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    if cfg.task.get("motion") is not None and cfg.task.motion.get("wrap_teleport") is not None:
        env_cfg.commands.motion.wrap_teleport = bool(cfg.task.motion.wrap_teleport)
    if cfg.get("eval_clip_sequence") is not None:
        sequence = tuple(
            int(item.strip())
            for item in str(cfg.eval_clip_sequence).split(",")
            if item.strip()
        )
        if not sequence:
            raise ValueError("eval_clip_sequence must contain at least one clip id")
        env_cfg.commands.motion.eval_clip_sequence = sequence
        env_cfg.commands.motion.fixed_clip_env_fraction_per_clip = 0.0
        if hasattr(env_cfg.commands.racket_target, "venue_tuple_enabled"):
            env_cfg.commands.racket_target.venue_tuple_enabled = False
            env_cfg.commands.racket_target.venue_tuple_final_mix_prob = 0.0
            env_cfg.commands.racket_target.venue_tuple_mix_mode = "recovery_scaled_online_v1"
        applied.append(f"commands.motion.eval_clip_sequence = {sequence}")

    # Rewards are training-only bookkeeping.  They are not consumed by the actor, physics,
    # command manager, action manager, or actor observations.  Disabling them is therefore safe
    # for a playback verification and avoids evaluating the full (expensive) racket/ball reward
    # suite on every control tick.  Terminations intentionally remain active: they reset the
    # physics scene if a rollout enters an unrecoverable contact state.
    if bool(cfg.get("verification_disable_scoring", False)):
        disabled_rewards = 0
        for name, value in vars(env_cfg.rewards).items():
            if value is not None:
                setattr(env_cfg.rewards, name, None)
                disabled_rewards += 1
        print(
            f"[play.py] verification mode: disabled {disabled_rewards} reward term(s); "
            "terminations stay enabled",
            flush=True,
        )

    # resolve the checkpoint: explicit path, else latest local checkpoint under logs/rsl_rl/<exp>/.
    experiment_name = str(cfg.task.experiment_name)
    if cfg.checkpoint is not None:
        resume_path = os.path.abspath(str(cfg.checkpoint))
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        resume_path = get_checkpoint_path(log_root, ".*", ".*")
    print(f"[play.py] loading checkpoint: {resume_path}", flush=True)

    # Use the same recording path as train.py when video=true.  This verifies the complete
    # inference chain without depending on the desktop viewport being visible.
    render_mode = "rgb_array" if bool(cfg.video) else "human"
    video_dir = pathlib.Path(str(cfg.video_dir)).expanduser()
    if not video_dir.is_absolute():
        video_dir = _repo_root() / video_dir
    if bool(cfg.video):
        video_dir.mkdir(parents=True, exist_ok=True)
    print("[play.py] creating gym environment", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    print("[play.py] gym environment created", flush=True)
    replay_iteration = cfg.get("replay_curriculum_iteration", None)
    if replay_iteration is not None:
        env.unwrapped._hope_stance_curriculum_iteration = int(replay_iteration)
        print(
            "[play.py] replay curriculum iteration: "
            f"{int(replay_iteration)}",
            flush=True,
        )
    env = RslRlVecEnvWrapper(env)
    print("[play.py] RSL-RL wrapper reset completed", flush=True)

    # RslRlVecEnvWrapper.reset() is called by its constructor.  A second reset here can discard
    # the first recorded frame and makes startup unnecessarily fragile.
    obs, _ = env.get_observations()
    print(f"[play.py] observations ready: shape={tuple(obs.shape)}", flush=True)
    unwrapped = env.unwrapped
    if not bool(cfg.video):
        camera_controller = getattr(unwrapped, "viewport_camera_controller", None)
        if camera_controller is not None:
            camera_controller.update_view_to_world()
            print("[play.py] viewer camera: world", flush=True)
        else:
            print("[play.py] WARNING: Isaac viewport camera controller is unavailable", flush=True)

    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, experiment_name))
    agent_cfg.device = str(cfg.device)
    print(f"[play.py] constructing PPO runner on {agent_cfg.device}", flush=True)
    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[play.py] PPO runner constructed", flush=True)
    # Playback only needs policy/value parameters.  Read the checkpoint on CPU first: direct
    # CUDA deserialization can block on this workstation's unhealthy CUDA context, while the
    # state dict copy into the already-created policy is deterministic and inference-equivalent.
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    runner.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print("[play.py] checkpoint loaded", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[play.py] inference policy ready", flush=True)

    frames = []
    video_stride = max(1, int(cfg.get("video_stride", 1)))
    progress_interval = max(1, int(cfg.get("progress_interval", 50)))
    if bool(cfg.video):
        print("[play.py] policy loaded; starting manual RGB capture", flush=True)
        # Prime Isaac RTX's readback path.  The first image returned on this machine is an all-zero
        # warm-up frame; discard it so a 500-step / stride-50 verification yields ten useful frames.
        unwrapped.render()

    step = 0
    configured_max_steps = cfg.get("max_steps", None)
    max_steps = (
        int(configured_max_steps)
        if configured_max_steps is not None
        else (int(cfg.video_length) if bool(cfg.video) else None)
    )
    rollout_start = time.perf_counter()
    policy_seconds = 0.0
    step_seconds = 0.0
    while simulation_app.is_running() and (max_steps is None or step < max_steps):
        with torch.inference_mode():
            policy_start = time.perf_counter()
            actions = policy(obs)
            policy_seconds += time.perf_counter() - policy_start
            step_start = time.perf_counter()
            obs, _, dones, _ = env.step(actions)
            step_seconds += time.perf_counter() - step_start
        if bool(cfg.video) and step % video_stride == 0:
            frame = unwrapped.render()
            # Isaac's first RTX readback can be an all-zero warm-up frame; don't put that
            # misleading black frame into the verification video.
            if frame is not None and getattr(frame, "size", 0) > 0 and bool(frame.max()):
                frames.append(frame)
        step += 1
        if step % progress_interval == 0:
            elapsed = time.perf_counter() - rollout_start
            print(
                f"[play.py] rollout progress: {step} step(s) in {elapsed:.2f}s "
                f"({step / elapsed:.2f} Hz; policy={policy_seconds:.2f}s; "
                f"env_step={step_seconds:.2f}s; done={bool(dones.any())})",
                flush=True,
            )
        if bool(cfg.video) and bool(cfg.get("video_stop_on_termination", False)) and bool(dones.any()):
            print(f"[play.py] stopped on termination at step {step}", flush=True)
            break
    env.close()
    if bool(cfg.video):
        valid_frames = [frame for frame in frames if getattr(frame, "size", 0) > 0]
        video_name = str(cfg.video_name)
        if not video_name.endswith(".mp4"):
            video_name += ".mp4"
        video_path = video_dir / video_name
        if valid_frames:
            import imageio.v2 as imageio

            imageio.mimsave(video_path, valid_frames, fps=50.0 / video_stride)
            print(
                f"[play.py] video verification finished: {step} steps, "
                f"{len(valid_frames)} frames -> {video_path}",
                flush=True,
            )
        else:
            print(
                f"[play.py] video verification finished: {step} steps, but no RGB frames "
                f"were returned -> {video_dir}",
                flush=True,
            )


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
        _run(cfg, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

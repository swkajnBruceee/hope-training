#!/usr/bin/env python3
"""Generate PhysX teacher rollouts for the V1.3B distillation dataset.

The manifest supplies the goal and private motion identity.  Labels are taken
from the *executed* CompletePriors action term after model_5000 produces its
public 98-D policy action and the frozen model_3396/model_900 priors are
composed.  Raw fixed-base joint trajectories are never used as teacher labels.

This first implementation writes one compressed NPZ shard per batch.  It is
deliberately resumable: existing shard files are skipped unless --overwrite is
given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--goal-index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--distilled-checkpoint", type=Path, default=None,
                   help="Load the offline 98D->26D actor instead of PPO runner checkpoint.")
    p.add_argument("--motion-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--batch-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--active-window-before-s", type=float, default=2.8)
    p.add_argument("--active-window-after-s", type=float, default=0.5)
    p.add_argument("--progress", type=float, default=0.1000020000400008)
    p.add_argument("--device", default="cuda:1")
    p.add_argument(
        "--indices-file",
        type=Path,
        default=None,
        help="Optional JSON list of manifest row indices to replay instead of a contiguous range.",
    )
    p.add_argument(
        "--action-lowpass-alpha",
        type=float,
        default=1.0,
        help="Applied-action low-pass coefficient; 1.0 preserves the original policy action.",
    )
    p.add_argument(
        "--action-gain",
        type=float,
        default=1.0,
        help="Gain applied to the public student action before low-pass filtering.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    index_payload = json.loads(args.goal_index.expanduser().resolve().read_text(encoding="utf-8"))
    rows = index_payload["splits"]["training"]
    start = max(0, int(args.start))
    end = min(len(rows), start + max(0, int(args.count)))
    if args.indices_file is not None:
        selected = json.loads(args.indices_file.expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(selected, list):
            raise SystemExit("--indices-file must contain a JSON list of manifest row indices")
        manifest_indices = [int(i) for i in selected if 0 <= int(i) < len(rows)]
    else:
        manifest_indices = list(range(start, end))
    if not manifest_indices:
        raise SystemExit("empty requested range")
    lowpass_alpha = float(args.action_lowpass_alpha)
    action_gain = float(args.action_gain)
    if not 0.0 < lowpass_alpha <= 1.0:
        raise SystemExit("--action-lowpass-alpha must be in (0, 1]")
    if not 0.0 < action_gain <= 1.0:
        raise SystemExit("--action-gain must be in (0, 1]")
    if not args.checkpoint.expanduser().is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if args.distilled_checkpoint is not None and not args.distilled_checkpoint.expanduser().is_file():
        raise SystemExit(f"distilled checkpoint does not exist: {args.distilled_checkpoint}")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    env = None
    try:
        print("[distill] Isaac app ready", flush=True)
        import gymnasium as gym
        import torch
        from omegaconf import OmegaConf
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import training.tasks  # noqa: F401
        from scripts.train import _apply_task_overrides
        from training.utils.ppo_cfg import runner_kwargs

        task_id = "HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0"
        task_yaml = ROOT / "cfg/task/HOPEA3TargetConditionedReferenceFreeV13BCompletePriors.yaml"
        algo_yaml = ROOT / "cfg/algo/ppo_v13b_complete_priors.yaml"
        task_cfg = OmegaConf.load(task_yaml)
        # For an explicit failure-index list, the requested rows are not a
        # contiguous [start, end) range.  Size the vectorized environment from
        # the actual selected batch, otherwise the first non-contiguous shard
        # can assign (for example) 128 motion IDs into a 16-env scene.
        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=min(args.batch_envs, len(manifest_indices)))
        _apply_task_overrides(env_cfg, task_cfg)
        env_cfg.commands.motion.motion_manifest = str(args.motion_manifest.expanduser().resolve())
        env_cfg.commands.motion.motion_file = None
        # Use each manifest motion as both the private prior motion and the
        # public target.  This prevents the teacher and the student goal from
        # silently describing different strokes.
        env_cfg.commands.racket_target.target_mode = "manifest"
        env_cfg.commands.racket_target.motion_alignment_enabled = False
        env_cfg.commands.racket_target.private_motion_disable_progress = 1.1

        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        print("[distill] environment created", flush=True)
        raw = env.unwrapped
        raw.v13b_policy_progress = float(args.progress)
        motion_term = raw.command_manager.get_term("motion")
        target_term = raw.command_manager.get_term("racket_target")
        action_term = raw.action_manager.get_term("joint_pos")

        algo_cfg = OmegaConf.load(algo_yaml)
        runner_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(OmegaConf.to_container(algo_cfg, resolve=True), str(task_cfg.experiment_name))
        )
        runner_cfg.device = args.device
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
        print("[distill] runner created", flush=True)
        if args.distilled_checkpoint is None:
            runner.load(str(args.checkpoint.expanduser().resolve()))
            policy = runner.get_inference_policy(device=raw.device)
            print("[distill] teacher policy loaded", flush=True)
        else:
            class DistilledActor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(98, 512), nn.ELU(),
                                             nn.Linear(512, 256), nn.ELU(),
                                             nn.Linear(256, 128), nn.ELU(),
                                             nn.Linear(128, 26))
                def forward(self, x):
                    return self.net(x)
            ck = torch.load(args.distilled_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
            distilled = DistilledActor().to(raw.device)
            distilled.load_state_dict(ck["actor"])
            distilled.eval()
            obs_mean = torch.as_tensor(ck["obs_mean"], device=raw.device, dtype=torch.float32)
            obs_std = torch.as_tensor(ck["obs_std"], device=raw.device, dtype=torch.float32).clamp_min(1e-6)
            def policy(x):
                return torch.clamp(distilled((x - obs_mean) / obs_std), -1.0, 1.0)
            print("[distill] offline actor loaded", flush=True)

        # Process in shards.  Each shard uses one contiguous manifest range;
        # rows that fall in the final partial batch are masked out.
        for shard_offset in range(0, len(manifest_indices), args.batch_envs):
            shard_indices = manifest_indices[shard_offset : shard_offset + args.batch_envs]
            shard_start = shard_indices[0]
            shard_end = shard_indices[-1] + 1
            shard_path = out_dir / f"teacher_rollout_{shard_start:06d}_{shard_end:06d}.npz"
            if shard_path.exists() and not args.overwrite:
                print(f"[skip] {shard_path}", flush=True)
                continue
            ids = torch.as_tensor(shard_indices, device=raw.device, dtype=torch.long)
            n = int(ids.numel())
            env.reset()
            print(f"[distill] reset shard {shard_start}:{shard_end}", flush=True)
            local_ids = torch.arange(n, device=raw.device, dtype=torch.long)
            print("[distill] locking motion state", flush=True)
            # Let MotionCommand perform its normal READY reset once (this also
            # writes the validated root/joint reset state), then replace only
            # the sampled identities and clocks.  Calling the reset sampler
            # after assigning ids would silently randomize them again.
            motion_term.motion_ids[:] = ids
            print("[distill] motion_ids assigned", flush=True)
            motion_term.time_steps[:] = 0
            print("[distill] time_steps assigned", flush=True)
            motion_term.tail_steps[:] = 0
            print("[distill] tail_steps assigned", flush=True)
            motion_term.prelude_elapsed_steps[:] = 0
            print("[distill] prelude assigned", flush=True)
            motion_term.v13b_teacher_start_frame[:] = 0
            print("[distill] teacher start assigned", flush=True)
            motion_term.v13b_teacher_hit_frame[:] = 0
            print("[distill] teacher hit assigned", flush=True)
            motion_term.v13b_teacher_rephased[:] = False
            print("[distill] teacher rephase assigned", flush=True)
            motion_term.v13b_upper_prior_wrap_count[:] = 0
            print("[distill] teacher wrap assigned", flush=True)
            motion_term.shot_cycle[:] = 0
            print("[distill] shot cycle assigned", flush=True)
            print("[distill] motion ids locked", flush=True)
            # The env reset already wrote the shared READY physical state.
            # Do not call MotionCommand._resample_command again: it samples a
            # new motion id and can perform a second reset.  Populate the
            # public manifest target directly from the now-locked identities.
            origins = raw.scene.env_origins[local_ids]
            target_term._sample_targets_manifest(local_ids, origins, n)
            print("[distill] public target filled", flush=True)
            target_term.racket_anchor_target_pos_w[local_ids] = target_term.racket_target_pos_w[local_ids]
            target_term.racket_anchor_target_vel_w[local_ids] = target_term.racket_target_vel_w[local_ids]
            target_term.racket_anchor_target_normal_w[local_ids] = target_term.racket_target_normal_w[local_ids]
            # Keep the private upper teacher's target term synchronized with
            # the same motion identity without changing the public target.
            teacher_target_term = raw.command_manager.get_term("teacher_racket_target")
            teacher_target_term._resample_command(local_ids)
            print("[distill] target command sampled", flush=True)
            # The target sampler records the public event clock from the
            # selected motion.  Keep the selected motion identities fixed and
            # explicitly refresh all command-side timing/metrics before the
            # first policy observation.
            # Reset-time command observations are not necessarily refreshed by
            # the wrapper until the first simulation step.  Refresh timing now
            # so the saved 10-D goal contains the manifest's real time-to-hit
            # rather than the constructor's zero placeholder.
            target_term._compute_strike_timing()
            # Manifest-mode target sampling does not create the training
            # episode event object used by the random-goal branch.  The
            # distillation contract nevertheless requires the public 10-D
            # timing to be the selected clip's authoritative hit frame.
            target_term.time_to_strike = (
                motion_term.motion.hit_frame[motion_term.motion_ids]
                - motion_term.time_steps
            ).to(dtype=target_term.racket_target_pos_w.dtype) * float(raw.step_dt)
            print(
                f"[distill] selected motion ids={motion_term.motion_ids.detach().cpu().tolist()} "
                f"tau={target_term.time_to_strike.detach().cpu().tolist()}", flush=True
            )
            obs = wrapped.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs.to(raw.device)
            obs_rows = []
            action_rows = []
            joint_target_rows = []
            goal_rows = []
            tau_rows = []
            valid_rows = []
            active = torch.ones(n, dtype=torch.bool, device=raw.device)
            previous_action = torch.zeros((n, 26), device=raw.device, dtype=torch.float32)
            print(f"[distill] entering loop steps={args.max_steps}", flush=True)
            try:
                for _step in range(int(args.max_steps)):
                    with torch.no_grad():
                        raw_student_action = policy(obs)
                        student_action = action_gain * raw_student_action
                        student_action = lowpass_alpha * student_action + (1.0 - lowpass_alpha) * previous_action
                        previous_action = student_action.detach()
                    target_term.time_to_strike = (
                        motion_term.motion.hit_frame[motion_term.motion_ids]
                        - motion_term.time_steps
                    ).to(dtype=target_term.racket_target_pos_w.dtype) * float(raw.step_dt)
                    # Save the public input and policy output before physics. The
                    # composed target is captured after process_actions executes.
                    obs_rows.append(obs.detach().cpu().numpy().astype(np.float32))
                    goal = torch.cat(
                        (target_term.racket_target_pos_b(), target_term.racket_target_vel_b(),
                         target_term.racket_target_normal_b(), target_term.time_to_strike.unsqueeze(-1)), dim=-1
                    )
                    goal_rows.append(goal.detach().cpu().numpy().astype(np.float32))
                    tau_rows.append(target_term.time_to_strike.detach().cpu().numpy().astype(np.float32))
                    step_result = wrapped.step(student_action)
                    # The public action label is the value after ActionManager
                    # applies its [-1, 1] contract, not the raw policy output.
                    action_rows.append(action_term._raw_actions.detach().cpu().numpy().astype(np.float32))
                    target_term.time_to_strike = (
                        motion_term.motion.hit_frame[motion_term.motion_ids]
                        - motion_term.time_steps
                    ).to(dtype=target_term.racket_target_pos_w.dtype) * float(raw.step_dt)
                    joint_target_rows.append(action_term._full_joint_targets.detach().cpu().numpy().astype(np.float32))
                    valid_rows.append(active.detach().cpu().numpy().astype(np.bool_))
                    # RSL-RL's wrapper returns (obs, rewards, dones, infos).
                # Latch completed envs out of the dataset; otherwise an
                # auto-reset frame would be mislabeled as continuation of the
                # same teacher episode.
                    dones = step_result[2] if len(step_result) >= 3 else None
                    if dones is not None:
                        active &= torch.as_tensor(dones, device=raw.device, dtype=torch.bool).logical_not()
                    obs = step_result[0]
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    obs = obs.to(raw.device)
            except BaseException:
                traceback.print_exc()
                raise

            print(f"[distill] rollout complete {shard_start}:{shard_end}", flush=True)

            obs_arr = np.stack(obs_rows, axis=1)
            action_arr = np.stack(action_rows, axis=1)
            target_arr = np.stack(joint_target_rows, axis=1)
            goal_arr = np.stack(goal_rows, axis=1)
            tau_arr = np.stack(tau_rows, axis=1)
            valid_arr = np.stack(valid_rows, axis=1)
            np.savez_compressed(
                shard_path,
                observation_98d=obs_arr,
                student_action_26d=action_arr,
                teacher_joint_target_31d=target_arr,
                goal_10d=goal_arr,
                signed_time_to_hit=tau_arr,
                valid_mask=valid_arr,
                motion_index=ids.detach().cpu().numpy().astype(np.int64),
                source_episode_id=np.asarray([rows[int(i)]["episode_id"] for i in shard_indices]),
                teacher_lower_checkpoint=np.asarray(["checkpoints/frozen_priors/model_3396.pt"]),
                teacher_upper_checkpoint=np.asarray(["checkpoints/frozen_priors/model_900.pt"]),
                teacher_student_checkpoint=np.asarray([str(args.checkpoint.expanduser().resolve())]),
                teacher_lower_alpha=np.asarray([1.0], dtype=np.float32),
                teacher_upper_alpha=np.asarray([0.9], dtype=np.float32),
                action_lowpass_alpha=np.asarray([lowpass_alpha], dtype=np.float32),
                action_gain=np.asarray([action_gain], dtype=np.float32),
                teacher_label_status=np.asarray(["physx_composed_joint_target"]),
            )
            print(f"[write] {shard_path} shape={obs_arr.shape}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()

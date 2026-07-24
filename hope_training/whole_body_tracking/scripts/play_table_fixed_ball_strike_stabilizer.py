"""Free-base fixed-ball strike replay with the Stage-A leg stabilizer.

This is a diagnostic bridge between the tracking task and the physical table
scene.  The motion command keeps the upper-body/waist reference, while the
loaded 14-DOF Stage-A actor supplies the bounded leg residual.  The table and
ball are real rigid bodies; the ball is launched toward the actually replayed
racket state after a deterministic calibration pass.

It is deliberately not a training task.  The first use is to answer whether
the existing leg checkpoint can coexist with a physical ball in the floating
base table scene.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy as np
import torch
from isaaclab.app import AppLauncher


_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_MANIFEST = _ROOT / "sample_motions/p2_strike_stabilizer_library_k17_v1/tracking_motion_manifest.json"
_DEFAULT_CHECKPOINT = _ROOT / (
    "logs/rsl_rl/agibot_a3_strike_stabilizer_a_unified_k8/"
    "2026-07-22_23-12-10_k17_96env_from_model2897_500it/model_3050.pt"
)
_DEFAULT_EPISODE = "T001_003_gao01_2p92_4p92"

parser = argparse.ArgumentParser(description="Free-base table-tennis strike with a Stage-A leg policy.")
parser.add_argument("--manifest", type=pathlib.Path, default=_DEFAULT_MANIFEST)
parser.add_argument("--episode-id", type=str, default=_DEFAULT_EPISODE)
parser.add_argument("--checkpoint", type=pathlib.Path, default=_DEFAULT_CHECKPOINT)
parser.add_argument("--flight-time", type=float, default=0.04)
parser.add_argument("--incoming-vx", type=float, default=2.0)
parser.add_argument("--incoming-vy", type=float, default=0.0)
parser.add_argument("--incoming-vz", type=float, default=0.30)
parser.add_argument(
    "--target-mode",
    choices=("manifest", "actual"),
    default="manifest",
    help="Ball target: original manifest strike state (default) or actual replayed racket state (diagnostic only).",
)
parser.add_argument(
    "--no-prelude",
    action="store_true",
    help="Disable the training ready-to-motion prelude (diagnostic only).",
)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--hold-seconds", type=float, default=5.0)
parser.add_argument(
    "--root-offset-x",
    type=float,
    default=0.0,
    help="Translate the replayed robot root along world +X (positive moves away from the table at the P2 end).",
)
parser.add_argument("--no-ball", action="store_true", help="Run the stabilizer without launching the ball.")
parser.add_argument(
    "--zero-residual",
    action="store_true",
    help="Disable the learned leg actor and replay the upper-body reference with zero Base residual.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_entry():
    data = json.loads(args_cli.manifest.expanduser().read_text())
    for entry in data.get("motions", []):
        if entry.get("episode_id") == args_cli.episode_id:
            return entry
    raise KeyError(f"episode_id not found: {args_cli.episode_id}")


def _quat_wxyz(npz):
    q = np.asarray(npz["body_quat_w"])[0, 0].astype(np.float32)
    q /= max(float(np.linalg.norm(q)), 1.0e-8)
    return tuple(float(v) for v in q)


def _ball_launch_state(target_pos, target_vel, flight_time):
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    t = float(flight_time)
    return target_pos - target_vel * t + 0.5 * gravity * t * t, target_vel - gravity * t


def _build_cfg(entry, motion, device):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from training.robots.agibot_a3 import AGIBOT_A3_CFG, A3_FEET_BODIES
    from training.tasks.table_tennis import geometry
    from training.tasks.table_tennis.table_tennis_env_cfg import (
        TableTennisSceneCfg,
        build_floor_cfg,
        build_table_top_cfg,
        build_net_cfg,
        build_net_post_cfg,
        build_center_line_cfg,
        build_ball_cfg,
        _MATS,
    )
    from training.tasks.tracking.config.agibot_a3.native_strike_env_cfg import (
        A3StrikeStabilizerAUnifiedEnvCfg,
    )

    @configclass
    class TableStrikeSceneCfg(TableTennisSceneCfg):
        robot: ArticulationCfg = copy.deepcopy(AGIBOT_A3_CFG)
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            force_threshold=10.0,
            debug_vis=False,
        )

    cfg = A3StrikeStabilizerAUnifiedEnvCfg()
    cfg.scene = TableStrikeSceneCfg(num_envs=1, env_spacing=6.0)
    cfg.sim.device = str(device)
    cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.scene.robot.spawn.fix_base = False

    # The tracking task normally runs on a z=0 plane and therefore applies a
    # +0.76 m loader offset.  The table scene floor is at -0.81 m, so retain
    # the original HOPE-world motion coordinates instead.
    cfg.commands.motion.motion_manifest = str(args_cli.manifest.expanduser().resolve())
    cfg.commands.motion.motion_file = None
    cfg.commands.motion.manifest_subset_size = None
    cfg.commands.motion.manifest_frame_z_offset = 0.0
    # Keep the same ready-to-motion prelude as the trained environment unless
    # explicitly disabled for a diagnostic. The physical hit time is shifted
    # by this duration in main().
    if args_cli.no_prelude:
        cfg.commands.motion.prelude_steps = 0
    cfg.commands.motion.debug_vis = False
    cfg.commands.racket_target.debug_vis = False
    # The fixed-ball bridge does not need reference-perturbation sampling.
    # Manifest mode also keeps the command compatible with the manifest-aware
    # MotionLibraryLoader (reference_perturbed is legacy single-NPZ logic).
    cfg.commands.racket_target.target_mode = "manifest"
    # When the whole robot is translated away from the table, keep the strike
    # target coupled to the translated base instead of leaving it at the old
    # world-frame position.  Otherwise the policy receives an artificial
    # 20-cm body-frame target jump that was not present during training.
    cfg.commands.racket_target.manifest_base_aligned = bool(abs(float(args_cli.root_offset_x)) > 1.0e-8)
    cfg.events.physics_material = None
    cfg.events.randomize_link_mass = None
    cfg.events.randomize_pd_gains = None
    init_pos = np.asarray(motion["body_pos_w"])[0, 0].astype(np.float32).copy()
    init_pos[0] += float(args_cli.root_offset_x)
    cfg.scene.robot.init_state.pos = tuple(float(v) for v in init_pos)
    cfg.scene.robot.init_state.rot = _quat_wxyz(motion)
    cfg.episode_length_s = max(5.0, float(entry.get("hit_event", {}).get("hit_time_from_start_s", 0.6)) + 3.0)
    cfg.terminations.base_height = None
    cfg.terminations.non_foot_ground_contact = None
    # The bridge script owns the ball launch and uses the table as a physical
    # object.  No table-task reward/termination is needed for this replay.
    return cfg


def _sync_motion(env, motion_cmd, motion_id, device):
    import torch

    with torch.inference_mode():
        motion_cmd.motion_ids.fill_(int(motion_id))
        motion_cmd.time_steps.zero_()
        motion_cmd.tail_steps.zero_()
        motion_cmd.prelude_elapsed_steps.zero_()
        motion_cmd._prev_motion_steps = motion_cmd.time_steps.clone()
        joint_pos = motion_cmd.joint_pos.clone()
        joint_vel = motion_cmd.joint_vel.clone()
        root_pos, root_ori, root_lin_vel, root_ang_vel = motion_cmd._motion_root_state_w()
        root_pos = root_pos.clone()
        root_pos[:, 0] += float(args_cli.root_offset_x)
        ids = torch.arange(env.num_envs, device=device)
        motion_cmd.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        motion_cmd.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=ids
        )
        motion_cmd._update_command()
        env.scene.write_data_to_sim()
        env.sim.forward()


def _obs_to_device(obs, device):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs.to(device)


def _policy_action(policy, obs, device, action_dim):
    if args_cli.zero_residual:
        return torch.zeros((obs.shape[0], action_dim), device=device)
    return policy(obs)


def main():
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from training.tasks.tracking.config.agibot_a3.native_strike_env_cfg import A3StrikeStabilizerAUnifiedEnvCfg
    from training.utils.ppo_cfg import load_ppo_params, runner_kwargs
    from training.tasks.table_tennis.mdp.racket import racket_state_w
    from isaaclab.managers import SceneEntityCfg

    entry = _load_entry()
    motion_path = pathlib.Path(entry["motion_npz"]).expanduser()
    motion = np.load(motion_path)
    hit = entry["strike_target"]
    cfg = _build_cfg(entry, motion, args_cli.device)
    env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    device = unwrapped.device
    source_hit_time = float(entry.get("hit_event", {}).get("hit_time_from_start_s", 0.6))
    prelude_time = float(cfg.commands.motion.prelude_steps) * float(unwrapped.step_dt)
    hit_time = source_hit_time + prelude_time
    flight_time = min(float(args_cli.flight_time), hit_time - 0.02)
    motion_cmd = unwrapped.command_manager.get_term("motion")
    motion_ids = [str(x) for x in motion_cmd.motion.episode_ids]
    if args_cli.episode_id not in motion_ids:
        raise RuntimeError(f"episode {args_cli.episode_id} not present in loaded manifest")
    motion_id = motion_ids.index(args_cli.episode_id)
    robot = unwrapped.scene["robot"]
    ball = unwrapped.scene["ball"]
    action_term = unwrapped.action_manager.get_term("joint_pos")
    # The current checkpoint was trained with the compact 256/128/64 actor,
    # not the repository-wide 512/256/128 default.  Read the saved run's
    # agent contract so inference cannot silently build a different network.
    ppo_params = load_ppo_params()
    ppo_params["policy"]["actor_hidden_dims"] = [256, 128, 64]
    ppo_params["policy"]["critic_hidden_dims"] = [256, 128, 64]
    ppo_params["policy"]["init_noise_std"] = 0.08
    ppo_params["algorithm"]["num_learning_epochs"] = 3
    ppo_params["algorithm"]["num_mini_batches"] = 2
    ppo_params["algorithm"]["learning_rate"] = 0.0003
    ppo_params["algorithm"]["entropy_coef"] = 0.002
    policy_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(ppo_params, "agibot_a3_strike_stabilizer_a_unified_k8"))
    policy_cfg.device = str(args_cli.device)
    if not args_cli.checkpoint.expanduser().exists():
        raise FileNotFoundError(args_cli.checkpoint)
    runner = OnPolicyRunner(env, policy_cfg.to_dict(), log_dir=None, device=policy_cfg.device)
    runner.load(str(args_cli.checkpoint.expanduser()), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    print(f"[stabilizer] checkpoint={args_cli.checkpoint}")
    print(f"[stabilizer] episode={args_cli.episode_id} motion_id={motion_id} ball={'off' if args_cli.no_ball else 'on'}")
    initial_obs = _obs_to_device(env.get_observations(), device)
    print(f"[stabilizer] obs_shape={tuple(initial_obs.shape)} action_shape={env.action_space.shape}")

    def reset_and_calibrate():
        with torch.inference_mode():
            env.reset()
            _sync_motion(unwrapped, motion_cmd, motion_id, device)
            obs = _obs_to_device(env.get_observations(), device)
            racket_pos = None
            racket_vel = None
            hit_step = int(round(hit_time / float(unwrapped.step_dt)))
            for step in range(hit_step + 1):
                if step == hit_step:
                    racket_pos, racket_vel, _ = racket_state_w(unwrapped, SceneEntityCfg("robot"))
                    racket_pos = racket_pos[0].detach().cpu().numpy().astype(np.float32)
                    racket_vel = racket_vel[0].detach().cpu().numpy().astype(np.float32)
                actions = _policy_action(policy, obs, device, unwrapped.action_manager.total_action_dim)
                obs, _, _, _ = env.step(actions.to(device))
                obs = _obs_to_device(obs, device)
            return racket_pos, racket_vel

    racket_pos, racket_vel = reset_and_calibrate()
    manifest_racket_pos = np.asarray(hit["racket_position_m"], dtype=np.float32)
    manifest_ball_pos = np.asarray(hit["ball_position_m"], dtype=np.float32)
    actual_racket_delta = racket_pos - manifest_racket_pos
    if args_cli.target_mode == "manifest":
        # Use the source strike state.  Do not silently move the ball down to a
        # fallen policy's racket: that would turn a pre-hit fall into a fake
        # successful contact at table height.
        hit_pos = manifest_ball_pos.copy()
        hit_vel = np.asarray(hit["ball_in_velocity_mps"], dtype=np.float32)
    else:
        rel = manifest_ball_pos - manifest_racket_pos
        hit_pos = racket_pos + rel
        hit_vel = np.array([args_cli.incoming_vx, args_cli.incoming_vy, args_cli.incoming_vz], dtype=np.float32)
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    launch_pos = hit_pos - hit_vel * flight_time + 0.5 * gravity * flight_time * flight_time
    launch_vel = hit_vel - gravity * flight_time
    root_at_hit = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
    strike_ids = torch.as_tensor(action_term._strike_joint_ids, device=device, dtype=torch.long)
    if not hasattr(main, "_strike_names_printed"):
        print(f"[stabilizer] strike_joint_ids={action_term._strike_joint_ids} names={[robot.joint_names[i] for i in action_term._strike_joint_ids]}")
        main._strike_names_printed = True
    actual_strike_q = robot.data.joint_pos[0, strike_ids].detach().cpu().numpy().astype(np.float32)
    target_strike_q = motion_cmd.joint_pos[0, strike_ids].detach().cpu().numpy().astype(np.float32)
    strike_q_error = actual_strike_q - target_strike_q
    print(f"[stabilizer] source_hit_time={source_hit_time:.3f}s prelude_time={prelude_time:.3f}s effective_hit_time={hit_time:.3f}s")
    print(f"[stabilizer] target_mode={args_cli.target_mode} manifest_racket_pos={manifest_racket_pos.tolist()}")
    print(f"[stabilizer] actual_racket_pos={racket_pos.tolist()} delta_to_manifest={actual_racket_delta.tolist()} vel={racket_vel.tolist()}")
    print(f"[stabilizer] root_at_hit={root_at_hit.tolist()} racket_z_error={float(actual_racket_delta[2]):.4f}m")
    print(f"[stabilizer] strike_q_actual={actual_strike_q.round(3).tolist()} target={target_strike_q.round(3).tolist()} error_max={float(np.max(np.abs(strike_q_error))):.4f}rad")
    print(f"[stabilizer] hit_pos={hit_pos.tolist()} launch_pos={launch_pos.tolist()} launch_vel={launch_vel.tolist()}")

    # Replay once more so the calibrated ball and the policy see the same
    # deterministic initial state.  The direct state writes are intentionally
    # inside inference mode; IsaacLab data buffers reject inference tensors.
    with torch.inference_mode():
        env.reset()
        _sync_motion(unwrapped, motion_cmd, motion_id, device)
        obs = _obs_to_device(env.get_observations(), device)
        control_dt = float(unwrapped.step_dt)
        launch_step = int(round((hit_time - flight_time) / control_dt))
        hit_step = int(round(hit_time / control_dt))
        launched = False
        contact = False
        returned = False
        min_dist = float("inf")
        incoming_vx = float("nan")
        return_vx = float("nan")
        min_root_z = float("inf")
        final_root_z = float("nan")
        final_root_speed = float("nan")
        final_root_ang_speed = float("nan")
        final_feet_contact = float("nan")
        for step in range(int(args_cli.steps)):
            if not args_cli.no_ball and step == launch_step:
                ball.write_root_pose_to_sim(
                    torch.cat(
                        [
                            torch.tensor(launch_pos, device=device),
                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
                        ]
                    ).view(1, 7),
                )
                ball.write_root_velocity_to_sim(
                    torch.cat(
                        [torch.tensor(launch_vel, device=device), torch.zeros(3, device=device)]
                    ).view(1, 6)
                )
                launched = True
                print(f"[stabilizer] ball launched step={step} t={step * control_dt:.3f}s")
            actions = _policy_action(policy, obs, device, unwrapped.action_manager.total_action_dim)
            obs, _, _, _ = env.step(actions.to(device))
            obs = _obs_to_device(obs, device)
            root_pos = robot.data.root_pos_w[0]
            root_lin = robot.data.root_lin_vel_w[0]
            root_ang = robot.data.root_ang_vel_w[0]
            min_root_z = min(min_root_z, float(root_pos[2].item()))
            final_root_z = float(root_pos[2].item())
            final_root_speed = float(torch.linalg.norm(root_lin).item())
            final_root_ang_speed = float(torch.linalg.norm(root_ang).item())
            contact_sensor = unwrapped.scene.sensors.get("contact_forces")
            if contact_sensor is not None:
                forces = contact_sensor.data.net_forces_w_history[:, -1]
                foot_ids = [i for i, name in enumerate(robot.body_names) if name in ("left_ankle_roll_Link", "right_ankle_roll_Link")]
                if foot_ids:
                    final_feet_contact = float((torch.linalg.norm(forces[0, foot_ids], dim=-1) > 10.0).float().sum().item())
            racket_pos_t, _, _ = racket_state_w(unwrapped, SceneEntityCfg("robot"))
            ball_pos_t = ball.data.root_pos_w
            distance = float(torch.linalg.norm(ball_pos_t[0] - racket_pos_t[0]).item())
            min_dist = min(min_dist, distance)
            if launched:
                vx = float(ball.data.root_lin_vel_w[0, 0].item())
                incoming_vx = vx if np.isnan(incoming_vx) else incoming_vx
                if distance < 0.10:
                    contact = True
                if step > hit_step and vx < -0.2:
                    returned = True
                    return_vx = vx
            if args_cli.realtime:
                import time
                time.sleep(control_dt)
        print(
            f"[stabilizer] result contact_candidate={contact} returned_toward_minus_x={returned} "
            f"closest_racket_distance_m={min_dist:.4f} incoming_vx={incoming_vx:.3f} return_vx={return_vx:.3f} "
            f"min_root_z={min_root_z:.3f} final_root_z={final_root_z:.3f} "
            f"final_root_speed={final_root_speed:.3f} final_root_ang_speed={final_root_ang_speed:.3f} "
            f"final_feet_contact_count={final_feet_contact:.0f}",
            flush=True,
        )
    if args_cli.hold_seconds > 0 and not args_cli.headless:
        import time
        time.sleep(float(args_cli.hold_seconds))
    unwrapped.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()

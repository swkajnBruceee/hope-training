"""Visualize one native A3 strike against a fixed incoming physical ball.

This is an isolated side-branch smoke test:

* full HOPE table/net/ball scene;
* A3 on the P2 side with the validated native strike joint replay;
* fixed incoming ball state derived from the manifest strike target;
* fixed base by default, so the first result answers only whether the racket
  and the dynamic ball actually make a physical return.

Run inside Isaac Lab:

    source setup_train_env.sh
    hope_isaac_py scripts/play_fixed_ball_native_strike.py

Use ``--no-fix-base`` only after the fixed-base contact test succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
from isaaclab.app import AppLauncher


_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_MANIFEST = _ROOT / "sample_motions/p2_strike_stabilizer_library_k17_v1/tracking_motion_manifest.json"
_DEFAULT_EPISODE = "T001_003_gao01_2p92_4p92"

parser = argparse.ArgumentParser(description="Replay one native strike against a fixed incoming ball.")
parser.add_argument("--manifest", type=pathlib.Path, default=_DEFAULT_MANIFEST)
parser.add_argument("--episode-id", type=str, default=_DEFAULT_EPISODE)
parser.add_argument("--motion-npz", type=pathlib.Path, default=None)
parser.add_argument("--flight-time", type=float, default=0.10, help="Ball flight time to the manifest hit state.")
parser.add_argument("--launch-lead", type=float, default=0.12, help="Seconds before hit to launch the ball.")
parser.add_argument(
    "--target-mode",
    choices=("replay", "manifest"),
    default="replay",
    help="Use the actual replayed racket state (recommended) or the manifest target.",
)
parser.add_argument("--incoming-vx", type=float, default=2.0)
parser.add_argument("--incoming-vy", type=float, default=0.0)
parser.add_argument("--incoming-vz", type=float, default=0.30)
parser.add_argument("--steps", type=int, default=220, help="Control steps to simulate.")
parser.add_argument("--realtime", action="store_true", help="Run the visual replay near real time.")
parser.add_argument("--hold-seconds", type=float, default=4.0, help="Keep the final visual state open.")
parser.add_argument("--no-fix-base", action="store_true", help="Allow the robot base to move/fall.")
parser.add_argument("--disable-aero", action="store_true")
parser.add_argument("--no-draw-trajectory", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_entry() -> dict:
    data = json.loads(args_cli.manifest.expanduser().read_text())
    for entry in data.get("motions", []):
        if entry.get("episode_id") == args_cli.episode_id:
            return entry
    raise KeyError(f"episode_id not found: {args_cli.episode_id}")


def _quat_wxyz_from_npz(npz: dict) -> tuple[float, float, float, float]:
    # Isaac Lab articulation data is wxyz.  The stored body_quat_w is also wxyz.
    q = np.asarray(npz["body_quat_w"])[0, 0].astype(float)
    q = q / max(float(np.linalg.norm(q)), 1.0e-8)
    return tuple(float(v) for v in q)


def _ball_launch_state(target_pos: np.ndarray, target_vel: np.ndarray, flight_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Back-propagate a short post-bounce flight under gravity to the launch point."""
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    t = float(flight_time)
    launch_pos = target_pos - target_vel * t + 0.5 * gravity * t * t
    launch_vel = target_vel - gravity * t
    return launch_pos, launch_vel


def _reset_motion_state(env, robot, q_ref: np.ndarray, joint_vel: np.ndarray, device: torch.device):
    """Reset manager state and align the articulation with the first replay frame."""
    import torch

    with torch.inference_mode():
        env.reset()
        q_init = torch.as_tensor(q_ref[0], device=device, dtype=torch.float32).unsqueeze(0)
        dq_init = torch.as_tensor(joint_vel[0], device=device, dtype=torch.float32).unsqueeze(0)
        robot.write_joint_state_to_sim(q_init, dq_init)
        robot.set_joint_position_target(q_init)


def main() -> None:
    import gymnasium as gym
    import torch

    import training.tasks.table_tennis.config.agibot_a3  # noqa: F401
    from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import AgibotA3TableTennisEnvCfg
    from training.tasks.table_tennis import geometry
    from training.tasks.table_tennis.mdp.racket import racket_state_w
    from isaaclab.managers import SceneEntityCfg

    entry = _load_entry()
    motion_path = args_cli.motion_npz or pathlib.Path(entry["motion_npz"])
    motion_path = motion_path.expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)
    motion = np.load(motion_path)
    q_ref = np.asarray(motion["joint_pos"], dtype=np.float32)
    dq_ref = np.asarray(motion["joint_vel"], dtype=np.float32)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    q0_root = np.asarray(motion["body_pos_w"])[0, 0].astype(np.float32)
    root_quat = _quat_wxyz_from_npz(motion)

    hit = entry["strike_target"]
    manifest_ball_pos = np.asarray(hit["ball_position_m"], dtype=np.float32)
    manifest_ball_vel = np.asarray(hit["ball_in_velocity_mps"], dtype=np.float32)
    hit_time = float(entry.get("hit_event", {}).get("hit_time_from_start_s", 0.6))
    flight_time = min(float(args_cli.flight_time), hit_time - 0.02)
    launch_time = hit_time - flight_time

    cfg = AgibotA3TableTennisEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.scene.robot.spawn.fix_base = not args_cli.no_fix_base
    # The source replay is a P2-side strike: root and racket coordinates are in the
    # same HOPE frame as the table scene.
    cfg.scene.robot.init_state.pos = (float(q0_root[0]), float(q0_root[1]), float(q0_root[2]))
    cfg.scene.robot.init_state.rot = root_quat
    cfg.events.reset_robot.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_robot.params["velocity_range"] = {}
    cfg.episode_length_s = max(3.0, hit_time + 2.0)
    if args_cli.disable_aero:
        cfg.ball_aerodynamics.enabled = False

    # Manual ball launch is used below; keep the reset serve safely in-bounds.
    cfg.events.serve_ball.params["serve_cfg"].pos_x_range = (1.0, 1.0)
    cfg.events.serve_ball.params["serve_cfg"].pos_y_range = (-0.35, -0.35)
    cfg.events.serve_ball.params["serve_cfg"].pos_z_range = (1.5, 1.5)
    cfg.events.serve_ball.params["serve_cfg"].vel_x_range = (0.0, 0.0)
    cfg.events.serve_ball.params["serve_cfg"].vel_y_range = (0.0, 0.0)
    cfg.events.serve_ball.params["serve_cfg"].vel_z_range = (0.0, 0.0)
    cfg.terminations.ball_out_of_bounds = None
    cfg.terminations.robot_fell = None if args_cli.no_fix_base else cfg.terminations.robot_fell

    env = gym.make("HOPE-TableTennis-AgibotA3-v0", cfg=cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    ball = unwrapped.scene["ball"]
    term = unwrapped.action_manager._terms["joint_pos"]
    _reset_motion_state(env, robot, q_ref, dq_ref, device)

    # Calibrate the actual paddle state produced by this table-scene replay.
    # The manifest target can belong to a different native executor contract;
    # using the replay state avoids falsely declaring a physics miss caused by
    # a reference-frame mismatch.
    replay_racket_pos = None
    replay_racket_vel = None
    if args_cli.target_mode == "replay":
        calib_steps = int(round(hit_time / (float(cfg.sim.dt * cfg.decimation)))) + 2
        for calib_step in range(calib_steps):
            t_calib = calib_step * float(cfg.sim.dt * cfg.decimation)
            idx_calib = min(int(round(t_calib * fps)), len(q_ref) - 1)
            q_calib = torch.as_tensor(q_ref[idx_calib], device=device, dtype=torch.float32).unsqueeze(0)
            scale_calib = term._scale if torch.is_tensor(term._scale) else torch.full_like(q_calib, float(term._scale))
            offset_calib = term._offset if torch.is_tensor(term._offset) else torch.full_like(q_calib, float(term._offset))
            action_calib = (q_calib - offset_calib) / torch.clamp(scale_calib, min=1.0e-8)
            with torch.inference_mode():
                env.step(action_calib)
            if calib_step == int(round(hit_time / (float(cfg.sim.dt * cfg.decimation)))):
                replay_racket_pos, replay_racket_vel, _ = racket_state_w(unwrapped, SceneEntityCfg("robot"))
                replay_racket_pos = replay_racket_pos[0].detach().cpu().numpy().astype(np.float32)
                replay_racket_vel = replay_racket_vel[0].detach().cpu().numpy().astype(np.float32)
        _reset_motion_state(env, robot, q_ref, dq_ref, device)

    manifest_racket_pos = np.asarray(hit["racket_position_m"], dtype=np.float32)
    relative_ball_to_racket = manifest_ball_pos - manifest_racket_pos
    if args_cli.target_mode == "replay":
        hit_pos = replay_racket_pos + relative_ball_to_racket
        hit_vel = np.array([args_cli.incoming_vx, args_cli.incoming_vy, args_cli.incoming_vz], dtype=np.float32)
        print(f"[fixed-ball] replay-calibrated racket_pos={replay_racket_pos.tolist()} racket_vel={replay_racket_vel.tolist()}")
    else:
        hit_pos = manifest_ball_pos
        hit_vel = manifest_ball_vel
    launch_pos, launch_vel = _ball_launch_state(hit_pos, hit_vel, flight_time)
    print(f"[fixed-ball] episode={args_cli.episode_id} stroke={entry.get('stroke_type')}")
    print(f"[fixed-ball] motion={motion_path} frames={len(q_ref)} fps={fps:.1f} hit_time={hit_time:.3f}s")
    print(f"[fixed-ball] q order={list(robot.joint_names)}")
    print(f"[fixed-ball] root_init={tuple(float(v) for v in q0_root)} quat_wxyz={root_quat}")
    print(f"[fixed-ball] hit_pos={hit_pos.tolist()} ball_in_vel={hit_vel.tolist()} target_mode={args_cli.target_mode}")
    print(f"[fixed-ball] launch_time={launch_time:.3f}s launch_pos={launch_pos.tolist()} launch_vel={launch_vel.tolist()}")
    q_ref0_t = torch.as_tensor(q_ref[0], device=device, dtype=torch.float32)
    q_default_t = robot.data.default_joint_pos[0].detach()
    print(f"[fixed-ball] q0_vs_default_max_abs_rad={float(torch.max(torch.abs(q_ref0_t - q_default_t)).item()):.6f}")

    zero = torch.zeros(env.action_space.shape, device=device)
    control_dt = float(cfg.sim.dt * cfg.decimation)
    t = 0.0
    launched = False
    contact_seen = False
    return_seen = False
    closest = float("inf")
    incoming_vx = float("nan")
    return_vx = float("nan")
    prev_ball_vel = None

    for step in range(args_cli.steps):
        # Interpolate the stored articulation-order reference at the current control time.
        idx = min(int(round(t * fps)), len(q_ref) - 1)
        q_target = torch.as_tensor(q_ref[idx], device=device, dtype=torch.float32).unsqueeze(0)
        # JointPositionAction uses processed = raw * scale + default_offset.
        scale = term._scale if torch.is_tensor(term._scale) else torch.full_like(q_target, float(term._scale))
        offset = term._offset if torch.is_tensor(term._offset) else torch.full_like(q_target, float(term._offset))
        action = (q_target - offset) / torch.clamp(scale, min=1.0e-8)

        if (not launched) and t >= launch_time:
            n = 1
            # Isaac Lab's manager step runs under inference mode in this
            # standalone replay.  State writes must use the same context when
            # the underlying root-state buffers are inference tensors.
            with torch.inference_mode():
                pos = torch.as_tensor(launch_pos, device=device, dtype=torch.float32).repeat(n, 1)
                pos = torch.cat([pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)], dim=1)
                vel = torch.as_tensor(launch_vel, device=device, dtype=torch.float32).repeat(n, 1)
                vel = torch.cat([vel, torch.zeros((n, 3), device=device)], dim=1)
                ball.write_root_pose_to_sim(pos)
                ball.write_root_velocity_to_sim(vel)
            launched = True
            print(f"[fixed-ball] launched at step={step} t={t:.3f}s")

        with torch.inference_mode():
            env.step(action)

        racket_pos, _, _ = racket_state_w(unwrapped, SceneEntityCfg("robot"))
        ball_pos = ball.data.root_pos_w[0]
        ball_vel = ball.data.root_lin_vel_w[0]
        if step in (int(round(hit_time / control_dt)) - 2, int(round(hit_time / control_dt)), int(round(hit_time / control_dt)) + 2):
            q_actual = robot.data.joint_pos[0].detach()
            print(
                f"[fixed-ball] t={t:.3f}s q_err_max={float(torch.max(torch.abs(q_actual - q_target[0])).item()):.5f} "
                f"racket={racket_pos[0].detach().cpu().numpy().round(4).tolist()} "
                f"ball={ball_pos.detach().cpu().numpy().round(4).tolist()}"
            )
        distance = float(torch.linalg.norm(ball_pos - racket_pos[0]).item())
        closest = min(closest, distance)
        if launched and not math.isnan(incoming_vx):
            pass
        if launched and math.isnan(incoming_vx) and float(ball_vel[0]) > 0.2:
            incoming_vx = float(ball_vel[0])
        if prev_ball_vel is not None and launched:
            # A return from the P2-side racket is toward -X.  Require the ball
            # to have first arrived with +X velocity to avoid false positives.
            if (not contact_seen) and distance < 0.09 and float(prev_ball_vel[0]) > 0.2:
                contact_seen = True
                print(f"[fixed-ball] near-racket contact candidate t={t:.3f}s d={distance:.4f} v={ball_vel.tolist()}")
            if contact_seen and float(ball_vel[0]) < -0.2:
                return_seen = True
                return_vx = float(ball_vel[0])
        prev_ball_vel = ball_vel.detach().clone()
        t += control_dt

        if args_cli.realtime:
            time.sleep(control_dt)

        if return_seen and t > hit_time + 0.8:
            break

    print(
        f"[fixed-ball] result contact_candidate={contact_seen} returned_toward_minus_x={return_seen} "
        f"closest_racket_distance_m={closest:.4f} incoming_vx={incoming_vx:.3f} return_vx={return_vx:.3f}"
    )
    print("[fixed-ball] Note: fixed-base is only the first physics/contact gate; it does not validate whole-body balance.")

    if not getattr(args_cli, "headless", False) and args_cli.hold_seconds > 0.0:
        print(f"[fixed-ball] holding visual window for {args_cli.hold_seconds:.1f}s; close the window to exit.")
        deadline = time.monotonic() + float(args_cli.hold_seconds)
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
            time.sleep(0.01)
    env.close()


if __name__ == "__main__":
    failed = False
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        os._exit(1)
    # Isaac Sim 4.5 can leave renderer workers alive after a standalone
    # manager-based environment closes.  This side-branch is an executable
    # replay, so force a clean process boundary after the result is printed.
    os._exit(0)

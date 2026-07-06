"""Launch and visualize the HOPE table-tennis match scene (physics + visualization, no policy).

Builds the full court (floor, table, net, ball, Agibot A3) in the HOPE frame, serves a ball each reset,
and steps the simulation holding the robot's default standing pose (zero action). Use this to verify the
physics (ball flight, drag, table/net bounce) and the scene layout before training a policy.

Run inside your Isaac Lab GPU environment after ``source setup_train_env.sh`` (which defines
``hope_isaac_py``, the Isaac Python launcher with the working-tree PYTHONPATH):

    # interactive window (default: 1 env, robot free-standing, aerodynamics on)
    hope_isaac_py scripts/play_table_tennis.py

    # several courts at once
    hope_isaac_py scripts/play_table_tennis.py --num_envs 9

    # pin the robot upright (stable view of the ball physics while no balance policy exists)
    hope_isaac_py scripts/play_table_tennis.py --fix_base

    # compare flight with/without aerodynamic drag, or enable Magnus (spin) lift
    hope_isaac_py scripts/play_table_tennis.py --disable_aero
    hope_isaac_py scripts/play_table_tennis.py --magnus 0.1

This uses the standard Isaac Lab ``AppLauncher`` standalone pattern (no Hydra/wandb), so it runs without
a trained checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize the HOPE table-tennis scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel courts to spawn.")
parser.add_argument("--fix_base", action="store_true", help="Pin the robot pelvis (stable visualization).")
parser.add_argument("--disable_aero", action="store_true", help="Disable ball aerodynamic drag.")
parser.add_argument(
    "--magnus", type=float, default=None, help="Magnus (spin lift) coefficient; also adds serve spin."
)
parser.add_argument("--serve-speed", type=float, default=None, help="Serve speed in m/s.")
parser.add_argument(
    "--serve-elevation-deg",
    type=float,
    default=None,
    help="Serve elevation angle in degrees. Positive raises the shot.",
)
parser.add_argument(
    "--serve-side-deg",
    type=float,
    default=None,
    help="Serve horizontal side angle in degrees relative to the default -X direction. Positive aims toward +Y.",
)
parser.add_argument("--serve-spin-x", type=float, default=None, help="Serve spin around HOPE X in rad/s.")
parser.add_argument("--serve-spin-y", type=float, default=None, help="Serve spin around HOPE Y in rad/s.")
parser.add_argument("--serve-spin-z", type=float, default=None, help="Serve spin around HOPE Z in rad/s.")
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps (0 = run until window closed).")
parser.add_argument(
    "--hide-robot",
    action="store_true",
    help="Move the robot far from the court and hide its visuals in the Isaac viewport.",
)
truth_group = parser.add_mutually_exclusive_group()
truth_group.add_argument(
    "--publish-ball-truth",
    dest="publish_ball_truth",
    action="store_true",
    help="Publish the simulated ball center as a ROS 2 PointStamped stream.",
)
truth_group.add_argument(
    "--no-publish-ball-truth",
    dest="publish_ball_truth",
    action="store_false",
    help="Disable the ROS 2 ball truth stream.",
)
parser.set_defaults(publish_ball_truth=True)
parser.add_argument("--ball-truth-topic", type=str, default="/ball/point", help="ROS 2 topic for ball truth.")
parser.add_argument("--ball-truth-frame-id", type=str, default="world", help="Frame id for ball truth.")
parser.add_argument("--ball-truth-env-index", type=int, default=0, help="Which env index to publish.")
parser.add_argument(
    "--ball-truth-udp-host",
    type=str,
    default="127.0.0.1",
    help="Local UDP host for the external C++ ROS 2 truth bridge.",
)
parser.add_argument(
    "--ball-truth-udp-port",
    type=int,
    default=19531,
    help="Local UDP port for the external C++ ROS 2 truth bridge.",
)
trajectory_group = parser.add_mutually_exclusive_group()
trajectory_group.add_argument(
    "--draw-trajectory",
    dest="draw_trajectory",
    action="store_true",
    help="Draw the live predicted ball trajectory in the Isaac viewport.",
)
trajectory_group.add_argument(
    "--no-draw-trajectory",
    dest="draw_trajectory",
    action="store_false",
    help="Disable the Isaac viewport trajectory overlay.",
)
parser.set_defaults(draw_trajectory=True)
parser.add_argument("--trajectory-env-index", type=int, default=0, help="Which env index to draw.")
parser.add_argument("--trajectory-horizon", type=float, default=1.2, help="Prediction horizon in seconds.")
parser.add_argument("--trajectory-draw-period", type=float, default=0.03, help="Overlay refresh period in seconds.")
parser.add_argument("--trajectory-udp-host", type=str, default="127.0.0.1", help="Local UDP host for trajectory overlay.")
parser.add_argument("--trajectory-udp-port", type=int, default=19532, help="Local UDP port for trajectory overlay.")
hit_overlay_group = parser.add_mutually_exclusive_group()
hit_overlay_group.add_argument(
    "--hit-overlay",
    dest="hit_overlay",
    action="store_true",
    help="Draw the externally-received /hit/state (hit point, target, racket velocity/normal) "
         "in the Isaac viewport.",
)
hit_overlay_group.add_argument(
    "--no-hit-overlay",
    dest="hit_overlay",
    action="store_false",
    help="Disable the Isaac viewport hit overlay.",
)
parser.set_defaults(hit_overlay=True)
parser.add_argument("--hit-overlay-udp-host", type=str, default="127.0.0.1",
                    help="Local UDP host for hit overlay (served by hit_state_udp_bridge).")
parser.add_argument("--hit-overlay-udp-port", type=int, default=19533,
                    help="Local UDP port for hit overlay (served by hit_state_udp_bridge).")
candidate_overlay_group = parser.add_mutually_exclusive_group()
candidate_overlay_group.add_argument(
    "--landing-candidate-overlay",
    dest="landing_candidate_overlay",
    action="store_true",
    help="Draw landing decision candidates in the Isaac viewport.",
)
candidate_overlay_group.add_argument(
    "--no-landing-candidate-overlay",
    dest="landing_candidate_overlay",
    action="store_false",
    help="Disable landing candidate overlay.",
)
parser.set_defaults(landing_candidate_overlay=True)
parser.add_argument("--landing-candidate-udp-host", type=str, default="127.0.0.1",
                    help="Local UDP host for landing candidate overlay.")
parser.add_argument("--landing-candidate-udp-port", type=int, default=19534,
                    help="Local UDP port for landing candidate overlay.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.fix_base:
    # Keep the visualization in a low-noise preview mode by default.
    # Users can still override this explicitly from the CLI if they want a heavier renderer.
    if getattr(args_cli, "rendering_mode", None) in (None, "balanced"):
        args_cli.rendering_mode = "performance"
    low_noise_kit_args = (
        "--/rtx/directLighting/sampledLighting/enabled=false "
        "--/rtx/reflections/enabled=false "
        "--/rtx/indirectDiffuse/enabled=false "
        "--/rtx/ambientOcclusion/enabled=false "
        "--/rtx/translucency/enabled=false "
        "--/rtx-transient/dldenoiser/enabled=true "
        "--/rtx/shadows/enabled=false"
    )
    existing_kit_args = getattr(args_cli, "kit_args", None)
    args_cli.kit_args = f"{existing_kit_args} {low_noise_kit_args}".strip() if existing_kit_args else low_noise_kit_args

# Launch Omniverse / Isaac Sim first; all isaaclab.* / task imports must come after this.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch

    # The ``training`` package lives under ``<repo>/training`` and must be
    # imported via its parent.  The standalone trajectory overlay package lives
    # under ``<repo>/show/trajectory`` and is imported from ``<repo>/show``.
    _repo_root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
    ))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    for _sub in ("training", "show"):
        _p = os.path.normpath(os.path.join(_repo_root, _sub))
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import training.tasks.table_tennis.config.agibot_a3  # noqa: F401 -- registers table-tennis Gym tasks
    from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (
        AgibotA3TableTennisEnvCfg,
    )
    from training.tasks.table_tennis.geometry import ServeConfig
    from trajectory import (  # type: ignore[import-not-found]
        IsaacTrajectoryOverlay,
        TrajectoryOverlayConfig,
    )
    from hit import (  # type: ignore[import-not-found]
        IsaacHitOverlay,
        HitOverlayConfig,
    )
    from landing_candidates import (  # type: ignore[import-not-found]
        IsaacLandingCandidateOverlay,
        LandingCandidateOverlayConfig,
    )
    from shared_debug_draw import (  # type: ignore[import-not-found]
        clear_shared_debug_draw,
        flush_shared_debug_draw,
    )

    task_id = "HOPE-TableTennis-AgibotA3-v0"

    def _hide_robot_visuals() -> None:
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_prim.IsValid():
            print("[play_table_tennis] warning: robot prim not found; could not hide robot.")
            return

        UsdGeom.Imageable(robot_prim).MakeInvisible()
        print("[play_table_tennis] robot visuals hidden.")

    env_cfg = AgibotA3TableTennisEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Publish the simulated ball truth from the physics callback so the point
    # stream follows the full sim step rate instead of the lower control rate.
    env_cfg.publish_ball_truth = bool(args_cli.publish_ball_truth)
    env_cfg.ball_truth_topic = args_cli.ball_truth_topic
    env_cfg.ball_truth_frame_id = args_cli.ball_truth_frame_id
    env_cfg.ball_truth_env_index = int(args_cli.ball_truth_env_index)
    env_cfg.ball_truth_udp_host = args_cli.ball_truth_udp_host
    env_cfg.ball_truth_udp_port = int(args_cli.ball_truth_udp_port)

    if env_cfg.publish_ball_truth and env_cfg.scene.num_envs != 1:
        print(
            "[play_table_tennis] warning: publish_ball_truth is enabled with multiple envs; "
            f"only env index {env_cfg.ball_truth_env_index} will be published."
        )
    if args_cli.draw_trajectory and args_cli.num_envs != 1:
        print(
            "[play_table_tennis] warning: draw_trajectory is enabled with multiple envs; "
            f"only env index {args_cli.trajectory_env_index} will be drawn."
        )

    if args_cli.fix_base:
        env_cfg.scene.robot.spawn.fix_base = True
        # Keep the visualization deterministic: start every episode from the same upright standing pose.
        env_cfg.events.reset_robot.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # Match HitFixedBaseTouch: keep speed/height fixed and randomize only the forehand-lane x/y spawn.
        env_cfg.events.serve_ball.params["serve_cfg"] = ServeConfig(
            pos_x_range=(2.00, 2.20),
            pos_y_range=(-0.9625, -0.5625),
            pos_z_range=(0.64, 0.64),
            vel_x_range=(-6.5, -5.5),
            vel_y_range=(-0.5, 0.5),
            vel_z_range=(0.08, 0.08),
        )

    if args_cli.hide_robot:
        x0, y0, z0 = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (20.0, 20.0, z0)
        env_cfg.events.reset_robot.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        print(
            "[play_table_tennis] robot moved to (20, 20) before simulation start; "
            "its collisions are away from the court."
        )

    serve_cfg = env_cfg.events.serve_ball.params["serve_cfg"]
    if args_cli.disable_aero:
        env_cfg.ball_aerodynamics.enabled = False
    if args_cli.magnus is not None:
        env_cfg.ball_aerodynamics.magnus_coefficient = float(args_cli.magnus)
        # Give the served ball some spin so the Magnus term is actually exercised.
        serve_cfg.spin_range = (-150.0, 150.0)

    if any(value is not None for value in (args_cli.serve_speed, args_cli.serve_elevation_deg, args_cli.serve_side_deg)):
        speed = 4.25 if args_cli.serve_speed is None else float(args_cli.serve_speed)
        elevation_deg = 2.0 if args_cli.serve_elevation_deg is None else float(args_cli.serve_elevation_deg)
        side_deg = 0.0 if args_cli.serve_side_deg is None else float(args_cli.serve_side_deg)
        elevation = math.radians(elevation_deg)
        side = math.radians(side_deg)
        horizontal_speed = speed * math.cos(elevation)
        vx = -horizontal_speed * math.cos(side)
        vy = horizontal_speed * math.sin(side)
        vz = speed * math.sin(elevation)
        serve_cfg.vel_x_range = (vx, vx)
        serve_cfg.vel_y_range = (vy, vy)
        serve_cfg.vel_z_range = (vz, vz)

    spin_components = (args_cli.serve_spin_x, args_cli.serve_spin_y, args_cli.serve_spin_z)
    if any(value is not None for value in spin_components):
        spin_x = 0.0 if args_cli.serve_spin_x is None else float(args_cli.serve_spin_x)
        spin_y = 0.0 if args_cli.serve_spin_y is None else float(args_cli.serve_spin_y)
        spin_z = 0.0 if args_cli.serve_spin_z is None else float(args_cli.serve_spin_z)
        serve_cfg.spin_range = (0.0, 0.0)
        serve_cfg.spin_x_range = (spin_x, spin_x)
        serve_cfg.spin_y_range = (spin_y, spin_y)
        serve_cfg.spin_z_range = (spin_z, spin_z)

    env = gym.make(task_id, cfg=env_cfg)
    print(f"[play_table_tennis] launched '{task_id}' with {env.unwrapped.num_envs} env(s).")
    print(f"[play_table_tennis] ball aerodynamics active: {getattr(env.unwrapped, '_aero_active', False)}")
    if any(value is not None for value in (args_cli.serve_speed, args_cli.serve_elevation_deg, args_cli.serve_side_deg)):
        print(
            "[play_table_tennis] serve velocity override (m/s): "
            f"vx={serve_cfg.vel_x_range[0]:.3f}, vy={serve_cfg.vel_y_range[0]:.3f}, vz={serve_cfg.vel_z_range[0]:.3f}"
        )
    if any(value is not None for value in spin_components):
        print(
            "[play_table_tennis] serve spin override (rad/s): "
            f"wx={spin_x:.3f}, wy={spin_y:.3f}, wz={spin_z:.3f}"
        )

    obs, _ = env.reset()
    if args_cli.hide_robot:
        _hide_robot_visuals()

    if args_cli.publish_ball_truth:
        print(
            "[play_table_tennis] ball truth UDP active: "
            f"udp://{args_cli.ball_truth_udp_host}:{args_cli.ball_truth_udp_port} "
            "(physics callback rate)"
        )

    # Zero action in the joint-position-with-default-offset space = hold the standing pose.
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    trajectory_overlay = None
    if args_cli.draw_trajectory:
        trajectory_overlay = IsaacTrajectoryOverlay(
            TrajectoryOverlayConfig(
                enabled=True,
                env_index=int(args_cli.trajectory_env_index),
                draw_period_s=float(args_cli.trajectory_draw_period),
                horizon_s=float(args_cli.trajectory_horizon),
                udp_host=args_cli.trajectory_udp_host,
                udp_port=int(args_cli.trajectory_udp_port),
            )
        )
        print(f"[play_table_tennis] trajectory overlay active: {trajectory_overlay.available}")

    hit_overlay = None
    if args_cli.hit_overlay:
        hit_overlay = IsaacHitOverlay(
            HitOverlayConfig(
                enabled=True,
                udp_host=args_cli.hit_overlay_udp_host,
                udp_port=int(args_cli.hit_overlay_udp_port),
            )
        )
        print(f"[play_table_tennis] hit overlay active: {hit_overlay.available}")

    landing_candidate_overlay = None
    if args_cli.landing_candidate_overlay:
        landing_candidate_overlay = IsaacLandingCandidateOverlay(
            LandingCandidateOverlayConfig(
                udp_host=args_cli.landing_candidate_udp_host,
                udp_port=int(args_cli.landing_candidate_udp_port),
            )
        )
        print(f"[play_table_tennis] landing candidate overlay active: {landing_candidate_overlay.available}")

    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    step = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(zero_action)
                ball = env.unwrapped.scene["ball"]
                sim_t = step * control_dt
                if trajectory_overlay is not None and trajectory_overlay.available:
                    env_index = int(args_cli.trajectory_env_index)
                    if 0 <= env_index < env.unwrapped.num_envs:
                        pos_w = ball.data.root_pos_w[env_index].detach().cpu().numpy()
                        origin = env.unwrapped.scene.env_origins[env_index].detach().cpu().numpy()
                        trajectory_overlay.push(step * control_dt, pos_w - origin)
                if hit_overlay is not None and hit_overlay.available:
                    hit_overlay.push(step * control_dt)
                if landing_candidate_overlay is not None and landing_candidate_overlay.available:
                    landing_candidate_overlay.push(step * control_dt)
                if (
                    (trajectory_overlay is not None and trajectory_overlay.available)
                    or (hit_overlay is not None and hit_overlay.available)
                    or (landing_candidate_overlay is not None and landing_candidate_overlay.available)
                ):
                    flush_shared_debug_draw()
            step += 1
            if args_cli.steps and step >= args_cli.steps:
                break
    finally:
        if trajectory_overlay is not None:
            trajectory_overlay.close()
        if hit_overlay is not None:
            hit_overlay.close()
        if landing_candidate_overlay is not None:
            landing_candidate_overlay.close()
        clear_shared_debug_draw()
        env.close()


if __name__ == "__main__":
    failed = False
    try:
        main()
    except Exception:
        import traceback

        print("\n[play_table_tennis.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)

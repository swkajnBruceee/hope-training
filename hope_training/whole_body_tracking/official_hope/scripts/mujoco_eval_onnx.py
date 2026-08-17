# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo sim-to-sim evaluation of the exported HOPE ONNX policy.

This drives the exported ``policy.onnx`` (single layout: observation[1, 110]
``hitter_pure`` -> raw_action[1, 31], no observation normalization) through the
SAME 110-D observation builder, ActionAdapter, swing lifecycle and RacketCommand
that the project reference deploy runner uses, so this is a faithful test of
the deploy contract rather than a second, divergent implementation.

``success_rate`` is measured from an ACTUAL simulated ball. A real ball (free joint,
sphere) is served from the opponent side toward the robot; it flies under gravity +
no-spin aerodynamic drag and really bounces off the racket, table and net inside
MuJoCo. A return succeeds when ALL of the following are observed on the real
trajectory:

  * **contact**   -- an actual MuJoCo contact between the ball geom and the racket
                     collision geom (or the ball passes within the contact radius of
                     the racket site while the racket is moving toward it);
  * **net clear** -- after contact the ball's x crosses the net plane with its centre
                     above the net top; and
  * **opponent-half first bounce** -- the ball's first downward crossing of the table
                     surface (z reaches ball_radius descending) lands on the opponent
                     half, inside the table bounds.

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

There is NO analytic predicted-landing substitute: every event above is read off the
real simulated ball. Forehand, backhand and all serves merge into the one number.
Emits only ``{"success_rate": <float>}`` to stdout (and optionally --json-out); a
one-line human summary goes to stderr.

Two evaluation modes (``--eval-mode``):

  * ``continuous`` (default) — deploy-faithful continuous rally: the robot, policy
    ``last_action``, lifecycle and fixed station are initialized ONCE; between serves
    the policy keeps running through its follow-through/recovery with no state reset
    and no teleport, exactly like the deploy runner. The serve side follows the
    pattern FH, FH, BH, BH, ... so all four adjacent transitions (FH->FH, FH->BH,
    BH->BH, BH->FH) are exercised; the transitions actually seen are reported on
    stderr. A fall keeps affecting later serves — that is the honest continuous
    measurement.
  * ``independent`` — independent-strike evaluation: every serve resets the robot to
    its stand with a fresh lifecycle/last_action/station. This measures isolated
    swings only; it does NOT validate between-swing transitions.

Requires ``mujoco``, ``onnxruntime``, ``pyyaml`` and ``numpy`` plus the shipped
reference deploy package (``a3_deploy/a3_deploy_example``) and the
``a3_pingpong`` MJCF. Runs OUTSIDE Isaac.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "a3_deploy").is_dir() and (parent / "hope_training").is_dir():
            return parent
        # The project training package is self-contained:
        # <workspace>/hope_training/whole_body_tracking/official_hope.
        # Return that directory itself so all defaults resolve to its
        # project-local mujoco_reference bundle, rather than the old
        # <workspace>/a3_deploy/a3_deploy_example layout.
        if (parent / "official_hope" / "source" / "whole_body_tracking" /
                "whole_body_tracking" / "utils" / "success_metric.py").is_file():
            return parent / "official_hope"
    return here.parents[3]


def _reference_pkg_dir(repo_root: pathlib.Path) -> pathlib.Path:
    candidates = (
        repo_root / "mujoco_reference" / "reference",
        repo_root / "a3_deploy" / "a3_deploy_example" / "reference",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _default_runtime_config(repo_root: pathlib.Path) -> pathlib.Path:
    candidates = (
        repo_root / "mujoco_reference" / "config" / "hope_pingpong_runtime.yaml",
        repo_root / "a3_deploy" / "a3_deploy_example" / "config" / "hope_pingpong_runtime.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_success_metric(repo_root: pathlib.Path):
    """Import the shared ``success_metric`` module by file path (pure NumPy).

    Importing it as ``whole_body_tracking.utils.success_metric`` would execute the
    training package ``__init__`` (which registers Isaac Lab gym tasks and needs
    gymnasium); this eval only needs the metric's SuccessRate accumulator, the
    TableGeometry.on_opponent_half definition, and the physics-config loader, so we
    load the single file standalone.
    """
    import importlib.util

    candidates = (
        repo_root / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
        / "success_metric.py",
        repo_root / "hope_training" / "whole_body_tracking" / "official_hope" / "source"
        / "whole_body_tracking" / "whole_body_tracking" / "utils" / "success_metric.py",
        repo_root / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
        / "whole_body_tracking" / "utils" / "success_metric.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("cannot locate the project success_metric.py")
    spec = importlib.util.spec_from_file_location("hope_success_metric", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (sys.modules[__module__]) works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------
# Example serve distribution (planner-less). These are NEUTRAL example incoming balls that arrive in
# front of the robot with a return toward the opponent half; they are NOT the training sampling boxes.
# Replace them (or feed your own planner's RacketCommand stream) for a deployment-faithful evaluation.
# All positions/velocities below are in the MuJoCo world (robot) frame, metres / m·s^-1.
# ---------------------------------------------------------------------------------------------------
def _side_box(args, side, attr: str):
    """Return the model metadata box for the current swing side, if present."""
    boxes = getattr(args, attr, None)
    if boxes is None or len(boxes) < 2:
        return None
    return np.asarray(boxes[0 if side >= 0 else 1], dtype=np.float64)


def _propagate_ball(pos, vel, duration, physics, dt=0.001):
    """Integrate the same no-spin flight model used by the evaluator."""
    p = np.asarray(pos, dtype=np.float64).copy()
    v = np.asarray(vel, dtype=np.float64).copy()
    elapsed = 0.0
    while elapsed < float(duration) - 1.0e-12:
        h = min(float(dt), float(duration) - elapsed)
        a = physics.acceleration(v)
        p = p + v * h + 0.5 * a * h * h
        v = v + a * h
        elapsed += h
    return p


def _solve_serve_velocity(origin, target, flight_t, physics):
    """Aim a serve at ``target`` while accounting for quadratic drag.

    The old evaluator used a no-drag ballistic velocity and hoped the online
    predictor would compensate.  That routinely moved the true crossing point
    outside the narrow model_21800 hitter_pure boxes.  A few shooting iterations
    keep the physical incoming ball and the command target in the same frame.
    """
    origin = np.asarray(origin, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    accel = np.array([0.0, 0.0, -physics.gravity], dtype=np.float64)
    velocity = (target - origin) / float(flight_t) - 0.5 * accel * float(flight_t)
    for _ in range(8):
        predicted = _propagate_ball(origin, velocity, flight_t, physics)
        error = target - predicted
        if float(np.linalg.norm(error)) < 1.0e-4:
            break
        velocity = velocity + error / float(flight_t)
    return velocity


def _sample_serve(rng, side, scene, physics, args):
    """Sample (strike_point, serve_origin, serve_velocity) in the MuJoCo world frame.

    A strike point in the robot's reachable zone is chosen (forehand to the robot's
    right, backhand to its left), a serve origin is placed on the opponent side, and
    the serve velocity is the ballistic (no-drag) solution that would carry the ball
    from the origin to the strike point in ``T`` seconds. Real drag then makes the
    ball arrive a little short/low, but the closed-loop intercept predictor re-reads
    the true ball each tick, so the exact serve is not relied upon.
    """
    strike_x = getattr(args, "strike_x_world", None)
    if strike_x is None:
        strike_x = scene.near_edge_x + float(args.strike_plane_x)
    pos_box = _side_box(args, side, "hitter_pure_pos_boxes")
    if pos_box is not None:
        # Stay inside the actor's trained position envelope, with a small margin
        # for MuJoCo/predictor integration error.
        y = rng.uniform(pos_box[2] + 0.025, pos_box[3] - 0.025)
        z_low = max(pos_box[4] + 0.025, scene.table_height + physics.ball_radius + 0.06)
        z_high = pos_box[5] - 0.025
        z = rng.uniform(z_low, z_high)
    elif side >= 0:  # FOREHAND -> robot's right = -y in the MuJoCo/robot frame
        y = rng.uniform(-0.45, -0.12)
        z = scene.table_height + rng.uniform(0.10, 0.22)
    else:          # BACKHAND -> robot's left = +y
        y = rng.uniform(0.05, 0.38)
        z = scene.table_height + rng.uniform(0.10, 0.22)
    strike_pt = np.array([strike_x, y, z], dtype=np.float64)

    origin = np.array(
        [
            scene.near_edge_x + rng.uniform(1.9, 2.4),
            y + rng.uniform(-0.05, 0.05),
            scene.table_height + rng.uniform(0.20, 0.30),
        ],
        dtype=np.float64,
    )
    flight_t = rng.uniform(0.85, 1.0)
    serve_vel = _solve_serve_velocity(origin, strike_pt, flight_t, physics)
    return strike_pt, origin, serve_vel


def _predict_command(ball_pos, ball_vel, side, scene, physics, args, task_id, revision, RacketCommand):
    """Lightweight inline no-spin intercept predictor -> a RacketCommand.

    Forward-integrates the true ball with the shared no-spin model (gravity +
    quadratic drag) until it crosses the fixed strike plane, producing the racket
    target position (where the ball will be), the time-to-strike, and a target racket
    velocity aimed at the fixed opponent-half landing point. ``swing_sign`` is the
    side locked for this task (harness-internal target semantics only — the policy
    never observes it). This mirrors the planner's job but reads the ball directly
    (the evaluator is the ground truth, so no mocap emulation is needed).
    """
    strike_x = getattr(args, "strike_x_world", None)
    if strike_x is None:
        strike_x = scene.near_edge_x + float(args.strike_plane_x)
    p = np.asarray(ball_pos, dtype=np.float64).copy()
    v = np.asarray(ball_vel, dtype=np.float64).copy()
    dt = 0.002
    t = 0.0
    y_cross, z_cross, tts = float(p[1]), float(p[2]), 0.0
    for _ in range(int(2.0 / dt)):
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        if p[0] >= strike_x > p_new[0]:  # ball moving -x through the strike plane
            dx = p_new[0] - p[0]
            frac = (strike_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
            y_cross = float(p[1] + frac * (p_new[1] - p[1]))
            z_cross = float(p[2] + frac * (p_new[2] - p[2]))
            tts = t + frac * dt
            break
        p, v = p_new, v_new
        t += dt

    target_pos = np.array([strike_x, y_cross, z_cross], dtype=np.float64)

    # Target racket velocity: the ballistic velocity that would send the ball from the
    # strike point to the fixed opponent-half landing target in a fixed time.
    landing_table = np.array(
        [(scene.net_x_table + scene.length) / 2.0, -scene.width / 2.0, 0.0], dtype=np.float64
    )
    landing_mujoco = landing_table + scene.offset
    t_out = 0.6
    accel = np.array([0.0, 0.0, -physics.gravity])
    target_vel = (landing_mujoco - target_pos) / t_out - 0.5 * accel * t_out

    # The evaluator's physical return aim can demand a velocity outside the
    # narrow reference-motion envelope.  Keep the planner command inside the
    # exported actor contract; the physical ball still determines contact and
    # return success below, so this is not an analytic success shortcut.
    vel_box = _side_box(args, side, "hitter_pure_vel_boxes")
    if vel_box is not None:
        target_vel = np.clip(
            target_vel,
            vel_box[[0, 2, 4]] + 0.02,
            vel_box[[1, 3, 5]] - 0.02,
        )

    return RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_sign=side,
        position=target_pos,
        velocity=target_vel,
        time_to_strike=float(tts),
    )


def run_eval(args) -> dict:
    repo_root = _repo_root()

    metric_config_path = repo_root / "configs" / "ball_physics.yaml"
    if not metric_config_path.is_file():
        raise FileNotFoundError(
            "ball-physics config not found: "
            f"{metric_config_path}. The physical evaluator must not fall back to defaults."
        )
    # Make the scoring model explicit and independent of the caller's working
    # directory or a stale environment from the old checkout.
    os.environ["HOPE_BALL_PHYSICS_CONFIG"] = str(metric_config_path)

    # Make the reference deploy package importable (shared 110-D obs / ActionAdapter /
    # lifecycle / RacketCommand / ONNX wrapper).
    ref_dir = pathlib.Path(args.reference_dir) if args.reference_dir else _reference_pkg_dir(repo_root)
    sys.path.insert(0, str(ref_dir))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.joint_order import HEAD_INDICES, JOINT_NAMES
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import build_observation
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import (
        BACKHAND,
        FOREHAND,
        QueueRacketCommandSource,
        RacketCommand,
    )

    # Shared success metric + physics config (pure NumPy; the DEFINITION only -- fed
    # real simulated events here, never an analytic predicted-landing rollout). Loaded
    # directly by file path so importing it does NOT drag in the Isaac Lab training
    # package (its __init__ registers gym tasks that need gymnasium/Isaac).
    metric = _load_success_metric(repo_root)
    resolved_metric_path = metric._find_config_path()
    if pathlib.Path(resolved_metric_path or "").resolve() != metric_config_path.resolve():
        raise RuntimeError(
            "success_metric resolved the wrong physics config: "
            f"{resolved_metric_path!r}; expected {str(metric_config_path)!r}"
        )
    print(f"[mujoco_eval] project_root={repo_root}", file=sys.stderr)
    print(f"[mujoco_eval] physics_config={metric_config_path}", file=sys.stderr)
    BallPhysics = metric.BallPhysics
    SuccessRate = metric.SuccessRate
    TableGeometry = metric.TableGeometry
    load_ball_physics_config = metric.load_ball_physics_config

    # The MuJoCo scene builder lives next to this script.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    runtime_cfg = RuntimeConfig.load(args.runtime_config or _default_runtime_config(repo_root))
    onnx_path = args.onnx or str(runtime_cfg.onnx_path)
    robot_xml = args.model_xml or str(runtime_cfg.model_xml_path)

    ball_cfg = load_ball_physics_config()
    physics = BallPhysics.from_config(ball_cfg)
    table = TableGeometry.from_config(ball_cfg)
    accumulator = SuccessRate()
    contact_count = 0
    net_clear_count = 0
    bounce_count = 0

    policy = OnnxPolicy(onnx_path)
    scene = PingPongRealPhysicsScene(
        robot_xml,
        ball_cfg,
        JOINT_NAMES,
        control_dt=runtime_cfg.control_dt,
        near_edge_x=args.near_edge_x,
        launch_viewer=args.view,
        video_path=args.video,
        video_width=args.video_width,
        video_height=args.video_height,
        initial_stance=(
            args.initial_stance
            if args.initial_stance is not None
            else getattr(runtime_cfg, "initial_stance", "standard")
        ),
        mu_ground_contact=args.mu_ground_contact,
    )

    # The published actor owns the authoritative strike plane and FH/BH
    # position/velocity envelopes. Feed those into the planner-less ball
    # generator so the generated command uses the same coordinate contract as
    # the policy, rather than the old generic table-edge defaults.
    args.hitter_pure_pos_boxes = policy.hitter_pure_pos_boxes
    args.hitter_pure_vel_boxes = policy.hitter_pure_vel_boxes
    if args.strike_plane_x is None:
        if policy.fixed_plane_x is not None:
            args.strike_plane_x = float(policy.fixed_plane_x) - float(args.near_edge_x)
        else:
            args.strike_plane_x = 0.0
    args.strike_x_world = float(args.near_edge_x) + float(args.strike_plane_x)
    print(
        f"[mujoco_eval] strike_x_world={args.strike_x_world:.3f} "
        f"model_fixed_plane_x={policy.fixed_plane_x!r}",
        file=sys.stderr,
    )

    default_q = runtime_cfg.action_adapter.default_q.copy()
    kp, kd = runtime_cfg.sim_kp.copy(), runtime_cfg.sim_kd.copy()
    head_idx = list(HEAD_INDICES)
    net_clear_z = table.net_height + physics.ball_radius
    contact_radius = args.contact_radius
    dt = runtime_cfg.control_dt
    max_ticks = max(1, int(round(args.max_trial_seconds / dt)))
    rng = np.random.default_rng(args.seed)
    continuous = args.eval_mode == "continuous"

    def _policy_tick(lifecycle, source, last_action, base_target_xy):
        """One 50 Hz policy step (identical to the deploy runner's tick).

        Returns (events, applied_action) — the caller keeps ``applied_action`` as
        the next tick's ``last_action``.
        """
        state = scene.read_robot_state()
        previous_phase = lifecycle.phase
        command = source.poll()
        target = lifecycle.update(command, state)
        if lifecycle.phase is not previous_phase:
            print(
                f"[mujoco_eval] phase={previous_phase.value}->{lifecycle.phase.value} "
                f"task={lifecycle.active_task_id} tts={target.time_to_strike:+.3f}",
                file=sys.stderr,
            )
        obs = build_observation(state, target, last_action, default_q, base_target_xy)
        raw_action = policy.infer_target(
            obs,
            target.time_to_strike,
            lifecycle.swing_sign,
            runtime_cfg.control_dt,
        )
        # Applied action = raw with the passive head columns zeroed, matching both
        # the deploy runner and training's zeroed last_action feedback.
        applied_action = np.asarray(raw_action, dtype=np.float64).copy()
        if runtime_cfg.passive_neck:
            applied_action[head_idx] = 0.0
        q_des = runtime_cfg.action_adapter.decode(applied_action)
        if runtime_cfg.passive_neck:
            q_des[head_idx] = default_q[head_idx]
        scene.write_targets(q_des, kp, kd)
        return scene.step(), applied_action

    def _park_ball():
        """Drop the ball out of play (past the far edge) between rally serves."""
        scene.set_ball(
            [scene.near_edge_x + scene.length + 1.0, 0.0, scene.table_height + 0.5],
            [0.0, 0.0, 0.0],
        )

    # Continuous mode: ONE initialization for the whole session — robot state,
    # last_action, lifecycle and the base target (the fixed station of this
    # harness, captured at spawn) all persist across serves (matching the deploy
    # runner). Independent mode re-creates them per serve.
    scene.reset_stand()
    lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
    lifecycle.configure_geometry(
        policy.hitter_pure_pos_boxes,
        policy.hitter_pure_vel_boxes,
        policy.reach_offsets,
        runtime_cfg.lifecycle.velocity_gate_margin,
    )
    source = QueueRacketCommandSource()
    last_action = np.zeros(31, dtype=np.float64)
    base_target_xy = scene.base_pos_w()[:2].copy()
    max_rest_ticks = max(1, int(round(args.max_rest_seconds / dt)))
    transitions_seen: set = set()
    prev_side = None
    task_counter = 0

    for trial in range(args.num_serves):
        # Side pattern FH, FH, BH, BH, ... exercises all four adjacent side
        # transitions (FH->FH, FH->BH, BH->BH, BH->FH) across the session.
        side = FOREHAND if (trial % 4) < 2 else BACKHAND
        if prev_side is not None:
            transitions_seen.add((prev_side, side))
        prev_side = side
        _strike_pt, serve_pos, serve_vel = _sample_serve(rng, side, scene, physics, args)

        if not continuous:
            # Independent-strike evaluation: fresh robot + policy state per serve.
            scene.reset_stand()
            lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
            lifecycle.configure_geometry(
                policy.hitter_pure_pos_boxes,
                policy.hitter_pure_vel_boxes,
                policy.reach_offsets,
                runtime_cfg.lifecycle.velocity_gate_margin,
            )
            source = QueueRacketCommandSource()
            last_action = np.zeros(31, dtype=np.float64)
            base_target_xy = scene.base_pos_w()[:2].copy()
        elif trial > 0:
            # Continuous rally: keep the policy running (no reset, no teleport)
            # through follow-through/recovery until it is ready for the next ball.
            _park_ball()
            for _ in range(max_rest_ticks):
                _events, last_action = _policy_tick(
                    lifecycle, source, last_action, base_target_xy
                )
                if lifecycle.phase.value == "ready":
                    break

        scene.set_ball(serve_pos, serve_vel)
        task_counter += 1
        task_id = task_counter
        revision = 0
        contacted = False
        net_clear = False
        first_bounce = None

        for _tick in range(max_ticks):
            ball_pos, ball_vel = scene.ball_state()
            cmd = _predict_command(
                ball_pos, ball_vel, side, scene, physics, args, task_id, revision, RacketCommand
            )
            revision += 1
            source.submit(cmd)

            events, last_action = _policy_tick(lifecycle, source, last_action, base_target_xy)
            scene.record_frame()

            # --- contact: real MuJoCo ball<->racket contact, or the allowed proximity
            #     fallback (ball within contact_radius of the racket site with the
            #     racket moving toward it). ---
            if not contacted:
                if events.ball_racket_contact:
                    contacted = True
                else:
                    r_pos, r_vel = scene.racket_site_state()
                    b_pos, _b_vel = scene.ball_state()
                    delta = b_pos - r_pos
                    if np.linalg.norm(delta) <= contact_radius and float(np.dot(r_vel, delta)) > 0.0:
                        contacted = True

            # --- after contact, watch the REAL outgoing ball for net clearance and its
            #     first bounce (both read off the simulated trajectory). ---
            if contacted and first_bounce is None:
                if not net_clear:
                    for z_cross, x_sign in events.net_crossings:
                        if x_sign > 0.0 and z_cross > net_clear_z:
                            net_clear = True
                            break
                for x_t, y_t, z_sign in events.surface_crossings:
                    if z_sign < 0.0:
                        first_bounce = (x_t, y_t)
                        break

            if first_bounce is not None:
                break
            # Incoming ball that flew past the robot without ever being contacted is a
            # definite miss (it can no longer produce an opponent-half return): stop the
            # trial early instead of waiting out the timeout.
            if not contacted and scene.ball_state()[0][0] < scene.near_edge_x - 0.8:
                break

        success = bool(
            contacted
            and net_clear
            and first_bounce is not None
            and table.on_opponent_half(first_bounce[0], first_bounce[1])
        )
        contact_count += int(contacted)
        net_clear_count += int(net_clear)
        bounce_count += int(first_bounce is not None)
        accumulator.add_bool(success)

    scene.close()
    if continuous:
        _names = {FOREHAND: "FH", BACKHAND: "BH"}
        seen = sorted(f"{_names[a]}->{_names[b]}" for a, b in transitions_seen)
        all_four = len(transitions_seen) == 4
        print(
            f"[mujoco_eval] adjacent side transitions exercised: {', '.join(seen) or 'none'}"
            + ("" if all_four else "  (increase --num-serves >= 5 to cover all four)"),
            file=sys.stderr,
        )
    print(
        f"[mujoco_eval] mode={args.eval_mode} serves={accumulator.attempts} "
        f"contacts={contact_count} net_clears={net_clear_count} "
        f"bounces={bounce_count} returns={accumulator.successes} "
        f"success_rate={accumulator.value:.4f}",
        file=sys.stderr,
    )
    result = accumulator.as_dict()
    result["mu_ground_contact"] = args.mu_ground_contact
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default=None, help="Exported policy.onnx (default: from runtime config).")
    parser.add_argument("--model-xml", default=None, help="a3_pingpong MJCF (default: from runtime config).")
    parser.add_argument("--runtime-config", default=None, help="hope_pingpong_runtime.yaml (default: shipped).")
    parser.add_argument("--reference-dir", default=None, help="Dir containing the reference deploy package.")
    parser.add_argument("--num-serves", type=int, default=50, help="Number of served balls (denominator).")
    parser.add_argument(
        "--eval-mode", choices=["continuous", "independent"], default="continuous",
        help="continuous (default): one uninterrupted rally session — robot/policy state persist "
             "across serves and all four adjacent side transitions are exercised. "
             "independent: reset the robot per serve (isolated-swing evaluation only).",
    )
    parser.add_argument(
        "--max-rest-seconds", type=float, default=2.0,
        help="continuous mode: max seconds the policy gets between serves to finish "
             "follow-through/recovery before the next ball is served.",
    )
    parser.add_argument(
        "--max-trial-seconds", type=float, default=3.0, help="Max simulated seconds per serve before scoring."
    )
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-site proximity contact fallback (m).")
    parser.add_argument(
        "--near-edge-x", type=float, default=0.30,
        help="MuJoCo x of the table's near edge (sets the robot-to-table placement).",
    )
    parser.add_argument(
        "--strike-plane-x", type=float, default=None,
        help="Override strike-plane offset from the MuJoCo table near edge; by default use the model metadata.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the example serve distribution.")
    parser.add_argument("--view", action="store_true", help="Launch the MuJoCo passive viewer (debug only).")
    parser.add_argument("--video", default=None, help="Record the real-ball MuJoCo replay to an MP4 at 50 fps.")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument(
        "--initial-stance", choices=["standard", "right_front", "left_front", "width50_parallel"],
        default=None,
        help="Initial robot pose; defaults to the runtime config (width50_parallel for the Curriculum-FT bundle).",
    )
    parser.add_argument(
        "--mu-ground-contact", type=float, default=None,
        help="set both A3 foot geoms and the robot floor to one effective sliding friction value",
    )
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    print(json.dumps(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

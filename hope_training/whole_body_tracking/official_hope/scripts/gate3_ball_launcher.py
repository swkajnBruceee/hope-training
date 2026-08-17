#!/usr/bin/env python3
"""Launch monotonic physical Gate3 balls and report latched contacts.

This is an evaluation helper only.  The simulator remains the authority for
ball motion and contact counts; this node only sends Gate3BallCommand messages.
"""

import argparse
from pathlib import Path
import random
import time

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point, Vector3
from mujoco_sim_msgs.msg import Gate3BallCommand, Gate3BallState
from rclpy.node import Node
from std_msgs.msg import Header


_TABLE_LENGTH = 2.74
_TABLE_WIDTH = 1.525
_TABLE_SURFACE_Z = 0.76
_NET_X = 1.37
_NET_HEIGHT = 0.1525
_BALL_RADIUS = 0.020

# The current closed-loop harness uses the fixed backhand station and the
# backhand hit plane.  These are deliberately conservative physical acceptance
# bounds, not planner predictions:
#   * the ball must clear the net by 30 mm beyond the ball-radius requirement;
#   * it must cross the current hit plane in the racket-height corridor;
#   * it must have touched the near-side table before crossing that plane;
#   * the random launch velocity must stay near the official Gate3 V1 profile
#     (vx=-2.8..-3.3 m/s, vz=1.9..2.5 m/s); the broader venue 1--7 m/s
#     acceptance envelope is not a planner-capability envelope;
#   * the flight apex must stay well below the A3 head region.
_HIT_PLANE_X = -0.07
_NET_CLEAR_Z_MIN = _TABLE_SURFACE_Z + _NET_HEIGHT + _BALL_RADIUS + 0.03
# The old ~5 m/s generator reached the hit plane in roughly 0.5--0.7 s.
# At the official-speed profile the same current launch geometry reaches it
# later, around 0.8--1.25 s.  Keep this preflight range wide enough for the
# official-speed distribution; the deployed MuJoCo run remains authoritative.
_HIT_TIME_RANGE = (0.80, 1.25)
_HIT_Y_RANGE = (-0.78, -0.50)
_MIXED_HIT_Y_RANGE = (-1.30, -0.75)
# Moderate lateral locomotion protocol: retain a clear FH/BH station change
# without sending the robot from one table edge to the other.  The two lane
# centers are about 0.90 m apart (the previous draft was about 1.34 m).
_WIDE_LATERAL_HIT_Y_RANGE = (-1.30, -0.30)
_HIT_Z_RANGE = (_TABLE_SURFACE_Z + _BALL_RADIUS - 0.005, 1.15)
_MAX_APEX_Z = 1.65
_VALID_LAUNCH_SPEED_RANGE = (3.3, 4.3)
_VALID_STRIKE_SPEED_RANGE = (1.0, 7.0)
_VALID_VX_RANGE = (-3.3, -2.8)
_VALID_VZ_RANGE = (1.9, 2.5)
_RANDOM_Z_RANGE = (1.30, 1.36)


def _find_pingpong_xml() -> Path:
    """Find the exact MuJoCo XML used by the current AimRT ping-pong binary."""
    here = Path(__file__).resolve()
    roots = [parent for parent in here.parents if (parent / "agibot").is_dir()]
    if not roots:
        raise RuntimeError("could not locate the HOPETableTennis project root")
    root = roots[0]
    sim_root = root / "agibot" / "A3_MuJoCo_Sim" / "aimrt_mujoco_sim"
    candidates = (
        sim_root / "build" / "install" / "bin" / "cfg" / "model" / "a3_pingpong" / "a3_pingpong.xml",
        sim_root / "build_ascii" / "install" / "bin" / "cfg" / "model" / "a3_pingpong" / "a3_pingpong.xml",
        sim_root / "src" / "models" / "bin" / "cfg" / "model" / "a3_pingpong" / "a3_pingpong.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("current AimRT MuJoCo a3_pingpong.xml was not found")


class _MuJoCoServeValidator:
    """Preflight the incoming trajectory with the same table/net XML physics.

    The launcher still lets MuJoCo remain the runtime authority.  This class is
    only a rejection sampler: a candidate is published only if its isolated
    ball trajectory clears the net, reaches the current strike corridor, has a
    near-side table contact, and stays inside the speed/apex envelope.
    """

    def __init__(self) -> None:
        try:
            import mujoco
        except ImportError:  # pragma: no cover - the ROS deployment env has no mujoco wheel
            self._mujoco = None
            self._mode = "analytic_fallback"
            return

        self._mujoco = mujoco
        self._mode = "mujoco_xml"
        self._model = mujoco.MjModel.from_xml_path(str(_find_pingpong_xml()))
        self._data = mujoco.MjData(self._model)
        self._joint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "gate3_ball_free_joint"
        )
        self._body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "gate3_ball"
        )
        self._qpos_adr = int(self._model.jnt_qposadr[self._joint_id])
        self._qvel_adr = int(self._model.jnt_dofadr[self._joint_id])

    @staticmethod
    def _finish_report(
        serve: tuple[float, ...],
        *,
        net_cross_z: float | None,
        hit: dict | None,
        max_z: float,
        table_contact: bool,
        net_contact: bool,
        mode: str,
    ) -> dict:
        position = np.asarray(serve[:3], dtype=np.float64)
        velocity = np.asarray(serve[3:], dtype=np.float64)
        launch_speed = float(np.linalg.norm(velocity))
        hit_position = None if hit is None else hit["position"]
        hit_velocity = None if hit is None else hit["velocity"]
        strike_speed = (
            None if hit_velocity is None else float(np.linalg.norm(hit_velocity))
        )
        checks = {
            "initial_arena": bool(
                _NET_X <= position[0] <= _TABLE_LENGTH
                and -_TABLE_WIDTH <= position[1] <= 0.0
                and position[2] > _TABLE_SURFACE_Z + _BALL_RADIUS
                and velocity[0] < 0.0
            ),
            "launch_speed": _VALID_LAUNCH_SPEED_RANGE[0] <= launch_speed <= _VALID_LAUNCH_SPEED_RANGE[1],
            "launch_vx": _VALID_VX_RANGE[0] <= velocity[0] <= _VALID_VX_RANGE[1],
            "launch_vz": _VALID_VZ_RANGE[0] <= velocity[2] <= _VALID_VZ_RANGE[1],
            "net_crossed": net_cross_z is not None,
            "net_clear": net_cross_z is not None and net_cross_z >= _NET_CLEAR_Z_MIN,
            "table_contact": table_contact,
            "hit_time": hit is not None and _HIT_TIME_RANGE[0] <= hit["time"] <= _HIT_TIME_RANGE[1],
            "hit_y": hit_position is not None and _HIT_Y_RANGE[0] <= hit_position[1] <= _HIT_Y_RANGE[1],
            "hit_z": hit_position is not None and _HIT_Z_RANGE[0] <= hit_position[2] <= _HIT_Z_RANGE[1],
            "strike_speed": strike_speed is not None and _VALID_STRIKE_SPEED_RANGE[0] <= strike_speed <= _VALID_STRIKE_SPEED_RANGE[1],
            "apex": max_z <= _MAX_APEX_Z,
            "no_net_contact": not net_contact,
        }
        return {
            "ok": bool(all(checks.values())),
            "mode": mode,
            "checks": checks,
            "net_cross_z": net_cross_z,
            "hit_time": None if hit is None else hit["time"],
            "hit_position": None if hit_position is None else hit_position.tolist(),
            "strike_speed": strike_speed,
            "launch_speed": launch_speed,
            "max_z": max_z,
        }

    def _validate_analytic(self, serve: tuple[float, ...]) -> dict:
        """Conservative XML-equivalent envelope check for the ROS-only env.

        The deployed AimRT binary remains the runtime authority.  This fallback
        uses its table/net geometry, 1 ms timestep and gravity, and clamps the
        first table contact to the ball-on-table plane with a deliberately low
        normal restitution.  The acceptance margins are wider than the
        resulting numerical uncertainty, so it is used only to reject unsafe
        candidates, never to certify a hit.
        """
        position = np.asarray(serve[:3], dtype=np.float64).copy()
        velocity = np.asarray(serve[3:], dtype=np.float64).copy()
        previous = position.copy()
        max_z = float(position[2])
        net_cross_z = None
        hit = None
        table_contact = False
        net_contact = False
        dt = 0.001
        table_ball_z = _TABLE_SURFACE_Z + _BALL_RADIUS
        net_ball_z = _TABLE_SURFACE_Z + _NET_HEIGHT + _BALL_RADIUS

        for step in range(1800):
            acceleration = np.array((0.0, 0.0, -9.81), dtype=np.float64)
            position = position + velocity * dt + 0.5 * acceleration * dt * dt
            velocity = velocity + acceleration * dt
            max_z = max(max_z, float(position[2]))

            if net_cross_z is None and previous[0] > _NET_X >= position[0]:
                net_cross_z = float(position[2])
                if net_cross_z < net_ball_z:
                    net_contact = True

            if (
                previous[2] > table_ball_z >= position[2]
                and velocity[2] < 0.0
                and 0.0 <= position[0] <= _TABLE_LENGTH
                and -_TABLE_WIDTH <= position[1] <= 0.0
            ):
                table_contact = True
                position[2] = table_ball_z
                # The current XML contact keeps the incoming ball close to the
                # table while it traverses the near-side half.  This low
                # conservative bounce avoids accepting a high post-bounce arc.
                velocity[2] = abs(velocity[2]) * 0.18

            if hit is None and previous[0] >= _HIT_PLANE_X > position[0]:
                hit = {
                    "time": step * dt,
                    "position": position.copy(),
                    "velocity": velocity.copy(),
                }
                break
            previous = position.copy()

        return self._finish_report(
            serve,
            net_cross_z=net_cross_z,
            hit=hit,
            max_z=max_z,
            table_contact=table_contact,
            net_contact=net_contact,
            mode=self._mode,
        )

    def _geom_name(self, geom_id: int) -> str:
        name = self._mujoco.mj_id2name(
            self._model, self._mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
        )
        return name or ""

    def validate(self, serve: tuple[float, ...]) -> dict:
        if self._mujoco is None:
            return self._validate_analytic(serve)

        mujoco = self._mujoco
        position = np.asarray(serve[:3], dtype=np.float64)
        velocity = np.asarray(serve[3:], dtype=np.float64)
        launch_speed = float(np.linalg.norm(velocity))

        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._qpos_adr : self._qpos_adr + 3] = position
        self._data.qpos[self._qpos_adr + 3 : self._qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self._data.qvel[self._qvel_adr : self._qvel_adr + 3] = velocity
        mujoco.mj_forward(self._model, self._data)

        previous = np.asarray(self._data.xpos[self._body_id], dtype=np.float64).copy()
        max_z = float(previous[2])
        net_cross_z = None
        hit = None
        table_contact = False
        net_contact = False

        # The current valid tuples reach the hit plane in <0.7 s.  Keep a
        # generous horizon so a malformed candidate is rejected, never emitted.
        for step in range(1800):
            mujoco.mj_step(self._model, self._data)
            current = np.asarray(self._data.xpos[self._body_id], dtype=np.float64).copy()
            current_velocity = np.asarray(
                self._data.qvel[self._qvel_adr : self._qvel_adr + 3], dtype=np.float64
            ).copy()
            max_z = max(max_z, float(current[2]))

            if net_cross_z is None and previous[0] > _NET_X >= current[0]:
                net_cross_z = float(current[2])
            if hit is None and previous[0] >= _HIT_PLANE_X > current[0]:
                hit = {
                    "time": step * float(self._model.opt.timestep),
                    "position": current.copy(),
                    "velocity": current_velocity.copy(),
                }

            for contact_index in range(self._data.ncon):
                contact = self._data.contact[contact_index]
                names = (
                    self._geom_name(contact.geom1),
                    self._geom_name(contact.geom2),
                )
                if "gate3_table_collision" in names:
                    table_contact = True
                if "gate3_net_collision" in names:
                    net_contact = True

            previous = current
            if hit is not None and current[0] < _HIT_PLANE_X - 0.20:
                break

        return self._finish_report(
            serve,
            net_cross_z=net_cross_z,
            hit=hit,
            max_z=max_z,
            table_contact=table_contact,
            net_contact=net_contact,
            mode=self._mode,
        )


def _validated_random_backhand_serves(count: int, seed: int):
    if int(count) != count or count < 1:
        raise ValueError("random Gate3 serve count must be positive")

    # These are the already successful current-harness backhand ranges.  The
    # official project uses a wider V17-R10 lane generator, but its station and
    # table-surface wire frame differ from this project; blindly copying it
    # would reintroduce out-of-range strikes here.
    rng = random.Random(int(seed))
    validator = _MuJoCoServeValidator()
    serves = []
    diagnostics = []
    attempts = 0
    while len(serves) < int(count) and attempts < int(count) * 500:
        attempts += 1
        candidate = (
            # Keep the launch center inside the official opponent-half table
            # arena (the reference Gate3 contract rejects starts beyond the
            # far edge), while leaving enough lead time for the current hit
            # plane.
            rng.uniform(2.64, 2.72),
            rng.uniform(-0.54, -0.50),
            rng.uniform(*_RANDOM_Z_RANGE),
            rng.uniform(*_VALID_VX_RANGE),
            rng.uniform(-0.05, 0.05),
            rng.uniform(*_VALID_VZ_RANGE),
        )
        report = validator.validate(candidate)
        if report["ok"]:
            serves.append(candidate)
            diagnostics.append(report)

    if len(serves) != int(count):
        raise RuntimeError(
            "could not generate the requested number of physically valid random "
            f"serves: {len(serves)}/{count} accepted after {attempts} candidates"
        )
    return serves, diagnostics


def make_random_safe_backhand_serves(count: int, seed: int):
    """Generate only MuJoCo-prevalidated random serves in the BH envelope."""
    serves, diagnostics = _validated_random_backhand_serves(count, seed)
    # Keep the public helper's original return type while making diagnostics
    # available to the ROS node startup log.
    make_random_safe_backhand_serves.last_diagnostics = diagnostics
    return serves


make_random_safe_backhand_serves.last_diagnostics = []


def _validated_random_mixed_serves(count: int, seed: int):
    """Generate balanced, side-neutral FH/BH serves for this deployment contract.

    This follows the official V17-R10 side-neutral principle, but uses the
    current floor-origin wire and current ~0.55 s hitter_pure timing.
    """
    if int(count) != count or count < 1:
        raise ValueError("mixed random Gate3 serve count must be positive")
    rng = random.Random(int(seed))
    validator = _MuJoCoServeValidator()
    serves = []
    diagnostics = []
    previous_range = _HIT_Y_RANGE
    globals()["_HIT_Y_RANGE"] = _MIXED_HIT_Y_RANGE
    try:
        for _ in range((int(count) + 1) // 2):
            lane_order = ["forehand", "backhand"]
            if rng.random() < 0.5:
                lane_order.reverse()
            for side in lane_order:
                if len(serves) >= int(count):
                    break
                lane = (-1.22, -1.18) if side == "forehand" else (-0.87, -0.83)
                for _ in range(500):
                    candidate = (
                        rng.uniform(2.64, 2.72),
                        rng.uniform(*lane),
                        rng.uniform(*_RANDOM_Z_RANGE),
                        rng.uniform(*_VALID_VX_RANGE),
                        rng.uniform(-0.05, 0.05),
                        rng.uniform(*_VALID_VZ_RANGE),
                    )
                    report = validator.validate(candidate)
                    if report["ok"]:
                        serves.append(candidate)
                        diagnostics.append({**report, "requested_side": side})
                        break
                else:
                    raise RuntimeError(
                        f"could not generate a physically valid {side} mixed serve"
                    )
    finally:
        globals()["_HIT_Y_RANGE"] = previous_range
    return serves, diagnostics


def make_random_mixed_serves(count: int, seed: int):
    """Generate balanced current-contract side-neutral FH/BH serves."""
    serves, diagnostics = _validated_random_mixed_serves(count, seed)
    make_random_mixed_serves.last_diagnostics = diagnostics
    return serves


make_random_mixed_serves.last_diagnostics = []


def _validated_random_wide_lateral_mixed_serves(count: int, seed: int):
    """Generate legal, speed-compliant serves across both lateral extremes.

    This is a locomotion-capability protocol, separate from the fixed-station
    mixed benchmark.  Side labels are only generator lanes; the planner still
    selects the deployed swing side from the observed target.
    """
    if int(count) != count or count < 1:
        raise ValueError("wide lateral mixed serve count must be positive")
    rng = random.Random(int(seed))
    validator = _MuJoCoServeValidator()
    serves = []
    diagnostics = []
    previous_range = _HIT_Y_RANGE
    globals()["_HIT_Y_RANGE"] = _WIDE_LATERAL_HIT_Y_RANGE
    try:
        for _ in range((int(count) + 1) // 2):
            lane_order = ["forehand", "backhand"]
            if rng.random() < 0.5:
                lane_order.reverse()
            for side in lane_order:
                if len(serves) >= int(count):
                    break
                lane = (-1.30, -1.20) if side == "forehand" else (-0.40, -0.30)
                for _ in range(500):
                    candidate = (
                        rng.uniform(2.64, 2.72),
                        rng.uniform(*lane),
                        rng.uniform(*_RANDOM_Z_RANGE),
                        rng.uniform(*_VALID_VX_RANGE),
                        rng.uniform(-0.05, 0.05),
                        rng.uniform(*_VALID_VZ_RANGE),
                    )
                    report = validator.validate(candidate)
                    if report["ok"]:
                        serves.append(candidate)
                        diagnostics.append({**report, "requested_side": side})
                        break
                else:
                    raise RuntimeError(
                        f"could not generate a physically valid wide {side} serve"
                    )
    finally:
        globals()["_HIT_Y_RANGE"] = previous_range
    return serves, diagnostics


def make_random_wide_lateral_mixed_serves(count: int, seed: int):
    """Generate the separate wide-lateral locomotion test sequence."""
    serves, diagnostics = _validated_random_wide_lateral_mixed_serves(count, seed)
    make_random_wide_lateral_mixed_serves.last_diagnostics = diagnostics
    return serves


make_random_wide_lateral_mixed_serves.last_diagnostics = []


class Gate3Launcher(Node):
    def __init__(self, args):
        super().__init__("gate3_ball_launcher")
        self.args = args
        self.serves = args.serves or [
            (args.x0, args.y0, args.z0, args.vx, args.vy, args.vz)
        ]
        self.pub = self.create_publisher(Gate3BallCommand, args.command_topic, 10)
        # Gate3BallState is published at high rate.  A single-threaded
        # executor can keep servicing that subscription and starve the
        # 10-ms launch/park timer after the first ball becomes active.
        self.sub = self.create_subscription(
            Gate3BallState, args.state_topic, self._on_state, 1
        )
        self.timer = self.create_timer(0.01, self._tick)
        self.shot_id = 0
        self.active = False
        self.park_pending = False
        self.next_park_retry = 0.0
        self.park_retry_deadline = 0.0
        self.seen_active_state = False
        self.next_launch = time.monotonic() + 1.0
        self.park_time = 0.0
        self.last_state = None
        self.launch_count = 0
        self._command_wait_logged = False
        # MuJoCo resets the per-ball contact counters when a shot is parked.
        # Keep the previous counter per shot; a single process-global counter
        # would silently miss contacts after the first successful ball.
        self.last_contact_count_by_shot = {}
        self.contacted_shots = set()
        self.get_logger().info("Gate3 launcher ready")

    def _publish(self, active: bool, shot_id: int):
        msg = Gate3BallCommand()
        msg.header = Header()
        msg.shot_id = shot_id
        msg.active = active
        if active:
            x0, y0, z0, vx, vy, vz = self.serves[self.launch_count - 1]
            msg.position = Point(x=x0, y=y0, z=z0)
            msg.linear_velocity = Vector3(
                x=vx, y=vy, z=vz
            )
        self.pub.publish(msg)

    def _tick(self):
        now = time.monotonic()
        # The launch command is intentionally a one-shot message. Do not
        # spend that one shot before the MuJoCo ROS subscriber has discovered
        # the publisher: otherwise the launcher log says "launch" while the
        # plant keeps shot_id=0 and no Ball object ever reaches the mocap
        # bridge. This race is especially visible after restarting AimRT.
        if not self.active and self.count_subscribers(self.args.command_topic) < 1:
            if not self._command_wait_logged:
                self.get_logger().info(
                    "waiting for MuJoCo Gate3 command subscriber before launch"
                )
                self._command_wait_logged = True
            return
        if not self.active and self._command_wait_logged:
            self.get_logger().info("MuJoCo Gate3 command subscriber ready")
            self._command_wait_logged = False
        if self.active:
            # The simulator is authoritative about the active/parked state.
            # Do not advance to the next shot until it acknowledges the park
            # for the current shot.  Repeating the same park command is safe:
            # the simulator treats an already-parked command as a duplicate.
            if self.park_pending:
                if now >= self.next_park_retry:
                    self._publish(False, self.shot_id)
                    self.get_logger().info(
                        "park request shot={} retry".format(self.shot_id)
                    )
                    self.next_park_retry = now + 0.10
                # Do not deadlock the whole sequence if the state publisher
                # drops the inactive acknowledgement.  The park command has
                # already been repeated several times; advance after a short
                # bounded window while preserving the same monotonic shot ID
                # on all park retries.
                if now >= self.park_retry_deadline:
                    self.park_pending = False
                    self.active = False
                    self.next_launch = now + self.args.inter_shot
                    self.get_logger().warning(
                        "park acknowledgement timeout shot={} "
                        "(advancing after retries)".format(self.shot_id)
                    )
                return
            if now < self.park_time:
                return
            # Every shot, including the final one, must emit an inactive
            # state so the evidence recorder can close its shot window.
            self.park_pending = True
            self.next_park_retry = now
            self.park_retry_deadline = now + max(0.8, self.args.contact_hold)
            self._publish(False, self.shot_id)
            self.get_logger().info(
                "park request shot={} initial".format(self.shot_id)
            )
            return
        if self.launch_count >= self.args.shots:
            return
        if now >= self.next_launch:
            self.shot_id += 1
            self._publish(True, self.shot_id)
            self.active = True
            self.park_pending = False
            self.seen_active_state = False
            self.launch_count += 1
            self.park_time = now + self.args.flight_window
            self.get_logger().info(
                "launch shot={} p=({:.2f},{:.2f},{:.2f}) v=({:.2f},{:.2f},{:.2f})".format(
                    self.shot_id,
                    *self.serves[self.launch_count - 1],
                )
            )

    def _on_state(self, msg: Gate3BallState):
        shot_id = int(msg.shot_id)
        if shot_id <= 0:
            return
        previous_count = self.last_contact_count_by_shot.get(shot_id)
        self.last_contact_count_by_shot[shot_id] = int(msg.racket_contact_count)
        self.last_state = msg
        if self.active and shot_id == self.shot_id and bool(msg.active):
            # A park acknowledgement is only valid after the simulator has
            # published the active state for this exact shot.  This prevents
            # a delayed pre-launch inactive state from being mistaken for a
            # successful park.
            self.seen_active_state = True
        if (
            self.active
            and self.park_pending
            and shot_id == self.shot_id
            and self.seen_active_state
            and not bool(msg.active)
        ):
            self.park_pending = False
            self.active = False
            self.next_launch = time.monotonic() + self.args.inter_shot
            self.get_logger().info(
                "park acknowledged shot={} next_launch_in={:.2f}s".format(
                    shot_id, self.args.inter_shot
                )
            )
        if previous_count is None:
            return
        if int(msg.racket_contact_count) > previous_count:
            if self.active:
                self.park_time = min(
                    self.park_time,
                    time.monotonic() + self.args.contact_hold,
                )
            if shot_id not in self.contacted_shots:
                self.contacted_shots.add(shot_id)
                self.get_logger().info(
                    "RACKET CONTACT shot={} count={} bits={} force={:.2f}N".format(
                        shot_id,
                        msg.racket_contact_count,
                        msg.contact_bits,
                        msg.racket_normal_force_n,
                    )
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--command-topic", default="/sim/gate3/ball_command")
    parser.add_argument("--state-topic", default="/sim/gate3/ball_state")
    parser.add_argument("--x0", type=float, default=2.20)
    parser.add_argument("--y0", type=float, default=-0.45)
    parser.add_argument("--z0", type=float, default=1.70)
    parser.add_argument("--vx", type=float, default=-3.00)
    parser.add_argument("--vy", type=float, default=0.20)
    parser.add_argument("--vz", type=float, default=2.50)
    parser.add_argument("--flight-window", type=float, default=1.5)
    parser.add_argument("--inter-shot", type=float, default=0.3)
    parser.add_argument(
        "--contact-hold",
        type=float,
        default=1.5,
        help="seconds to keep a contacted ball active for post-racket net/table evidence",
    )
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--randomize-mixed",
        action="store_true",
        help="balanced side-neutral current-contract FH/BH random serves",
    )
    parser.add_argument(
        "--randomize-wide-lateral-mixed",
        action="store_true",
        help="balanced legal FH/BH serves spanning wide lateral lanes for locomotion testing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--serve",
        action="append",
        default=None,
        metavar="x,y,z,vx,vy,vz",
        help="repeatable physical Gate3 serve tuple; z uses floor-origin metres",
    )
    args = parser.parse_args()
    if args.serve:
        if args.randomize or args.randomize_mixed or args.randomize_wide_lateral_mixed:
            parser.error("--serve and random serve modes cannot be combined")
        args.serves = []
        for item in args.serve:
            values = tuple(float(value) for value in item.split(","))
            if len(values) != 6:
                parser.error("--serve must contain x,y,z,vx,vy,vz")
            args.serves.append(values)
        args.shots = len(args.serves)
    elif args.randomize or args.randomize_mixed or args.randomize_wide_lateral_mixed:
        if sum(bool(value) for value in (args.randomize, args.randomize_mixed, args.randomize_wide_lateral_mixed)) > 1:
            parser.error("random serve modes cannot be combined")
        if args.randomize_wide_lateral_mixed:
            generator = make_random_wide_lateral_mixed_serves
        elif args.randomize_mixed:
            generator = make_random_mixed_serves
        else:
            generator = make_random_safe_backhand_serves
        args.serves = generator(args.shots, args.seed)
        for index, report in enumerate(generator.last_diagnostics, start=1):
            hit = report["hit_position"]
            self_info = (
                f"validated shot={index} mode={report['mode']} "
                f"net_z={report['net_cross_z']:.3f} "
                f"hit_t={report['hit_time']:.3f} hit=({hit[0]:.3f},{hit[1]:.3f},{hit[2]:.3f}) "
                f"speed={report['strike_speed']:.3f} apex={report['max_z']:.3f}"
            )
            print(f"[gate3 preflight] {self_info}", flush=True)
    else:
        args.serves = None
    rclpy.init()
    node = Gate3Launcher(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and node.launch_count < args.shots:
            executor.spin_once(timeout_sec=0.02)
        end = time.monotonic() + args.flight_window + 0.5
        while rclpy.ok() and time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)
    finally:
        executor.shutdown(timeout_sec=1.0)
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

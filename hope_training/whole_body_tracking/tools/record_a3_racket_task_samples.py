#!/usr/bin/env python3
"""Record official A3 MuJoCo racket samples for standalone qualification.

The recorder consumes the simulator's ``/sim/a3/pelvis_pose`` and
``/sim/a3/right_racket_pose`` ``PoseStamped`` topics.  It converts the world
(``odom``) poses into the pelvis base frame and writes the four arrays consumed
by ``a3_standalone_qualification.py``.  It intentionally does *not* create a
target or decide that a replay passes a task gate.

Run this recorder before the RobotIO replay.  The standard standalone runner
emits 150 PD-STAND waist commands before the first replay command, so the
default capture origin is the 151st waist command observed by this process.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RACKET_NORMAL_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def quaternion_to_matrix_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the active local-to-world rotation matrix for a ROS quaternion."""

    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite [x, y, z, w] vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_racket_to_base(
    pelvis_position_w_m: np.ndarray,
    pelvis_quaternion_w_xyzw: np.ndarray,
    racket_position_w_m: np.ndarray,
    racket_quaternion_w_xyzw: np.ndarray,
    racket_velocity_w_mps: np.ndarray,
    racket_normal_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express an official-model racket state in the pelvis base frame.

    This deliberately follows ``training/tasks/table_tennis/mdp/racket.py``:
    velocity is the world linear velocity rotated into base coordinates, not
    the time derivative of base-relative position (which would additionally
    include rotating-frame terms).
    """

    base_position = np.asarray(pelvis_position_w_m, dtype=np.float64)
    racket_position = np.asarray(racket_position_w_m, dtype=np.float64)
    racket_velocity = np.asarray(racket_velocity_w_mps, dtype=np.float64)
    local_normal = np.asarray(racket_normal_local, dtype=np.float64)
    for name, value in (
        ("pelvis_position_w_m", base_position),
        ("racket_position_w_m", racket_position),
        ("racket_velocity_w_mps", racket_velocity),
        ("racket_normal_local", local_normal),
    ):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite 3-vector")
    normal_norm = float(np.linalg.norm(local_normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("racket_normal_local must be non-zero")

    rotation_wb = quaternion_to_matrix_xyzw(pelvis_quaternion_w_xyzw)
    rotation_wr = quaternion_to_matrix_xyzw(racket_quaternion_w_xyzw)
    rotation_bw = rotation_wb.T
    position_b = rotation_bw @ (racket_position - base_position)
    velocity_b = rotation_bw @ racket_velocity
    normal_b = rotation_bw @ (rotation_wr @ (local_normal / normal_norm))
    normal_b /= np.linalg.norm(normal_b)
    return position_b, velocity_b, normal_b


def estimate_world_velocity(position_w_m: np.ndarray, timestamp_s: np.ndarray) -> np.ndarray:
    """Estimate world linear velocity by finite differences on pose samples."""

    position = np.asarray(position_w_m, dtype=np.float64)
    timestamp = np.asarray(timestamp_s, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position_w_m must be [T,3]")
    if timestamp.shape != (len(position),) or len(timestamp) < 3:
        raise ValueError("timestamp_s must be strictly increasing with at least three samples")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(timestamp)) or not np.all(np.diff(timestamp) > 0.0):
        raise ValueError("position and timestamps must be finite; timestamps must be strictly increasing")
    return np.gradient(position, timestamp, axis=0, edge_order=1)


@dataclass(frozen=True)
class PoseRecord:
    receive_monotonic_ns: int
    header_timestamp_s: float
    position_w_m: np.ndarray
    quaternion_w_xyzw: np.ndarray


@dataclass(frozen=True)
class PairedPoseRecord:
    base: PoseRecord
    racket: PoseRecord
    pair_skew_s: float


def _header_timestamp_s(header: Any) -> float:
    stamp = header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    return value if value > 0.0 and math.isfinite(value) else math.nan


def _pose_record(message: Any, receive_monotonic_ns: int) -> PoseRecord:
    pose = message.pose
    return PoseRecord(
        receive_monotonic_ns=receive_monotonic_ns,
        header_timestamp_s=_header_timestamp_s(message.header),
        position_w_m=np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
        quaternion_w_xyzw=np.asarray(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float64
        ),
    )


class RacketTaskSampleRecorder:
    """ROS2 callback collector; imported only after the ROS environment exists."""

    def __init__(self, node: Any, args: argparse.Namespace, pose_type: Any, command_type: Any, command_qos: Any):
        self.node = node
        self.args = args
        self.capture_start_ns: int | None = None
        self.capture_end_ns: int | None = None
        self.command_count = 0
        self.command_receive_ns: list[int] = []
        self.base_records: deque[PoseRecord] = deque(maxlen=512)
        self.pairs: list[PairedPoseRecord] = []
        self.stand_gate_result: dict[str, float | bool | int] | None = None
        self.node.create_subscription(pose_type, args.pelvis_topic, self._on_pelvis, 100)
        self.node.create_subscription(pose_type, args.racket_topic, self._on_racket, 100)
        # The RobotIOBackend command publishers are explicitly BEST_EFFORT.
        # A default RELIABLE rclpy subscription is incompatible with them.
        self.node.create_subscription(command_type, args.command_topic, self._on_command, command_qos)

    @property
    def done(self) -> bool:
        return self.capture_end_ns is not None

    def _on_command(self, _message: Any) -> None:
        receive_ns = time.monotonic_ns()
        self.command_count += 1
        self.command_receive_ns.append(receive_ns)
        if self.command_count == self.args.skip_command_messages and self.args.stand_gate_file:
            self.stand_gate_result = self._evaluate_stand_gate(receive_ns)
            self._write_stand_gate(self.stand_gate_result)
            if not bool(self.stand_gate_result["passed"]):
                self.node.get_logger().error(
                    "PD-STAND gate rejected: "
                    f"min_height={self.stand_gate_result['min_pelvis_height_m']:.3f} m, "
                    f"max_tilt={self.stand_gate_result['max_pelvis_tilt_deg']:.1f} deg"
                )
                return
        if self.capture_start_ns is None and self.command_count > self.args.skip_command_messages:
            if self.stand_gate_result is not None and not bool(self.stand_gate_result["passed"]):
                return
            self.capture_start_ns = receive_ns
            self.node.get_logger().info(
                f"capture origin set at waist command {self.command_count} "
                f"(skipped {self.args.skip_command_messages} PD-STAND commands)"
            )

    def _evaluate_stand_gate(self, command_receive_ns: int) -> dict[str, float | bool | int]:
        window_begin_ns = command_receive_ns - int(self.args.stand_window_s * 1.0e9)
        records = [record for record in self.base_records if record.receive_monotonic_ns >= window_begin_ns]
        if not records:
            return {"passed": False, "sample_count": 0, "min_pelvis_height_m": math.nan, "max_pelvis_tilt_deg": math.inf}
        heights = np.asarray([record.position_w_m[2] for record in records], dtype=np.float64)
        tilts = []
        for record in records:
            up_world = quaternion_to_matrix_xyzw(record.quaternion_w_xyzw)[:, 2]
            tilts.append(float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0)))))
        min_height = float(np.min(heights))
        max_tilt = float(np.max(tilts))
        passed = (
            len(records) >= self.args.min_stand_samples
            and min_height >= self.args.min_pelvis_height_m
            and max_tilt <= self.args.max_pelvis_tilt_deg
        )
        return {
            "passed": bool(passed),
            "sample_count": len(records),
            "min_pelvis_height_m": min_height,
            "max_pelvis_tilt_deg": max_tilt,
        }

    def _write_stand_gate(self, result: dict[str, float | bool | int]) -> None:
        path = self.args.stand_gate_file.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _on_pelvis(self, message: Any) -> None:
        self.base_records.append(_pose_record(message, time.monotonic_ns()))

    def _on_racket(self, message: Any) -> None:
        racket = _pose_record(message, time.monotonic_ns())
        start_ns = self.capture_start_ns
        if start_ns is None or racket.receive_monotonic_ns < start_ns or not self.base_records:
            return
        # Header stamps reflect simulator time and are preferred for matching;
        # receive clocks are the fallback when a publisher leaves the stamp at 0.
        def clock(record: PoseRecord) -> float:
            return record.header_timestamp_s if math.isfinite(record.header_timestamp_s) else record.receive_monotonic_ns * 1.0e-9

        selected = min(self.base_records, key=lambda base: abs(clock(base) - clock(racket)))
        skew_s = abs(clock(selected) - clock(racket))
        if skew_s <= self.args.max_pair_skew_ms * 1.0e-3:
            self.pairs.append(PairedPoseRecord(base=selected, racket=racket, pair_skew_s=skew_s))
        if racket.receive_monotonic_ns - start_ns >= int(self.args.duration_s * 1.0e9):
            self.capture_end_ns = racket.receive_monotonic_ns


def _load_target_metadata(target_spec_path: Path | None) -> dict[str, np.ndarray]:
    if target_spec_path is None:
        return {}
    from a3_strike_contract import verify_target_spec

    payload = json.loads(target_spec_path.read_text(encoding="utf-8"))
    digest = verify_target_spec(payload)
    return {
        "source_target_sha256": np.asarray([digest]),
        "racket_mount_contract_id": np.asarray([payload["racket_mount_contract_id"]]),
    }


def build_task_sample_payload(
    pairs: list[PairedPoseRecord],
    command_receive_ns: list[int],
    capture_start_ns: int,
    racket_normal_local: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    stand_gate_passed: bool = False,
) -> dict[str, np.ndarray]:
    """Build a qualification-compatible NPZ payload from synchronized poses."""

    if len(pairs) < 3:
        raise ValueError(f"need at least three matched pelvis/racket poses, received {len(pairs)}")
    ordered = sorted(pairs, key=lambda pair: pair.racket.receive_monotonic_ns)
    deduped: list[PairedPoseRecord] = []
    for pair in ordered:
        if not deduped or pair.racket.receive_monotonic_ns > deduped[-1].racket.receive_monotonic_ns:
            deduped.append(pair)
    if len(deduped) < 3:
        raise ValueError("need at least three unique racket receive timestamps")

    receive_ns = np.asarray([pair.racket.receive_monotonic_ns for pair in deduped], dtype=np.int64)
    timestamp_s = (receive_ns - int(capture_start_ns)).astype(np.float64) * 1.0e-9
    racket_position_w = np.asarray([pair.racket.position_w_m for pair in deduped], dtype=np.float64)
    racket_velocity_w = estimate_world_velocity(racket_position_w, timestamp_s)
    position_b, velocity_b, normal_b = [], [], []
    for index, pair in enumerate(deduped):
        position, velocity, normal = world_racket_to_base(
            pair.base.position_w_m,
            pair.base.quaternion_w_xyzw,
            pair.racket.position_w_m,
            pair.racket.quaternion_w_xyzw,
            racket_velocity_w[index],
            racket_normal_local,
        )
        position_b.append(position)
        velocity_b.append(velocity)
        normal_b.append(normal)

    command_ns = np.asarray(command_receive_ns, dtype=np.int64)
    command_index = np.searchsorted(command_ns, receive_ns, side="right") - 1
    command_index = np.clip(command_index, 0, max(len(command_ns) - 1, 0))
    command_time_s = (command_ns[command_index] - int(capture_start_ns)).astype(np.float64) * 1.0e-9
    normal_semantics = (
        "right_racket_site_local_plus_y_red_face"
        if np.allclose(racket_normal_local, DEFAULT_RACKET_NORMAL_LOCAL, atol=1.0e-12)
        else "explicit_racket_normal_local_override"
    )
    payload: dict[str, np.ndarray] = {
        "timestamp_s": timestamp_s,
        "racket_position_b_m": np.asarray(position_b, dtype=np.float64),
        "racket_velocity_b_mps": np.asarray(velocity_b, dtype=np.float64),
        "racket_normal_b": np.asarray(normal_b, dtype=np.float64),
        "pelvis_pose_timestamp_s": np.asarray([pair.base.header_timestamp_s for pair in deduped], dtype=np.float64),
        "racket_pose_timestamp_s": np.asarray([pair.racket.header_timestamp_s for pair in deduped], dtype=np.float64),
        "pelvis_pose_receive_time_s": (
            np.asarray([pair.base.receive_monotonic_ns for pair in deduped], dtype=np.int64) - int(capture_start_ns)
        ).astype(np.float64) * 1.0e-9,
        "racket_pose_receive_time_s": timestamp_s.copy(),
        "last_command_receive_time_s": command_time_s,
        "pair_skew_s": np.asarray([pair.pair_skew_s for pair in deduped], dtype=np.float64),
        "racket_normal_local": np.asarray(racket_normal_local, dtype=np.float64),
        "source_frame": np.asarray(["odom"]),
        "base_frame": np.asarray(["pelvis_pose"]),
        "velocity_semantics": np.asarray(["finite_difference_world_linear_velocity_rotated_to_pelvis_base"]),
        "normal_semantics": np.asarray([normal_semantics]),
        "capture_start_monotonic_ns": np.asarray([capture_start_ns], dtype=np.int64),
        "stand_gate_passed": np.asarray([stand_gate_passed], dtype=np.bool_),
    }
    if metadata:
        payload.update(metadata)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=3.0, help="Capture duration after replay command zero (default: 3.0).")
    parser.add_argument("--start-timeout-s", type=float, default=15.0, help="Fail if the replay command zero is not observed in this time.")
    parser.add_argument("--skip-command-messages", type=int, default=150,
                        help="Waist command messages to ignore before replay time zero (default: runner PD-STAND length 150).")
    parser.add_argument("--max-pair-skew-ms", type=float, default=10.0)
    parser.add_argument("--pelvis-topic", default="/sim/a3/pelvis_pose")
    parser.add_argument("--racket-topic", default="/sim/a3/right_racket_pose")
    parser.add_argument("--command-topic", default="/body_drive/waist_joint_command")
    parser.add_argument("--racket-normal-local", nargs=3, type=float, default=DEFAULT_RACKET_NORMAL_LOCAL.tolist(), metavar=("X", "Y", "Z"),
                        help="Right-racket site local face normal; default is documented +Y/red face.")
    parser.add_argument("--target-spec", type=Path,
                        help="Optional immutable target to embed its hash and racket mount ID in the evidence.")
    parser.add_argument("--stand-gate-file", type=Path,
                        help="When set, write PD-STAND gate JSON after the skipped command phase; required by the repeat harness.")
    parser.add_argument("--stand-window-s", type=float, default=0.5)
    parser.add_argument("--min-pelvis-height-m", type=float, default=0.75)
    parser.add_argument("--max-pelvis-tilt-deg", type=float, default=25.0)
    parser.add_argument("--min-stand-samples", type=int, default=20)
    args = parser.parse_args()
    if (args.duration_s <= 0.0 or args.start_timeout_s <= 0.0 or args.max_pair_skew_ms < 0.0
            or args.skip_command_messages < 0 or args.stand_window_s <= 0.0
            or args.min_pelvis_height_m <= 0.0 or args.max_pelvis_tilt_deg < 0.0 or args.min_stand_samples < 1):
        parser.error("invalid duration, stand-gate, skew, or skipped-command value")
    normal = np.asarray(args.racket_normal_local, dtype=np.float64)
    if not np.all(np.isfinite(normal)) or float(np.linalg.norm(normal)) <= 1.0e-12:
        parser.error("--racket-normal-local must be finite and non-zero")
    args.racket_normal_local = normal / np.linalg.norm(normal)
    return args


def main() -> None:
    args = _parse_args()
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from joint_msgs.msg import JointCommand
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise SystemExit(
            "ROS2 Python messages are unavailable. Source tools/setup_a3_mujoco_sim_env.sh "
            "(or the equivalent ROS2 and joint_msgs overlays) before running this recorder."
        ) from exc

    metadata = _load_target_metadata(args.target_spec.expanduser().resolve() if args.target_spec else None)
    rclpy.init()
    node = rclpy.create_node("a3_racket_task_sample_recorder")
    command_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    recorder = RacketTaskSampleRecorder(node, args, PoseStamped, JointCommand, command_qos)
    deadline_ns = time.monotonic_ns() + int(args.start_timeout_s * 1.0e9)
    try:
        node.get_logger().info(
            f"armed; waiting for waist command {args.skip_command_messages + 1} to define replay time zero"
        )
        while rclpy.ok() and not recorder.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if recorder.capture_start_ns is None and time.monotonic_ns() > deadline_ns:
                raise TimeoutError("did not observe the replay command origin before --start-timeout-s")
        if recorder.capture_start_ns is None:
            raise RuntimeError("ROS shutdown before replay command origin was observed")
        payload = build_task_sample_payload(
            recorder.pairs,
            recorder.command_receive_ns,
            recorder.capture_start_ns,
            args.racket_normal_local,
            metadata,
            stand_gate_passed=bool(recorder.stand_gate_result and recorder.stand_gate_result["passed"]),
        )
        if recorder.stand_gate_result:
            payload.update({
                "stand_gate_min_pelvis_height_m": np.asarray([recorder.stand_gate_result["min_pelvis_height_m"]], dtype=np.float64),
                "stand_gate_max_pelvis_tilt_deg": np.asarray([recorder.stand_gate_result["max_pelvis_tilt_deg"]], dtype=np.float64),
                "stand_gate_sample_count": np.asarray([recorder.stand_gate_result["sample_count"]], dtype=np.int64),
            })
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        print(json.dumps({
            "output": str(output),
            "sample_count": int(len(payload["timestamp_s"])),
            "duration_s": float(payload["timestamp_s"][-1]),
            "max_pair_skew_ms": float(np.max(payload["pair_skew_s"]) * 1.0e3),
            "capture_start_command_index": args.skip_command_messages + 1,
        }, ensure_ascii=False, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

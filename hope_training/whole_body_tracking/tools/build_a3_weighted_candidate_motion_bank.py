#!/usr/bin/env python3
"""Build a 50 Hz, 31-DOF A3 reference bank from offline 10-DOF IK clips.

This is a kinematic reference packaging step.  It embeds the offline waist and
right-arm trajectory into the A3 reset pose, refreshes the 32-body FK state in
Isaac Lab, and writes the contract consumed by MotionLibraryLoader.  It does
not claim floating-base, actuator, self-collision, fall, or teacher-data
qualification.  Core and soft-limit boundary samples retain their explicit
weights from the fixed-base audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weighted-index", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--max-candidates", type=int, default=0, help="0 means all rows.")
parser.add_argument("--start-index", type=int, default=0, help="Zero-based CSV row index.")
parser.add_argument("--manifest-every", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_candidates < 0:
    parser.error("--max-candidates must be >= 0")
if args_cli.start_index < 0:
    parser.error("--start-index must be >= 0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from training.robots.agibot_a3 import AGIBOT_A3_CFG  # noqa: E402


@configclass
class CandidateSceneCfg(InteractiveSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    # No simulation stepping is used.  A fixed base makes the FK packaging
    # deterministic and avoids turning this stage into a floating-base audit.
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=AGIBOT_A3_CFG.spawn.replace(fix_base=True),
    )


def _scalar(data: np.lib.npyio.NpzFile, name: str, default: float | None = None) -> float:
    if name not in data:
        if default is None:
            raise KeyError(name)
        return float(default)
    return float(np.asarray(data[name]).reshape(-1)[0])


def _text(data: np.lib.npyio.NpzFile, name: str, default: str) -> str:
    if name not in data:
        return default
    return str(np.asarray(data[name]).reshape(-1)[0])


def _vec(data: np.lib.npyio.NpzFile, name: str, default: tuple[float, float, float]) -> np.ndarray:
    if name not in data:
        return np.asarray(default, dtype=np.float32)
    value = np.asarray(data[name], dtype=np.float32).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid {name}: shape={value.shape}")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty candidate index: {path}")
    required = {"stroke", "goal_id", "trajectory_npz", "dataset_role", "sample_weight"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"candidate index missing columns: {sorted(missing)}")
    return rows


def _safe_stroke(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return value or "unknown"


def _load_checkpoint(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    completed: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    if not path.exists():
        return completed, failures
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at line {line_no}: {exc}") from exc
            key = str(record.get("episode_id", ""))
            if not key:
                continue
            if record.get("status") == "completed" and isinstance(record.get("entry"), dict):
                completed[key] = record["entry"]
            elif record.get("status") == "failed":
                failures[key] = record
    return completed, failures


def _write_manifest(path: Path, entries: list[dict], failures: list[dict], source_index: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "a3_weighted_candidate_reference_bank/v1",
        "status": "candidate_reference_pending_training_split",
        "source_index": str(source_index.expanduser().resolve()),
        "notice_contract": "project_manifest",
        "reference_semantics": "fixed-base FK-expanded offline IK candidate reference",
        "coordinate_contract": "current_root_relative_initial_heading",
        "root_pose_contract": {
            "root_position_w_m": [0.0, 0.0, 1.0684],
            "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "tcp_contract": {
            "body": "right_wrist_yaw_Link",
            "mount_offset_local_m": [0.210211399202899, 0.0320784994676765, 0.0320358706296689],
            "normal_axis": "+Y",
        },
        "waist_contract": {
            "waist_yaw": "preserved_or_ready_state_defined",
            "waist_pitch": "forward_only_nonnegative_joint_pitch",
            "backward_tilt_allowed": False,
            "forward_tilt_limit_deg": 20.0,
            "waist_roll_abs_limit_deg": 20.0,
        },
        "physics_qualified": False,
        "teacher_approved": False,
        "training_admission": False,
        "floating_base_replay_done": False,
        "self_collision_observable": False,
        "weights": {"core": 1.0, "boundary": 0.25},
        "counts": {
            "completed": len(entries),
            "failed": len(failures),
            "core": sum(e.get("dataset_role") == "core" for e in entries),
            "boundary": sum(e.get("dataset_role") == "boundary" for e in entries),
        },
        "motions": entries,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _downsample_indices(frame_count: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError(f"trajectory needs at least 2 frames, got {frame_count}")
    indices = np.arange(0, frame_count, 2, dtype=np.int64)
    if indices[-1] != frame_count - 1:
        indices = np.concatenate([indices, np.asarray([frame_count - 1], dtype=np.int64)])
    return indices


def _build_motion(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    raw_ids: torch.Tensor,
    source_path: Path,
    output_path: Path,
) -> tuple[int, int, list[int], list[int]]:
    with np.load(source_path, allow_pickle=False) as data:
        raw_names = tuple(str(x) for x in np.asarray(data["joint_names"]).reshape(-1))
        q_raw = np.asarray(data["joint_pos"], dtype=np.float32)
        qd_raw = np.asarray(data["joint_vel"], dtype=np.float32)
        hit_raw = int(_scalar(data, "hit_frame", q_raw.shape[0] - 1))
        if q_raw.ndim != 2 or qd_raw.shape != q_raw.shape:
            raise ValueError(f"invalid raw q/qd shapes: {q_raw.shape} vs {qd_raw.shape}")
        if len(raw_names) != q_raw.shape[1] or not np.isfinite(q_raw).all() or not np.isfinite(qd_raw).all():
            raise ValueError("raw joint names/trajectory are invalid or non-finite")
        if not 0 <= hit_raw < q_raw.shape[0]:
            raise ValueError(f"invalid raw hit_frame={hit_raw} for {q_raw.shape[0]} frames")
        source_goal_id = _text(data, "source_goal_id", source_path.stem)
        swing_type = _text(data, "selected_swing_type", _text(data, "requested_swing_type", "unknown"))
        canonical_position = _vec(data, "canonical_position", (0.0, 0.0, 0.0))
        canonical_normal = _vec(data, "canonical_normal", (0.0, 0.0, 1.0))
        canonical_velocity = _vec(data, "canonical_velocity", (0.0, 0.0, 0.0))
        strike_time = _scalar(data, "requested_strike_time_s", hit_raw * 0.01)

    indices = _downsample_indices(q_raw.shape[0])
    q = q_raw[indices]
    qd = qd_raw[indices]
    hit = int(np.flatnonzero(indices == hit_raw)[0]) if hit_raw in set(indices.tolist()) else int(round(hit_raw / 2.0))
    hit = min(max(hit, 0), q.shape[0] - 1)

    device = sim.device
    default_q = robot.data.default_joint_pos[0].detach().clone()
    default_qd = robot.data.default_joint_vel[0].detach().clone()
    q_full = default_q.unsqueeze(0).repeat(q.shape[0], 1)
    qd_full = default_qd.unsqueeze(0).repeat(q.shape[0], 1)
    q_full[:, raw_ids] = torch.as_tensor(q, dtype=torch.float32, device=device)
    qd_full[:, raw_ids] = torch.as_tensor(qd, dtype=torch.float32, device=device)

    root = robot.data.default_root_state[0].detach().clone()
    root_state = root.unsqueeze(0).clone()
    root_state[:, 7:] = 0.0
    joint_pos_out: list[np.ndarray] = []
    joint_vel_out: list[np.ndarray] = []
    body_pos_out: list[np.ndarray] = []
    body_quat_out: list[np.ndarray] = []
    body_lin_out: list[np.ndarray] = []
    body_ang_out: list[np.ndarray] = []
    for frame in range(q.shape[0]):
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(q_full[frame : frame + 1], qd_full[frame : frame + 1])
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0.0)
        joint_pos_out.append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=True))
        joint_vel_out.append(robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32, copy=True))
        body_pos_out.append(robot.data.body_pos_w[0].detach().cpu().numpy().astype(np.float32, copy=True))
        body_quat_out.append(robot.data.body_quat_w[0].detach().cpu().numpy().astype(np.float32, copy=True))
        body_lin_out.append(robot.data.body_lin_vel_w[0].detach().cpu().numpy().astype(np.float32, copy=True))
        body_ang_out.append(robot.data.body_ang_vel_w[0].detach().cpu().numpy().astype(np.float32, copy=True))

    arrays = {
        "fps": np.asarray(50, dtype=np.int32),
        "joint_names": np.asarray(list(robot.joint_names)),
        "joint_pos": np.stack(joint_pos_out),
        "joint_vel": np.stack(joint_vel_out),
        "body_pos_w": np.stack(body_pos_out),
        "body_quat_w": np.stack(body_quat_out),
        "body_lin_vel_w": np.stack(body_lin_out),
        "body_ang_vel_w": np.stack(body_ang_out),
        "hit_frame": np.asarray(hit, dtype=np.int32),
        "physics_qualified": np.asarray(False, dtype=np.bool_),
        "teacher_approved": np.asarray(False, dtype=np.bool_),
        "source_goal_id": np.asarray(source_goal_id),
        "selected_swing_type": np.asarray(swing_type),
        "canonical_position": canonical_position,
        "canonical_normal": canonical_normal,
        "canonical_velocity": canonical_velocity,
        "canonical_strike_time_s": np.asarray(strike_time, dtype=np.float32),
    }
    for name, value in arrays.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite generated array: {name}")
    if arrays["joint_pos"].shape[1:] != (31,) or arrays["body_pos_w"].shape[1:] != (32, 3):
        raise ValueError(
            f"A3 runtime contract mismatch: joint_pos={arrays['joint_pos'].shape}, body_pos_w={arrays['body_pos_w'].shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return q.shape[0], hit, list(arrays["joint_pos"].shape), list(arrays["body_pos_w"].shape)


def main() -> None:
    source_index = args_cli.weighted_index.expanduser().resolve()
    output_root = args_cli.output_root.expanduser().resolve()
    manifest_path = args_cli.manifest.expanduser().resolve()
    checkpoint = (args_cli.checkpoint or output_root / "conversion_checkpoint.jsonl").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(source_index)
    selected = rows[args_cli.start_index :]
    if args_cli.max_candidates:
        selected = selected[: args_cli.max_candidates]

    completed, failures_by_id = _load_checkpoint(checkpoint)
    all_failures = dict(failures_by_id)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_handle = checkpoint.open("a", encoding="utf-8", buffering=1)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(CandidateSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    joint_names = list(robot.joint_names)
    if len(joint_names) != 31 or robot.num_bodies != 32:
        raise RuntimeError(f"unexpected A3 dimensions: joints={len(joint_names)}, bodies={robot.num_bodies}")
    raw_ids_cache: dict[tuple[str, ...], torch.Tensor] = {}
    manifest_entries = dict(completed)
    processed = 0
    try:
        for offset, row in enumerate(selected, start=args_cli.start_index):
            episode_id = f"candidate_{offset:05d}"
            if episode_id in manifest_entries:
                continue
            stroke = _safe_stroke(row["stroke"])
            source_path = Path(row["trajectory_npz"]).expanduser().resolve()
            output_rel = Path("motions") / stroke / f"{episode_id}.npz"
            output_path = output_root / output_rel
            record = {"episode_id": episode_id, "row_index": offset, "source_npz": str(source_path)}
            try:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                with np.load(source_path, allow_pickle=False) as source_data:
                    raw_names = tuple(str(x) for x in np.asarray(source_data["joint_names"]).reshape(-1))
                missing = [name for name in raw_names if name not in joint_names]
                if missing:
                    raise ValueError(f"raw joints missing from A3 articulation: {missing}")
                if raw_names not in raw_ids_cache:
                    raw_ids_cache[raw_names] = torch.as_tensor(
                        [joint_names.index(name) for name in raw_names], dtype=torch.long, device=sim.device
                    )
                frames, hit, q_shape, body_shape = _build_motion(
                    sim, scene, robot, raw_ids_cache[raw_names], source_path, output_path
                )
                with np.load(source_path, allow_pickle=False) as source_data:
                    canonical_normal = _vec(source_data, "canonical_normal", (0.0, 0.0, 1.0))
                    canonical_velocity = _vec(source_data, "canonical_velocity", (0.0, 0.0, 0.0))
                entry = {
                    "episode_id": episode_id,
                    "motion_id": episode_id,
                    "stroke_type": str(row["stroke"]).lower(),
                    "motion_npz": str(output_rel),
                    "library_motion_npz": str(output_rel),
                    "canonical_motion_npz": True,
                    "fps": 50,
                    "joint_pos_shape": q_shape,
                    "body_pos_w_shape": body_shape,
                    "hit_event": {
                        "motion_hit_frame": hit,
                        "source_hit_frame_100hz": int(round(hit * 2)),
                        "strike_time_s": float(row["strike_time_s"]),
                    },
                    "strike_target": {
                        "racket_position_m": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                        "racket_velocity_mps": [float(x) for x in canonical_velocity],
                        "racket_normal_w": [float(x) for x in canonical_normal],
                    },
                    "canonical_goal_10d": {
                        "position_m": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                        "normal_w": [float(x) for x in canonical_normal],
                        "linear_velocity_mps": [float(x) for x in canonical_velocity],
                        "time_to_hit_s": float(row["strike_time_s"]),
                    },
                    "source_goal_id": row["goal_id"],
                    "source_npz": str(source_path),
                    "fixed_base_physx_status": row["fixed_base_physx_status"],
                    "dataset_role": row["dataset_role"],
                    "sample_weight": float(row["sample_weight"]),
                    "screening_reason": row["screening_reason"],
                    "physics_qualified": False,
                    "teacher_approved": False,
                    "training_admission": False,
                }
                manifest_entries[episode_id] = entry
                checkpoint_handle.write(json.dumps({"status": "completed", "episode_id": episode_id, "entry": entry}, ensure_ascii=False) + "\n")
                all_failures.pop(episode_id, None)
                processed += 1
                if processed == 1 or processed % max(1, args_cli.manifest_every) == 0:
                    entries = [manifest_entries[k] for k in sorted(manifest_entries)]
                    _write_manifest(manifest_path, entries, list(all_failures.values()), source_index)
                    print(json.dumps({"processed_this_run": processed, "completed_total": len(entries), "last": episode_id}, ensure_ascii=False), flush=True)
            except Exception as exc:
                failure = {**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                all_failures[episode_id] = failure
                checkpoint_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(json.dumps(failure, ensure_ascii=False), flush=True)
    finally:
        checkpoint_handle.close()
        entries = [manifest_entries[k] for k in sorted(manifest_entries)]
        _write_manifest(manifest_path, entries, list(all_failures.values()), source_index)
        watchdog = threading.Timer(8.0, os._exit, args=(0,))
        watchdog.daemon = True
        watchdog.start()
        simulation_app.close()
    print(json.dumps({"completed": len(manifest_entries), "failed": len(all_failures), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive MuJoCo replay for A3 retargeted motion NPZ files.

The pelvis is welded to the world by removing the free joint at load time.
Each frame is applied kinematically, so this viewer answers "does the motion
look right?" without lower-body balance, contacts, or actuator dynamics.

Example:
    python tools/mujoco_replay_fixed_waist.py --limit 20
    python tools/mujoco_replay_fixed_waist.py --manifest path/to/manifest.json --limit 100
    python tools/mujoco_replay_fixed_waist.py --grid-library /path/to/grid_action_library.npz --limit 10
    python tools/mujoco_replay_fixed_waist.py --motion path/to/motion.npz

Requires the Python ``mujoco`` package and a graphical session.
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

import numpy as np


DEFAULT_MODEL = Path(
    "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
DEFAULT_MOTION_ROOT = Path("data/analysis/mocap_cleaning_outputs")
DEFAULT_MANIFEST = Path(
    "a3_ik_point_offline_wrapper_v2/training_reference_bank_merged_20260807/training_manifest.json"
)
GRID_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
GRID_DEFAULT_LIBRARY = Path(
    "/home/bistu/桌面/A3_strike_grid_library_v1/data/grid_action_library.npz"
)
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.1311,
    "right_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "right_hip_roll_joint": -0.0056,
    "left_hip_yaw_joint": -0.0348,
    "right_hip_yaw_joint": 0.0348,
    "left_knee_joint": 0.2468,
    "right_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "right_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_ankle_roll_joint": 0.0078,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
    "left_elbow_joint": 0.8,
    "right_elbow_joint": 0.8,
}
DEFAULT_JOINTS = [
    # The large 23k candidate bank was exported from Isaac articulation order,
    # not MuJoCo/SDK XML order. Many legacy files omit joint_names, so this
    # fallback order is part of the replay contract.
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--motion", type=Path, help="One NPZ file to replay")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_MOTION_ROOT,
        help="Fallback search root for **/optimized_motion_npz/*.npz",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--grid-library",
        type=Path,
        default=None,
        help="Replay grid_action_library.npz launch->hit->follow actions",
    )
    parser.add_argument("--index", type=int, default=0, help="Sorted motion index")
    parser.add_argument("--limit", type=int, default=0, help="Replay at most N motions; 0 means all")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop-motion", action="store_true")
    parser.add_argument("--list", action="store_true", help="List discovered motions and exit")
    return parser.parse_args()


def discover(root: Path, manifest: Path | None) -> list[Path]:
    if manifest is not None and manifest.exists():
        import json

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths: list[Path] = []
        for motion in payload.get("motions", []):
            raw = str(motion.get("motion_npz", ""))
            candidate = Path(raw)
            if not candidate.exists() and "/HOPETableTennis/" in raw:
                candidate = Path.cwd() / raw.split("/HOPETableTennis/", 1)[1]
            if candidate.exists():
                paths.append(candidate)
        if paths:
            return paths
        print(f"warning: manifest has no local motion files: {manifest}")
    return sorted(root.glob("**/optimized_motion_npz/*.npz"))


def load_fixed_model(model_path: Path):
    import mujoco

    xml = model_path.read_text(encoding="utf-8")
    fixed_xml, count = re.subn(
        r"\s*<(?:freejoint\s+name=\"[^\"]+\"\s*/|joint\s+name=\"[^\"]+\"\s+type=\"free\"\s*/)>\s*",
        "\n",
        xml,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"free root joint not found in {model_path}")
    # The source model's stand keyframe also contains the seven free-joint
    # coordinates. It is only a convenience keyframe, so discard it after
    # converting the root to a welded body rather than leaving an invalid qpos.
    fixed_xml = re.sub(r"\s*<keyframe>.*?</keyframe>\s*", "\n", fixed_xml, count=1, flags=re.DOTALL)
    # Keep the temporary XML beside the original so compiler meshdir="../meshes"
    # continues to resolve exactly as it does for the checked-in model.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="a3_fixed_", dir=model_path.parent, delete=False
    ) as handle:
        handle.write(fixed_xml)
        temporary_path = Path(handle.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)
    return model


def motion_joint_map(npz: np.lib.npyio.NpzFile, available: set[str]) -> list[tuple[int, str]]:
    names = npz.get("joint_names")
    if names is None:
        source_names = DEFAULT_JOINTS
    else:
        source_names = [str(x) for x in names.tolist()]
    missing = sorted(set(source_names) - available)
    if missing:
        raise ValueError(f"motion contains unknown A3 joints: {missing}")
    return [(i, name) for i, name in enumerate(source_names)]


def apply_frame(mujoco, model, data, frame: np.ndarray, source_names: list[str]) -> None:
    if frame.ndim != 1 or frame.shape[0] != len(source_names):
        raise ValueError(f"joint frame must have shape [{len(source_names)}], got {frame.shape}")
    for column, name in enumerate(source_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = frame[column]
    mujoco.mj_forward(model, data)


def model_qpos_from_named_values(mujoco, model, values: dict[str, float]) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    for name, value in values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint {name} not found in model")
        q[model.jnt_qposadr[joint_id]] = value
    return q


def grid_archive_size(archive: np.lib.npyio.NpzFile) -> int:
    if "q_hit" in archive.files:
        return int(archive["q_hit"].shape[0] * archive["q_hit"].shape[1])
    if all(f"q{i}" in archive.files for i in range(7)):
        return int(archive["q0"].shape[0])
    raise ValueError("grid library has neither v1 q_hit arrays nor v2 q0..q6 columns")


def _grid_action(archive: np.lib.npyio.NpzFile, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return launch/hit/follow and metadata for either grid-library format."""
    total = grid_archive_size(archive)
    if not 0 <= index < total:
        raise IndexError(f"grid action index must be in [0, {total - 1}]")
    if "q_hit" in archive.files:
        slots = int(archive["q_hit"].shape[1])
        voxel, slot = divmod(index, slots)
        launch = np.asarray(archive["q_launch"][voxel, slot], dtype=np.float64)
        hit = np.asarray(archive["q_hit"][voxel, slot], dtype=np.float64)
        follow = np.asarray(archive["q_follow"][voxel, slot], dtype=np.float64)
        arrival = float(archive["T"][voxel, slot])
        solution = str(archive["solution_types"][slot])
        info = {
            "format": "v1",
            "voxel": int(archive["voxel_id"][voxel]),
            "slot": slot + 1,
            "solution": solution,
            "arrival_s": arrival,
            "position": np.asarray(archive["p"][voxel, slot]).round(3).tolist(),
            "speed_mps": float(archive["speed"][voxel, slot]),
        }
    else:
        launch = np.asarray([archive[f"launch{i}"][index] for i in range(7)], dtype=np.float64)
        hit = np.asarray([archive[f"q{i}"][index] for i in range(7)], dtype=np.float64)
        follow = np.asarray([archive[f"follow{i}"][index] for i in range(7)], dtype=np.float64)
        arrival = float(archive["arrival_time_s"][index])
        solution = str(archive["solution_type"][index])
        info = {
            "format": "v2",
            "voxel": int(archive["voxel_id"][index]),
            "slot": int(archive["slot"][index]),
            "solution": solution,
            "arrival_s": arrival,
            "position": [
                round(float(archive["p_x"][index]), 3),
                round(float(archive["p_y"][index]), 3),
                round(float(archive["p_z"][index]), 3),
            ],
            "speed_mps": float(archive["speed_mps"][index]),
        }
    return launch, hit, follow, info


def grid_frames(mujoco, model, archive: np.lib.npyio.NpzFile, index: int) -> tuple[np.ndarray, dict[str, object]]:
    launch_values, hit_values, follow_values, info = _grid_action(archive, index)
    base = model_qpos_from_named_values(mujoco, model, DEFAULT_POSE)
    launch = np.array(base, copy=True)
    hit = np.array(base, copy=True)
    follow = np.array(base, copy=True)
    for name, q0, q1, q2 in zip(
        GRID_JOINTS,
        launch_values,
        hit_values,
        follow_values,
        strict=True,
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        address = model.jnt_qposadr[joint_id]
        launch[address], hit[address], follow[address] = q0, q1, q2
    arrival = float(info["arrival_s"])
    fps = 50.0
    n1 = max(2, int(round(arrival * fps)))
    n2 = max(2, int(round(0.30 * fps)))
    frames = np.concatenate(
        (
            np.linspace(launch, hit, n1, endpoint=False),
            np.linspace(hit, follow, n2),
        ),
        axis=0,
    )
    return frames, info


def replay_one(mujoco, viewer, model, data, path: Path, name_to_qpos: dict[str, int], speed: float, loop: bool) -> None:
    with np.load(path, allow_pickle=False) as npz:
        q = np.asarray(npz["joint_pos"], dtype=np.float64)
        names = npz.get("joint_names")
        source_names = DEFAULT_JOINTS if names is None else [str(x) for x in names.tolist()]
        fps = float(np.asarray(npz.get("fps", np.asarray(120.0))).reshape(-1)[0])
    if fps <= 0 or speed <= 0:
        raise ValueError("fps and --speed must be positive")
    print(f"replay {path.name}: {len(q)} frames @ {fps:g} Hz")
    while True:
        start = time.perf_counter()
        for frame in q:
            apply_frame(mujoco, model, data, frame, source_names)
            viewer.sync()
            target = (time.perf_counter() - start) + 1.0 / (fps * speed)
            time.sleep(max(0.0, target - (time.perf_counter() - start)))
            if not viewer.is_running():
                return
        if not loop:
            return


def replay_grid_one(mujoco, viewer, model, data, archive, index: int, speed: float) -> bool:
    frames, info = grid_frames(mujoco, model, archive, index)
    print(f"grid[{index}] {info}")
    start = time.perf_counter()
    for frame in frames:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        viewer.sync()
        target = (time.perf_counter() - start) + 1.0 / (50.0 * speed)
        time.sleep(max(0.0, target - (time.perf_counter() - start)))
        if not viewer.is_running():
            return False
    return True


def main() -> int:
    args = parse_args()
    if args.grid_library:
        motions: list[Path] = []
    else:
        motions = [args.motion] if args.motion else discover(args.root, args.manifest)
    if not args.grid_library and not motions:
        raise SystemExit(f"no optimized motion NPZ files found under {args.root}")
    if args.list and not args.grid_library:
        try:
            for i, path in enumerate(motions):
                print(f"{i:5d}  {path}")
        except BrokenPipeError:
            pass
        return 0
    if not args.grid_library and args.motion is None:
        if args.index < 0 or args.index >= len(motions):
            raise SystemExit(f"--index must be in [0, {len(motions) - 1}]")
        motions = motions[args.index :]
    if args.limit > 0:
        motions = motions[: args.limit]

    import mujoco
    import mujoco.viewer

    model = load_fixed_model(args.model)
    data = mujoco.MjData(model)
    name_to_qpos = {}
    for name in DEFAULT_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint {name} not found in {args.model}")
        name_to_qpos[name] = int(model.jnt_qposadr[joint_id])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Let the viewer open on the robot rather than the floor.
        viewer.cam.lookat[:] = [0.0, 0.0, 1.35]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        if args.grid_library:
            with np.load(args.grid_library, allow_pickle=False) as archive:
                total = grid_archive_size(archive)
                indices = range(args.index, total)
                if args.limit > 0:
                    indices = range(args.index, min(total, args.index + args.limit))
                for index in indices:
                    if not replay_grid_one(mujoco, viewer, model, data, archive, index, args.speed):
                        break
            return 0
        for path in motions:
            replay_one(mujoco, viewer, model, data, path, name_to_qpos, args.speed, args.loop_motion)
            if not viewer.is_running():
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

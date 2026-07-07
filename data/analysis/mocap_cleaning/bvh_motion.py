"""Minimal BVH parser and sampler for generic retarget initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointChannels:
    name: str
    channels: tuple[str, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class BvhJoint:
    name: str
    parent: str | None
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class BvhMotion:
    path: str
    frame_time: float
    frames: int
    joints: dict[str, JointChannels]
    hierarchy: dict[str, BvhJoint]
    values: np.ndarray

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.frames, dtype=np.float64) * self.frame_time

    def sample_channel_series(self, joint: str, channel: str) -> np.ndarray:
        item = self.joints[joint]
        idx = item.channels.index(channel)
        return self.values[:, item.indices[idx]]

    def joint_offset(self, joint: str) -> np.ndarray:
        return np.asarray(self.hierarchy[joint].offset, dtype=np.float64)


def load_bvh(path: str | Path) -> BvhMotion:
    bvh_path = Path(path)
    lines = bvh_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints: dict[str, JointChannels] = {}
    hierarchy: dict[str, BvhJoint] = {}
    channel_cursor = 0
    i = 0
    stack: list[str] = []
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTION"):
            break
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            name = line.split()[1]
            parent = stack[-1] if stack else None
            hierarchy[name] = BvhJoint(name=name, parent=parent, offset=(0.0, 0.0, 0.0))
            stack.append(name)
        elif line.startswith("OFFSET"):
            if not stack:
                raise ValueError(f"offset without joint at line {i+1}")
            parts = line.split()
            hierarchy[stack[-1]] = BvhJoint(
                name=stack[-1],
                parent=hierarchy[stack[-1]].parent,
                offset=(float(parts[1]), float(parts[2]), float(parts[3])),
            )
        elif line.startswith("CHANNELS"):
            parts = line.split()
            count = int(parts[1])
            channels = tuple(parts[2 : 2 + count])
            if not stack:
                raise ValueError(f"channel block without joint at line {i+1}")
            joints[stack[-1]] = JointChannels(
                name=stack[-1],
                channels=channels,
                indices=tuple(range(channel_cursor, channel_cursor + count)),
            )
            channel_cursor += count
        elif line.startswith("}"):
            if stack:
                stack.pop()
        i += 1

    if i >= len(lines) or not lines[i].strip().startswith("MOTION"):
        raise ValueError(f"invalid BVH, MOTION section not found: {bvh_path}")
    frames = int(lines[i + 1].split(":")[1].strip())
    frame_time = float(lines[i + 2].split(":")[1].strip())
    motion_rows = []
    for raw in lines[i + 3 : i + 3 + frames]:
        if raw.strip():
            motion_rows.append([float(x) for x in raw.split()])
    values = np.asarray(motion_rows, dtype=np.float64)
    return BvhMotion(
        path=str(bvh_path),
        frame_time=frame_time,
        frames=frames,
        joints=joints,
        hierarchy=hierarchy,
        values=values,
    )


def sample_joint_channels_at_times(
    motion: BvhMotion,
    joint: str,
    channels: tuple[str, ...],
    times: np.ndarray,
) -> np.ndarray:
    src_t = motion.time
    out = []
    for channel in channels:
        series = motion.sample_channel_series(joint, channel)
        out.append(np.interp(times, src_t, series))
    return np.stack(out, axis=1)


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _joint_local_transform(motion: BvhMotion, joint: str, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    pos = motion.joint_offset(joint).copy()
    rot = np.eye(3, dtype=np.float64)
    if joint not in motion.joints:
        return pos, rot
    values = motion.values[frame_index]
    desc = motion.joints[joint]
    for channel, idx in zip(desc.channels, desc.indices):
        value = float(values[idx])
        if channel == "Xposition":
            pos[0] += value
        elif channel == "Yposition":
            pos[1] += value
        elif channel == "Zposition":
            pos[2] += value
        elif channel == "Xrotation":
            rot = rot @ _rot_x(np.deg2rad(value))
        elif channel == "Yrotation":
            rot = rot @ _rot_y(np.deg2rad(value))
        elif channel == "Zrotation":
            rot = rot @ _rot_z(np.deg2rad(value))
    return pos, rot


def joint_global_transform(motion: BvhMotion, joint: str, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    lineage = []
    cur = joint
    while cur is not None:
        lineage.append(cur)
        cur = motion.hierarchy[cur].parent
    lineage.reverse()
    pos = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    for name in lineage:
        local_pos, local_rot = _joint_local_transform(motion, name, frame_index)
        pos = pos + rot @ local_pos
        rot = rot @ local_rot
    return pos, rot

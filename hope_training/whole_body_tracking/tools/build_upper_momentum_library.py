#!/usr/bin/env python3
"""Build an immutable A3 motion library with canonical upper-body momentum.

The source NPZ velocity fields are deliberately ignored. Link poses, COM
velocities, and angular velocities are derived from joint positions and the
same prepared A3 URDF used by Isaac Lab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "training/assets/agibot_a3/urdf/model.urdf"
DEFAULT_ARTICULATION_METADATA = REPO_ROOT / "docs/a3_articulation_metadata.json"
DEFAULT_MANIFEST = REPO_ROOT / "sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "sample_motions/p2_data260708_backhand_strike_only_v2_momentum"
GRAVITY = 9.81


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class Inertial:
    mass: float
    xyz: np.ndarray
    rpy: np.ndarray
    inertia: np.ndarray


def _vector(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {text!r}")
    return value


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        raise ValueError("Revolute joint has a zero axis")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _quat_matrix(quaternion: np.ndarray, convention: str) -> np.ndarray:
    if convention == "wxyz":
        w, x, y, z = quaternion
    elif convention == "xyzw":
        x, y, z, w = quaternion
    else:
        raise ValueError(f"Unknown quaternion convention {convention!r}")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        raise ValueError("Zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]),
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return 0.5 * skew
    return 0.5 * angle / math.sin(angle) * skew


def _load_articulation_metadata(path: Path) -> tuple[list[str], list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    joint_names = value.get("joint_names")
    body_names = value.get("body_names")
    if not isinstance(joint_names, list) or not all(isinstance(item, str) for item in joint_names):
        raise RuntimeError(f"Invalid joint_names in {path}")
    if not isinstance(body_names, list) or not all(isinstance(item, str) for item in body_names):
        raise RuntimeError(f"Invalid body_names in {path}")
    if value.get("num_joints") != len(joint_names) or value.get("num_bodies") != len(body_names):
        raise RuntimeError(f"Count mismatch in {path}")
    return joint_names, body_names


class UrdfModel:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.path = path.resolve()
        self.links = {element.attrib["name"] for element in root.findall("link")}
        self.joints: list[Joint] = []
        self.joint_by_name: dict[str, Joint] = {}
        self.children: dict[str, list[Joint]] = {}
        self.inertials: dict[str, Inertial] = {}

        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            joint = Joint(
                name=element.attrib["name"],
                kind=element.attrib["type"],
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                xyz=_vector(origin.attrib.get("xyz") if origin is not None else None),
                rpy=_vector(origin.attrib.get("rpy") if origin is not None else None),
                axis=_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            )
            self.joints.append(joint)
            self.joint_by_name[joint.name] = joint
            self.children.setdefault(joint.parent, []).append(joint)

        for element in root.findall("link"):
            inertial = element.find("inertial")
            if inertial is None:
                continue
            origin = inertial.find("origin")
            inertia = inertial.find("inertia")
            values = {key: float(inertia.attrib[key]) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            matrix = np.array(
                (
                    (values["ixx"], values["ixy"], values["ixz"]),
                    (values["ixy"], values["iyy"], values["iyz"]),
                    (values["ixz"], values["iyz"], values["izz"]),
                ),
                dtype=np.float64,
            )
            self.inertials[element.attrib["name"]] = Inertial(
                mass=float(inertial.find("mass").attrib["value"]),
                xyz=_vector(origin.attrib.get("xyz") if origin is not None else None),
                rpy=_vector(origin.attrib.get("rpy") if origin is not None else None),
                inertia=matrix,
            )

    def descendants(self, root_link: str) -> set[str]:
        result = {root_link}
        stack = [root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                if joint.child not in result:
                    result.add(joint.child)
                    stack.append(joint.child)
        return result

    def fk(self, joint_values: dict[str, float], root_link: str = "pelvis_link") -> dict[str, np.ndarray]:
        poses = {root_link: np.eye(4, dtype=np.float64)}
        stack = [root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                origin = _transform(_rpy_matrix(joint.rpy), joint.xyz)
                motion = np.eye(4, dtype=np.float64)
                value = float(joint_values.get(joint.name, 0.0))
                if joint.kind in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_angle(joint.axis, value)
                elif joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * value
                elif joint.kind != "fixed":
                    raise ValueError(f"Unsupported joint type {joint.kind!r} for {joint.name}")
                poses[joint.child] = poses[parent] @ origin @ motion
                stack.append(joint.child)
        if poses.keys() != self.links:
            missing = sorted(self.links - poses.keys())
            raise RuntimeError(f"URDF FK did not reach links: {missing}")
        return poses


def _angular_velocity(rotations: np.ndarray, dt: float) -> np.ndarray:
    count = rotations.shape[0]
    result = np.zeros((count, 3), dtype=np.float64)
    for index in range(count):
        before = max(index - 1, 0)
        after = min(index + 1, count - 1)
        elapsed = (after - before) * dt
        if elapsed <= 0.0:
            continue
        relative = rotations[before].T @ rotations[after]
        local = _rotation_log(relative) / elapsed
        result[index] = rotations[index] @ local
    return result


def _compute_momentum(
    model: UrdfModel,
    joint_pos: np.ndarray,
    joint_names: list[str],
    upper_links: list[str],
    dt: float,
) -> tuple[np.ndarray, float, float, dict[str, np.ndarray]]:
    frame_count = joint_pos.shape[0]
    link_positions = {name: np.zeros((frame_count, 3), dtype=np.float64) for name in upper_links}
    link_rotations = {name: np.zeros((frame_count, 3, 3), dtype=np.float64) for name in upper_links}
    link_origins: dict[str, np.ndarray] = {}
    for frame in range(frame_count):
        poses = model.fk(dict(zip(joint_names, joint_pos[frame], strict=True)))
        if frame == 0:
            link_origins = {name: pose[:3, 3].copy() for name, pose in poses.items()}
        for name in upper_links:
            pose = poses[name]
            inertial = model.inertials[name]
            link_positions[name][frame] = pose[:3, 3] + pose[:3, :3] @ inertial.xyz
            link_rotations[name][frame] = pose[:3, :3] @ _rpy_matrix(inertial.rpy)

    masses = np.asarray([model.inertials[name].mass for name in upper_links], dtype=np.float64)
    total_mass = float(masses.sum())
    if total_mass <= 0.0:
        raise RuntimeError("Upper-body mass is zero")
    positions = np.stack([link_positions[name] for name in upper_links], axis=1)
    length_scale = float(np.max(np.linalg.norm(positions[0], axis=-1)))
    if length_scale <= 0.0:
        raise RuntimeError("Invalid upper-body length scale")

    linear_momentum = np.zeros((frame_count, 3), dtype=np.float64)
    angular_momentum = np.zeros((frame_count, 3), dtype=np.float64)
    for name in upper_links:
        inertial = model.inertials[name]
        position = link_positions[name]
        velocity = np.gradient(position, dt, axis=0, edge_order=2)
        omega = _angular_velocity(link_rotations[name], dt)
        linear = inertial.mass * velocity
        linear_momentum += linear
        for frame in range(frame_count):
            rotation = link_rotations[name][frame]
            inertia_pelvis = rotation @ inertial.inertia @ rotation.T
            angular_momentum[frame] += inertia_pelvis @ omega[frame] + np.cross(position[frame], linear[frame])

    return (
        np.concatenate((linear_momentum, angular_momentum), axis=-1).astype(np.float32),
        total_mass,
        length_scale,
        link_origins,
    )


def _body_fk_audit(
    model: UrdfModel,
    joint_names: list[str],
    body_names: list[str],
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    sample_frames: list[int],
) -> dict:
    if body_pos_w.shape[1] != len(body_names):
        raise ValueError(f"body_pos_w has {body_pos_w.shape[1]} bodies, expected {len(body_names)}")
    conventions: dict[str, dict] = {}
    for convention in ("wxyz", "xyzw"):
        errors: list[float] = []
        per_link: dict[str, list[float]] = {name: [] for name in body_names}
        for frame in sample_frames:
            poses = model.fk(dict(zip(joint_names, joint_pos[frame], strict=True)))
            root_pos = body_pos_w[frame, 0]
            root_rotation = _quat_matrix(body_quat_w[frame, 0], convention)
            relative = (root_rotation.T @ (body_pos_w[frame] - root_pos).T).T
            for index, name in enumerate(body_names):
                error = float(np.linalg.norm(relative[index] - poses[name][:3, 3]))
                errors.append(error)
                per_link[name].append(error)
        conventions[convention] = {
            "mean_position_error_m": float(np.mean(errors)),
            "max_position_error_m": float(np.max(errors)),
            "per_link_max_position_error_m": {name: float(max(values)) for name, values in per_link.items()},
        }
    selected = min(conventions, key=lambda key: conventions[key]["mean_position_error_m"])
    return {"selected_quaternion_convention": selected, "conventions": conventions}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest.resolve()
    output_dir = args.output.resolve()
    output_motion_dir = output_dir / "motion_npz"
    output_motion_dir.mkdir(parents=True, exist_ok=True)

    model = UrdfModel(args.urdf.resolve())
    joint_names, body_names = _load_articulation_metadata(args.articulation_metadata.resolve())
    if set(joint_names) - model.joint_by_name.keys():
        raise RuntimeError(f"Joint order contains names absent from URDF: {sorted(set(joint_names) - model.joint_by_name.keys())}")
    if set(body_names) - model.links:
        raise RuntimeError(
            "Articulation body contract contains links absent from the URDF: "
            f"{sorted(set(body_names) - model.links)}"
        )
    upper_links = sorted(
        name
        for name in model.descendants("waist_yaw_Link")
        if name in model.inertials and model.inertials[name].mass > 0.0
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_entries = []
    output_entries = []
    for entry in manifest["motions"]:
        source_path = Path(entry["motion_npz"]).expanduser().resolve()
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        fps = int(np.asarray(arrays["fps"]).reshape(-1)[0])
        dt = 1.0 / fps
        joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
        momentum, total_mass, length_scale, _ = _compute_momentum(
            model, joint_pos, joint_names, upper_links, dt
        )
        sample_frames = sorted({0, joint_pos.shape[0] // 2, joint_pos.shape[0] - 1})
        fk_audit = _body_fk_audit(
            model,
            joint_names,
            body_names,
            joint_pos,
            np.asarray(arrays["body_pos_w"], dtype=np.float64),
            np.asarray(arrays["body_quat_w"], dtype=np.float64),
            sample_frames,
        )
        selected = fk_audit["conventions"][fk_audit["selected_quaternion_convention"]]
        if selected["max_position_error_m"] > args.max_fk_error:
            raise RuntimeError(
                f"{entry['episode_id']}: canonical FK/body max error "
                f"{selected['max_position_error_m']:.6f}m exceeds {args.max_fk_error:.6f}m"
            )

        destination = output_motion_dir / f"{entry['episode_id']}.npz"
        np.savez_compressed(
            destination,
            **arrays,
            upper_momentum_pelvis=momentum,
            upper_mass_kg=np.asarray([total_mass], dtype=np.float32),
            upper_length_scale_m=np.asarray([length_scale], dtype=np.float32),
        )
        output_entry = dict(entry)
        output_entry["motion_npz"] = str(destination)
        output_entry["momentum_preview"] = {
            "field": "upper_momentum_pelvis",
            "frame": "pelvis",
            "linear_units": "kg*m/s",
            "angular_units": "kg*m^2/s",
            "upper_mass_kg": total_mass,
            "length_scale_m": length_scale,
            "source_urdf_sha256": _sha256(args.urdf),
        }
        output_entries.append(output_entry)
        report_entries.append(
            {
                "episode_id": entry["episode_id"],
                "source_motion": str(source_path),
                "output_motion": str(destination),
                "frames": int(joint_pos.shape[0]),
                "upper_mass_kg": total_mass,
                "length_scale_m": length_scale,
                "max_abs_linear_momentum": np.max(np.abs(momentum[:, :3]), axis=0).tolist(),
                "max_abs_angular_momentum": np.max(np.abs(momentum[:, 3:]), axis=0).tolist(),
                "fk_body_audit": fk_audit,
            }
        )

    output_manifest = dict(manifest)
    output_manifest["motions"] = output_entries
    output_manifest["momentum_preview_contract"] = {
        "version": 1,
        "source_manifest": str(manifest_path),
        "source_urdf": str(args.urdf.resolve()),
        "source_urdf_sha256": _sha256(args.urdf),
        "articulation_metadata": str(args.articulation_metadata.resolve()),
        "articulation_metadata_sha256": _sha256(args.articulation_metadata),
        "joint_names": joint_names,
        "body_names": body_names,
        "root_frame": "pelvis",
        "upper_root_link": "waist_yaw_Link",
        "upper_links": upper_links,
        "velocity_source": "finite_difference_of_urdf_fk",
        "npz_velocity_fields_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "manifest": str(manifest_out),
        "source_manifest": str(manifest_path),
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": _sha256(args.urdf),
        "articulation_metadata": str(args.articulation_metadata.resolve()),
        "articulation_metadata_sha256": _sha256(args.articulation_metadata),
        "upper_links": upper_links,
        "motions": report_entries,
    }
    report_path = output_dir / "momentum_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_out), "report": str(report_path), "motions": len(report_entries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--articulation-metadata", type=Path, default=DEFAULT_ARTICULATION_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-fk-error", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False))


if __name__ == "__main__":
    main()

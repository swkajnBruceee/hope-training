#!/usr/bin/env python3
"""Audit the zero-action free-base support geometry before further Stand tuning.

The report is diagnostic only.  It records whole-body COM, per-foot normal
load, foot drift, ankle/waist state, and an explicitly approximate support
margin derived from the lowest 5 mm of the current URDF collision meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument(
    "--task",
    choices=("A3BaseStand-v0", "A3BaseStandAuthorityCandidate-v0"),
    default="A3BaseStand-v0",
)
parser.add_argument(
    "--contact-geometry",
    choices=("current_convex_hull", "conservative_sole_box"),
    default="current_convex_hull",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps < 1:
    parser.error("--steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch
import trimesh

import isaaclab.utils.math as math_utils

import training.tasks.base_locomotion.config.a3  # noqa: F401
from training.robots.agibot_a3 import A3_ANCHOR_BODY, A3_FEET_BODIES


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _signed_margin(point: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    """Signed distance to the closest convex-hull edge; positive is inside."""
    if len(hull) < 3:
        return -math.inf
    distances = []
    for start, end in zip(hull, hull[1:] + hull[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        edge_length = math.hypot(dx, dy)
        distances.append((dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / edge_length)
    return min(distances)


def _tilt_rad(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(-projected_gravity_b[..., 2], min=-1.0, max=1.0))


def _mesh_sole_vertices(side: str, device: str) -> torch.Tensor:
    mesh_path = PROJECT_ROOT / "training" / "assets" / "agibot_a3" / "meshes" / f"{side}_ankle_roll_Link.STL"
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
    sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.005]
    if sole.shape[0] < 3:
        raise RuntimeError(f"Could not derive {side} foot sole vertices from {mesh_path}")
    return sole


def _conservative_sole_box(side: str) -> dict[str, list[float]]:
    """Fit a deliberately inset box to the lowest 1 mm of the source sole."""
    mesh_path = (
        PROJECT_ROOT
        / "training"
        / "assets"
        / "agibot_a3"
        / "meshes"
        / f"{side}_ankle_roll_Link.STL"
    )
    vertices = torch.tensor(trimesh.load(mesh_path, force="mesh").vertices, dtype=torch.float)
    sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.001]
    minimum = sole.amin(dim=0)
    maximum = sole.amax(dim=0)
    inset_xy_m = 0.005
    thickness_m = 0.010
    lower = torch.tensor(
        [minimum[0] + inset_xy_m, minimum[1] + inset_xy_m, minimum[2]], dtype=torch.float
    )
    upper = torch.tensor(
        [maximum[0] - inset_xy_m, maximum[1] - inset_xy_m, minimum[2] + thickness_m],
        dtype=torch.float,
    )
    if torch.any(upper <= lower):
        raise RuntimeError(f"Invalid conservative sole box for {side}: {lower}, {upper}")
    return {
        "center_xyz_m": ((lower + upper) * 0.5).tolist(),
        "size_xyz_m": (upper - lower).tolist(),
        "lower_xyz_m": lower.tolist(),
        "upper_xyz_m": upper.tolist(),
    }


def _box_vertices(box: dict[str, list[float]], device: str) -> torch.Tensor:
    lower = box["lower_xyz_m"]
    upper = box["upper_xyz_m"]
    return torch.tensor(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=torch.float,
        device=device,
    )


def _write_sole_box_variant(output_dir: Path, boxes: dict[str, dict]) -> tuple[Path, dict]:
    """Generate an audit-only URDF changing exactly the two foot collisions."""
    source = PROJECT_ROOT / "training" / "assets" / "agibot_a3" / "urdf" / "model.urdf"
    tree = ET.parse(source)
    root = tree.getroot()
    changed_links = []

    # Make all unchanged mesh paths absolute because the variant lives beside
    # audit artifacts rather than beside the source URDF.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", str((source.parent / filename).resolve()))

    for side, link_name in zip(("left", "right"), A3_FEET_BODIES):
        links = [link for link in root.findall("link") if link.get("name") == link_name]
        if len(links) != 1:
            raise RuntimeError(f"Expected one URDF link named {link_name}, found {len(links)}")
        collisions = links[0].findall("collision")
        if len(collisions) != 1:
            raise RuntimeError(f"Expected one collision on {link_name}, found {len(collisions)}")
        collision = collisions[0]
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        if origin is None or geometry is None:
            raise RuntimeError(f"Malformed collision on {link_name}")
        origin.set("xyz", " ".join(f"{value:.9g}" for value in boxes[side]["center_xyz_m"]))
        origin.set("rpy", "0 0 0")
        for child in list(geometry):
            geometry.remove(child)
        ET.SubElement(
            geometry,
            "box",
            {"size": " ".join(f"{value:.9g}" for value in boxes[side]["size_xyz_m"])},
        )
        changed_links.append(link_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "a3_contact_audit_conservative_sole_box.urdf"
    tree.write(target, encoding="utf-8", xml_declaration=True)
    metadata = {
        "source_urdf": str(source),
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "variant_urdf": str(target),
        "variant_urdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "changed_collision_links": changed_links,
        "unchanged_contract": "visuals, inertials, joints, actuators, initial pose, and all non-foot collisions",
    }
    return target, metadata


def main() -> int:
    env = None
    try:
        cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 1
        cfg.seed = 0
        cfg.sim.device = args_cli.device
        contact_asset = {
            "requested_geometry": args_cli.contact_geometry,
            "current_importer_collider_type": cfg.scene.robot.spawn.collider_type,
            "raw_triangle_mesh_contact_claimed": False,
        }
        boxes = {
            "left": _conservative_sole_box("left"),
            "right": _conservative_sole_box("right"),
        }
        if args_cli.contact_geometry == "conservative_sole_box":
            variant_path, variant_metadata = _write_sole_box_variant(
                args_cli.output.expanduser().resolve().parent / "contact_assets", boxes
            )
            cfg.scene.robot.spawn.asset_path = str(variant_path)
            cfg.scene.robot.spawn.force_usd_conversion = True
            cfg.scene.robot.spawn.usd_dir = str(
                args_cli.output.expanduser().resolve().parent / "contact_assets" / "usd_sole_box"
            )
            contact_asset.update(variant_metadata)
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset(seed=0)

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        sensor = unwrapped.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(A3_FEET_BODIES, preserve_order=True)
        if foot_names != A3_FEET_BODIES:
            raise RuntimeError(f"A3 foot mapping changed: {foot_names}")
        sensor_foot_ids = [sensor.body_names.index(name) for name in A3_FEET_BODIES]
        torso_ids, torso_names = robot.find_bodies([A3_ANCHOR_BODY], preserve_order=True)
        if torso_names != [A3_ANCHOR_BODY]:
            raise RuntimeError("A3 torso mapping changed")
        joint_names = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        joint_ids, resolved_joint_names = robot.find_joints(joint_names, preserve_order=True)
        if resolved_joint_names != joint_names:
            raise RuntimeError(f"A3 support joint mapping changed: {resolved_joint_names}")

        sole_vertices = (
            [_mesh_sole_vertices("left", unwrapped.device), _mesh_sole_vertices("right", unwrapped.device)]
            if args_cli.contact_geometry == "current_convex_hull"
            else [_box_vertices(boxes["left"], unwrapped.device), _box_vertices(boxes["right"], unwrapped.device)]
        )
        masses = robot.data.default_mass[0].to(unwrapped.device)
        total_mass = float(masses.sum().item())
        actions = torch.zeros((1, 14), device=unwrapped.device)
        trace = []
        termination_labels: list[str] = []
        runtime_finite = True

        for step in range(args_cli.steps):
            body_com = robot.data.body_com_pos_w[0]
            system_com = torch.sum(body_com * masses[:, None], dim=0) / masses.sum()
            support_points: list[tuple[float, float]] = []
            foot_rows = []
            for side_index, (body_id, sensor_id) in enumerate(zip(foot_ids, sensor_foot_ids)):
                foot_pos = robot.data.body_pos_w[0, body_id]
                foot_quat = robot.data.body_quat_w[0, body_id]
                local = sole_vertices[side_index]
                world_vertices = math_utils.quat_apply(
                    foot_quat.unsqueeze(0).expand(local.shape[0], -1), local
                ) + foot_pos
                support_points.extend(
                    (float(point[0]), float(point[1])) for point in world_vertices[:, :2].cpu().tolist()
                )
                contact_force = sensor.data.net_forces_w[0, sensor_id]
                foot_rows.append(
                    {
                        "body_name": A3_FEET_BODIES[side_index],
                        "origin_w_m": foot_pos.tolist(),
                        "linear_velocity_w_mps": robot.data.body_lin_vel_w[0, body_id].tolist(),
                        "normal_contact_force_w_n": contact_force.tolist(),
                    }
                )
            support_hull = _convex_hull(support_points)
            margin = _signed_margin(
                (float(system_com[0].item()), float(system_com[1].item())), support_hull
            )
            foot_fz = [max(0.0, row["normal_contact_force_w_n"][2]) for row in foot_rows]
            total_fz = sum(foot_fz)
            torso_quat = robot.data.body_quat_w[0, torso_ids[0]]
            gravity_w = robot.data.GRAVITY_VEC_W
            if gravity_w.ndim > 1:
                gravity_w = gravity_w[0]
            torso_gravity = math_utils.quat_rotate_inverse(torso_quat, gravity_w)
            trace.append(
                {
                    "policy_step": step,
                    "time_s": step * float(unwrapped.step_dt),
                    "system_com_w_m": system_com.tolist(),
                    "support_margin_approx_m": margin,
                    "support_hull_xy_m": support_hull,
                    "foot_normal_load_fraction": [
                        force / total_fz if total_fz > 1.0e-6 else None for force in foot_fz
                    ],
                    "feet": foot_rows,
                    "root_height_m": float(robot.data.root_pos_w[0, 2].item()),
                    "root_tilt_rad": float(_tilt_rad(robot.data.projected_gravity_b[0]).item()),
                    "torso_tilt_rad": float(_tilt_rad(torso_gravity).item()),
                    "joint_position_rad": {
                        name: float(robot.data.joint_pos[0, joint_id].item())
                        for name, joint_id in zip(joint_names, joint_ids)
                    },
                    "applied_torque_nm": {
                        name: float(robot.data.applied_torque[0, joint_id].item())
                        for name, joint_id in zip(joint_names, joint_ids)
                    },
                }
            )
            finite = (
                torch.isfinite(system_com).all()
                and torch.isfinite(robot.data.root_state_w).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
            )
            runtime_finite = runtime_finite and bool(finite)
            _obs, _reward, terminated, truncated, _extras = env.step(actions)
            if bool((terminated | truncated).item()):
                termination_labels = [
                    name
                    for name in unwrapped.termination_manager.active_terms
                    if bool(unwrapped.termination_manager.get_term(name)[0])
                ]
                break

        initial = trace[0]
        final = trace[-1]
        result = {
            "schema_version": 1,
            "audit_id": "a3_base_stand_zero_action_support_audit_v1",
            "task": args_cli.task,
            "simulation_only": True,
            "contact_geometry": args_cli.contact_geometry,
            "contact_asset": contact_asset,
            "sole_box_parameters": boxes if args_cli.contact_geometry == "conservative_sole_box" else None,
            "support_margin_semantics": (
                "convex_hull_of_lowest_5mm_source_foot_mesh_vertices_projected_to_world_xy_approximation"
                if args_cli.contact_geometry == "current_convex_hull"
                else "convex_hull_of_exact_conservative_sole_box_vertices_projected_to_world_xy"
            ),
            "total_mass_kg": total_mass,
            "requested_policy_steps": args_cli.steps,
            "recorded_policy_steps": len(trace),
            "termination_labels": termination_labels,
            "runtime_integrity_passed": runtime_finite,
            "initial_summary": {
                "system_com_w_m": initial["system_com_w_m"],
                "support_margin_approx_m": initial["support_margin_approx_m"],
                "foot_normal_load_fraction": initial["foot_normal_load_fraction"],
            },
            "last_alive_summary": {
                "time_s": final["time_s"],
                "system_com_w_m": final["system_com_w_m"],
                "support_margin_approx_m": final["support_margin_approx_m"],
                "foot_normal_load_fraction": final["foot_normal_load_fraction"],
                "root_height_m": final["root_height_m"],
                "root_tilt_rad": final["root_tilt_rad"],
                "torso_tilt_rad": final["torso_tilt_rad"],
            },
            "trace": trace,
            "changes_authorized_by_this_report": [],
            "stand_phase1_qualified": False,
            "stand_long_training_approved": False,
            "deployment_approved": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "trace"}, indent=2))
        return 0 if runtime_finite else 2
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

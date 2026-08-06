#!/usr/bin/env python3
"""Build an AimSim A3 T2D5 overlay with the proven ``a3_pingpong`` right hand.

The official AimSim package is never modified.  The generated XML retains all
official joints, actuators, sensors, and controller-facing names.  It replaces
the stock right-hand visual with the repository's existing ``a3_pingpong``
gripping-hand mesh and its matching red/black racket meshes.  A fixed 0.18 kg
racket body below ``right_wrist_yaw_Link`` preserves a physical payload;
its body frame is the racket frame and the red hitting face normal is local
``+Y``.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


RACKET_LINK_NAME = "right_racket_link"
RACKET_CENTER_SITE = "right_racket_center"
RACKET_RED_FACE_SITE = "right_racket_red_face_y"
PINGPONG_HAND_MESH = "right_hand_pingpang_Link"
PINGPONG_RED_MESH = "pingpang_red_Link"
PINGPONG_BLACK_MESH = "pingpang_black_Link"
PINGPONG_FACE_COLLISION_MESH = "collision_right_racket_face"

# World W follows the official MuJoCo convention: +Z is up and X/Y span the
# ground plane.  The robot's measured fixed-strike path reaches the near half
# from +X; the opponent and incoming-ball side are -X.  The first task is a
# gravity-compensated straight-line calibration fixture, deliberately not a
# claim of a physically realistic served ball.
TABLE_CENTER_W_M = (-1.15, 0.0, 0.760)
BALL_RADIUS_M = 0.020
BALL_MASS_KG = 0.0027
CALIBRATION_CONTACT_BALL_CENTER_W_M = (-0.5178691058, -0.0959257017, 0.9205844204)
CALIBRATION_INCOMING_VELOCITY_W_MPS = (1.6, 0.0, 0.0)
CALIBRATION_HIT_TIME_S = 1.10
CALIBRATION_BALL_SPAWN_W_M = tuple(
    CALIBRATION_CONTACT_BALL_CENTER_W_M[index] - CALIBRATION_INCOMING_VELOCITY_W_MPS[index] * CALIBRATION_HIT_TIME_S
    for index in range(3)
)


def _overlay_mesh_name(source_name: str) -> str:
    return f"a3_pingpong_overlay_{source_name}"


def _require(path: Path, description: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} is missing: {result}")
    return result


def _one(parent: ET.Element, xpath: str, description: str) -> ET.Element:
    values = parent.findall(xpath)
    if len(values) != 1:
        raise ValueError(f"expected exactly one {description}, found {len(values)}")
    return values[0]


def _racket_body() -> ET.Element:
    # The attachment transform is the existing a3_pingpong wrist-to-racket
    # centre transform.  The matching visual and collision meshes remain on
    # the wrist body below, exactly as in that proven model.
    body = ET.Element("body", {
        "name": RACKET_LINK_NAME,
        "pos": "0.21021 0.032078 0.032036",
        "quat": "1 0 0 0",
    })
    ET.SubElement(body, "inertial", {
        "mass": "0.18", "pos": "0 0 0", "diaginertia": "0.00085 0.00036 0.00072",
    })
    ET.SubElement(body, "site", {"name": RACKET_CENTER_SITE, "pos": "0 0 0", "size": "0.004", "rgba": "0 0 0 0"})
    # MuJoCo site local +Z is aligned to racket-frame +Y. This makes the
    # exported site orientation unambiguous while preserving the contract
    # wording: red face normal is racket-frame local +Y.
    ET.SubElement(body, "site", {
        "name": RACKET_RED_FACE_SITE, "pos": "0 0.0068 0", "size": "0.004",
        "zaxis": "0 1 0", "rgba": "1 0 0 0.5",
    })
    return body


def _add_pingpong_assets(root: ET.Element, pingpong_model: Path) -> None:
    """Import only the four proven pingpong meshes under overlay-only names."""
    source_root = ET.parse(pingpong_model).getroot()
    source_asset = _one(source_root, "asset", "a3_pingpong asset element")
    source_mesh_dir = pingpong_model.parent / "meshes"
    target_asset = _one(root, "asset", "official asset element")
    required = (PINGPONG_HAND_MESH, PINGPONG_RED_MESH, PINGPONG_BLACK_MESH, PINGPONG_FACE_COLLISION_MESH)
    for source_name in required:
        source_mesh = _one(source_asset, f"mesh[@name='{source_name}']", f"a3_pingpong mesh {source_name}")
        file_name = source_mesh.get("file")
        if not file_name:
            raise ValueError(f"a3_pingpong mesh {source_name} does not declare a file")
        mesh_path = (source_mesh_dir / file_name).resolve()
        _require(mesh_path, f"a3_pingpong mesh {source_name}")
        ET.SubElement(target_asset, "mesh", {
            "name": _overlay_mesh_name(source_name),
            "content_type": source_mesh.get("content_type", "model/stl"),
            "file": str(mesh_path),
        })


def _replace_right_hand_with_pingpong(wrist: ET.Element) -> None:
    """Apply the established a3_pingpong wrist-level visual/collision layout."""
    for geom in list(wrist.findall("geom")):
        if geom.get("mesh") == "right_hand_Link":
            wrist.remove(geom)
    ET.SubElement(wrist, "geom", {
        "name": "right_hand_pingpong_visual", "class": "visual", "type": "mesh",
        "rgba": "1 1 1 1", "mesh": _overlay_mesh_name(PINGPONG_HAND_MESH),
    })
    racket_pos = "0.21021 0.032078 0.032036"
    ET.SubElement(wrist, "geom", {
        "name": "right_racket_red_face_visual", "class": "visual", "type": "mesh",
        "pos": racket_pos, "quat": "1 0 0 0", "rgba": "1 0 0 1",
        "mesh": _overlay_mesh_name(PINGPONG_RED_MESH),
    })
    ET.SubElement(wrist, "geom", {
        "name": "right_racket_black_face_visual", "class": "visual", "type": "mesh",
        "pos": racket_pos, "quat": "1 0 0 0", "rgba": "0.1 0.1 0.1 1",
        "mesh": _overlay_mesh_name(PINGPONG_BLACK_MESH),
    })
    ET.SubElement(wrist, "geom", {
        "name": "right_racket_blade_collision", "class": "collision", "type": "mesh",
        "mesh": _overlay_mesh_name(PINGPONG_FACE_COLLISION_MESH),
        "pos": "0.206194 0.025474 0.028020",
    })
    ET.SubElement(wrist, "geom", {
        "name": "right_racket_handle_collision", "class": "collision", "type": "capsule",
        "fromto": "0.060 0.019 -0.095 0.150 0.023 -0.015", "size": "0.018",
    })


def _append_pingpong_calibration_task(root: ET.Element) -> None:
    """Append a fixed incoming-ball coordinate/contact calibration fixture."""
    worldbody = _one(root, "worldbody", "official worldbody element")
    table = ET.SubElement(worldbody, "body", {
        "name": "pingpong_table", "pos": "-1.15 0 0.745",
    })
    ET.SubElement(table, "geom", {
        "name": "pingpong_table_top", "type": "box", "size": "1.37 0.7625 0.015",
        "rgba": "0.05 0.22 0.48 1", "contype": "0", "conaffinity": "0",
    })
    # The net is visual-only during the coordinate/contact fixture.  Making
    # it non-colliding avoids treating a deliberately gravity-compensated,
    # straight-line calibration ball as a physical over-net rally.
    net = ET.SubElement(worldbody, "body", {
        "name": "pingpong_net", "pos": "-1.15 0 0.83625",
    })
    ET.SubElement(net, "geom", {
        "name": "pingpong_net_visual", "type": "box", "size": "0.0075 0.7625 0.07625",
        "rgba": "0.92 0.92 0.92 0.55", "contype": "0", "conaffinity": "0",
    })
    ball = ET.SubElement(worldbody, "body", {
        "name": "pingpong_ball",
        "pos": "{:.10f} {:.10f} {:.10f}".format(*CALIBRATION_BALL_SPAWN_W_M),
        "gravcomp": "1",
    })
    ET.SubElement(ball, "freejoint", {"name": "pingpong_ball_freejoint"})
    ET.SubElement(ball, "geom", {
        "name": "pingpong_ball_geom", "type": "sphere", "size": str(BALL_RADIUS_M),
        "mass": str(BALL_MASS_KG), "rgba": "0.96 0.72 0.05 1",
        "contype": "1", "conaffinity": "7",
        "friction": "0.25 0.005 0.0001", "solref": "0.004 1", "solimp": "0.95 0.99 0.001",
    })


def _extend_keyframes_for_pingpong_ball(root: ET.Element) -> None:
    """Keep vendor stand/reset keyframes valid after adding the ball free joint."""
    ball_qpos = "{:.10f} {:.10f} {:.10f} 1 0 0 0".format(*CALIBRATION_BALL_SPAWN_W_M)
    for key in root.findall("./keyframe/key"):
        qpos = (key.get("qpos") or "").strip()
        if not qpos:
            raise ValueError("cannot extend a keyframe without qpos for pingpong_ball")
        key.set("qpos", f"{qpos} {ball_qpos}")
        qvel = (key.get("qvel") or "").strip()
        if qvel:
            key.set("qvel", f"{qvel} 0 0 0 0 0 0")


def prepare(base_xml: Path, pingpong_model: Path, robot_model_info: Path, aimrt_config: Path, output_root: Path, *, with_pingpong_task: bool) -> dict[str, str]:
    base_xml = _require(base_xml, "official base XML")
    pingpong_model = _require(pingpong_model, "repository a3_pingpong XML")
    robot_model_info = _require(robot_model_info, "robot model-info YAML")
    aimrt_config = _require(aimrt_config, "official AimRT SIL config")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    compiler = _one(root, "compiler", "compiler element")
    mesh_dir = (base_xml.parent.parent / "meshes").resolve()
    compiler.set("meshdir", str(mesh_dir))
    for include in root.findall("include"):
        source = include.get("file", "")
        if "terrain/mujoco" in source:
            include.set("file", str((base_xml.parent / source).resolve()))
    wrist = _one(root, ".//body[@name='right_wrist_yaw_Link']", "right wrist-yaw body")
    if wrist.find(f"./body[@name='{RACKET_LINK_NAME}']") is not None:
        raise ValueError("base XML already contains the requested right racket overlay")
    _add_pingpong_assets(root, pingpong_model)
    _replace_right_hand_with_pingpong(wrist)
    wrist.append(_racket_body())
    if with_pingpong_task:
        _append_pingpong_calibration_task(root)
        _extend_keyframes_for_pingpong_ball(root)
    ET.indent(tree, space="  ")
    model_path = output_root / "raise_a3_t2d5_right_racket.xml"
    tree.write(model_path, encoding="utf-8", xml_declaration=True)

    cfg_root = output_root / "mujoco_cfg"
    cfg_robot = cfg_root / "raise_a3_t2d5"
    cfg_robot.mkdir(parents=True, exist_ok=True)
    sim_cfg = cfg_robot / "mujoco_simulator.yaml"
    sim_cfg.write_text(
        "\n".join((
            "# Generated project-side AimSim racket overlay; do not edit the vendor package.",
            f"robot_model_info_path: {json.dumps(str(robot_model_info))}",
            f"xml_path: {json.dumps(str(model_path))}",
            f"aimrt_path: {json.dumps(str(aimrt_config))}",
            "sim_mode: sil",
            "",
        )), encoding="utf-8",
    )
    launcher_cfg = output_root / "config_mujoco.yaml"
    launcher_cfg.write_text(
        "robot: \"raise_a3_t2d5\"\nsim_mode: \"sil\"\nros_domain_id: 232\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "base_model": str(base_xml),
        "visual_source_model": str(pingpong_model),
        "visual_source": "repository a3_pingpong gripping hand + racket meshes",
        "generated_model": str(model_path),
        "racket_link": RACKET_LINK_NAME,
        "racket_center_site": RACKET_CENTER_SITE,
        "red_face_site": RACKET_RED_FACE_SITE,
        "red_face_normal_racket_frame": [0.0, 1.0, 0.0],
        "racket_mass_kg": 0.18,
        "pingpong_task": {
            "enabled": with_pingpong_task,
            "world_frame": "official_mujoco_world_W: +Z up; table long axis +X/-X",
            "table_frame_T": {
                "origin_w_m": list(TABLE_CENTER_W_M),
                "x_axis_w": [1.0, 0.0, 0.0],
                "y_axis_w": [0.0, 1.0, 0.0],
                "z_axis_w": [0.0, 0.0, 1.0],
                "robot_side": "+X", "opponent_side": "-X",
            },
            "calibration_fixture": {
                "kind": "gravity_compensated_straight_incoming_ball",
                "ball_body": "pingpong_ball",
                "ball_radius_m": BALL_RADIUS_M,
                "ball_mass_kg": BALL_MASS_KG,
                "spawn_w_m": list(CALIBRATION_BALL_SPAWN_W_M),
                "incoming_velocity_w_mps": list(CALIBRATION_INCOMING_VELOCITY_W_MPS),
                "planned_contact_ball_center_w_m": list(CALIBRATION_CONTACT_BALL_CENTER_W_M),
                "planned_contact_after_arm_start_s": CALIBRATION_HIT_TIME_S,
                "table_collision": False,
                "net_collision": False,
                "not_a_physical_rally": True,
            },
        },
        "configuration": {"mujoco_config": str(launcher_cfg), "app_cfg_root": str(cfg_root)},
    }
    metadata_path = output_root / "racket_attachment_contract.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": str(model_path), "config": str(launcher_cfg), "app_cfg_root": str(cfg_root), "metadata": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-xml", type=Path, required=True)
    parser.add_argument("--pingpong-model", type=Path, required=True)
    parser.add_argument("--robot-model-info", type=Path, required=True)
    parser.add_argument("--aimrt-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--with-pingpong-task", action="store_true", help="Append the fixed-ball coordinate/contact calibration fixture.")
    args = parser.parse_args()
    print(json.dumps(prepare(args.base_xml, args.pingpong_model, args.robot_model_info, args.aimrt_config, args.output_root, with_pingpong_task=args.with_pingpong_task), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

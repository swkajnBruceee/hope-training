from pathlib import Path

import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def _load_world_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "hope_world_frame.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["hope_world"]


def _static_tf(parent_frame, child_frame, xyz, rpy):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{child_frame}_static_tf".replace("/", "_"),
        arguments=[
            "--x", str(xyz[0]),
            "--y", str(xyz[1]),
            "--z", str(xyz[2]),
            "--roll", str(rpy[0]),
            "--pitch", str(rpy[1]),
            "--yaw", str(rpy[2]),
            "--frame-id", parent_frame,
            "--child-frame-id", child_frame,
        ],
    )


def generate_launch_description():
    config = _load_world_config()
    frames = config["frames"]
    landmarks = config["landmarks_m"]
    offsets = config["mocap_to_base_link"]
    x_hit = config["planner"]["x_hit"]

    nodes = [
        _static_tf(frames["world"], frames["table_center"], landmarks["table_center"], [0.0, 0.0, 0.0]),
        _static_tf(frames["world"], frames["p1_half_center"], landmarks["p1_half_center"], [0.0, 0.0, 0.0]),
        _static_tf(frames["world"], frames["p2_half_center"], landmarks["p2_half_center"], [0.0, 0.0, 0.0]),
        _static_tf(frames["world"], frames["net_center"], landmarks["net_center"], [0.0, 0.0, 0.0]),
        _static_tf(frames["world"], frames["floor_origin"], landmarks["floor_origin"], [0.0, 0.0, 0.0]),
        _static_tf(frames["world"], frames["virtual_hit_plane"], [x_hit, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _static_tf(frames["p1_mocap"], frames["p1_base_link"], offsets["p1_xyz"], offsets["p1_rpy"]),
        _static_tf(frames["p2_mocap"], frames["p2_base_link"], offsets["p2_xyz"], offsets["p2_rpy"]),
    ]
    return LaunchDescription(nodes)

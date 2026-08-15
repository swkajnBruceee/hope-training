"""Launch the HOPE solver (C++) with the default config."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("solver")) / "config" / "solver.yaml"
    return LaunchDescription([
        Node(
            package="solver",
            executable="solver_node",
            name="solver",
            output="screen",
            parameters=[str(config)],
        ),
    ])

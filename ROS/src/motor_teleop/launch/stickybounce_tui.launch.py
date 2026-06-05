from __future__ import annotations

import os
import shlex

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _build_tmux_launch(context, *args, **kwargs):
    ros_root = os.path.expanduser(LaunchConfiguration("ros_root").perform(context))
    session_name = LaunchConfiguration("session_name").perform(context)
    server_uri = LaunchConfiguration("server_uri").perform(context)
    arduino_port = LaunchConfiguration("arduino_port").perform(context)
    robot_calibration_file = os.path.expanduser(LaunchConfiguration("robot_calibration_file").perform(context))
    motor_cmd_topic = LaunchConfiguration("motor_cmd_topic").perform(context)
    source_ros = "source /opt/ros/humble/setup.bash"
    source_ws = f"source {shlex.quote(f'{ros_root}/install/setup.bash')}"

    stack_cmd = (
        f"cd {shlex.quote(ros_root)} && "
        f"{source_ros} && {source_ws} && "
        "ros2 launch motor_teleop stickybounce_v2.launch.py "
        f"server_uri:={shlex.quote(server_uri)} "
        f"arduino_port:={shlex.quote(arduino_port)} "
        f"robot_calibration_file:={shlex.quote(robot_calibration_file)} "
        "launch_teleop:=false"
    )
    teleop_cmd = (
        f"cd {shlex.quote(ros_root)} && "
        f"{source_ros} && {source_ws} && "
        "ros2 run motor_teleop teleop"
    )
    monitor_cmd = (
        f"cd {shlex.quote(ros_root)} && "
        f"{source_ros} && {source_ws} && "
        f"ros2 topic echo {shlex.quote(motor_cmd_topic)}"
    )

    tmux_script = "\n".join([
        "set -e",
        f"session={shlex.quote(session_name)}",
        "if tmux has-session -t \"$session\" 2>/dev/null; then",
        "  tmux kill-session -t \"$session\"",
        "fi",
        f"tmux new-session -d -s \"$session\" -n stack bash -lc {shlex.quote(stack_cmd)}",
        f"tmux split-window -h -t \"$session\":0 bash -lc {shlex.quote(teleop_cmd)}",
        f"tmux split-window -v -t \"$session\":0.0 bash -lc {shlex.quote(monitor_cmd)}",
        "tmux select-layout -t \"$session\":0 tiled",
        "tmux attach -t \"$session\"",
    ])

    return [
        ExecuteProcess(
            cmd=["bash", "-lc", tmux_script],
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("ros_root", default_value="~/Documents/robotics_final/ROS"),
        DeclareLaunchArgument("session_name", default_value="stickybounce"),
        DeclareLaunchArgument("server_uri", default_value="ws://192.168.0.9:8765"),
        DeclareLaunchArgument("arduino_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument(
            "robot_calibration_file",
            default_value="~/Documents/robotics_final/StickyBounce/robot_calibration.npz",
        ),
        DeclareLaunchArgument(
            "motor_cmd_topic",
            default_value="/stickybounce/motor_cmd_json",
        ),
        OpaqueFunction(function=_build_tmux_launch),
    ])

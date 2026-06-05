from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    server_uri = LaunchConfiguration("server_uri")
    arduino_port = LaunchConfiguration("arduino_port")
    robot_calibration_file = LaunchConfiguration("robot_calibration_file")
    launch_teleop = LaunchConfiguration("launch_teleop")

    return LaunchDescription([
        DeclareLaunchArgument("server_uri", default_value="ws://127.0.0.1:8765"),
        DeclareLaunchArgument("arduino_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("robot_calibration_file", default_value=""),
        DeclareLaunchArgument("launch_teleop", default_value="false"),
        Node(
            package="motor_teleop",
            executable="bridge",
            name="stickybounce_bridge",
            output="screen",
            parameters=[{"server_uri": server_uri}],
        ),
        Node(
            package="motor_teleop",
            executable="game",
            name="stickybounce_game",
            output="screen",
            parameters=[{"robot_calibration_file": robot_calibration_file}],
        ),
        Node(
            package="motor_teleop",
            executable="motor_driver",
            name="motor_driver",
            output="screen",
            parameters=[{"port": arduino_port}],
        ),
        Node(
            package="motor_teleop",
            executable="teleop",
            name="motor_teleop",
            output="screen",
            condition=IfCondition(launch_teleop),
        ),
    ])

import os

from setuptools import setup

package_name = "motor_teleop"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), [
            "launch/stickybounce_v2.launch.py",
            "launch/stickybounce_tui.launch.py",
        ]),
    ],
    install_requires=["setuptools", "numpy", "pyserial", "websockets"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="ROS bridge, game, teleop, and motor control for StickyBounce.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "teleop = motor_teleop.teleop:main",
            "bridge = motor_teleop.bridge:main",
            "game = motor_teleop.game:main",
            "motor_driver = motor_teleop.motor_driver:main",
        ],
    },
)

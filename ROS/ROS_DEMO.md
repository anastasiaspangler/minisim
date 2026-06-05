# ROS Instructions

StickyBounce browser -> StickyBounce websocket server -> ROS bridge -> ROS game node -> ROS motor driver -> Arduino

Manual teleop -> ROS game node -> ROS motor driver

## Build once

From the `ROS/` directory:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select motor_teleop
source install/setup.bash
```

## Run the full stack

### Option 1: launch file

```bash
ros2 launch motor_teleop stickybounce_v2.launch.py server_uri:=ws://<stickybounce-host>:8765 arduino_port:=/dev/cu.usbmodemB43A4536DA882
```

Add `launch_teleop:=true` if you want the keyboard node in the same launch process.

### Option 2: separate terminals

Run the motor driver:

```bash
ros2 run motor_teleop motor_driver --ros-args \
  -p port:=/dev/cu.usbmodemB43A4536DA882 \
  -p baud:=115200
```

Run the ROS game node:

```bash
ros2 run motor_teleop game
```

Run the StickyBounce bridge:

```bash
ros2 run motor_teleop bridge --ros-args \
  -p server_uri:=ws://<stickybounce-host>:8765
```

Run manual teleop in its own terminal:

```bash
ros2 run motor_teleop teleop --ros-args -p topic:=/stickybounce/manual_action_json
```

Use:

- Right arrow or `F` for forward
- Left arrow or `R` for reverse
- Space or `S` for stop
- `V` to toggle vacuum
- `H` to home
- `Q` or `Ctrl-C` to quit

## Run StickyBounce

On the camera machine:

```bash
uv run python server.py
```

## Open the browser

Open `StickyBounce/index.html` on the same machine as `server.py`, or use:

```text
file:///Users/anastasiaspangler/PycharmProjects/minisim/StickyBounce/index.html?ws=<stickybounce-host>
```

## Expected behavior

- Browser shows `ws: connected`
- Browser shows `vacuum: on/off`
- ROS bridge publishes browser state into ROS
- ROS game node publishes normalized robot state and captured ball ids
- ROS motor driver receives motor commands and telemetry
- Arduino receives `F`, `R`, `S`, `V`, and `H`

import json
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Teleop(Node):
    def __init__(self) -> None:
        super().__init__("motor_teleop")
        self.topic = str(self.declare_parameter("topic", "/stickybounce/manual_action_json").value)
        self.hold_timeout = float(self.declare_parameter("hold_timeout", 0.2).value)
        self.print_period = float(self.declare_parameter("print_period", 0.5).value)

        self.pub = self.create_publisher(String, self.topic, 10)
        self.current_rail = "S"
        self.current_rail_at = 0.0
        self.last_print_at = 0.0
        self.last_status = "ready"
        self.buf = b""
        self.publish_action(rail="S", announce=False)

    def publish_action(
        self,
        *,
        rail: str | None = None,
        toggle_vacuum: bool = False,
        home: bool = False,
        swing: bool = False,
        announce: bool = True,
    ) -> None:
        if rail is not None:
            rail = (rail or "S").strip().upper()
            if rail not in {"F", "R", "S"}:
                rail = "S"
            self.current_rail = rail
            self.current_rail_at = time.monotonic()
        else:
            rail = self.current_rail

        payload = {
            "type": "manual_action",
            "rail": rail,
            "toggle_vacuum": bool(toggle_vacuum),
            "home": bool(home),
            "swing": bool(swing),
            "source": "keyboard",
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.pub.publish(msg)

        if toggle_vacuum:
            self.last_status = "vacuum toggle"
        elif home:
            self.last_status = "home"
        elif swing:
            self.last_status = "swing"
        else:
            self.last_status = f"sent {rail}"

        if announce:
            self.get_logger().info(f"cmd {payload}")

    def feed(self, data: bytes) -> bool:
        self.last_status = "listening"
        self.buf += data
        while self.buf:
            if self.buf.startswith(b"\x1b"):
                if len(self.buf) < 3:
                    return False
                seq, self.buf = self.buf[:3], self.buf[3:]
                if seq == b"\x1b[C":
                    self.publish_action(rail="F")
                    self.last_status = "right arrow"
                elif seq == b"\x1b[D":
                    self.publish_action(rail="R")
                    self.last_status = "left arrow"
                elif seq == b"\x1b[A":
                    self.publish_action(rail="F")
                    self.last_status = "up arrow"
                elif seq == b"\x1b[B":
                    self.publish_action(rail="R")
                    self.last_status = "down arrow"
                continue

            ch, self.buf = self.buf[:1], self.buf[1:]
            if ch in (b"q", b"Q", b"\x03"):
                return True
            if ch == b" ":
                self.publish_action(rail="S")
                self.last_status = "stop"
            elif ch == b"f":
                self.publish_action(rail="F")
                self.last_status = "forward"
            elif ch == b"r":
                self.publish_action(rail="R")
                self.last_status = "reverse"
            elif ch == b"s":
                self.publish_action(rail="S")
                self.last_status = "stop"
            elif ch in (b"v", b"V"):
                self.publish_action(toggle_vacuum=True)
                self.last_status = "vacuum"
            elif ch in (b"h", b"H"):
                self.publish_action(home=True)
                self.last_status = "home"
            elif ch in (b"x", b"X"):
                self.publish_action(swing=True)
                self.last_status = "swing"
        return False

    def tick(self) -> None:
        if self.current_rail != "S" and (time.monotonic() - self.current_rail_at) > self.hold_timeout:
            self.publish_action(rail="S", announce=False)
            self.last_status = "auto-stop"

        if (time.monotonic() - self.last_print_at) >= self.print_period:
            self.last_print_at = time.monotonic()
            elapsed = time.monotonic() - self.current_rail_at
            print(
                f"\r[teleop] topic={self.topic} rail={self.current_rail} age={elapsed:0.2f}s status={self.last_status}   ",
                end="",
                flush=True,
            )


def main() -> None:
    rclpy.init()
    node = Teleop()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    print("ROS rail teleop ready")
    print("Right arrow / F = forward")
    print("Left arrow / R = reverse")
    print("Space / S = stop")
    print("V = toggle vacuum")
    print("H = home")
    print("Q or Ctrl-C = quit")
    try:
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                should_quit = node.feed(os.read(fd, 32))
                if should_quit:
                    break
            node.tick()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_action(rail="S", announce=False)
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

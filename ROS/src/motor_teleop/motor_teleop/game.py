import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_STICKYBOUNCE_DIR = Path(__file__).resolve().parents[4] / "StickyBounce"
DEFAULT_ROBOT_CALIBRATION = DEFAULT_STICKYBOUNCE_DIR / "robot_calibration.npz"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StickyBounceGame(Node):
    def __init__(self) -> None:
        super().__init__("stickybounce_game")
        self.browser_state_topic = str(self.declare_parameter("browser_state_topic", "/stickybounce/browser_state_json").value)
        self.telemetry_topic = str(self.declare_parameter("telemetry_topic", "/stickybounce/telemetry_json").value)
        self.manual_action_topic = str(self.declare_parameter("manual_action_topic", "/stickybounce/manual_action_json").value)
        self.motor_cmd_topic = str(self.declare_parameter("motor_cmd_topic", "/stickybounce/motor_cmd_json").value)
        self.game_state_topic = str(self.declare_parameter("game_state_topic", "/stickybounce/game_state_json").value)

        self.game_width = float(self.declare_parameter("game_width", 1920.0).value)
        self.game_height = float(self.declare_parameter("game_height", 1080.0).value)
        self.rail_min_encoder = float(self.declare_parameter("rail_min_encoder", 0.0).value)
        self.rail_max_encoder = float(self.declare_parameter("rail_max_encoder", 1200.0).value)
        self.rail_min_x = float(self.declare_parameter("rail_min_x", 80.0).value)
        self.rail_max_x = float(self.declare_parameter("rail_max_x", self.game_width - 80.0).value)
        self.robot_center_y_offset_px = float(self.declare_parameter("robot_center_y_offset_px", -380.0).value)
        self.vacuum_off_angle_deg = float(self.declare_parameter("vacuum_off_angle_deg", 90.0).value)
        self.vacuum_on_angle_deg = float(self.declare_parameter("vacuum_on_angle_deg", 70.0).value)
        self.robot_calibration_file = str(
            self.declare_parameter("robot_calibration_file", str(DEFAULT_ROBOT_CALIBRATION)).value
        )

        self.robot_calibration = self._load_robot_calibration()
        self.robot_width = float(self.robot_calibration.get("width", 220.0)) if self.robot_calibration else 220.0
        self.robot_height = float(self.robot_calibration.get("height", 180.0)) if self.robot_calibration else 180.0
        self.robot_anchor_y = float(self.robot_calibration.get("anchor_y", self.game_height * 0.8)) if self.robot_calibration else self.game_height * 0.8
        self.robot_bottom_y = float(self.robot_calibration.get("bottom_y", self.robot_anchor_y + self.robot_height / 2.0)) if self.robot_calibration else self.robot_anchor_y + self.robot_height / 2.0

        self.browser_state: dict = {"balls": [], "vacuum_on": False}
        self.telemetry: dict = {}
        self.robot_state: dict = {}

        self.desired_rail_cmd = "S"
        self.desired_vacuum_on = False
        self.actual_vacuum_on = False
        self.pending_home = False
        self.pending_swing = False
        self.captured_ball_ids: set[int] = set()

        self.last_published_motor_cmd = ""
        self.last_published_game_state = ""

        self.motor_pub = self.create_publisher(String, self.motor_cmd_topic, 10)
        self.game_state_pub = self.create_publisher(String, self.game_state_topic, 10)

        self.create_subscription(String, self.browser_state_topic, self._on_browser_state, 10)
        self.create_subscription(String, self.telemetry_topic, self._on_telemetry, 10)
        self.create_subscription(String, self.manual_action_topic, self._on_manual_action, 10)
        self.create_timer(0.05, self._tick)

        self.get_logger().info(f"browser state topic: {self.browser_state_topic}")
        self.get_logger().info(f"telemetry topic: {self.telemetry_topic}")
        self.get_logger().info(f"manual action topic: {self.manual_action_topic}")
        self.get_logger().info(f"motor command topic: {self.motor_cmd_topic}")
        self.get_logger().info(f"game state topic: {self.game_state_topic}")

    def _load_robot_calibration(self) -> dict | None:
        path = Path(self.robot_calibration_file)
        if not path.exists():
            self.get_logger().warn(f"robot calibration missing: {path}")
            return None

        data = np.load(path)
        if "width" not in data or "height" not in data:
            self.get_logger().warn(f"robot calibration missing width/height: {path}")
            return None

        center = data["center"].astype(np.float32) if "center" in data else None
        center_anchor = data["center_anchor"].astype(np.float32) if "center_anchor" in data else None
        anchor_y = float(center_anchor[1]) if center_anchor is not None else float(center[1]) if center is not None else self.game_height * 0.8

        return {
            "width": float(data["width"]),
            "height": float(data["height"]),
            "center": center,
            "center_anchor": center_anchor,
            "anchor_y": anchor_y,
            "bottom_y": float(data["bottom_y"]) if "bottom_y" in data else anchor_y + float(data["height"]) / 2.0,
            "top_y": float(data["top_y"]) if "top_y" in data else anchor_y - float(data["height"]) / 2.0,
        }

    def _parse_json(self, raw: str) -> dict | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data
        return None

    def _on_browser_state(self, msg: String) -> None:
        data = self._parse_json(msg.data)
        if data is None:
            return

        balls = data.get("balls")
        if isinstance(balls, list):
            self.browser_state["balls"] = balls

        if isinstance(data.get("vacuum_on"), bool):
            self.desired_vacuum_on = bool(data["vacuum_on"])
            self.browser_state["vacuum_on"] = self.desired_vacuum_on

        if isinstance(data.get("camera_robot"), dict):
            self.browser_state["camera_robot"] = data["camera_robot"]

        self.browser_state["updated_at"] = data.get("timestamp_ms", time.time() * 1000.0)

    def _on_telemetry(self, msg: String) -> None:
        data = self._parse_json(msg.data)
        if data is None:
            return

        self.telemetry = data
        if isinstance(data.get("vacuum_on"), bool):
            self.actual_vacuum_on = bool(data["vacuum_on"])

    def _on_manual_action(self, msg: String) -> None:
        data = self._parse_json(msg.data)
        if data is None:
            return

        rail = data.get("rail")
        if isinstance(rail, str) and rail in {"F", "R", "S"}:
            self.desired_rail_cmd = rail

        if data.get("home"):
            self.pending_home = True
            self.desired_rail_cmd = "S"

        if data.get("swing"):
            self.pending_swing = True

        if data.get("toggle_vacuum"):
            self.desired_vacuum_on = not self.desired_vacuum_on
            self.browser_state["vacuum_on"] = self.desired_vacuum_on

        if "vacuum_on" in data and isinstance(data["vacuum_on"], bool):
            self.desired_vacuum_on = bool(data["vacuum_on"])
            self.browser_state["vacuum_on"] = self.desired_vacuum_on

    def _normalized_robot_from_telemetry(self) -> dict | None:
        encoder_value = self.telemetry.get("encoder_count", self.telemetry.get("encoder"))
        if encoder_value is None:
            return None

        try:
            encoder_count = float(encoder_value)
        except Exception:
            return None

        span = self.rail_max_encoder - self.rail_min_encoder
        if abs(span) < 1e-9:
            rail_frac = 0.0
        else:
            rail_frac = _clamp((encoder_count - self.rail_min_encoder) / span, 0.0, 1.0)

        center_x = self.rail_min_x + rail_frac * (self.rail_max_x - self.rail_min_x)
        center_y = self.robot_anchor_y - self.robot_center_y_offset_px
        vacuum_angle_deg = float(self.telemetry.get(
            "vacuum_angle_deg",
            self.vacuum_on_angle_deg if self.actual_vacuum_on else self.vacuum_off_angle_deg,
        ))
        angle_span = self.vacuum_on_angle_deg - self.vacuum_off_angle_deg
        vacuum_angle_norm = 0.0 if abs(angle_span) < 1e-9 else _clamp((vacuum_angle_deg - self.vacuum_off_angle_deg) / angle_span, 0.0, 1.0)

        return {
            "status": "ok",
            "source": "encoder",
            "x": center_x - self.robot_width / 2.0,
            "y": center_y - self.robot_height / 2.0,
            "width": self.robot_width,
            "height": self.robot_height,
            "center_x": center_x,
            "center_y": center_y,
            "encoder_count": encoder_count,
            "rail_fraction": rail_frac,
            "vacuum_on": self.actual_vacuum_on,
            "vacuum_angle_deg": vacuum_angle_deg,
            "vacuum_angle_norm": vacuum_angle_norm,
            "age": 0.0,
            "frame_width": self.game_width,
            "frame_height": self.game_height,
            "calibrated": True,
        }

    def _camera_robot_fallback(self) -> dict | None:
        camera_robot = self.browser_state.get("camera_robot")
        if not isinstance(camera_robot, dict):
            return None

        robot = dict(camera_robot)
        robot.setdefault("status", "ok")
        robot.setdefault("source", "camera_fallback")
        robot.setdefault("frame_width", self.game_width)
        robot.setdefault("frame_height", self.game_height)
        robot.setdefault("vacuum_on", self.actual_vacuum_on or self.desired_vacuum_on)
        robot.setdefault("vacuum_angle_deg", self.vacuum_on_angle_deg if robot["vacuum_on"] else self.vacuum_off_angle_deg)
        angle_span = self.vacuum_on_angle_deg - self.vacuum_off_angle_deg
        robot.setdefault(
            "vacuum_angle_norm",
            0.0 if abs(angle_span) < 1e-9 else _clamp((float(robot["vacuum_angle_deg"]) - self.vacuum_off_angle_deg) / angle_span, 0.0, 1.0),
        )
        return robot

    def _active_robot_state(self) -> dict | None:
        robot = self._normalized_robot_from_telemetry()
        if robot is not None:
            self.robot_state = robot
            return robot

        robot = self._camera_robot_fallback()
        if robot is not None:
            self.robot_state = robot
            return robot

        self.robot_state = {}
        return None

    def _capture_ball_ids(self, robot: dict | None) -> list[int]:
        if robot is None:
            return []
        if not self.desired_vacuum_on and not self.actual_vacuum_on:
            return []

        x1 = float(robot.get("x", 0.0))
        y1 = float(robot.get("y", 0.0))
        x2 = x1 + float(robot.get("width", 0.0))
        y2 = y1 + float(robot.get("height", 0.0))
        captured: list[int] = []

        for ball in self.browser_state.get("balls", []):
            if not isinstance(ball, dict):
                continue
            ball_id = ball.get("id")
            if ball_id is None:
                continue
            try:
                ball_id_int = int(ball_id)
            except Exception:
                continue
            if ball_id_int in self.captured_ball_ids:
                continue
            bx = float(ball.get("x", 0.0))
            by = float(ball.get("y", 0.0))
            if x1 <= bx <= x2 and y1 <= by <= y2:
                self.captured_ball_ids.add(ball_id_int)
                captured.append(ball_id_int)

        return captured

    def _motor_cmd_payload(self) -> dict:
        return {
            "type": "motor_cmd",
            "rail": self.desired_rail_cmd,
            "vacuum_on": self.desired_vacuum_on,
            "home": bool(self.pending_home),
            "swing": bool(self.pending_swing),
            "source": "game",
        }

    def _game_state_payload(self, robot: dict | None, captured_ids: list[int]) -> dict:
        vacuum_angle_deg = self.telemetry.get(
            "vacuum_angle_deg",
            self.vacuum_on_angle_deg if self.actual_vacuum_on else self.vacuum_off_angle_deg,
        )
        angle_span = self.vacuum_on_angle_deg - self.vacuum_off_angle_deg
        vacuum_angle_norm = 0.0 if abs(angle_span) < 1e-9 else _clamp((float(vacuum_angle_deg) - self.vacuum_off_angle_deg) / angle_span, 0.0, 1.0)
        return {
            "type": "game_state",
            "robot": robot,
            "vacuum": {
                "on": self.actual_vacuum_on if "vacuum_on" in self.telemetry else self.desired_vacuum_on,
                "desired": self.desired_vacuum_on,
                "angle_deg": float(vacuum_angle_deg),
                "angle_norm": vacuum_angle_norm,
            },
            "captured_ball_ids": captured_ids,
            "browser_vacuum_on": self.browser_state.get("vacuum_on", False),
            "telemetry": self.telemetry,
            "status": "ok" if robot is not None else "waiting_for_robot",
        }

    def _publish_json(self, publisher, payload: dict) -> str:
        encoded = json.dumps(payload, sort_keys=True)
        msg = String()
        msg.data = encoded
        publisher.publish(msg)
        return encoded

    def _tick(self) -> None:
        robot = self._active_robot_state()
        captured_ids = self._capture_ball_ids(robot)

        motor_payload = self._motor_cmd_payload()
        motor_encoded = json.dumps(motor_payload, sort_keys=True)
        if motor_encoded != self.last_published_motor_cmd:
            self._publish_json(self.motor_pub, motor_payload)
            self.last_published_motor_cmd = motor_encoded

        if self.pending_home:
            self.pending_home = False
        if self.pending_swing:
            self.pending_swing = False

        game_payload = self._game_state_payload(robot, captured_ids)
        game_encoded = json.dumps(game_payload, sort_keys=True)
        if game_encoded != self.last_published_game_state or captured_ids:
            self._publish_json(self.game_state_pub, game_payload)
            self.last_published_game_state = game_encoded


def main() -> None:
    rclpy.init()
    node = StickyBounceGame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

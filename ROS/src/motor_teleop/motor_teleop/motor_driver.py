import json
import time

import rclpy
from rclpy.node import Node
from serial import Serial
from std_msgs.msg import String


class MotorDriver(Node):
    def __init__(self) -> None:
        super().__init__("motor_driver")
        self.port = str(self.declare_parameter("port", "/dev/ttyACM0").value)
        self.baud = int(self.declare_parameter("baud", 115200).value)
        self.command_topic = str(self.declare_parameter("command_topic", "/stickybounce/motor_cmd_json").value)
        self.telemetry_topic = str(self.declare_parameter("telemetry_topic", "/stickybounce/telemetry_json").value)
        self.telemetry_period = float(self.declare_parameter("telemetry_period", 0.05).value)

        self.command = {
            "rail": "S",
            "vacuum_on": False,
            "home": False,
            "swing": False,
        }
        self.current_rail = "S"
        self.current_vacuum_on = False
        self.encoder_count = 0.0
        self.vacuum_angle_deg = 90.0
        self.status = "disabled"
        self.enabled = False
        self.ser = None
        self._read_buffer = ""
        self._last_telemetry_pub = 0.0

        self.telemetry_pub = self.create_publisher(String, self.telemetry_topic, 10)
        self.create_subscription(String, self.command_topic, self._on_command, 10)
        self.create_timer(0.02, self._tick)

        if self.port:
            try:
                self.ser = Serial(self.port, self.baud, timeout=0)
                time.sleep(2.0)
                self.ser.write(b"S")
                self.enabled = True
                self.status = "active"
                self.get_logger().info(f"serial connected to {self.port} @ {self.baud}")
            except Exception as exc:  # pragma: no cover - hardware-specific
                self.status = f"disabled:{exc}"
                self.get_logger().warn(f"serial unavailable: {exc}")
        else:
            self.get_logger().warn("motor driver started without ARDUINO_PORT; running disabled")

        self.get_logger().info(f"command topic: {self.command_topic}")
        self.get_logger().info(f"telemetry topic: {self.telemetry_topic}")

    def _normalize_rail(self, raw: str | None) -> str:
        rail = (raw or "S").strip().upper()
        return rail if rail in {"F", "R", "S"} else "S"

    def _write(self, cmd: str) -> None:
        if not self.enabled or self.ser is None:
            return
        try:
            self.ser.write(cmd.encode("ascii"))
        except Exception as exc:  # pragma: no cover - hardware-specific
            self.status = f"fault:{exc}"
            self.enabled = False
            self.get_logger().error(f"serial write failed: {exc}")
            raise

    def _apply_command(self, data: dict) -> None:
        rail = self._normalize_rail(data.get("rail"))
        vacuum_on = data.get("vacuum_on")
        home = bool(data.get("home"))
        swing = bool(data.get("swing"))

        self.command["rail"] = rail
        self.command["vacuum_on"] = bool(vacuum_on) if isinstance(vacuum_on, bool) else self.command["vacuum_on"]
        self.command["home"] = home
        self.command["swing"] = swing

        if home:
            self.get_logger().info("home requested")
            self._write("H")
            self.encoder_count = 0.0

        if rail != self.current_rail:
            self.get_logger().info(f"rail -> {rail}")
            self._write(rail)
            self.current_rail = rail

        if isinstance(vacuum_on, bool) and vacuum_on != self.current_vacuum_on:
            self.get_logger().info(f"vacuum -> {'ON' if vacuum_on else 'OFF'}")
            self._write("V")
            self.current_vacuum_on = vacuum_on
            self.vacuum_angle_deg = 70.0 if vacuum_on else 90.0

        if swing:
            self.get_logger().info("swing requested, ignored in v2 vacuum mode")

    def _on_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("received malformed motor command JSON")
            return
        if not isinstance(data, dict):
            return
        self._apply_command(data)

    def _parse_telemetry_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return

        fields: dict[str, str] = {}
        for chunk in line.split(","):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            fields[key.strip().upper()] = value.strip()

        if "ENC" in fields:
            try:
                self.encoder_count = float(fields["ENC"])
            except ValueError:
                pass
        if "VAC" in fields:
            self.current_vacuum_on = fields["VAC"] in {"1", "true", "TRUE", "on", "ON"}
        if "ANG" in fields:
            try:
                self.vacuum_angle_deg = float(fields["ANG"])
            except ValueError:
                pass

    def _drain_serial(self) -> None:
        if not self.enabled or self.ser is None:
            return

        try:
            waiting = self.ser.in_waiting
        except Exception:
            waiting = 0
        if waiting <= 0:
            return

        try:
            chunk = self.ser.read(waiting).decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - hardware-specific
            self.status = f"fault:{exc}"
            self.get_logger().error(f"serial read failed: {exc}")
            return

        self._read_buffer += chunk
        while "\n" in self._read_buffer:
            line, self._read_buffer = self._read_buffer.split("\n", 1)
            self._parse_telemetry_line(line)

    def _publish_telemetry(self) -> None:
        payload = {
            "type": "telemetry",
            "encoder_count": self.encoder_count,
            "vacuum_on": self.current_vacuum_on,
            "vacuum_angle_deg": self.vacuum_angle_deg,
            "rail_cmd": self.current_rail,
            "status": self.status,
            "timestamp_ms": int(time.time() * 1000),
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.telemetry_pub.publish(msg)

    def _tick(self) -> None:
        self._drain_serial()
        now = time.monotonic()
        if (now - self._last_telemetry_pub) >= self.telemetry_period:
            self._last_telemetry_pub = now
            self._publish_telemetry()


def main() -> None:
    rclpy.init()
    node = MotorDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.ser is not None:
                node.ser.write(b"S")
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

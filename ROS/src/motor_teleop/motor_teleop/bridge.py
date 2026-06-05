import asyncio
import contextlib
import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import websockets


class StickyBounceBridge(Node):
    def __init__(self) -> None:
        super().__init__("stickybounce_bridge")
        self.server_uri = str(self.declare_parameter("server_uri", "ws://127.0.0.1:8765").value)
        self.browser_state_topic = str(self.declare_parameter("browser_state_topic", "/stickybounce/browser_state_json").value)
        self.game_state_topic = str(self.declare_parameter("game_state_topic", "/stickybounce/game_state_json").value)

        self.browser_state_pub = self.create_publisher(String, self.browser_state_topic, 10)
        self.create_subscription(String, self.game_state_topic, self._on_game_state, 10)

        self._lock = threading.Lock()
        self._pending_game_state_raw = ""
        self._last_sent_game_state = ""
        self._last_published_browser_state = ""

        self.get_logger().info(f"server uri: {self.server_uri}")
        self.get_logger().info(f"browser state topic: {self.browser_state_topic}")
        self.get_logger().info(f"game state topic: {self.game_state_topic}")

    def _publish_json(self, publisher, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        publisher.publish(msg)

    def _on_game_state(self, msg: String) -> None:
        with self._lock:
            self._pending_game_state_raw = msg.data.strip()

    def _handle_server_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        if data.get("type") != "snapshot":
            return

        browser = data.get("browser") if isinstance(data.get("browser"), dict) else {}
        payload = {
            "type": "browser_state",
            "balls": browser.get("balls", []),
            "vacuum_on": bool(browser.get("vacuum_on", False)),
            "camera_robot": data.get("camera_robot") if isinstance(data.get("camera_robot"), dict) else None,
            "notes": data.get("notes", []),
            "updated_at": browser.get("updated_at", time.time() * 1000.0),
        }
        encoded = json.dumps(payload, sort_keys=True)
        if encoded != self._last_published_browser_state:
            self._publish_json(self.browser_state_pub, payload)
            self._last_published_browser_state = encoded

    async def _send_game_state(self, websocket) -> None:
        while rclpy.ok():
            with self._lock:
                raw = self._pending_game_state_raw
            if raw and raw != self._last_sent_game_state:
                await websocket.send(raw)
                self._last_sent_game_state = raw
            await asyncio.sleep(0.05)

    async def run(self) -> None:
        while rclpy.ok():
            try:
                self.get_logger().info(f"connecting to {self.server_uri}")
                async with websockets.connect(self.server_uri) as websocket:
                    self.get_logger().info("connected")
                    sender = asyncio.create_task(self._send_game_state(websocket))
                    try:
                        async for raw in websocket:
                            self._handle_server_message(raw)
                    finally:
                        sender.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await sender
            except Exception as exc:  # pragma: no cover - depends on network setup
                if rclpy.ok():
                    self.get_logger().warn(f"bridge disconnected: {exc}")
                    await asyncio.sleep(2.0)


def main() -> None:
    rclpy.init()
    node = StickyBounceBridge()

    bridge_thread = threading.Thread(target=lambda: asyncio.run(node.run()), daemon=True)
    bridge_thread.start()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        bridge_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

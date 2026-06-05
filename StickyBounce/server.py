#!/usr/bin/env python3
"""
Ball Falling Game - Python backend
Detects pink sticky notes via webcam and broadcasts their screen-space
positions over WebSocket to the browser game.

Usage:
    uv run server.py              # auto-picks camera
    uv run server.py --camera 1   # use camera index 1
"""

import argparse
import asyncio
import contextlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import websockets

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0  # overridden by --camera arg
WEBSOCKET_HOST    = "0.0.0.0"
WEBSOCKET_PORT    = 8765
PREVIEW_PORT      = int(os.environ.get("PREVIEW_PORT", "8000"))
DETECTION_FPS     = 15
CALIBRATION_FILE  = "calibration.npz"
ROBOT_CALIBRATION_FILE = os.environ.get("ROBOT_CALIBRATION_FILE", "robot_calibration.npz")
PREVIEW_TEMPLATE_PATH = Path(__file__).resolve().with_name("templates") / "preview.html"
GAME_WIDTH        = int(os.environ.get("GAME_WIDTH", "1920"))
GAME_HEIGHT       = int(os.environ.get("GAME_HEIGHT", "1080"))
ROBOT_TEMPLATE_PATH = Path(__file__).resolve().with_name("robot.png")
ROBOT_SEARCH_TOP = float(os.environ.get("ROBOT_SEARCH_TOP", "0.55"))
ROBOT_MATCH_THRESHOLD = float(os.environ.get("ROBOT_MATCH_THRESHOLD", "0.45"))
ROBOT_MATCH_SCALES = tuple(float(v) for v in os.environ.get("ROBOT_MATCH_SCALES", "0.35,0.45,0.55,0.65,0.75").split(","))
ROBOT_MAX_STALE_SEC = float(os.environ.get("ROBOT_MAX_STALE_SEC", "1.5"))
ROBOT_MIN_AREA = int(os.environ.get("ROBOT_MIN_AREA", "1200"))
ROBOT_BOX_HOLD_SEC = float(os.environ.get("ROBOT_BOX_HOLD_SEC", "0.6"))
ROBOT_SMOOTH_ALPHA = float(os.environ.get("ROBOT_SMOOTH_ALPHA", "0.18"))
ROBOT_MAX_JUMP_PX = float(os.environ.get("ROBOT_MAX_JUMP_PX", "260"))
ROBOT_CENTER_Y_OFFSET_PX = float(os.environ.get("ROBOT_CENTER_Y_OFFSET_PX", "-380"))
ROBOT_ORANGE_LOWER = np.array([5, 60, 60])
ROBOT_ORANGE_UPPER = np.array([28, 255, 255])
ROBOT_BLUE_LOWER = np.array([90, 55, 35])
ROBOT_BLUE_UPPER = np.array([135, 255, 255])

# Neon pink sticky-note HSV ranges (OpenCV: H 0-180, S/V 0-255)
PINK_LOWER_1 = np.array([138,  55,  55])
PINK_UPPER_1 = np.array([179, 255, 255])
PINK_LOWER_2 = np.array([138,  55,  55])  # same — no wrap-around needed for neon pink
PINK_UPPER_2 = np.array([179, 255, 255])

MIN_CONTOUR_AREA = 800    # px² — ignore small noise
MAX_CONTOUR_AREA = 60_000 # px² — ignore huge blobs

DEBUG_MODE = False

# ── Shared state (GIL-safe for simple list replacement) ───────────────────────
detected_notes: list[dict] = []
external_notes: list[dict] = []
latest_balls: list[dict] = []
latest_ball_update_at = 0.0
latest_camera_preview: bytes | None = None
latest_camera_preview_at = 0.0
detected_robot: dict | None = None
detected_robot_at = 0.0
homography: np.ndarray | None = None
state_lock = threading.Lock()
robot_template: dict | None = None
robot_calibration: dict | None = None
last_robot_box: dict | None = None
latest_browser_state: dict = {"balls": [], "vacuum_on": False}
latest_browser_state_at = 0.0
latest_game_state: dict = {}
latest_game_state_at = 0.0


def snapshot_notes() -> list[dict]:
    with state_lock:
        return [*detected_notes, *external_notes]


def snapshot_balls() -> tuple[list[dict], float]:
    with state_lock:
        return [*latest_balls], latest_ball_update_at


def snapshot_browser_state() -> dict:
    with state_lock:
        return {
            "balls": [*latest_balls],
            "vacuum_on": bool(latest_browser_state.get("vacuum_on", False)),
            "updated_at": latest_browser_state_at,
            "ball_update_at": latest_ball_update_at,
        }


def snapshot_game_state() -> dict:
    with state_lock:
        return dict(latest_game_state)


def snapshot_camera_preview() -> tuple[bytes | None, float]:
    with state_lock:
        return latest_camera_preview, latest_camera_preview_at


def snapshot_robot() -> dict | None:
    with state_lock:
        return None if detected_robot is None else dict(detected_robot)


def snapshot_robot_age() -> float:
    with state_lock:
        return time.time() - detected_robot_at if detected_robot_at else float("inf")


def update_external_notes(notes: list[dict]) -> None:
    global external_notes
    with state_lock:
        external_notes = list(notes)


def update_latest_balls(balls: list[dict], timestamp: float | None = None) -> None:
    global latest_balls, latest_ball_update_at
    with state_lock:
        latest_balls = list(balls)
        latest_ball_update_at = timestamp if timestamp is not None else time.time()


def update_browser_state(state: dict) -> None:
    global latest_browser_state, latest_browser_state_at
    with state_lock:
        latest_browser_state = dict(state)
        latest_browser_state_at = time.time()


def update_game_state(state: dict) -> None:
    global latest_game_state, latest_game_state_at
    with state_lock:
        latest_game_state = dict(state)
        latest_game_state_at = time.time()


def update_camera_preview(frame: np.ndarray) -> None:
    global latest_camera_preview, latest_camera_preview_at
    preview = frame
    height, width = frame.shape[:2]
    target_width = 480
    if width > target_width:
        target_height = max(1, int(height * (target_width / width)))
        preview = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        return
    with state_lock:
        latest_camera_preview = encoded.tobytes()
        latest_camera_preview_at = time.time()


def update_robot(robot: dict | None) -> None:
    global detected_robot, detected_robot_at
    with state_lock:
        detected_robot = None if robot is None else dict(robot)
        if robot is not None:
            detected_robot_at = time.time()


def load_robot_calibration() -> dict | None:
    if not os.path.exists(ROBOT_CALIBRATION_FILE):
        print(f"[robot] no robot calibration found at {ROBOT_CALIBRATION_FILE}")
        return None

    data = np.load(ROBOT_CALIBRATION_FILE)
    points = data["points"].astype(np.float32) if "points" in data else None
    width = float(data["width"]) if "width" in data else None
    height = float(data["height"]) if "height" in data else None
    center = data["center"].astype(np.float32) if "center" in data else None
    center_anchor = data["center_anchor"].astype(np.float32) if "center_anchor" in data else None
    H = data["H"] if "H" in data else None
    camera_index = int(data["camera_index"]) if "camera_index" in data else None

    if width is None or height is None or center is None or points is None:
        print(f"[robot] calibration file {ROBOT_CALIBRATION_FILE} is missing fields")
        return None

    bottom_y = float(data["bottom_y"]) if "bottom_y" in data else float(np.max(points[:, 1]))
    top_y = float(data["top_y"]) if "top_y" in data else float(np.min(points[:, 1]))
    anchor_y = float(center_anchor[1]) if center_anchor is not None else float(center[1])

    print(f"[robot] Loaded calibration from {ROBOT_CALIBRATION_FILE}")
    return {
        "points": points,
        "width": width,
        "height": height,
        "center": center,
        "center_anchor": center_anchor,
        "anchor_y": anchor_y,
        "bottom_y": bottom_y,
        "top_y": top_y,
        "homography": H,
        "camera_index": camera_index,
        "half_width": width / 2.0,
        "half_height": height / 2.0,
    }


def normalize_robot_box(box: dict | None, calibration: dict | None) -> dict | None:
    if box is None:
        return None
    if calibration is None:
        return dict(box)

    frame_w = float(box.get("frame_width", GAME_WIDTH))
    center_x = float(box["x"]) + float(box["width"]) / 2.0

    # Keep the vertical geometry fixed on the rail using the fifth calibration
    # click if available. Fall back to the old center/bottom geometry.
    if "anchor_y" in calibration:
        fixed_center_y = float(calibration["anchor_y"])
    elif "center" in calibration:
        fixed_center_y = float(calibration["center"][1])
    elif "bottom_y" in calibration:
        fixed_center_y = float(calibration["bottom_y"]) - float(calibration["height"]) / 2.0
    else:
        fixed_center_y = float(box["y"]) + float(box["height"]) / 2.0
    fixed_center_y -= ROBOT_CENTER_Y_OFFSET_PX

    width = float(calibration["width"])
    height = float(calibration["height"])

    prev = snapshot_robot()
    if prev is not None and snapshot_robot_age() <= ROBOT_BOX_HOLD_SEC:
        prev_center_x = float(prev.get("center_x", float(prev["x"]) + float(prev["width"]) / 2.0))
        delta_x = center_x - prev_center_x
        if abs(delta_x) > ROBOT_MAX_JUMP_PX:
            center_x = prev_center_x
        else:
            center_x = prev_center_x + (delta_x * ROBOT_SMOOTH_ALPHA)

    half_w = width / 2.0
    half_h = height / 2.0
    center_x = max(half_w, min(frame_w - half_w, center_x))

    normalized = dict(box)
    normalized.update({
        "center_x": center_x,
        "center_y": fixed_center_y,
        "x": center_x - width / 2.0,
        "y": fixed_center_y - height / 2.0,
        "width": width,
        "height": height,
        "calibrated": True,
    })
    return normalized


def load_robot_template() -> dict | None:
    if not ROBOT_TEMPLATE_PATH.exists():
        print(f"[robot] reference image not found: {ROBOT_TEMPLATE_PATH}")
        return None

    template = cv2.imread(str(ROBOT_TEMPLATE_PATH), cv2.IMREAD_COLOR)
    if template is None:
        print(f"[robot] failed to load reference image: {ROBOT_TEMPLATE_PATH}")
        return None

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.GaussianBlur(template_gray, (3, 3), 0)
    template_edges = cv2.Canny(template_gray, 50, 150)
    return {
        "gray": template_gray,
        "edges": template_edges,
        "width": template_edges.shape[1],
        "height": template_edges.shape[0],
        "path": str(ROBOT_TEMPLATE_PATH),
    }


def contour_bbox_from_mask(mask: np.ndarray, frame_top: int = 0) -> dict | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < ROBOT_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        score = float(area / max(1, w * h))
        candidate = {
            "score": score,
            "x": float(x),
            "y": float(y + frame_top),
            "width": float(w),
            "height": float(h),
            "frame_width": float(mask.shape[1]),
            "frame_height": float(mask.shape[0] + frame_top),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def detect_robot(frame: np.ndarray, template: dict | None) -> dict | None:
    frame_h, frame_w = frame.shape[:2]
    roi_top = int(frame_h * ROBOT_SEARCH_TOP)
    roi = frame[roi_top:, :]
    if roi.size == 0:
        return None

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(roi_hsv, ROBOT_ORANGE_LOWER, ROBOT_ORANGE_UPPER)
    blue_mask = cv2.inRange(roi_hsv, ROBOT_BLUE_LOWER, ROBOT_BLUE_UPPER)
    color_mask = cv2.bitwise_or(orange_mask, blue_mask)
    kernel = np.ones((7, 7), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_box = contour_bbox_from_mask(color_mask, roi_top)
    if color_box is not None:
        return color_box

    if template is None:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    roi_edges = cv2.Canny(roi_gray, 50, 150)

    best = None
    for scale in ROBOT_MATCH_SCALES:
        tw = max(16, int(template["width"] * scale))
        th = max(16, int(template["height"] * scale))
        if tw >= roi_edges.shape[1] or th >= roi_edges.shape[0]:
            continue

        resized = cv2.resize(
            template["edges"],
            (tw, th),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        result = cv2.matchTemplate(roi_edges, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if best is None or max_val > best["score"]:
            best = {
                "score": float(max_val),
                "x": float(max_loc[0]),
                "y": float(max_loc[1] + roi_top),
                "width": float(tw),
                "height": float(th),
                "frame_width": float(frame_w),
                "frame_height": float(frame_h),
            }

    if best is None or best["score"] < ROBOT_MATCH_THRESHOLD:
        return None

    return best


def camera_robot_status_snapshot() -> dict:
    robot = snapshot_robot()
    if robot is None:
        return {"status": "missing", "age": snapshot_robot_age()}
    payload = dict(robot)
    payload["status"] = "ok"
    payload["age"] = snapshot_robot_age()
    if robot_calibration is not None:
        payload["calibrated"] = True
    return payload


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_pink_notes(frame: np.ndarray, H: np.ndarray | None) -> list[dict]:
    """
    Return a list of dicts {x, y, width, height, angle} in screen coordinates.
    If H (homography) is None, returns raw camera coordinates.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1),
        cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2),
    )

    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    notes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA):
            continue

        rect              = cv2.minAreaRect(cnt)
        center, (w, h), _ = rect

        if H is not None:
            # Transform all four corners through the homography then re-fit
            box    = cv2.boxPoints(rect).reshape(1, -1, 2).astype(np.float32)
            box_t  = cv2.perspectiveTransform(box, H).reshape(-1, 2)
            rect_t = cv2.minAreaRect(box_t)
            cx, cy = rect_t[0]
            rw, rh = rect_t[1]
            angle  = rect_t[2]
        else:
            cx, cy = center
            rw, rh = w, h
            angle  = rect[2]

        notes.append({
            "x":      float(cx),
            "y":      float(cy),
            "width":  float(max(rw, rh)),
            "height": float(min(rw, rh)),
            "angle":  float(angle),
        })

    return notes


# ── Camera loop (runs in background thread) ───────────────────────────────────

def camera_loop() -> None:
    global detected_notes, homography, robot_template, robot_calibration, last_robot_box

    camera_index = CAMERA_INDEX
    if os.path.exists(CALIBRATION_FILE):
        data       = np.load(CALIBRATION_FILE)
        homography = data["H"]
        if "camera_index" in data:
            camera_index = int(data["camera_index"])
        print(f"[camera] Loaded calibration from {CALIBRATION_FILE} (camera {camera_index})")
    else:
        print("[camera] WARNING: no calibration file found — "
              "positions may not align with the projection.")
        print("[camera] Run  uv run calibrate.py  first.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera {camera_index}")
        return

    interval = 1.0 / DETECTION_FPS
    print(f"[camera] Capturing at {DETECTION_FPS} fps (camera {camera_index})")

    robot_template = load_robot_template()
    robot_calibration = load_robot_calibration()
    if robot_template is not None:
        print(f"[robot] Loaded reference template from {robot_template['path']}")

    show_debug = DEBUG_MODE
    if show_debug:
        try:
            cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug", 960, 540)
            print("[camera] Debug window open — press Q to close it")
        except cv2.error as exc:
            show_debug = False
            print(f"[camera] Debug window disabled: {exc}")

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if ret:
            update_camera_preview(frame)
            notes = detect_pink_notes(frame, homography)
            robot_box = detect_robot(frame, robot_template)
            robot_box = normalize_robot_box(robot_box, robot_calibration)
            if robot_box is None and last_robot_box is not None:
                if snapshot_robot_age() <= ROBOT_BOX_HOLD_SEC:
                    robot_box = dict(last_robot_box)
            if robot_box is not None:
                last_robot_box = dict(robot_box)
            with state_lock:
                detected_notes = notes
            update_robot(robot_box)

            if show_debug:
                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.bitwise_or(
                        cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1),
                        cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2),
                    )
                    debug = frame.copy()
                    # Highlight detected mask in cyan
                    debug[mask > 0] = (255, 255, 0)
                    # Draw bounding boxes
                    for note in notes:
                        x1 = int(note["x"])
                        y1 = int(note["y"])
                        x2 = int(x1 + note["width"])
                        y2 = int(y1 + note["height"])
                        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if robot_box is not None:
                        x1 = int(robot_box["x"])
                        y1 = int(robot_box["y"])
                        x2 = int(x1 + robot_box["width"])
                        y2 = int(y1 + robot_box["height"])
                        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(
                            debug,
                            f"robot {robot_box['score']:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2,
                        )
                    cv2.putText(debug, f"notes: {len(notes)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if robot_box is not None:
                        cv2.putText(
                            debug,
                            f"robot: {robot_box['x']:.0f},{robot_box['y']:.0f}",
                            (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
                        )
                    cv2.imshow("Debug", debug)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        show_debug = False
                        cv2.destroyAllWindows()
                except cv2.error as exc:
                    show_debug = False
                    print(f"[camera] Debug rendering disabled: {exc}")

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))

    cap.release()


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(websocket) -> None:
    addr = websocket.remote_address
    print(f"[ws] Browser connected: {addr}")
    async def send_loop() -> None:
        try:
            while True:
                payload = {
                    "type": "snapshot",
                    "notes": snapshot_notes(),
                    "browser": snapshot_browser_state(),
                    "camera_robot": camera_robot_status_snapshot(),
                    "game": snapshot_game_state(),
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(1.0 / DETECTION_FPS)
        except websockets.exceptions.ConnectionClosed:
            return

    sender_task = asyncio.create_task(send_loop())
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                data_type = data.get("type")
                if data_type in {"browser_state", "state"}:
                    if "balls" in data:
                        update_latest_balls(data.get("balls", []))
                    update_browser_state({
                        "balls": data.get("balls", []),
                        "vacuum_on": bool(data.get("vacuum_on", False)),
                        "updated_at": data.get("timestamp_ms", time.time() * 1000.0),
                    })
                elif data_type in {"game_state", "capture"}:
                    update_game_state(data)
                elif "notes" in data:
                    update_external_notes(data.get("notes", []))
                elif "boxes" in data:
                    update_external_notes(data.get("boxes", []))
    except websockets.exceptions.ConnectionClosed:
        print(f"[ws] Browser disconnected: {addr}")
    finally:
        sender_task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await sender_task

class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        if self.path in {"/", "/preview", "/preview/"}:
            body = build_preview_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/preview.jpg"):
            body, _ = snapshot_camera_preview()
            if body is None:
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(b"Preview not ready")
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - stdlib signature
        return


def build_preview_html() -> bytes:
    if not PREVIEW_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"missing preview template: {PREVIEW_TEMPLATE_PATH}")

    template = PREVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace("__PREVIEW_PORT__", str(PREVIEW_PORT))
    return template.encode("utf-8")


def start_preview_server() -> None:
    try:
        server = ThreadingHTTPServer((WEBSOCKET_HOST, PREVIEW_PORT), PreviewHandler)
    except OSError as exc:
        print(f"[preview] failed to start HTTP server on {PREVIEW_PORT}: {exc}")
        return
    print(f"[preview] Preview available at http://127.0.0.1:{PREVIEW_PORT}/")
    server.serve_forever()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (omit to use calibrate.py's choice or default 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Show live camera window with detection overlay")
    args = parser.parse_args()
    if args.camera is not None:
        global CAMERA_INDEX
        CAMERA_INDEX = args.camera  # overrides calibration file
    global DEBUG_MODE
    DEBUG_MODE = args.debug

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    preview_thread = threading.Thread(target=start_preview_server, daemon=True)
    preview_thread.start()

    print(f"[ws] Listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print("[ws] Open index.html in your browser, press F for fullscreen.")
    print(f"[preview] Open http://127.0.0.1:{PREVIEW_PORT}/ for a tiny camera preview.")

    try:
        async with websockets.serve(ws_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
            await asyncio.Future()  # run forever
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Robot calibration tool for StickyBounce.

How it works:
  1. Open the camera view showing the robot at a representative position.
  2. Click the four corners of the robot in order:
       1. top-left
       2. top-right
       3. bottom-right
       4. bottom-left
  3. Click the center of the robot as the fifth click to lock the robot's
     vertical anchor.
  4. The robot geometry is saved to robot_calibration.npz and used by server.py.

Usage:
    uv run calibrate_robot.py              # list cameras, then pick one
    uv run calibrate_robot.py --camera 1   # use camera index 1 directly
"""

import argparse
import math
import sys

import cv2
import numpy as np

ROBOT_CALIBRATION_FILE = "robot_calibration.npz"
DISPLAY_W = 960
DISPLAY_H = 540

WIN = "Robot Calibration"
camera_points: list[list[float]] = []
current_frame_size: tuple[int, int] | None = None


def list_cameras(max_test: int = 6) -> list[int]:
    found = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


def pick_camera() -> int:
    print("Scanning for cameras…")
    indices = list_cameras()
    if not indices:
        print("ERROR: no cameras found.")
        sys.exit(1)
    print(f"Found cameras: {indices}")
    if len(indices) == 1:
        print(f"Using camera {indices[0]} (only one available).")
        return indices[0]
    choice = input(f"Enter camera index to use {indices}: ").strip()
    try:
        idx = int(choice)
        if idx not in indices:
            raise ValueError
        return idx
    except ValueError:
        print(f"Invalid choice — using {indices[0]}")
        return indices[0]


def on_click(event, x, y, flags, param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or len(camera_points) >= 5:
        return

    if current_frame_size is None:
        camera_points.append([float(x), float(y)])
        print(f"  Point {len(camera_points)}/5 captured: ({x}, {y})")
        return

    frame_w, frame_h = current_frame_size
    scaled_x = float(x) * float(frame_w) / float(DISPLAY_W)
    scaled_y = float(y) * float(frame_h) / float(DISPLAY_H)
    camera_points.append([scaled_x, scaled_y])
    print(f"  Point {len(camera_points)}/5 captured: ({x}, {y}) -> ({scaled_x:.1f}, {scaled_y:.1f})")


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def compute_geometry(points: np.ndarray) -> dict:
    tl, tr, br, bl, center_anchor = points
    width = 0.5 * (dist(tl, tr) + dist(bl, br))
    height = 0.5 * (dist(tl, bl) + dist(tr, br))
    center = np.array([
        0.5 * (float(tl[0]) + float(tr[0])),
        float(center_anchor[1]),
    ], dtype=np.float32)
    top_y = 0.5 * (float(tl[1]) + float(tr[1]))
    bottom_y = 0.5 * (float(br[1]) + float(bl[1]))
    plane = np.array([
        [0.0, 0.0],
        [width, 0.0],
        [width, height],
        [0.0, height],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(plane, points)
    return {
        "points": points.astype(np.float32),
        "plane": plane,
        "H": H,
        "width": np.float32(width),
        "height": np.float32(height),
        "center": center.astype(np.float32),
        "center_anchor": center_anchor.astype(np.float32),
        "top_y": np.float32(top_y),
        "bottom_y": np.float32(bottom_y),
    }


def draw_overlay(frame: np.ndarray, points: list[list[float]]) -> np.ndarray:
    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
    frame_h, frame_w = frame.shape[:2]
    scale_x = DISPLAY_W / float(frame_w)
    scale_y = DISPLAY_H / float(frame_h)
    for i, pt in enumerate(points):
        px = int(pt[0] * scale_x)
        py = int(pt[1] * scale_y)
        cv2.circle(display, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(display, str(i + 1), (px + 12, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return display


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (omit to auto-pick)")
    args = parser.parse_args()

    camera_index = args.camera if args.camera is not None else pick_camera()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}")
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(WIN, on_click)

    print("\n=== Robot Calibration ===")
    print("Click the four robot corners in order:")
    print("  1. top-left")
    print("  2. top-right")
    print("  3. bottom-right")
    print("  4. bottom-left")
    print("  5. center anchor (sets the robot's Y height)")
    print("Press Q to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        global current_frame_size
        current_frame_size = (frame.shape[1], frame.shape[0])

        display = draw_overlay(frame, camera_points)
        n = len(camera_points)
        if n < 4:
            msg = f"LEFT-CLICK corner {n + 1}/4"
            color = (0, 200, 255)
        elif n == 4:
            msg = "LEFT-CLICK center anchor 5/5"
            color = (255, 170, 0)
        else:
            msg = "All 5 done! Saving..."
            color = (0, 255, 0)

        cv2.rectangle(display, (0, 0), (display.shape[1], 55), (0, 0, 0), -1)
        cv2.putText(display, msg, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("Calibration cancelled.")
            break

        if len(camera_points) == 5:
            points = np.array(camera_points, dtype=np.float32)
            geom = compute_geometry(points)
            np.savez(
                ROBOT_CALIBRATION_FILE,
                points=geom["points"],
                plane=geom["plane"],
                H=geom["H"],
                width=geom["width"],
                height=geom["height"],
                center=geom["center"],
                center_anchor=geom["center_anchor"],
                top_y=geom["top_y"],
                bottom_y=geom["bottom_y"],
                camera_index=np.array(camera_index),
            )
            print(f"\nRobot calibration saved to {ROBOT_CALIBRATION_FILE}")
            print("Run: uv run server.py")
            cv2.waitKey(1500)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

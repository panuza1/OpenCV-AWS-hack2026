"""HSV-based tracking for one orange ball in fixed-camera video."""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


HSV_LOWER = np.array((5, 100, 100), dtype=np.uint8)
HSV_UPPER = np.array((25, 255, 255), dtype=np.uint8)
MIN_AREA_PX2 = 100.0
MAX_AREA_RATIO = 0.1
MAX_JUMP_PX_PER_FRAME = 50.0
_KERNEL = np.ones((5, 5), dtype=np.uint8)
FIELDS = ("frame", "time_s", "x_px", "y_px", "area_px2", "valid")


class VideoError(ValueError):
    """Raised when an input cannot produce a valid video track."""


def _detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = frame.shape[0] * frame.shape[1] * MAX_AREA_RATIO
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA_PX2 <= area <= max_area:
            moments = cv2.moments(contour)
            if moments["m00"]:
                candidates.append(
                    (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], area)
                )
    return max(candidates, key=lambda candidate: candidate[2], default=None)


def _video_fps(capture, fps_override):
    fps = float(fps_override) if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise VideoError("Video FPS is unavailable; provide a positive --fps override")
    return fps


def track_video(video_path, fps_override=None):
    if not cv2.__version__.startswith("5."):
        raise RuntimeError(f"PhysicsEye requires OpenCV 5, found {cv2.__version__}")

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")

    try:
        fps = _video_fps(capture, fps_override)
        observations = []
        last_valid = None
        frame_index = 0
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            detection = _detect_ball(frame)
            valid = detection is not None
            if valid and last_valid is not None:
                x_px, y_px, _ = detection
                elapsed_frames = frame_index - last_valid[0]
                jump = math.hypot(x_px - last_valid[1], y_px - last_valid[2])
                valid = jump <= MAX_JUMP_PX_PER_FRAME * elapsed_frames

            if valid:
                x_px, y_px, area_px2 = detection
                last_valid = (frame_index, x_px, y_px)
                observation = {
                    "frame": frame_index,
                    "time_s": frame_index / fps,
                    "x_px": x_px,
                    "y_px": y_px,
                    "area_px2": area_px2,
                    "valid": True,
                }
            else:
                observation = {
                    "frame": frame_index,
                    "time_s": frame_index / fps,
                    "x_px": None,
                    "y_px": None,
                    "area_px2": None,
                    "valid": False,
                }
            observations.append(observation)
            frame_index += 1
    finally:
        capture.release()

    if not observations:
        raise VideoError(f"Video contains no decodable frames: {video_path}")
    return observations


def write_track(observations, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_stem}.json")
    csv_path = Path(f"{output_stem}.csv")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(observations, stream, indent=2)
        stream.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(observations)
    return json_path, csv_path


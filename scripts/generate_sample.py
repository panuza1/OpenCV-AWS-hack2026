#!/usr/bin/env python3
"""Generate deterministic PhysicsEye sample videos and ground truth."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


FPS = 30.0
FRAME_COUNT = 150
FRAME_SIZE = (320, 240)
BALL_RADIUS_PX = 12
SCENARIOS = {
    "constant_speed": {"initial_speed_px_s": 40.0, "deceleration_px_s2": 0.0},
    "slowing_down": {"initial_speed_px_s": 60.0, "deceleration_px_s2": 12.0},
}
CODECS = (("mp4v", ".mp4"), ("MJPG", ".avi"))


def _ground_truth(initial_speed_px_s, deceleration_px_s2):
    rows = []
    for frame in range(FRAME_COUNT):
        time_s = frame / FPS
        x_px = round(40.0 + initial_speed_px_s * time_s - 0.5 * deceleration_px_s2 * time_s**2)
        rows.append({"frame": frame, "time_s": time_s, "x_px": x_px, "y_px": 120})
    return rows


def _verify_video(path, expected_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    decoded_frames = 0
    while capture.read()[0]:
        decoded_frames += 1
    capture.release()
    return (
        reported_frames == expected_frames
        and decoded_frames == expected_frames
        and abs(reported_fps - FPS) < 0.01
    )


def _write_video(output_dir, name, rows):
    for codec, suffix in CODECS:
        path = output_dir / f"{name}{suffix}"
        path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), FPS, FRAME_SIZE
        )
        if not writer.isOpened():
            writer.release()
            continue
        for row in rows:
            frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 18, dtype=np.uint8)
            cv2.circle(
                frame,
                (row["x_px"], row["y_px"]),
                BALL_RADIUS_PX,
                (0, 140, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()
        if _verify_video(path, len(rows)):
            return path, codec
        path.unlink(missing_ok=True)
    raise RuntimeError(f"No tested video codec could write and reopen {name}")


def generate_samples(output_dir):
    if not cv2.__version__.startswith("5."):
        raise RuntimeError(f"PhysicsEye requires OpenCV 5, found {cv2.__version__}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, motion in SCENARIOS.items():
        rows = _ground_truth(**motion)
        video_path, codec = _write_video(output_dir, name, rows)
        csv_path = output_dir / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("frame", "time_s", "x_px", "y_px"))
            writer.writeheader()
            writer.writerows(rows)
        generated.append(
            {
                "scenario": name,
                "video": str(video_path),
                "ground_truth": str(csv_path),
                "codec": codec,
                "fps": FPS,
                "frames": len(rows),
            }
        )
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("examples/generated"))
    args = parser.parse_args()
    print(json.dumps(generate_samples(args.output_dir), indent=2))


if __name__ == "__main__":
    main()


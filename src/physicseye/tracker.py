"""HSV and user-initialized ROI tracking for one ball in fixed-camera video."""

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
ROI_CONFIDENCE_THRESHOLD = 0.55
ROI_TEMPLATE_UPDATE_THRESHOLD = 0.7
ROI_SCALES = (0.88, 0.94, 1.0, 1.06, 1.12)
ROI_MAX_TRACKING_DIMENSION = 960
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


def _invalid_roi_observation(frame_index, fps, status, confidence=0.0):
    return {
        "frame": frame_index,
        "time_s": frame_index / fps,
        "x_px": None,
        "y_px": None,
        "area_px2": None,
        "valid": False,
        "confidence": max(0.0, float(confidence)),
        "status": status,
        "bbox": None,
    }


def _roi_observation(frame_index, fps, bbox, confidence, status, coordinate_scale=1.0):
    x, y, width, height = bbox
    x *= coordinate_scale
    y *= coordinate_scale
    width *= coordinate_scale
    height *= coordinate_scale
    return {
        "frame": frame_index,
        "time_s": frame_index / fps,
        "x_px": x + width / 2.0,
        "y_px": y + height / 2.0,
        "area_px2": float(width * height),
        "valid": True,
        "confidence": float(confidence),
        "status": status,
        "bbox": [x, y, width, height],
    }


def _validate_roi(frame, roi):
    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise VideoError("ROI must contain x, y, width, and height")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in roi):
        raise VideoError("ROI values must be numbers")
    x, y, width, height = (round(float(value)) for value in roi)
    if width < 8 or height < 8:
        raise VideoError("ROI width and height must be at least 8 pixels")
    if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise VideoError("ROI must stay inside the initialization frame")
    return x, y, width, height


def _gray(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def _match_roi(frame_gray, template, bbox, velocity, missed_frames, initial_size):
    x, y, width, height = bbox
    center_x = x + width / 2.0 + velocity[0] * (missed_frames + 1)
    center_y = y + height / 2.0 + velocity[1] * (missed_frames + 1)
    radius = max(60.0, 2.0 * max(width, height)) * min(
        1.0 + 0.35 * missed_frames, 2.0
    )
    left = max(0, round(center_x - radius))
    top = max(0, round(center_y - radius))
    right = min(frame_gray.shape[1], round(center_x + radius))
    bottom = min(frame_gray.shape[0], round(center_y + radius))
    search = frame_gray[top:bottom, left:right]

    best = None
    for scale in ROI_SCALES:
        candidate_width = max(8, round(template.shape[1] * scale))
        candidate_height = max(8, round(template.shape[0] * scale))
        if not (
            0.45 * initial_size[0] <= candidate_width <= 2.5 * initial_size[0]
            and 0.45 * initial_size[1] <= candidate_height <= 2.5 * initial_size[1]
        ):
            continue
        if candidate_width > search.shape[1] or candidate_height > search.shape[0]:
            continue
        candidate = cv2.resize(template, (candidate_width, candidate_height))
        scores = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        match_x = left + location[0]
        match_y = top + location[1]
        match_center_x = match_x + candidate_width / 2.0
        match_center_y = match_y + candidate_height / 2.0
        distance = math.hypot(match_center_x - center_x, match_center_y - center_y)
        rank = score - 0.03 * abs(math.log(scale)) - 0.04 * distance / max(radius, 1.0)
        if best is None or rank > best[0]:
            best = (
                rank,
                float(score),
                (match_x, match_y, candidate_width, candidate_height),
                candidate,
                distance,
            )
    return best


def _track_roi_frame(frame, frame_index, fps, state, confidence_threshold):
    frame_gray = _gray(frame)
    if frame_index < state["initial_frame"]:
        return _invalid_roi_observation(frame_index, fps, "before_initialization")
    if frame_index == state["initial_frame"]:
        bbox = _validate_roi(frame, state["roi"])
        x, y, width, height = bbox
        state.update({
            "bbox": bbox,
            "template": frame_gray[y:y + height, x:x + width].copy(),
            "initial_size": (width, height),
            "valid_centers": [(frame_index, x + width / 2.0, y + height / 2.0)],
            "missed_frames": 0,
        })
        return _roi_observation(
            frame_index, fps, bbox, 1.0, "initialized", state["coordinate_scale"]
        )
    if state.get("left_frame"):
        return _invalid_roi_observation(frame_index, fps, "target_left_frame")

    centers = state["valid_centers"]
    if len(centers) >= 2:
        first, second = centers[-2:]
        elapsed = second[0] - first[0]
        velocity = ((second[1] - first[1]) / elapsed, (second[2] - first[2]) / elapsed)
    else:
        velocity = (0.0, 0.0)
    match = _match_roi(
        frame_gray,
        state["template"],
        state["bbox"],
        velocity,
        state["missed_frames"],
        state["initial_size"],
    )
    if match is None:
        state["missed_frames"] += 1
        return _invalid_roi_observation(frame_index, fps, "no_search_candidate")

    _, confidence, bbox, resized_template, prediction_error = match
    previous = state["bbox"]
    elapsed = state["missed_frames"] + 1
    jump = math.hypot(
        bbox[0] + bbox[2] / 2.0 - (previous[0] + previous[2] / 2.0),
        bbox[1] + bbox[3] / 2.0 - (previous[1] + previous[3] / 2.0),
    )
    max_jump = max(75.0, 1.5 * max(previous[2], previous[3])) * elapsed
    max_prediction_error = max(50.0, 1.25 * max(previous[2], previous[3])) * (
        min(1.0 + 0.25 * state["missed_frames"], 2.0)
    )
    if (
        confidence < confidence_threshold
        or jump > max_jump
        or prediction_error > max_prediction_error
    ):
        state["missed_frames"] += 1
        reason = (
            "low_confidence" if confidence < confidence_threshold
            else "position_discontinuity"
        )
        return _invalid_roi_observation(frame_index, fps, reason, confidence)

    x, y, width, height = bbox
    if x <= 1 or y <= 1 or x + width >= frame.shape[1] - 1 or y + height >= frame.shape[0] - 1:
        state["left_frame"] = True
        return _invalid_roi_observation(frame_index, fps, "target_at_frame_edge", confidence)
    patch = frame_gray[y:y + height, x:x + width]
    state["template"] = (
        cv2.addWeighted(resized_template, 0.75, patch, 0.25, 0)
        if confidence >= ROI_TEMPLATE_UPDATE_THRESHOLD else resized_template
    )
    state["bbox"] = bbox
    state["valid_centers"].append((frame_index, x + width / 2.0, y + height / 2.0))
    state["valid_centers"] = state["valid_centers"][-2:]
    state["missed_frames"] = 0
    return _roi_observation(
        frame_index, fps, bbox, confidence, "tracked", state["coordinate_scale"]
    )


def track_video(
    video_path,
    fps_override=None,
    *,
    mode="hsv",
    initial_frame=0,
    roi=None,
    confidence_threshold=ROI_CONFIDENCE_THRESHOLD,
):
    if not cv2.__version__.startswith("5."):
        raise RuntimeError(f"PhysicsEye requires OpenCV 5, found {cv2.__version__}")

    if mode not in ("hsv", "roi_template"):
        raise VideoError(f"Unknown tracking mode: {mode}")
    if mode == "roi_template":
        if not isinstance(initial_frame, int) or isinstance(initial_frame, bool) or initial_frame < 0:
            raise VideoError("initial_frame must be a nonnegative integer")
        if not isinstance(confidence_threshold, (int, float)) or not 0 < confidence_threshold <= 1:
            raise VideoError("confidence_threshold must be between 0 and 1")

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")

    try:
        fps = _video_fps(capture, fps_override)
        observations = []
        last_valid = None
        roi_state = {"initial_frame": initial_frame, "roi": roi}
        frame_index = 0
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            if mode == "roi_template":
                if "working_scale" not in roi_state:
                    working_scale = min(
                        1.0, ROI_MAX_TRACKING_DIMENSION / max(frame.shape[:2])
                    )
                    roi_state["working_scale"] = working_scale
                    roi_state["coordinate_scale"] = 1.0 / working_scale
                    if roi is not None:
                        roi_state["roi"] = [value * working_scale for value in roi]
                working_scale = roi_state["working_scale"]
                working_frame = (
                    cv2.resize(frame, None, fx=working_scale, fy=working_scale)
                    if working_scale < 1.0 else frame
                )
                observations.append(
                    _track_roi_frame(
                        working_frame, frame_index, fps, roi_state, confidence_threshold
                    )
                )
                frame_index += 1
                continue
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
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(observations)
    return json_path, csv_path


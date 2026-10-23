"""Small, testable data helpers for the local Streamlit dashboard."""

import csv
import io
import json
import math
import tempfile
from pathlib import Path

import cv2

from physicseye.physics import predict_trajectory
from physicseye.tracker import track_video


MODEL_META = {
    "constant_velocity": ("Constant Velocity", "#1f77b4", (180, 95, 20)),
    "fitted_deceleration": ("Refitted Deceleration", "#ff7f0e", (14, 127, 255)),
    "frozen_deceleration": ("Frozen Deceleration", "#9467bd", (189, 103, 148)),
    "adaptive_deceleration": ("Adaptive Deceleration", "#2ca02c", (44, 160, 44)),
}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_DURATION_S = 60.0


def validate_video_upload(filename, data):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Upload an MP4, AVI, or MOV video")
    if not data:
        raise ValueError("The uploaded video is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded video exceeds the 100 MB limit")
    return suffix


def track_video_bytes(filename, data, **tracker_options):
    suffix = validate_video_upload(filename, data)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(data)
        _video_metadata(path)
        return track_video(path, **tracker_options)


def _video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video")
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video metadata is incomplete")
    duration_s = frame_count / fps
    if duration_s > MAX_VIDEO_DURATION_S:
        raise ValueError(f"Video exceeds the {MAX_VIDEO_DURATION_S:g}-second limit")
    return {
        "width": width, "height": height, "frame_count": frame_count,
        "fps": fps, "duration_s": duration_s,
    }


def video_metadata_bytes(filename, data):
    suffix = validate_video_upload(filename, data)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"video{suffix}"
        path.write_bytes(data)
        return _video_metadata(path)


def render_roi_preview_bytes(filename, data, frame_index, roi):
    suffix = validate_video_upload(filename, data)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"video{suffix}"
        path.write_bytes(data)
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            readable, frame = capture.read()
        finally:
            capture.release()
    if not readable:
        raise ValueError(f"Could not decode initialization frame {frame_index}")
    x, y, width, height = (round(value) for value in roi)
    if width < 8 or height < 8 or x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("ROI must stay inside the initialization frame")
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), max(2, frame.shape[1] // 500))
    cv2.putText(frame, "selected target", (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, frame.shape[1] / 2500), (0, 255, 0), max(1, frame.shape[1] // 1000), cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def prepare_analysis(observations, cutoff_time_s, horizons_s, prediction_result=None):
    """Predict from a prefix, then join later observations for evaluation only."""
    history = [point for point in observations if point["time_s"] <= cutoff_time_s]
    result = (
        prediction_result
        if prediction_result is not None
        else predict_trajectory(history, cutoff_time_s, horizons_s)
    )
    analysis = {
        "cutoff_time_s": cutoff_time_s,
        "horizons_s": sorted(horizons_s),
        "history": history,
        "prediction": result,
        "comparisons": [],
    }
    if result["status"] != "ok":
        return analysis

    later = [
        point for point in observations
        if point["time_s"] > cutoff_time_s and point.get("valid")
    ]
    frame_interval_s = min(
        (second["time_s"] - first["time_s"] for first, second in zip(observations, observations[1:])),
        default=0.0,
    )
    for model_name, model in result["models"].items():
        for prediction in model["predictions"]:
            actual = min(
                later,
                key=lambda point: abs(point["time_s"] - prediction["target_time_s"]),
                default=None,
            )
            if actual and abs(actual["time_s"] - prediction["target_time_s"]) > 1.5 * frame_interval_s:
                actual = None
            error_px = None
            if actual is not None:
                error_px = math.hypot(
                    prediction["x_px"] - actual["x_px"],
                    prediction["y_px"] - actual["y_px"],
                )
            analysis["comparisons"].append(
                {
                    "model": model_name,
                    "model_label": MODEL_META[model_name][0],
                    "horizon_s": prediction["horizon_s"],
                    "target_time_s": prediction["target_time_s"],
                    "predicted_x_px": prediction["x_px"],
                    "predicted_y_px": prediction["y_px"],
                    "actual_time_s": actual["time_s"] if actual else None,
                    "actual_x_px": actual["x_px"] if actual else None,
                    "actual_y_px": actual["y_px"] if actual else None,
                    "error_px": error_px,
                }
            )
    return analysis


def analysis_csv(analysis):
    if not analysis["comparisons"]:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=analysis["comparisons"][0].keys())
    writer.writeheader()
    writer.writerows(analysis["comparisons"])
    return stream.getvalue()


def analysis_json(analysis):
    return json.dumps(analysis, indent=2) + "\n"


def load_benchmark(path=Path("outputs/evaluation.json")):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            result = json.load(stream)
        required = {"configuration", "scenarios", "aggregate_summaries", "acceptance"}
        if not required.issubset(result):
            raise ValueError("benchmark JSON is missing required fields")
        if not result["scenarios"] or not result["aggregate_summaries"]:
            raise ValueError("benchmark JSON contains no results")
        return {"status": "ok", "data": result}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "error", "error": str(error), "data": None}


def render_annotated_frame(video_path, observations, analysis, time_s):
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(time_s * fps)))
        readable, frame = capture.read()
    finally:
        capture.release()
    if not readable:
        raise ValueError(f"Could not decode the frame at {time_s:.2f} s")

    current = min(
        observations,
        key=lambda point: abs(point["time_s"] - time_s),
        default=None,
    )
    if current and "status" in current:
        if current.get("valid") and current.get("bbox"):
            x, y, width, height = map(round, current["bbox"])
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            tracking_text = f'{current["status"]} confidence={current["confidence"]:.2f}'
            color = (0, 255, 0)
        else:
            tracking_text = f'tracking invalid: {current["status"]}'
            color = (0, 0, 255)
        cv2.putText(frame, tracking_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)

    cutoff = analysis["cutoff_time_s"]
    history = [point for point in observations if point["time_s"] <= cutoff and point.get("valid")]
    for first, second in zip(history, history[1:]):
        cv2.line(frame, _point(first), _point(second), (210, 210, 210), 2)
    if history:
        anchor = _point(history[-1])
        cv2.circle(frame, anchor, 5, (255, 255, 255), -1)
        for model_name, model in analysis["prediction"].get("models", {}).items():
            color = MODEL_META[model_name][2]
            previous = anchor
            for prediction in model["predictions"]:
                target = _point(prediction)
                cv2.line(frame, previous, target, color, 2)
                cv2.circle(frame, target, 5, color, -1)
                previous = target

    actual = min(
        (point for point in observations if point.get("valid")),
        key=lambda point: abs(point["time_s"] - time_s),
        default=None,
    )
    if actual and actual["time_s"] > cutoff:
        cv2.circle(frame, _point(actual), 8, (255, 255, 255), 2)
        cv2.putText(frame, "actual (evaluation only)", (_point(actual)[0] + 10, _point(actual)[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"cutoff {cutoff:.2f}s | frame {time_s:.2f}s", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_video_bytes(filename, data, observations, analysis, time_s):
    suffix = validate_video_upload(filename, data)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"video{suffix}"
        path.write_bytes(data)
        return render_annotated_frame(path, observations, analysis, time_s)


def _point(value):
    return round(value["x_px"]), round(value["y_px"])

"""Small, testable data helpers for the local Streamlit dashboard."""

import csv
import io
import json
import math
import shutil
import subprocess
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
MAX_VIDEO_PIXELS = 1920 * 1080


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
    decoded_frame_mib = width * height * 3 / (1024 * 1024)
    if width * height > MAX_VIDEO_PIXELS:
        raise ValueError(
            f"Video resolution {width}x{height} decodes to {decoded_frame_mib:.1f} MiB "
            "per BGR frame and exceeds the 2.1-megapixel limit; "
            "resize to 1920x1080 or smaller"
        )
    duration_s = frame_count / fps
    if duration_s > MAX_VIDEO_DURATION_S:
        raise ValueError(f"Video exceeds the {MAX_VIDEO_DURATION_S:g}-second limit")
    return {
        "width": width, "height": height, "frame_count": frame_count,
        "fps": fps, "duration_s": duration_s,
        "decoded_frame_mib": decoded_frame_mib,
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


def prepare_automatic_predictions(observations, frame_width):
    """Build a bounded forecast timeline from observation prefixes only."""
    if len(observations) < 2:
        return [None] * len(observations)
    frame_interval_s = observations[1]["time_s"] - observations[0]["time_s"]
    step = max(1, round(0.25 / frame_interval_s))
    timeline = []
    latest = None
    for index, observation in enumerate(observations):
        if index % step == 0 and observation["time_s"] >= 0.75:
            history = observations[:index + 1]
            probe = predict_trajectory(history, observation["time_s"], [0.25])
            if probe["status"] == "ok":
                speed = probe["models"]["adaptive_deceleration"]["parameters"]["speed_px_s"]
                horizon_s = 0.25 if speed >= 0.5 * frame_width else 0.5
                horizons = [horizon_s / 3.0, 2.0 * horizon_s / 3.0, horizon_s]
                prediction = predict_trajectory(history, observation["time_s"], horizons)
                if prediction["status"] == "ok":
                    model = prediction["models"]["adaptive_deceleration"]
                    latest_valid = next(
                        (point for point in reversed(history) if point.get("valid")),
                        None,
                    )
                    latest = {
                        "cutoff_frame": index,
                        "cutoff_time_s": observation["time_s"],
                        "horizon_s": horizon_s,
                        "anchor_x_px": latest_valid["x_px"],
                        "anchor_y_px": latest_valid["y_px"],
                        "direction_angle_deg": math.degrees(math.atan2(
                            model["parameters"]["velocity_y_px_s"],
                            model["parameters"]["velocity_x_px_s"],
                        )),
                        "prediction": prediction,
                    }
        timeline.append(latest)
    return timeline


def _transcode_browser_mp4(source_path, output_path, expected_frames, expected_fps):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError(
            "FFmpeg and ffprobe are required for browser-compatible H.264 video; "
            "install the ffmpeg system package"
        )
    conversion = subprocess.run(
        [
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(source_path),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ],
        capture_output=True, text=True,
    )
    if conversion.returncode:
        Path(output_path).unlink(missing_ok=True)
        detail = conversion.stderr.strip() or "unknown FFmpeg error"
        raise ValueError(f"Could not convert comparison video to H.264: {detail}")

    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,avg_frame_rate,nb_read_frames,duration",
            "-of", "json", str(output_path),
        ],
        capture_output=True, text=True,
    )
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        actual_fps = numerator / denominator
        frame_count = int(stream["nb_read_frames"])
        duration_s = float(stream["duration"])
    except (
        KeyError, IndexError, TypeError, ValueError, ZeroDivisionError,
        json.JSONDecodeError,
    ) as error:
        Path(output_path).unlink(missing_ok=True)
        detail = probe.stderr.strip() or str(error)
        raise ValueError(f"Could not verify the H.264 comparison video: {detail}") from error
    expected_duration_s = expected_frames / expected_fps
    if (
        probe.returncode
        or stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or frame_count != expected_frames
        or abs(actual_fps - expected_fps) > 0.01
        or abs(duration_s - expected_duration_s) > 0.02
    ):
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("H.264 comparison video failed codec or timing verification")
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "duration_s": duration_s,
    }


def _comparison_forecast(forecast, observation, trajectory_kind):
    if trajectory_kind == "flight":
        return (
            (forecast["anchor_x_px"], forecast["anchor_y_px"]),
            [
                point for point in forecast["predicted_points"]
                if point["time_s"] > observation["time_s"]
            ],
            f'flight {forecast["direction_angle_deg"]:.1f} deg | fit R2 {forecast["fit_r2"]:.3f}',
        )
    model = forecast["prediction"]["models"]["adaptive_deceleration"]
    return (
        (forecast["anchor_x_px"], forecast["anchor_y_px"]),
        [
            point for point in model["predictions"]
            if point["target_time_s"] > observation["time_s"]
        ],
        f'direction {forecast["direction_angle_deg"]:.1f} deg | horizon {forecast["horizon_s"]:.2f}s',
    )


def render_comparison_video(
    video_path, observations, prediction_timeline, output_path,
    *, trajectory_kind="rolling",
):
    """Encode original and annotated frames side by side without buffering frames."""
    if trajectory_kind not in ("rolling", "flight"):
        raise ValueError(f"Unknown trajectory kind: {trajectory_kind}")
    output_path = Path(output_path)
    intermediate_path = output_path.with_name(f".{output_path.stem}-mpeg4.mp4")
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    render_finished = False
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, 1280 / width, 720 / height)
        render_width = max(2, round(width * scale))
        render_height = max(2, round(height * scale))
        render_width -= render_width % 2
        render_height -= render_height % 2
        writer = cv2.VideoWriter(
            str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (render_width * 2, render_height),
        )
        if not writer.isOpened():
            raise ValueError("Could not create the side-by-side MP4")

        history = []
        frame_index = 0
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            original = cv2.resize(frame, (render_width, render_height))
            annotated = original.copy()
            observation = observations[frame_index]
            if observation.get("valid"):
                center = (
                    round(observation["x_px"] * scale),
                    round(observation["y_px"] * scale),
                )
                history.append(center)
                history = history[-30:]
                if observation.get("bbox"):
                    x, y, box_width, box_height = observation["bbox"]
                    first = (round(x * scale), round(y * scale))
                    second = (
                        round((x + box_width) * scale),
                        round((y + box_height) * scale),
                    )
                    cv2.rectangle(annotated, first, second, (0, 255, 0), 2)
                cv2.circle(annotated, center, 5, (0, 255, 0), -1)
            for first, second in zip(history, history[1:]):
                cv2.line(annotated, first, second, (210, 210, 210), 2)

            forecast = prediction_timeline[frame_index]
            if forecast is not None:
                anchor, future_points, label = _comparison_forecast(
                    forecast, observation, trajectory_kind
                )
                previous = (round(anchor[0] * scale), round(anchor[1] * scale))
                for point in future_points:
                    target = (
                        round(point["x_px"] * scale),
                        round(point["y_px"] * scale),
                    )
                    _draw_dashed_line(annotated, previous, target, (14, 127, 255))
                    previous = target
                if future_points:
                    cv2.arrowedLine(
                        annotated,
                        (round(anchor[0] * scale), round(anchor[1] * scale)),
                        previous,
                        (14, 127, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.12,
                    )
                    cv2.circle(annotated, previous, 7, (14, 127, 255), 2)
                cv2.putText(
                    annotated, label,
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (14, 127, 255), 2, cv2.LINE_AA,
                )
            status = observation.get("status", "tracked" if observation.get("valid") else "invalid")
            color = (0, 255, 0) if observation.get("valid") else (0, 0, 255)
            cv2.putText(
                annotated, f"{frame_index / fps:.2f}s | {status}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            writer.write(cv2.hconcat((original, annotated)))
            frame_index += 1
        render_finished = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not render_finished:
            intermediate_path.unlink(missing_ok=True)
    try:
        if frame_index != len(observations):
            raise ValueError("Decoded frame count does not match tracking observations")
        codec_metadata = _transcode_browser_mp4(
            intermediate_path, output_path, frame_index, fps
        )
        return {
            "frames": frame_index,
            "fps": fps,
            "width": render_width * 2,
            "height": render_height,
            **codec_metadata,
        }
    finally:
        intermediate_path.unlink(missing_ok=True)


def render_comparison_video_bytes(
    filename, data, observations, prediction_timeline, *, trajectory_kind="rolling"
):
    suffix = validate_video_upload(filename, data)
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / f"input{suffix}"
        output_path = Path(directory) / "physicseye-comparison.mp4"
        input_path.write_bytes(data)
        metadata = _video_metadata(input_path)
        render_metadata = render_comparison_video(
            input_path, observations, prediction_timeline, output_path,
            trajectory_kind=trajectory_kind,
        )
        return output_path.read_bytes(), metadata | {
            f"output_{key}": value for key, value in render_metadata.items()
        }


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


def _draw_dashed_line(frame, start, end, color, width=2, dash_px=10):
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance <= 0:
        return
    for offset in range(0, round(distance), dash_px * 2):
        first_ratio = offset / distance
        second_ratio = min((offset + dash_px) / distance, 1.0)
        first = (
            round(start[0] + (end[0] - start[0]) * first_ratio),
            round(start[1] + (end[1] - start[1]) * first_ratio),
        )
        second = (
            round(start[0] + (end[0] - start[0]) * second_ratio),
            round(start[1] + (end[1] - start[1]) * second_ratio),
        )
        cv2.line(frame, first, second, color, width, cv2.LINE_AA)


def _point(value):
    return round(value["x_px"]), round(value["y_px"])

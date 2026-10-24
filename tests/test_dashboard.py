import copy
import json

import cv2
import numpy as np
import pytest

import physicseye.dashboard as dashboard
from physicseye.dashboard import (
    MODEL_META,
    load_benchmark,
    prepare_analysis,
    prepare_automatic_predictions,
    render_annotated_frame,
    render_comparison_video_bytes,
    render_roi_preview_bytes,
    track_video_bytes,
    validate_video_upload,
    video_metadata_bytes,
)
from physicseye.tracker import track_video


SAMPLE = "examples/generated/slowing_down.mp4"


@pytest.mark.parametrize("sample", ("examples/generated/constant_speed.mp4", SAMPLE))
def test_video_analysis_is_causal_and_uses_all_four_models(sample):
    observations = track_video(sample)
    first = prepare_analysis(observations, 2.0, [0.25, 0.5])
    changed_future = copy.deepcopy(observations)
    for point in changed_future:
        if point["time_s"] > 2.0 and point["valid"]:
            point["x_px"] += 1000
    second = prepare_analysis(changed_future, 2.0, [0.25, 0.5])

    assert first["prediction"] == second["prediction"]
    assert set(first["prediction"]["models"]) == set(MODEL_META)
    assert all(point["time_s"] <= 2.0 for point in first["history"])
    assert all(
        update["available_at_s"] <= 2.0
        for update in first["prediction"]["adaptive_state"]["update_trace"]
    )
    assert first["comparisons"] != second["comparisons"]


def test_upload_validation_and_tracking_integration():
    data = open(SAMPLE, "rb").read()
    observations = track_video_bytes("../../slowing_down.mp4", data)
    assert len(observations) == 150
    with pytest.raises(ValueError, match="MP4"):
        validate_video_upload("unsafe.txt", data)
    with pytest.raises(ValueError, match="empty"):
        validate_video_upload("empty.mp4", b"")


def test_roi_preview_and_tracking_integration():
    data = open(SAMPLE, "rb").read()
    metadata = video_metadata_bytes("slowing_down.mp4", data)
    assert metadata == {
        "width": 320, "height": 240, "frame_count": 150,
        "fps": 30.0, "duration_s": 5.0, "decoded_frame_mib": 0.2197265625,
    }
    preview = render_roi_preview_bytes("slowing_down.mp4", data, 0, (27, 107, 26, 26))
    assert preview.shape == (240, 320, 3)

    observations = track_video_bytes(
        "slowing_down.mp4", data, mode="roi_template",
        initial_frame=0, roi=(27, 107, 26, 26),
    )
    assert len(observations) == 150
    assert sum(point["valid"] for point in observations) >= 145
    assert all("confidence" in point and "status" in point for point in observations)


def test_upload_duration_limit(tmp_path):
    video = tmp_path / "long.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (64, 64)
    )
    assert writer.isOpened()
    for _ in range(61):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="60-second limit"):
        track_video_bytes("long.avi", video.read_bytes())


def test_4k_upload_is_rejected_before_frame_decode(monkeypatch):
    class MetadataOnlyCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 4096,
                cv2.CAP_PROP_FRAME_HEIGHT: 2160,
                cv2.CAP_PROP_FRAME_COUNT: 197,
                cv2.CAP_PROP_FPS: 25,
            }[property_id]

        def release(self):
            pass

    monkeypatch.setattr(dashboard.cv2, "VideoCapture", lambda _path: MetadataOnlyCapture())
    with pytest.raises(ValueError, match=r"4096x2160.*25\.3 MiB.*2.1-megapixel"):
        dashboard._video_metadata("4k.mp4")


def test_benchmark_loading_and_missing_file(tmp_path):
    missing = load_benchmark(tmp_path / "missing.json")
    assert missing["status"] == "error"
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"configuration": {}, "scenarios": [], "aggregate_summaries": [], "acceptance": {}}))
    assert load_benchmark(path)["status"] == "error"

    benchmark = load_benchmark()
    assert benchmark["status"] == "ok"
    constant_velocity = next(
        row for row in benchmark["data"]["aggregate_summaries"]
        if row["model"] == "constant_velocity"
    )
    assert constant_velocity["mae_px"] == pytest.approx(1.3296881414822217)


def test_annotated_frame_uses_real_video():
    observations = track_video(SAMPLE)
    analysis = prepare_analysis(observations, 2.0, [0.25])
    image = render_annotated_frame(SAMPLE, observations, analysis, 2.25)
    assert image.shape == (240, 320, 3)
    assert image.dtype.name == "uint8"


def test_automatic_predictions_are_causal_and_video_is_synchronized(tmp_path):
    data = open(SAMPLE, "rb").read()
    observations = track_video_bytes("slowing_down.mp4", data, mode="auto_motion")
    first = prepare_automatic_predictions(observations, 320)
    changed = copy.deepcopy(observations)
    for point in changed[91:]:
        if point["valid"]:
            point["x_px"] += 1000
    second = prepare_automatic_predictions(changed, 320)

    assert first[:91] == second[:91]
    assert all(
        item is None or item["cutoff_frame"] <= frame
        for frame, item in enumerate(first)
    )
    video, metadata = render_comparison_video_bytes(
        "slowing_down.mp4", data, observations, first
    )
    assert video
    output_path = tmp_path / "comparison.mp4"
    with output_path.open("wb") as stream:
        stream.write(video)
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(observations)
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 30
        assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
        assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
    finally:
        capture.release()
    assert metadata["output_frames"] == len(observations)


@pytest.mark.parametrize("size", ((1280, 720), (1920, 1080)))
def test_automatic_upload_supports_hd_and_bounds_output_resolution(tmp_path, size):
    video_path = tmp_path / f"input-{size[0]}.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, size
    )
    assert writer.isOpened()
    for frame_index in range(20):
        frame = np.full((size[1], size[0], 3), 24, dtype=np.uint8)
        center = (
            round(size[0] * (0.2 + 0.02 * frame_index)),
            size[1] // 2,
        )
        cv2.circle(frame, center, max(14, size[1] // 35), (255, 80, 20), -1)
        writer.write(frame)
    writer.release()
    data = video_path.read_bytes()

    metadata = video_metadata_bytes(video_path.name, data)
    observations = track_video_bytes(video_path.name, data, mode="auto_motion")
    timeline = prepare_automatic_predictions(observations, size[0])
    output, output_metadata = render_comparison_video_bytes(
        video_path.name, data, observations, timeline
    )
    assert metadata["width"] == size[0]
    assert metadata["height"] == size[1]
    assert sum(point["valid"] for point in observations) >= 15
    assert output
    assert output_metadata["output_width"] == 2560
    assert output_metadata["output_height"] == 720

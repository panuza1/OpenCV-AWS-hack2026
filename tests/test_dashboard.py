import copy
import json

import cv2
import numpy as np
import pytest

from physicseye.dashboard import (
    MODEL_META,
    load_benchmark,
    prepare_analysis,
    render_annotated_frame,
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
        "fps": 30.0, "duration_s": 5.0,
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

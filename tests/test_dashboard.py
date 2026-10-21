import copy
import json

import pytest

from physicseye.dashboard import (
    MODEL_META,
    load_benchmark,
    prepare_analysis,
    render_annotated_frame,
    track_video_bytes,
    validate_video_upload,
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

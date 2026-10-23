import csv
import math
from pathlib import Path

import pytest

from physicseye.tracker import track_video


CONFIGURATIONS = {
    "basketball_6077402": {"initial_frame": 80, "roi": (1175, 120, 430, 430), "max_mae": 35.0},
    "billiards_6254504": {"initial_frame": 0, "roi": (776, 1539, 240, 240), "max_mae": 40.0},
    "disco_5751090": {"initial_frame": 0, "roi": (1880, 500, 1150, 1150), "max_mae": 30.0},
}


def _annotations(name):
    with open("docs/real_video_roi_annotations.csv", newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["video"] == name]


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_roi_tracker_matches_real_video_annotations(name):
    video = Path(f"examples/real/{name}.mp4")
    if not video.is_file():
        pytest.skip("licensed real-video fixture is downloaded separately")
    configuration = CONFIGURATIONS[name]
    observations = track_video(
        video,
        mode="roi_template",
        initial_frame=configuration["initial_frame"],
        roi=configuration["roi"],
    )

    errors = []
    false_positives = 0
    for annotation in _annotations(name):
        observation = observations[int(annotation["frame"])]
        if annotation["ball_visible"] == "false":
            false_positives += observation["valid"]
            continue
        assert observation["valid"], (name, annotation["frame"], observation["status"])
        error = math.hypot(
            observation["x_px"] - float(annotation["x_px"]),
            observation["y_px"] - float(annotation["y_px"]),
        )
        errors.append(error)
        assert error <= float(annotation["radius_px"])

    assert false_positives == 0
    assert sum(errors) / len(errors) <= configuration["max_mae"]

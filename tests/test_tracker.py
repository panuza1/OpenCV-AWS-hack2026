import csv
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

import physicseye.tracker as tracker
from physicseye.tracker import VideoError, track_video
from scripts.generate_sample import generate_samples


def _truth(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_tracks_synthetic_samples_with_low_error(tmp_path):
    for sample in generate_samples(tmp_path / "samples"):
        observations = track_video(sample["video"])
        truth = _truth(sample["ground_truth"])
        errors = [
            ((observation["x_px"] - float(expected["x_px"])) ** 2
             + (observation["y_px"] - float(expected["y_px"])) ** 2) ** 0.5
            for observation, expected in zip(observations, truth)
            if observation["valid"]
        ]
        assert len(observations) == len(truth) == 150
        assert len(errors) == 150
        assert sum(errors) / len(errors) <= 5.0


def test_missing_ball_and_large_jump_are_invalid(tmp_path):
    video = tmp_path / "edge_cases.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240)
    )
    assert writer.isOpened()
    for center in ((40, 120), None, (42, 120), (250, 120)):
        frame = np.full((240, 320, 3), 18, dtype=np.uint8)
        cv2.circle(frame, (10, 10), 2, (0, 140, 255), -1)
        if center:
            cv2.circle(frame, center, 12, (0, 140, 255), -1)
        writer.write(frame)
    writer.release()

    observations = track_video(video)
    assert [observation["valid"] for observation in observations] == [True, False, True, False]
    for observation in (observations[1], observations[3]):
        assert observation["x_px"] is observation["y_px"] is observation["area_px2"] is None


def test_unreadable_video_has_clear_error(tmp_path):
    with pytest.raises(VideoError, match="Cannot open video"):
        track_video(tmp_path / "missing.mp4")

    result = subprocess.run(
        [sys.executable, "scripts/analyze_video.py", "--video", str(tmp_path / "missing.mp4")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error: Cannot open video" in result.stderr


def test_empty_video_and_unknown_fps_are_rejected(monkeypatch):
    class EmptyCapture:
        def isOpened(self):
            return True

        def get(self, _property):
            return 30.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(tracker.cv2, "VideoCapture", lambda _path: EmptyCapture())
    with pytest.raises(VideoError, match="no decodable frames"):
        track_video("empty.mp4")

    capture = EmptyCapture()
    capture.get = lambda _property: 0.0
    with pytest.raises(VideoError, match="provide a positive --fps override"):
        tracker._video_fps(capture, None)
    assert tracker._video_fps(capture, 24.0) == 24.0


def test_cli_writes_json_and_csv(tmp_path):
    sample = generate_samples(tmp_path / "samples")[0]
    output_dir = tmp_path / "tracks"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_video.py",
            "--video",
            sample["video"],
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["frames"] == summary["valid_frames"] == 150
    with open(summary["json"], encoding="utf-8") as stream:
        observations = json.load(stream)
    assert observations[0].keys() == {
        "frame", "time_s", "x_px", "y_px", "area_px2", "valid"
    }
    assert (output_dir / "constant_speed.track.csv").is_file()


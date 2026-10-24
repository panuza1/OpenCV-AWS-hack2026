import copy
import math

import cv2
import numpy as np
import pytest

from physicseye.dashboard import render_comparison_video_bytes
from physicseye.flight import fit_flight_trajectory, prepare_flight_predictions
from physicseye.tracker import track_video


@pytest.fixture
def airborne_video(tmp_path):
    path = tmp_path / "airborne.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240)
    )
    assert writer.isOpened()
    truth = []
    for frame_index in range(60):
        frame = np.full((240, 320, 3), 24, dtype=np.uint8)
        center = (
            40 + 4 * frame_index,
            185 - 5 * frame_index + 0.08 * frame_index**2,
        )
        truth.append(center)
        if not 25 <= frame_index <= 27:
            rounded = tuple(round(value) for value in center)
            cv2.circle(frame, rounded, 12, (40, 180, 240), -1)
            cv2.circle(frame, (rounded[0] - 3, rounded[1] - 3), 3, (240, 240, 240), -1)
        writer.write(frame)
    writer.release()
    return path, truth


def test_flight_tracker_follows_parabolic_ball_and_keeps_occlusion_invalid(airborne_video):
    path, truth = airborne_video
    observations = track_video(path, mode="flight")
    errors = [
        math.hypot(observation["x_px"] - expected[0], observation["y_px"] - expected[1])
        for observation, expected in zip(observations, truth)
        if observation["valid"]
    ]
    assert len(errors) >= 48
    assert sum(errors) / len(errors) <= 5.0
    assert all(not observations[index]["valid"] for index in range(25, 28))
    assert all(
        "confidence" in observation and "status" in observation
        for observation in observations
    )


def test_projectile_fit_is_accurate_and_causal():
    observations = [
        {
            "frame": index,
            "time_s": index * 0.05,
            "x_px": 40.0 + 100.0 * index * 0.05,
            "y_px": 200.0 - 120.0 * index * 0.05 + 50.0 * (index * 0.05) ** 2,
            "valid": True,
        }
        for index in range(41)
    ]
    first = fit_flight_trajectory(observations, 1.0, 0.5)
    changed_future = copy.deepcopy(observations)
    for point in changed_future:
        if point["time_s"] > 1.0:
            point["x_px"] += 1000.0
            point["y_px"] -= 1000.0
    second = fit_flight_trajectory(changed_future, 1.0, 0.5)

    assert first == second
    assert first["status"] == "ok"
    assert first["fit_r2"] == pytest.approx(1.0)
    assert first["fit_rmse_px"] == pytest.approx(0.0, abs=1e-9)
    assert max(point["time_s"] for point in first["observed_points"]) <= 1.0
    endpoint = first["predicted_points"][-1]
    assert endpoint["x_px"] == pytest.approx(190.0)
    assert endpoint["y_px"] == pytest.approx(132.5)


def test_flight_timeline_and_browser_video_are_causal_and_synchronized(
    airborne_video, tmp_path
):
    path, truth = airborne_video
    observations = track_video(path, mode="flight")
    first = prepare_flight_predictions(observations, 320)
    changed_future = copy.deepcopy(observations)
    for point in changed_future[36:]:
        if point["valid"]:
            point["x_px"] += 1000.0
    second = prepare_flight_predictions(changed_future, 320)
    assert first[:36] == second[:36]

    endpoint_errors = []
    seen_cutoffs = set()
    for fit in first:
        if not fit or fit["cutoff_frame"] in seen_cutoffs:
            continue
        seen_cutoffs.add(fit["cutoff_frame"])
        endpoint = fit["predicted_points"][-1]
        target_frame = round(endpoint["time_s"] * 30.0)
        if target_frame < len(truth):
            endpoint_errors.append(math.hypot(
                endpoint["x_px"] - truth[target_frame][0],
                endpoint["y_px"] - truth[target_frame][1],
            ))
    assert endpoint_errors
    assert sum(endpoint_errors) / len(endpoint_errors) <= 15.0

    video, metadata = render_comparison_video_bytes(
        path.name, path.read_bytes(), observations, first, trajectory_kind="flight"
    )
    output = tmp_path / "flight.mp4"
    output.write_bytes(video)
    capture = cv2.VideoCapture(str(output))
    try:
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(observations)
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 30
    finally:
        capture.release()
    assert metadata["output_codec"] == "h264"
    assert metadata["output_pixel_format"] == "yuv420p"
    assert metadata["output_duration_s"] == pytest.approx(2.0)

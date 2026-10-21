import math

import pytest

from physicseye.physics import (
    forecast_position,
    predict_trajectory,
    travel_distance,
    update_deceleration,
)


def _constant_deceleration_observations(
    *, initial_speed=40.0, deceleration=4.0, duration=2.0, step=0.05
):
    observations = []
    count = round(duration / step) + 1
    for index in range(count):
        time_s = index * step
        stop_time = initial_speed / deceleration if deceleration else math.inf
        active_time = min(time_s, stop_time)
        x_px = 10.0 + initial_speed * active_time - 0.5 * deceleration * active_time**2
        observations.append(
            {"time_s": time_s, "x_px": x_px, "y_px": 20.0, "valid": True}
        )
    return observations


def _changing_deceleration_observations(
    *, initial_speed=40.0, first_d=2.0, second_d=10.0, change_time=1.5,
    duration=4.0, step=0.05
):
    observations = []
    x_at_change = 10.0 + initial_speed * change_time - 0.5 * first_d * change_time**2
    speed_at_change = initial_speed - first_d * change_time
    for index in range(round(duration / step) + 1):
        time_s = index * step
        if time_s <= change_time:
            x_px = 10.0 + initial_speed * time_s - 0.5 * first_d * time_s**2
        else:
            elapsed = time_s - change_time
            x_px = x_at_change + speed_at_change * elapsed - 0.5 * second_d * elapsed**2
        observations.append(
            {"time_s": time_s, "x_px": x_px, "y_px": 20.0, "valid": True}
        )
    return observations


def _at_time(observations, time_s):
    return next(observation for observation in observations if observation["time_s"] == pytest.approx(time_s))


def test_analytic_forecasts_and_stopping_distance():
    observations = _constant_deceleration_observations()
    cutoff = 1.0
    result = predict_trajectory(
        [observation for observation in observations if observation["time_s"] <= cutoff],
        cutoff,
        [0.25, 0.5],
    )

    assert result["status"] == "ok"
    fitted = result["models"]["fitted_deceleration"]
    assert fitted["parameters"]["deceleration_px_s2"] == pytest.approx(4.0)
    for prediction in fitted["predictions"]:
        actual = _at_time(observations, prediction["target_time_s"])
        assert prediction["x_px"] == pytest.approx(actual["x_px"], abs=1e-9)
        assert prediction["y_px"] == pytest.approx(actual["y_px"], abs=1e-9)

    assert travel_distance(10.0, 2.0, 5.0) == pytest.approx(25.0)
    assert travel_distance(10.0, 2.0, 50.0) == pytest.approx(25.0)
    motion = {
        "x_px": 3.0,
        "y_px": 4.0,
        "direction_x": 1.0,
        "direction_y": 0.0,
        "speed_px_s": 10.0,
    }
    assert forecast_position(motion, 2.0, 50.0) == pytest.approx((28.0, 4.0))


def test_exact_residual_update_math_with_alpha_one():
    candidate, updated = update_deceleration(
        current_px_s2=2.0,
        residual_px=0.25,
        speed_px_s=20.0,
        elapsed_s=0.25,
        alpha=1.0,
    )
    assert candidate == pytest.approx(10.0)
    assert updated == pytest.approx(candidate)


def test_future_outcome_cannot_change_an_earlier_prediction():
    observations = _changing_deceleration_observations()
    cutoff = 1.75
    prefix = [observation for observation in observations if observation["time_s"] <= cutoff]
    before = predict_trajectory(prefix, cutoff, [0.25])

    future = _at_time(observations, 2.0)
    rejected = predict_trajectory(prefix + [future], cutoff, [0.25])
    after = predict_trajectory(prefix, cutoff, [0.25])

    assert before == after
    assert rejected["status"] == "error"
    assert "after cutoff_time_s" in rejected["error"]
    assert all(
        update["available_at_s"] <= cutoff
        for update in before["adaptive_state"]["update_trace"]
    )


def test_irregular_timestamps_missing_observation_and_deterministic_replay():
    observations = _constant_deceleration_observations(step=0.05)
    observations[6] = {"time_s": observations[6]["time_s"], "x_px": None, "y_px": None, "valid": False}
    observations = [observation for index, observation in enumerate(observations) if index not in {9, 14}]
    cutoff = observations[-1]["time_s"]

    first = predict_trajectory(observations, cutoff, [0.25])
    second = predict_trajectory(observations, cutoff, [0.25])

    assert first == second
    assert first["status"] == "ok"
    assert all(
        math.isfinite(prediction[coordinate])
        for model in first["models"].values()
        for prediction in model["predictions"]
        for coordinate in ("x_px", "y_px")
    )


def test_irregular_timestamps_preserve_analytic_prediction():
    times = [0.0, 0.07, 0.19, 0.31, 0.52, 0.74, 1.0]
    observations = [
        {
            "time_s": time_s,
            "x_px": 10.0 + 40.0 * time_s - 2.0 * time_s**2,
            "y_px": 20.0,
            "valid": True,
        }
        for time_s in times
    ]
    result = predict_trajectory(observations, 1.0, [0.25])
    prediction = result["models"]["fitted_deceleration"]["predictions"][0]

    assert result["status"] == "ok"
    assert prediction["x_px"] == pytest.approx(10.0 + 40.0 * 1.25 - 2.0 * 1.25**2)
    assert prediction["y_px"] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "observations, cutoff, message",
    [
        ([], 0.0, "at least 4"),
        (
            [
                {"time_s": 0.0, "x_px": 0.0, "y_px": 0.0, "valid": True},
                {"time_s": 0.0, "x_px": 1.0, "y_px": 0.0, "valid": True},
            ],
            0.0,
            "strictly increasing",
        ),
        (
            [
                {"time_s": index * 0.1, "x_px": float("nan"), "y_px": 0.0, "valid": True}
                for index in range(4)
            ],
            0.3,
            "finite coordinates",
        ),
    ],
)
def test_invalid_inputs_return_explicit_error_status(observations, cutoff, message):
    result = predict_trajectory(observations, cutoff, [0.25])
    assert result["status"] == "error"
    assert message in result["error"]
    assert result["models"] == {}


def test_stationary_and_ambiguous_feedback_are_safe():
    stationary = [
        {"time_s": index * 0.1, "x_px": 5.0, "y_px": 5.0, "valid": True}
        for index in range(10)
    ]
    result = predict_trajectory(stationary, 0.9, [0.25])
    assert result["status"] == "ok"
    assert result["adaptive_state"]["update_count"] == 0
    assert "stationary" in result["warnings"][0]
    for model in result["models"].values():
        assert model["predictions"][0]["x_px"] == pytest.approx(5.0)
        assert model["predictions"][0]["y_px"] == pytest.approx(5.0)

    turning = _constant_deceleration_observations(duration=1.2, step=0.05)
    turning[20]["y_px"] += 20.0
    turned = predict_trajectory(turning, turning[-1]["time_s"], [0.25])
    assert turned["status"] == "ok"
    assert turned["adaptive_state"]["skipped_updates"]


def test_adaptive_beats_frozen_by_ten_percent_after_dynamics_change():
    observations = _changing_deceleration_observations()
    horizon = 0.25
    frozen_errors = []
    adaptive_errors = []

    for cutoff_index in range(40, 71, 5):
        cutoff = observations[cutoff_index]["time_s"]
        target = observations[cutoff_index + 5]
        prefix = observations[: cutoff_index + 1]
        result = predict_trajectory(prefix, cutoff, [horizon])
        assert result["status"] == "ok"
        assert len(
            {model["parameters"]["speed_px_s"] for model in result["models"].values()}
        ) == 1
        for model_name, errors in (
            ("frozen_deceleration", frozen_errors),
            ("adaptive_deceleration", adaptive_errors),
        ):
            prediction = result["models"][model_name]["predictions"][0]
            errors.append(math.hypot(prediction["x_px"] - target["x_px"], prediction["y_px"] - target["y_px"]))

    frozen_mean = sum(frozen_errors) / len(frozen_errors)
    adaptive_mean = sum(adaptive_errors) / len(adaptive_errors)
    improvement = (frozen_mean - adaptive_mean) / frozen_mean
    assert improvement >= 0.10, (
        f"adaptive improvement was {improvement:.1%}: "
        f"frozen={frozen_mean:.6f}px adaptive={adaptive_mean:.6f}px"
    )

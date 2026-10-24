"""Causal short-window projectile fitting for tracked airborne balls."""

import math

import numpy as np


MIN_FLIGHT_POINTS = 6
FLIGHT_WINDOW_S = 1.0
FLIGHT_HORIZON_S = 0.5


def fit_flight_trajectory(
    observations,
    cutoff_time_s,
    horizon_s=FLIGHT_HORIZON_S,
    *,
    window_s=FLIGHT_WINDOW_S,
    prediction_steps=15,
):
    """Fit x(t) linearly and y(t) quadratically using observations through cutoff."""
    if not math.isfinite(cutoff_time_s):
        raise ValueError("Flight fit cutoff must be finite")
    if not all(math.isfinite(value) and value > 0 for value in (horizon_s, window_s)):
        raise ValueError("Flight fit horizon and window must be positive")
    if (
        not isinstance(prediction_steps, int)
        or isinstance(prediction_steps, bool)
        or prediction_steps <= 0
    ):
        raise ValueError("Flight prediction steps must be a positive integer")
    history = [
        point for point in observations
        if point.get("valid")
        and cutoff_time_s - window_s <= point["time_s"] <= cutoff_time_s
    ]
    if len(history) < MIN_FLIGHT_POINTS:
        return {
            "status": "insufficient_history",
            "cutoff_time_s": cutoff_time_s,
            "observed_points": history,
            "predicted_points": [],
        }
    times = np.asarray([point["time_s"] - cutoff_time_s for point in history])
    if np.ptp(times) < 0.1:
        return {
            "status": "insufficient_time_span",
            "cutoff_time_s": cutoff_time_s,
            "observed_points": history,
            "predicted_points": [],
        }
    x_values = np.asarray([point["x_px"] for point in history])
    y_values = np.asarray([point["y_px"] for point in history])
    try:
        x_coefficients = np.polyfit(times, x_values, 1)
        y_coefficients = np.polyfit(times, y_values, 2)
    except np.linalg.LinAlgError:
        return {
            "status": "fit_failed",
            "cutoff_time_s": cutoff_time_s,
            "observed_points": history,
            "predicted_points": [],
        }

    fitted_x = np.polyval(x_coefficients, times)
    fitted_y = np.polyval(y_coefficients, times)
    squared_error = float(np.sum((x_values - fitted_x) ** 2 + (y_values - fitted_y) ** 2))
    total_variance = float(
        np.sum((x_values - np.mean(x_values)) ** 2 + (y_values - np.mean(y_values)) ** 2)
    )
    fit_r2 = 1.0 if total_variance == 0 and squared_error == 0 else (
        max(-1.0, min(1.0, 1.0 - squared_error / total_variance))
        if total_variance else 0.0
    )
    future_times = np.linspace(horizon_s / prediction_steps, horizon_s, prediction_steps)
    predicted_points = [
        {
            "time_s": float(cutoff_time_s + delta),
            "x_px": float(np.polyval(x_coefficients, delta)),
            "y_px": float(np.polyval(y_coefficients, delta)),
        }
        for delta in future_times
    ]
    velocity_x = float(x_coefficients[0])
    velocity_y = float(y_coefficients[1])
    return {
        "status": "ok",
        "cutoff_time_s": cutoff_time_s,
        "horizon_s": float(horizon_s),
        "observed_points": history,
        "predicted_points": predicted_points,
        "anchor_x_px": float(np.polyval(x_coefficients, 0.0)),
        "anchor_y_px": float(np.polyval(y_coefficients, 0.0)),
        "direction_angle_deg": math.degrees(math.atan2(velocity_y, velocity_x)),
        "speed_px_s": math.hypot(velocity_x, velocity_y),
        "vertical_acceleration_px_s2": float(2.0 * y_coefficients[0]),
        "fit_r2": fit_r2,
        "fit_rmse_px": math.sqrt(squared_error / len(history)),
        "coefficients": {
            "x": [float(value) for value in x_coefficients],
            "y": [float(value) for value in y_coefficients],
        },
    }


def prepare_flight_predictions(observations, frame_width):
    """Create a per-frame causal flight forecast timeline."""
    if len(observations) < 2:
        return [None] * len(observations)
    frame_interval_s = observations[1]["time_s"] - observations[0]["time_s"]
    step = max(1, round(0.2 / frame_interval_s))
    timeline = []
    latest = None
    for index, observation in enumerate(observations):
        if index % step == 0:
            history = observations[:index + 1]
            fit = fit_flight_trajectory(history, observation["time_s"])
            if fit["status"] == "ok":
                horizon_s = min(
                    0.75,
                    max(0.25, 0.2 * frame_width / max(fit["speed_px_s"], 1.0)),
                )
                latest = fit_flight_trajectory(
                    history, observation["time_s"], horizon_s
                )
                latest["cutoff_frame"] = index
        timeline.append(latest)
    return timeline

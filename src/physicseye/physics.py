"""Causal, dependency-free trajectory prediction for PhysicsEye tracks."""

import math


MIN_OBSERVATIONS = 4
FIT_WINDOW_S = 0.75
WARMUP_DURATION_S = FIT_WINDOW_S
FEEDBACK_HORIZON_S = 0.25
ADAPTATION_ALPHA = 0.25
MAX_FEEDBACK_LAG_S = 0.125
MIN_SPEED_PX_S = 1e-6


class PhysicsError(ValueError):
    """Raised internally when a prediction request is invalid."""


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_request(observations, cutoff_time_s, horizons_s):
    if not _finite_number(cutoff_time_s) or cutoff_time_s < 0:
        raise PhysicsError("cutoff_time_s must be a finite nonnegative number")
    if not isinstance(observations, (list, tuple)):
        raise PhysicsError("observations must be a list or tuple")
    if not isinstance(horizons_s, (list, tuple)) or not horizons_s:
        raise PhysicsError("horizons_s must contain at least one horizon")

    horizons = []
    for horizon in horizons_s:
        if not _finite_number(horizon) or horizon <= 0:
            raise PhysicsError("horizons_s must contain only finite positive numbers")
        horizons.append(float(horizon))
    if len(set(horizons)) != len(horizons):
        raise PhysicsError("horizons_s must not contain duplicates")

    previous_time = -math.inf
    validated = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise PhysicsError(f"observation {index} must be an object")
        time_s = observation.get("time_s")
        if not _finite_number(time_s) or time_s < 0:
            raise PhysicsError(f"observation {index} has invalid time_s")
        time_s = float(time_s)
        if time_s <= previous_time:
            raise PhysicsError("observation times must be strictly increasing")
        if time_s > cutoff_time_s:
            raise PhysicsError("observations after cutoff_time_s are not allowed")
        previous_time = time_s

        valid = observation.get("valid")
        if not isinstance(valid, bool):
            raise PhysicsError(f"observation {index} must have a boolean valid field")
        if valid:
            x_px = observation.get("x_px")
            y_px = observation.get("y_px")
            if not _finite_number(x_px) or not _finite_number(y_px):
                raise PhysicsError(f"valid observation {index} must have finite coordinates")
            validated.append({"time_s": time_s, "x_px": float(x_px), "y_px": float(y_px)})
    return validated, sorted(horizons)


def _linear_fit(times, values):
    mean_time = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    denominator = sum((time_s - mean_time) ** 2 for time_s in times)
    if denominator <= 0:
        raise PhysicsError("fitting timestamps have no measurable span")
    slope = sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(times, values)
    ) / denominator
    return mean_value - slope * mean_time, slope


def estimate_motion(valid_observations, window_s=FIT_WINDOW_S):
    """Estimate current direction, speed and deceleration from recent valid points."""
    if len(valid_observations) < MIN_OBSERVATIONS:
        raise PhysicsError(f"at least {MIN_OBSERVATIONS} valid observations are required")
    end_time = valid_observations[-1]["time_s"]
    window = [
        observation
        for observation in valid_observations
        if observation["time_s"] >= end_time - window_s
    ]
    if len(window) < MIN_OBSERVATIONS:
        window = valid_observations[-MIN_OBSERVATIONS:]

    times = [observation["time_s"] for observation in window]
    _, slope_x = _linear_fit(times, [observation["x_px"] for observation in window])
    _, slope_y = _linear_fit(times, [observation["y_px"] for observation in window])
    slope_speed = math.hypot(slope_x, slope_y)
    if slope_speed <= MIN_SPEED_PX_S:
        direction_x = direction_y = 0.0
    else:
        direction_x = slope_x / slope_speed
        direction_y = slope_y / slope_speed

    segment_times = []
    segment_speeds = []
    for first, second in zip(window, window[1:]):
        elapsed = second["time_s"] - first["time_s"]
        segment_times.append((first["time_s"] + second["time_s"]) / 2.0)
        segment_speeds.append(
            ((second["x_px"] - first["x_px"]) * direction_x
             + (second["y_px"] - first["y_px"]) * direction_y) / elapsed
        )

    speed_intercept, speed_slope = _linear_fit(segment_times, segment_speeds)
    speed_px_s = max(0.0, speed_intercept + speed_slope * end_time)
    deceleration_px_s2 = max(0.0, -speed_slope)
    return {
        "anchor_time_s": end_time,
        "x_px": window[-1]["x_px"],
        "y_px": window[-1]["y_px"],
        "direction_x": direction_x,
        "direction_y": direction_y,
        "speed_px_s": speed_px_s,
        "velocity_x_px_s": direction_x * speed_px_s,
        "velocity_y_px_s": direction_y * speed_px_s,
        "deceleration_px_s2": deceleration_px_s2,
        "window_start_s": window[0]["time_s"],
        "window_end_s": end_time,
        "observation_count": len(window),
    }


def travel_distance(speed_px_s, deceleration_px_s2, elapsed_s):
    """Return straight-line travel distance, preserving distance after stopping."""
    if not all(_finite_number(value) for value in (speed_px_s, deceleration_px_s2, elapsed_s)):
        raise PhysicsError("speed, deceleration and elapsed time must be finite")
    if speed_px_s < 0 or deceleration_px_s2 < 0 or elapsed_s < 0:
        raise PhysicsError("speed, deceleration and elapsed time must be nonnegative")
    if deceleration_px_s2 == 0:
        return speed_px_s * elapsed_s
    stop_time_s = speed_px_s / deceleration_px_s2
    active_time_s = min(elapsed_s, stop_time_s)
    return speed_px_s * active_time_s - 0.5 * deceleration_px_s2 * active_time_s**2


def forecast_position(motion, deceleration_px_s2, elapsed_s):
    distance_px = travel_distance(motion["speed_px_s"], deceleration_px_s2, elapsed_s)
    return (
        motion["x_px"] + motion["direction_x"] * distance_px,
        motion["y_px"] + motion["direction_y"] * distance_px,
    )


def update_deceleration(current_px_s2, residual_px, speed_px_s, elapsed_s, alpha):
    """Apply the declared bounded residual update and return candidate plus new value."""
    values = (current_px_s2, residual_px, speed_px_s, elapsed_s, alpha)
    if not all(_finite_number(value) for value in values):
        raise PhysicsError("adaptive update inputs must be finite")
    if current_px_s2 < 0 or speed_px_s <= 0 or elapsed_s <= 0 or not 0 <= alpha <= 1:
        raise PhysicsError("adaptive update inputs are outside their valid bounds")
    candidate = current_px_s2 + 2.0 * residual_px / elapsed_s**2
    candidate = min(max(candidate, 0.0), speed_px_s / elapsed_s)
    updated = (1.0 - alpha) * current_px_s2 + alpha * candidate
    return candidate, updated


def _issue_feedback(motion, deceleration_px_s2, horizon_s):
    return {
        "issued_at_s": motion["anchor_time_s"],
        "due_at_s": motion["anchor_time_s"] + horizon_s,
        "direction_x": motion["direction_x"],
        "direction_y": motion["direction_y"],
        "speed_px_s": motion["speed_px_s"],
        "deceleration_px_s2": deceleration_px_s2,
        "origin_x_px": motion["x_px"],
        "origin_y_px": motion["y_px"],
    }


def _replay_adaptation(valid_observations, alpha, feedback_horizon_s):
    state = {
        "d_initial_px_s2": None,
        "d_current_px_s2": None,
        "alpha": alpha,
        "update_count": 0,
        "update_trace": [],
        "skipped_updates": [],
    }
    pending = None
    for index, observation in enumerate(valid_observations):
        history = valid_observations[: index + 1]
        if state["d_current_px_s2"] is None:
            if (
                len(history) < MIN_OBSERVATIONS
                or history[-1]["time_s"] - history[0]["time_s"] < WARMUP_DURATION_S
            ):
                continue
            motion = estimate_motion(history)
            initial = motion["deceleration_px_s2"]
            state["d_initial_px_s2"] = initial
            state["d_current_px_s2"] = initial
            if motion["speed_px_s"] > MIN_SPEED_PX_S:
                pending = _issue_feedback(motion, initial, feedback_horizon_s)
            continue

        if pending is None or observation["time_s"] < pending["due_at_s"]:
            continue

        elapsed_s = observation["time_s"] - pending["issued_at_s"]
        skip_reason = None
        if observation["time_s"] - pending["due_at_s"] > MAX_FEEDBACK_LAG_S:
            skip_reason = "feedback target missing or too late"
        elif pending["speed_px_s"] <= MIN_SPEED_PX_S:
            skip_reason = "speed is too small"
        elif pending["deceleration_px_s2"] > 0 and pending["speed_px_s"] <= pending["deceleration_px_s2"] * elapsed_s:
            skip_reason = "predicted stop occurs inside feedback interval"
        else:
            actual_dx = observation["x_px"] - pending["origin_x_px"]
            actual_dy = observation["y_px"] - pending["origin_y_px"]
            along_px = actual_dx * pending["direction_x"] + actual_dy * pending["direction_y"]
            cross_px = abs(actual_dx * pending["direction_y"] - actual_dy * pending["direction_x"])
            if along_px < 0 or cross_px > max(2.0, 0.25 * max(along_px, 1.0)):
                skip_reason = "motion direction changed or is ambiguous"

        if skip_reason:
            state["skipped_updates"].append(
                {
                    "issued_at_s": pending["issued_at_s"],
                    "available_at_s": observation["time_s"],
                    "reason": skip_reason,
                }
            )
        else:
            predicted_displacement = travel_distance(
                pending["speed_px_s"], pending["deceleration_px_s2"], elapsed_s
            )
            predicted_x_px = pending["origin_x_px"] + pending["direction_x"] * predicted_displacement
            predicted_y_px = pending["origin_y_px"] + pending["direction_y"] * predicted_displacement
            actual_displacement = (
                (observation["x_px"] - pending["origin_x_px"]) * pending["direction_x"]
                + (observation["y_px"] - pending["origin_y_px"]) * pending["direction_y"]
            )
            residual_px = predicted_displacement - actual_displacement
            candidate, updated = update_deceleration(
                state["d_current_px_s2"],
                residual_px,
                pending["speed_px_s"],
                elapsed_s,
                alpha,
            )
            state["update_trace"].append(
                {
                    "issued_at_s": pending["issued_at_s"],
                    "available_at_s": observation["time_s"],
                    "elapsed_s": elapsed_s,
                    "predicted_x_px": predicted_x_px,
                    "predicted_y_px": predicted_y_px,
                    "actual_x_px": observation["x_px"],
                    "actual_y_px": observation["y_px"],
                    "d_before_px_s2": state["d_current_px_s2"],
                    "residual_px": residual_px,
                    "d_candidate_px_s2": candidate,
                    "d_after_px_s2": updated,
                }
            )
            state["d_current_px_s2"] = updated
            state["update_count"] += 1

        motion = estimate_motion(history)
        pending = (
            _issue_feedback(motion, state["d_current_px_s2"], feedback_horizon_s)
            if motion["speed_px_s"] > MIN_SPEED_PX_S
            else None
        )
    return state


def _model_result(name, motion, deceleration_px_s2, cutoff_time_s, horizons_s):
    predictions = []
    for horizon_s in horizons_s:
        elapsed_s = cutoff_time_s + horizon_s - motion["anchor_time_s"]
        x_px, y_px = forecast_position(motion, deceleration_px_s2, elapsed_s)
        predictions.append(
            {
                "horizon_s": horizon_s,
                "target_time_s": cutoff_time_s + horizon_s,
                "x_px": x_px,
                "y_px": y_px,
            }
        )
    return {
        "model": name,
        "status": "ok",
        "parameters": {
            "speed_px_s": motion["speed_px_s"],
            "velocity_x_px_s": motion["velocity_x_px_s"],
            "velocity_y_px_s": motion["velocity_y_px_s"],
            "deceleration_px_s2": deceleration_px_s2,
        },
        "predictions": predictions,
    }


def predict_trajectory(
    observations,
    cutoff_time_s,
    horizons_s=(0.25, 0.5, 1.0),
    *,
    alpha=ADAPTATION_ALPHA,
    feedback_horizon_s=FEEDBACK_HORIZON_S,
):
    """Return four causal forecasts and an auditable adaptive replay state."""
    try:
        if not _finite_number(alpha) or not 0 <= alpha <= 1:
            raise PhysicsError("alpha must be between 0 and 1")
        if not _finite_number(feedback_horizon_s) or feedback_horizon_s <= 0:
            raise PhysicsError("feedback_horizon_s must be finite and positive")
        valid_observations, horizons = _validate_request(
            observations, cutoff_time_s, horizons_s
        )
        motion = estimate_motion(valid_observations)
        adaptive_state = _replay_adaptation(valid_observations, alpha, feedback_horizon_s)
        if adaptive_state["d_current_px_s2"] is None:
            raise PhysicsError(
                f"adaptive state requires at least {WARMUP_DURATION_S:g} seconds of valid history"
            )

        d_fit = motion["deceleration_px_s2"]
        d_initial = adaptive_state["d_initial_px_s2"]
        d_current = adaptive_state["d_current_px_s2"]
        models = {}
        for name, deceleration in (
            ("constant_velocity", 0.0),
            ("fitted_deceleration", d_fit),
            ("frozen_deceleration", d_initial),
            ("adaptive_deceleration", d_current),
        ):
            models[name] = _model_result(
                name, motion, deceleration, float(cutoff_time_s), horizons
            )

        warnings = []
        if motion["anchor_time_s"] < cutoff_time_s:
            warnings.append("latest valid observation precedes cutoff_time_s")
        if motion["speed_px_s"] <= MIN_SPEED_PX_S:
            warnings.append("object is stationary or motion direction is unavailable")
        return {
            "status": "ok",
            "cutoff_time_s": float(cutoff_time_s),
            "fitting_window": {
                "start_time_s": motion["window_start_s"],
                "end_time_s": motion["window_end_s"],
                "valid_observations": motion["observation_count"],
            },
            "models": models,
            "adaptive_state": adaptive_state,
            "warnings": warnings,
        }
    except (PhysicsError, KeyError, TypeError, OverflowError) as error:
        return {
            "status": "error",
            "error": str(error),
            "cutoff_time_s": cutoff_time_s,
            "models": {},
            "adaptive_state": None,
            "warnings": [],
        }

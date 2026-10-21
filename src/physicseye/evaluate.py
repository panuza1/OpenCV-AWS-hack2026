"""Leakage-safe walk-forward evaluation and benchmark exports."""

import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

from physicseye.physics import ADAPTATION_ALPHA, predict_trajectory


MODELS = (
    "constant_velocity",
    "fitted_deceleration",
    "frozen_deceleration",
    "adaptive_deceleration",
)
HORIZONS_S = (0.25, 0.5, 1.0)
BENCHMARK_SEED = 20261021
DETAIL_FIELDS = (
    "scenario", "source", "cutoff_time_s", "target_time_s", "horizon_s",
    "model", "actual_x_px", "actual_y_px", "predicted_x_px", "predicted_y_px",
    "error_px", "deceleration_px_s2", "true_deceleration_px_s2",
    "parameter_error_px_s2", "history_observations", "history_valid_observations",
    "history_max_time_s", "adaptive_updates", "adaptive_skips", "latency_ms",
)


class EvaluationError(ValueError):
    """Raised when a benchmark scenario cannot be evaluated fairly."""


def calculate_metrics(errors):
    """Return standard pixel-error metrics for a nonempty sequence."""
    if not errors:
        return {"sample_count": 0, "mae_px": None, "median_error_px": None, "rmse_px": None}
    values = [float(error) for error in errors]
    if any(not math.isfinite(error) or error < 0 for error in values):
        raise EvaluationError("errors must be finite and nonnegative")
    return {
        "sample_count": len(values),
        "mae_px": sum(values) / len(values),
        "median_error_px": statistics.median(values),
        "rmse_px": math.sqrt(sum(error**2 for error in values) / len(values)),
    }


def _interpolate_truth(ground_truth, target_time_s):
    previous = None
    for point in ground_truth:
        time_s = float(point["time_s"])
        if math.isclose(time_s, target_time_s, abs_tol=1e-9):
            return {
                "time_s": target_time_s,
                "x_px": float(point["x_px"]),
                "y_px": float(point["y_px"]),
                "deceleration_px_s2": float(point.get("deceleration_px_s2", 0.0)),
            }
        if time_s > target_time_s:
            if previous is None:
                return None
            first_time = float(previous["time_s"])
            weight = (target_time_s - first_time) / (time_s - first_time)
            return {
                "time_s": target_time_s,
                "x_px": float(previous["x_px"]) + weight * (float(point["x_px"]) - float(previous["x_px"])),
                "y_px": float(previous["y_px"]) + weight * (float(point["y_px"]) - float(previous["y_px"])),
                "deceleration_px_s2": float(previous.get("deceleration_px_s2", 0.0)) + weight * (
                    float(point.get("deceleration_px_s2", 0.0))
                    - float(previous.get("deceleration_px_s2", 0.0))
                ),
            }
        previous = point
    return None


def _model_summary(scenario_name, model, rows, failed_predictions, endpoint_horizon_s):
    model_rows = [row for row in rows if row["model"] == model]
    metrics = calculate_metrics([row["error_px"] for row in model_rows])
    endpoint_rows = [row for row in model_rows if row["horizon_s"] == endpoint_horizon_s]
    parameter_errors = [row["parameter_error_px_s2"] for row in model_rows]
    final_parameter_error = parameter_errors[-1] if parameter_errors else None
    return {
        "scenario": scenario_name,
        "model": model,
        **metrics,
        "endpoint_error_px": (
            sum(row["error_px"] for row in endpoint_rows) / len(endpoint_rows)
            if endpoint_rows else None
        ),
        "parameter_mae_px_s2": (
            sum(parameter_errors) / len(parameter_errors) if parameter_errors else None
        ),
        "parameter_std_px_s2": (
            statistics.pstdev(row["deceleration_px_s2"] for row in model_rows)
            if model_rows else None
        ),
        "final_parameter_error_px_s2": final_parameter_error,
        "failed_predictions": failed_predictions,
        "mean_latency_ms": (
            sum(row["latency_ms"] for row in model_rows) / len(model_rows)
            if model_rows else None
        ),
        "adaptive_gain_vs_frozen_pct": None,
        "adaptive_gain_vs_fitted_pct": None,
        "adaptive_win_rate_vs_frozen": None,
        "adaptive_win_rate_vs_fitted": None,
    }


def _add_adaptive_comparisons(summaries, rows):
    by_model = {summary["model"]: summary for summary in summaries}
    adaptive = by_model["adaptive_deceleration"]
    for comparator, gain_key, win_key in (
        ("frozen_deceleration", "adaptive_gain_vs_frozen_pct", "adaptive_win_rate_vs_frozen"),
        ("fitted_deceleration", "adaptive_gain_vs_fitted_pct", "adaptive_win_rate_vs_fitted"),
    ):
        baseline = by_model[comparator]["mae_px"]
        adaptive[gain_key] = (
            100.0 * (baseline - adaptive["mae_px"]) / baseline
            if baseline is not None and baseline > 1e-12 else None
        )
        paired = {}
        for row in rows:
            if row["model"] in ("adaptive_deceleration", comparator):
                key = (row["cutoff_time_s"], row["horizon_s"])
                paired.setdefault(key, {})[row["model"]] = row["error_px"]
        pairs = [values for values in paired.values() if len(values) == 2]
        adaptive[win_key] = (
            sum(values["adaptive_deceleration"] < values[comparator] for values in pairs) / len(pairs)
            if pairs else None
        )


def evaluate_scenario(scenario):
    """Evaluate one scenario with a fresh causal prefix at every cutoff."""
    observations = scenario["observations"]
    ground_truth = scenario["ground_truth"]
    horizons = tuple(scenario.get("horizons_s", HORIZONS_S))
    rows = []
    failures = []
    audits = []
    final_trace = {"update_count": 0, "update_trace": [], "skipped_updates": []}

    for cutoff_time_s in scenario["cutoffs_s"]:
        history = [point for point in observations if float(point["time_s"]) <= cutoff_time_s]
        if not history:
            failures.append(
                {"cutoff_time_s": cutoff_time_s, "reason": "no historical observations", "prediction_count": len(horizons)}
            )
            continue
        if any(float(point["time_s"]) > cutoff_time_s for point in history):
            raise EvaluationError("future observation entered a prediction history")

        started = time.perf_counter()
        result = predict_trajectory(history, cutoff_time_s, list(horizons))
        latency_ms = (time.perf_counter() - started) * 1000.0
        audit = {
            "cutoff_time_s": cutoff_time_s,
            "history_observations": len(history),
            "history_valid_observations": sum(point["valid"] for point in history),
            "history_max_time_s": max(float(point["time_s"]) for point in history),
            "status": result["status"],
            "latency_ms": latency_ms,
        }
        audits.append(audit)
        if result["status"] != "ok":
            failures.append(
                {"cutoff_time_s": cutoff_time_s, "reason": result["error"], "prediction_count": len(horizons)}
            )
            continue

        adaptive_state = result["adaptive_state"]
        if any(update["available_at_s"] > cutoff_time_s for update in adaptive_state["update_trace"]):
            raise EvaluationError("adaptive update used an outcome after its cutoff")
        final_trace = adaptive_state
        truth_at_cutoff = _interpolate_truth(ground_truth, cutoff_time_s)
        if truth_at_cutoff is None:
            failures.append(
                {"cutoff_time_s": cutoff_time_s, "reason": "cutoff ground truth unavailable", "prediction_count": len(horizons)}
            )
            continue

        targets = {}
        for horizon_s in horizons:
            target_time_s = cutoff_time_s + horizon_s
            if target_time_s <= cutoff_time_s:
                raise EvaluationError("evaluation target must be strictly after cutoff")
            target = _interpolate_truth(ground_truth, target_time_s)
            if target is None:
                failures.append(
                    {"cutoff_time_s": cutoff_time_s, "horizon_s": horizon_s, "reason": "target unavailable"}
                )
            else:
                targets[horizon_s] = target

        for model in MODELS:
            if model not in result["models"]:
                raise EvaluationError(f"prediction result is missing model {model}")
            predictions = {
                prediction["horizon_s"]: prediction
                for prediction in result["models"][model]["predictions"]
            }
            for horizon_s, target in targets.items():
                prediction = predictions.get(horizon_s)
                if prediction is None:
                    failures.append(
                        {"cutoff_time_s": cutoff_time_s, "horizon_s": horizon_s, "model": model, "reason": "prediction unavailable"}
                    )
                    continue
                error_px = math.hypot(
                    prediction["x_px"] - target["x_px"],
                    prediction["y_px"] - target["y_px"],
                )
                deceleration = result["models"][model]["parameters"]["deceleration_px_s2"]
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "source": scenario["source"],
                        "cutoff_time_s": cutoff_time_s,
                        "target_time_s": target["time_s"],
                        "horizon_s": horizon_s,
                        "model": model,
                        "actual_x_px": target["x_px"],
                        "actual_y_px": target["y_px"],
                        "predicted_x_px": prediction["x_px"],
                        "predicted_y_px": prediction["y_px"],
                        "error_px": error_px,
                        "deceleration_px_s2": deceleration,
                        "true_deceleration_px_s2": truth_at_cutoff["deceleration_px_s2"],
                        "parameter_error_px_s2": abs(deceleration - truth_at_cutoff["deceleration_px_s2"]),
                        "history_observations": len(history),
                        "history_valid_observations": sum(point["valid"] for point in history),
                        "history_max_time_s": max(float(point["time_s"]) for point in history),
                        "adaptive_updates": adaptive_state["update_count"],
                        "adaptive_skips": len(adaptive_state["skipped_updates"]),
                        "latency_ms": latency_ms,
                    }
                )

    failed_by_model = {
        model: sum(
            failure.get("prediction_count", 1)
            for failure in failures if failure.get("model") in (None, model)
        )
        for model in MODELS
    }
    summaries = [
        _model_summary(scenario["name"], model, rows, failed_by_model[model], max(horizons))
        for model in MODELS
    ]
    _add_adaptive_comparisons(summaries, rows)
    horizon_summaries = []
    for horizon_s in horizons:
        for model in MODELS:
            horizon_rows = [
                row for row in rows
                if row["model"] == model and row["horizon_s"] == horizon_s
            ]
            horizon_summaries.append(
                {
                    "scenario": scenario["name"],
                    "horizon_s": horizon_s,
                    "model": model,
                    **calculate_metrics([row["error_px"] for row in horizon_rows]),
                }
            )
    attempts = final_trace["update_count"] + len(final_trace["skipped_updates"])
    measurement_errors = []
    for observation in observations:
        if observation["valid"]:
            target = _interpolate_truth(ground_truth, float(observation["time_s"]))
            if target:
                measurement_errors.append(
                    math.hypot(
                        observation["x_px"] - target["x_px"],
                        observation["y_px"] - target["y_px"],
                    )
                )
    return {
        "name": scenario["name"],
        "source": scenario["source"],
        "description": scenario["description"],
        "horizons_s": list(horizons),
        "evaluation_window_s": [min(scenario["cutoffs_s"]), max(scenario["cutoffs_s"])],
        "rows": rows,
        "summaries": summaries,
        "horizon_summaries": horizon_summaries,
        "failures": failures,
        "audits": audits,
        "adaptive_trace": final_trace,
        "skipped_update_rate": len(final_trace["skipped_updates"]) / attempts if attempts else 0.0,
        "measurement_mae_px": calculate_metrics(measurement_errors)["mae_px"],
    }


def _deceleration(name, time_s):
    if name == "constant_velocity":
        return 0.0
    if name in ("constant_deceleration_quantized", "irregular_missing_noisy"):
        return 6.0
    if name == "abrupt_deceleration_change":
        return 2.0 if time_s < 1.5 else 10.0
    if name == "gradual_change_noisy":
        if time_s < 1.5:
            return 2.0
        if time_s > 3.5:
            return 10.0
        return 2.0 + 4.0 * (time_s - 1.5)
    raise EvaluationError(f"unknown synthetic dynamics: {name}")


def _ground_truth(name, duration_s=5.0, step_s=0.05):
    points = [{"time_s": 0.0, "x_px": 10.0, "y_px": 20.0, "deceleration_px_s2": _deceleration(name, 0.0)}]
    x_px = 10.0
    speed_px_s = 40.0
    for index in range(1, round(duration_s / step_s) + 1):
        previous_time = (index - 1) * step_s
        time_s = index * step_s
        deceleration = _deceleration(name, previous_time + step_s / 2.0)
        active_time = min(step_s, speed_px_s / deceleration) if deceleration else step_s
        x_px += speed_px_s * active_time - 0.5 * deceleration * active_time**2
        speed_px_s = max(0.0, speed_px_s - deceleration * step_s)
        points.append(
            {
                "time_s": time_s,
                "x_px": x_px,
                "y_px": 20.0,
                "deceleration_px_s2": _deceleration(name, time_s),
            }
        )
    return points


def _observations(ground_truth, *, rng, noise_px=0.0, quantized=False, irregular=False, missing=False):
    if irregular:
        times = [0.0]
        intervals = (0.04, 0.06, 0.05, 0.07, 0.03)
        index = 0
        while times[-1] + intervals[index % len(intervals)] <= ground_truth[-1]["time_s"]:
            times.append(round(times[-1] + intervals[index % len(intervals)], 10))
            index += 1
        source_points = [_interpolate_truth(ground_truth, time_s) for time_s in times]
    else:
        source_points = ground_truth

    observations = []
    for index, point in enumerate(source_points):
        time_s = float(point["time_s"])
        invalid = missing and index > 0 and index % 13 == 0
        if invalid:
            observations.append({"time_s": time_s, "x_px": None, "y_px": None, "valid": False})
            continue
        x_px = float(point["x_px"]) + (rng.gauss(0.0, noise_px) if noise_px else 0.0)
        y_px = float(point["y_px"]) + (rng.gauss(0.0, noise_px) if noise_px else 0.0)
        if quantized:
            x_px, y_px = round(x_px), round(y_px)
        observations.append({"time_s": time_s, "x_px": x_px, "y_px": y_px, "valid": True})
    return observations


def build_synthetic_scenarios(seed=BENCHMARK_SEED):
    """Return the frozen deterministic Phase 3 synthetic benchmark matrix."""
    definitions = (
        ("constant_velocity", "Ideal constant velocity control", {}),
        ("constant_deceleration_quantized", "Constant deceleration with integer-pixel measurements", {"quantized": True}),
        ("abrupt_deceleration_change", "Held-out abrupt change from 2 to 10 px/s^2 at 1.5 s", {}),
        ("gradual_change_noisy", "Gradual 2-to-10 px/s^2 change with 0.35 px Gaussian measurement noise", {"noise_px": 0.35}),
        ("irregular_missing_noisy", "Constant deceleration with irregular timestamps, 0.25 px noise, and every 13th observation missing", {"noise_px": 0.25, "irregular": True, "missing": True}),
    )
    cutoffs = [1.0 + 0.25 * index for index in range(13)]
    scenarios = []
    for offset, (name, description, conditions) in enumerate(definitions):
        truth = _ground_truth(name)
        scenarios.append(
            {
                "name": name,
                "source": "synthetic_ground_truth",
                "description": description,
                "observations": _observations(
                    truth, rng=random.Random(seed + offset), **conditions
                ),
                "ground_truth": truth,
                "cutoffs_s": cutoffs,
                "horizons_s": HORIZONS_S,
            }
        )
    return scenarios


def build_tracker_scenario(name, observations, ground_truth, nominal_deceleration_px_s2):
    """Build an evaluation scenario while keeping tracker measurements separate from truth."""
    truth = [
        {
            "time_s": float(point["time_s"]),
            "x_px": float(point["x_px"]),
            "y_px": float(point["y_px"]),
            "deceleration_px_s2": float(nominal_deceleration_px_s2),
        }
        for point in ground_truth
    ]
    return {
        "name": name,
        "source": "opencv5_synthetic_video",
        "description": "OpenCV 5 HSV/contour measurements from a generated video, scored against separately loaded rendered centers",
        "observations": observations,
        "ground_truth": truth,
        "cutoffs_s": [1.0 + 0.5 * index for index in range(6)],
        "horizons_s": HORIZONS_S,
    }


def run_benchmark(scenarios):
    evaluated = [evaluate_scenario(scenario) for scenario in scenarios]
    all_rows = [row for scenario in evaluated for row in scenario["rows"]]
    aggregate = [
        _model_summary(
            "ALL", model, all_rows,
            sum(
                next(summary for summary in scenario["summaries"] if summary["model"] == model)["failed_predictions"]
                for scenario in evaluated
            ),
            max(HORIZONS_S),
        )
        for model in MODELS
    ]
    _add_adaptive_comparisons(aggregate, all_rows)

    acceptance_rows = [
        row for row in all_rows
        if row["scenario"] == "abrupt_deceleration_change"
        and 2.0 <= row["cutoff_time_s"] <= 3.5
        and row["horizon_s"] == 0.25
    ]
    acceptance_means = {
        model: calculate_metrics([row["error_px"] for row in acceptance_rows if row["model"] == model])["mae_px"]
        for model in MODELS
    }
    frozen = acceptance_means["frozen_deceleration"]
    adaptive = acceptance_means["adaptive_deceleration"]
    improvement = 100.0 * (frozen - adaptive) / frozen if frozen else None
    return {
        "configuration": {
            "seed": BENCHMARK_SEED,
            "alpha": ADAPTATION_ALPHA,
            "horizons_s": list(HORIZONS_S),
            "units": {"position_error": "px", "deceleration": "px/s^2", "latency": "ms"},
            "acceptance_window": {"scenario": "abrupt_deceleration_change", "cutoff_start_s": 2.0, "cutoff_end_s": 3.5, "horizon_s": 0.25, "threshold_pct": 10.0},
        },
        "scenarios": evaluated,
        "aggregate_summaries": aggregate,
        "acceptance": {
            "sample_count_per_model": len(acceptance_rows) // len(MODELS),
            "model_mae_px": acceptance_means,
            "adaptive_improvement_vs_frozen_pct": improvement,
            "passed": improvement is not None and improvement >= 10.0,
        },
    }


def _markdown_number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown_report(benchmark):
    lines = [
        "# PhysicsEye Phase 3 Evaluation",
        "",
        "All forecasts use pixel-space observations available at or before each cutoff. Synthetic ground truth and tracker observations are stored separately. Targets are revealed only after prediction.",
        "",
        "## Configuration",
        "",
        f"- Seed: `{benchmark['configuration']['seed']}`",
        f"- Horizons: `{benchmark['configuration']['horizons_s']}` seconds",
        f"- Adaptive alpha: `{benchmark['configuration']['alpha']}`",
        "- MAE/RMSE/endpoint units: pixels; deceleration units: px/s²",
        "- Endpoint error: mean error at the largest requested horizon (1.0 s)",
        "- Scenario definitions, seed, alpha, horizons, cutoffs, and 10% threshold were frozen before this run",
        "",
        "## Scenario results",
        "",
        "| Scenario | Model | n | MAE px | Median px | RMSE px | Endpoint px | Parameter MAE px/s² | Parameter std px/s² | Failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = [summary for scenario in benchmark["scenarios"] for summary in scenario["summaries"]]
    for summary in summaries:
        lines.append(
            f"| {summary['scenario']} | {summary['model']} | {summary['sample_count']} | "
            f"{_markdown_number(summary['mae_px'])} | {_markdown_number(summary['median_error_px'])} | "
            f"{_markdown_number(summary['rmse_px'])} | {_markdown_number(summary['endpoint_error_px'])} | "
            f"{_markdown_number(summary['parameter_mae_px_s2'])} | {_markdown_number(summary['parameter_std_px_s2'])} | "
            f"{summary['failed_predictions']} |"
        )

    lines.extend([
        "",
        "## Aggregate error by horizon",
        "",
        "| Horizon s | Model | n | MAE px | RMSE px |",
        "|---:|---|---:|---:|---:|",
    ])
    for horizon_s in benchmark["configuration"]["horizons_s"]:
        for model in MODELS:
            horizon_rows = [
                row for scenario in benchmark["scenarios"] for row in scenario["rows"]
                if row["horizon_s"] == horizon_s and row["model"] == model
            ]
            metrics = calculate_metrics([row["error_px"] for row in horizon_rows])
            lines.append(
                f"| {horizon_s:.2f} | {model} | {metrics['sample_count']} | "
                f"{_markdown_number(metrics['mae_px'])} | {_markdown_number(metrics['rmse_px'])} |"
            )

    acceptance = benchmark["acceptance"]
    lines.extend([
        "",
        "## Adaptive acceptance result",
        "",
        f"The frozen-model MAE is **{acceptance['model_mae_px']['frozen_deceleration']:.6f} px** and adaptive MAE is **{acceptance['model_mae_px']['adaptive_deceleration']:.6f} px** over {acceptance['sample_count_per_model']} held-out post-change targets. Adaptive improvement is **{acceptance['adaptive_improvement_vs_frozen_pct']:.3f}%**; required threshold is 10%. Result: **{'PASS' if acceptance['passed'] else 'FAIL'}**.",
        "",
        "## Representative trajectory comparison",
        "",
        "The following predictions share the same history, cutoff, horizon, and ground-truth target:",
        "",
        "| Cutoff s | Horizon s | Ground truth x/y px | Constant velocity x/y px | Frozen x/y px | Adaptive x/y px |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    representative = [
        row for scenario in benchmark["scenarios"] if scenario["name"] == "abrupt_deceleration_change"
        for row in scenario["rows"] if row["cutoff_time_s"] == 2.0 and row["horizon_s"] == 0.25
    ]
    by_model = {row["model"]: row for row in representative}
    actual = representative[0]
    lines.append(
        f"| 2.000 | 0.250 | {actual['actual_x_px']:.3f}/{actual['actual_y_px']:.3f} | "
        f"{by_model['constant_velocity']['predicted_x_px']:.3f}/{by_model['constant_velocity']['predicted_y_px']:.3f} | "
        f"{by_model['frozen_deceleration']['predicted_x_px']:.3f}/{by_model['frozen_deceleration']['predicted_y_px']:.3f} | "
        f"{by_model['adaptive_deceleration']['predicted_x_px']:.3f}/{by_model['adaptive_deceleration']['predicted_y_px']:.3f} |"
    )

    lines.extend(["", "## Measurement and adaptive outcomes by scenario", ""])
    for scenario in benchmark["scenarios"]:
        adaptive_summary = next(summary for summary in scenario["summaries"] if summary["model"] == "adaptive_deceleration")
        lines.append(
            f"- **{scenario['name']}**: observation-vs-truth MAE `{_markdown_number(scenario['measurement_mae_px'])} px`; "
            f"gain vs frozen `{_markdown_number(adaptive_summary['adaptive_gain_vs_frozen_pct'])}%`; "
            f"gain vs refitted `{_markdown_number(adaptive_summary['adaptive_gain_vs_fitted_pct'])}%`; "
            f"skipped-update rate `{scenario['skipped_update_rate']:.3f}`."
        )

    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "- Synthetic scenarios are controlled tests, not evidence of real-world accuracy.",
        "- OpenCV scenarios use generated videos; tracking error is included, but no independently recorded real rolling-ball video is available yet.",
        "- Noisy, quantized, missing, and irregular observations can make the adaptive model worse; negative gains above are retained.",
        "- Ground-truth targets between sampled video frames are linearly interpolated and remain strictly after the cutoff.",
        "- Timing values are host-dependent and excluded from deterministic metric assertions.",
        "",
    ])
    return "\n".join(lines)


def write_benchmark(benchmark, output_dir="outputs", report_path="docs/EVALUATION.md"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    detail_rows = [row for scenario in benchmark["scenarios"] for row in scenario["rows"]]
    detail_csv = output_dir / "evaluation.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = [summary for scenario in benchmark["scenarios"] for summary in scenario["summaries"]]
    summary_rows.extend(benchmark["aggregate_summaries"])
    summary_csv = output_dir / "evaluation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    horizon_rows = [
        summary for scenario in benchmark["scenarios"]
        for summary in scenario["horizon_summaries"]
    ]
    horizon_csv = output_dir / "evaluation_by_horizon.csv"
    with horizon_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=horizon_rows[0].keys())
        writer.writeheader()
        writer.writerows(horizon_rows)

    traces = {
        scenario["name"]: {
            "update_trace": scenario["adaptive_trace"]["update_trace"],
            "skipped_updates": scenario["adaptive_trace"]["skipped_updates"],
        }
        for scenario in benchmark["scenarios"]
    }
    trace_json = output_dir / "adaptive_traces.json"
    with trace_json.open("w", encoding="utf-8") as stream:
        json.dump(traces, stream, indent=2, allow_nan=False)
        stream.write("\n")

    detail_json = output_dir / "evaluation.json"
    with detail_json.open("w", encoding="utf-8") as stream:
        json.dump(benchmark, stream, indent=2, allow_nan=False)
        stream.write("\n")

    report_path.write_text(_markdown_report(benchmark), encoding="utf-8")
    return {
        "details_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "horizon_csv": str(horizon_csv),
        "details_json": str(detail_json),
        "traces_json": str(trace_json),
        "report": str(report_path),
    }

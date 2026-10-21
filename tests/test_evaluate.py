import csv
import json
import math

import pytest

import physicseye.evaluate as evaluation
from physicseye.evaluate import (
    MODELS,
    build_synthetic_scenarios,
    build_tracker_scenario,
    calculate_metrics,
    evaluate_scenario,
    run_benchmark,
    write_benchmark,
)
from physicseye.tracker import track_video
from scripts.generate_sample import SCENARIOS, generate_samples


@pytest.fixture(scope="module")
def synthetic_benchmark():
    return run_benchmark(build_synthetic_scenarios())


def _without_timing(result):
    return [
        {key: value for key, value in row.items() if key != "latency_ms"}
        for row in result["rows"]
    ]


def test_metric_correctness():
    metrics = calculate_metrics([3.0, 4.0])
    assert metrics == {
        "sample_count": 2,
        "mae_px": 3.5,
        "median_error_px": 3.5,
        "rmse_px": pytest.approx(math.sqrt(12.5)),
    }
    assert calculate_metrics([])["sample_count"] == 0
    with pytest.raises(evaluation.EvaluationError, match="nonnegative"):
        calculate_metrics([1.0, -1.0])


def test_walk_forward_passes_only_causal_history(monkeypatch):
    scenario = build_synthetic_scenarios()[2]
    scenario["cutoffs_s"] = [1.0, 1.25]
    scenario["horizons_s"] = (0.25,)
    calls = []
    real_predict = evaluation.predict_trajectory

    def checked_predict(history, cutoff, horizons):
        calls.append((cutoff, max(point["time_s"] for point in history)))
        assert all(point["time_s"] <= cutoff for point in history)
        return real_predict(history, cutoff, horizons)

    monkeypatch.setattr(evaluation, "predict_trajectory", checked_predict)
    result = evaluate_scenario(scenario)

    assert calls == [(1.0, 1.0), (1.25, 1.25)]
    assert all(audit["history_max_time_s"] <= audit["cutoff_time_s"] for audit in result["audits"])
    assert all(
        update["available_at_s"] <= result["evaluation_window_s"][1]
        for update in result["adaptive_trace"]["update_trace"]
    )


def test_future_observations_do_not_change_an_earlier_result():
    original = build_synthetic_scenarios()[2]
    original["cutoffs_s"] = [1.0]
    original["horizons_s"] = (0.25,)
    changed = {**original, "observations": [dict(point) for point in original["observations"]]}
    for point in changed["observations"]:
        if point["time_s"] > 1.0 and point["valid"]:
            point["x_px"] += 10000.0

    before = evaluate_scenario(original)
    after = evaluate_scenario(changed)
    assert _without_timing(before) == _without_timing(after)


def test_seeded_benchmark_is_deterministic_without_host_timing():
    first = evaluate_scenario(build_synthetic_scenarios()[3])
    second = evaluate_scenario(build_synthetic_scenarios()[3])
    assert _without_timing(first) == _without_timing(second)
    for first_summary, second_summary in zip(first["summaries"], second["summaries"]):
        assert {
            key: value for key, value in first_summary.items() if key != "mean_latency_ms"
        } == {
            key: value for key, value in second_summary.items() if key != "mean_latency_ms"
        }


def test_missing_and_irregular_observations_are_evaluated_and_counted():
    scenario = next(
        scenario for scenario in build_synthetic_scenarios()
        if scenario["name"] == "irregular_missing_noisy"
    )
    intervals = [
        second["time_s"] - first["time_s"]
        for first, second in zip(scenario["observations"], scenario["observations"][1:])
    ]
    result = evaluate_scenario(scenario)

    assert len({round(interval, 3) for interval in intervals}) > 1
    assert any(not point["valid"] for point in scenario["observations"])
    assert len(result["rows"]) == len(scenario["cutoffs_s"]) * len(scenario["horizons_s"]) * len(MODELS)
    assert all(summary["failed_predictions"] == 0 for summary in result["summaries"])

    insufficient = {**scenario, "cutoffs_s": [0.1], "horizons_s": (0.25, 0.5)}
    failed = evaluate_scenario(insufficient)
    assert all(summary["failed_predictions"] == 2 for summary in failed["summaries"])


def test_all_models_share_cutoffs_targets_and_history(synthetic_benchmark):
    for scenario in synthetic_benchmark["scenarios"]:
        groups = {}
        for row in scenario["rows"]:
            groups.setdefault((row["cutoff_time_s"], row["horizon_s"]), []).append(row)
        for rows in groups.values():
            assert {row["model"] for row in rows} == set(MODELS)
            assert len({(row["actual_x_px"], row["actual_y_px"]) for row in rows}) == 1
            assert len({(row["history_observations"], row["history_valid_observations"], row["history_max_time_s"]) for row in rows}) == 1


def test_adaptive_acceptance_and_honest_noisy_result(synthetic_benchmark):
    acceptance = synthetic_benchmark["acceptance"]
    assert acceptance["sample_count_per_model"] == 7
    assert acceptance["adaptive_improvement_vs_frozen_pct"] == pytest.approx(83.10609105966913)
    assert acceptance["passed"] is True

    gradual = next(
        scenario for scenario in synthetic_benchmark["scenarios"]
        if scenario["name"] == "gradual_change_noisy"
    )
    adaptive = next(
        summary for summary in gradual["summaries"]
        if summary["model"] == "adaptive_deceleration"
    )
    assert gradual["measurement_mae_px"] > 0
    assert adaptive["adaptive_gain_vs_frozen_pct"] < 0


def test_exports_are_complete_and_machine_readable(tmp_path, synthetic_benchmark):
    paths = write_benchmark(
        synthetic_benchmark,
        tmp_path / "outputs",
        tmp_path / "docs" / "EVALUATION.md",
    )
    assert set(paths) == {
        "details_csv", "summary_csv", "horizon_csv", "details_json", "traces_json", "report"
    }
    assert all((tmp_path / ("docs" if key == "report" else "outputs") / path.split("/")[-1]).is_file() for key, path in paths.items())

    with open(paths["details_csv"], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["model"] for row in rows} == set(MODELS)
    assert {"actual_x_px", "predicted_x_px", "error_px"} <= rows[0].keys()

    with open(paths["details_json"], encoding="utf-8") as stream:
        exported = json.load(stream)
    assert exported["acceptance"]["passed"] is True
    assert "PASS" in open(paths["report"], encoding="utf-8").read()


def test_opencv5_tracker_pipeline_benchmark(tmp_path):
    sample = generate_samples(tmp_path / "samples")[1]
    with open(sample["ground_truth"], newline="", encoding="utf-8") as stream:
        truth = list(csv.DictReader(stream))
    scenario = build_tracker_scenario(
        "opencv_smoke",
        track_video(sample["video"]),
        truth,
        SCENARIOS["slowing_down"]["deceleration_px_s2"],
    )
    scenario["cutoffs_s"] = [1.0]
    scenario["horizons_s"] = (0.5,)
    result = evaluate_scenario(scenario)

    assert result["source"] == "opencv5_synthetic_video"
    assert result["measurement_mae_px"] <= 5.0
    assert len(result["rows"]) == len(MODELS)
    assert not result["failures"]

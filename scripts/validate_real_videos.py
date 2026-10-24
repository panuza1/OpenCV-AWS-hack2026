#!/usr/bin/env python3
"""Reproduce Phase 6 ROI tracking and causal real-video prediction metrics."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physicseye.evaluate import MODELS, calculate_metrics
from physicseye.physics import predict_trajectory
from physicseye.tracker import track_video


CONFIGURATIONS = {
    "basketball_6077402": {
        "initial_frame": 80, "roi": (1175, 120, 430, 430), "visible_frames": (80, 146)
    },
    "billiards_6254504": {
        "initial_frame": 0, "roi": (776, 1539, 240, 240), "visible_frames": (0, 321)
    },
    "disco_5751090": {
        "initial_frame": 0, "roi": (1880, 500, 1150, 1150), "visible_frames": (0, 170)
    },
}


def _read_annotations(path):
    grouped = {name: [] for name in CONFIGURATIONS}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["video"]].append(row)
    return grouped


def _tracking_metrics(observations, annotations, visible_frames, latency_ms):
    start, end = visible_frames
    visible_observations = observations[start:end + 1]
    errors = []
    on_ball = 0
    false_positives = 0
    visible_annotations = 0
    for annotation in annotations:
        observation = observations[int(annotation["frame"])]
        if annotation["ball_visible"] == "false":
            false_positives += int(observation["valid"])
            continue
        visible_annotations += 1
        if not observation["valid"]:
            continue
        error = math.hypot(
            observation["x_px"] - float(annotation["x_px"]),
            observation["y_px"] - float(annotation["y_px"]),
        )
        errors.append(error)
        on_ball += error <= float(annotation["radius_px"])
    error_metrics = calculate_metrics(errors)
    recoveries = sum(
        not first["valid"] and second["valid"]
        for first, second in zip(observations[start:end], observations[start + 1:end + 1])
    )
    return {
        "frames": len(observations),
        "visible_frame_range": [start, end],
        "visible_frames": len(visible_observations),
        "valid_visible_frames": sum(point["valid"] for point in visible_observations),
        "valid_visible_coverage_pct": 100.0 * sum(
            point["valid"] for point in visible_observations
        ) / len(visible_observations),
        "annotated_visible_frames": visible_annotations,
        "on_ball_detections": on_ball,
        "on_ball_rate_pct": 100.0 * on_ball / visible_annotations,
        "annotated_false_positives": false_positives,
        "center_error": error_metrics,
        "maximum_center_error_px": max(errors),
        "recoveries": recoveries,
        "invalid_status_counts": {
            status: sum(point.get("status") == status for point in observations if not point["valid"])
            for status in sorted({point.get("status") for point in observations if not point["valid"]})
        },
        "mean_valid_confidence": sum(
            point["confidence"] for point in observations if point["valid"]
        ) / sum(point["valid"] for point in observations),
        "tracking_latency_ms": latency_ms,
    }


def _walk_forward(observations, annotations):
    visible = [
        {
            "frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "x_px": float(row["x_px"]),
            "y_px": float(row["y_px"]),
        }
        for row in annotations if row["ball_visible"] == "true"
    ]
    errors = {model: [] for model in MODELS}
    rows = []
    audits = []
    start_time = visible[0]["time_s"]
    for cutoff, target in zip(visible, visible[1:]):
        if cutoff["time_s"] - start_time < 0.75:
            continue
        horizon_s = round(target["time_s"] - cutoff["time_s"], 10)
        history = [point for point in observations if point["time_s"] <= cutoff["time_s"]]
        result = predict_trajectory(history, cutoff["time_s"], [horizon_s])
        if result["status"] != "ok":
            continue
        history_max = max(point["time_s"] for point in history)
        updates_max = max(
            (row["available_at_s"] for row in result["adaptive_state"]["update_trace"]),
            default=None,
        )
        assert history_max <= cutoff["time_s"]
        assert updates_max is None or updates_max <= cutoff["time_s"]
        audits.append({
            "cutoff_time_s": cutoff["time_s"],
            "target_time_s": target["time_s"],
            "horizon_s": horizon_s,
            "history_max_time_s": history_max,
            "adaptive_updates": result["adaptive_state"]["update_count"],
            "adaptive_updates_max_time_s": updates_max,
        })
        for model in MODELS:
            prediction = result["models"][model]["predictions"][0]
            error = math.hypot(
                prediction["x_px"] - target["x_px"],
                prediction["y_px"] - target["y_px"],
            )
            errors[model].append(error)
            rows.append({
                "cutoff_time_s": cutoff["time_s"], "target_time_s": target["time_s"],
                "horizon_s": horizon_s, "model": model, "error_px": error,
                "predicted_x_px": prediction["x_px"], "predicted_y_px": prediction["y_px"],
                "actual_x_px": target["x_px"], "actual_y_px": target["y_px"],
            })
    summaries = {}
    for model, values in errors.items():
        summaries[model] = calculate_metrics(values)
    frozen = summaries["frozen_deceleration"]["mae_px"]
    adaptive = summaries["adaptive_deceleration"]["mae_px"]
    return {
        "horizons_s": sorted({row["horizon_s"] for row in rows}),
        "summaries": summaries,
        "adaptive_gain_vs_frozen_pct": (
            100.0 * (frozen - adaptive) / frozen if frozen and adaptive is not None else None
        ),
        "rows": rows,
        "causal_audits": audits,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("examples/real"))
    parser.add_argument("--annotations", type=Path, default=Path("docs/real_video_roi_annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real_roi_validation.json"))
    args = parser.parse_args(argv)
    annotations = _read_annotations(args.annotations)
    result = {
        "tracking": {}, "physics": {},
        "automatic_tracking": {}, "automatic_physics": {},
    }
    passing_videos = 0
    for name, configuration in CONFIGURATIONS.items():
        video = args.video_dir / f"{name}.mp4"
        if not video.is_file():
            parser.error(f"missing licensed fixture: {video}")
        started = time.perf_counter()
        observations = track_video(
            video, mode="roi_template", initial_frame=configuration["initial_frame"],
            roi=configuration["roi"],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        result["tracking"][name] = _tracking_metrics(
            observations, annotations[name], configuration["visible_frames"], latency_ms
        )
        result["physics"][name] = _walk_forward(observations, annotations[name])
        automatic_started = time.perf_counter()
        automatic_observations = track_video(video, mode="auto_motion")
        automatic_latency_ms = (time.perf_counter() - automatic_started) * 1000.0
        result["automatic_tracking"][name] = _tracking_metrics(
            automatic_observations,
            annotations[name],
            configuration["visible_frames"],
            automatic_latency_ms,
        )
        result["automatic_physics"][name] = _walk_forward(
            automatic_observations, annotations[name]
        )
        tracking = result["tracking"][name]
        if (
            tracking["on_ball_rate_pct"] >= 90.0
            and tracking["annotated_false_positives"] == 0
            and tracking["center_error"]["mae_px"] <= 40.0
        ):
            passing_videos += 1
    result["acceptance"] = {
        "required_passing_videos": 2,
        "passing_videos": passing_videos,
        "passed": passing_videos >= 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "acceptance": result["acceptance"],
        "tracking": {
            name: {
                "coverage_pct": row["valid_visible_coverage_pct"],
                "on_ball_rate_pct": row["on_ball_rate_pct"],
                "center_mae_px": row["center_error"]["mae_px"],
                "false_positives": row["annotated_false_positives"],
                "latency_ms": row["tracking_latency_ms"],
            } for name, row in result["tracking"].items()
        },
        "physics": {
            name: {
                "horizons_s": row["horizons_s"],
                "adaptive_gain_vs_frozen_pct": row["adaptive_gain_vs_frozen_pct"],
                "summaries": row["summaries"],
            } for name, row in result["physics"].items()
        },
        "automatic_tracking": {
            name: {
                "coverage_pct": row["valid_visible_coverage_pct"],
                "on_ball_rate_pct": row["on_ball_rate_pct"],
                "center_mae_px": row["center_error"]["mae_px"],
                "false_positives": row["annotated_false_positives"],
                "latency_ms": row["tracking_latency_ms"],
            } for name, row in result["automatic_tracking"].items()
        },
        "automatic_physics": {
            name: {
                "horizons_s": row["horizons_s"],
                "adaptive_gain_vs_frozen_pct": row["adaptive_gain_vs_frozen_pct"],
                "summaries": row["summaries"],
            } for name, row in result["automatic_physics"].items()
        },
        "output": str(args.output),
    }, indent=2, allow_nan=False))
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

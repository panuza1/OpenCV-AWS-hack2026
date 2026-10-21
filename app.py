"""PhysicsEye local tracking dashboard with optional direct AWS prediction."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from physicseye.dashboard import (
    MODEL_META,
    analysis_csv,
    analysis_json,
    load_benchmark,
    prepare_analysis,
    render_video_bytes,
    track_video_bytes,
)
from physicseye.cloud import execute_prediction
from physicseye.tracker import VideoError


SAMPLES = {
    "Constant speed": Path("examples/generated/constant_speed.mp4"),
    "Slowing down": Path("examples/generated/slowing_down.mp4"),
}


def _line_chart(rows, x_field, y_field, color_field, y_title):
    st.vega_lite_chart(
        {"values": rows},
        {
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": x_field, "type": "quantitative", "title": x_field.replace("_", " ")},
                "y": {"field": y_field, "type": "quantitative", "title": y_title},
                "color": {"field": color_field, "type": "nominal"},
                "tooltip": [x_field, y_field, color_field],
            },
        },
        width="stretch",
    )


def _video_source():
    source = st.radio("Video source", ("Sample", "Upload"), horizontal=True)
    if source == "Sample":
        name = st.selectbox("Sample video", tuple(SAMPLES))
        path = SAMPLES[name]
        if not path.is_file():
            st.error(f"Missing sample: {path}. Run `python scripts/generate_sample.py`.")
            return None, None
        return path.name, path.read_bytes()
    upload = st.file_uploader("Video", type=("mp4", "avi", "mov"))
    return (upload.name, upload.getvalue()) if upload else (None, None)


st.set_page_config(page_title="PhysicsEye", page_icon="👁️", layout="wide")
st.title("PhysicsEye")
st.caption("A vision system that observes motion, forecasts it, and updates physical parameters from measured errors.")
execution_mode = st.radio("Prediction execution", ("LOCAL", "AWS"), horizontal=True)
if execution_mode == "LOCAL":
    st.info("LOCAL MODE — OpenCV tracking and physics prediction run on this computer.")
else:
    st.warning("AWS MODE — OpenCV tracking stays local. Prediction runs only after you explicitly invoke the configured Lambda.")

benchmark = load_benchmark()
overview, video_tab, comparison_tab, adaptive_tab, benchmark_tab = st.tabs(
    ("Overview", "Video Analysis", "Model Comparison", "Adaptive Learning", "Benchmark Results")
)

with overview:
    st.subheader("Measured, causal trajectory forecasting")
    st.write(
        "Choose a video, track the orange ball with OpenCV 5, select a cutoff, and compare four forecasts. "
        "Predictions use only observations at or before the cutoff; later points are revealed afterward for evaluation."
    )
    if benchmark["status"] == "ok":
        acceptance = benchmark["data"]["acceptance"]
        st.metric("Held-out adaptive improvement vs frozen", f'{acceptance["adaptive_improvement_vs_frozen_pct"]:.1f}%')
        st.caption("This acceptance result is synthetic and does not establish real-world adaptive accuracy.")

with video_tab:
    st.subheader("Analyze a video")
    filename, video_bytes = _video_source()
    if filename and st.button("Analyze", type="primary"):
        try:
            st.session_state.track = track_video_bytes(filename, video_bytes)
            st.session_state.video_name = filename
            st.session_state.video_bytes = video_bytes
            st.session_state.pop("analysis", None)
            st.session_state.pop("prediction_execution", None)
            st.session_state.pop("prediction_execution_key", None)
        except (ValueError, VideoError, RuntimeError) as error:
            st.error(str(error))
    observations = st.session_state.get("track")
    if observations:
        valid_count = sum(point["valid"] for point in observations)
        duration = observations[-1]["time_s"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames", len(observations))
        col2.metric("Valid tracked positions", valid_count)
        col3.metric("Duration", f"{duration:.2f} s")
        if valid_count < 4:
            st.error("Too few valid tracked positions for physics prediction.")
        else:
            cutoff = st.slider("Prediction cutoff (s)", 0.75, max(0.75, duration), min(2.0, duration), 0.05)
            horizons = st.multiselect("Prediction horizons (s)", (0.25, 0.5, 1.0), default=(0.25, 0.5, 1.0))
            if not horizons:
                st.warning("Select at least one prediction horizon.")
            else:
                request_key = (execution_mode, cutoff, tuple(horizons), len(observations))
                execution = None
                if execution_mode == "LOCAL":
                    execution = execute_prediction("LOCAL", observations, cutoff, horizons)
                else:
                    if st.button("Invoke AWS Lambda", type="primary"):
                        with st.spinner("Invoking AWS Lambda..."):
                            st.session_state.prediction_execution = execute_prediction(
                                "AWS", observations, cutoff, horizons
                            )
                            st.session_state.prediction_execution_key = request_key
                    if st.session_state.get("prediction_execution_key") == request_key:
                        execution = st.session_state.get("prediction_execution")
                    else:
                        st.session_state.pop("analysis", None)
                        st.info("Invoke Lambda to calculate predictions for this cutoff and horizon selection.")

                if execution and execution["status"] == "error":
                    st.session_state.pop("analysis", None)
                    st.error(f'{execution_mode} prediction failed: {execution["error"]}')
                    st.caption("No local fallback was used.")
                elif execution:
                    analysis = prepare_analysis(
                        observations, cutoff, horizons, execution["prediction"]
                    )
                    analysis["execution"] = {
                        key: execution.get(key)
                        for key in (
                            "engine", "success_count", "failure_count",
                            "local_latency_ms", "parity", "metadata",
                        )
                    }
                    st.session_state.analysis = analysis
                    st.caption(
                        f'{execution["engine"]} prediction completed; local reference '
                        f'{execution["local_latency_ms"]:.3f} ms.'
                    )
                    if execution_mode == "AWS":
                        metadata = execution["metadata"]
                        cloud_cols = st.columns(4)
                        cloud_cols[0].metric("AWS invocation", f'{metadata["invocation_latency_ms"]:.2f} ms')
                        cloud_cols[1].metric("Lambda duration", f'{metadata["lambda_duration_ms"]:.2f} ms' if metadata["lambda_duration_ms"] is not None else "unavailable")
                        cloud_cols[2].metric("Parity", "PASS" if execution["parity"]["passed"] else "FAIL")
                        cloud_cols[3].metric("Max numeric difference", f'{execution["parity"]["max_absolute_difference"]:.3g}')
                        st.caption(
                            f'AWS request {metadata["request_id"] or "unavailable"}; '
                            f'{metadata["region"]}; function {metadata["function_name"]}; '
                            f'{metadata["timestamp_utc"]}'
                        )
                    st.caption("Gray is observed history. Colored paths are forecasts anchored at the cutoff. White rings after the cutoff are later observations used only for evaluation.")
                    left, right = st.columns(2)
                    left.image(render_video_bytes(st.session_state.video_name, st.session_state.video_bytes, observations, analysis, cutoff), caption="Cutoff frame")
                    target_time = min(duration, cutoff + max(horizons))
                    right.image(render_video_bytes(st.session_state.video_name, st.session_state.video_bytes, observations, analysis, target_time), caption=f"Target frame at {target_time:.2f} s")
                    st.download_button("Download analysis JSON", analysis_json(analysis), "physicseye-analysis.json", "application/json")
                    st.download_button("Download comparisons CSV", analysis_csv(analysis), "physicseye-comparisons.csv", "text/csv")

with comparison_tab:
    st.subheader("Model comparison")
    analysis = st.session_state.get("analysis")
    if not analysis or analysis["prediction"]["status"] != "ok":
        st.warning("Analyze a video to compare models.")
    else:
        rows = analysis["comparisons"]
        st.dataframe(rows, width="stretch", hide_index=True)
        scored = [row for row in rows if row["error_px"] is not None]
        if scored:
            _line_chart(scored, "horizon_s", "error_px", "model_label", "Euclidean error (px)")
        st.caption("Actual positions are tracker measurements revealed after each forecast, never predictor inputs.")
        parameters = []
        for name, model in analysis["prediction"]["models"].items():
            parameters.append({
                "model": MODEL_META[name][0],
                "speed_px_s": model["parameters"]["speed_px_s"],
                "deceleration_px_s2": model["parameters"]["deceleration_px_s2"],
            })
        st.dataframe(parameters, width="stretch", hide_index=True)

with adaptive_tab:
    st.subheader("Adaptive learning chronology")
    analysis = st.session_state.get("analysis")
    if not analysis or analysis["prediction"]["status"] != "ok":
        st.warning("Analyze a video to inspect adaptive updates.")
    else:
        state = analysis["prediction"]["adaptive_state"]
        cols = st.columns(4)
        cols[0].metric("d_fit", f'{analysis["prediction"]["models"]["fitted_deceleration"]["parameters"]["deceleration_px_s2"]:.3f} px/s²')
        cols[1].metric("d_initial", f'{state["d_initial_px_s2"]:.3f} px/s²')
        cols[2].metric("d_current", f'{state["d_current_px_s2"]:.3f} px/s²')
        cols[3].metric("Updates / skipped", f'{state["update_count"]} / {len(state["skipped_updates"])}')
        trace = [dict(row, series="deceleration") for row in state["update_trace"]]
        if trace:
            _line_chart(trace, "available_at_s", "d_after_px_s2", "series", "Estimated deceleration (px/s²)")
            error_trace = [dict(row, absolute_residual_px=abs(row["residual_px"]), series="absolute residual") for row in trace]
            _line_chart(error_trace, "available_at_s", "absolute_residual_px", "series", "Prediction residual (px)")
            st.dataframe(trace, width="stretch", hide_index=True)
        else:
            st.info("No eligible adaptive updates occurred before this cutoff.")
        if state["skipped_updates"]:
            st.write("Skipped updates")
            st.dataframe(state["skipped_updates"], width="stretch", hide_index=True)
        st.caption("Every update is shown at available_at_s, when its outcome became observable; all entries are at or before the selected cutoff.")

with benchmark_tab:
    st.subheader("Phase 3 benchmark results")
    if benchmark["status"] == "error":
        st.warning(f'Benchmark unavailable: {benchmark["error"]}. Run `python scripts/run_benchmark.py`.')
    else:
        data = benchmark["data"]
        aggregate = []
        for row in data["aggregate_summaries"]:
            aggregate.append({
                "model": MODEL_META[row["model"]][0],
                "samples": row["sample_count"],
                "MAE (px)": row["mae_px"],
                "RMSE (px)": row["rmse_px"],
                "Endpoint error (px)": row["endpoint_error_px"],
                "Adaptive gain vs frozen (%)": row.get("adaptive_gain_vs_frozen_pct"),
            })
        st.dataframe(aggregate, width="stretch", hide_index=True)
        best = min(aggregate, key=lambda row: row["MAE (px)"])
        st.info(f'Lowest aggregate MAE: {best["model"]} ({best["MAE (px)"]:.3f} px). Adaptive is not assumed to be the best model.')
        scenario_names = [scenario["name"] for scenario in data["scenarios"]]
        chosen = st.selectbox("Scenario", scenario_names)
        scenario = next(item for item in data["scenarios"] if item["name"] == chosen)
        summaries = [dict(row, model=MODEL_META[row["model"]][0]) for row in scenario["summaries"]]
        st.dataframe(summaries, width="stretch", hide_index=True)
        horizon_rows = [dict(row, model=MODEL_META[row["model"]][0]) for row in scenario["horizon_summaries"]]
        _line_chart(horizon_rows, "horizon_s", "mae_px", "model", "MAE (px)")
        adaptive = next(row for row in scenario["summaries"] if row["model"] == "adaptive_deceleration")
        gain = adaptive.get("adaptive_gain_vs_frozen_pct")
        if gain is not None:
            (st.success if gain >= 0 else st.error)(f"Adaptive gain vs frozen: {gain:.1f}%")
        st.caption(f'{scenario["description"]} Units: pixels. Evaluation window: {scenario["evaluation_window_s"]}.')
        st.caption("Synthetic and generated-video results are reproducible benchmarks; real-world adaptive performance remains unverified.")

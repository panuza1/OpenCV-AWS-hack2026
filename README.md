# PhysicsEye

PhysicsEye turns fixed-camera video of an orange rolling ball into measured trajectories, causal physics forecasts, and reproducible benchmark results. It compares constant-velocity, refitted-deceleration, frozen-parameter, and adaptive-deceleration models locally or through a private AWS Lambda function.

## Quick start

Python 3.10 was used for the verified setup on Linux x86_64.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -c 'import cv2; print(cv2.__version__); assert cv2.__version__.startswith("5.")'
python scripts/generate_sample.py
python scripts/analyze_video.py --video examples/generated/constant_speed.mp4
pytest
```

The generator creates two 5-second, 30 FPS videos under `examples/generated/`, each with a same-named CSV containing the exact rendered center for every frame. It tries MP4 (`mp4v`) first, then AVI (`MJPG`), and accepts a codec only after OpenCV reopens and decodes the expected frame count and FPS.

The analyzer writes JSON and CSV tracks under `outputs/`. Each frame includes its zero-based frame number, time in seconds, detected center and contour area, plus a `valid` flag. Missing or rejected detections have null coordinates and area. Use `--fps NUMBER` only when a video's FPS metadata is unavailable.

## Physics prediction API

`predict_trajectory` accepts observations through an explicit cutoff and rejects later observations. All models use the same current position, direction, and speed estimate; only their deceleration parameter differs.

```bash
PYTHONPATH=src python - <<'PY'
import json
from physicseye.physics import predict_trajectory

with open("outputs/slowing_down.track.json", encoding="utf-8") as stream:
    track = json.load(stream)
cutoff = 2.0
history = [point for point in track if point["time_s"] <= cutoff]
print(json.dumps(predict_trajectory(history, cutoff, [0.25, 0.5, 1.0]), indent=2))
PY
```

Predictions remain in pixels, pixels/second, and pixels/second². The adaptive trace records when each forecast was issued, when its observation became available, its residual, and the resulting bounded parameter update.

## Reproduce the Phase 3 benchmark

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_benchmark.py
```

The seeded CPU benchmark covers constant velocity, constant and changing deceleration, noise, pixel quantization, irregular timestamps, missing observations, and both generated videos through the OpenCV 5 tracker. It writes:

- `outputs/evaluation.csv`: every prediction with cutoff, target, ground truth, prediction, and error
- `outputs/evaluation_summary.csv`: scenario/model MAE, median, RMSE, endpoint, parameter, improvement, and failure metrics
- `outputs/evaluation_by_horizon.csv`: metrics for 0.25, 0.5, and 1.0-second horizons
- `outputs/evaluation.json`: full machine-readable benchmark and causal audit records
- `outputs/adaptive_traces.json`: adaptive updates and skipped-update reasons
- `docs/EVALUATION.md`: concise benchmark report including negative results

The bundled videos are synthetic fixtures. Their tracker measurements are evaluated against separately loaded rendered-center ground truth; these results do not establish accuracy on independently recorded real video.

## Phase 6 real-video validation

Three permitted Pexels clips were downloaded through their official free-download redirects and tested without changing the detector. The basketball produced 40.1% raw coverage on floor features, the white billiard ball produced no detections, and the disco clip produced 100% coverage of a hanging cord. A 32-frame manual spot check found zero on-ball detections, so tracker-relative forecast errors are documented only as pipeline diagnostics.

See [real-video validation](docs/REAL_VIDEO_VALIDATION.md) for sources, license restrictions, authorized download commands, media hashes, scene inspection, all four model results, the manual checks, and real Lambda parity evidence. Downloaded originals remain under the Git-ignored `examples/real/` directory.

## Run the Phase 4 dashboard

From the repository root:

```bash
env -u PYTHONPATH .venv/bin/streamlit run app.py
```

The Streamlit dashboard can analyze either bundled sample or an uploaded MP4, AVI, or MOV file up to 100 MB. It shows OpenCV 5 tracked history, forecasts from all four Phase 2 models, later tracker observations labeled as evaluation-only, pixel-space errors, model parameters, chronological adaptive updates and skipped-update reasons. Analysis can be downloaded as JSON or CSV.

The Benchmark Results tab reads the existing Phase 3 artifact at `outputs/evaluation.json`; run `env -u PYTHONPATH .venv/bin/python scripts/run_benchmark.py` if it is missing. The dashboard reports regressions as well as improvements and does not claim that adaptive prediction is always best.

`LOCAL` is the default execution mode. `AWS` keeps OpenCV tracking local and sends only the causal observation prefix to the configured Lambda. It has no silent fallback and displays local/cloud parity and real invocation metadata. See [AWS deployment](docs/AWS_DEPLOYMENT.md) for the request contract, least-privilege IAM setup, deployment, invocation, evidence, and cleanup commands. Cloud deployment and verification require explicit authorization and valid AWS credentials.

Predictions remain causal: the selected cutoff determines the observation prefix sent to the physics engine. Later tracker points are joined only after prediction for evaluation and are visually labeled. Pixel-space measurements are not calibrated physical units. Independently sourced real-video validation currently fails at perception, so real-world adaptive performance remains unverified.

## Tracking constraints

The detector expects one orange ball, a dark or plain background, a fixed camera, even lighting, and little occlusion. It thresholds HSV values from `(5, 100, 100)` through `(25, 255, 255)`, removes small mask noise with 5x5 morphological operations, accepts contour areas from 100 pixels² through 10% of the frame, and rejects center jumps over 50 pixels per elapsed frame. These values target the provided samples and are not generalized object recognition.

If your shell exports a ROS `PYTHONPATH`, prefix the Python and pytest commands with `env -u PYTHONPATH` to keep the virtual environment isolated.


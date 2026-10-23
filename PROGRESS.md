# PhysicsEye progress

## Current phase

**Phase 6 — COMPLETE / PASS: trustworthy ROI-initialized real-ball tracking validated** (2026-10-23, Asia/Bangkok)

Three permitted Pexels videos were tested end to end. The unchanged HSV path remains the synthetic default; a small ROI template mode now uses target appearance, local motion/size continuity, and confidence rejection. Independent annotations show trustworthy target identity in all three clips, while the physics results retain perspective, hand-contact, reflection, and annotation limitations. Full evidence is in `docs/REAL_VIDEO_VALIDATION.md`.

## Completed tasks

- Inspected the initial repository (`MASTER_PLAN.md` and the unrelated `note.pd`).
- Created `.venv` with Python 3.10.12 and installed the verified project dependencies.
- Pinned the directly used runtime and development packages.
- Confirmed OpenCV reports version 5.0.0 and exposes `VideoWriter` and `VideoCapture`.
- Added a deterministic generator for constant-speed and slowing-down orange-ball samples.
- Added MP4/AVI codec fallback and mandatory OpenCV reopen/decode validation.
- Generated both samples with per-frame CSV ground truth and documented their provenance.
- Added the Phase 0 test and quick-start instructions.
- Added the OpenCV 5 HSV detector and sequential single-ball tracker.
- Added valid/invalid per-frame observations with stable JSON semantics and CSV/JSON export.
- Added the tracking-only `scripts/analyze_video.py` CLI with explicit input and FPS errors.
- Added filtering for small/large contours and implausible center jumps.
- Compared both generated sample tracks with their ground truth and documented tracking constraints.
- Added a dependency-free causal physics engine with one shared request/result interface.
- Added constant-velocity, independently refitted deceleration, frozen-initial-parameter, and adaptive-deceleration forecasts.
- Added time-based velocity/deceleration fitting, stopping-distance handling, and support for irregular timestamps and invalid tracking frames.
- Added deterministic adaptive replay with a 0.75-second warm-up, one pending feedback forecast at a time, bounded residual updates, skipped-update reasons, and a complete update trace.
- Enforced strictly increasing timestamps and rejected every observation later than the declared cutoff.
- Added deterministic analytic, causal-ordering, missing-data, stationary/turning, numerical-input, and changing-deceleration tests.
- Added a strict walk-forward evaluator that passes only observations at or before each cutoff to the shared Phase 2 predictor.
- Added seeded scenarios for constant velocity, quantized constant deceleration, abrupt and gradual changes, Gaussian measurement noise, irregular timestamps, and missing observations.
- Added two end-to-end OpenCV 5 benchmarks using generated videos while keeping rendered-center ground truth separate from tracker observations.
- Added MAE, median, RMSE, endpoint error, per-horizon error, parameter convergence/stability, skipped-update rate, adaptive gains, win rates, failure counts, measurement error, and host latency metrics.
- Added detailed CSV/JSON exports, adaptive trace export, summary/per-horizon CSVs, and `docs/EVALUATION.md`.
- Added automated metric, causality, leakage, determinism, missing/irregular data, export, fairness, acceptance-threshold, and OpenCV pipeline tests.
- Added a five-tab Streamlit dashboard for overview, video analysis, model comparison, adaptive learning, and Phase 3 benchmark results.
- Added safe MP4/AVI/MOV upload handling with a 100 MB limit and temporary generated filenames that do not trust client paths.
- Reused the Phase 1 tracker and Phase 2 predictor without changing either implementation.
- Added cutoff-anchored annotated frames, all four model trajectories, tracker-only evaluation errors, parameter tables, adaptive update/skip traces, and JSON/CSV downloads.
- Added honest aggregate, per-scenario, and per-horizon benchmark views from the existing Phase 3 JSON artifact, including negative adaptive results.
- Added dashboard tests for causal isolation, both bundled OpenCV videos, upload validation/path traversal resistance, missing benchmark handling, and real-frame rendering.
- Added a strict dependency-free Lambda handler that calls the existing physics engine and returns all four models, per-model parameters, the adaptive trace, warnings, request context, and `engine=aws_lambda`.
- Added 1 MB, 5,000-observation, and 20-horizon request limits plus exact top-level field validation; the shared engine still enforces timestamps, finite values, observation structure, and the cutoff boundary.
- Added a direct boto3 `RequestResponse` client using the standard credential chain, configured function/region environment variables, visible errors, response validation, and no silent local fallback.
- Added exact-schema local/Lambda parity checks with a `1e-9` numerical tolerance, invocation/local/Lambda duration fields, UTC timestamp, request ID, success/failure counts, and downloadable execution metadata.
- Added a LOCAL/AWS dashboard selector. Tracking remains local; AWS prediction requires an explicit button press and reports parity and invocation metadata.
- Added a reproducible 8 KB ZIP packager/deployer containing only the Lambda handler and shared standard-library physics module; OpenCV and NumPy are not packaged.
- Added least-privilege IAM, deployment, invocation, evidence, and cleanup instructions in `docs/AWS_DEPLOYMENT.md` based on current official AWS documentation.
- Added Phase 5 tests for valid/malformed events, schema compatibility, causal rejection, missing/irregular observations, mocked invocation parity and timing metadata, errors without fallback, and package contents.
- Deployed the dedicated `physicseye-lambda-role` execution role with `AWSLambdaBasicExecutionRole` and the private `physicseye-predict` Lambda function using Python 3.13, 128 MB memory, and a 10-second timeout.
- Invoked the deployed function with the OpenCV slowing-video track, verified all four models and four adaptive updates, and measured local/cloud parity against the shared local engine.
- Verified three Pexels source pages and the Pexels license, then downloaded their UHD MP4s through the official public download actions into Git-ignored `examples/real/`.
- Inspected codec, resolution, FPS, duration, file hash, camera stability, lighting, visibility, blur, occlusion, perspective, and target-color compatibility for each real clip.
- Ran the unchanged OpenCV 5 tracker on 690 real-video frames and recorded raw coverage of 40.10% for basketball, 0% for billiards, and 100% for disco.
- Manually checked 32 frames and found zero on-ball detections: basketball followed floor features, billiards rejected the white ball, and disco followed a hanging cord.
- Ran all four predictors at equal causal cutoffs on the basketball and disco tracks, clearly labeling the resulting metrics as wrong-target tracker diagnostics; billiards could not predict because it had zero valid observations.
- Invoked the real Lambda with the basketball observation stream and verified all four models, exact schema parity, and a maximum local/cloud difference of `1.1368683772161603e-13`.
- Added dashboard ROI initialization from an actual frame with a target preview, tracking-mode selector, confidence/status, current bounding box/position, and failure warnings.
- Added CPU-friendly grayscale template matching at bounded nearby scales, a motion-constrained search, size and appearance gates, explicit invalid observations, and safe termination when the target leaves the frame.
- Preserved the original orange HSV behavior and added synthetic occlusion/recovery and real-video annotation regression tests.
- Manually annotated 34 real-video reference rows, including three absent-target basketball frames, with explicit center uncertainty and method notes.
- Improved annotated on-ball results from 0/32 with HSV to 7/7 basketball, 12/12 billiards, and 12/12 disco with ROI tracking; all annotated false-positive counts are zero.
- Added a reproducible strict walk-forward real-video validator with causal audit rows, four-model metrics, adaptive gains/regressions, tracking latency, confidence, failure statuses, and JSON export.
- Added a 60-second dashboard video limit alongside the existing 100 MB limit and documented the two-path architecture in `docs/ARCHITECTURE.md`.
- Invoked Lambda with trustworthy basketball ROI observations and verified all four models, three adaptive updates, exact schema parity, and a maximum local/cloud difference of `5.684341886080802e-14`.

## Verified environment

- OS: Linux 6.8.0-138-generic x86_64 GNU/Linux
- CPU/GPU: 12th Gen Intel Core i5-12450H / NVIDIA GeForce RTX 3050 Laptop GPU (ROI tracking used CPU only)
- Python: 3.10.12
- OpenCV package: `opencv-python-headless==5.0.0.93`; runtime `cv2.__version__ == 5.0.0`
- NumPy: 2.2.6
- Streamlit: 1.65.0
- boto3: 1.43.116
- pytest: 9.1.1
- AWS CLI: 2.38.4
- Working video output: `.mp4` with OpenCV fourcc `mp4v`; ffprobe identifies MPEG-4 video, 320x240, 30 FPS, 150 frames

## Test evidence

Dependency consistency:

```text
$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.
```

Acceptance test:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest -q
.                                                                        [100%]
1 passed in 0.39s
```

Reproducible smoke test:

```text
$ env -u PYTHONPATH .venv/bin/python scripts/generate_sample.py
constant_speed: mp4v, 30.0 FPS, 150 frames
slowing_down: mp4v, 30.0 FPS, 150 frames

$ env -u PYTHONPATH .venv/bin/python <OpenCV reopen/CSV validation snippet>
constant_speed.mp4: codec=mp4v fps=30.0 decoded=150 truth_rows=150
slowing_down.mp4: codec=mp4v fps=30.0 decoded=150 truth_rows=150
```

The smoke validation also asserted strictly increasing timestamps and in-frame coordinates. Ground-truth CSV SHA-256 values were:

```text
bbe0386cdcf6407168e3549add6068d5ce0d41244a31ecc8b0d634fc39512e71  constant_speed.csv
9b5df91ec0bca8f9f1725b6eed1240b46aecf0949f965b93d7650d148e686248  slowing_down.csv
```

Phase 1 tests:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_tracker.py -q
.....                                                                    [100%]
5 passed in 0.99s

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
......                                                                   [100%]
6 passed in 1.21s
```

Phase 1 CLI smoke test:

```text
$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/generated/constant_speed.mp4
frames=150 valid_frames=150

$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/generated/slowing_down.mp4
frames=150 valid_frames=150

constant_speed: frames=150 valid=150 invalid=0 MAE=0.0355px
slowing_down: frames=150 valid=150 invalid=0 MAE=0.0321px

$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video does-not-exist.mp4
error: Cannot open video: does-not-exist.mp4
exit status 2
```

Phase 2 tests:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_physics.py -q
..........                                                               [100%]
10 passed in 0.03s

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
................                                                         [100%]
16 passed in 1.03s
```

Changing-deceleration acceptance case (`v0=40 px/s`, `d=2→10 px/s²` at `t=1.5 s`, `dt=0.05 s`, 0.25-second horizon, seven post-change cutoffs):

```text
frozen mean error:   0.265837585 px
adaptive mean error: 0.044910360 px
relative improvement: 83.106%
```

Every target used for those errors was held out from the corresponding predictor input. All models received the identical observation prefix and current-motion estimate. This passes the requested >=10% adaptive-versus-frozen criterion on the declared deterministic scenario; it is not a claim about real video.

Phase 2 passes its acceptance criteria: analytic forecasts, stopping distance, invalid-input status, deterministic replay, causal separation, finite outputs, exact residual-update math, and the requested adaptive-improvement threshold are covered by passing tests.

OpenCV-to-physics smoke test on the generated slowing-down video at a 2.0-second cutoff:

```text
OpenCV 5.0.0; 61 historical tracked frames; status=ok
models=constant_velocity,fitted_deceleration,frozen_deceleration,adaptive_deceleration
adaptive_updates=4; d_initial=17.273275809 px/s²; d_current=14.974707429 px/s²
```

Phase 3 tests and full regression suite:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_evaluate.py -q
.........                                                                [100%]
9 passed in 0.64s

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
.........................                                                [100%]
25 passed in 1.49s

$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.
```

Reproducible benchmark smoke test:

```text
$ env -u PYTHONPATH .venv/bin/python scripts/run_benchmark.py
phase3_acceptance=PASS
adaptive_improvement_vs_frozen_pct=83.10609105966913
scenario_count=7
prediction_rows=924
```

Aggregate results across all seven scenarios and three horizons:

| Model | n | MAE px | RMSE px | Endpoint px |
|---|---:|---:|---:|---:|
| Constant velocity | 231 | 1.329688 | 2.168196 | 2.659600 |
| Refitted deceleration | 231 | 1.306162 | 2.242586 | 2.427702 |
| Frozen deceleration | 231 | 1.496590 | 2.417208 | 2.963820 |
| Adaptive deceleration | 231 | 1.358028 | 2.348350 | 2.630440 |

The designated abrupt-change acceptance window retained the Phase 2 boundary (`2.0–3.5 s`, `0.25 s` horizon, seven targets): frozen MAE `0.265837585 px`, adaptive MAE `0.044910360 px`, improvement `83.106%` — **PASS** against the required 10%.

Negative results were retained: adaptive MAE was 27.097% worse than frozen on gradual noisy motion, 170.806% worse on the OpenCV constant-speed fixture, and 17.207% worse on the OpenCV slowing fixture. Adaptive aggregate MAE was 9.259% better than frozen but 3.971% worse than independently refitted deceleration.

Phase 4 dashboard verification:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_dashboard.py -q
.....                                                                    [100%]
5 passed

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
..............................                                           [100%]
30 passed

$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.

$ env -u PYTHONPATH .venv/bin/streamlit run app.py --server.headless true
Local URL: http://localhost:8501
```

The Streamlit headless app test loaded all five tabs with no exception, selected and analyzed each bundled sample, found all 150 tracked positions, and rendered calculated model/learning results. The server startup smoke test succeeded. No graphical browser surface was available in this environment, so visual browser inspection and screenshots were not completed.

Phase 5 tests, package smoke test, and real AWS verification:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_lambda_handler.py -q
..........                                                               [100%]
10 passed in 0.24s

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
........................................                                 [100%]
40 passed in 1.81s

$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.

$ PYTHON=.venv/bin/python scripts/deploy_lambda.sh package /tmp/.../physicseye-lambda.zip
lambda_handler.py
physicseye/__init__.py
physicseye/physics.py
8.0K total ZIP size
```

The packaged handler was imported directly from the ZIP and executed 50 times against the OpenCV slowing-video track at cutoff `2.0 s`: all four models returned, the adaptive trace contained four updates, and measured local handler latency was median `0.299204 ms`, minimum `0.292669 ms`, maximum `0.395071 ms`. These are host-local timings, not AWS measurements.

The mocked boto3 test returned exact local/Lambda parity (`max_absolute_difference=0`) and verified parsing of a synthetic `1.23 ms` Lambda duration. That duration is test fixture data and is not reported as cloud performance. The Streamlit harness also verified LOCAL analysis and a visible AWS missing-configuration error with no fallback.

Real AWS evidence captured with `AWS_PROFILE=physicseye-cli` and `AWS_REGION=ap-southeast-1`:

```text
IAM identity: arn:aws:iam::502657994526:user/physicseye-cli
Execution role: arn:aws:iam::502657994526:role/physicseye-lambda-role
Function: arn:aws:lambda:ap-southeast-1:502657994526:function:physicseye-predict
Runtime/state: python3.13 / Active
UTC invocation timestamp: 2026-10-23T06:19:56.617656+00:00
AWS request ID: 009fd967-62b1-418d-acc3-753664a0148d
Invocation latency: 429.531348 ms
Lambda duration: 17.25 ms
Lambda billed duration: 106 ms
Local prediction latency: 0.345083 ms
Adaptive updates: 4
Parity tolerance: 1e-9
Maximum absolute difference: 5.3290705182007514e-14
Schema mismatches: 0
Result: PASS
```

This was a real synchronous cold invocation through boto3 using the standard named-profile credential chain. The request contained only 61 causal tracker observations at or before the `2.0 s` cutoff and horizons `[0.25, 0.5, 1.0]`; no video or future observation was sent to Lambda.

Phase 6 ROI tracker and real-video validation:

```text
$ env -u PYTHONPATH .venv/bin/python scripts/validate_real_videos.py
acceptance: required=2 passing=3 passed=true
basketball: visible coverage=100%; on-ball=100%; annotated FP=0; center MAE=16.121255 px
billiards: visible coverage=100%; on-ball=100%; annotated FP=0; center MAE=18.143355 px
disco: visible coverage=100%; on-ball=100%; annotated FP=0; center MAE=3.680787 px
```

The baseline HSV spot check was 0/32 on-ball. ROI evaluation uses 31 visible center annotations and three absent-target frames. A deterministic synthetic test separately proves invalid output and recovery after a four-frame occlusion; none of the real clips contains an unambiguous recovery event.

Strict walk-forward physics results use only tracker observations at or before each cutoff and reveal the next manual target afterward. Basketball (`n=4` per model, 0.4 s horizon) produced MAE `29.877476 px` for every model and 0% adaptive gain. Billiards (`n=10`, 1.16–1.20 s) produced MAE `1288.729656 / 1055.900189 / 1288.729656 / 838.960819 px` for constant/refitted/frozen/adaptive and 34.900% adaptive gain, but repeated hand impulses make the errors physically unsuitable. Disco (`n=9`, 0.60–0.64 s) produced MAE `18.801271 / 15.184220 / 15.527389 / 13.657234 px` and 12.044% adaptive gain; its ±25 px manual uncertainty is larger than the model difference. No calibrated physical-accuracy claim is made.

Real-video AWS parity at the basketball `4.8 s` cutoff:

```text
UTC: 2026-10-23T07:37:43.527915+00:00
AWS request ID: a2bddbaa-c2dd-4943-9b41-fc837936c4c9
Invocation / Lambda / billed duration: 397.621249 ms / 11.43 ms / 86 ms
Local prediction latency: 0.246337 ms
Models / adaptive updates: 4 / 3
Parity tolerance: 1e-9
Maximum absolute difference: 5.684341886080802e-14
Schema mismatches: 0
Result: PASS
```

Phase 6 targeted regression verification:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/test_tracker.py tests/test_dashboard.py tests/test_real_video_tracker.py -q
................                                                         [100%]
16 passed in 12.08s

$ env -u PYTHONPATH .venv/bin/python -m pytest -q
..............................................                           [100%]
46 passed in 11.71s

$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.

$ fresh_venv/bin/python -m pytest -q
..............................................                           [100%]
46 passed in 12.91s

$ env -u PYTHONPATH .venv/bin/streamlit run app.py --server.headless true --server.port 8502
Uvicorn server started on :::8502
$ curl --fail http://localhost:8502/_stcore/health
ok
```

The fresh virtual environment was created with Python 3.10.12 and installed only `requirements.txt` plus `requirements-dev.txt`; OpenCV reported `5.0.0` before the full suite ran.

## Known issues

- This host exports a ROS `PYTHONPATH`, which exposes unrelated ROS packages even inside a normal virtual environment. Verification commands used `env -u PYTHONPATH`; the README documents this host-specific workaround.
- Codec support depends on the OpenCV build and host. The generator validates `mp4v` and falls back to `MJPG`/AVI rather than assuming either works.
- Generated videos and CSVs are reproducible local artifacts under `examples/generated/` and are intentionally Git-ignored.
- HSV thresholds target one orange ball on a plain dark background. ROI mode handles the three validation targets but assumes a fixed camera and recognizable appearance; long occlusion, extreme scale change, or a similar nearby object can still produce invalid or incorrect detections.
- Video time uses `frame / fps`; variable-frame-rate media may require transcoding or an explicit trustworthy FPS override.
- Physics estimates assume approximately straight, direction-stable motion in image space. They do not model turns, collisions, perspective, or calibrated physical units.
- Adaptive feedback uses the first valid observation at/after its 0.25-second target, up to 0.125 seconds late; a later target is logged and skipped.
- The required adaptive threshold passes only on the designated noiseless abrupt-change scenario; noisy and tracker scenarios include losses and do not support a universal-improvement claim.
- OpenCV benchmark fixtures are generated videos, not independently recorded real motion. Their tracking MAE is `0.035545 px` for constant speed and `0.032061 px` for slowing motion.
- Ground-truth targets falling between video frames are linearly interpolated. Latency is host-dependent and excluded from deterministic assertions.
- The dashboard accepts prerecorded uploads only; Phase 4 deliberately excludes webcam streaming.
- The three independently sourced real clips remain Git-ignored under `examples/real/` because they total about 63 MB and the Pexels license restricts standalone redistribution. Reproduction uses the official source links and downloads documented in `docs/REAL_VIDEO_VALIDATION.md`.
- The 31 visible and three absent manual reference rows validate target identity only at sparse frames. Annotation uncertainty is ±8–25 px; the disco model differences are below that uncertainty.
- Basketball perspective creates apparent acceleration, billiards hands repeatedly redirect the ball, and disco reflections/near-stationarity violate or weaken the simple deceleration model. The real results do not establish calibrated physical parameters or universal adaptive benefit.
- Dashboard actual points are later tracker measurements, not calibrated ground truth. They are used only after forecasts are produced and may contain tracker error.
- The deployed Lambda function, execution role, and CloudWatch log group are AWS resources that may incur charges until removed with the documented cleanup commands. They were intentionally left deployed for the dashboard and competition demonstration.
- The recorded `429.531348 ms` invocation and `106 ms` billed duration include a cold start and are one measurement, not a latency benchmark.
- `LogType=Tail` allows the client to parse Lambda duration when present, but AWS documents that returning log tails can add to synchronous response latency.

## Next action

Phase 6 is complete. Phase 7 is the next planned phase but has not been started; begin it only on explicit request.


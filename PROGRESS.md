# PhysicsEye progress

## Current phase

**Phase 6 — BLOCKED: internet-video validation complete; detector unsuitable for supplied clips** (2026-10-23, Asia/Bangkok)

Three permitted Pexels videos were downloaded through official free-download redirects and tested end to end. The unchanged orange HSV tracker did not follow the target ball in any clip, so real-ball prediction accuracy and adaptive improvement cannot be measured honestly. The failure evidence, tracker-relative pipeline diagnostics, and real-video local/AWS parity are documented in `docs/REAL_VIDEO_VALIDATION.md`.

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

## Verified environment

- OS: Linux 6.8.0-138-generic x86_64 GNU/Linux
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

Phase 6 internet-video validation:

```text
$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/basketball_6077402.mp4 --output-dir outputs/real
frames=197 valid_frames=79

$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/billiards_6254504.mp4 --output-dir outputs/real
frames=322 valid_frames=0

$ env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/disco_5751090.mp4 --output-dir outputs/real
frames=171 valid_frames=171

manual spot checks: 32 visible-ball frames; 15 accepted wrong-object detections; 0 on-ball detections
```

Tracker-relative four-model diagnostics were available for basketball (`n=25` per model) and disco (`n=39` per model). Basketball adaptive MAE was `342.075553 px`, identical to frozen because no eligible updates occurred; disco adaptive MAE was `3.218922 px`, but this predicts a cord rather than a ball. Billiards produced no prediction samples. These are not real-ball accuracy results.

Real-video AWS parity at the basketball `5.0 s` cutoff:

```text
UTC: 2026-10-23T06:32:30.492919+00:00
AWS request ID: 5e9b648a-1e66-4e65-a53d-40e8a2a81283
Invocation / Lambda / billed duration: 463.882241 ms / 3.36 ms / 113 ms
Local prediction latency: 0.167895 ms
Parity tolerance: 1e-9
Maximum absolute difference: 1.1368683772161603e-13
Schema mismatches: 0
Result: PASS
```

Phase 6 regression verification:

```text
$ env -u PYTHONPATH .venv/bin/python -m pytest -q
........................................                                 [100%]
40 passed in 1.46s

$ env -u PYTHONPATH .venv/bin/python -m pip check
No broken requirements found.

$ env -u PYTHONPATH .venv/bin/python <spot-check CSV assertions>
spot_checks=32 visible=32 detector_valid=15 on_ball=0
```

## Known issues

- This host exports a ROS `PYTHONPATH`, which exposes unrelated ROS packages even inside a normal virtual environment. Verification commands used `env -u PYTHONPATH`; the README documents this host-specific workaround.
- Codec support depends on the OpenCV build and host. The generator validates `mp4v` and falls back to `MJPG`/AVI rather than assuming either works.
- Generated videos and CSVs are reproducible local artifacts under `examples/generated/` and are intentionally Git-ignored.
- Tracking thresholds target one orange ball on a plain dark background with a fixed camera. Lighting shifts, occlusion, motion blur, similarly colored objects, and camera motion can produce invalid or incorrect detections.
- Video time uses `frame / fps`; variable-frame-rate media may require transcoding or an explicit trustworthy FPS override.
- Physics estimates assume approximately straight, direction-stable motion in image space. They do not model turns, collisions, perspective, or calibrated physical units.
- Adaptive feedback uses the first valid observation at/after its 0.25-second target, up to 0.125 seconds late; a later target is logged and skipped.
- The required adaptive threshold passes only on the designated noiseless abrupt-change scenario; noisy and tracker scenarios include losses and do not support a universal-improvement claim.
- OpenCV benchmark fixtures are generated videos, not independently recorded real motion. Their tracking MAE is `0.035545 px` for constant speed and `0.032061 px` for slowing motion.
- Ground-truth targets falling between video frames are linearly interpolated. Latency is host-dependent and excluded from deterministic assertions.
- The dashboard accepts prerecorded uploads only; Phase 4 deliberately excludes webcam streaming.
- The three independently sourced real clips remain Git-ignored under `examples/real/` because they total about 63 MB and the Pexels license restricts standalone redistribution. Reproduction uses the official source links and downloads documented in `docs/REAL_VIDEO_VALIDATION.md`.
- The current tracker failed to follow the target ball in all three real clips. Raw valid-frame coverage can be high while following the wrong object, so real-world tracking and adaptive performance remain unverified.
- Dashboard actual points are later tracker measurements, not calibrated ground truth. They are used only after forecasts are produced and may contain tracker error.
- The deployed Lambda function, execution role, and CloudWatch log group are AWS resources that may incur charges until removed with the documented cleanup commands. They were intentionally left deployed for the dashboard and competition demonstration.
- The recorded `429.531348 ms` invocation and `106 ms` billed duration include a cold start and are one measurement, not a latency benchmark.
- `LogType=Tail` allows the client to parse Lambda duration when present, but AWS documents that returning log tails can add to synchronous response latency.

## Next action

Phase 6 is blocked on a trustworthy real-ball trajectory. Record a matching MVP scene with one saturated orange ball on a plain contrasting surface, or separately validate a scoped ROI/color-calibration change with independent center annotations. Do not start Phase 7 until Phase 6 has meaningful real-ball metrics.


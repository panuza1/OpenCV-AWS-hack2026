# PhysicsEye — Master Implementation Plan

> **Competition:** OpenCV AI Competition 2026, powered by AWS  
> **Project:** PhysicsEye — *A Vision-Based System That Predicts, Observes, and Learns from Object Motion*  
> **Document status:** Research-informed implementation plan; repository progress is tracked separately in `PROGRESS.md` (Phases 0–1 complete at this revision).  
> **Plan date:** 2026-10-21 (Asia/Bangkok); research revision based on the repository heads listed below.

## 0. Codex operating contract — READ FIRST

This plan is the single source of truth for building PhysicsEye. Work **one phase at a time**, and **stop after the current phase**. Do not begin another phase until explicitly requested.

For every phase:
1. Inspect the existing repository and relevant files before editing.
2. Implement **only** that phase's scope and its necessary tests; preserve prior working behavior.
3. Use the simplest working solution. No unrequested frameworks, architecture layers, neural networks, databases, paid AI APIs, or speculative features.
4. Check library/API availability before writing code. Do not invent methods, API arguments, AWS resources, or benchmark results.
5. Run applicable tests and a reproducible smoke test. Report the exact commands and outcomes.
6. Update `PROGRESS.md`: current phase, completed tasks, test evidence, known issues, next phase. Create it in Phase 0.
7. At completion, summarize files changed, decisions, how to run, test results, and blockers; **then stop**.
8. If AWS credentials/hardware are unavailable, complete offline portions, clearly mark cloud validation blocked, and do not claim the phase has fully passed.
9. Never commit secrets, `.env`, personal videos, AWS tokens, or unnecessary binary data. Never create paid AWS infrastructure without explicit authorization.
10. If a phase cannot meet its acceptance criteria, mark **BLOCKED**, explain why, and do not silently substitute a lower standard.

**The final project must not rely on simulated results being presented as real-world measurements.**

---

## 1. Product and competition goal

### One-sentence pitch

PhysicsEye turns ordinary videos of a moving object into measurable trajectories, **physics-informed future-position predictions**, and an adaptive model that updates its measured deceleration from earlier prediction errors before making later predictions.

### Real-world problem

Predicting physical motion is useful for visual process monitoring, laboratory demonstrations, robotics prototyping, and quality control, but most basic demos only detect objects in the present frame. PhysicsEye adds *forecasting* and *measurement* to a conventional OpenCV pipeline.

### Intended user and scenario

- Primary user: student, engineer, or developer testing simple motion predictions from a video.
- **MVP scene:** one brightly colored ball rolling on a flat, matte surface, recorded by a **fixed overhead camera** (phone or webcam).
- Demo constraints: one object, visible throughout most frames, reasonably uniform lighting, no major occlusions or camera movement.
- Output units: **pixels and pixels/second** unless an actual spatial calibration is implemented and documented. Never label pixel distances as meters.

### Demonstrable user journey

1. Upload a short video (or use a bundled, legitimately sourced sample).
2. OpenCV 5 detects the colored ball, tracks its position over time, and extracts timestamped measurements.
3. Pick an observation cutoff frame: only samples at or before this point are input to a prediction.
4. Send observations to an AWS Lambda prediction service (or explicitly labeled local fallback during development).
5. Predict the ball's position at short future horizons using (a) constant velocity, (b) a non-adaptive deceleration model fitted from observations available at that cutoff, and (c) a persistent deceleration model updated only from outcomes already observed before the current cutoff.
6. Compare each prediction with *later observed video frames*, then use an eligible past residual to update the adaptive model for the next cutoff.
7. Show annotated frames, trajectory overlays, prediction errors, the learned deceleration trace, and an exportable JSON/CSV report.

### What makes it compelling

- Video perception + interpretable physical model, not black-box detection alone.
- Prediction is made **before** future frames are inspected for accuracy.
- The adaptive model has a measurable contract: it starts from the same parameter as the ordinary physics model at the first eligible cutoff and may update only after the corresponding outcome becomes observable.
- The UI shows where the system is wrong and explains failures.
- AWS executes a **meaningful prediction workload**, not just static file storage.
- Quantitative comparison against a constant-velocity baseline, including failed cases.

### Non-goals for the competition MVP

No multi-object tracking, hand/robot control, 3D reconstruction, collisions/bounces, live distributed streaming, neural video generation, reinforcement learning, agent frameworks, large-model training, 3D units without calibration, or unsafe physical-control claims. These belong in post-submission extensions only.

---

## 2. Official competition constraints and evidence

Competition sources:
- Overview: https://opencv26.devpost.com/
- Rules: https://opencv26.devpost.com/rules
- OpenCV 5 installation: https://docs.opencv.org/5.0/py_tutorials/py_setup/py_pip_install/py_pip_install.html
- AWS Lambda docs: https://docs.aws.amazon.com/lambda/

**Must satisfy:**
- Substantive use of **OpenCV 5** in real image/video analysis.
- A meaningful application component running on **AWS** and an explanation of what runs there.
- Judge-accessible repo/archive with pinned dependencies and reproducible instructions.
- Technical report: problem, target users, architecture, OpenCV implementation, AWS, evaluation, limits, responsible use.
- Architecture diagram and working endpoint **or** arranged live screen-share demo.
- Public/unlisted judge-accessible demo video **at most 5 minutes**.
- Measured outcomes and failure cases.

**Submission deadline discrepancy:** the overview/header displays **Oct 26, 2026, 11:45 p.m. PDT**, while overview body says **11:59 p.m. Pacific Time**. Treat **11:45 p.m. PDT as the conservative hard deadline**, confirm within Devpost, and aim to submit on **Oct 25–26 Bangkok time** rather than rely on the last minute.

**Judging priorities:** Technical Execution 30%, Innovation 20%, Real-world Impact 20%, User Experience 10%, Documentation/Presentation 10%, AWS/reproducibility/responsible operation 10%.

**Special award choice:** Do **not** advertise Agentic Vision eligibility unless the running system genuinely uses OpenCV outputs to change a later action/tool call. A physics model adjusting coefficients is not automatically an autonomous agent. The reliable target is a strong **overall-award submission**.

---

## 3. Engineering constraints and design decisions

### Research review (verified 2026-10-21)

Method: each public repository was opened on GitHub and its checked-out default-branch head, source files, dependency declarations and root license files were inspected. The repositories were **not executed** because this planning task forbids installing their dependencies. “OpenCV 5 compatibility” below therefore means API-level compatibility from static inspection against the installed OpenCV 5.0.0 API, not an end-to-end runtime claim.

| Project (inspected head) | Actual approach and dependencies | License / OpenCV 5 status | Reusable lesson, limitation, complexity and fit |
|---|---|---|---|
| [trflorian/ball-tracking-live-plot](https://github.com/trflorian/ball-tracking-live-plot) (`2c6270a`, 2025-06-30) | HSV mask + MOG2 foreground mask, morphology, largest contour, short `deque`, optional exponential position smoothing; a separate demo uses NumPy polynomial fits and Matplotlib/Rerun. Declares Python >=3.13, OpenCV >=4.10, Matplotlib and Rerun. | MIT. Its inspected `cvtColor`, `inRange`, MOG2, morphology, two-return `findContours` and writer APIs exist in OpenCV 5, but its application was not run on OpenCV 5. | Adapt the **idea** of inspecting separate color/motion masks and bounded history; do not add Matplotlib/Rerun to the MVP. Low complexity, CPU-suitable. Existing Phase 1 already implements the smaller HSV-only path. |
| [MehakArora/Ball_Tracking](https://github.com/MehakArora/Ball_Tracking) (`7ad5a5c`, 2020-05-14) | Grayscale MOG2, contour merging/shape matching and a custom six-state position/velocity/acceleration Kalman filter using NumPy; also imports OpenCV and Matplotlib. | **No license found: no code reuse.** Most called OpenCV APIs still exist, but `run.py` contains invalid import syntax and the Kalman `dt` multiplies by FPS instead of frame duration, so the repository is not runnable/correct as inspected. | Kalman predict/correct is a useful optional smoothing reference, but implement independently with `cv2.KalmanFilter` only if measured jitter warrants it. Medium validation effort; CPU-suitable after repair. |
| [NattyBumppo/Ball-Tracking](https://github.com/NattyBumppo/Ball-Tracking) (`3a7e61c`, 2018-03-18) | OpenCV 2/Python 2 HSV filtering, MOG2, k-means for several balls, consecutive-frame velocity, Euler gravity rollout and fixed weighted blending of prediction/observation. Hard-codes FPS and a guessed pixels-per-meter value. | **No license found: no code reuse.** Incompatible as-is with Python 3/OpenCV 5: Python 2 `print` statements and the old three-return `findContours` signature are present. | The explicit predict→observe→correct loop motivates PhysicsEye’s adaptive loop, but multi-ball k-means, guessed calibration and fixed weights are unsuitable. Medium-to-high porting effort; algorithm is CPU-friendly but obsolete. |
| [turman17/tennis-ball-detection-and-tracking](https://github.com/turman17/tennis-ball-detection-and-tracking) (`be46b32`, 2026-03-08) | Notebook trains YOLO11n with Ultralytics, estimates range from apparent ball diameter and focal-length calibration, uses nearest-neighbor association + `cv2.KalmanFilter`, windowed velocity and a fixed per-step friction factor (`0.93`). Requires Ultralytics, OpenCV, NumPy and Jupyter; configured for Apple MPS. | **No license found: no code or dataset reuse.** No pinned versions, trained weight or complete licensed dataset was found. OpenCV calls are available in OpenCV 5, but Ultralytics/OpenCV 5 dependency compatibility and the notebook were not run. | Calibration caveats, windowed velocity and Kalman state design are useful ideas. YOLO training and a second OpenCV wheel risk the deadline and 4 GB GPU. High complexity; reject for mandatory MVP. |
| [vincent-leguen/PhyDNet](https://github.com/vincent-leguen/PhyDNet) (`23a992d`, 2022-03-01) | PyTorch PhyCell + ConvLSTM two-branch unsupervised frame prediction with PDE-moment constraints; Moving-MNIST example defaults to 2,001 epochs and includes a checkpoint. Depends on PyTorch, torchvision, NumPy, SciPy and scikit-image. | MIT. It does not use OpenCV, and old code imports removed `skimage.measure.compare_ssim`; modern dependency compatibility is unverified. | Strong research reference for separating physical structure from residual appearance, but it predicts pixels, requires neural training and does not expose the interpretable ball parameters needed here. Very high complexity; unsuitable for the RTX 3050 4 GB/deadline. |
| [ccyydd/PhysMind](https://github.com/ccyydd/PhysMind) (`5d39b16`, 2026-09-13) | Training-free but very large executable-world stack: VLM planning, SAM-based tracking/segmentation, metric depth, 3D reconstruction, system identification, differentiable analytic rollout, PyBullet/Blender and benchmark tooling. Its friction path initializes velocity/deceleration by least squares, fits bounded physical parameters against trajectory RMSE, then rolls out futures/counterfactuals. | **No root license found: no code reuse.** Pins OpenCV 4.10 plus PyTorch/CUDA/xFormers and many compiled/large-model packages; requires CUDA 12, gated weights, submodules and optional paid model APIs. Not compatible with the current minimal OpenCV 5 environment as a package set. | Adopt only the principles of bounded system identification, held-out rollout and visible fit diagnostics. The full stack exceeds 4 GB VRAM and the hackathon schedule. Very high complexity; reject for MVP. |

License rule: only the MIT-licensed projects above permit code adaptation subject to preserving their notices. PhysicsEye will still implement its small tracker and physics code directly so behavior, tests and OpenCV 5 compatibility remain controlled. Public repositories without a license are reference material only; their code, datasets and media must not be copied.

### Research-informed scope decisions

**Mandatory MVP, implemented directly:**

- Keep the existing OpenCV 5 HSV/contour tracker and raw valid/invalid observations. Add MOG2 or Kalman filtering only after a measured real-video failure shows it is needed.
- Estimate velocity with a short time-based least-squares fit rather than a single frame difference.
- Compare three causal predictors: constant velocity, a non-adaptive constant-deceleration fit, and a persistent adaptive version. Retain a frozen-initial-parameter ablation in evaluation to prove that updates have an effect.
- Learn one bounded scalar, deceleration in `px/s²`, from revealed forecast residuals. This captures the useful system-identification idea from PhysMind without its 3D/neural stack.
- Show the parameter history, update count, warnings and before/after error. A model is not “learning” merely because it fitted one batch once.
- Keep the Lambda dependency-light: the shared prediction/adaptation code uses the Python standard library where practical, avoiding compiled NumPy packaging in Lambda.

**Optional only after the mandatory gates pass:**

- OpenCV Kalman smoothing, compared against raw measurements and disabled by default. Never score a smoothed point as if it were independently observed ground truth.
- A clearly labeled what-if control that varies deceleration without changing the fitted parameter or evaluation record.
- Camera calibration, bounce/collision logic, and background-motion masking.
- YOLO, PhyDNet, full 3D reconstruction and VLM/agent pipelines remain out of scope for the competition MVP.

| Technique | MVP decision | Reason and validation gate |
|---|---|---|
| Ball detection/tracking | **Mandatory; keep Phase 1 HSV + morphology + contours** | Already passes synthetic tracking with OpenCV 5. Test on real light/background variation before adding MOG2. |
| Kalman filtering | Optional | OpenCV 5 supports it, but smoothing can hide detector failures and contaminate “actual” targets. Add only if held-out raw-versus-Kalman metrics improve. |
| Velocity/acceleration estimation | **Mandatory** | Fit `x(t)` and `y(t)` slopes over several recent valid timestamps; derive speed changes with real `dt`. Unit-test irregular gaps and noise. |
| Physics trajectory prediction | **Mandatory** | Constant velocity and straight-line constant deceleration are interpretable, analytic and cheap. Test stopping distance and no-future-data rules. |
| Rolling friction/deceleration | **Mandatory as effective `d`** | Learn `px/s²`; do not label it friction coefficient `mu` without calibrated geometry and gravity/surface assumptions. |
| Bounce/collision prediction | Optional after submission core | Requires event/wall detection, restitution and separate evaluation; it changes the MVP scene and risks the deadline. |
| Adaptive physical parameter estimation | **Mandatory innovation** | Use the residual update and chronological trace above; compare against non-adaptive and frozen ablations. |
| Prediction error analysis | **Mandatory** | Euclidean position error by horizon, coverage, latency, adaptive gain and failure counts make claims auditable. |
| What-if physics simulation | Optional | A slider is inexpensive but is hypothetical; label it clearly and exclude it from observed-performance metrics. |

### Core innovation and measurable adaptive contract

**Concept:** “A Vision System That Learns Physical Dynamics from Observation and Improves Its Predictions Over Time.”

Ordinary prediction independently fits `d` from the recent observations available at each cutoff and does not retain forecast-error feedback. PhysicsEye’s adaptive run carries an auditable model state through a chronological sequence:

1. Fit initial velocity and nonnegative deceleration using only the warm-up observations.
2. At cutoff `t_i`, save the prediction and the exact model state used.
3. Only after the target at `t_i + h_update` is observed, compute the signed displacement residual along the prior motion direction: `e_s = s_pred - s_actual`.
4. For a non-stopping, direction-stable interval, derive the residual-corrected candidate `d_candidate = clip(d + 2*e_s/h_update², 0, v/h_update)` and apply a predeclared exponential update `d_next = (1-alpha)*d + alpha*d_candidate`.
5. Use `d_next` only for a later cutoff. Log skipped updates caused by missing observations, near-zero speed, a turn, a stop inside the feedback interval or a non-finite value.

Use `h_update = 0.25 s` and choose a single `alpha` before the held-out benchmark; the initial plan value is `alpha = 0.25`. Any change must be recorded before evaluating the final real-video test set. This small update is interpretable, CPU-friendly and directly tied to a measured forecast error.

The adaptive claim is supported only when a chronological benchmark reports all of the following against the **same cutoffs and targets**:

- constant-velocity error;
- non-adaptive deceleration error (fresh causal fit at each cutoff) plus a frozen-initial-`d` ablation;
- adaptive-deceleration error and per-cutoff `d` history;
- adaptive gain against both non-adaptive and frozen physics, win rate and sample count.

Required deterministic proof: on a held-out synthetic trajectory whose deceleration changes after warm-up, the adaptive model must reduce post-change mean position error by at least 10% relative to the frozen model. This threshold does not apply to every real video; real losses and negative gains must be reported honestly.

### Measurement validity and practical limits

- With an uncalibrated fixed overhead camera, positions, speeds and deceleration remain `px`, `px/s` and `px/s²`. The fitted `d` is an image-space effective deceleration, **not** a dimensionless friction coefficient and not proof of a material property.
- Perspective makes pixel scale vary across the frame. Keep the ball on a nearly planar surface and within a limited field of view; optional spatial calibration must use a measured reference plane before reporting centimeters.
- Apparent ball diameter alone does not provide trustworthy planar distance under arbitrary camera pose. Do not adopt the tennis repository’s range formula without a documented camera model and calibration capture.
- Detection jitter can masquerade as acceleration and adaptation can absorb lighting/tracking bias as if it were physics. Use time-window fits, parameter bounds, skipped-update warnings and raw-versus-fit plots.
- Tracked future positions are evaluation observations, not perfect ground truth. Synthetic data supplies exact ground truth; real-video reporting must state that forecast error includes tracking error and should include manual spot checks or uncertainty notes.
- The constant-deceleration model assumes approximately straight rolling motion on one surface. Turns, slopes, impacts, bounces, camera motion and rolling-to-sliding transitions violate the model and must be labeled as failure cases rather than silently learned.

### Hardware and development environment

- Development-first on Linux + Python 3.11 or another compatible version supported by installed wheels. Verified repository environment currently uses Python 3.10.12.
- CPU-only MVP; GPU is **not** required. Keep 4 GB VRAM laptop GPUs out of the critical path.
- Available hardware is an RTX 3050 Laptop GPU with 4 GB VRAM and 32 GB RAM. Treat the GPU as optional demo hardware, not a required runtime or training target.
- Use an isolated virtual environment.
- Install **one** OpenCV wheel flavor to avoid `cv2` namespace conflicts.
- Preferred package: `opencv-python-headless==5.0.0.93` (verify wheel availability on the working Python/platform in Phase 0); import as `cv2` and assert `cv2.__version__.split('.')[0] == '5'`.
- Other minimal packages: `numpy`, `streamlit`, `boto3`, `pytest`. Pin actual successfully installed versions after environment validation; do not guess pins.

### Input contract

- MVP input: `.mp4` or `.mov` if the environment's OpenCV decoder supports it.
- Recommended videos: 5–15 seconds, 25–60 fps, moderate 640p–1080p resolution.
- Read FPS from `cv2.VideoCapture`; reject invalid FPS or require a **clearly labeled** FPS override, never assume one silently.
- Use `frame_index / fps` as measured video time in seconds for constant-frame-rate recordings. Document that variable-frame-rate videos may need transcoding or true timestamps.
- Prefer filmed samples with an orange ball against a dark/plain background. Do not advertise generalized recognition.

### Vision technique (OpenCV 5)

- `cv2.VideoCapture` to decode frames.
- `cv2.cvtColor(..., cv2.COLOR_BGR2HSV)` to convert to HSV.
- `cv2.inRange` for a color mask, followed by a small morphological open/close (`cv2.morphologyEx`).
- `cv2.findContours`, `cv2.contourArea`, `cv2.minEnclosingCircle` or moments to select and locate the ball.
- Reject implausibly small/large contours and out-of-frame or sudden large jumps.
- `cv2.circle`, `cv2.polylines`, `cv2.putText` for trajectory annotations.
- Allow configurable HSV bounds only as necessary after first reliable sample; prefer a known working preset for MVP.
- Lost tracking should be flagged, not silently replaced with fabricated coordinates.

### Prediction models

Use three **minimal and interpretable** causal models, based exclusively on observations and adaptive feedback available at/before the cutoff:

1. **Constant velocity baseline:** `p(t + h) = p(t) + v * h`.
2. **Non-adaptive constant deceleration baseline:** at each cutoff, independently estimate a nonnegative speed-deceleration magnitude `d_fit` from the recent measured speed history. Predict travel distance `s(h) = v*h - 0.5*d*h^2` until stopping, with stopping time `v/d` and correct stopped-distance behavior. Apply distance along the estimated current motion direction. `d=0` reduces to constant velocity.
3. **Adaptive constant deceleration:** initialize from the same estimate as model 2 at the first eligible cutoff, then carry state and apply only the chronological residual updates defined above. At each later cutoff it uses `d_current`, never a value learned from that cutoff’s future target.

Implementation notes:
- Velocity must have units `px/s`; deceleration `px/s²` without physical calibration.
- Estimate velocity robustly from several **recent valid** points; document method (e.g., slope fit on `x(t)` and `y(t)`).
- Estimate deceleration from observed change in speed, clamp negative deceleration estimates to zero, and guard near-zero speeds/noisy tracks.
- For a stopping event, travel distance at times beyond stopping is `v²/(2d)`, **not zero**.
- Preserve both raw tracked positions and any filtered/fit values; do not overwrite measurements with model output.
- Adaptive state must include `d_initial_px_s2`, `d_current_px_s2`, `alpha`, `update_count`, the timestamped update trace and skipped-update warnings.
- Replaying the same ordered observations must produce the same adaptive state. Reject duplicate/non-monotonic times rather than letting update order vary.
- No prediction through wall collisions or turns in the MVP. Show degraded accuracy openly in such scenes.
- Include a minimum-observation threshold and explicit insufficient-data result.
- Horizons: default `0.25 s`, `0.5 s`, and `1.0 s`, if future video duration permits.
- Do not claim that the physics-informed or adaptive model *always* outperforms a baseline. “Improves over time” means the declared aggregate metric improved on the declared sequence, not that every cutoff won.

### Cloud architecture (meaningful workload)

```text
Uploaded video
     |
     v
Local Streamlit UI + OpenCV 5 tracker
     | timestamped observations through cutoff (JSON)
     v
AWS Lambda — causal replay + parameter update + forecasts [REAL AWS COMPUTE]
     | three-model predictions, adaptive trace, execution metadata
     v
Local UI — overlay future predictions against later video frames
     | errors + learned-parameter chart + CSV/JSON
     v
Judge demo + technical report
```

| Component | Responsibility | Trust/leakage boundary |
|---|---|---|
| Local OpenCV 5 tracker | Decode real frames; emit raw valid/invalid pixel observations and diagnostic masks/overlays. | It measures the whole uploaded video, but each prediction request slices observations to `time <= cutoff`. |
| Shared pure-Python physics core | Validate ordered observations; estimate velocity/deceleration; replay eligible adaptive feedback; produce three forecasts and trace. | The same deterministic module runs locally for tests and inside Lambda. |
| AWS Lambda | Execute the real causal replay, adaptive update history and requested forecasts; return engine/request metadata. | Reject observations later than declared cutoff and invalid request sizes. No silent local substitution. |
| Local evaluator/UI | After a forecast is stored, reveal the corresponding later tracked point, score all models, advance to the next cutoff and visualize the audit trail. | Evaluation targets never enter an earlier Lambda request; exports retain cutoff/target timestamps. |

For an uploaded video, adaptation is demonstrated with several ordered cutoffs. At cutoff `i`, Lambda sees only that cutoff’s past. When the UI advances to cutoff `i+1`, outcomes that are now historical may be included and used to reconstruct the next adaptive state. This produces a visible improvement curve without persistent databases or server-side sessions.

- The Lambda's business logic must compute the actual forward predictions and adaptive parameter history from the observations available through the requested cutoff. It must not be a ping endpoint or upload-only storage.
- Avoid API Gateway for MVP: use an authenticated `boto3.client('lambda').invoke(InvocationType='RequestResponse', ...)` from the local application when credentials are configured.
- Package the same pure-Python physics module for local tests and Lambda to avoid mathematical drift; use Python standard-library math where feasible to avoid Lambda wheel/layer packaging for compiled libraries.
- UI explicitly reports `engine=aws_lambda` or `engine=local_fallback`, plus AWS request ID where available.
- **Cloud completion requires a live Lambda invocation with evidence**, not only mocks.
- AWS region, Lambda name and credentials come from environment variables or the standard AWS credentials chain; never hard-code a private value.
- Optionally archive a sample analysis JSON to S3 **only if it adds demonstrable value** and is authorized. Lambda prediction is the core AWS proof.
- Prefer reconstructing adaptive state deterministically from ordered observations over accepting an opaque client-supplied learned state. This makes local/Lambda parity and leakage audits reproducible.

---

## 4. Minimal repository structure

Create files **when their phase starts**, not as empty placeholders:

```text
physicseye/
├── MASTER_PLAN.md                # This plan
├── PROGRESS.md                   # Phase log / evidence
├── README.md                     # How to install, run, reproduce
├── requirements.txt              # Exact, tested runtime pins
├── requirements-dev.txt          # Exact, tested pytest pin
├── .gitignore
├── .env.example                  # Variable NAMES only, no secrets
├── app.py                        # Streamlit UI (Phase 4)
├── src/physicseye/
│   ├── __init__.py
│   ├── tracker.py                # OpenCV tracking (Phase 1)
│   ├── physics.py                # Shared pure prediction math (Phase 2)
│   ├── evaluate.py               # Time-based evaluations (Phase 3)
│   └── render.py                 # Visual overlays (Phase 4)
├── scripts/
│   ├── generate_sample.py        # Synthetic input with ground truth (Phase 0)
│   ├── analyze_video.py          # Thin CLI for repeatability (Phase 1/3)
│   └── deploy_lambda.sh          # Minimal AWS deployment helper (Phase 5)
├── cloud/
│   └── lambda_handler.py         # Invokes shared physics module (Phase 5)
├── tests/
│   ├── test_tracker.py
│   ├── test_physics.py
│   ├── test_evaluate.py
│   └── test_lambda_handler.py
├── examples/
│   └── README.md                 # Provenance for sample video(s)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── TECHNICAL_REPORT.md
│   └── SUBMISSION_CHECKLIST.md
└── outputs/                      # Local generated outputs; gitignored
```

Avoid adding FastAPI, Redis, databases, Kubernetes, Docker, custom tracking frameworks, or PyTorch unless a later requirement demonstrates a real need.

### Core data shape (stable after Phase 1)

Each observation is structured like:

```json
{"frame": 12, "time_s": 0.4, "x_px": 186.5, "y_px": 219.0, "area_px2": 390.0, "valid": true}
```

- `frame` is zero-based, `time_s` increases strictly, `x_px` and `y_px` are object-center pixels.
- Invalid frames contain `valid: false` and null coordinates; no fake numeric values.
- Store detection quality/threshold diagnostics separately if useful; do not call uncalibrated heuristics an accurate probability.
- Preserve the same JSON semantics between CLI, UI, tests and Lambda.

---

## 5. Ordered implementation phases

### PHASE 0 — Environment + reproducible sample

**Goal:** Establish a verified OpenCV 5 development environment and a synthetic video with known ground-truth motion.

**Tasks**
- [ ] Inspect repository; create minimum folders, `.gitignore`, `.venv`, dependency files, `PROGRESS.md`.
- [ ] Create and activate Python venv; install one verified OpenCV 5 wheel and other actual needed dependencies; pin successful versions.
- [ ] Record Python, OpenCV, OS, FPS/video-codec observations; fail fast if `cv2.__version__` does not start with `5.`.
- [ ] Generate two short synthetic videos with a single orange ball on dark background: one constant speed, one slowing down. Save exact position-vs-time ground truth beside the video.
- [ ] Avoid assuming `VideoWriter` codecs work: write, reopen with `VideoCapture`, verify frame count, and choose a supported format if needed.
- [ ] Write README quick-start and sample provenance note (`synthetic; created by project`).

**Acceptance criteria**
- Clean environment can import OpenCV 5 and generate/reopen the synthetic videos.
- Synthetic trajectory ground truth is valid and reproducible.
- `pytest` command starts and succeeds for any tests introduced.
- `PROGRESS.md` reports actual versions and execution evidence.

**Stop:** do not implement tracking in Phase 0.

### PHASE 1 — OpenCV 5 ball tracking

**Goal:** Extract reliable 2D ball centers from input video.

**Tasks**
- [ ] Implement minimal HSV-based detector and sequential tracker using OpenCV 5.
- [ ] Store valid/invalid frame observations in a structured list and CSV/JSON output.
- [ ] Add command-line path `python scripts/analyze_video.py --video <path>` (no prediction yet).
- [ ] Handle empty video, failed decoder, unknown FPS, missing ball, noisy contours and large false jumps without crashing.
- [ ] Compare detected synthetic positions to generator ground truth.
- [ ] Document lighting/background constraints and key thresholds.

**Acceptance criteria**
- Extracted trajectory visually follows the synthetic ball.
- On clean synthetic sequences, center-position MAE **target <= 5 px** (adjustable only with documented evidence and reason).
- Missing frames are labeled invalid, never silently filled.
- CLI returns a clear error for unreadable media and writes a valid track for supported input.
- `pytest tests/test_tracker.py` passes.

**Stop:** do not implement a predictor or UI in Phase 1.

### PHASE 2 — Physical prediction + adaptive learning core

**Goal:** Predict future 2D positions with causal baselines and an interpretable online deceleration update.

**Tasks**
- [ ] Implement `src/physicseye/physics.py` with pure functions for motion-parameter estimation and forecast.
- [ ] Include constant velocity, independently refitted deceleration and adaptive deceleration/stopping-distance models; no heavy fitting libraries.
- [ ] Accept observations only up to a specified cutoff timestamp and explicit requested horizons.
- [ ] Implement the residual-derived deceleration candidate and fixed-alpha update from the adaptive contract; update only after the feedback target is in the past relative to the next cutoff.
- [ ] Return model name, x/y forecasts, fitting-window info, motion parameters with pixel-based units, adaptive update trace, and warning/status fields.
- [ ] Deal with insufficient observations, duplicate/non-monotonic time, stationary ball, tracking gaps, and impossible inputs.
- [ ] Add deterministic tests using known synthetic paths, including travel distance after stop, exact residual update math, skipped ambiguous updates and deterministic replay.

**Acceptance criteria**
- Predictions agree with known analytic trajectories to numeric tolerance on noiseless synthetic data.
- No access to future frames inside predictor.
- All three models produce finite values for valid inputs and explicit error status for invalid ones.
- With `alpha=1` on a valid noiseless non-stopping feedback interval, the update recovers its analytic deceleration candidate to numeric tolerance.
- A test proves that an outcome at time `t` cannot affect any prediction issued before `t`.
- `pytest tests/test_physics.py` passes.

**Stop:** no AWS or UI yet.

### PHASE 3 — Fair evaluation + benchmark

**Goal:** Quantify forecasting error without data leakage.

**Tasks**
- [ ] Implement evaluation with one or more clearly chosen time cutoffs and horizons (0.25, 0.5, 1.0 s).
- [ ] At each cutoff, fit using only observations `time <= cutoff`; retrieve actual positions strictly **after** cutoff for error calculation.
- [ ] Use walk-forward evaluation: reveal a target, score it, make any eligible adaptive update, then advance to the next cutoff. Never batch-fit the adaptive model on the whole video.
- [ ] Compare all three forecasts to the same actual tracked position at the corresponding target frame/time; skip unavailable horizons and report sample counts.
- [ ] Compute per-model mean/median Euclidean position error (px), per-horizon error, adaptive gain versus non-adaptive physics, cutoff win rate, update/skip counts, parameter stability, trajectory tracking accuracy on synthetic ground truth, and inference/processing latency (ms). Also report a frozen-initial-`d` ablation for the changing-dynamics scenario.
- [ ] Evaluate constant-speed, constant-deceleration and changing-deceleration synthetic scenarios and, when available, two independently recorded real rolling-ball videos.
- [ ] Freeze `alpha`, fitting-window length and thresholds before the final real-video evaluation; record configuration and do not tune on the reported test videos.
- [ ] Export `outputs/evaluation.csv` and `docs/EVALUATION.md` including setup, numerical results, thresholds, and failure examples.
- [ ] Write tests verifying no temporal leakage and correct metric calculations.

**Acceptance criteria**
- All three models are measured at the same cutoff/horizon/valid ground truth where comparable.
- At least one reproducible benchmark table with **actual computed** values, even if the physics model loses.
- On the held-out noiseless changing-deceleration synthetic scenario, adaptive post-change mean error is at least 10% lower than the frozen-initial-`d` ablation; otherwise Phase 3 is **BLOCKED** pending diagnosis, not threshold relaxation. Adaptive-versus-refit results are reported without a guaranteed win.
- Real-video results report adaptive gain and win rate even when they are zero or negative; no universal-improvement claim is allowed.
- Missing/failed predictions are counted and not quietly excluded.
- `pytest tests/test_evaluate.py` passes; full test suite still passes.

**Stop:** do not optimize metrics with unrecorded manual test-set tuning.

### PHASE 4 — Judge-friendly interactive demo

**Goal:** Turn the pipeline into a compact, working Streamlit demo.

**Tasks**
- [ ] Build `app.py`: video upload / sample selection, analyze button, cutoff selection, prediction horizon selector, clear status/errors.
- [ ] Visualize observed vs predicted trajectories: colored points/lines and target frame annotated images. Label actual and predicted distinctly.
- [ ] Show side-by-side constant-velocity vs friction-informed errors and parameters with units.
- [ ] Distinguish non-adaptive and adaptive physics, and show `d_fit`, `d_initial`, `d_current`, update count, skipped updates, and an error/parameter-over-time chart.
- [ ] Provide JSON/CSV download of analysis. Add a simple chart and a few diagnostic messages.
- [ ] Reuse the existing tracker/predictor/evaluator; keep the interface thin.
- [ ] Test with synthetic and at least one real sample; verify there is no crash for missing ball or invalid video.

**Acceptance criteria**
- `streamlit run app.py` can complete the full local flow on two sample videos.
- The prediction is visibly anchored at the selected cutoff, not retroactively drawn as if known earlier.
- The adaptive chart makes each predict→observe→update event chronological; the selected cutoff cannot display feedback from its own future.
- Outputs are actually calculated, not screenshots or fabricated sample figures.
- UI states it is running **LOCAL MODE** until AWS has been integrated.

**Stop:** do not add webcam real-time processing yet.

### PHASE 5 — AWS Lambda integration (**required**)

**Goal:** Make AWS perform the project’s actual physics prediction workload.

**Tasks**
- [ ] Implement `cloud/lambda_handler.py` calling the shared `physics.py` module with strict JSON validation and structured JSON result.
- [ ] Define request contract `{observations, cutoff_time_s, horizons_s}` with request limits and response contract `{predictions, parameters, adaptive_trace, warnings, engine}`. Lambda causally reconstructs adaptive state from the supplied ordered observations.
- [ ] Unit-test the Lambda handler locally; package Python sources so Lambda can import them without local editable installs.
- [ ] Deploy a minimal Python AWS Lambda function using AWS CLI / zip. No API Gateway required.
- [ ] Allow UI to invoke the Lambda with `boto3.client('lambda').invoke(InvocationType='RequestResponse', FunctionName=...)`; decode payload; check function errors and return metadata.
- [ ] Use environment variables for AWS region and Lambda name. Prefer standard AWS profile/role chain; no secrets in repo.
- [ ] Keep an explicitly labeled local fallback for offline demos. Never claim AWS was used if fallback occurred.
- [ ] Capture proof of a real AWS request: timestamp, region (nonsecret), Lambda function identifier (redacted if needed), request ID, adaptive update count, and measured result. Include deployment and cost cleanup instructions.

**Acceptance criteria**
- Live upload flow uses OpenCV 5 locally to extract observations and **AWS Lambda actually computes the predictions**.
- Cloud and local forecasts **and adaptive traces** agree within an appropriate deterministic tolerance for identical inputs.
- Errors/timeouts are visible and do not silently swap to fallback.
- Cloud run evidence is available for the demo/report.
- `pytest tests/test_lambda_handler.py` and full suite pass.

**AWS access blocker:** If credentials/permission do not exist, do not pretend Phase 5 passed. Complete packaging/tests and tell the user exactly which AWS authorization/deployment action is needed.

### PHASE 6 — Quality, reproducibility and measured impact

**Goal:** Demonstrate correctness, robustness, limitations and real-world usefulness.

**Tasks**
- [ ] Film/obtain at least two real-world videos under permitted use and document their capture conditions and provenance.
- [ ] Run the same end-to-end evaluation locally and with Lambda for selected samples.
- [ ] Include 1+ honest failure case (occlusion, motion blur, changing light, ball stopping, or camera shake).
- [ ] Add an ablation table for constant velocity, non-adaptive refit, frozen-initial-`d` and adaptive physics. If optional Kalman smoothing is tested, report raw-versus-filtered tracking and forecast metrics separately.
- [ ] Validate adaptive repeatability, strict chronological ordering, parameter bounds, missing-feedback behavior and reset behavior across videos.
- [ ] Document runtime/latency numbers with hardware description, FPS, frames analyzed and trial count.
- [ ] Add useful input-size / duration limits; handle unexpected media safely; no untrusted command execution or public AWS exposure.
- [ ] Test a fresh environment using only README instructions and dependency pins; fix blockers.
- [ ] Produce a simple mermaid architecture diagram in `docs/ARCHITECTURE.md`.

**Acceptance criteria**
- Reproducible install/test/analyze instructions validated, not just written.
- Evaluation includes baselines, raw metrics, sample sizes, uncertainty/limitations, and AWS evidence.
- Evaluation includes the full adaptive parameter trace and enough ordered input/output data to audit that no future observation affected an earlier forecast.
- No hard-coded personal machine paths, credentials, or falsified measurement claims.
- UI + CLI remain functional.

### PHASE 7 — Final submission and release

**Goal:** Package a judge-accessible competition entry, not just source files.

**Tasks**
- [ ] Finish `README.md` with problem, demo, setup, reproducibility and cloud deployment summary.
- [ ] Finish `docs/TECHNICAL_REPORT.md`: target users, method, research references/licenses, OpenCV 5, adaptive update rule, physics assumptions, AWS, benchmarking, results, failures, security, limitations, and future work.
- [ ] Finish `docs/SUBMISSION_CHECKLIST.md` with links/placeholders for repo, accessible demo, and video; never invent URLs.
- [ ] Prepare a **maximum 5-minute demo video script/storyboard**: setup/problem (25 s), actual OpenCV tracking (50 s), chronological prediction/error/update and three-model comparison (105 s), real Lambda request and architecture (55 s), quantified results + failures (60 s). Adjust to <=300 seconds.
- [ ] Ensure every screenshot, visualization and measurement shown in the video is from the running system.
- [ ] Verify repository access, license rights for samples, no secrets, dependency pins, architecture graphic, report completeness, and video availability.
- [ ] Run complete test suite and capture the final output; produce the final release notes.

**Acceptance criteria**
- Every organizer-required artifact is present or explicitly noted as an external submission step not yet performed.
- A judge can reproduce one sample run from documented commands.
- Video length <=5:00, judge-accessible and demonstrates actual AWS execution.
- No claim that a Devpost submission is complete unless it was actually submitted and verified.

---

## 6. Optional extension — ONLY after Phase 7 is ready

**Not required; do not build unless requested.**

- **Adaptive visual verification:** when prediction error from a prior interval crosses a declared threshold, issue a new OpenCV re-tracking request using a tighter ROI, perform it, and log the actual multi-step evidence loop. This is closer to a legitimate Agentic Vision workflow than an LLM describing a video, but still requires proof of decision→action. Do not market it as contest-eligible without verification.
- **Collision prediction:** balls bouncing off a clearly visible wall; requires separate event detection and collision model/evaluation.
- **Spatial calibration:** ruler or checkerboard for converting pixels to centimeters; document camera geometry and error.
- **OpenCV Kalman smoothing:** `[x, y, vx, vy]` state with timestamp-derived `dt`; preserve raw measurements and add only if a fixed evaluation shows lower held-out tracking/forecast error.
- **What-if physics simulation:** a deceleration slider or alternative initial velocity, labeled hypothetical and excluded from measured benchmark results.
- **Live webcam mode:** streaming tracking and prediction with latency budgeting.
- **Multiple objects:** track identity and interactions.

Avoid these additions if they threaten tests, deployment, or competition submission.

---

## 7. Timeline (Bangkok time, suggested)

| Day | Priorities | Deliverable |
|---|---|---|
| Oct 21 | Phase 0 → start Phase 1 | Verified OpenCV 5 + synthetic video |
| Oct 22 | Phase 1 → Phase 2 | Tracking + three prediction modes + adaptive update tests |
| Oct 23 | Phase 3 | Leakage-safe adaptive benchmark + test suite |
| Oct 24 | Phase 4 | End-to-end local UI |
| Oct 25 | Phase 5 | AWS Lambda working + evidence |
| Oct 26 | Phases 6–7 / submit | Real videos, docs, final demo |

This is an **aggressive target**, not a guarantee. If behind schedule, prioritize, in this order: **OpenCV 5 live result → constant/non-adaptive/adaptive forecasts → leakage-safe metrics → real AWS call → judge-accessible submission → polish**. Do not add Kalman, calibration, collision or what-if controls until every mandatory gate below passes.

### MVP priority and cut line

1. **Must ship:** existing OpenCV 5 tracker, causal constant-velocity/non-adaptive/adaptive prediction, residual-based parameter updates, fair benchmark, compact Streamlit visualization, real Lambda computation, reproducible docs/demo.
2. **Ship if evidence demands it:** MOG2 masking or OpenCV Kalman smoothing for a documented tracking failure.
3. **Cut first:** calibration, what-if UI, bounce/collision, live webcam, YOLO and every deep/3D model.

### Ship/no-ship gate

A project is **not competition-ready** unless all are true:

- [ ] OpenCV 5 runtime verified in the application.
- [ ] Real frames processed into actual tracked observations.
- [ ] Prediction is made using only past observations.
- [ ] Constant velocity, non-adaptive deceleration and adaptive deceleration are compared at identical cutoffs with measured error; the changing-dynamics benchmark also includes a frozen-initial-parameter ablation.
- [ ] Adaptive updates are chronological, auditable, and beat frozen physics by >=10% on the held-out changing-deceleration synthetic acceptance case.
- [ ] A real AWS workload runs and is demonstrated.
- [ ] UI or live screen-share path works reliably.
- [ ] Repo, pins, technical report, architecture, and <=5-minute video are judge-accessible.
- [ ] Known shortcomings and at least one failure case are reported honestly.

---

## 8. Ready-to-paste Codex instructions

### First Codex prompt — copy this

```text
Read MASTER_PLAN.md completely and follow Section 0 (Codex operating contract).
Implement PHASE 0 ONLY. Do not implement later phases.
Inspect the repository and verify dependencies/APIs before writing code.
Run the acceptance tests and smoke test. Update PROGRESS.md.
Report exact files changed, commands/results, blockers and next phase, then STOP.
```

### Next phases — reuse one prompt at a time

For Phase 1 through Phase 7, send (substitute the phase number only):

```text
Read MASTER_PLAN.md and PROGRESS.md. Implement PHASE <N> ONLY.
Preserve all passed prior-phase acceptance criteria. Do not build later phases.
Run new phase tests plus the complete existing test suite.
Update PROGRESS.md with actual evidence, problems and next action.
Summarize changes and test results, then STOP.
```

### When Codex gets stuck

```text
Read MASTER_PLAN.md and PROGRESS.md. Diagnose the current phase's blocker.
Make the smallest fix necessary for that phase only.
Do not skip acceptance criteria or invent test results.
Rerun the failing test and complete suite, update PROGRESS.md, then STOP.
```

---

## 9. Reference and verification notes

- Organizer's overview explicitly lists **physics-informed video prediction for process control** as a suggested project area, not a promise of judging preference or award.
- OpenCV version and Python wheel compatibility must be verified *in the actual development environment*. The PyPI version `5.0.0.93` is a known published OpenCV 5 wheel release, but a given platform may not have an installable wheel.
- OpenCV 5 documents the Python [`cv.KalmanFilter`](https://docs.opencv.org/5.0/main_modules/classcv_1_1KalmanFilter.html), two-return [`cv.findContours`](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html) and [`cv.VideoCapture`](https://docs.opencv.org/5.0/main_modules/classcv_1_1VideoCapture.html) APIs. The current Phase 0 environment also exposes these APIs; this does not prove an external repository runs unchanged.
- Lambda deployment needs a real AWS account/permissions; do not assert connectivity until tested.
- Keep Lambda prediction/adaptation pure Python when possible. AWS documents that native packages such as NumPy require Linux-compatible packaging or a layer: https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html
- Devpost header vs body deadline differ; use the earlier time and verify within Devpost before submission.

Research repositories inspected for this revision:

- https://github.com/trflorian/ball-tracking-live-plot — MIT; usable reference for classical masking/trajectory visualization.
- https://github.com/MehakArora/Ball_Tracking — no license found; reference only.
- https://github.com/NattyBumppo/Ball-Tracking — no license found; reference only and based on OpenCV 2/Python 2.
- https://github.com/turman17/tennis-ball-detection-and-tracking — no license found; reference only; YOLO assets/dependencies were not adopted.
- https://github.com/vincent-leguen/PhyDNet — MIT; research reference only; deep frame prediction was rejected for the MVP.
- https://github.com/ccyydd/PhysMind — no root license found; reference only; bounded system-identification principles were simplified rather than copied.

**Plan success means a reproducible, honest working entry — not a claim of winning.**

# PhysicsEye demo video script

Target duration: **4:50**. Hard limit: **5:00**. The official submission requires a public or unlisted judge-accessible video that shows the team, working application, architecture, and principal results. This document is a recording plan; no demo video has been recorded or verified yet.

## Recording preflight

Use a 1920x1080 screen recording at readable browser zoom. Disable notifications and hide the AWS account number, profile files, terminal history, and unrelated windows. Do not show credentials. Record narration live or add it afterward without changing what the application actually displays.

Generate local artifacts before recording:

```bash
cd /path/to/OpenCV-AWS-hack2026
source .venv/bin/activate
env -u PYTHONPATH .venv/bin/python scripts/generate_sample.py
env -u PYTHONPATH .venv/bin/python scripts/run_benchmark.py
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

Verify the private cloud path:

```bash
export AWS_PROFILE=physicseye-cli
export AWS_REGION=ap-southeast-1
export PHYSICSEYE_LAMBDA_NAME=physicseye-predict
aws sts get-caller-identity
aws lambda get-function-configuration \
  --function-name "$PHYSICSEYE_LAMBDA_NAME" \
  --query '{State:State,Runtime:Runtime,Memory:MemorySize,Timeout:Timeout}'
```

Start the dashboard:

```bash
env -u PYTHONPATH .venv/bin/streamlit run app.py
```

Open the dashboard, terminal, [`ARCHITECTURE.md`](ARCHITECTURE.md), and [`EVALUATION.md`](EVALUATION.md) before recording. Download the permitted basketball clip using the official command in [`REAL_VIDEO_VALIDATION.md`](REAL_VIDEO_VALIDATION.md); do not commit or redistribute the source file separately.

## Shot list and narration

### 0:00–0:25 — Team, problem, and result

**Screen:** Camera view with the entrant/team and the PhysicsEye title, then transition to the dashboard Overview tab.

**Narration:**

> We built PhysicsEye for the OpenCV AI Competition 2026. Tracking where an object is now is useful, but many experiments need to know where it will go and whether the model is improving. PhysicsEye turns ball video into causal forecasts, measures later errors, and updates one interpretable motion parameter.

Do not state that the project has won, is calibrated in meters, or always improves.

### 0:25–1:15 — OpenCV 5 real-video tracking

**Dashboard actions:**

1. Open **Video Analysis**.
2. Choose **Upload** and select `basketball_6077402.mp4`.
3. Choose **Select target ROI**.
4. Set initialization frame `80`.
5. Set horizontal range approximately `1175–1605 px` and vertical range `120–550 px`.
6. Show the preview box tightly around the ball, then click **Analyze**.
7. Show valid count, mean confidence, cutoff position, target box, and the later invalid warning after the ball leaves the frame.

**Narration:**

> OpenCV 5 decodes every frame locally. The original orange HSV detector worked on generated fixtures but followed the basketball court and failed on white or mirrored balls. ROI mode learns appearance from one user box, searches locally at nearby scales, and checks confidence, motion, and size. When confidence is insufficient or the ball leaves the frame, it returns an invalid observation—never a fabricated position.

Overlay a small text card: `HSV baseline: 0/32 checked positions on-ball` and `ROI: 31/31 visible references on-ball; 0/3 absent-frame false positives`. Keep “sparse manual references, ±8–25 px uncertainty” visible.

### 1:15–2:30 — Causal forecasts and four-model comparison

**Dashboard actions:**

1. Switch to the generated **Slowing down** sample and **Orange HSV** mode.
2. Keep **LOCAL** execution.
3. Click **Analyze**, choose cutoff `2.00 s`, and horizons `0.25`, `0.5`, `1.0 s`.
4. Pause on the cutoff/target overlay and point out observed history, future forecasts, and evaluation-only later points.
5. Open **Model Comparison** and show constant velocity, refitted, frozen, and adaptive rows.

**Narration:**

> Every model receives the identical observation prefix through this cutoff. Constant velocity is the simplest baseline. Refitted deceleration estimates a new value from recent history. Frozen deceleration preserves its initial value. Adaptive deceleration changes persistent state only from an earlier forecast whose outcome is already observable. White rings are later tracker measurements used for scoring; they were not available to the predictor.

### 2:30–3:15 — Adaptive update evidence

**Dashboard actions:**

1. Open **Adaptive Learning**.
2. Show `d_fit`, `d_initial`, `d_current`, update count, parameter trace, residual trace, and any skipped-update table.
3. Move the cutoff backward briefly to demonstrate that later updates disappear, then return to `2.00 s`.

**Narration:**

> The update uses the signed displacement residual at a fixed 0.25-second feedback horizon, clamps the candidate to physical bounds, and blends it with alpha 0.25. Each trace row records when the prediction was issued and when its target became available. Tests prove that changing future observations cannot change an earlier forecast.

### 3:15–4:00 — Real AWS Lambda invocation

**Dashboard actions:**

1. Return to **Video Analysis** with the slowing sample analyzed.
2. Select **AWS**.
3. Keep cutoff `2.00 s` and all three horizons.
4. Click **Invoke AWS Lambda** once.
5. Wait for and clearly show function name, region, AWS request ID, invocation/Lambda duration, **Parity PASS**, and maximum numeric difference.
6. Briefly show the architecture diagram.

**Narration:**

> Video and OpenCV remain local. The dashboard sends only causal numeric observations to a private Lambda. Lambda reconstructs the adaptive history and calculates all four forecasts using the same dependency-free physics source as local mode. This is a real synchronous invocation; an AWS error is shown rather than silently replaced with local output.

Read the values currently displayed. Do not reuse an old request ID as though it came from this recording session.

### 4:00–4:35 — Benchmark, failure cases, and limitations

**Dashboard actions:**

1. Open **Benchmark Results**.
2. Show the aggregate table and the abrupt-change scenario.
3. Briefly display `docs/EVALUATION.md` or an edited card containing verified numbers.

**Narration:**

> On seven controlled scenarios, refitted deceleration has the lowest aggregate MAE at 1.306 pixels; adaptive is 1.358, so we do not claim it is always best. In the predeclared changing-dynamics acceptance window, adaptive reduces error 83.106 percent versus frozen. It also performs worse in noisy and generated-video cases. Real clips expose perspective, hand impacts, reflections, and annotation uncertainty, so their pixel-space gains are not calibrated physics claims.

### 4:35–4:50 — Final summary

**Screen:** Return to the architecture or dashboard overview with repository/report links.

**Narration:**

> PhysicsEye combines substantive OpenCV 5 tracking, interpretable causal learning, fair quantitative evaluation, and real AWS computation in a reproducible CPU-friendly MVP. The repository includes pinned dependencies, tests, raw benchmark exports, deployment instructions, and documented failures.

End without a fabricated Devpost/demo URL or award claim.

## Editing and verification checklist

- [ ] Final runtime is <=4:50 target and never exceeds 5:00.
- [ ] Team/entrant appears on camera.
- [ ] Real OpenCV tracking, real calculated predictions, architecture, and principal results are visible.
- [ ] The AWS segment shows a request ID generated during recording and `engine=aws_lambda`/parity evidence.
- [ ] Text is readable at normal playback size and narration is audible.
- [ ] Account IDs, credentials, tokens, home-directory details, and unrelated terminal history are hidden.
- [ ] Every number matches the dashboard or committed reports.
- [ ] Synthetic versus real-video evidence is labeled.
- [ ] Negative results and pixel-space limitations remain in the final edit.
- [ ] Pexels clips are credited in the description and used consistently with their license.
- [ ] The full exported file is watched once after rendering.
- [ ] The uploaded public/unlisted URL works in a signed-out browser.

Record the checked duration, upload URL, recording date, and AWS request ID in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) only after they are verified.

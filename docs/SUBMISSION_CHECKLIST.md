# PhysicsEye submission checklist

Verified against the official [OpenCV AI Competition 2026 overview](https://opencv26.devpost.com/) and [rules](https://opencv26.devpost.com/rules) on 2026-10-23.

## Deadline and eligibility

The Devpost header and rules page show **October 26, 2026 at 11:45 p.m. PDT**. The overview body says 11:59 p.m. Pacific. Use the earlier time as the submission deadline: **October 27, 2026 at 1:45 p.m. Asia/Bangkok (ICT)**, and submit earlier to allow for upload/review time.

- [ ] Confirm the countdown shown inside the signed-in Devpost submission form.
- [ ] Confirm every team member satisfies the current eligibility rules and the team representative is identified.
- [ ] Confirm the project is joined to the competition and the final project form is editable.
- [ ] Submit before the conservative 11:45 p.m. PDT deadline.

These items require the entrant's Devpost account and have not been completed by this repository update.

## Official final-submission requirements

| Official requirement | Prepared evidence | Status |
|---|---|---|
| Technical report covering problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, and responsible use | [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) | Ready |
| Judge-accessible repository or archive | `https://github.com/panuza1/OpenCV-AWS-hack2026` | Ready: unauthenticated HTTP request returned 200 on 2026-10-23; visually recheck before submission |
| Pinned dependencies and build/deploy/test instructions | `requirements*.txt`, [`README.md`](../README.md), [`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md) | Ready |
| Architecture diagram showing OpenCV 5 and AWS | [`ARCHITECTURE.md`](ARCHITECTURE.md) and technical report | Ready |
| Working web endpoint or arranged live screen-share | Local Streamlit dashboard and private Lambda are verified | Manual: arrange live screen-share; no public endpoint exists |
| Public or unlisted judge-accessible video <=5 minutes showing the team, working app, architecture, and principal results | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | **Blocked: record, inspect, upload, and add URL** |
| Evaluation evidence including failure cases or limitations | [`EVALUATION.md`](EVALUATION.md), [`REAL_VIDEO_VALIDATION.md`](REAL_VIDEO_VALIDATION.md) | Ready |

PhysicsEye targets the Overall Awards. Do not select **Best Use of COOL** or **Agentic Vision** unless the implementation materially changes and the extra official evidence is produced; the current project uses neither COOL on Graviton nor an agentic perception-decision-action loop.

## Devpost copy deck

### Short summary

PhysicsEye uses OpenCV 5 to track a ball, compares four causal trajectory models, and updates an interpretable image-space deceleration estimate from earlier forecast errors. The same lightweight prediction engine runs locally and in a private AWS Lambda with measured parity.

### Final project description

PhysicsEye turns ordinary fixed-camera ball videos into tracked trajectories, short-horizon forecasts, and an auditable predict-observe-update loop. OpenCV 5 performs either deterministic orange-HSV detection or user-initialized ROI template tracking with confidence, motion, and size rejection. Four CPU-friendly models—constant velocity, refitted deceleration, frozen deceleration, and adaptive deceleration—receive identical causal observation histories. Later observations are revealed only after prediction, so forecast error can update the adaptive parameter only at a later cutoff.

The Streamlit dashboard shows target selection, tracking status, cutoff-anchored overlays, per-model errors, deceleration updates, skipped updates, benchmark results, and downloadable artifacts. A private AWS Lambda reconstructs the same causal state and returns all four forecasts; a verified real-video invocation matched local output within `5.684e-14`.

The controlled abrupt-change benchmark reduced adaptive error by 83.106% versus the frozen ablation, while aggregate and real-video results also preserve cases where adaptive prediction ties or loses. Three independently sourced real videos validate ROI target identity, but perspective, hand impulses, reflections, sparse annotations, and pixel units prevent claims of calibrated physical accuracy.

### Technology stack

- Python 3.10 locally; Python 3.13 AWS Lambda runtime
- OpenCV 5.0.0 and NumPy for video decoding, HSV/contour detection, ROI template matching, and rendering
- Pure Python standard-library physics/adaptation core
- Streamlit dashboard
- AWS Lambda invoked directly with boto3 and standard AWS credentials
- Pytest deterministic unit and integration tests

### Key innovations

- A causal predict → observe → measure error → update loop with timestamped evidence.
- Four fair models using identical observation prefixes, including frozen and refitted ablations.
- One bounded, interpretable effective-deceleration parameter rather than a large learned model.
- Real-video tracking that becomes invalid instead of inventing occluded positions.
- Identical shared physics code locally and in Lambda with numerical parity checks.
- Honest reporting of adaptive regressions, wrong-target HSV failures, and model-assumption violations.

### Suggested tags

`OpenCV 5`, `Computer Vision`, `Physics-Informed`, `AWS Lambda`, `Serverless`, `Streamlit`, `Trajectory Prediction`, `Adaptive Learning`

## Repository release checks

- [x] OpenCV runtime asserts major version 5.
- [x] Four prediction models and chronological adaptive traces are implemented.
- [x] Synthetic ground truth remains separate from tracker observations.
- [x] Real ROI annotations, uncertainty, failure states, and provenance are documented.
- [x] Lambda is private, uses shared physics code, and has real invocation evidence.
- [x] Runtime/dev dependencies are pinned.
- [x] `.venv`, `.env*`, `.aws/`, keys, archives, generated outputs, and videos are ignored.
- [x] Pexels originals are excluded from Git and downloaded only through documented official links.
- [x] Unauthenticated access to the GitHub repository returned HTTP 200 on 2026-10-23.
- [ ] Open the GitHub URL signed out and visually confirm every judge-required file renders correctly immediately before submission.
- [ ] Add an owner-selected repository license only if public reuse rights are intended; competition submission does not require open-source licensing.

## Reproduction preflight

```bash
git clone https://github.com/panuza1/OpenCV-AWS-hack2026.git
cd OpenCV-AWS-hack2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -c 'import cv2; print(cv2.__version__); assert cv2.__version__.startswith("5.")'
python scripts/generate_sample.py
python scripts/analyze_video.py --video examples/generated/constant_speed.mp4
python scripts/run_benchmark.py
pytest -q
streamlit run app.py
```

For AWS mode, follow [`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md). For independently sourced media, follow [`REAL_VIDEO_VALIDATION.md`](REAL_VIDEO_VALIDATION.md). Do not bundle the downloaded Pexels originals in the submission archive.

## Demo and AWS preflight

- [ ] Verify `AWS_PROFILE=physicseye-cli` resolves to the intended non-root IAM identity.
- [ ] Verify `physicseye-predict` is Active in `ap-southeast-1` immediately before recording/live judging.
- [ ] Run one AWS dashboard prediction and capture the request ID and PASS parity result.
- [ ] Record with the exact flow in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).
- [ ] Verify the final file duration is <=5:00 and audio/text are readable.
- [ ] Watch the entire exported video and confirm every measurement shown came from the running app.
- [ ] Upload public or unlisted and test the URL in a signed-out browser.
- [ ] Replace the placeholder below only after that check.

Demo video URL: **MANUAL — not recorded or published**

## Final Devpost actions

- [ ] Enter the final project title and copy reviewed text from this file.
- [ ] Add the repository URL.
- [ ] Add the verified demo video URL.
- [ ] Add architecture/report links or upload the required files.
- [ ] State that the demonstration path is an arranged live screen-share; provide scheduling/contact details in Devpost if requested.
- [ ] Do not claim COOL or Agentic Vision eligibility.
- [ ] Preview the entry while signed out and test every link.
- [ ] Review Devpost's submitted-material license and confirm rights to every screenshot/video segment.
- [ ] Submit and capture Devpost confirmation plus the final public project URL.

Devpost project URL: **MANUAL — not submitted**

Phase 7 cannot satisfy its final acceptance gate until the demo video and Devpost entry are created and verified by the entrant.

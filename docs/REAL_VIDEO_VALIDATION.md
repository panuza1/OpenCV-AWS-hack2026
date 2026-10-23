# Phase 6 real-video validation

Validated on 2026-10-23 with OpenCV 5.0.0. This report preserves the Phase 3 synthetic benchmarks and evaluates the unchanged Phase 1 orange-ball detector on three independently produced Pexels clips.

## Sources, permission, and download

Each source page existed, named its contributor, and displayed a **Free download** action and free license:

| Local name | Pexels source and contributor | Source-page metadata |
|---|---|---|
| `basketball_6077402.mp4` | [A Ball Rolling on the Floor](https://www.pexels.com/video/a-ball-rolling-on-the-floor-6077402/) by KoolShooters | 4096x2160, 8 seconds, page reports 50 FPS |
| `billiards_6254504.mp4` | [Man Rolling a Billiard Ball on the Top of the Table](https://www.pexels.com/video/man-rolling-a-billiard-ball-on-the-top-of-the-table-6254504/) by Tima Miroshnichenko | 3840x2160, 13 seconds, 25 FPS |
| `disco_5751090.mp4` | [Disco Ball Rolling on the Floor](https://www.pexels.com/video/disco-ball-rolling-on-the-floor-5751090/) by Kaboompics | 4096x2160, 7 seconds, 25 FPS |

The [Pexels license](https://www.pexels.com/license/) permits free use and modification without required attribution. It prohibits selling unaltered copies, redistributing content on another stock platform, implying endorsement, and using content as a trademark. PhysicsEye uses the clips for validation and credits the sources; the original files are excluded from Git.

The files were downloaded through each page's official public download redirect. These commands reproduce that authorized path without scraping page internals or bypassing access controls:

```bash
mkdir -p examples/real
curl --fail --location --output examples/real/basketball_6077402.mp4 https://www.pexels.com/download/video/6077402/
curl --fail --location --output examples/real/billiards_6254504.mp4 https://www.pexels.com/download/video/6254504/
curl --fail --location --output examples/real/disco_5751090.mp4 https://www.pexels.com/download/video/5751090/
```

The first source page reports 50 FPS, while the UHD file returned by its official download action is 25 FPS. Evaluation uses the decoded file metadata.

## Downloaded media inspection

`ffprobe` and OpenCV agree on the following decoded properties:

| Video | Codec | Resolution | FPS | Frames | Duration | Size | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Basketball | H.264 | 4096x2160 | 25 | 197 | 7.88 s | 22,777,799 B | `80b3812a5df08a349508aa7c68c672b216bea2b395dc2b72f1a3bab8643421b8` |
| Billiards | H.264 | 3840x2160 | 25 | 322 | 12.88 s | 22,770,994 B | `2b92e9eb407af69c2ea42049031ccef46c62477da5b43e9bed72ff8314fd1660` |
| Disco | H.264 | 4096x2160 | 25 | 171 | 6.84 s | 19,043,173 B | `66138f884ddbca440857ed1232f18ec529ee5dab64438a715427fb545e0b0d98` |

| Video | Camera and visibility inspection | Fit to current detector |
|---|---|---|
| Basketball | Fixed low camera; ball enters near the top, rolls diagonally toward the camera, grows substantially under perspective, and becomes soft near the frame edge. The orange-brown court fills most of the image. | Poor. Ball and floor have overlapping hue; the accepted contours follow floor features rather than the ball. Perspective also changes ball area substantially. |
| Billiards | Fixed frontal camera; white ball remains visible but is approached and touched by hands. Strong blue/purple illumination changes across the clip and the moving ball has occasional blur. | Incompatible. A low-saturation white ball is outside the hard-coded orange HSV range. |
| Disco | Fixed low camera; several reflective balls are present. Specular highlights and projected light spots change as the large mirrored ball moves. | Incompatible. There is no single orange target; the largest accepted orange contour is a hanging cord. |

## Tracker results and manual spot check

The unchanged detector uses HSV `[5,100,100]` through `[25,255,255]`. Raw `valid` coverage alone is misleading:

| Video | Valid frames | Raw coverage | Manual result |
|---|---:|---:|---|
| Basketball | 79 / 197 | 40.10% | 0 / 8 checked visible-ball frames had an on-ball detection |
| Billiards | 0 / 322 | 0.00% | 0 / 12 checked visible-ball frames had an on-ball detection |
| Disco | 171 / 171 | 100.00% | 0 / 12 checked visible-ball frames had an on-ball detection; all accepted contours were on a cord |

The 32 frame-level checks are recorded in [`real_video_spot_checks.csv`](real_video_spot_checks.csv). The manual label is deliberately binary: whether the reported centroid lies inside the visible target ball. It is not a subpixel ground-truth center annotation. Across this subset, the on-ball hit rate was **0/32**. Fifteen frames contained a false positive on another object, and all 32 visible target balls were missed.

Commands used:

```bash
env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/basketball_6077402.mp4 --output-dir outputs/real
env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/billiards_6254504.mp4 --output-dir outputs/real
env -u PYTHONPATH .venv/bin/python scripts/analyze_video.py --video examples/real/disco_5751090.mp4 --output-dir outputs/real
```

## Four-model pipeline diagnostics

All four models ran on basketball and disco observation streams at causal cutoffs spaced 0.4 seconds apart, with horizons of 0.25, 0.5, and 1.0 seconds. Billiards could not reach the physics engine because it had zero valid observations. Later tracker positions were joined only after each prediction.

These errors measure prediction of the tracker's **wrong target**, not ball trajectory accuracy. They are retained to show why internally consistent forecasts cannot validate perception.

| Video / model | Samples | MAE px | RMSE px | 1.0 s samples | 1.0 s endpoint MAE px |
|---|---:|---:|---:|---:|---:|
| Basketball / Constant velocity | 25 | 510.296502 | 758.938358 | 9 | 669.319247 |
| Basketball / Refitted deceleration | 25 | 510.296502 | 758.938358 | 9 | 669.319247 |
| Basketball / Frozen deceleration | 25 | 342.075553 | 492.603209 | 9 | 408.809467 |
| Basketball / Adaptive deceleration | 25 | 342.075553 | 492.603209 | 9 | 408.809467 |
| Disco / Constant velocity | 39 | 3.263410 | 4.302695 | 13 | 4.987862 |
| Disco / Refitted deceleration | 39 | 3.320568 | 4.374922 | 13 | 5.095203 |
| Disco / Frozen deceleration | 39 | 3.219076 | 4.035971 | 13 | 4.794369 |
| Disco / Adaptive deceleration | 39 | 3.218922 | 4.035118 | 13 | 4.793803 |

At the last sampled cutoffs, basketball produced zero adaptive updates because its sparse false detections did not supply eligible feedback. Disco produced one update and one skipped update while following the cord. Reset behavior remained correct because every video prediction started from a fresh causal replay; no state crossed between videos.

No real-ball MAE, RMSE, endpoint error, deceleration convergence, or adaptive improvement is reported. Those values require a correctly tracked ball and independent ground-truth centers.

## Real AWS parity

The basketball track was replayed at a `5.0 s` cutoff through the existing private Lambda with the same observation prefix and `[0.25, 0.5, 1.0]` horizons:

```text
UTC: 2026-10-23T06:32:30.492919+00:00
Region/function: ap-southeast-1 / physicseye-predict
AWS request ID: 5e9b648a-1e66-4e65-a53d-40e8a2a81283
Invocation latency: 463.882241 ms
Lambda duration / billed duration: 3.36 ms / 113 ms
Local prediction latency: 0.167895 ms
Models returned: 4
Adaptive updates: 0
Parity tolerance: 1e-9
Maximum absolute difference: 1.1368683772161603e-13
Schema mismatches: 0
Result: PASS
```

This establishes deterministic local/cloud execution on observations extracted from a real video. It does not establish ball-tracking or physical-prediction accuracy because the observations follow a floor feature.

## Result and blocker

The current orange HSV detector is unsuitable for all three supplied clips. Phase 6 real-world accuracy remains blocked: no clip yielded a trustworthy target trajectory, so the four-model ablation cannot be interpreted as ball physics.

The smallest reliable unblock is a fixed-camera recording that matches the declared MVP scene: one saturated orange ball, plain contrasting surface, uniform light, little blur, and no occlusion. A broader detector would require a separately validated ROI/color calibration or object model plus new ground-truth annotations; changing the physics engine would not fix these failures.

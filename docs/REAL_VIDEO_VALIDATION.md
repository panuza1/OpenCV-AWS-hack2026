# Phase 6 real-video validation

Validated on 2026-10-23 with OpenCV 5.0.0. The original orange-HSV results are retained as the before baseline. The completed validation uses the user-initialized ROI template tracker; synthetic samples continue to use the unchanged HSV tracker.

## Sources, permission, and download

| Local name | Pexels source and contributor | Decoded file |
|---|---|---|
| `basketball_6077402.mp4` | [A Ball Rolling on the Floor](https://www.pexels.com/video/a-ball-rolling-on-the-floor-6077402/) by KoolShooters | 4096x2160, 25 FPS, 197 frames |
| `billiards_6254504.mp4` | [Man Rolling a Billiard Ball on the Top of the Table](https://www.pexels.com/video/man-rolling-a-billiard-ball-on-the-top-of-the-table-6254504/) by Tima Miroshnichenko | 3840x2160, 25 FPS, 322 frames |
| `disco_5751090.mp4` | [Disco Ball Rolling on the Floor](https://www.pexels.com/video/disco-ball-rolling-on-the-floor-5751090/) by Kaboompics | 4096x2160, 25 FPS, 171 frames |

The [Pexels license](https://www.pexels.com/license/) permits free use and modification without required attribution. It prohibits selling unaltered copies, redistributing content on another stock platform, implying endorsement, and using content as a trademark. The originals remain Git-ignored.

```bash
mkdir -p examples/real
curl --fail --location --output examples/real/basketball_6077402.mp4 https://www.pexels.com/download/video/6077402/
curl --fail --location --output examples/real/billiards_6254504.mp4 https://www.pexels.com/download/video/6254504/
curl --fail --location --output examples/real/disco_5751090.mp4 https://www.pexels.com/download/video/5751090/
```

The downloaded SHA-256 values were `80b3812a…3421b8`, `2b92e9eb…1660`, and `66138f88…e0d98`, respectively. Basketball has strong perspective and leaves the frame; billiards includes hand contact and partial occlusion; the mirrored ball has changing reflections and nearby similar objects.

## Failure diagnosis and minimal fix

The fixed HSV interval `[5,100,100]`–`[25,255,255]` contains the basketball court but not the white billiard ball or mirrored disco ball. Selecting the largest accepted contour supplies no target identity, and the old global detector has no appearance, scale, or local-motion check. Raw coverage therefore hid wrong-object tracks: 40.10% basketball coverage followed floor texture, billiards had 0%, and 100% disco coverage followed a cord. Manual review found 0/32 on-ball detections.

ROI mode takes one initialization box, converts the patch to blurred grayscale, and runs normalized template matching in a motion-constrained local search. Five nearby scales handle modest size change. Confidence, displacement, predicted-position, and size bounds reject background jumps. A low-confidence frame has null coordinates; no occluded position is interpolated. A confident patch may update the template slowly. UHD frames are searched at a maximum dimension of 960 pixels and positions are mapped back to source coordinates. The original HSV mode and synthetic benchmark are unchanged.

Dashboard users choose **Select target ROI**, move to an initialization frame, and set horizontal and vertical bounds around only the ball. The preview, tracked bounding box, confidence, status, current position, and failures are displayed.

## Independent annotations and tracking results

The 34 reference rows in [`real_video_roi_annotations.csv`](real_video_roi_annotations.csv) were manually placed from visible ball outlines before scoring. The three basketball absent-target rows test false recovery. Center uncertainty is ±8 px for basketball, ±12 px for clear billiards frames, ±25 px for partly occluded billiards, and ±25 px for disco. “On ball” means the reported center is within the annotated visible radius; it is separate from coverage.

```bash
env -u PYTHONPATH .venv/bin/python scripts/validate_real_videos.py
```

| Video | HSV on-ball | ROI visible coverage | ROI on-ball | Annotated FP | Center MAE | Real recovery |
|---|---:|---:|---:|---:|---:|---:|
| Basketball | 0/8 | 67/67 (100%) | 7/7 (100%) | 0/3 | 16.121 px | 0 |
| Billiards | 0/12 | 322/322 (100%) | 12/12 (100%) | 0 | 18.143 px | 0 |
| Disco | 0/12 | 171/171 (100%) | 12/12 (100%) | 0 | 3.681 px | 0 |

The declared validation gate requires at least two videos with >=90% annotated on-ball rate, zero annotated false positives, and center MAE <=40 px; 3/3 pass. A separate deterministic test covers loss and recovery after a four-frame occlusion. None of the real clips provides an unambiguous recovery event. Basketball correctly becomes invalid when the target reaches the frame edge and does not recover onto the floor.

One CPU trial on a 12th Gen Intel Core i5-12450H took 2.617 s for 197 basketball frames, 3.526 s for 322 billiards frames, and 4.868 s for 171 disco frames. Mean valid confidence was 0.964, 0.964, and 0.626. The RTX 3050 GPU was not used. Basketball logged 80 pre-initialization frames, one edge failure, and 49 post-exit invalid frames; the other clips had no invalid frames. These host-specific single trials are not a throughput benchmark.

These sparse, manually marked frames validate target identity, not subpixel precision. The disco MAE is below its ±25 px annotation uncertainty, so it must not be read as a 3.7 px ground-truth claim.

## Strict walk-forward physics results

At each annotated cutoff, only tracker observations at or before the cutoff were passed to all four models. The next manual annotation was revealed only after prediction. The JSON export includes prediction rows and causal audits showing history and adaptive-update times did not exceed each cutoff.

| Video / model | n | Horizons | MAE px | RMSE px |
|---|---:|---|---:|---:|
| Basketball / Constant velocity | 4 | 0.40 s | 29.877 | 30.469 |
| Basketball / Refitted deceleration | 4 | 0.40 s | 29.877 | 30.469 |
| Basketball / Frozen deceleration | 4 | 0.40 s | 29.877 | 30.469 |
| Basketball / Adaptive deceleration | 4 | 0.40 s | 29.877 | 30.469 |
| Billiards / Constant velocity | 10 | 1.16–1.20 s | 1288.730 | 1541.619 |
| Billiards / Refitted deceleration | 10 | 1.16–1.20 s | 1055.900 | 1353.826 |
| Billiards / Frozen deceleration | 10 | 1.16–1.20 s | 1288.730 | 1541.619 |
| Billiards / Adaptive deceleration | 10 | 1.16–1.20 s | 838.961 | 1113.656 |
| Disco / Constant velocity | 9 | 0.60–0.64 s | 18.801 | 22.665 |
| Disco / Refitted deceleration | 9 | 0.60–0.64 s | 15.184 | 18.840 |
| Disco / Frozen deceleration | 9 | 0.60–0.64 s | 15.527 | 20.349 |
| Disco / Adaptive deceleration | 9 | 0.60–0.64 s | 13.657 | 18.699 |

Adaptive gain versus frozen was 0% for basketball, 34.900% for billiards, and 12.044% for disco. These are honest image-space forecast results, not calibrated mechanics. Basketball apparently accelerates toward the camera, so a nonnegative-deceleration model collapses to constant velocity. Hands repeatedly redirect/relaunch the billiard ball, causing errors over 800 px despite the numerical adaptive gain. Disco's model differences are smaller than the annotation uncertainty. No real clip supports a strong physical-parameter claim.

The complete machine-readable result, including observations-derived forecasts and update audits, is written to the Git-ignored `outputs/real_roi_validation.json`. The Phase 3 synthetic benchmark and its exact ground truth remain unchanged.

## Real AWS parity

The trusted basketball ROI stream through the 4.8 s cutoff was invoked against the private Lambda with horizons `[0.4, 0.8]`:

```text
UTC: 2026-10-23T07:37:43.527915+00:00
Region/function: ap-southeast-1 / physicseye-predict
AWS request ID: a2bddbaa-c2dd-4943-9b41-fc837936c4c9
Invocation / Lambda / billed duration: 397.621249 ms / 11.43 ms / 86 ms
Local prediction latency: 0.246337 ms
Models returned / adaptive updates: 4 / 3
Parity tolerance / maximum difference: 1e-9 / 5.684341886080802e-14
Schema mismatches: 0
Result: PASS
```

Only the causal observation prefix—not video or future annotations—was sent. This is one invocation, not a latency benchmark.

## Result and limitations

Phase 6 passes its master-plan criteria: three permitted real videos have reproducible target tracking, honest failure cases, four-model causal results, auditable JSON, real Lambda evidence, size/duration limits, tests, dependency checks, and an architecture diagram. The validation also demonstrates why coverage is not accuracy.

The clips do not match the fixed-overhead, free-rolling MVP assumption. Perspective, hand impulses, occlusion, reflections, sparse annotations, and pixel units prevent claims of calibrated physical accuracy. ROI initialization also assumes the selected appearance remains recognizable and the camera is approximately fixed; confidence gates can reject a real ball or accept a similar nearby object.

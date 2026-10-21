# PhysicsEye Phase 3 Evaluation

All forecasts use pixel-space observations available at or before each cutoff. Synthetic ground truth and tracker observations are stored separately. Targets are revealed only after prediction.

## Configuration

- Seed: `20261021`
- Horizons: `[0.25, 0.5, 1.0]` seconds
- Adaptive alpha: `0.25`
- MAE/RMSE/endpoint units: pixels; deceleration units: px/s²
- Endpoint error: mean error at the largest requested horizon (1.0 s)
- Scenario definitions, seed, alpha, horizons, cutoffs, and 10% threshold were frozen before this run

## Scenario results

| Scenario | Model | n | MAE px | Median px | RMSE px | Endpoint px | Parameter MAE px/s² | Parameter std px/s² | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_velocity | constant_velocity | 39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| constant_velocity | fitted_deceleration | 39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| constant_velocity | frozen_deceleration | 39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| constant_velocity | adaptive_deceleration | 39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| constant_deceleration_quantized | constant_velocity | 39 | 1.244 | 0.744 | 1.822 | 2.517 | 6.000 | 0.000 | 0 |
| constant_deceleration_quantized | fitted_deceleration | 39 | 1.479 | 0.878 | 2.185 | 2.769 | 2.945 | 3.588 | 0 |
| constant_deceleration_quantized | frozen_deceleration | 39 | 1.603 | 1.146 | 2.213 | 3.197 | 5.429 | 0.000 | 0 |
| constant_deceleration_quantized | adaptive_deceleration | 39 | 1.409 | 0.925 | 1.877 | 2.608 | 2.902 | 2.061 | 0 |
| abrupt_deceleration_change | constant_velocity | 39 | 2.068 | 1.250 | 2.885 | 4.737 | 8.769 | 0.000 | 0 |
| abrupt_deceleration_change | fitted_deceleration | 39 | 0.419 | 0.000 | 1.039 | 0.968 | 1.231 | 3.427 | 0 |
| abrupt_deceleration_change | frozen_deceleration | 39 | 1.630 | 1.000 | 2.305 | 3.737 | 6.769 | 0.000 | 0 |
| abrupt_deceleration_change | adaptive_deceleration | 39 | 0.501 | 0.108 | 1.066 | 1.155 | 1.606 | 3.209 | 0 |
| gradual_change_noisy | constant_velocity | 39 | 1.558 | 0.953 | 2.240 | 2.832 | 6.000 | 0.000 | 0 |
| gradual_change_noisy | fitted_deceleration | 39 | 2.020 | 1.405 | 2.654 | 3.604 | 3.759 | 4.268 | 0 |
| gradual_change_noisy | frozen_deceleration | 39 | 1.459 | 1.101 | 1.917 | 2.336 | 2.827 | 0.000 | 0 |
| gradual_change_noisy | adaptive_deceleration | 39 | 1.854 | 1.469 | 2.533 | 3.264 | 3.901 | 3.772 | 0 |
| irregular_missing_noisy | constant_velocity | 39 | 1.751 | 1.124 | 2.550 | 3.195 | 6.000 | 0.000 | 0 |
| irregular_missing_noisy | fitted_deceleration | 39 | 2.245 | 1.362 | 3.219 | 4.136 | 3.961 | 4.831 | 0 |
| irregular_missing_noisy | frozen_deceleration | 39 | 2.948 | 1.645 | 4.073 | 6.084 | 11.744 | 0.000 | 0 |
| irregular_missing_noisy | adaptive_deceleration | 39 | 2.533 | 1.499 | 3.641 | 5.074 | 8.597 | 7.288 | 0 |
| opencv_tracker_constant_speed | constant_velocity | 18 | 0.440 | 0.354 | 0.509 | 0.763 | 0.000 | 0.000 | 0 |
| opencv_tracker_constant_speed | fitted_deceleration | 18 | 0.440 | 0.354 | 0.509 | 0.763 | 0.000 | 0.000 | 0 |
| opencv_tracker_constant_speed | frozen_deceleration | 18 | 0.440 | 0.354 | 0.509 | 0.763 | 0.000 | 0.000 | 0 |
| opencv_tracker_constant_speed | adaptive_deceleration | 18 | 1.190 | 0.509 | 1.872 | 2.849 | 6.604 | 3.683 | 0 |
| opencv_tracker_slowing_down | constant_velocity | 18 | 2.279 | 1.142 | 3.143 | 4.594 | 12.000 | 0.000 | 0 |
| opencv_tracker_slowing_down | fitted_deceleration | 18 | 2.969 | 2.509 | 3.727 | 5.527 | 6.509 | 5.949 | 0 |
| opencv_tracker_slowing_down | frozen_deceleration | 18 | 2.213 | 1.379 | 2.950 | 4.003 | 5.273 | 0.000 | 0 |
| opencv_tracker_slowing_down | adaptive_deceleration | 18 | 2.594 | 1.519 | 3.813 | 4.690 | 7.330 | 4.725 | 0 |

## Aggregate error by horizon

| Horizon s | Model | n | MAE px | RMSE px |
|---:|---|---:|---:|---:|
| 0.25 | constant_velocity | 77 | 0.447 | 0.600 |
| 0.25 | fitted_deceleration | 77 | 0.492 | 0.719 |
| 0.25 | frozen_deceleration | 77 | 0.488 | 0.681 |
| 0.25 | adaptive_deceleration | 77 | 0.468 | 0.686 |
| 0.50 | constant_velocity | 77 | 0.882 | 1.190 |
| 0.50 | fitted_deceleration | 77 | 0.999 | 1.468 |
| 0.50 | frozen_deceleration | 77 | 1.038 | 1.405 |
| 0.50 | adaptive_deceleration | 77 | 0.976 | 1.406 |
| 1.00 | constant_velocity | 77 | 2.660 | 3.511 |
| 1.00 | fitted_deceleration | 77 | 2.428 | 3.524 |
| 1.00 | frozen_deceleration | 77 | 2.964 | 3.885 |
| 1.00 | adaptive_deceleration | 77 | 2.630 | 3.755 |

## Adaptive acceptance result

The frozen-model MAE is **0.265838 px** and adaptive MAE is **0.044910 px** over 7 held-out post-change targets. Adaptive improvement is **83.106%**; required threshold is 10%. Result: **PASS**.

## Representative trajectory comparison

The following predictions share the same history, cutoff, horizon, and ground-truth target:

| Cutoff s | Horizon s | Ground truth x/y px | Constant velocity x/y px | Frozen x/y px | Adaptive x/y px |
|---:|---:|---:|---:|---:|---:|
| 2.000 | 0.250 | 92.687/20.000 | 93.111/20.000 | 93.048/20.000 | 92.883/20.000 |

## Measurement and adaptive outcomes by scenario

- **constant_velocity**: observation-vs-truth MAE `0.000 px`; gain vs frozen `n/a%`; gain vs refitted `n/a%`; skipped-update rate `0.000`.
- **constant_deceleration_quantized**: observation-vs-truth MAE `0.236 px`; gain vs frozen `12.107%`; gain vs refitted `4.775%`; skipped-update rate `0.000`.
- **abrupt_deceleration_change**: observation-vs-truth MAE `0.000 px`; gain vs frozen `69.284%`; gain vs refitted `-19.573%`; skipped-update rate `0.000`.
- **gradual_change_noisy**: observation-vs-truth MAE `0.428 px`; gain vs frozen `-27.097%`; gain vs refitted `8.213%`; skipped-update rate `0.000`.
- **irregular_missing_noisy**: observation-vs-truth MAE `0.309 px`; gain vs frozen `14.068%`; gain vs refitted `-12.837%`; skipped-update rate `0.000`.
- **opencv_tracker_constant_speed**: observation-vs-truth MAE `0.036 px`; gain vs frozen `-170.806%`; gain vs refitted `-170.806%`; skipped-update rate `0.000`.
- **opencv_tracker_slowing_down**: observation-vs-truth MAE `0.032 px`; gain vs frozen `-17.207%`; gain vs refitted `12.617%`; skipped-update rate `0.000`.

## Interpretation and limits

- Synthetic scenarios are controlled tests, not evidence of real-world accuracy.
- OpenCV scenarios use generated videos; tracking error is included, but no independently recorded real rolling-ball video is available yet.
- Noisy, quantized, missing, and irregular observations can make the adaptive model worse; negative gains above are retained.
- Ground-truth targets between sampled video frames are linearly interpolated and remain strictly after the cutoff.
- Timing values are host-dependent and excluded from deterministic metric assertions.

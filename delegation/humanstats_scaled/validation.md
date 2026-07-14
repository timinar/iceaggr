# NPE-15 validation gate

Batches [651, 1], 3000 events, 180610 DOM rows joined on (event_id, sensor_id). Unmatched DOM rows: 0.

Our un-normalized features (float64, raw geometry) vs Johann's precomputed store. PASS = max rel < 0.001 or max abs < 1e-06.

| feature | max_abs | max_rel | PASS |
|---|---|---|---|
| c_total | 0.000e+00 | 0.000e+00 | PASS |
| c_500ns | 0.000e+00 | 0.000e+00 | PASS |
| c_100ns | 0.000e+00 | 0.000e+00 | PASS |
| t_first | 0.000e+00 | 0.000e+00 | PASS |
| t_last | 0.000e+00 | 0.000e+00 | PASS |
| t_20 | 0.000e+00 | 0.000e+00 | PASS |
| t_50 | 0.000e+00 | 0.000e+00 | PASS |
| t_mean | 0.000e+00 | 0.000e+00 | PASS |
| t_std | 5.368e-09 | 1.140e-09 | PASS |
| x | 0.000e+00 | 0.000e+00 | PASS |
| y | 0.000e+00 | 0.000e+00 | PASS |
| z | 0.000e+00 | 0.000e+00 | PASS |
| x_rel | 4.547e-13 | 1.994e-12 | PASS |
| y_rel | 5.684e-13 | 3.722e-12 | PASS |
| z_rel | 1.023e-12 | 4.494e-11 | PASS |

**Overall: PASS**

Notes:
- t_20 and t_50 are compared to the store's values, which equal t_first for every DOM. The reference generator's polars `map_elements` runs the charge-cumulative percentile per pulse (not per DOM) and takes `.first()`, collapsing both quantiles to the earliest pulse time. `compute_dom_features_npe` reproduces this by default so the scaled run is a faithful scale-up of the same encoding; pass `correct_percentiles=True` for genuine quantile times.

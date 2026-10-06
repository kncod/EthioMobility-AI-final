# D — Modeling & Evaluation

**Split:** train `pickup_hour < 2025-10-18`, validate `2025-10-18`–`2025-10-31`.  
**Final model:** HistGradientBoostingRegressor (sklearn) with weather + event + calendar/lag features.  
**Winner on holdout:** `hist_gbm` — RMSE 10.664, MAE 6.686.

Code: `src/modeling.py` · Notebook: `notebooks/04_modeling_and_evaluation.ipynb`  
Artifacts: `models/final_model.joblib`, `submission/team_addis_demand_ai_submission.csv`  
Team: **Addis Demand AI**

---

## D1 Baselines

         model   family      rmse       mae  train_seconds
seasonal_naive baseline 12.078873  7.438533       0.050802
 mean_baseline baseline 27.878944 20.809506       0.002895

**Interpretation:** Seasonal naive (zone×dow×hour) is the real bar to beat; a global mean is far too weak.

## D2 Model comparison

         model   family      rmse       mae  train_seconds
      hist_gbm       ml 10.664282  6.686266      10.371406
 random_forest       ml 10.933406  6.749322      28.660057
seasonal_naive baseline 12.078873  7.438533       0.050802
         ridge       ml 19.592150 13.336310       0.422604
 mean_baseline baseline 27.878944 20.809506       0.002895

**Winner:** `hist_gbm` — best RMSE/MAE tradeoff on the same chronological split and features. Trees capture nonlinear rain/event effects better than ridge; HistGBM trains faster than a deep RF here.

## D3 Rolling-origin validation

 fold  train_end valid_start  valid_end  hist_gbm_rmse  hist_gbm_mae  seasonal_naive_rmse  n_train  n_valid
    1 2025-08-23  2025-08-23 2025-09-06      10.573204      6.879218            12.287298    63460     3937
    2 2025-09-06  2025-09-06 2025-09-20      11.255245      7.167531            13.670984    67397     3930
    3 2025-09-20  2025-09-20 2025-10-04      11.298576      7.162298            14.086733    71327     3926
    4 2025-10-04  2025-10-04 2025-10-18      11.510908      6.797141            12.151328    75253     3935
    5 2025-10-18  2025-10-18 2025-11-01      10.664282      6.686266            12.078873    79188     3914

- HistGBM RMSE mean±std: **11.060 ± 0.416**
- Seasonal naive RMSE mean±std: **12.855 ± 0.949**

**Interpretation:** Spread across folds shows stability; any fold that is unusually bad usually coincides with holiday/event-dense fortnights where residual event risk is higher.

## D4 Feature availability & leakage audit

Full table: `reports/D_tables/D4_leakage_audit.csv`.

Excluded from model inputs: `avg_fare_birr`, `avg_wait_min`, `active_drivers` — not known at forecast time.

Leaky demo (illegal `active_drivers` included): RMSE = **5.928** vs honest model ~10.160. Looks better on validation but would be unavailable for Nov forecasts — production failure / leakage.

## D5 Ablation

Ablation rerun with **tuned** HistGBM hyperparameters (fair comparison):

| feature_set | n_features | rmse | mae | rmse_delta_vs_calendar |
|-------------|------------|------|-----|------------------------|
| calendar_zone_trend | 10 | 10.433 | 6.503 | 0.000 |
| plus_weather | 16 | 10.453 | 6.532 | +0.020 |
| plus_events | 19 | 10.351 | 6.485 | −0.082 |
| plus_both | 25 | **10.160** | **6.405** | **−0.273** |

**Interpretation:** With tuned params, events alone improve RMSE (~0.08). Weather alone is nearly flat on this fortnight, but **weather+events together** give the best score (−0.27 vs calendar-only). Both joins are kept (Rule 5); weather still matters for rainy/event interactions and generalization beyond this holdout.

## D6 Tuning

- Trials: **12** (random search, time-ordered single validation fortnight)
- RMSE before → after: **10.664 → 10.160**
- Best params: `{'min_samples_leaf': 40, 'max_iter': 350, 'max_depth': 10, 'learning_rate': 0.08, 'l2_regularization': 1.0}`

Search space:
```json
{
  "learning_rate": [
    0.05,
    0.08,
    0.12
  ],
  "max_depth": [
    6,
    8,
    10
  ],
  "max_iter": [
    150,
    250,
    350
  ],
  "min_samples_leaf": [
    10,
    20,
    40
  ],
  "l2_regularization": [
    0.0,
    0.1,
    1.0
  ]
}
```

## D7 Error analysis

Overall validation RMSE/MAE: **10.160 / 6.405**

### By zone

     zone      rmse      mae     n
  Merkato 13.544703 8.206861 328.0
   Piassa 13.393601 6.706240 324.0
Megenagna 13.189269 8.416415 325.0
Kazanchis 13.147412 8.177055 331.0
     Bole 11.090821 7.810951 320.0
   Lideta  8.815292 5.909062 322.0
      CMC  8.663015 6.180815 324.0
Arat Kilo  7.762604 5.319405 327.0
    Kolfe  7.756945 5.545024 332.0
    Gerji  7.273656 5.479205 323.0
   Sarbet  6.700733 4.896422 328.0
     Ayat  5.969748 4.249309 330.0

### By day type

day_type      rmse      mae      n
 weekday  9.717489 6.467259 2805.0
 weekend 11.200659 6.248181 1109.0

### Top 10 absolute errors

     zone         pickup_hour  hour  trips       pred  abs_error  is_football_window  is_public_holiday  rain_mm                                                             hypothesis
   Piassa 2025-10-19 12:00:00    12  192.0  27.780477 164.219523                   0                  0     0.00                 Possible unlisted event, local spike, or lag mismatch.
  Merkato 2025-10-18 06:00:00     6  192.0  52.592316 139.407684                   0                  0     0.00                 Possible unlisted event, local spike, or lag mismatch.
Kazanchis 2025-10-25 18:00:00    18  192.0  76.714712 115.285288                   1                  0     0.00   Football window — event magnitude may exceed average uplift feature.
Megenagna 2025-10-28 02:00:00     2   96.0   8.515830  87.484170                   0                  0     0.00                 Possible unlisted event, local spike, or lag mismatch.
Megenagna 2025-10-28 18:00:00    18  174.0  92.195503  81.804497                   0                  0     0.00        Peak commute hour — residual peak sharpness not fully captured.
Kazanchis 2025-10-28 18:00:00    18  176.0  99.215723  76.784277                   0                  0     0.00        Peak commute hour — residual peak sharpness not fully captured.
     Bole 2025-10-26 18:00:00    18  137.0  64.361103  72.638897                   0                  0     0.00        Peak commute hour — residual peak sharpness not fully captured.
Megenagna 2025-10-29 18:00:00    18  183.0 115.150750  67.849250                   0                  0     3.50 Heavy rain hour — nonlinear rain response / sparse heavy-rain history.
   Piassa 2025-10-29 20:00:00    20  110.0  45.092272  64.907728                   0                  0     1.15                 Possible unlisted event, local spike, or lag mismatch.
   Lideta 2025-10-28 18:00:00    18  114.0  66.387699  47.612301                   0                  0     0.00        Peak commute hour — residual peak sharpness not fully captured.

## D8 Response to findings

- Change: Added is_peak_hour and rain_x_weekend based on peak-hour / rain error patterns
- RMSE 10.160 → 10.277 (helped=False)
- Extra features kept in final model: none (reverted)

## D9 Plain-language metric

On the Oct 18–31 validation fortnight, typical absolute error is about **6.4 trips per zone-hour** (MAE), roughly **23% of mean demand** (28.4 trips/hour). RMSE is **10.2 trips** (~36% of mean), penalizing larger misses. At ~1.3 trips per active driver-hour, that is roughly **±4.9 drivers** on average (RMSE ≈ **7.8 drivers**) for staffing a single zone-hour — useful as a buffer when deploying.

---

## Submission hygiene

- Rows: 4032
- Min/max prediction: 1.776 / 128.148
- Features used (25): ['zone', 'hour', 'dow', 'is_weekend', 'month', 'is_payday_window', 'week_index', 'temp_c', 'rain_mm', 'humidity_pct', 'wind_kmh', 'rain_class', 'rain_last_3h', 'in_event_window', 'is_public_holiday', 'is_football_window', 'is_concert_window', 'is_road_closure', 'hours_to_next_football', 'hours_since_football', 'event_attendance_nearby', 'n_events_overlapping', 'trips_lag_168h', 'trips_roll_mean_24h', 'trips_roll_mean_168h']

# Model card — Addis Demand AI (HistGBM)

## Intended use

Hourly **ride-demand forecasts** for **12 Addis Ababa zones** for the scoring window **1–14 November 2025**.

- **Primary users:** hackathon judges and ops staff exploring a 24h zone plan.
- **Inputs at inference:** zone + calendar date only (demo). Weather and confirmed events are looked up from bundled tables; the model never receives fare, wait, or active-driver columns.
- **Outputs:** predicted trips per zone-hour; demo also shows drivers ≈ trips ÷ 1.3, day fares ≈ trips × zone avg fare, and an uncertainty band of ± validation MAE.

**Out of scope:** real-time dispatch, pricing, or cities other than the 12 training zones.

## Training & validation data

| | Window | Notes |
|--|--------|--------|
| **Train** | `pickup_hour < 2025-10-18` | Chronological only (Rule 7) |
| **Validate** | 2025-10-18 → 2025-10-31 | Same feature set as production |
| **Score / demo** | 2025-11-01 → 2025-11-14 | Test file; no target used in training |
| **Timezone** | Africa/Addis_Ababa (EAT) | Weather converted from UTC |

Sources joined into `master_*`: cleaned trips, hourly weather, confirmed events (football/concert ±2h windows).

## Model

| Field | Value |
|-------|--------|
| **Family** | `sklearn.ensemble.HistGradientBoostingRegressor` |
| **Artifact** | `models/final_model.joblib` (`pipeline`, `features`, `params`, `val_rmse`, `val_mae`) |
| **Seed** | `random_state=42` |
| **Validation (honest)** | RMSE ≈ **10.16** · MAE ≈ **6.41** trips/hour |
| **vs seasonal naive** | ≈ 12.08 RMSE on the same split |
| **Rolling-origin (5 folds)** | RMSE **11.06 ± 0.42** (see D3) |

### Tuned hyperparameters (D6 winner)

```text
learning_rate:      0.08
max_depth:          10
max_iter:           350
min_samples_leaf:   40
l2_regularization:  1.0
```

(D2 untuned HistGBM on the same split ≈ **10.66** RMSE — headline **10.16** is after tuning.)

### Features included (25)

Calendar: `zone`, `hour`, `dow`, `is_weekend`, `month`, `is_payday_window`, `week_index`  
Weather: `temp_c`, `rain_mm`, `humidity_pct`, `wind_kmh`, `rain_class`, `rain_last_3h`  
Events: `in_event_window`, `is_public_holiday`, `is_football_window`, `is_concert_window`, `is_road_closure`, `hours_to_next_football`, `hours_since_football`, `event_attendance_nearby`, `n_events_overlapping`  
Lags: `trips_lag_168h`, `trips_roll_mean_24h`, `trips_roll_mean_168h`

### Features excluded (leakage)

`avg_fare_birr`, `avg_wait_min`, `active_drivers` — not known at forecast time. Including `active_drivers` illegally drops val RMSE to ~5.93 and would fail in production (D4).

## How to retrain

```bash
python src/run_pipeline.py          # refresh masters if raw data changed
python src/modeling.py              # D1–D9, overwrite model + submission
# or
python -m streamlit run app/app.py  # demo uses models/ + app/assets/
```

## Limitations & risks

1. **Ayat late launch** — fewer history hours than other zones; lag features and errors are less reliable. The demo surfaces a caution when Ayat is selected.
2. **Weather-forecast error** — Nov inference uses *provided* weather tables as if known; real ops would substitute imperfect forecasts, so rain-driven lifts/drops are optimistic.
3. **Event calendar coverage** — only listed, **confirmed** events attach; unlisted spikes (see B3.4) are under-explained.
4. **Chronological drift** — rolling-origin RMSE (~11.1) is higher than the Oct holdout (~10.2); expect some degradation further into November.
5. **Zone coverage** — model is fit only on the 12 canonical zones; aliases outside the map are dropped/mapped at clean time.
6. **Uncertainty band** — demo ±MAE is a simple global band, not a calibrated prediction interval per hour.

## Ethics & dual use

Forecasts are for staffing/planning discussion in a competition setting. Do not treat scores as ground truth for safety-critical or contractual decisions without local validation.

## References

- Evaluation write-up: `reports/D_model_evaluation.md`
- Feature dictionary: `data/processed/data_dictionary_master.csv`
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)

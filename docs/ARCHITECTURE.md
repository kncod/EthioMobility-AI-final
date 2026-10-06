# Architecture — Addis Demand AI

End-to-end flow from raw CSVs to the Streamlit demo and Nov 2025 submission.

## Data flow

```text
data/raw/
  ride_demand_train.csv
  ride_demand_test.csv
  weather_hourly.csv
  events_calendar.csv
  submission_template.csv
        │
        ▼
┌───────────────────────┐
│  src/cleaning.py      │  A — clean, standardize zones/tz, join, features
│  (run_pipeline.py)    │
└───────────┬───────────┘
            │
            ▼
data/processed/
  master_train.csv · master_test.csv
  data_dictionary_master.csv
  weather_clean.csv · events_clean.csv
            │
     ┌──────┼──────────────────┐
     ▼      ▼                  ▼
┌────────┐ ┌────────────────┐ ┌──────────────────┐
│analysis│ │ visualizations │ │ modeling.py      │
│  .py   │ │     .py        │ │ D — train/eval   │
│ B-report│ │ C — fig01–12   │ │ final_model      │
└────────┘ └────────────────┘ └────────┬─────────┘
                                       │
                    models/final_model.joblib
                    submission/team_*_submission.csv
                    app/assets/* (demo lookups)
                                       │
                                       ▼
                              ┌────────────────┐
                              │  app/app.py    │  E — zone + date only
                              │  Streamlit     │
                              └────────────────┘
```

## Component map

| Module | Role | Main outputs |
|--------|------|----------------|
| `src/cleaning.py` | Parse timestamps (UTC→EAT), canonicalize zones/`event_type`, weather join, event windows (±2h football/concert), calendar/lag features, integrity checks | `data/processed/*`, `reports/A*`, `app/assets/weather_clean.csv`, `events_clean.csv` |
| `src/run_pipeline.py` | CLI entry for cleaning + export | same as above |
| `src/analysis.py` | B1–B4 demand / weather / event / gap analysis | `reports/B_analysis_report.md`, `reports/B_tables/` |
| `src/visualizations.py` | Twelve required figures (≥1200px) + captions | `figures/fig01`–`fig12`, `figure_captions.md` |
| `src/modeling.py` | Baselines, HistGBM, rolling-origin, leakage, ablation, tune, submission | `models/final_model.joblib`, `submission/…csv`, `reports/D_*` |
| `src/make_slides.py` | Build 5-slide deck from pack figures | `presentation/*.pptx` |
| `app/app.py` | Ops demo: zone + date → 24h forecast, drivers, fares, city overview | Live Streamlit UI |

## Notebooks (same order)

1. `notebooks/01_cleaning_and_integration.ipynb` → A  
2. `notebooks/02_analysis_report.ipynb` → B  
3. `notebooks/03_visualizations.ipynb` → C  
4. `notebooks/04_modeling_and_evaluation.ipynb` → D  

## Paths & reproducibility

- Code uses **relative paths** from the repo root (`Path(__file__).parents[1]`).
- Seeds: `random_state=42` in modeling / viz model snapshots.
- Lag and imputation statistics are fit on **train only** (Rule 8).
- Demo never asks for weather/events — it looks them up from bundled `app/assets/`.

See also: [MODEL_CARD.md](MODEL_CARD.md), root [README.md](../README.md).

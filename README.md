# Addis Ride Demand Forecasting — Addis Demand AI

Qiyas AI Hackathon #2: Addis Ride Demand Forecasting Challenge  
**Track:** Intelligent Data & AI Engineering (IDAE)  
**Program:** EDI-Qiyas-CoDiST Advanced Digital Skills Program  
**Repo:** [kncod/EthioMobility-AI](https://github.com/kncod/EthioMobility-AI)

We forecast hourly ride demand for 12 Addis Ababa zones (1–14 Nov 2025) by cleaning and joining trips, weather, and events; engineering calendar/weather/event/lag features known at forecast time; and training a chronological HistGradientBoosting model. Validation RMSE ≈ **10.16** trips/hour (MAE ≈ 6.41) vs seasonal naive ≈ 12.08. A Streamlit demo lets ops pick a zone and date for a 24h plan.

## Team

| | |
|--|--|
| **Team name** | Addis Demand AI |
| **Members** | Abraham Getachew, Ahmed Hussen, Natanim Masresha, Nigus Shiferaw, Tekilu Asefa |

## Setup

```bash
git clone https://github.com/kncod/EthioMobility-AI.git
cd EthioMobility-AI
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Full local stack: `pip install -r requirements-dev.txt`.  
Demo only: `pip install -r app/requirements.txt`.

## Run order

1. **Cleaning & integration (A)** — `python src/run_pipeline.py` or `notebooks/01_cleaning_and_integration.ipynb`
2. **Analysis (B)** — `python src/analysis.py` or `notebooks/02_analysis_report.ipynb`
3. **Figures (C)** — `python src/visualizations.py` or `notebooks/03_visualizations.ipynb`
4. **Modeling (D)** — `python src/modeling.py` or `notebooks/04_modeling_and_evaluation.ipynb`
5. **Demo (E)** — `python -m streamlit run app/app.py`
6. **Slides (F)** — `presentation/team_addis_demand_ai_slides.pptx` (`python src/make_slides.py`)

## Deliverables

- Raw CSVs in `data/raw/` (never edited)
- Processed: `data/processed/master_train.csv`, `master_test.csv`, `data_dictionary_master.csv`
- Reports: `reports/A_*.md`, `B_analysis_report.md`, `D_model_evaluation.md`
- Figures: `figures/fig01`–`fig12` + `figure_captions.md`
- Model: `models/final_model.joblib`
- Submission: `submission/team_addis_demand_ai_submission.csv` (4,032 rows)
- Demo: `app/app.py` (local: http://localhost:8501)
- Slides: `presentation/team_addis_demand_ai_slides.pptx`

## Validation score (chronological, 18–31 Oct)

- **Tuned HistGBM RMSE ≈ 10.16** · MAE ≈ 6.41 (final model / D6) — headline score
- Untuned HistGBM (D2 default params / fig10 bar) RMSE ≈ **10.66** — same family before D6 tuning
- Rolling-origin (D3 + fig10 error bar, 5 folds): RMSE **11.06 ± 0.42**
- Seasonal naive RMSE ≈ 12.08 · Mean baseline ≈ 27.88

## Demo

**Live:** https://ethiomobility-ai-6vn5hrkftvxgek3su5xkie.streamlit.app/

```bash
python -m streamlit run app/app.py
```

Pick a zone and any date in **1–14 Nov 2025**. Weather/events load automatically.  
Dashboard includes uncertainty band, vs-typical %, city overview tab, and CSV download.

## Notes

- Timestamps after cleaning use **Africa/Addis_Ababa**.
- Model excludes `avg_fare_birr`, `avg_wait_min`, `active_drivers` (not known at forecast time).
- Validation is chronological — never random split, never score on the test file.
- **Streamlit Cloud (important):** Main file = `app/app.py`, requirements = `app/requirements.txt`, Python **3.11**.  
  Cloud ignores `.python-version`. To change Python you must **delete the app and redeploy**, then in **Advanced settings** pick Python 3.11 (do not leave 3.14). See [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

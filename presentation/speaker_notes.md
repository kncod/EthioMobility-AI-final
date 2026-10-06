# F — 5-minute talk guide — Addis Demand AI

**Team:** Addis Demand AI  
**Members:** Abraham Getachew, Ahmed Hussen, Natanim Masresha, Nigus Shiferaw, Tekilu Asefa  
**Track:** IDAE · EDI-Qiyas-CoDiST Advanced Digital Skills Program

Open: `presentation/team_addis_demand_ai_slides.pptx`  
Rebuild: `python src/make_slides.py`  
Demo: `python -m streamlit run app/app.py` → http://localhost:8501

## Timing (~5 min)

| Slide | ~Time | Say |
|-------|-------|-----|
| 1 Problem | 45s | Introduce team briefly. Ops needs zone×hour demand 1–14 Nov. Three messy tables. |
| 2 Cleaning | 75s | Zones + **UTC→EAT clock**. Weather 100% match. Events ±2h. Show fig06. |
| 3 Findings | 60s | Growth ~30%. Football uplift before kickoff. Show fig03 + fig08. |
| 4 Modeling | 75s | Seasonal 12.1 → HistGBM **10.16**. Ablation −0.27. Rolling ±0.4. Show fig10. |
| 5 Demo/next | 45s | Errors on peaks/events. Demo link. Next: intervals, Ayat, drift. |

## Live demo (Q&A)

1. Open Streamlit.
2. Zone **Kazanchis**, date **2025-11-09** (match day if present).
3. Point to lookup line (rain/events), peak hour, drivers column.
4. Let judges pick another zone/date.

## Checklist before presenting

- [ ] Slides open offline
- [ ] Demo running locally
- [ ] Submission CSV: `submission/team_addis_demand_ai_submission.csv`
- [ ] Know RMSE 10.16 / MAE 6.41 / seasonal 12.1 by heart

# E — Deployed Forecast Demo (upgraded)

**Team:** Addis Demand AI  
**Members:** Abraham Getachew, Ahmed Hussen, Natanim Masresha, Nigus Shiferaw, Tekilu Asefa

## Run locally

```bash
python -m streamlit run app/app.py
```

→ http://localhost:8501

## What’s in the dashboard

### Tab 1 — Zone forecast
- Team branding header
- Metrics: peak hour/trips, drivers, **day total**, **% vs typical weekday**, day fares
- Auto **lookup** line (weather + named events)
- Events table for the day
- Chart: forecast + **uncertainty band** (± validation MAE) + typical profile + event shading
- Weather subplot (rain + temp)
- Ayat late-launch caution badge
- 24h table with low/mid/high + **CSV download**
- Sidebar model card (no leakage features)

### Tab 2 — City overview (stretch: interactive explorer)
- All 12 zones for the selected date
- Day totals bar chart + % vs typical
- Peak hour table + CSV download

## Ops guidance for uncertainty
Staff to the mid forecast; use the high band as a driver buffer on rain/event days (~trips_high ÷ 1.3).

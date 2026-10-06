# A — Data Cleaning & Integration Pipeline

Team: **Addis Demand AI** (Abraham Getachew, Ahmed Hussen, Natanim Masresha, Nigus Shiferaw, Tekilu Asefa)  
Clock used everywhere after cleaning: **Africa/Addis_Ababa (EAT, UTC+3)**.

Code: `src/cleaning.py` · Notebook: `notebooks/01_cleaning_and_integration.ipynb`  
Run: `python src/run_pipeline.py`

---

## A1 Cleaning log

See `reports/A1_cleaning_log.csv` and `data/processed/cleaning_log.csv` (**22** issues logged across all three tables).

Highlights:

| File | Issue | Fix |
|------|-------|-----|
| trips | 55 zone spellings | Mapped to 12 canonical zones |
| trips | Mixed datetime formats | Parsed ISO+03 / Y-M-D / D/M/Y; floored to hour |
| trips | Negative trips / wait | Set to NaN; trips capped at 99.9th pct |
| weather | UTC `Z` timestamps | Converted to EAT |
| weather | `rain_mm = -9999` | Treated as missing |
| weather | `temp_c > 45` | Set to NaN |
| weather | Duplicate hours | Mean-aggregated to unique hour |
| events | Messy types/status | Normalized; cancelled kept but not used in features |
| events | Multi-zone / citywide | Expanded to long event×zone rows |
| events | Missing/inverted ends | Type-specific defaults |

---

## A2 Time & key standardization

### Zones before → after

- **Before (train):** 55 raw labels (case, spaces, aliases like `Piazza`, `Kazanches`, `Bole Rd`, `C.M.C`, `Kolfe Keranio`, `Mercato`, `Megenaga`).
- **After:** exactly the 12 canonical zones:  
  Arat Kilo, Ayat, Bole, CMC, Gerji, Kazanchis, Kolfe, Lideta, Megenagna, Merkato, Piassa, Sarbet  
  (same list on train, test, and expanded events).

### `event_type` before → after

| | Unique labels |
|--|--|
| **Before (raw)** | 20 messy labels: `CONCERT`, `Concert`, `concert`, `Football Match`, `football match `, `football_match`, `Public Holiday`, `public holiday`, `public_holiday`, `Road closure` / `road_closure`, `School Break` / `school_break`, `Sports Run` / `sports_run`, `Conference` / `conference`, `Exhibition` / `exhibition` |
| **After** | 8 canonical types: `concert`, `conference`, `exhibition`, `football_match`, `public_holiday`, `road_closure`, `school_break`, `sports_run` |

### Timestamp formats parsed

| Table | Formats |
|-------|---------|
| trips | `YYYY-MM-DD HH:MM`, `DD/MM/YYYY HH:MM`, ISO with `+03:00` |
| weather | ISO `...Z` (UTC), some `DD/MM/YYYY HH:MM` |
| events | `YYYY-MM-DD HH:MM`, `DD/MM/YYYY HH:MM`, `Mon DD, YYYY h:mm AM/PM` |

Slash dates are parsed **day-first**.

### Timezone proof

1. **Trip evidence:** a large share of `pickup_hour` values carry an explicit `+03:00` offset → EAT.
2. **Weather evidence:** most `timestamp` values end in `Z` → UTC.
3. **Conversion:** `UTC → Africa/Addis_Ababa`, then drop tz for naive EAT wall-clock joins.
4. **Sanity check:** after conversion, mean `temp_c` peaks in mid-afternoon EAT (see notebook figure), which matches local climate expectation. Joining without the +3h shift would misalign rain vs demand.

---

## A3 Join map

```
ride_demand_*  (LEFT, grain = zone × hour)
      |
      | many-to-one: pickup_hour == weather.timestamp (EAT)
      v
weather_hourly (deduped to 1 row/hour)
      |
      | interval join: zone match AND hour ∈ [start−pre, end+post]
      v
events_calendar (confirmed only; citywide → all 12 zones)
      |
      v
master_train / master_test + features
```

**Why trips are left:** every scored zone-hour must survive.  
**Event windows:** football/concert **±2h**; sports_run ±1h; holidays/closures exact interval; **confirmed only**.

---

## A4 Join audit

From `reports/A4_join_audit.json` (re-run pipeline to refresh):

- **Weather:** left row count unchanged (many-to-one). After reindexing weather onto a complete hourly spine + interpolation, join **match rate = 100%** (0 zone-hours without weather).
- **Events:** 159 confirmed event IDs; all matched ≥1 zone-hour under the window rule; 6 cancelled excluded from feature attachment.

---

## A5 Join proof

Three concrete zone-hours from `master_train` after the weather + events joins (EAT). Each shows the attached weather row fields and resulting event features.

### 1) Rain — Arat Kilo · 2025-05-02 17:00 EAT

| Field | Value |
|-------|-------|
| `trips` | 96 |
| Weather row (`weather_clean`) | `temp_c=21.5`, `rain_mm=21.0`, `humidity_pct=76.0` |
| `rain_class` | `heavy` |
| Event flags | `in_event_window=0`, `is_football_window=0`, `is_public_holiday=0`, `n_events_overlapping=0` |

**Proof:** heavy rain attaches from the weather spine; no event features fire.

### 2) Football window — Kazanchis · 2025-01-26 13:00 EAT

| Field | Value |
|-------|-------|
| `trips` | 37 |
| Weather | `temp_c=22.6`, `rain_mm=0.0`, `rain_class=none` |
| Event flags | `in_event_window=1`, `is_football_window=1`, `is_public_holiday=0`, `event_attendance_nearby=34756`, `n_events_overlapping=2` |
| Matched event | `EVT-0007` Premier league match · `football_match` · 15:00–17:00 · attendance 34756 · **confirmed** (±2h window → 13:00 included) |

**Proof:** pre-kickoff hour is inside the ±2h football window; attendance and football flag attach.

### 3) Public holiday — Arat Kilo · 2025-01-07 00:00 EAT

| Field | Value |
|-------|-------|
| `trips` | 4 |
| Weather | `temp_c=8.7`, `rain_mm=0.0`, `rain_class=none` |
| Event flags | `in_event_window=1`, `is_public_holiday=1`, `is_football_window=0`, `n_events_overlapping=1` |
| Matched event | `EVT-0001` Genna (Ethiopian Christmas) · `public_holiday` · all day · **confirmed** |

**Proof:** holiday interval join sets `is_public_holiday` without a football window.

---

## A6 Feature engineering

≥8 engineered features with forecast-time flags **and** why-each-helps — full table in `data/processed/data_dictionary_master.csv` (columns include `known_at_forecast_time` and `why_expect_help`).

Includes:

- **Calendar (≥3):** `hour`, `dow`, `is_weekend`, `month`, `is_payday_window`, `week_index`
- **Weather (≥2):** `rain_class`, `rain_last_3h` (+ raw weather columns)
- **Events (≥3):** `in_event_window`, `is_football_window`, `hours_to_next_football`, `event_attendance_nearby`, …
- **Lag/trend (≥1):** `trips_lag_168h`, `trips_roll_mean_24h`, `trips_roll_mean_168h`, `week_index`

| Feature | Known at forecast? | Why we expect it to help |
|---------|--------------------|--------------------------|
| `hour` / `dow` / `is_weekend` | yes | Diurnal + weekday/weekend shapes dominate demand |
| `rain_mm` / `rain_class` / `rain_last_3h` | yes | Rain suppresses walking and lifts ride hailing |
| `is_football_window` / `event_attendance_nearby` | yes | Stadium crowds create sharp local spikes |
| `is_public_holiday` | yes | Holiday calendars change zone mix |
| `trips_lag_168h` / roll means | yes (history) | Same weekday-hour last week is a strong baseline |

**Excluded from model inputs (leakage):** `avg_fare_birr`, `avg_wait_min`, `active_drivers`.

---

## A7 Integrity checks

See `reports/A7_integrity_checks.md`. Latest run: **all PASS** (unique zone-hours, 12 zones, date ranges, no negative trips, weather complete, test=4032 rows, no leaky features in model list).

---

## A8 Master tables

| File | Location |
|------|----------|
| `master_train.csv` | `data/processed/` |
| `master_test.csv` | `data/processed/` (same feature columns; **target/ops columns dropped**) |
| `data_dictionary_master.csv` | `data/processed/` |

Test rows use the **same pipeline**; lag medians and imputations are learned from train only. Exported `master_test.csv` omits `trips` / `trips_raw` / `avg_*` / `active_drivers`.

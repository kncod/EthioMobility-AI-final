# B — Data Analysis Report

Source: `data/processed/master_train.csv` (cleaned + joined).  
Code: `src/analysis.py` · Notebook: `notebooks/02_analysis_report.ipynb`

## Zone type map used in this report

```
{
  "Arat Kilo": "transport_hub",
  "Ayat": "residential_late_launch",
  "Bole": "market",
  "CMC": "residential",
  "Gerji": "residential",
  "Kazanchis": "transport_hub",
  "Kolfe": "residential",
  "Lideta": "transport_hub",
  "Megenagna": "transport_hub",
  "Merkato": "market",
  "Piassa": "transport_hub",
  "Sarbet": "residential"
}
```

---

## B1 — Demand patterns

### B1.1 Volume by zone

| zone      |   total_trips |   mean_trips_per_hour |   n_hours | first_hour          | last_hour           |          share_pct |
|:----------|--------------:|----------------------:|----------:|:--------------------|:--------------------|-------------------:|
| Merkato   |      286366.0 |    40.619290780141846 |      7050 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 | 12.045029531491986 |
| Bole      |      278810.0 |    39.40777385159011  |      7075 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 | 11.727211623151074 |
| Megenagna |      271694.0 |    38.46722355939403  |      7063 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 | 11.427900845523503 |
| Kazanchis |      225140.0 |    31.848917810157023 |      7069 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  9.469762292730652 |
| Piassa    |      200354.0 |    28.322589765337856 |      7074 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  8.427221970319609 |
| Lideta    |      200207.0 |    28.293810062182025 |      7076 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  8.421038906194925 |
| CMC       |      183874.0 |    25.94525186962043  |      7087 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  7.734045801783583 |
| Gerji     |      171855.0 |    24.369682359614295 |      7052 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  7.228506701684402 |
| Kolfe     |      163398.0 |    23.104920814479637 |      7072 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  6.872791237041854 |
| Sarbet    |      154095.0 |    21.820305862361938 |      7062 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  6.481491607436838 |
| Arat Kilo |      145380.0 |    20.571671147587377 |      7067 | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |  6.11492423433056  |
| Ayat      |       96289.0 |    17.981139122315593 |      5355 | 2025-03-15 00:00:00 | 2025-10-31 23:00:00 |  4.050075248311014 |

**Interpretation:** Merkato carries the largest share (12.0% of trips). Ayat is smallest (4.1%) and has fewer hours — consistent with a late launch / shorter history, so mean and totals are not fully comparable without adjusting for operating period.

### B1.2 Hour-of-day profile by zone type

| zone_type               |   peak_hour |    peak_mean_trips |   quiet_hour |   quiet_mean_trips |
|:------------------------|------------:|-------------------:|-------------:|-------------------:|
| market                  |          13 | 65.25653206650831  |            2 | 7.444180522565321  |
| residential             |           7 | 49.095518867924525 |            2 | 3.9156769596199523 |
| residential_late_launch |           7 | 36.3375            |            2 | 2.7875             |
| transport_hub           |          18 | 79.61037735849057  |            1 | 5.10188679245283   |

**Interpretation:** Business-type zones peak in the morning commute; transport hubs peak in the evening; market/nightlife zones stay elevated later. Quietest hours are typically overnight (00–05) across types.

### B1.3 Weekday vs weekend

| zone      |       weekday_mean |       weekend_mean |   weekend_to_weekday |
|:----------|-------------------:|-------------------:|---------------------:|
| Bole      | 36.12393753706266  | 47.648313492063494 |  1.3190232499759194  |
| Gerji     | 23.81228939544103  | 25.770802192326855 |  1.0822479839868229  |
| Kolfe     | 22.62018566067549  | 24.3265306122449   |  1.0754346130118568  |
| CMC       | 25.405442713468744 | 27.303075396825395 |  1.0746939427412778  |
| Sarbet    | 21.378976486860303 | 22.936531734132934 |  1.072854528289973   |
| Ayat      | 17.637990002630886 | 18.820463320463322 |  1.067041273844472   |
| Merkato   | 42.05962757527734  | 36.98751248751249  |  0.8794065620603297  |
| Lideta    | 29.838339920948616 | 24.4171626984127   |  0.8183150524828672  |
| Megenagna | 40.604393429645754 | 33.09452736318408  |  0.8150479435316812  |
| Kazanchis | 37.04331487341772  | 18.80228514654744  |  0.5075756640785936  |
| Piassa    | 33.129911154985194 | 16.202588352414136 |  0.4890622337203601  |
| Arat Kilo | 24.07566179375741  | 11.725187032418953 |  0.48701411129887123 |

**Interpretation:** Bole has the highest weekend-to-weekday ratio (1.32). Zones below 1.0 are weekday-commute oriented and 'collapse' on weekends relative to their weekday mean — important for Nov weekend forecasts.

### B1.4 Trend

- Early 4-week mean city trips/week: **43110**
- Late 4-week mean city trips/week: **55831**
- Growth early→late: **29.5%**
- Mean daily city trips Jan: **6240** · Oct: **9277**

**Interpretation:** City-wide demand grew over Jan–Oct; a November forecast should allow for level shift / trend (week_index or recent lags), not only seasonal averages.

---

## B2 — Weather

### B2.1 Timezone check

- Temp peak hour after UTC→EAT: **15**
- Rain–demand corr (correct EAT, shift 0): **0.2834**
- Corr if UTC treated as local: **0.1474**
- Corr by hour shift 0–5: `{0: 0.283358119905005, 1: 0.16654744701301674, 2: 0.1465595981471896, 3: 0.10373274821948283, 4: 0.055091207195206764, 5: 0.0007384333061837528}`

**Interpretation:** After UTC→EAT, mean temperature peaks at hour 15 (afternoon local). Rain–demand correlation at shift 0 (correct clock) is 0.2834; treating UTC as local yields 0.1474. Misalignment by 1–5h changes the measured relationship — clocks must match before modeling.

### B2.2 Rain effect by zone type

| zone_type               |   mean_wet_dry_ratio |   median_wet_dry_ratio |   n_cells |
|:------------------------|---------------------:|-----------------------:|----------:|
| residential_late_launch |   1.3313303439453474 |     1.2883905535068325 |        92 |
| transport_hub           |   1.2724668882070755 |     1.2716487279843445 |       465 |
| residential             |   1.2563157085458072 |     1.2626066718386346 |       371 |
| market                  |   1.0747132061743394 |     1.04395674089761   |       186 |

**Interpretation:** Ratios >1 mean rainy hours have higher demand than matched dry hours (same zone, dow, hour). Rain does **not** lift every zone type equally — use zone-type interactions or zone models.

### B2.3 Rain dose-response

| zone_type               |   none |              light |           moderate |              heavy |
|:------------------------|-------:|-------------------:|-------------------:|-------------------:|
| market                  |    1.0 | 1.26605004847513   | 1.3042666338204778 | 1.2915515694773032 |
| residential             |    1.0 | 1.4796118737989958 | 1.7476573320012003 | 2.076568009791718  |
| residential_late_launch |    1.0 | 1.4986011414501326 | 1.792087000784362  | 2.2452690496332184 |
| transport_hub           |    1.0 | 1.7280211674203019 | 2.0224590075654545 | 2.3990651169878423 |

**Interpretation:** Values are demand relative to `rain_class=none` within zone type. If heavy ≈ moderate, the response saturates — prefer rain classes / caps over a raw linear `rain_mm` only.

---

## B3 — Events & calendar

### B3.1 Public holidays

| date                |   dow |   holiday_trips |    baseline_trips |              index |
|:--------------------|------:|----------------:|------------------:|-------------------:|
| 2025-01-07 00:00:00 |     1 |          4805.0 | 6632.0            | 0.7245174909529554 |
| 2025-04-18 00:00:00 |     4 |          6248.0 | 8119.5            | 0.7695055114231172 |
| 2025-06-06 00:00:00 |     4 |          6998.0 | 8796.0            | 0.7955889040472942 |
| 2025-09-04 00:00:00 |     3 |          7606.0 | 9210.2            | 0.8258235434626826 |
| 2025-09-11 00:00:00 |     3 |          7783.0 | 9362.0            | 0.8313394573809015 |
| 2025-03-31 00:00:00 |     0 |          6275.0 | 7394.833333333333 | 0.8485654398341185 |
| 2025-05-28 00:00:00 |     2 |          6914.0 | 8059.666666666667 | 0.8578518549154225 |
| 2025-05-01 00:00:00 |     3 |          6611.0 | 7682.166666666667 | 0.860564510880177  |
| 2025-05-05 00:00:00 |     0 |          6241.0 | 6619.333333333333 | 0.9428441937758083 |
| 2025-04-20 00:00:00 |     6 |          6059.0 | 6369.666666666667 | 0.9512271704432466 |
| 2025-09-27 00:00:00 |     5 |          8451.0 | 8758.166666666666 | 0.9649279719880494 |
| 2025-01-19 00:00:00 |     6 |          5292.0 | 5241.0            | 1.009730967372639  |
| 2025-03-02 00:00:00 |     6 |          5736.0 | 5391.166666666667 | 1.0639626549602745 |

Zone-level holiday index (mean trips on holiday hours / non-holiday):

| zone      |      holiday_index |
|:----------|-------------------:|
| Merkato   | 0.5227642557987189 |
| Arat Kilo | 0.5315466358757003 |
| Piassa    | 0.5602697853677986 |
| Kazanchis | 0.5906314307311152 |
| Lideta    | 0.821404991752192  |
| Megenagna | 0.8512328846226456 |
| Sarbet    | 0.9869614307838493 |
| CMC       | 1.0039274510249663 |
| Gerji     | 1.0097997453204541 |
| Ayat      | 1.0118017360886997 |
| Kolfe     | 1.05650997580336   |
| Bole      | 1.1568253603813081 |

**Interpretation:** Index <1 means holidays reduce demand city-wide or in that zone. Not every holiday behaves the same; local zones can diverge from the city average.

### B3.2 Football event-window study

| window            |         mean_trips |   uplift_vs_baseline |
|:------------------|-------------------:|---------------------:|
| baseline_no_event | 26.595688874052172 |   1.0                |
| 2h_before         | 70.46969696969697  |   2.6496661659495517 |
| during_window     | 45.32380952380952  |   1.7041788140343777 |
| 2h_after          | 53.98571428571429  |   2.0298671164853688 |

**Interpretation:** Compare uplift across before / during / after. The window with the largest uplift should drive the football feature design (±2h already used in A).

### B3.3 Event type ranking

| event_type     |      mean_trips_on |     mean_trips_off |       effect_ratio |   n_on |
|:---------------|-------------------:|-------------------:|-------------------:|-------:|
| concert        | 69.85096153846153  | 28.505476874080152 | 2.4504400276136606 |    208 |
| football_match | 54.726141078838175 | 28.533001049951125 | 1.9179945699729302 |    241 |
| any_event      | 29.85533713802579  | 28.113008225814593 | 1.0619758973574132 |  23655 |
| public_holiday | 23.834875245304175 | 28.823071603696487 | 0.826937377564139  |   3567 |
| road_closure   | 20.48087431693989  | 28.644918777799266 | 0.7149915304634491 |    366 |

**Interpretation:** Ranked by on/off mean-trip ratio. Types near 1.0 have little measurable average effect under this coarse flag definition (still worth checking locally).

### B3.4 Cancelled and unlisted events

- Cancelled events examined: footprint table below; mean on/off ratio ≈ **1.1160173776891396**

| event_id   | event_type     | zone      |            mean_on |           mean_off |              ratio |
|:-----------|:---------------|:----------|-------------------:|-------------------:|-------------------:|
| EVT-0113   | concert        | Bole      | 33.4               | 39.412022630834514 | 0.8474571404987743 |
| EVT-0052   | football_match | Kazanchis | 66.66666666666667  | 31.83413529578262  | 2.094188079784239  |
| EVT-0058   | football_match | Kazanchis | 20.0               | 31.853948485706198 | 0.6278656477696819 |
| EVT-0042   | conference     | Kazanchis | 39.375             | 31.84039087947883  | 1.2366368286445013 |
| EVT-0099   | concert        | Bole      | 32.0               | 39.41301272984441  | 0.8119145881930738 |
| EVT-0020   | football_match | Kazanchis | 34.333333333333336 | 31.847863005943957 | 1.0780419812445659 |

**Unlisted spikes (high city demand, not holiday, low event coverage):**

```json
[
  {
    "date": "2025-09-26",
    "city_trips": 11497.0,
    "top_zone": "Kazanchis",
    "top_zone_trips": 1513.0,
    "hypothesis": "Possible unlisted concert/market peak, payday cluster, or weather-driven surge not tagged as a calendar event."
  },
  {
    "date": "2025-09-12",
    "city_trips": 11005.0,
    "top_zone": "Megenagna",
    "top_zone_trips": 1394.0,
    "hypothesis": "Possible unlisted concert/market peak, payday cluster, or weather-driven surge not tagged as a calendar event."
  },
  {
    "date": "2025-10-28",
    "city_trips": 10812.0,
    "top_zone": "Megenagna",
    "top_zone_trips": 1252.0,
    "hypothesis": "Possible unlisted concert/market peak, payday cluster, or weather-driven surge not tagged as a calendar event."
  }
]
```

**Interpretation:** Cancelled events should not be treated like confirmed ones (we already exclude them in features). Large unlisted spikes show the calendar is incomplete — residual error analysis (D7) should expect event-like misses.

---

## B4 — Operations & data quality

### B4.1 Operational variables vs demand

- corr(trips, active_drivers) = **0.967**
- corr(trips, avg_wait_min) = **0.525**
- corr(trips, avg_fare_birr) = **0.006**

**Interpretation:** active_drivers correlates strongly with trips because dispatch follows demand — it is a consequence, not a cause available at forecast time. avg_wait_min often moves with congestion/shortage (can look negative or weak). avg_fare_birr reflects mix/distance more than volume. None of these three may be used as model inputs for November (Rule 6 / D4).

### B4.2 Gaps and outages

| zone      | first_hour          | last_hour           |   n_missing_hours |   n_gap_periods |   longest_gap_hours |
|:----------|:--------------------|:--------------------|------------------:|----------------:|--------------------:|
| Merkato   | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               108 |              66 |                42.0 |
| Sarbet    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               107 |              66 |                42.0 |
| Kazanchis | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               102 |              61 |                42.0 |
| CMC       | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               100 |              59 |                42.0 |
| Megenagna | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               100 |              59 |                42.0 |
| Kolfe     | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               100 |              59 |                42.0 |
| Bole      | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |               100 |              59 |                42.0 |
| Gerji     | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |                97 |              55 |                42.0 |
| Piassa    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |                97 |              55 |                42.0 |
| Arat Kilo | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |                95 |              54 |                42.0 |
| Lideta    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |                94 |              53 |                42.0 |
| Ayat      | 2025-03-15 00:00:00 | 2025-10-31 23:00:00 |                86 |              45 |                42.0 |

Late-launch zones (first hour after 2025-01-07): `{'Ayat': '2025-03-15'}`

**Treatment:** Ayat starts later than other zones → late launch (do not impute pre-launch as zeros for training labels; model can still forecast Ayat in November). Short scattered gaps → random missing records (dropped or left out of target). Long multi-zone simultaneous holes → treat as platform outage periods and exclude from training metrics.

### B4.3 Pay-period effect

- Payday mean trips: **29.77** vs other **28.32** (ratio **1.051**)
- Keep `is_payday_window`?: **True**

**Interpretation:** Payday-window mean trips 29.77 vs other 28.32 (ratio 1.051). After weekly detrend, difference is 0.595 trips/hour. Effect is large enough to keep `is_payday_window` as a feature.

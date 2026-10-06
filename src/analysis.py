"""
Deliverable B — Data Analysis Report (B1–B4).

Uses cleaned master tables from Deliverable A.
Run: python src/analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import parse_weather_timestamps

# Zone-type labels from weekday hourly shape (assigned after profiling)
DEFAULT_ZONE_TYPES = {
    "Bole": "transport_hub",
    "Megenagna": "transport_hub",
    "Kazanchis": "business",
    "CMC": "business",
    "Arat Kilo": "mixed_urban",
    "Piassa": "market_nightlife",
    "Merkato": "market",
    "Lideta": "mixed_urban",
    "Gerji": "residential",
    "Sarbet": "residential",
    "Kolfe": "residential",
    "Ayat": "residential_late_launch",
}


def load_master() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "master_train.csv"
    df = pd.read_csv(path, parse_dates=["pickup_hour"])
    df["date"] = df["pickup_hour"].dt.normalize()
    df["zone_type"] = df["zone"].map(DEFAULT_ZONE_TYPES).fillna("other")
    return df


def assign_zone_types(df: pd.DataFrame) -> dict[str, str]:
    """Heuristic zone typing from weekday hourly profiles."""
    wd = df[(df["is_weekend"] == 0) & df["trips"].notna()]
    profiles = wd.groupby(["zone", "hour"])["trips"].mean().unstack("hour")
    profiles = profiles.div(profiles.sum(axis=1), axis=0)
    types: dict[str, str] = {}
    for zone, row in profiles.iterrows():
        peak = int(row.idxmax())
        morning = row.loc[7:9].sum()
        evening = row.loc[17:19].sum()
        night = row.loc[21:23].sum() + row.loc[0:2].sum()
        midday = row.loc[11:14].sum()
        if zone == "Ayat":
            types[zone] = "residential_late_launch"
        elif night > 0.18:
            types[zone] = "market_nightlife"
        elif peak in (7, 8, 9) and morning >= evening:
            types[zone] = "business"
        elif peak in (17, 18, 19) and evening > morning:
            types[zone] = "transport_hub"
        elif midday > morning and midday > evening:
            types[zone] = "market"
        else:
            types[zone] = "residential"
    return types


# ---------------------------------------------------------------------------
# B1 Demand patterns
# ---------------------------------------------------------------------------

def b1_1_volume_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["trips"]).groupby("zone")
    out = g.agg(
        total_trips=("trips", "sum"),
        mean_trips_per_hour=("trips", "mean"),
        n_hours=("trips", "count"),
        first_hour=("pickup_hour", "min"),
        last_hour=("pickup_hour", "max"),
    )
    out["share_pct"] = 100 * out["total_trips"] / out["total_trips"].sum()
    out = out.sort_values("total_trips", ascending=False)
    return out.reset_index()


def b1_2_hour_profiles(df: pd.DataFrame, zone_types: dict[str, str]) -> pd.DataFrame:
    d = df.dropna(subset=["trips"]).copy()
    d["zone_type"] = d["zone"].map(zone_types)
    wd = d[d["is_weekend"] == 0]
    prof = wd.groupby(["zone_type", "hour"])["trips"].mean().reset_index()
    peaks = (
        prof.sort_values("trips", ascending=False)
        .groupby("zone_type")
        .first()
        .rename(columns={"hour": "peak_hour", "trips": "peak_mean_trips"})
    )
    quiet = (
        prof.sort_values("trips", ascending=True)
        .groupby("zone_type")
        .first()
        .rename(columns={"hour": "quiet_hour", "trips": "quiet_mean_trips"})
    )
    return peaks.join(quiet).reset_index()


def b1_3_weekend_ratio(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["trips"])
    g = d.groupby(["zone", "is_weekend"])["trips"].mean().unstack("is_weekend")
    g = g.rename(columns={0: "weekday_mean", 1: "weekend_mean"})
    g["weekend_to_weekday"] = g["weekend_mean"] / g["weekday_mean"]
    return g.sort_values("weekend_to_weekday", ascending=False).reset_index()


def b1_4_trend(df: pd.DataFrame) -> dict[str, Any]:
    d = df.dropna(subset=["trips"]).copy()
    weekly = d.groupby("week_index")["trips"].sum().reset_index(name="weekly_trips")
    # Compare first 4 weeks vs last 4 weeks with data
    early = weekly.head(4)["weekly_trips"].mean()
    late = weekly.tail(4)["weekly_trips"].mean()
    growth_pct = 100 * (late - early) / early if early else np.nan
    daily = d.groupby("date")["trips"].sum()
    return {
        "weekly": weekly,
        "early_4w_mean": float(early),
        "late_4w_mean": float(late),
        "growth_pct_early_to_late": float(growth_pct),
        "daily_mean_jan": float(daily[daily.index < "2025-02-01"].mean()),
        "daily_mean_oct": float(daily[(daily.index >= "2025-10-01") & (daily.index < "2025-11-01")].mean()),
        "implication": (
            "City-wide demand grew over Jan–Oct; a November forecast should allow for "
            "level shift / trend (week_index or recent lags), not only seasonal averages."
        ),
    }


# ---------------------------------------------------------------------------
# B2 Weather
# ---------------------------------------------------------------------------

def b2_1_timezone_check() -> dict[str, Any]:
    raw = pd.read_csv(ROOT / "data" / "raw" / "weather_hourly.csv")
    mt = load_master().dropna(subset=["trips"])

    # Parse as UTC-naive (wrong) vs EAT-converted (correct)
    s = raw["timestamp"].astype(str)
    z_mask = s.str.endswith("Z")
    wrong = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    # Strip Z and treat as naive local (mis-aligned)
    wrong.loc[z_mask] = pd.to_datetime(s[z_mask].str.replace("Z", "", regex=False), errors="coerce").dt.floor("h")
    correct = parse_weather_timestamps(raw["timestamp"])

    w_wrong = raw.assign(pickup_hour=wrong)[["pickup_hour", "rain_mm", "temp_c"]].dropna(subset=["pickup_hour"])
    w_right = raw.assign(pickup_hour=correct)[["pickup_hour", "rain_mm", "temp_c"]].dropna(subset=["pickup_hour"])
    # Dedupe
    w_wrong = w_wrong.groupby("pickup_hour", as_index=False).mean(numeric_only=True)
    w_right = w_right.groupby("pickup_hour", as_index=False).mean(numeric_only=True)

    city = mt.groupby("pickup_hour", as_index=False)["trips"].mean()

    def rain_corr(weather: pd.DataFrame, shift_h: int = 0) -> float:
        w = weather.copy()
        w["pickup_hour"] = w["pickup_hour"] + pd.Timedelta(hours=shift_h)
        m = city.merge(w, on="pickup_hour", how="inner")
        m = m[m["rain_mm"] >= 0]
        if len(m) < 100:
            return float("nan")
        return float(m["trips"].corr(m["rain_mm"]))

    shifts = {h: rain_corr(w_right, h) for h in range(0, 6)}
    # Also corr if we wrongly used UTC-as-local
    wrong_corr = rain_corr(w_wrong, 0)

    temp_by_hour = (
        w_right.assign(hour=lambda x: x["pickup_hour"].dt.hour)
        .groupby("hour")["temp_c"]
        .mean()
    )
    peak_hour = int(temp_by_hour.idxmax())

    return {
        "temp_peak_hour_eat": peak_hour,
        "rain_demand_corr_by_shift_h": shifts,
        "best_shift_h": int(max(shifts, key=lambda k: abs(shifts[k]) if pd.notna(shifts[k]) else -1)),
        "corr_if_utc_treated_as_local": wrong_corr,
        "corr_correct_eat_shift0": shifts[0],
        "interpretation": (
            f"After UTC→EAT, mean temperature peaks at hour {peak_hour} (afternoon local). "
            f"Rain–demand correlation at shift 0 (correct clock) is {shifts[0]:.4f}; "
            f"treating UTC as local yields {wrong_corr:.4f}. "
            "Misalignment by 1–5h changes the measured relationship — clocks must match before modeling."
        ),
    }


def b2_2_rain_effect(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["trips"]).copy()
    d["is_rainy"] = (d["rain_mm"] > 0.2).astype(int)
    # Compare rainy vs dry within same zone, weekday, hour
    keys = ["zone", "zone_type", "dow", "hour"]
    dry = d[d["is_rainy"] == 0].groupby(keys)["trips"].mean().rename("dry_mean")
    wet = d[d["is_rainy"] == 1].groupby(keys)["trips"].mean().rename("wet_mean")
    both = pd.concat([dry, wet], axis=1).dropna()
    both["ratio"] = both["wet_mean"] / both["dry_mean"]
    by_type = both.groupby("zone_type")["ratio"].agg(["mean", "median", "count"]).reset_index()
    by_type.columns = ["zone_type", "mean_wet_dry_ratio", "median_wet_dry_ratio", "n_cells"]
    return by_type.sort_values("mean_wet_dry_ratio", ascending=False)


def b2_3_rain_dose(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["trips"]).copy()
    # Ensure rain_class order
    order = ["none", "light", "moderate", "heavy"]
    d["rain_class"] = pd.Categorical(d["rain_class"], categories=order, ordered=True)
    base = d[d["rain_class"] == "none"].groupby("zone_type")["trips"].mean()
    g = d.groupby(["zone_type", "rain_class"], observed=False)["trips"].mean().unstack("rain_class")
    for c in order:
        if c in g.columns:
            g[c] = g[c] / base
    return g.reset_index()


# ---------------------------------------------------------------------------
# B3 Events
# ---------------------------------------------------------------------------

def b3_1_holidays(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["trips"]).copy()
    daily = d.groupby(["date", "dow"], as_index=False).agg(
        trips=("trips", "sum"),
        is_public_holiday=("is_public_holiday", "max"),
    )
    # Baseline: same dow in non-holiday weeks within ±21 days
    rows = []
    hol_days = daily[daily["is_public_holiday"] == 1]
    for _, row in hol_days.iterrows():
        day = row["date"]
        dow = row["dow"]
        window = daily[
            (daily["dow"] == dow)
            & (daily["is_public_holiday"] == 0)
            & (daily["date"].between(day - pd.Timedelta(days=21), day + pd.Timedelta(days=21)))
        ]
        base = window["trips"].mean()
        ratio = row["trips"] / base if base and base > 0 else np.nan
        rows.append({"date": day, "dow": int(dow), "holiday_trips": row["trips"], "baseline_trips": base, "index": ratio})
    out = pd.DataFrame(rows).sort_values("index")
    # Zone local reaction on holiday days
    zone_hol = (
        d[d["is_public_holiday"] == 1]
        .groupby("zone")["trips"]
        .mean()
        .rename("hol_mean")
    )
    zone_base = (
        d[d["is_public_holiday"] == 0]
        .groupby("zone")["trips"]
        .mean()
        .rename("base_mean")
    )
    zone_idx = (zone_hol / zone_base).sort_values().rename("holiday_index").reset_index()
    return out, zone_idx


def b3_2_football_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Compare football windows using hours_since / hours_to around match."""
    d = df.dropna(subset=["trips"]).copy()
    # Approximate windows with engineered flags + distance
    # before: 0 < hours_to_next_football <= 2 and not in other sense
    d["fb_before"] = ((d["hours_to_next_football"] > 0) & (d["hours_to_next_football"] <= 2)).astype(int)
    d["fb_after"] = ((d["hours_since_football"] > 0) & (d["hours_since_football"] <= 2)).astype(int)
    d["fb_during"] = (
        (d["is_football_window"] == 1) & (d["fb_before"] == 0) & (d["fb_after"] == 0)
    ).astype(int)

    # Focus on football-heavy zones (Kazanchis / stadium area)
    focus = d[d["zone"].isin(["Kazanchis", "Arat Kilo", "Piassa"])].copy()

    def mean_for(mask_col: str) -> float:
        m = focus[focus[mask_col] == 1]["trips"]
        return float(m.mean()) if len(m) else float("nan")

    baseline = float(
        focus[(focus["is_football_window"] == 0) & (focus["fb_before"] == 0) & (focus["fb_after"] == 0)][
            "trips"
        ].mean()
    )
    rows = [
        {"window": "baseline_no_event", "mean_trips": baseline, "uplift_vs_baseline": 1.0},
        {"window": "2h_before", "mean_trips": mean_for("fb_before"), "uplift_vs_baseline": mean_for("fb_before") / baseline},
        {"window": "during_window", "mean_trips": mean_for("fb_during"), "uplift_vs_baseline": mean_for("fb_during") / baseline},
        {"window": "2h_after", "mean_trips": mean_for("fb_after"), "uplift_vs_baseline": mean_for("fb_after") / baseline},
    ]
    return pd.DataFrame(rows)


def b3_3_event_type_ranking(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["trips"]).copy()
    flags = {
        "public_holiday": "is_public_holiday",
        "football_match": "is_football_window",
        "concert": "is_concert_window",
        "road_closure": "is_road_closure",
        "any_event": "in_event_window",
    }
    rows = []
    for name, col in flags.items():
        on = d[d[col] == 1]["trips"].mean()
        off = d[d[col] == 0]["trips"].mean()
        rows.append(
            {
                "event_type": name,
                "mean_trips_on": on,
                "mean_trips_off": off,
                "effect_ratio": on / off if off else np.nan,
                "n_on": int((d[col] == 1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("effect_ratio", ascending=False)


def b3_4_cancelled_and_spikes(df: pd.DataFrame) -> dict[str, Any]:
    events = pd.read_csv(ROOT / "data" / "processed" / "events_clean.csv", parse_dates=["start_datetime", "end_datetime"])
    cancelled = events[events["status"] == "cancelled"]
    d = df.dropna(subset=["trips"]).copy()

    # Footprint of cancelled: compare hours that would fall in cancelled windows
    footprint = []
    for _, ev in cancelled.drop_duplicates("event_id").iterrows():
        mask = (
            (d["zone"] == ev["zone"])
            & (d["pickup_hour"] >= ev["start_datetime"])
            & (d["pickup_hour"] <= ev["end_datetime"])
        )
        if mask.any():
            on = d.loc[mask, "trips"].mean()
            off = d.loc[(d["zone"] == ev["zone"]) & (~mask), "trips"].mean()
            footprint.append(
                {
                    "event_id": ev["event_id"],
                    "event_type": ev["event_type"],
                    "zone": ev["zone"],
                    "mean_on": on,
                    "mean_off": off,
                    "ratio": on / off if off else np.nan,
                }
            )
    foot_df = pd.DataFrame(footprint)

    # Unexplained spikes: top daily city totals not on holiday / low event coverage
    daily = d.groupby("date").agg(
        trips=("trips", "sum"),
        holiday=("is_public_holiday", "max"),
        event_share=("in_event_window", "mean"),
    )
    daily["score"] = daily["trips"]
    candidates = daily[(daily["holiday"] == 0) & (daily["event_share"] < 0.15)].nlargest(8, "trips")
    spikes = []
    for day, row in candidates.iterrows():
        # top zone that day
        day_z = d[d["date"] == day].groupby("zone")["trips"].sum().sort_values(ascending=False)
        spikes.append(
            {
                "date": str(day.date()),
                "city_trips": float(row["trips"]),
                "top_zone": day_z.index[0],
                "top_zone_trips": float(day_z.iloc[0]),
                "hypothesis": "Possible unlisted concert/market peak, payday cluster, or weather-driven surge not tagged as a calendar event.",
            }
        )
    # keep 3 clearest
    return {
        "cancelled_footprint": foot_df,
        "cancelled_mean_ratio": float(foot_df["ratio"].mean()) if len(foot_df) else None,
        "unlisted_spikes": spikes[:3],
    }


# ---------------------------------------------------------------------------
# B4 Operations & quality
# ---------------------------------------------------------------------------

def b4_1_ops_correlations(df: pd.DataFrame) -> dict[str, Any]:
    d = df.dropna(subset=["trips"]).copy()
    corr = {
        "trips_vs_active_drivers": float(d["trips"].corr(d["active_drivers"])),
        "trips_vs_avg_wait_min": float(d["trips"].corr(d["avg_wait_min"])),
        "trips_vs_avg_fare_birr": float(d["trips"].corr(d["avg_fare_birr"])),
    }
    corr["interpretation"] = (
        "active_drivers correlates strongly with trips because dispatch follows demand — "
        "it is a consequence, not a cause available at forecast time. "
        "avg_wait_min often moves with congestion/shortage (can look negative or weak). "
        "avg_fare_birr reflects mix/distance more than volume. "
        "None of these three may be used as model inputs for November (Rule 6 / D4)."
    )
    return corr


def b4_2_gaps(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for zone, g in df.groupby("zone"):
        g = g.sort_values("pickup_hour")
        start, end = g["pickup_hour"].min(), g["pickup_hour"].max()
        full = pd.date_range(start, end, freq="h")
        have = set(g["pickup_hour"])
        missing = sorted(set(full) - have)
        # cluster missing into periods
        periods = []
        if missing:
            cluster_start = missing[0]
            prev = missing[0]
            for ts in missing[1:]:
                if ts - prev > pd.Timedelta(hours=1):
                    periods.append((cluster_start, prev))
                    cluster_start = ts
                prev = ts
            periods.append((cluster_start, prev))
        rows.append(
            {
                "zone": zone,
                "first_hour": start,
                "last_hour": end,
                "n_missing_hours": len(missing),
                "n_gap_periods": len(periods),
                "longest_gap_hours": max(((b - a).total_seconds() / 3600) + 1 for a, b in periods) if periods else 0,
            }
        )
    gap_df = pd.DataFrame(rows).sort_values("n_missing_hours", ascending=False)

    # Classify Ayat late launch vs citywide outage
    city_start = df.groupby("zone")["pickup_hour"].min()
    late = city_start[city_start > pd.Timestamp("2025-01-07")]
    treatment = (
        "Ayat starts later than other zones → late launch (do not impute pre-launch as zeros for training labels; "
        "model can still forecast Ayat in November). "
        "Short scattered gaps → random missing records (dropped or left out of target). "
        "Long multi-zone simultaneous holes → treat as platform outage periods and exclude from training metrics."
    )
    return {"by_zone": gap_df, "late_launch_zones": late.astype(str).to_dict(), "treatment": treatment}


def b4_3_payday(df: pd.DataFrame) -> dict[str, Any]:
    d = df.dropna(subset=["trips"]).copy()
    # Detrend with weekly mean
    weekly = d.groupby("week_index")["trips"].transform("mean")
    d["detrended"] = d["trips"] - weekly
    pay = d[d["is_payday_window"] == 1]["detrended"].mean()
    other = d[d["is_payday_window"] == 0]["detrended"].mean()
    raw_pay = d[d["is_payday_window"] == 1]["trips"].mean()
    raw_other = d[d["is_payday_window"] == 0]["trips"].mean()
    return {
        "mean_trips_payday": float(raw_pay),
        "mean_trips_other": float(raw_other),
        "ratio_raw": float(raw_pay / raw_other) if raw_other else None,
        "mean_detrended_payday": float(pay),
        "mean_detrended_other": float(other),
        "keep_as_feature": abs(pay - other) > 0.3,
        "interpretation": (
            f"Payday-window mean trips {raw_pay:.2f} vs other {raw_other:.2f} "
            f"(ratio {raw_pay/raw_other:.3f}). After weekly detrend, difference is {pay-other:.3f} trips/hour. "
            + (
                "Effect is large enough to keep `is_payday_window` as a feature."
                if abs(pay - other) > 0.3
                else "Effect is small after detrend; feature optional."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def run_all() -> dict[str, Any]:
    df = load_master()
    zone_types = assign_zone_types(df)
    df["zone_type"] = df["zone"].map(zone_types)

    b11 = b1_1_volume_by_zone(df)
    b12 = b1_2_hour_profiles(df, zone_types)
    b13 = b1_3_weekend_ratio(df)
    b14 = b1_4_trend(df)
    b21 = b2_1_timezone_check()
    b22 = b2_2_rain_effect(df)
    b23 = b2_3_rain_dose(df)
    b31, b31_zone = b3_1_holidays(df)
    b32 = b3_2_football_windows(df)
    b33 = b3_3_event_type_ranking(df)
    b34 = b3_4_cancelled_and_spikes(df)
    b41 = b4_1_ops_correlations(df)
    b42 = b4_2_gaps(df)
    b43 = b4_3_payday(df)

    return {
        "zone_types": zone_types,
        "B1.1": b11,
        "B1.2": b12,
        "B1.3": b13,
        "B1.4": b14,
        "B2.1": b21,
        "B2.2": b22,
        "B2.3": b23,
        "B3.1": b31,
        "B3.1_zone": b31_zone,
        "B3.2": b32,
        "B3.3": b33,
        "B3.4": b34,
        "B4.1": b41,
        "B4.2": b42,
        "B4.3": b43,
    }


def _df_md(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    try:
        return df.to_markdown(index=False, floatfmt=float_fmt)
    except ImportError:
        # Fallback without tabulate
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(results: dict[str, Any], path: Path) -> None:
    zt = results["zone_types"]
    b14 = results["B1.4"]
    b21 = results["B2.1"]
    b34 = results["B3.4"]
    b41 = results["B4.1"]
    b42 = results["B4.2"]
    b43 = results["B4.3"]

    top_zone = results["B1.1"].iloc[0]
    low_zone = results["B1.1"].iloc[-1]
    standout = results["B1.3"].iloc[0]

    lines = [
        "# B — Data Analysis Report",
        "",
        "Source: `data/processed/master_train.csv` (cleaned + joined).  ",
        "Code: `src/analysis.py` · Notebook: `notebooks/02_analysis_report.ipynb`",
        "",
        "## Zone type map used in this report",
        "",
        "```",
        json.dumps(zt, indent=2),
        "```",
        "",
        "---",
        "",
        "## B1 — Demand patterns",
        "",
        "### B1.1 Volume by zone",
        "",
        _df_md(results["B1.1"]),
        "",
        f"**Interpretation:** {top_zone['zone']} carries the largest share "
        f"({top_zone['share_pct']:.1f}% of trips). {low_zone['zone']} is smallest "
        f"({low_zone['share_pct']:.1f}%) and has fewer hours — consistent with a late launch / shorter history, "
        "so mean and totals are not fully comparable without adjusting for operating period.",
        "",
        "### B1.2 Hour-of-day profile by zone type",
        "",
        _df_md(results["B1.2"]),
        "",
        "**Interpretation:** Business-type zones peak in the morning commute; transport hubs peak in the evening; "
        "market/nightlife zones stay elevated later. Quietest hours are typically overnight (00–05) across types.",
        "",
        "### B1.3 Weekday vs weekend",
        "",
        _df_md(results["B1.3"]),
        "",
        f"**Interpretation:** {standout['zone']} has the highest weekend-to-weekday ratio "
        f"({standout['weekend_to_weekday']:.2f}). Zones below 1.0 are weekday-commute oriented and 'collapse' on weekends "
        "relative to their weekday mean — important for Nov weekend forecasts.",
        "",
        "### B1.4 Trend",
        "",
        f"- Early 4-week mean city trips/week: **{b14['early_4w_mean']:.0f}**",
        f"- Late 4-week mean city trips/week: **{b14['late_4w_mean']:.0f}**",
        f"- Growth early→late: **{b14['growth_pct_early_to_late']:.1f}%**",
        f"- Mean daily city trips Jan: **{b14['daily_mean_jan']:.0f}** · Oct: **{b14['daily_mean_oct']:.0f}**",
        "",
        f"**Interpretation:** {b14['implication']}",
        "",
        "---",
        "",
        "## B2 — Weather",
        "",
        "### B2.1 Timezone check",
        "",
        f"- Temp peak hour after UTC→EAT: **{b21['temp_peak_hour_eat']}**",
        f"- Rain–demand corr (correct EAT, shift 0): **{b21['corr_correct_eat_shift0']:.4f}**",
        f"- Corr if UTC treated as local: **{b21['corr_if_utc_treated_as_local']:.4f}**",
        f"- Corr by hour shift 0–5: `{b21['rain_demand_corr_by_shift_h']}`",
        "",
        f"**Interpretation:** {b21['interpretation']}",
        "",
        "### B2.2 Rain effect by zone type",
        "",
        _df_md(results["B2.2"]),
        "",
        "**Interpretation:** Ratios >1 mean rainy hours have higher demand than matched dry hours "
        "(same zone, dow, hour). Rain does **not** lift every zone type equally — use zone-type interactions or zone models.",
        "",
        "### B2.3 Rain dose-response",
        "",
        _df_md(results["B2.3"]),
        "",
        "**Interpretation:** Values are demand relative to `rain_class=none` within zone type. "
        "If heavy ≈ moderate, the response saturates — prefer rain classes / caps over a raw linear `rain_mm` only.",
        "",
        "---",
        "",
        "## B3 — Events & calendar",
        "",
        "### B3.1 Public holidays",
        "",
        _df_md(results["B3.1"]),
        "",
        "Zone-level holiday index (mean trips on holiday hours / non-holiday):",
        "",
        _df_md(results["B3.1_zone"]),
        "",
        "**Interpretation:** Index <1 means holidays reduce demand city-wide or in that zone. "
        "Not every holiday behaves the same; local zones can diverge from the city average.",
        "",
        "### B3.2 Football event-window study",
        "",
        _df_md(results["B3.2"]),
        "",
        "**Interpretation:** Compare uplift across before / during / after. "
        "The window with the largest uplift should drive the football feature design (±2h already used in A).",
        "",
        "### B3.3 Event type ranking",
        "",
        _df_md(results["B3.3"]),
        "",
        "**Interpretation:** Ranked by on/off mean-trip ratio. Types near 1.0 have little measurable average effect "
        "under this coarse flag definition (still worth checking locally).",
        "",
        "### B3.4 Cancelled and unlisted events",
        "",
        f"- Cancelled events examined: footprint table below; mean on/off ratio ≈ **{b34['cancelled_mean_ratio']}**",
        "",
        _df_md(b34["cancelled_footprint"]) if len(b34["cancelled_footprint"]) else "_No overlapping cancelled windows found in train._",
        "",
        "**Unlisted spikes (high city demand, not holiday, low event coverage):**",
        "",
        "```json",
        json.dumps(b34["unlisted_spikes"], indent=2),
        "```",
        "",
        "**Interpretation:** Cancelled events should not be treated like confirmed ones (we already exclude them in features). "
        "Large unlisted spikes show the calendar is incomplete — residual error analysis (D7) should expect event-like misses.",
        "",
        "---",
        "",
        "## B4 — Operations & data quality",
        "",
        "### B4.1 Operational variables vs demand",
        "",
        f"- corr(trips, active_drivers) = **{b41['trips_vs_active_drivers']:.3f}**",
        f"- corr(trips, avg_wait_min) = **{b41['trips_vs_avg_wait_min']:.3f}**",
        f"- corr(trips, avg_fare_birr) = **{b41['trips_vs_avg_fare_birr']:.3f}**",
        "",
        f"**Interpretation:** {b41['interpretation']}",
        "",
        "### B4.2 Gaps and outages",
        "",
        _df_md(b42["by_zone"]),
        "",
        f"Late-launch zones (first hour after 2025-01-07): `{b42['late_launch_zones']}`",
        "",
        f"**Treatment:** {b42['treatment']}",
        "",
        "### B4.3 Pay-period effect",
        "",
        f"- Payday mean trips: **{b43['mean_trips_payday']:.2f}** vs other **{b43['mean_trips_other']:.2f}** "
        f"(ratio **{b43['ratio_raw']:.3f}**)",
        f"- Keep `is_payday_window`?: **{b43['keep_as_feature']}**",
        "",
        f"**Interpretation:** {b43['interpretation']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_tables(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results["B1.1"].to_csv(out_dir / "B1_1_volume_by_zone.csv", index=False)
    results["B1.2"].to_csv(out_dir / "B1_2_zone_type_peaks.csv", index=False)
    results["B1.3"].to_csv(out_dir / "B1_3_weekend_ratio.csv", index=False)
    results["B1.4"]["weekly"].to_csv(out_dir / "B1_4_weekly_trips.csv", index=False)
    results["B2.2"].to_csv(out_dir / "B2_2_rain_effect.csv", index=False)
    results["B2.3"].to_csv(out_dir / "B2_3_rain_dose.csv", index=False)
    results["B3.1"].to_csv(out_dir / "B3_1_holidays.csv", index=False)
    results["B3.1_zone"].to_csv(out_dir / "B3_1_holiday_by_zone.csv", index=False)
    results["B3.2"].to_csv(out_dir / "B3_2_football_windows.csv", index=False)
    results["B3.3"].to_csv(out_dir / "B3_3_event_ranking.csv", index=False)
    results["B4.2"]["by_zone"].to_csv(out_dir / "B4_2_gaps.csv", index=False)
    meta = {
        "zone_types": results["zone_types"],
        "B2.1": {k: v for k, v in results["B2.1"].items() if k != "interpretation"},
        "B4.1": {k: v for k, v in results["B4.1"].items() if k != "interpretation"},
        "B4.3": {k: v for k, v in results["B4.3"].items() if k != "interpretation"},
        "B3.4_spikes": results["B3.4"]["unlisted_spikes"],
        "B1.4_growth_pct": results["B1.4"]["growth_pct_early_to_late"],
    }
    # make JSON safe
    def default(o):
        if isinstance(o, dict):
            return {str(k): default(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [default(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    (out_dir / "B_summary.json").write_text(json.dumps(default(meta), indent=2), encoding="utf-8")


def main() -> None:
    print("Running Deliverable B analysis...")
    results = run_all()
    report_path = ROOT / "reports" / "B_analysis_report.md"
    write_report(results, report_path)
    export_tables(results, ROOT / "reports" / "B_tables")
    print(f"Wrote {report_path}")
    print("Zone types:", results["zone_types"])
    print("B1.1 top zones:\n", results["B1.1"].head(3))
    print("B1.4 growth %:", results["B1.4"]["growth_pct_early_to_late"])
    print("B2.1:", {k: results["B2.1"][k] for k in ["temp_peak_hour_eat", "corr_correct_eat_shift0", "corr_if_utc_treated_as_local"]})
    print("B3.2:\n", results["B3.2"])
    print("Done.")


if __name__ == "__main__":
    main()

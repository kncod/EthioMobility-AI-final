"""
Cleaning, time alignment, joins, and feature engineering for the
Qiyas Addis Ride Demand Forecasting challenge.

All clocks are standardized to Africa/Addis_Ababa (EAT, UTC+3, no DST).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TZ = "Africa/Addis_Ababa"

CANONICAL_ZONES = [
    "Arat Kilo",
    "Ayat",
    "Bole",
    "CMC",
    "Gerji",
    "Kazanchis",
    "Kolfe",
    "Lideta",
    "Megenagna",
    "Merkato",
    "Piassa",
    "Sarbet",
]

# After lower/strip/collapse spaces
ZONE_ALIASES = {
    "arat kilo": "Arat Kilo",
    "ayat": "Ayat",
    "bole": "Bole",
    "bole rd": "Bole",
    "cmc": "CMC",
    "c.m.c": "CMC",
    "gerji": "Gerji",
    "kazanchis": "Kazanchis",
    "kazanches": "Kazanchis",
    "kolfe": "Kolfe",
    "kolfe keranio": "Kolfe",
    "lideta": "Lideta",
    "megenagna": "Megenagna",
    "megenaga": "Megenagna",
    "merkato": "Merkato",
    "mercato": "Merkato",
    "piassa": "Piassa",
    "piazza": "Piassa",
    "sarbet": "Sarbet",
}

EVENT_TYPE_ALIASES = {
    "football_match": "football_match",
    "football match": "football_match",
    "conference": "conference",
    "concert": "concert",
    "public_holiday": "public_holiday",
    "public holiday": "public_holiday",
    "road_closure": "road_closure",
    "road closure": "road_closure",
    "sports_run": "sports_run",
    "sports run": "sports_run",
    "school_break": "school_break",
    "school break": "school_break",
    "exhibition": "exhibition",
}

# Event windows: hours before start / after end by type
EVENT_WINDOWS = {
    "football_match": (2, 2),
    "concert": (2, 2),
    "sports_run": (1, 1),
    "conference": (0, 0),
    "exhibition": (0, 0),
    "road_closure": (0, 0),
    "public_holiday": (0, 0),
    "school_break": (0, 0),
}

CITYWIDE_TOKENS = {
    "citywide",
    "city-wide",
    "city wide",
    "all",
    "all zones",
    "all zone",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def processed_dir() -> Path:
    return project_root() / "data" / "processed"


# ---------------------------------------------------------------------------
# Cleaning log helper
# ---------------------------------------------------------------------------

class CleaningLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        file: str,
        columns: str,
        issue_type: str,
        n_affected: int,
        n_total: int,
        fix: str,
        why: str,
    ) -> None:
        pct = round(100.0 * n_affected / n_total, 2) if n_total else 0.0
        self.rows.append(
            {
                "file": file,
                "columns": columns,
                "issue_type": issue_type,
                "n_affected": n_affected,
                "pct_affected": pct,
                "fix": fix,
                "why": why,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------------------
# Zone & label normalization
# ---------------------------------------------------------------------------

def normalize_zone_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    # Drop parenthetical notes: "Kazanchis (Kirkos)" -> "Kazanchis"
    text = re.sub(r"\s*\(.*?\)\s*", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    key = text.lower()
    return ZONE_ALIASES.get(key)


def normalize_zones_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_zone_label)


def expand_event_zones(raw_zone: Any) -> list[str]:
    """Map an events.zone cell to one or more canonical zones."""
    if pd.isna(raw_zone):
        return []
    text = str(raw_zone).strip()
    text = re.sub(r"\s+", " ", text)
    lower = text.lower()

    if lower in CITYWIDE_TOKENS:
        return list(CANONICAL_ZONES)

    # Split multi-zone cells: "Arat Kilo & SARBET", "Ayat & Lideta"
    parts = re.split(r"\s*&\s*|\s*,\s*|\s*/\s*", text)
    zones: list[str] = []
    for part in parts:
        z = normalize_zone_label(part)
        if z and z not in zones:
            zones.append(z)
    return zones


def normalize_event_type(value: Any) -> str | None:
    if pd.isna(value):
        return None
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    return EVENT_TYPE_ALIASES.get(key, key.replace(" ", "_"))


def normalize_status(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    key = str(value).strip().lower()
    if key.startswith("cancel"):
        return "cancelled"
    if key.startswith("confirm"):
        return "confirmed"
    return key


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_mixed_timestamps(series: pd.Series, assume_tz: str | None = TZ) -> pd.Series:
    """
    Parse mixed timestamp formats seen in the raw exports.

    Formats observed:
      - YYYY-MM-DD HH:MM
      - DD/MM/YYYY HH:MM
      - ISO with +03:00
      - ISO with Z (UTC)
      - Mon DD, YYYY h:mm AM/PM
    """
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    # ISO with timezone offset or Z
    iso_mask = s.str.contains(r"T", regex=True, na=False)
    if iso_mask.any():
        parsed = pd.to_datetime(s[iso_mask], utc=True, errors="coerce")
        # Convert to naive EAT wall time for consistent joins
        parsed = parsed.dt.tz_convert(TZ).dt.tz_localize(None)
        out.loc[iso_mask] = parsed

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")

    # Day-first slash dates: DD/MM/YYYY HH:MM
    slash_mask = remaining & s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}")
    if slash_mask.any():
        parsed = pd.to_datetime(s[slash_mask], dayfirst=True, errors="coerce")
        out.loc[slash_mask] = parsed

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")

    # Month-name formats: "Feb 06, 2025 08:00 PM"
    mon_mask = remaining & s.str.match(r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}")
    if mon_mask.any():
        parsed = pd.to_datetime(s[mon_mask], errors="coerce")
        out.loc[mon_mask] = parsed

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")

    # ISO-like without T: YYYY-MM-DD HH:MM
    ymd_mask = remaining & s.str.match(r"^\d{4}-\d{2}-\d{2}")
    if ymd_mask.any():
        parsed = pd.to_datetime(s[ymd_mask], errors="coerce")
        out.loc[ymd_mask] = parsed

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")
    if remaining.any():
        parsed = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)
        out.loc[remaining] = parsed

    # Floor to hour for join keys
    out = pd.to_datetime(out).dt.floor("h")
    return out


def parse_weather_timestamps(series: pd.Series) -> pd.Series:
    """
    Weather timestamps are predominantly UTC (trailing Z). Convert to EAT.
    Non-Z rows (DD/MM/YYYY) are treated as already-local EAT after dayfirst parse,
    unless they also look like UTC ISO.
    """
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    z_mask = s.str.endswith("Z") | s.str.contains(r"\+00:00", regex=True, na=False)
    if z_mask.any():
        parsed = pd.to_datetime(s[z_mask], utc=True, errors="coerce")
        out.loc[z_mask] = parsed.dt.tz_convert(TZ).dt.tz_localize(None)

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")
    # Other ISO with offset
    iso_mask = remaining & s.str.contains("T", na=False)
    if iso_mask.any():
        parsed = pd.to_datetime(s[iso_mask], utc=True, errors="coerce")
        out.loc[iso_mask] = parsed.dt.tz_convert(TZ).dt.tz_localize(None)

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")
    slash_mask = remaining & s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}")
    if slash_mask.any():
        # Ambiguous clock: treated as EAT wall time (documented in notebook A2)
        parsed = pd.to_datetime(s[slash_mask], dayfirst=True, errors="coerce")
        out.loc[slash_mask] = parsed

    remaining = out.isna() & s.notna() & (s != "") & (s != "nan")
    if remaining.any():
        parsed = pd.to_datetime(s[remaining], errors="coerce", utc=True)
        if getattr(parsed.dt, "tz", None) is not None or str(parsed.dtype).startswith("datetime64[ns, "):
            out.loc[remaining] = parsed.dt.tz_convert(TZ).dt.tz_localize(None)
        else:
            # If parsed naive from UTC-looking strings without Z, still shift +3?
            # Safer: assume UTC for leftover ISO-like, EAT otherwise.
            out.loc[remaining] = parsed

    return pd.to_datetime(out).dt.floor("h")


# ---------------------------------------------------------------------------
# Table cleaners
# ---------------------------------------------------------------------------

def clean_trips(df: pd.DataFrame, log: CleaningLog, is_train: bool = True) -> pd.DataFrame:
    file = "ride_demand_train.csv" if is_train else "ride_demand_test.csv"
    out = df.copy()
    n = len(out)

    # Zone cleanup
    before_zones = out["zone"].nunique()
    out["zone_raw"] = out["zone"]
    out["zone"] = normalize_zones_series(out["zone"])
    unmapped = out["zone"].isna().sum()
    log.add(
        file,
        "zone",
        "inconsistent_spelling",
        before_zones,
        n,
        f"Mapped {before_zones} raw labels -> {out['zone'].nunique()} canonical zones; unmapped={unmapped}",
        "Joins and models need one label per zone",
    )
    if unmapped:
        # Keep rows but flag — should be rare after alias map
        out = out.dropna(subset=["zone"])

    # Timestamps
    raw_ts = out["pickup_hour"].copy()
    out["pickup_hour"] = parse_mixed_timestamps(raw_ts)
    bad_ts = out["pickup_hour"].isna().sum()
    log.add(
        file,
        "pickup_hour",
        "mixed_datetime_formats",
        n,
        n,
        "Parsed ISO+03, YYYY-MM-DD HH:MM, DD/MM/YYYY HH:MM; floored to hour; clock=EAT",
        "Mixed formats cause silent day/month swaps if parsed naively",
    )
    if bad_ts:
        log.add(file, "pickup_hour", "unparseable_timestamp", bad_ts, n, "Dropped rows with NaT pickup_hour", "Invalid join keys")
        out = out.dropna(subset=["pickup_hour"])

    if is_train:
        # Negative trips
        neg = (out["trips"] < 0).sum()
        if neg:
            out.loc[out["trips"] < 0, "trips"] = np.nan
            log.add(file, "trips", "impossible_negative", int(neg), n, "Set trips<0 to NaN", "Trip counts cannot be negative")

        null_trips = int(out["trips"].isna().sum())
        log.add(
            file,
            "trips",
            "missing_target",
            null_trips,
            len(out),
            "Kept as NaN for analysis; dropped only when fitting models",
            "Missing targets should not be silently filled before evaluation design",
        )

        # Extreme spikes — cap at 99.9th percentile of non-null for modeling column
        if out["trips"].notna().any():
            cap = float(out["trips"].quantile(0.999))
            spikes = int((out["trips"] > cap).sum())
            out["trips_raw"] = out["trips"]
            out.loc[out["trips"] > cap, "trips"] = cap
            log.add(
                file,
                "trips",
                "extreme_outlier",
                spikes,
                len(out),
                f"Capped trips at 99.9th pct ({cap:.1f}); kept trips_raw",
                "A few extreme spikes distort RMSE and tree splits",
            )

        # Negative wait
        neg_wait = int((out["avg_wait_min"] < 0).sum())
        if neg_wait:
            out.loc[out["avg_wait_min"] < 0, "avg_wait_min"] = np.nan
            log.add(file, "avg_wait_min", "impossible_negative", neg_wait, len(out), "Set avg_wait_min<0 to NaN", "Wait time cannot be negative")

        null_fare = int(out["avg_fare_birr"].isna().sum())
        log.add(
            file,
            "avg_fare_birr",
            "missing_values",
            null_fare,
            len(out),
            "Left as NaN (analysis only; not a forecast-time feature)",
            "Operational column — useful for EDA/demo fares, not model input",
        )

    # Deduplicate zone-hour if any
    id_col = "record_id" if "record_id" in out.columns else "row_id"
    before = len(out)
    out = out.sort_values(["zone", "pickup_hour", id_col])
    dup_mask = out.duplicated(subset=["zone", "pickup_hour"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        out = out.loc[~dup_mask].copy()
        log.add(
            file,
            "zone,pickup_hour",
            "duplicate_zone_hour",
            n_dup,
            before,
            "Kept first row per zone-hour",
            "Many-to-one weather join requires unique left keys",
        )

    out["pickup_hour"] = pd.to_datetime(out["pickup_hour"])
    return out.reset_index(drop=True)


def clean_weather(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    file = "weather_hourly.csv"
    out = df.copy()
    n = len(out)

    out["timestamp_raw"] = out["timestamp"]
    out["timestamp"] = parse_weather_timestamps(out["timestamp"])
    log.add(
        file,
        "timestamp",
        "timezone_and_formats",
        n,
        n,
        "UTC (Z) converted to Africa/Addis_Ababa; DD/MM rows parsed dayfirst as EAT; floored to hour",
        "Trip table is EAT; joining UTC hours without shift misaligns rain vs demand by ~3h",
    )

    # Sentinel rain
    sentinel = int((out["rain_mm"] <= -999).sum())
    out.loc[out["rain_mm"] <= -999, "rain_mm"] = np.nan
    log.add(
        file,
        "rain_mm",
        "sentinel_missing_code",
        sentinel,
        n,
        "Replaced rain_mm<=-999 with NaN",
        "-9999 is a missing-reading code, not rainfall",
    )

    # Impossible temperature (Addis rarely >40C; >50 looks like bad unit/error)
    hot = int((out["temp_c"] > 45).sum())
    out.loc[out["temp_c"] > 45, "temp_c"] = np.nan
    log.add(
        file,
        "temp_c",
        "impossible_value",
        hot,
        n,
        "Set temp_c>45 to NaN",
        "Values like 76C are not plausible for Addis Ababa air temperature",
    )

    null_temp = int(out["temp_c"].isna().sum())
    null_hum = int(out["humidity_pct"].isna().sum())
    log.add(file, "temp_c", "missing_values", null_temp, n, "Imputed later with hourly seasonal median from train period", "Keep series complete for joins")
    log.add(file, "humidity_pct", "missing_values", null_hum, n, "Imputed later with hourly seasonal median from train period", "Keep series complete for joins")

    # Deduplicate hours — mean of numeric conflicts
    before = len(out)
    dup = int(out["timestamp"].duplicated().sum())
    num_cols = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]
    data_type = (
        out.sort_values("timestamp")
        .groupby("timestamp", as_index=False)
        .agg(
            {
                **{c: "mean" for c in num_cols},
                "data_type": "first",
            }
        )
    )
    out = data_type
    log.add(
        file,
        "timestamp",
        "duplicate_hours",
        dup,
        before,
        f"Aggregated to unique hours by mean; rows {before}->{len(out)}",
        "Duplicate keys would multiply trip rows on join",
    )

    # Reindex to a complete hourly spine and interpolate gaps
    out = out.sort_values("timestamp").set_index("timestamp")
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="h")
    gaps = int(len(full_idx) - len(out))
    out = out.reindex(full_idx)
    out["data_type"] = out["data_type"].fillna("imputed_gap")
    for col in num_cols:
        out[col] = out[col].interpolate(method="time", limit_direction="both")
        # residual NaNs (edges): hour-of-day median
        hour_med = out.groupby(out.index.hour)[col].transform("median")
        out[col] = out[col].fillna(hour_med).fillna(out[col].median())
    out = out.reset_index().rename(columns={"index": "timestamp"})
    log.add(
        file,
        "timestamp",
        "missing_hours_in_span",
        gaps,
        len(full_idx),
        "Reindexed to hourly spine; interpolated numeric gaps",
        "Weather export has holes; without a spine the trip join misses ~11% of hours",
    )

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.reset_index(drop=True)


def parse_attendance(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    text = text.replace(",", "").replace("approx", "").replace("about", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else np.nan


def clean_events(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    file = "events_calendar.csv"
    out = df.copy()
    n = len(out)

    before_types = out["event_type"].nunique()
    out["event_type_raw"] = out["event_type"]
    out["event_type"] = out["event_type"].map(normalize_event_type)
    log.add(
        file,
        "event_type",
        "inconsistent_spelling",
        before_types,
        n,
        f"Normalized {before_types} labels -> {out['event_type'].nunique()} types",
        "Case/spacing variants must collapse for event-type features",
    )

    before_status = out["status"].nunique()
    out["status_raw"] = out["status"]
    out["status"] = out["status"].map(normalize_status)
    log.add(
        file,
        "status",
        "inconsistent_spelling",
        before_status,
        n,
        "Normalized to confirmed/cancelled",
        "Cancelled events should not drive demand features the same way",
    )

    out["start_datetime"] = parse_mixed_timestamps(out["start_datetime"])
    out["end_datetime"] = parse_mixed_timestamps(out["end_datetime"])

    # Fix end < start or missing end: default +2h for matches/concerts else +1h / end-of-day for holidays
    missing_end = int(out["end_datetime"].isna().sum())
    bad_order = int((out["end_datetime"] < out["start_datetime"]).sum())

    def default_end(row: pd.Series) -> pd.Timestamp:
        start = row["start_datetime"]
        end = row["end_datetime"]
        et = row["event_type"]
        if pd.isna(start):
            return end
        if pd.isna(end) or (pd.notna(end) and end < start):
            if et == "public_holiday":
                return start.normalize() + pd.Timedelta(hours=23)
            if et in {"football_match", "concert"}:
                return start + pd.Timedelta(hours=2)
            if et == "road_closure":
                return start + pd.Timedelta(hours=12)
            return start + pd.Timedelta(hours=1)
        return end

    out["end_datetime"] = out.apply(default_end, axis=1)
    log.add(
        file,
        "end_datetime",
        "missing_or_inverted_end",
        missing_end + bad_order,
        n,
        "Filled/fixed ends with type-specific defaults",
        "Interval join needs a valid [start, end] window",
    )

    out["expected_attendance_num"] = out["expected_attendance"].map(parse_attendance)
    messy_att = int(out["expected_attendance"].notna().sum() - out["expected_attendance_num"].notna().sum())
    log.add(
        file,
        "expected_attendance",
        "free_text_numeric",
        int(out["expected_attendance"].notna().sum()),
        n,
        "Parsed digits from free text into expected_attendance_num",
        "Attendance entered as text (commas, 'approx') cannot be modeled raw",
    )

    # Expand zones to long format
    rows = []
    excluded = 0
    for _, row in out.iterrows():
        zones = expand_event_zones(row["zone"])
        if not zones:
            excluded += 1
            continue
        for z in zones:
            r = row.to_dict()
            r["zone"] = z
            r["zone_raw"] = row["zone"]
            rows.append(r)

    long = pd.DataFrame(rows)
    log.add(
        file,
        "zone",
        "multi_zone_and_aliases",
        n,
        n,
        f"Expanded to long format: {n} events -> {len(long)} event-zone rows; excluded={excluded}",
        "Citywide and '&' zones must attach to each affected zone-hour",
    )
    return long.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Joins & features
# ---------------------------------------------------------------------------

def join_weather(trips: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    left_n = len(trips)
    w = weather.rename(columns={"timestamp": "pickup_hour"})
    merged = trips.merge(w, on="pickup_hour", how="left", validate="m:1")
    matched = int(merged["temp_c"].notna().sum())
    audit = {
        "left_rows_before": left_n,
        "rows_after": len(merged),
        "match_rate": round(matched / left_n, 4) if left_n else 0,
        "zone_hours_without_weather": int(merged["temp_c"].isna().sum()),
        "join_type": "left many-to-one on pickup_hour",
    }
    # Fill any residual weather gaps with global medians
    for col in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        merged[col] = merged[col].fillna(merged[col].median())
    if "data_type" in merged.columns:
        merged["data_type"] = merged["data_type"].fillna("missing")
    return merged, audit


def attach_events(trips: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Interval join: flag whether a zone-hour falls inside an expanded event window.
    Window = [start - pre_h, end + post_h] for confirmed events only.
    """
    confirmed = events[events["status"] == "confirmed"].copy()
    cancelled = events[events["status"] == "cancelled"]

    # Precompute windows
    pre_list, post_list = [], []
    for et in confirmed["event_type"]:
        pre, post = EVENT_WINDOWS.get(et, (0, 0))
        pre_list.append(pre)
        post_list.append(post)
    confirmed["pre_h"] = pre_list
    confirmed["post_h"] = post_list
    confirmed["win_start"] = confirmed["start_datetime"] - pd.to_timedelta(confirmed["pre_h"], unit="h")
    confirmed["win_end"] = confirmed["end_datetime"] + pd.to_timedelta(confirmed["post_h"], unit="h")

    # Initialize feature columns
    out = trips.copy()
    out["in_event_window"] = 0
    out["is_public_holiday"] = 0
    out["is_football_window"] = 0
    out["is_concert_window"] = 0
    out["is_road_closure"] = 0
    out["hours_to_next_football"] = np.nan
    out["hours_since_football"] = np.nan
    out["event_attendance_nearby"] = 0.0
    out["n_events_overlapping"] = 0
    out["event_names"] = ""

    matched_events = set()
    # Zone-wise interval tagging (12 zones — manageable)
    for zone in CANONICAL_ZONES:
        z_idx = out.index[out["zone"] == zone]
        if len(z_idx) == 0:
            continue
        z_events = confirmed[confirmed["zone"] == zone]
        if z_events.empty:
            continue
        hours = out.loc[z_idx, "pickup_hour"]
        for _, ev in z_events.iterrows():
            mask = (hours >= ev["win_start"]) & (hours <= ev["win_end"])
            if not mask.any():
                continue
            matched_events.add(ev["event_id"])
            hit_idx = z_idx[mask.values]
            out.loc[hit_idx, "in_event_window"] = 1
            out.loc[hit_idx, "n_events_overlapping"] += 1
            att = ev.get("expected_attendance_num", np.nan)
            if pd.notna(att):
                out.loc[hit_idx, "event_attendance_nearby"] = np.maximum(
                    out.loc[hit_idx, "event_attendance_nearby"], att
                )
            et = ev["event_type"]
            if et == "public_holiday":
                out.loc[hit_idx, "is_public_holiday"] = 1
            elif et == "football_match":
                out.loc[hit_idx, "is_football_window"] = 1
            elif et == "concert":
                out.loc[hit_idx, "is_concert_window"] = 1
            elif et == "road_closure":
                out.loc[hit_idx, "is_road_closure"] = 1

        # Hours to / since nearest football match start in this zone
        foot = z_events[z_events["event_type"] == "football_match"]["start_datetime"].sort_values()
        if len(foot):
            starts = foot.values.astype("datetime64[ns]")
            hvals = hours.values.astype("datetime64[ns]")
            # vector approx via searchsorted
            pos = np.searchsorted(starts, hvals, side="left")
            # next
            next_start = np.full(len(hvals), np.datetime64("NaT"), dtype="datetime64[ns]")
            prev_start = np.full(len(hvals), np.datetime64("NaT"), dtype="datetime64[ns]")
            valid_next = pos < len(starts)
            next_start[valid_next] = starts[pos[valid_next]]
            valid_prev = pos > 0
            prev_start[valid_prev] = starts[pos[valid_prev] - 1]
            to_next = (next_start - hvals) / np.timedelta64(1, "h")
            since = (hvals - prev_start) / np.timedelta64(1, "h")
            out.loc[z_idx, "hours_to_next_football"] = to_next
            out.loc[z_idx, "hours_since_football"] = since

    # Cap distance features for model stability
    out["hours_to_next_football"] = out["hours_to_next_football"].clip(upper=168).fillna(168)
    out["hours_since_football"] = out["hours_since_football"].clip(upper=168).fillna(168)

    audit = {
        "events_total": int(events["event_id"].nunique()) if "event_id" in events.columns else len(events),
        "confirmed_event_rows": int(confirmed["event_id"].nunique()),
        "cancelled_event_rows": int(cancelled["event_id"].nunique()) if len(cancelled) else 0,
        "events_matched_at_least_one_zone_hour": len(matched_events),
        "events_excluded_or_unmatched": int(confirmed["event_id"].nunique()) - len(matched_events),
        "trip_rows_in_any_event_window": int((out["in_event_window"] == 1).sum()),
        "window_rule": "type-specific [start-pre, end+post]; football/concert pre=2h post=2h; confirmed only",
    }
    return out, audit


def add_calendar_and_lag_features(df: pd.DataFrame, train_end: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    ts = out["pickup_hour"]
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek  # 0=Mon
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    # Payday proxy: last 3 days of month or first 3 days
    out["is_payday_window"] = (
        (out["day"] <= 3) | (out["day"] >= (ts.dt.days_in_month - 2))
    ).astype(int)
    # Trend: weeks since 2025-01-01
    origin = pd.Timestamp("2025-01-01")
    out["week_index"] = ((ts - origin).dt.days // 7).astype(int)

    # Weather-derived
    out["rain_mm"] = out["rain_mm"].fillna(0)
    out["rain_class"] = pd.cut(
        out["rain_mm"],
        bins=[-0.01, 0.0, 1.0, 5.0, 50],
        labels=["none", "light", "moderate", "heavy"],
    ).astype(str)

    # Rain last 3 hours per zone (requires sort)
    out = out.sort_values(["zone", "pickup_hour"])
    out["rain_last_3h"] = (
        out.groupby("zone")["rain_mm"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Lag / rolling demand — train-safe: computed from historical trips only.
    # For test rows, we will merge lags from the end of train in build_master.
    if "trips" in out.columns:
        g = out.groupby("zone")["trips"]
        out["trips_lag_168h"] = g.shift(168)  # same hour last week
        out["trips_roll_mean_24h"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
        out["trips_roll_mean_168h"] = g.transform(lambda s: s.shift(1).rolling(168, min_periods=1).mean())

    return out


FEATURE_DICTIONARY = [
    # identity
    ("row_id", "id", "test/submission", "Row id", "pass-through", "n/a"),
    ("record_id", "id", "train", "Record id", "pass-through", "n/a"),
    ("zone", "categorical", "trips", "Canonical pickup zone", "alias map", "yes"),
    ("pickup_hour", "datetime", "trips", "Hour start in EAT", "parsed/floored", "yes"),
    ("trips", "target", "trips", "Trips in zone-hour", "cleaned/capped", "n/a (target)"),
    ("hour", "int", "calendar", "Hour of day 0-23", "pickup_hour.dt.hour", "yes"),
    ("dow", "int", "calendar", "Day of week Mon=0", "pickup_hour.dt.dayofweek", "yes"),
    ("is_weekend", "int", "calendar", "Sat/Sun flag", "dow>=5", "yes"),
    ("month", "int", "calendar", "Month number", "pickup_hour.dt.month", "yes"),
    ("is_payday_window", "int", "calendar", "Payday proximity flag", "day<=3 or last 3 days", "yes"),
    ("week_index", "int", "calendar/trend", "Weeks since 2025-01-01", "(date-origin)//7", "yes"),
    ("temp_c", "float", "weather", "Temperature C", "joined weather", "yes (forecast in Nov)"),
    ("rain_mm", "float", "weather", "Rain mm in hour", "joined weather; sentinel fixed", "yes"),
    ("humidity_pct", "float", "weather", "Relative humidity %", "joined weather", "yes"),
    ("wind_kmh", "float", "weather", "Wind km/h", "joined weather", "yes"),
    ("rain_class", "categorical", "weather", "none/light/moderate/heavy", "binned rain_mm", "yes"),
    ("rain_last_3h", "float", "weather", "Sum of rain previous 3h incl current", "rolling sum by zone", "yes"),
    ("in_event_window", "int", "events", "Any confirmed event window", "interval join", "yes"),
    ("is_public_holiday", "int", "events", "Public holiday flag", "event_type window", "yes"),
    ("is_football_window", "int", "events", "Football pre/during/post window", "±2h window", "yes"),
    ("is_concert_window", "int", "events", "Concert window", "±2h window", "yes"),
    ("is_road_closure", "int", "events", "Road closure active", "event window", "yes"),
    ("hours_to_next_football", "float", "events", "Hours until next match start", "zone match calendar", "yes"),
    ("hours_since_football", "float", "events", "Hours since last match start", "zone match calendar", "yes"),
    ("event_attendance_nearby", "float", "events", "Max attendance in overlapping events", "parsed attendance", "yes"),
    ("n_events_overlapping", "int", "events", "Count of overlapping events", "interval join", "yes"),
    ("trips_lag_168h", "float", "lag", "Trips same zone 168h ago", "shift(168)", "yes (from history)"),
    ("trips_roll_mean_24h", "float", "lag", "Mean trips prior 24h", "shift+rolling", "yes (from history)"),
    ("trips_roll_mean_168h", "float", "lag", "Mean trips prior 168h", "shift+rolling", "yes (from history)"),
    ("avg_fare_birr", "float", "ops", "Avg fare (train only)", "raw cleaned", "no — not known at forecast time"),
    ("avg_wait_min", "float", "ops", "Avg wait (train only)", "raw cleaned", "no — not known at forecast time"),
    ("active_drivers", "int", "ops", "Active drivers (train only)", "raw", "no — not known at forecast time"),
]


MODEL_FEATURES = [
    "zone",
    "hour",
    "dow",
    "is_weekend",
    "month",
    "is_payday_window",
    "week_index",
    "temp_c",
    "rain_mm",
    "humidity_pct",
    "wind_kmh",
    "rain_class",
    "rain_last_3h",
    "in_event_window",
    "is_public_holiday",
    "is_football_window",
    "is_concert_window",
    "is_road_closure",
    "hours_to_next_football",
    "hours_since_football",
    "event_attendance_nearby",
    "n_events_overlapping",
    "trips_lag_168h",
    "trips_roll_mean_24h",
    "trips_roll_mean_168h",
]


def build_master_tables(
    raw: Path | None = None,
) -> dict[str, Any]:
    raw = raw or raw_dir()
    log = CleaningLog()

    train_raw = pd.read_csv(raw / "ride_demand_train.csv")
    test_raw = pd.read_csv(raw / "ride_demand_test.csv")
    weather_raw = pd.read_csv(raw / "weather_hourly.csv")
    events_raw = pd.read_csv(raw / "events_calendar.csv")

    zone_before = {
        "train": sorted(train_raw["zone"].astype(str).unique().tolist()),
        "test": sorted(test_raw["zone"].astype(str).unique().tolist()),
        "events": sorted(events_raw["zone"].astype(str).unique().tolist()),
    }
    event_type_before = sorted(events_raw["event_type"].astype(str).unique().tolist())

    train = clean_trips(train_raw, log, is_train=True)
    test = clean_trips(test_raw, log, is_train=False)
    weather = clean_weather(weather_raw, log)
    events = clean_events(events_raw, log)

    zone_after = {
        "train": sorted(train["zone"].dropna().unique().tolist()),
        "test": sorted(test["zone"].dropna().unique().tolist()),
        "events": sorted(events["zone"].dropna().unique().tolist()),
    }
    event_type_after = sorted(events["event_type"].dropna().unique().tolist())

    # Combine train+test for consistent lag features across boundary, then split
    train["__split"] = "train"
    test["__split"] = "test"
    # Align columns
    for col in ["trips", "avg_fare_birr", "avg_wait_min", "active_drivers", "trips_raw"]:
        if col not in test.columns:
            test[col] = np.nan
    if "record_id" not in test.columns:
        test["record_id"] = np.nan
    if "row_id" not in train.columns:
        train["row_id"] = train.get("record_id")

    common_cols = sorted(set(train.columns) | set(test.columns))
    train = train.reindex(columns=common_cols)
    test = test.reindex(columns=common_cols)
    both = pd.concat([train, test], ignore_index=True)

    both, weather_audit = join_weather(both, weather)
    both, events_audit = attach_events(both, events)
    both = add_calendar_and_lag_features(both)

    # Fill lag NaNs with zone medians learned from train only
    train_mask = both["__split"] == "train"
    for col in ["trips_lag_168h", "trips_roll_mean_24h", "trips_roll_mean_168h"]:
        zone_med = both.loc[train_mask].groupby("zone")[col].transform("median")
        # For all rows, map zone -> train median
        med_map = both.loc[train_mask].groupby("zone")[col].median()
        both[col] = both[col].fillna(both["zone"].map(med_map))
        both[col] = both[col].fillna(both.loc[train_mask, col].median())

    master_train = both.loc[train_mask].drop(columns=["__split"]).reset_index(drop=True)
    master_test = both.loc[~train_mask].drop(columns=["__split"]).reset_index(drop=True)

    # Drop target from test export of modeling features presence check
    dictionary = pd.DataFrame(
        FEATURE_DICTIONARY,
        columns=["column", "type", "source", "description", "derived_how", "known_at_forecast_time"],
    )

    return {
        "master_train": master_train,
        "master_test": master_test,
        "weather_clean": weather,
        "events_clean": events,
        "cleaning_log": log.to_frame(),
        "weather_audit": weather_audit,
        "events_audit": events_audit,
        "zone_before": zone_before,
        "zone_after": zone_after,
        "event_type_before": event_type_before,
        "event_type_after": event_type_after,
        "data_dictionary": dictionary,
        "model_features": MODEL_FEATURES,
    }


def integrity_checks(master_train: pd.DataFrame, master_test: pd.DataFrame) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str) -> None:
        checks.append((name, bool(cond), detail))

    check(
        "train_unique_zone_hour",
        not master_train.duplicated(subset=["zone", "pickup_hour"]).any(),
        f"dupes={master_train.duplicated(subset=['zone','pickup_hour']).sum()}",
    )
    check(
        "test_unique_zone_hour",
        not master_test.duplicated(subset=["zone", "pickup_hour"]).any(),
        f"dupes={master_test.duplicated(subset=['zone','pickup_hour']).sum()}",
    )
    check(
        "zones_are_canonical",
        set(master_train["zone"].unique()).issubset(CANONICAL_ZONES)
        and set(master_test["zone"].unique()).issubset(CANONICAL_ZONES),
        f"train={sorted(master_train['zone'].unique())}",
    )
    check(
        "timestamps_in_range_train",
        master_train["pickup_hour"].min() >= pd.Timestamp("2025-01-01")
        and master_train["pickup_hour"].max() < pd.Timestamp("2025-11-01"),
        f"{master_train['pickup_hour'].min()} .. {master_train['pickup_hour'].max()}",
    )
    check(
        "timestamps_in_range_test",
        master_test["pickup_hour"].min() >= pd.Timestamp("2025-11-01")
        and master_test["pickup_hour"].max() < pd.Timestamp("2025-11-15"),
        f"{master_test['pickup_hour'].min()} .. {master_test['pickup_hour'].max()}",
    )
    check(
        "no_negative_trips_left",
        not ((master_train["trips"].dropna() < 0).any()),
        f"min={master_train['trips'].min()}",
    )
    check(
        "weather_columns_complete",
        master_train[["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]].notna().all().all(),
        "all weather features non-null on train",
    )
    check(
        "same_model_feature_columns",
        set(MODEL_FEATURES).issubset(master_train.columns)
        and set(MODEL_FEATURES).issubset(master_test.columns),
        "model feature list present on train and test",
    )
    leak_cols = {"avg_fare_birr", "avg_wait_min", "active_drivers"}
    check(
        "leaky_ops_not_in_model_features",
        leak_cols.isdisjoint(MODEL_FEATURES),
        f"model_features overlap leaky={leak_cols & set(MODEL_FEATURES)}",
    )
    check(
        "test_row_count_4032",
        len(master_test) == 4032,
        f"len={len(master_test)}",
    )
    return checks


def export_outputs(bundle: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = project_root() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    assets = project_root() / "app" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    bundle["master_train"].to_csv(out_dir / "master_train.csv", index=False)
    bundle["master_test"].to_csv(out_dir / "master_test.csv", index=False)
    bundle["data_dictionary"].to_csv(out_dir / "data_dictionary_master.csv", index=False)
    bundle["cleaning_log"].to_csv(out_dir / "cleaning_log.csv", index=False)
    bundle["weather_clean"].to_csv(out_dir / "weather_clean.csv", index=False)
    bundle["events_clean"].to_csv(out_dir / "events_clean.csv", index=False)

    # Bundle lookup tables for the future demo
    bundle["weather_clean"].to_csv(assets / "weather_clean.csv", index=False)
    bundle["events_clean"].to_csv(assets / "events_clean.csv", index=False)

    # Write integrity check output
    checks = integrity_checks(bundle["master_train"], bundle["master_test"])
    lines = ["# A7 Integrity checks\n"]
    for name, ok, detail in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name} — {detail}")
    (reports / "A7_integrity_checks.md").write_text("\n".join(lines), encoding="utf-8")

    return out_dir

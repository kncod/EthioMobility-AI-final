"""
Addis Ride Demand — operations forecast demo (Deliverable E + stretch extras).

Run from project root:
    python -m streamlit run app/app.py

User picks zone + date (1–14 Nov 2025). Weather and events are looked up
automatically; the saved model returns a 24h forecast with uncertainty band.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
TRIPS_PER_DRIVER = 1.3
# Residual-based buffer from validation MAE (~6.4); use 1.0×MAE as half-width
UNCERTAINTY_K = 1.0
MIN_DATE = date(2025, 11, 1)
MAX_DATE = date(2025, 11, 14)
TEAM = "Addis Demand AI"
MEMBERS = "Abraham Getachew · Ahmed Hussen · Natanim Masresha · Nigus Shiferaw · Tekilu Asefa"
ZONES = [
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
# Zones with thinner history / higher risk (from analysis)
WATCH_ZONES = {"Ayat"}


@st.cache_resource
def load_model():
    import warnings

    path = ASSETS / "final_model.joblib"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "models" / "final_model.joblib"
    # Cloud may install a patch sklearn newer than the pickle; warn-only is OK.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        return joblib.load(path)


@st.cache_data
def load_tables():
    features = pd.read_csv(ASSETS / "master_test_features.csv", parse_dates=["pickup_hour"])
    weather = pd.read_csv(ASSETS / "weather_clean.csv", parse_dates=["timestamp"])
    events = pd.read_csv(ASSETS / "events_clean.csv", parse_dates=["start_datetime", "end_datetime"])
    fare = pd.read_csv(ASSETS / "zone_avg_fare.csv").set_index("zone")["avg_fare_birr"]
    profile = pd.read_csv(ASSETS / "zone_typical_profile.csv")
    return features, weather, events, fare, profile


def day_events(zone: str, day: date, events: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()))
    end = start + pd.Timedelta(hours=23)
    ev = events[
        (events["zone"] == zone)
        & (events["status"] == "confirmed")
        & (events["start_datetime"] <= end)
        & (events["end_datetime"] >= start)
    ].drop_duplicates("event_id")
    return ev.sort_values("start_datetime")


def lookup_context(zone: str, day: date, weather: pd.DataFrame, events: pd.DataFrame) -> str:
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()))
    end = start + pd.Timedelta(hours=23)
    wday = weather[(weather["timestamp"] >= start) & (weather["timestamp"] <= end)].copy()
    bits = []
    if len(wday):
        wet = wday[wday["rain_mm"] > 0.2]
        if len(wet):
            peak = wet.loc[wet["rain_mm"].idxmax()]
            bits.append(f"rain {peak['rain_mm']:.1f} mm at {peak['timestamp'].strftime('%H:%M')}")
        else:
            bits.append("no meaningful rain forecast")
        bits.append(f"temp {wday['temp_c'].min():.0f}–{wday['temp_c'].max():.0f} °C")
    else:
        bits.append("weather lookup missing for this day")

    ev = day_events(zone, day, events)
    if len(ev):
        for _, row in ev.head(5).iterrows():
            bits.append(
                f"{row['event_type']} ({row.get('event_name', '')}) "
                f"{row['start_datetime'].strftime('%H:%M')}–{row['end_datetime'].strftime('%H:%M')}"
            )
    else:
        bits.append("no confirmed local events on calendar")
    return "; ".join(bits)


def event_windows_for_day(zone: str, day: date, events: pd.DataFrame) -> list[tuple[float, float, str]]:
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()))
    end = start + pd.Timedelta(days=1)
    ev = events[
        (events["zone"] == zone)
        & (events["status"] == "confirmed")
        & (events["start_datetime"] < end)
        & (events["end_datetime"] >= start)
    ].drop_duplicates("event_id")
    windows = []
    for _, row in ev.iterrows():
        h0 = max(0.0, (row["start_datetime"] - start).total_seconds() / 3600)
        h1 = min(24.0, (row["end_datetime"] - start).total_seconds() / 3600)
        if h1 > h0:
            windows.append((h0, h1, str(row["event_type"])))
    return windows


def forecast_day(zone: str, day: date, model_bundle, features_df: pd.DataFrame) -> pd.DataFrame:
    day_ts = pd.Timestamp(day)
    mask = (features_df["zone"] == zone) & (features_df["pickup_hour"].dt.normalize() == day_ts)
    rows = features_df.loc[mask].sort_values("pickup_hour").copy()
    if rows.empty:
        return rows
    feats = model_bundle["features"]
    missing = [c for c in feats if c not in rows.columns]
    if missing:
        raise ValueError(f"Feature columns missing from assets: {missing}")
    pred = np.clip(model_bundle["pipeline"].predict(rows[feats]), 0, None)
    mae = float(model_bundle.get("val_mae", 6.4))
    half = UNCERTAINTY_K * mae
    rows["forecast_trips"] = pred
    rows["forecast_low"] = np.clip(pred - half, 0, None)
    rows["forecast_high"] = pred + half
    rows["drivers_needed"] = rows["forecast_trips"] / TRIPS_PER_DRIVER
    rows["drivers_high"] = rows["forecast_high"] / TRIPS_PER_DRIVER
    return rows


def weather_for_day(day: date, weather: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()))
    end = start + pd.Timedelta(hours=23)
    w = weather[(weather["timestamp"] >= start) & (weather["timestamp"] <= end)].copy()
    if len(w):
        w["hour"] = w["timestamp"].dt.hour
    return w.sort_values("timestamp")


def city_day_totals(day: date, model_bundle, features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z in ZONES:
        fc = forecast_day(z, day, model_bundle, features_df)
        if fc.empty:
            continue
        rows.append(
            {
                "zone": z,
                "day_trips": float(fc["forecast_trips"].sum()),
                "peak_hour": int(fc.loc[fc["forecast_trips"].idxmax(), "hour"]),
                "peak_trips": float(fc["forecast_trips"].max()),
                "drivers_at_peak": float(fc["drivers_needed"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("day_trips", ascending=False)


def render_zone_forecast(zone, day, model, features_df, weather, events, fare, profile):
    try:
        fc = forecast_day(zone, day, model, features_df)
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        return

    if fc.empty:
        st.warning("No feature rows found for that zone/date in the bundled test table.")
        return

    avg_fare = float(fare.get(zone, fare.mean()))
    fc = fc.copy()
    fc["expected_fare_birr"] = fc["forecast_trips"] * avg_fare

    typ = profile[profile["zone"] == zone].set_index("hour")["trips"]
    typical_day = float(sum(typ.get(h, 0) for h in range(24)))
    day_trips = float(fc["forecast_trips"].sum())
    vs_pct = 100.0 * (day_trips - typical_day) / typical_day if typical_day > 0 else 0.0

    peak_idx = fc["forecast_trips"].idxmax()
    peak_row = fc.loc[peak_idx]
    context = lookup_context(zone, day, weather, events)
    ev = day_events(zone, day, events)
    wday = weather_for_day(day, weather)
    mae = float(model.get("val_mae", 6.4))

    if zone in WATCH_ZONES:
        st.warning(
            f"**{zone}** has a shorter history (late launch). Treat forecasts with extra caution "
            "and prefer the high end of the uncertainty band when staffing."
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Peak hour", f"{int(peak_row['hour']):02d}:00")
    m2.metric("Peak trips", f"{peak_row['forecast_trips']:.1f}")
    m3.metric("Drivers at peak", f"{peak_row['drivers_needed']:.1f}")
    m4.metric("Day total trips", f"{day_trips:.0f}")
    m5.metric("vs typical weekday", f"{vs_pct:+.1f}%")
    m6.metric("Day gross fares", f"{fc['expected_fare_birr'].sum():,.0f} ETB")

    st.info(f"**Looked up:** {context}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Events on this day")
        if len(ev):
            show_ev = ev[["event_name", "event_type", "start_datetime", "end_datetime", "venue"]].copy()
            show_ev["start_datetime"] = show_ev["start_datetime"].dt.strftime("%H:%M")
            show_ev["end_datetime"] = show_ev["end_datetime"].dt.strftime("%H:%M")
            st.dataframe(show_ev, width="stretch", hide_index=True)
        else:
            st.caption("No confirmed events in this zone for the selected date.")
    with right:
        st.subheader("How to use the band")
        st.markdown(
            f"- **Mid (blue):** point forecast for dispatch planning  \n"
            f"- **Band:** ≈ ±{mae:.1f} trips/hour (validation MAE)  \n"
            f"- **High band → drivers:** peak buffer ≈ "
            f"**{peak_row['drivers_high']:.1f}** drivers  \n"
            f"- Staff to mid; keep high as contingency on event/rain days."
        )

    st.subheader(f"{zone} · {day.isoformat()} — hourly forecast")
    hours = fc["hour"].values
    forecast = fc["forecast_trips"].values
    low = fc["forecast_low"].values
    high = fc["forecast_high"].values
    typical = np.array([typ.get(h, np.nan) for h in hours])

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax = axes[0]
    ax.fill_between(hours, low, high, color="#56B4E9", alpha=0.35, label="Uncertainty band")
    ax.plot(hours, forecast, "o-", color="#0072B2", lw=2, label="Forecast")
    ax.plot(hours, typical, "--", color="#666666", lw=1.5, label="Typical weekday")
    for h0, h1, label in event_windows_for_day(zone, day, events):
        ax.axvspan(h0, h1, color="#E69F00", alpha=0.2, label=f"Event: {label}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)
    ax.set_ylabel("Trips")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Demand forecast vs typical profile")

    axw = axes[1]
    if len(wday):
        axw.bar(wday["hour"], wday["rain_mm"], color="#0072B2", alpha=0.7, label="Rain mm")
        axw2 = axw.twinx()
        axw2.plot(wday["hour"], wday["temp_c"], "o-", color="#D55E00", lw=1.5, label="Temp °C")
        axw.set_ylabel("Rain (mm)")
        axw2.set_ylabel("Temp (°C)")
        h1, l1 = axw.get_legend_handles_labels()
        h2, l2 = axw2.get_legend_handles_labels()
        axw.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        axw.text(0.5, 0.5, "No weather rows", ha="center", va="center", transform=axw.transAxes)
    axw.set_xlabel("Hour (EAT)")
    axw.set_xticks(range(0, 24, 2))
    axw.grid(True, alpha=0.25)
    axw.set_title("Looked-up weather for this date")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    show = fc[
        [
            "pickup_hour",
            "forecast_low",
            "forecast_trips",
            "forecast_high",
            "drivers_needed",
            "drivers_high",
            "expected_fare_birr",
            "temp_c",
            "rain_mm",
            "is_football_window",
            "is_public_holiday",
            "in_event_window",
        ]
    ].copy()
    show["pickup_hour"] = show["pickup_hour"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        show.style.format(
            {
                "forecast_low": "{:.1f}",
                "forecast_trips": "{:.1f}",
                "forecast_high": "{:.1f}",
                "drivers_needed": "{:.1f}",
                "drivers_high": "{:.1f}",
                "expected_fare_birr": "{:,.0f}",
                "temp_c": "{:.1f}",
                "rain_mm": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    csv_bytes = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download 24h plan (CSV)",
        data=csv_bytes,
        file_name=f"forecast_{zone.replace(' ', '_').lower()}_{day.isoformat()}.csv",
        mime="text/csv",
    )

    st.caption(
        "Fares = forecast trips × historical average fare for the zone. "
        f"Drivers ≈ trips ÷ {TRIPS_PER_DRIVER}. "
        "Uncertainty band = point forecast ± validation MAE (residual-based)."
    )


def render_city_overview(day, model, features_df, profile):
    st.subheader(f"City overview · {day.isoformat()}")
    st.caption("All 12 zones for the selected date — totals from the same HistGBM model.")
    with st.spinner("Forecasting all zones…"):
        totals = city_day_totals(day, model, features_df)
    if totals.empty:
        st.warning("No forecasts available for this date.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("City day trips", f"{totals['day_trips'].sum():,.0f}")
    c2.metric("Busiest zone", totals.iloc[0]["zone"])
    c3.metric("Quietest zone", totals.iloc[-1]["zone"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(totals["zone"], totals["day_trips"], color="#0072B2")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Forecast day total trips")
    axes[0].set_title("Day demand by zone")
    axes[0].set_xlim(left=0)

    # vs typical
    typ_day = (
        profile.groupby("zone")["trips"].sum().rename("typical_day")
        if "hour" in profile.columns
        else profile.groupby("zone")["trips"].sum()
    )
    # profile is hourly means — sum over 24 hours for a synthetic typical day
    typ_sum = profile.groupby("zone")["trips"].sum()
    merged = totals.merge(typ_sum.rename("typical_day"), on="zone", how="left")
    merged["vs_pct"] = 100 * (merged["day_trips"] - merged["typical_day"]) / merged["typical_day"]
    colors = ["#009E73" if v >= 0 else "#D55E00" for v in merged["vs_pct"]]
    axes[1].barh(merged["zone"], merged["vs_pct"], color=colors)
    axes[1].axvline(0, color="#333", lw=1)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("% vs typical weekday total")
    axes[1].set_title("Relative to typical weekday")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Peak hour heatmap-like table
    st.markdown("#### Peak hour by zone")
    st.dataframe(
        totals.assign(
            peak_hour=totals["peak_hour"].map(lambda h: f"{h:02d}:00"),
            day_trips=totals["day_trips"].round(1),
            peak_trips=totals["peak_trips"].round(1),
            drivers_at_peak=totals["drivers_at_peak"].round(1),
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download city overview (CSV)",
        data=totals.to_csv(index=False).encode("utf-8"),
        file_name=f"city_overview_{day.isoformat()}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Addis Demand AI — Ride Forecast",
        page_icon="🚕",
        layout="wide",
    )

    st.title("Addis Ride Demand Forecast")
    st.markdown(f"**{TEAM}**  \n{MEMBERS}  \n*IDAE · EDI-Qiyas-CoDiST Advanced Digital Skills Program*")
    st.caption(
        "Operations demo — pick a zone and a November 2025 date. "
        "Weather and events are looked up automatically; no manual weather/event inputs."
    )

    try:
        model = load_model()
        features_df, weather, events, fare, profile = load_tables()
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load demo assets/model: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Inputs")
        zone = st.selectbox("Zone", ZONES, index=ZONES.index("Kazanchis"))
        day = st.date_input(
            "Date",
            value=MIN_DATE,
            min_value=date(2025, 1, 1),
            max_value=date(2025, 12, 31),
            help="Supported forecast window: 1–14 November 2025",
        )
        st.markdown("---")
        st.subheader("Model card")
        st.markdown(
            f"""
- **Model:** HistGBM (sklearn)  
- **Val RMSE:** `{model['val_rmse']:.2f}`  
- **Val MAE:** `{model['val_mae']:.2f}`  
- **Features:** calendar + weather + events + lags  
- **Excluded:** fare, wait, active drivers (leakage)  
- **Drivers rule:** trips ÷ {TRIPS_PER_DRIVER}  
- **Uncertainty:** ± MAE trips/hour  
"""
        )

    if day < MIN_DATE or day > MAX_DATE:
        st.warning(
            f"Please choose a date between **{MIN_DATE.isoformat()}** and "
            f"**{MAX_DATE.isoformat()}** (forecast fortnight)."
        )
        st.stop()

    tab_zone, tab_city = st.tabs(["Zone forecast", "City overview"])
    with tab_zone:
        render_zone_forecast(zone, day, model, features_df, weather, events, fare, profile)
    with tab_city:
        render_city_overview(day, model, features_df, profile)


if __name__ == "__main__":
    main()

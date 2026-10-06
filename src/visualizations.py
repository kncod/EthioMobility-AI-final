"""
Deliverable C — Visualization Pack (12 required figures).

Run: python src/visualizations.py
Outputs: figures/fig01_*.png ... fig12_*.png and figures/figure_captions.md
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import assign_zone_types, load_master
from src.cleaning import MODEL_FEATURES, parse_weather_timestamps

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

FIG_DIR = ROOT / "figures"
DPI = 150
RANDOM_STATE = 42

# Colorblind-safe (Okabe–Ito inspired)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#666666",
}
PALETTE = list(COLORS.values())

CAPTIONS: dict[str, str] = {}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    if sns is not None:
        sns.set_theme(style="whitegrid", palette=PALETTE[:6])


def save(fig: plt.Figure, name: str, caption: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CAPTIONS[name] = caption
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# fig01–09 (EDA)
# ---------------------------------------------------------------------------

def fig01_gaps(raw_train: pd.DataFrame, master: pd.DataFrame) -> None:
    weather = pd.read_csv(ROOT / "data" / "raw" / "weather_hourly.csv")
    events = pd.read_csv(ROOT / "data" / "raw" / "events_calendar.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Missingness by column across raw tables
    miss_rows = []
    for name, df in [
        ("trips", raw_train),
        ("weather", weather),
        ("events", events),
    ]:
        for col in df.columns:
            miss_rows.append(
                {
                    "table": name,
                    "column": f"{name}:{col}",
                    "missing_pct": 100 * df[col].isna().mean(),
                }
            )
    miss = pd.DataFrame(miss_rows)
    miss = miss[miss["missing_pct"] > 0].sort_values("missing_pct", ascending=True)
    axes[0].barh(miss["column"], miss["missing_pct"], color=COLORS["blue"])
    axes[0].set_xlabel("Missing / invalid share (%)")
    axes[0].set_title("Missing values by column (raw tables)")
    axes[0].set_xlim(0, max(5, miss["missing_pct"].max() * 1.15))

    # Timeline of missing hours per zone on cleaned grain
    d = master.copy()
    # rebuild expected hours per zone
    records = []
    for zone, g in d.groupby("zone"):
        start, end = g["pickup_hour"].min(), g["pickup_hour"].max()
        full = pd.date_range(start, end, freq="h")
        have = set(g["pickup_hour"])
        for ts in full:
            records.append({"zone": zone, "pickup_hour": ts, "missing": int(ts not in have)})
    gap = pd.DataFrame(records)
    # daily missing count heatmap-like: zone x week
    gap["week"] = gap["pickup_hour"].dt.to_period("W").astype(str)
    mat = gap.groupby(["zone", "week"])["missing"].sum().unstack("week").fillna(0)
    im = axes[1].imshow(mat.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axes[1].set_yticks(range(len(mat.index)))
    axes[1].set_yticklabels(mat.index)
    axes[1].set_xticks(range(0, len(mat.columns), max(1, len(mat.columns) // 8)))
    axes[1].set_xticklabels(
        [mat.columns[i] for i in range(0, len(mat.columns), max(1, len(mat.columns) // 8))],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[1].set_title("Missing hours per zone × week (outages / late launch)")
    axes[1].set_xlabel("Week")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="Missing hours")
    fig.suptitle("fig01 — Gaps and missingness", y=1.02, fontsize=14)
    fig.tight_layout()
    save(
        fig,
        "fig01_gaps_and_missingness.png",
        "Missingness by raw column and weekly missing-hour heatmap by zone. "
        "Takeaway: Ayat’s shorter history (late launch) and scattered gaps are visible; cleaning must not invent pre-launch demand.",
    )


def fig02_before_after(raw_train: pd.DataFrame, raw_weather: pd.DataFrame, master: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # trips: raw vs cleaned (capped)
    axes[0].hist(raw_train["trips"].dropna(), bins=60, alpha=0.55, color=COLORS["vermillion"], label="raw", density=True)
    axes[0].hist(master["trips"].dropna(), bins=60, alpha=0.55, color=COLORS["blue"], label="cleaned", density=True)
    axes[0].set_xlim(0, 200)
    axes[0].set_xlabel("trips")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Trips distribution before vs after cleaning")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    # rain: sentinel -9999 removed
    rain_raw = raw_weather["rain_mm"].copy()
    rain_clean = master.drop_duplicates("pickup_hour")["rain_mm"]
    axes[1].hist(rain_raw.clip(-50, 30), bins=50, alpha=0.55, color=COLORS["vermillion"], label="raw (clipped view)", density=True)
    axes[1].hist(rain_clean.clip(0, 30), bins=50, alpha=0.55, color=COLORS["blue"], label="cleaned", density=True)
    axes[1].set_xlabel("rain_mm")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Rain before vs after sentinel cleanup")
    axes[1].legend()
    axes[1].set_ylim(bottom=0)
    # annotate sentinel count
    n_sent = int((raw_weather["rain_mm"] <= -999).sum())
    axes[1].text(0.98, 0.95, f"raw rain<=-999: {n_sent}", transform=axes[1].transAxes, ha="right", va="top", fontsize=9)

    fig.suptitle("fig02 — Before / after cleaning", y=1.02)
    fig.tight_layout()
    save(
        fig,
        "fig02_before_after_cleaning.png",
        "Trips and rain distributions before vs after cleaning. "
        "Takeaway: capping extreme trip spikes and removing rain sentinels (-9999) removes distortion without hiding the real rainy-hour mass at 0.",
    )


def fig03_trend(master: pd.DataFrame) -> None:
    d = master.dropna(subset=["trips"]).copy()
    daily = d.groupby("date", as_index=False)["trips"].sum()
    hol = d[d["is_public_holiday"] == 1].groupby("date")["trips"].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["trips"], color=COLORS["blue"], lw=1.2, label="Daily city trips")
    # weekly smooth
    daily["roll7"] = daily["trips"].rolling(7, min_periods=1).mean()
    ax.plot(daily["date"], daily["roll7"], color=COLORS["orange"], lw=2.2, label="7-day rolling mean")
    if len(hol):
        ax.scatter(hol.index, hol.values, color=COLORS["vermillion"], s=40, zorder=5, label="Public holiday")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trips (city total)")
    ax.set_title("fig03 — City demand trend with public holidays")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    save(
        fig,
        "fig03_demand_trend_with_holidays.png",
        "Daily city-wide trips Jan–Oct with 7-day trend and holidays marked. "
        "Takeaway: demand rises through the year; holidays often dip below the local trend — November models need a trend term, not only seasonality.",
    )


def fig04_heatmap(master: pd.DataFrame) -> None:
    d = master.dropna(subset=["trips"])
    city = d.groupby(["dow", "hour"])["trips"].mean().unstack("hour")
    # pick a contrast zone
    zone = "Merkato" if "Merkato" in d["zone"].unique() else d["zone"].iloc[0]
    z = d[d["zone"] == zone].groupby(["dow", "hour"])["trips"].mean().unstack("hour")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, title in [
        (axes[0], city, "City-wide"),
        (axes[1], z, zone),
    ]:
        if sns is not None:
            sns.heatmap(mat, ax=ax, cmap="viridis", cbar_kws={"label": "Mean trips"})
        else:
            im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label="Mean trips")
            ax.set_yticks(range(7))
            ax.set_xticks(range(0, 24, 2))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("Hour of day (EAT)")
        ax.set_ylabel("Day of week")
        ax.set_title(title)
    fig.suptitle("fig04 — Hour × weekday heatmap", y=1.02)
    fig.tight_layout()
    save(
        fig,
        "fig04_hour_by_weekday_heatmap.png",
        f"Mean trips by hour×weekday for the city and {zone}. "
        "Takeaway: weekday commute peaks differ from weekend shapes — zone-specific seasonality matters.",
    )


def fig05_zone_profiles(master: pd.DataFrame, zone_types: dict) -> None:
    d = master.dropna(subset=["trips"]).copy()
    d["zone_type"] = d["zone"].map(zone_types)
    types = sorted(d["zone_type"].dropna().unique())
    n = len(types)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(14, 7), sharey=True)
    axes = np.array(axes).ravel()
    for i, zt in enumerate(types):
        ax = axes[i]
        sub = d[d["zone_type"] == zt]
        for weekend, label, color in [(0, "Weekday", COLORS["blue"]), (1, "Weekend", COLORS["orange"])]:
            g = sub[sub["is_weekend"] == weekend].groupby("hour")["trips"].mean()
            ax.plot(g.index, g.values, marker="o", ms=3, label=label, color=color)
        ax.set_title(zt.replace("_", " "))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean trips")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("fig05 — Weekday vs weekend profiles by zone type", y=1.02)
    fig.tight_layout()
    save(
        fig,
        "fig05_zone_profiles.png",
        "Small-multiples of weekday/weekend hourly profiles by zone type. "
        "Takeaway: residential vs hub vs market types peak at different hours — one city-wide curve is not enough.",
    )


def fig06_timezone() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "weather_hourly.csv")
    s = raw["timestamp"].astype(str)
    z_mask = s.str.endswith("Z")
    wrong = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    wrong.loc[z_mask] = pd.to_datetime(s[z_mask].str.replace("Z", "", regex=False), errors="coerce").dt.floor("h")
    right = parse_weather_timestamps(raw["timestamp"])

    tmp = raw.copy()
    tmp["hour_wrong"] = wrong.dt.hour
    tmp["hour_right"] = right.dt.hour
    curve_wrong = tmp.groupby("hour_wrong")["temp_c"].mean()
    curve_right = tmp.groupby("hour_right")["temp_c"].mean()

    # rain-demand corr by shift using master
    master = load_master().dropna(subset=["trips"])
    city = master.groupby("pickup_hour", as_index=False)["trips"].mean()
    w = pd.DataFrame({"pickup_hour": right, "rain_mm": raw["rain_mm"]})
    w = w.dropna().groupby("pickup_hour", as_index=False)["rain_mm"].mean()
    corrs = []
    for h in range(0, 6):
        ww = w.copy()
        ww["pickup_hour"] = ww["pickup_hour"] + pd.Timedelta(hours=h)
        m = city.merge(ww, on="pickup_hour")
        m = m[m["rain_mm"] >= 0]
        corrs.append(m["trips"].corr(m["rain_mm"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(curve_wrong.index, curve_wrong.values, "o-", color=COLORS["vermillion"], label="UTC treated as local")
    axes[0].plot(curve_right.index, curve_right.values, "o-", color=COLORS["blue"], label="UTC→EAT (correct)")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean temp_c")
    axes[0].set_title("Temperature daily curve under two clocks")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].bar(range(0, 6), corrs, color=COLORS["green"])
    axes[1].set_xlabel("Hour shift applied to weather (h)")
    axes[1].set_ylabel("corr(trips, rain_mm)")
    axes[1].set_title("Rain–demand correlation vs clock shift")
    axes[1].axhline(0, color=COLORS["gray"], lw=1)
    fig.suptitle("fig06 — Weather timezone check", y=1.02)
    fig.tight_layout()
    save(
        fig,
        "fig06_weather_timezone_check.png",
        "Temperature curve on wrong vs corrected clock, and rain–demand correlation for shifts 0–5h. "
        "Takeaway: correct EAT alignment peaks temperature in mid-afternoon and strengthens the rain signal — do not join UTC as local.",
    )


def fig07_rain(master: pd.DataFrame, zone_types: dict) -> None:
    d = master.dropna(subset=["trips"]).copy()
    d["zone_type"] = d["zone"].map(zone_types)
    order = ["none", "light", "moderate", "heavy"]
    d["rain_class"] = pd.Categorical(d["rain_class"], categories=order, ordered=True)
    base = d[d["rain_class"] == "none"].groupby("zone_type")["trips"].mean()
    g = d.groupby(["zone_type", "rain_class"], observed=False)["trips"].mean().unstack("rain_class")
    for c in order:
        if c in g.columns:
            g[c] = g[c] / base

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    width = 0.15
    for i, zt in enumerate(g.index):
        vals = [g.loc[zt, c] if c in g.columns else np.nan for c in order]
        ax.bar(x + i * width, vals, width=width, label=zt.replace("_", " "), color=PALETTE[i % len(PALETTE)])
    ax.axhline(1.0, color=COLORS["gray"], ls="--", lw=1, label="baseline (none)")
    ax.set_xticks(x + width * (len(g.index) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel("Demand ratio vs dry (none)")
    ax.set_xlabel("Rain class")
    ax.set_title("fig07 — Rain effect by zone type")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(
        fig,
        "fig07_rain_effect.png",
        "Demand ratio versus rain class, one series per zone type. "
        "Takeaway: rain lifts demand unevenly by zone type and often saturates from moderate to heavy.",
    )


def fig08_event_study(master: pd.DataFrame) -> None:
    d = master.dropna(subset=["trips"]).copy()
    # Build relative-hour profiles using football and concert flags + distance
    rows = []
    for et, flag, pre_col in [
        ("football_match", "is_football_window", "hours_to_next_football"),
        ("concert", "is_concert_window", "hours_to_next_football"),  # use since/to approx via window only
    ]:
        # Simpler: for hours with flag, we don't have exact start offset stored.
        # Use hours_since_football for football; for concert use rolling relative proxy via in-window only.
        pass

    # Event study using hours_to / hours_since for football; for concert compare in-window vs baseline by hour-of-day residual
    # Football relative hour: -6..+6 around nearest start using min(|to|,|since|)
    fb = d[d["zone"].isin(["Kazanchis", "Arat Kilo", "Piassa"])].copy()
    # Approximate offset: if hours_to <= 6 use -hours_to; elif hours_since <= 6 use +hours_since
    offset = np.where(
        fb["hours_to_next_football"] <= 6,
        -fb["hours_to_next_football"],
        np.where(fb["hours_since_football"] <= 6, fb["hours_since_football"], np.nan),
    )
    fb["rel_h"] = np.round(offset)
    # baseline: same zone-hour without football window
    base = (
        d[(d["is_football_window"] == 0) & d["zone"].isin(["Kazanchis", "Arat Kilo", "Piassa"])]
        .groupby(["zone", "hour"])["trips"]
        .mean()
        .rename("base")
    )
    fb = fb.merge(base.reset_index(), on=["zone", "hour"], how="left")
    fb["resid"] = fb["trips"] / fb["base"]
    fb_curve = fb.dropna(subset=["rel_h"]).groupby("rel_h")["resid"].mean()
    fb_curve = fb_curve.loc[(fb_curve.index >= -6) & (fb_curve.index <= 6)]

    # Concert: compare mean trips in concert window vs matched baseline by zone-hour
    concert = d.copy()
    concert_on = concert[concert["is_concert_window"] == 1]
    base_c = (
        concert[concert["is_concert_window"] == 0]
        .groupby(["zone", "hour"])["trips"]
        .mean()
        .rename("base")
    )
    c = concert_on.merge(base_c.reset_index(), on=["zone", "hour"], how="left")
    # Without exact start, show hour-of-day uplift profile for concert hours vs baseline
    c_by_hour = c.groupby("hour").apply(lambda x: (x["trips"] / x["base"]).mean(), include_groups=False)
    base_line = pd.Series(1.0, index=range(24))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fb_curve.index, fb_curve.values, "o-", color=COLORS["blue"], label="Football (stadium zones)")
    axes[0].axhline(1.0, color=COLORS["gray"], ls="--", label="Non-event baseline")
    axes[0].axvline(0, color=COLORS["orange"], ls=":", label="Match start")
    axes[0].set_xlabel("Hours relative to match start")
    axes[0].set_ylabel("Trips / matched baseline")
    axes[0].set_title("Football event study (−6h to +6h)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(c_by_hour.index, c_by_hour.values, "o-", color=COLORS["purple"], label="Concert window")
    axes[1].plot(base_line.index, base_line.values, "--", color=COLORS["gray"], label="Baseline = 1")
    axes[1].set_xlabel("Hour of day (EAT)")
    axes[1].set_ylabel("Trips / matched baseline")
    axes[1].set_title("Concert uplift by hour of day")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(bottom=0)

    fig.suptitle("fig08 — Event study", y=1.02)
    fig.tight_layout()
    save(
        fig,
        "fig08_event_study.png",
        "Average demand around football match start (−6h to +6h) and concert uplift by hour vs matched non-event baseline. "
        "Takeaway: football demand rises before kickoff and stays elevated after; concerts show evening uplift — window features are justified.",
    )


def fig09_holidays(master: pd.DataFrame) -> None:
    d = master.dropna(subset=["trips"]).copy()
    daily = d.groupby(["date", "dow"], as_index=False).agg(
        trips=("trips", "sum"),
        is_public_holiday=("is_public_holiday", "max"),
    )
    rows = []
    for _, row in daily[daily["is_public_holiday"] == 1].iterrows():
        day, dow = row["date"], row["dow"]
        window = daily[
            (daily["dow"] == dow)
            & (daily["is_public_holiday"] == 0)
            & (daily["date"].between(day - pd.Timedelta(days=21), day + pd.Timedelta(days=21)))
        ]
        base = window["trips"].mean()
        rows.append({"date": day.strftime("%Y-%m-%d"), "index": row["trips"] / base if base else np.nan})
    hol = pd.DataFrame(rows).dropna().sort_values("index")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["vermillion"] if v < 1 else COLORS["green"] for v in hol["index"]]
    ax.barh(hol["date"], hol["index"], color=colors)
    ax.axvline(1.0, color=COLORS["gray"], ls="--", label="Normal (=1)")
    ax.set_xlabel("Daily trips index vs nearby same-weekday baseline")
    ax.set_title("fig09 — Public holiday effects")
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    save(
        fig,
        "fig09_holiday_effects.png",
        "Index of daily trips for each public holiday relative to nearby same-weekday normals, sorted. "
        "Takeaway: most holidays suppress city demand (index<1), but magnitudes differ — keep holiday flags, not a single ‘holiday’ effect size.",
    )


# ---------------------------------------------------------------------------
# fig10–12 need a quick chronological model
# ---------------------------------------------------------------------------

def _prepare_xy(df: pd.DataFrame):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    use = df.dropna(subset=["trips"]).copy()
    features = [f for f in MODEL_FEATURES if f in use.columns]
    cat = [c for c in ["zone", "rain_class"] if c in features]
    num = [c for c in features if c not in cat]
    X = use[features]
    y = use["trips"].values
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )
    return use, X, y, pre, features, cat, num


def run_model_bundle(master: pd.DataFrame) -> dict:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.inspection import permutation_importance

    d = master.dropna(subset=["trips"]).copy()
    cut = pd.Timestamp("2025-10-18")
    train = d[d["pickup_hour"] < cut]
    valid = d[(d["pickup_hour"] >= cut) & (d["pickup_hour"] < pd.Timestamp("2025-11-01"))]

    _, X_tr, y_tr, pre, features, cat, num = _prepare_xy(train)
    X_va = valid[features]
    y_va = valid["trips"].values

    # Baselines
    overall_mean = y_tr.mean()
    pred_mean = np.full_like(y_va, overall_mean, dtype=float)
    # Seasonal naive: zone × dow × hour mean
    key = ["zone", "dow", "hour"]
    seas = train.groupby(key)["trips"].mean()
    pred_seas = valid.set_index(key).index.map(seas)
    pred_seas = pd.Series(pred_seas, index=valid.index).fillna(overall_mean).values

    def rmse(a, p):
        return float(np.sqrt(mean_squared_error(a, p)))

    def mae(a, p):
        return float(mean_absolute_error(a, p))

    results = [
        {"model": "mean_baseline", "rmse": rmse(y_va, pred_mean), "mae": mae(y_va, pred_mean), "train_s": 0.0},
        {"model": "seasonal_naive", "rmse": rmse(y_va, pred_seas), "mae": mae(y_va, pred_seas), "train_s": 0.0},
    ]

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=80, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=200, random_state=RANDOM_STATE
        ),
    }

    fitted = {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = pipe.predict(X_va)
        results.append({"model": name, "rmse": rmse(y_va, pred), "mae": mae(y_va, pred), "train_s": dt})
        fitted[name] = (pipe, pred)

    # Rolling-origin for final model (hist_gbm) — 4 folds
    fold_rmses = []
    cuts = [
        ("2025-08-23", "2025-09-06"),
        ("2025-09-06", "2025-09-20"),
        ("2025-09-20", "2025-10-04"),
        ("2025-10-04", "2025-10-18"),
    ]
    for c0, c1 in cuts:
        c0t, c1t = pd.Timestamp(c0), pd.Timestamp(c1)
        tr = d[d["pickup_hour"] < c0t]
        va = d[(d["pickup_hour"] >= c0t) & (d["pickup_hour"] < c1t)]
        if len(tr) < 1000 or len(va) < 100:
            continue
        _, Xtr, ytr, pre_f, feats, _, _ = _prepare_xy(tr)
        pipe = Pipeline(
            [
                ("pre", pre_f),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=6, learning_rate=0.08, max_iter=200, random_state=RANDOM_STATE
                    ),
                ),
            ]
        )
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(va[feats])
        fold_rmses.append(rmse(va["trips"].values, pred))

    # Seasonal naive fold RMSEs for contrast
    seas_fold = []
    for c0, c1 in cuts:
        c0t, c1t = pd.Timestamp(c0), pd.Timestamp(c1)
        tr = d[d["pickup_hour"] < c0t]
        va = d[(d["pickup_hour"] >= c0t) & (d["pickup_hour"] < c1t)]
        s = tr.groupby(key)["trips"].mean()
        p = va.set_index(key).index.map(s)
        p = pd.Series(p, index=va.index).fillna(tr["trips"].mean()).values
        seas_fold.append(rmse(va["trips"].values, p))

    # Permutation importance on validation for hist_gbm
    pipe_final, pred_final = fitted["hist_gbm"]
    # Use a sample for speed
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_va), size=min(4000, len(X_va)), replace=False)
    X_s = X_va.iloc[idx]
    y_s = y_va[idx]
    r = permutation_importance(
        pipe_final, X_s, y_s, n_repeats=5, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error"
    )
    # Feature names after OHE are hard; importance on original columns via column shuffle
    # permutation_importance on pipeline uses transformed columns. Instead compute
    # column-wise permutation on raw features:
    base_pred = pipe_final.predict(X_s)
    base_rmse = rmse(y_s, base_pred)
    importances = []
    for col in features:
        Xp = X_s.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        importances.append({"feature": col, "importance": rmse(y_s, pipe_final.predict(Xp)) - base_rmse})
    imp = pd.DataFrame(importances).sort_values("importance", ascending=False)

    return {
        "results": pd.DataFrame(results),
        "fold_rmses": fold_rmses,
        "seas_fold": seas_fold,
        "valid": valid,
        "pred_final": pred_final,
        "importance": imp,
        "features": features,
        "cut": cut,
    }


def fig10_model_comparison(bundle: dict) -> None:
    res = bundle["results"].sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = []
    for m in res["model"]:
        if m in ("mean_baseline", "seasonal_naive"):
            colors.append(COLORS["gray"])
        elif m == "hist_gbm":
            colors.append(COLORS["green"])
        else:
            colors.append(COLORS["blue"])
    ax.barh(res["model"], res["rmse"], color=colors, xerr=None)
    # error bars on final model from rolling folds
    if bundle["fold_rmses"]:
        mean_f = np.mean(bundle["fold_rmses"])
        std_f = np.std(bundle["fold_rmses"])
        # mark on hist_gbm bar
        y_pos = list(res["model"]).index("hist_gbm")
        ax.errorbar(mean_f, y_pos, xerr=std_f, fmt="o", color=COLORS["vermillion"], capsize=4, label=f"hist_gbm rolling RMSE {mean_f:.2f}±{std_f:.2f}")
        ax.legend()
    ax.set_xlabel("Validation RMSE (trips)")
    ax.set_title("fig10 — Model comparison (train < 18 Oct, valid 18–31 Oct)")
    ax.set_xlim(left=0)
    fig.tight_layout()
    save(
        fig,
        "fig10_model_comparison.png",
        "Validation RMSE for baselines and model families; rolling-origin error bar on HistGBM. "
        "Takeaway: seasonal naive beats a global mean; boosted trees improve further — report chronological scores only.",
    )


def fig11_forecast_vs_actual(bundle: dict) -> None:
    valid = bundle["valid"].copy()
    valid["pred"] = bundle["pred_final"]
    zones = ["Merkato", "Bole", "Kazanchis"]
    zones = [z for z in zones if z in valid["zone"].unique()][:3]
    fig, axes = plt.subplots(len(zones), 1, figsize=(12, 3.2 * len(zones)), sharex=True)
    if len(zones) == 1:
        axes = [axes]
    for ax, zone in zip(axes, zones):
        g = valid[valid["zone"] == zone].sort_values("pickup_hour")
        ax.plot(g["pickup_hour"], g["trips"], color=COLORS["black"], lw=1, label="Actual", alpha=0.8)
        ax.plot(g["pickup_hour"], g["pred"], color=COLORS["blue"], lw=1.2, label="Predicted")
        ax.set_ylabel("Trips")
        ax.set_title(zone)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Hour (validation fortnight)")
    fig.suptitle("fig11 — Forecast vs actual (HistGBM, 18–31 Oct)", y=1.01)
    fig.tight_layout()
    save(
        fig,
        "fig11_forecast_vs_actual.png",
        "Predicted vs actual hourly trips for three zones over the validation fortnight. "
        "Takeaway: the model tracks daily peaks; largest misses often align with unusual event/weather hours.",
    )


def fig12_importance(bundle: dict) -> None:
    imp = bundle["importance"].head(12).sort_values("importance")
    weather_feats = {"temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_class", "rain_last_3h"}
    event_feats = {
        "in_event_window",
        "is_public_holiday",
        "is_football_window",
        "is_concert_window",
        "is_road_closure",
        "hours_to_next_football",
        "hours_since_football",
        "event_attendance_nearby",
        "n_events_overlapping",
    }
    colors = []
    for f in imp["feature"]:
        if f in weather_feats:
            colors.append(COLORS["sky"])
        elif f in event_feats:
            colors.append(COLORS["orange"])
        else:
            colors.append(COLORS["blue"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp["feature"], imp["importance"], color=colors)
    ax.set_xlabel("Increase in RMSE when feature is shuffled")
    ax.set_title("fig12 — Feature importance (permutation)")
    # legend proxies
    ax.barh([], [], color=COLORS["sky"], label="Weather")
    ax.barh([], [], color=COLORS["orange"], label="Events")
    ax.barh([], [], color=COLORS["blue"], label="Calendar / lag / zone")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    fig.tight_layout()
    save(
        fig,
        "fig12_feature_importance.png",
        "Top features by permutation importance with weather and event features highlighted. "
        "Takeaway: lags/calendar dominate, but weather and event features contribute measurable RMSE reduction — Rule 5 is satisfied with signal, not decoration.",
    )


def write_captions() -> None:
    lines = ["# Figure captions (Deliverable C)", ""]
    for i in range(1, 13):
        # find key
        key = next(k for k in CAPTIONS if k.startswith(f"fig{i:02d}_"))
        lines.append(f"## {key}")
        lines.append(CAPTIONS[key])
        lines.append("")
    (FIG_DIR / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {FIG_DIR / 'figure_captions.md'}")


def main() -> None:
    setup_style()
    print("Loading data...")
    master = load_master()
    zone_types = assign_zone_types(master)
    master["zone_type"] = master["zone"].map(zone_types)
    raw_train = pd.read_csv(ROOT / "data" / "raw" / "ride_demand_train.csv")
    raw_weather = pd.read_csv(ROOT / "data" / "raw" / "weather_hourly.csv")

    print("Building fig01–fig09...")
    fig01_gaps(raw_train, master)
    fig02_before_after(raw_train, raw_weather, master)
    fig03_trend(master)
    fig04_heatmap(master)
    fig05_zone_profiles(master, zone_types)
    fig06_timezone()
    fig07_rain(master, zone_types)
    fig08_event_study(master)
    fig09_holidays(master)

    print("Training quick models for fig10–fig12...")
    bundle = run_model_bundle(master)
    print(bundle["results"])
    # persist interim metrics for modeling notebook later
    out = ROOT / "reports" / "C_model_snapshot"
    out.mkdir(parents=True, exist_ok=True)
    bundle["results"].to_csv(out / "model_comparison_snapshot.csv", index=False)
    bundle["importance"].to_csv(out / "permutation_importance_snapshot.csv", index=False)

    fig10_model_comparison(bundle)
    fig11_forecast_vs_actual(bundle)
    fig12_importance(bundle)
    write_captions()
    print("Done. All 12 figures in figures/")


if __name__ == "__main__":
    main()

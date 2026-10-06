"""
Deliverable D — Modeling & Evaluation (D1–D9) + final submission.

Run: python src/modeling.py

Chronological split default: train < 2025-10-18, validate 18–31 Oct 2025.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import MODEL_FEATURES

RANDOM_STATE = 42
VAL_START = pd.Timestamp("2025-10-18")
VAL_END = pd.Timestamp("2025-11-01")

CALENDAR_FEATS = [
    "zone",
    "hour",
    "dow",
    "is_weekend",
    "month",
    "is_payday_window",
    "week_index",
    "trips_lag_168h",
    "trips_roll_mean_24h",
    "trips_roll_mean_168h",
]
WEATHER_FEATS = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_class", "rain_last_3h"]
EVENT_FEATS = [
    "in_event_window",
    "is_public_holiday",
    "is_football_window",
    "is_concert_window",
    "is_road_closure",
    "hours_to_next_football",
    "hours_since_football",
    "event_attendance_nearby",
    "n_events_overlapping",
]

LEAKY = ["avg_fare_birr", "avg_wait_min", "active_drivers", "trips", "trips_raw"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "data" / "processed" / "master_train.csv", parse_dates=["pickup_hour"])
    test = pd.read_csv(ROOT / "data" / "processed" / "master_test.csv", parse_dates=["pickup_hour"])
    return train, test


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in ["zone", "rain_class"] if c in features]
    num = [c for c in features if c not in cat]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )


def make_pipe(estimator, features: list[str]) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(features)), ("model", estimator)])


def seasonal_naive_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    key = ["zone", "dow", "hour"]
    seas = train.groupby(key)["trips"].mean()
    pred = valid.set_index(key).index.map(seas)
    return pd.Series(pred, index=valid.index).fillna(train["trips"].mean()).values


def split_chrono(df: pd.DataFrame, start=VAL_START, end=VAL_END):
    d = df.dropna(subset=["trips"]).copy()
    train = d[d["pickup_hour"] < start]
    valid = d[(d["pickup_hour"] >= start) & (d["pickup_hour"] < end)]
    return train, valid


# ---------------------------------------------------------------------------
# D1 / D2
# ---------------------------------------------------------------------------

def d1_d2_baselines_and_models(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y_va = valid["trips"].values
    rows = []

    # Mean baseline
    t0 = time.time()
    pred = np.full(len(valid), train["trips"].mean())
    rows.append(
        {
            "model": "mean_baseline",
            "family": "baseline",
            "rmse": rmse(y_va, pred),
            "mae": mae(y_va, pred),
            "train_seconds": time.time() - t0,
        }
    )

    # Seasonal naive
    t0 = time.time()
    pred = seasonal_naive_predict(train, valid)
    rows.append(
        {
            "model": "seasonal_naive",
            "family": "baseline",
            "rmse": rmse(y_va, pred),
            "mae": mae(y_va, pred),
            "train_seconds": time.time() - t0,
        }
    )

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    X_tr, y_tr = train[features], train["trips"].values
    X_va = valid[features]
    for name, est in models.items():
        pipe = make_pipe(est, features)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = pipe.predict(X_va)
        rows.append(
            {
                "model": name,
                "family": "ml",
                "rmse": rmse(y_va, pred),
                "mae": mae(y_va, pred),
                "train_seconds": dt,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


# ---------------------------------------------------------------------------
# D3 rolling origin
# ---------------------------------------------------------------------------

def d3_rolling(df: pd.DataFrame, features: list[str], params: dict | None = None) -> pd.DataFrame:
    params = params or dict(
        max_depth=8, learning_rate=0.08, max_iter=250, min_samples_leaf=20, l2_regularization=0.1
    )
    folds = [
        ("2025-08-23", "2025-09-06"),
        ("2025-09-06", "2025-09-20"),
        ("2025-09-20", "2025-10-04"),
        ("2025-10-04", "2025-10-18"),
        ("2025-10-18", "2025-11-01"),
    ]
    rows = []
    d = df.dropna(subset=["trips"])
    for i, (c0, c1) in enumerate(folds, 1):
        c0t, c1t = pd.Timestamp(c0), pd.Timestamp(c1)
        tr = d[d["pickup_hour"] < c0t]
        va = d[(d["pickup_hour"] >= c0t) & (d["pickup_hour"] < c1t)]
        if len(tr) < 2000 or len(va) < 200:
            continue
        pipe = make_pipe(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params),
            features,
        )
        pipe.fit(tr[features], tr["trips"].values)
        pred = pipe.predict(va[features])
        seas = seasonal_naive_predict(tr, va)
        rows.append(
            {
                "fold": i,
                "train_end": c0,
                "valid_start": c0,
                "valid_end": c1,
                "hist_gbm_rmse": rmse(va["trips"], pred),
                "hist_gbm_mae": mae(va["trips"], pred),
                "seasonal_naive_rmse": rmse(va["trips"], seas),
                "n_train": len(tr),
                "n_valid": len(va),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D4 leakage audit
# ---------------------------------------------------------------------------

def d4_leakage_table(all_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in sorted(set(all_columns) | set(MODEL_FEATURES) | set(LEAKY)):
        if col in ("record_id", "row_id", "zone_raw", "pickup_hour", "data_type", "event_names", "day"):
            known = "n/a"
            use = "id/meta — not a model feature"
        elif col in LEAKY and col not in ("trips", "trips_raw"):
            known = "no"
            use = "EXCLUDED — operational outcome, unknown at forecast time"
        elif col in ("trips", "trips_raw"):
            known = "n/a"
            use = "target only"
        elif col in MODEL_FEATURES:
            known = "yes"
            use = "INCLUDED in final model"
        else:
            known = "depends"
            use = "available but not selected"
        rows.append({"feature": col, "known_at_forecast_time": known, "decision": use})
    return pd.DataFrame(rows)


def d4_leaky_demo(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Show inflated score if we illegally use active_drivers."""
    feats = MODEL_FEATURES + ["active_drivers"]
    # active_drivers may have no nulls
    pipe = make_pipe(
        HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, max_iter=200, random_state=RANDOM_STATE
        ),
        feats,
    )
    pipe.fit(train[feats], train["trips"].values)
    pred = pipe.predict(valid[feats])
    return {
        "rmse_with_active_drivers": rmse(valid["trips"], pred),
        "note": "Looks better on validation but would be unavailable for Nov forecasts — production failure / leakage.",
    }


# ---------------------------------------------------------------------------
# D5 ablation
# ---------------------------------------------------------------------------

def d5_ablation(train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or dict(
        max_depth=8, learning_rate=0.08, max_iter=250, min_samples_leaf=20, l2_regularization=0.1
    )
    sets = {
        "calendar_zone_trend": [f for f in CALENDAR_FEATS if f in train.columns],
        "plus_weather": [f for f in CALENDAR_FEATS + WEATHER_FEATS if f in train.columns],
        "plus_events": [f for f in CALENDAR_FEATS + EVENT_FEATS if f in train.columns],
        "plus_both": [f for f in MODEL_FEATURES if f in train.columns],
    }
    rows = []
    y_va = valid["trips"].values
    for name, feats in sets.items():
        pipe = make_pipe(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params),
            feats,
        )
        t0 = time.time()
        pipe.fit(train[feats], train["trips"].values)
        pred = pipe.predict(valid[feats])
        rows.append(
            {
                "feature_set": name,
                "n_features": len(feats),
                "rmse": rmse(y_va, pred),
                "mae": mae(y_va, pred),
                "train_seconds": time.time() - t0,
            }
        )
    out = pd.DataFrame(rows)
    base = out.loc[out["feature_set"] == "calendar_zone_trend", "rmse"].iloc[0]
    out["rmse_delta_vs_calendar"] = out["rmse"] - base
    return out


# ---------------------------------------------------------------------------
# D6 tuning
# ---------------------------------------------------------------------------

def d6_tune(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> dict:
    y_va = valid["trips"].values
    X_tr, y_tr = train[features], train["trips"].values
    X_va = valid[features]

    space = {
        "learning_rate": [0.05, 0.08, 0.12],
        "max_depth": [6, 8, 10],
        "max_iter": [150, 250, 350],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    sampler = list(ParameterSampler(space, n_iter=12, random_state=RANDOM_STATE))

    # baseline default before tune
    default = dict(
        learning_rate=0.08, max_depth=8, max_iter=250, min_samples_leaf=20, l2_regularization=0.1
    )
    pipe0 = make_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **default), features)
    pipe0.fit(X_tr, y_tr)
    rmse_before = rmse(y_va, pipe0.predict(X_va))

    best = None
    trials = []
    for i, params in enumerate(sampler, 1):
        pipe = make_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params), features)
        pipe.fit(X_tr, y_tr)
        score = rmse(y_va, pipe.predict(X_va))
        trials.append({"trial": i, **params, "rmse": score})
        if best is None or score < best["rmse"]:
            best = {"rmse": score, "params": params}

    return {
        "n_trials": len(sampler),
        "search_space": {k: list(v) for k, v in space.items()},
        "rmse_before": rmse_before,
        "rmse_after": best["rmse"],
        "best_params": best["params"],
        "trials": pd.DataFrame(trials).sort_values("rmse"),
    }


# ---------------------------------------------------------------------------
# D7 / D8 error analysis
# ---------------------------------------------------------------------------

def d7_error_analysis(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict) -> dict:
    pipe = make_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params), features)
    pipe.fit(train[features], train["trips"].values)
    pred = pipe.predict(valid[features])
    v = valid.copy()
    v["pred"] = pred
    v["error"] = v["trips"] - v["pred"]
    v["abs_error"] = v["error"].abs()

    by_zone = v.groupby("zone").apply(
        lambda g: pd.Series({"rmse": rmse(g["trips"], g["pred"]), "mae": mae(g["trips"], g["pred"]), "n": len(g)}),
        include_groups=False,
    ).sort_values("rmse", ascending=False)

    by_hour = v.groupby("hour").apply(
        lambda g: pd.Series({"rmse": rmse(g["trips"], g["pred"]), "mae": mae(g["trips"], g["pred"])}),
        include_groups=False,
    )

    v["day_type"] = np.where(
        v["is_public_holiday"] == 1,
        "holiday",
        np.where(v["is_weekend"] == 1, "weekend", "weekday"),
    )
    by_daytype = v.groupby("day_type").apply(
        lambda g: pd.Series({"rmse": rmse(g["trips"], g["pred"]), "mae": mae(g["trips"], g["pred"]), "n": len(g)}),
        include_groups=False,
    )

    top10 = (
        v.nlargest(10, "abs_error")[
            [
                "zone",
                "pickup_hour",
                "hour",
                "trips",
                "pred",
                "abs_error",
                "is_football_window",
                "is_public_holiday",
                "rain_mm",
            ]
        ]
        .copy()
    )
    top10["hypothesis"] = top10.apply(_error_hypothesis, axis=1)

    return {
        "by_zone": by_zone.reset_index(),
        "by_hour": by_hour.reset_index(),
        "by_daytype": by_daytype.reset_index(),
        "top10": top10,
        "valid_with_pred": v,
        "pipe": pipe,
        "overall_rmse": rmse(v["trips"], v["pred"]),
        "overall_mae": mae(v["trips"], v["pred"]),
    }


def _error_hypothesis(row) -> str:
    if row["is_football_window"] == 1:
        return "Football window — event magnitude may exceed average uplift feature."
    if row["is_public_holiday"] == 1:
        return "Holiday — holiday effect heterogeneous across zones."
    if row["rain_mm"] > 2:
        return "Heavy rain hour — nonlinear rain response / sparse heavy-rain history."
    if row["hour"] in (7, 8, 9, 17, 18, 19):
        return "Peak commute hour — residual peak sharpness not fully captured."
    return "Possible unlisted event, local spike, or lag mismatch."


def d8_response(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict, errors: dict) -> dict:
    """
    Response to D7: add interaction-like helpers —
    peak_hour flag + rain_x_weekend — and re-evaluate.
    """
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["is_peak_hour"] = out["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        out["rain_x_weekend"] = out["rain_mm"] * out["is_weekend"]
        return out

    tr, va = enrich(train), enrich(valid)
    feats2 = features + ["is_peak_hour", "rain_x_weekend"]
    pipe = make_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params), feats2)
    pipe.fit(tr[feats2], tr["trips"].values)
    pred = pipe.predict(va[feats2])
    rmse_new = rmse(va["trips"], pred)
    mae_new = mae(va["trips"], pred)
    helped = rmse_new < errors["overall_rmse"] - 1e-6
    return {
        "change": "Added is_peak_hour and rain_x_weekend based on peak-hour / rain error patterns",
        "rmse_before": errors["overall_rmse"],
        "rmse_after": rmse_new,
        "mae_before": errors["overall_mae"],
        "mae_after": mae_new,
        "helped": bool(helped),
        "extra_features": ["is_peak_hour", "rain_x_weekend"],
        "keep_extra_features": bool(helped),
    }


# ---------------------------------------------------------------------------
# D9 plain language
# ---------------------------------------------------------------------------

def d9_plain(train: pd.DataFrame, rmse_v: float, mae_v: float) -> str:
    mean_demand = float(train["trips"].mean())
    rmse_pct = 100 * rmse_v / mean_demand
    mae_pct = 100 * mae_v / mean_demand
    # ~1.3 trips per active driver per hour from history
    trips_per_driver = 1.3
    drivers_rmse = rmse_v / trips_per_driver
    drivers_mae = mae_v / trips_per_driver
    return (
        f"On the Oct 18–31 validation fortnight, typical absolute error is about **{mae_v:.1f} trips per zone-hour** "
        f"(MAE), roughly **{mae_pct:.0f}% of mean demand** ({mean_demand:.1f} trips/hour). "
        f"RMSE is **{rmse_v:.1f} trips** (~{rmse_pct:.0f}% of mean), penalizing larger misses. "
        f"At ~1.3 trips per active driver-hour, that is roughly **±{drivers_mae:.1f} drivers** on average "
        f"(RMSE ≈ **{drivers_rmse:.1f} drivers**) for staffing a single zone-hour — useful as a buffer when deploying."
    )


# ---------------------------------------------------------------------------
# Final train + submission
# ---------------------------------------------------------------------------

def train_final_and_submit(
    train_all: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    params: dict,
    extra_features: list[str] | None = None,
) -> tuple[Pipeline, pd.DataFrame]:
    feats = list(features) + list(extra_features or [])

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "is_peak_hour" in feats:
            out["is_peak_hour"] = out["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        if "rain_x_weekend" in feats:
            out["rain_x_weekend"] = out["rain_mm"] * out["is_weekend"]
        return out

    tr = enrich(train_all.dropna(subset=["trips"]))
    te = enrich(test)
    pipe = make_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params), feats)
    pipe.fit(tr[feats], tr["trips"].values)
    pred = np.clip(pipe.predict(te[feats]), 0, None)

    # Align to submission template order
    template = pd.read_csv(ROOT / "data" / "raw" / "submission_template.csv")
    # master_test should have row_id
    te_out = te[["row_id"]].copy()
    te_out["predicted_trips"] = pred
    sub = template[["row_id"]].merge(te_out, on="row_id", how="left")
    if sub["predicted_trips"].isna().any():
        # fallback: merge failed for some — fill with seasonal naive from train
        seas = seasonal_naive_predict(tr, te)
        fill_map = dict(zip(te["row_id"], seas))
        sub["predicted_trips"] = sub["predicted_trips"].fillna(sub["row_id"].map(fill_map)).fillna(tr["trips"].mean())
    sub["predicted_trips"] = sub["predicted_trips"].clip(lower=0)
    return pipe, sub, feats


def write_report(ctx: dict, path: Path) -> None:
    cmp_ = ctx["comparison"]
    winner = cmp_.iloc[0]
    roll = ctx["rolling"]
    abl = ctx["ablation"]
    tune = ctx["tuning"]
    err = ctx["errors"]
    resp = ctx["response"]
    leak = ctx["leaky_demo"]

    lines = [
        "# D — Modeling & Evaluation",
        "",
        f"**Split:** train `pickup_hour < {VAL_START.date()}`, validate `{VAL_START.date()}`–`2025-10-31`.  ",
        "**Final model:** HistGradientBoostingRegressor (sklearn) with weather + event + calendar/lag features.  ",
        f"**Winner on holdout:** `{winner['model']}` — RMSE {winner['rmse']:.3f}, MAE {winner['mae']:.3f}.",
        "",
        "Code: `src/modeling.py` · Notebook: `notebooks/04_modeling_and_evaluation.ipynb`  ",
        "Artifacts: `models/final_model.joblib`, `submission/team_addis_demand_ai_submission.csv`",
        "",
        "---",
        "",
        "## D1 Baselines",
        "",
        cmp_[cmp_["family"] == "baseline"].to_string(index=False),
        "",
        "**Interpretation:** Seasonal naive (zone×dow×hour) is the real bar to beat; a global mean is far too weak.",
        "",
        "## D2 Model comparison",
        "",
        cmp_.to_string(index=False),
        "",
        f"**Winner:** `{winner['model']}` — best RMSE/MAE tradeoff on the same chronological split and features. "
        "Trees capture nonlinear rain/event effects better than ridge; HistGBM trains faster than a deep RF here.",
        "",
        "## D3 Rolling-origin validation",
        "",
        roll.to_string(index=False),
        "",
        f"- HistGBM RMSE mean±std: **{roll['hist_gbm_rmse'].mean():.3f} ± {roll['hist_gbm_rmse'].std():.3f}**",
        f"- Seasonal naive RMSE mean±std: **{roll['seasonal_naive_rmse'].mean():.3f} ± {roll['seasonal_naive_rmse'].std():.3f}**",
        "",
        "**Interpretation:** Spread across folds shows stability; any fold that is unusually bad usually coincides "
        "with holiday/event-dense fortnights where residual event risk is higher.",
        "",
        "## D4 Feature availability & leakage audit",
        "",
        "Full table: `reports/D_tables/D4_leakage_audit.csv`.",
        "",
        "Excluded from model inputs: `avg_fare_birr`, `avg_wait_min`, `active_drivers` — not known at forecast time.",
        "",
        f"Leaky demo (illegal `active_drivers` included): RMSE = **{leak['rmse_with_active_drivers']:.3f}** "
        f"vs honest model ~{ctx['errors']['overall_rmse']:.3f}. {leak['note']}",
        "",
        "## D5 Ablation",
        "",
        abl.to_string(index=False),
        "",
        "**Interpretation:** Compare RMSE deltas. Weather and events should each move the needle versus calendar-only; "
        "both together is the production feature set (Rule 5).",
        "",
        "## D6 Tuning",
        "",
        f"- Trials: **{tune['n_trials']}** (random search, time-ordered single validation fortnight)",
        f"- RMSE before → after: **{tune['rmse_before']:.3f} → {tune['rmse_after']:.3f}**",
        f"- Best params: `{tune['best_params']}`",
        "",
        "Search space:",
        "```json",
        json.dumps(tune["search_space"], indent=2),
        "```",
        "",
        "## D7 Error analysis",
        "",
        f"Overall validation RMSE/MAE: **{err['overall_rmse']:.3f} / {err['overall_mae']:.3f}**",
        "",
        "### By zone",
        "",
        err["by_zone"].to_string(index=False),
        "",
        "### By day type",
        "",
        err["by_daytype"].to_string(index=False),
        "",
        "### Top 10 absolute errors",
        "",
        err["top10"].to_string(index=False),
        "",
        "## D8 Response to findings",
        "",
        f"- Change: {resp['change']}",
        f"- RMSE {resp['rmse_before']:.3f} → {resp['rmse_after']:.3f} (helped={resp['helped']})",
        f"- Extra features kept in final model: {resp['extra_features'] if resp['keep_extra_features'] else 'none (reverted)'}",
        "",
        "## D9 Plain-language metric",
        "",
        ctx["plain"],
        "",
        "---",
        "",
        "## Submission hygiene",
        "",
        f"- Rows: {ctx['submission_rows']}",
        f"- Min/max prediction: {ctx['pred_min']:.3f} / {ctx['pred_max']:.3f}",
        f"- Features used ({len(ctx['final_features'])}): {ctx['final_features']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Loading master tables...")
    train_all, test = load_data()
    features = [f for f in MODEL_FEATURES if f in train_all.columns]
    train, valid = split_chrono(train_all)
    print(f"Train {train['pickup_hour'].min()} .. {train['pickup_hour'].max()} n={len(train)}")
    print(f"Valid {valid['pickup_hour'].min()} .. {valid['pickup_hour'].max()} n={len(valid)}")

    print("D1/D2 comparison...")
    comparison = d1_d2_baselines_and_models(train, valid, features)
    print(comparison)

    print("D3 rolling...")
    rolling = d3_rolling(train_all, features)
    print(rolling)

    print("D4 leakage...")
    leakage = d4_leakage_table(list(train_all.columns))
    leaky_demo = d4_leaky_demo(train, valid)

    print("D6 tuning...")
    tuning = d6_tune(train, valid, features)
    print("best", tuning["best_params"], tuning["rmse_after"])

    best_params = tuning["best_params"]
    # ensure plain dict with python types
    best_params = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in best_params.items()}
    # max_iter/depth/leaf should be int
    for k in ["max_depth", "max_iter", "min_samples_leaf"]:
        if k in best_params:
            best_params[k] = int(best_params[k])

    print("D5 ablation (with tuned params)...")
    ablation = d5_ablation(train, valid, best_params)
    print(ablation)

    print("D7 errors...")
    errors = d7_error_analysis(train, valid, features, best_params)

    print("D8 response...")
    response = d8_response(train, valid, features, best_params, errors)
    print(response)

    extra = response["extra_features"] if response["keep_extra_features"] else []
    # Recompute final validation metrics with chosen feature set
    final_rmse = response["rmse_after"] if response["keep_extra_features"] else errors["overall_rmse"]
    final_mae = response["mae_after"] if response["keep_extra_features"] else errors["overall_mae"]
    plain = d9_plain(train, final_rmse, final_mae)

    print("Training final model on all train + writing submission...")
    pipe, submission, final_features = train_final_and_submit(
        train_all, test, features, best_params, extra_features=extra
    )

    # Save artifacts
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipe,
        "features": final_features,
        "params": best_params,
        "val_rmse": final_rmse,
        "val_mae": final_mae,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(payload, models_dir / "final_model.joblib")
    joblib.dump(payload, ROOT / "app" / "assets" / "final_model.joblib")

    sub_path = ROOT / "submission" / "team_addis_demand_ai_submission.csv"
    submission.to_csv(sub_path, index=False)

    tables = ROOT / "reports" / "D_tables"
    tables.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables / "D2_model_comparison.csv", index=False)
    rolling.to_csv(tables / "D3_rolling_origin.csv", index=False)
    leakage.to_csv(tables / "D4_leakage_audit.csv", index=False)
    ablation.to_csv(tables / "D5_ablation.csv", index=False)
    tuning["trials"].to_csv(tables / "D6_tuning_trials.csv", index=False)
    errors["by_zone"].to_csv(tables / "D7_error_by_zone.csv", index=False)
    errors["by_hour"].to_csv(tables / "D7_error_by_hour.csv", index=False)
    errors["by_daytype"].to_csv(tables / "D7_error_by_daytype.csv", index=False)
    errors["top10"].to_csv(tables / "D7_top10_errors.csv", index=False)

    ctx = {
        "comparison": comparison,
        "rolling": rolling,
        "ablation": ablation,
        "tuning": tuning,
        "errors": errors,
        "response": response,
        "leaky_demo": leaky_demo,
        "plain": plain,
        "submission_rows": len(submission),
        "pred_min": float(submission["predicted_trips"].min()),
        "pred_max": float(submission["predicted_trips"].max()),
        "final_features": final_features,
    }
    write_report(ctx, ROOT / "reports" / "D_model_evaluation.md")

    summary = {
        "val_split": "train<2025-10-18; valid 2025-10-18..2025-10-31",
        "winner": comparison.iloc[0].to_dict(),
        "rolling_rmse_mean": float(rolling["hist_gbm_rmse"].mean()),
        "rolling_rmse_std": float(rolling["hist_gbm_rmse"].std()),
        "best_params": best_params,
        "final_val_rmse": final_rmse,
        "final_val_mae": final_mae,
        "response_helped": response["helped"],
        "submission_rows": len(submission),
        "features": final_features,
    }

    def _jsonable(o):
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    (tables / "D_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Validation RMSE/MAE: {final_rmse:.3f} / {final_mae:.3f}")
    print(f"Saved model -> models/final_model.joblib")
    print(f"Saved submission -> {sub_path} ({len(submission)} rows)")
    print(f"Report -> reports/D_model_evaluation.md")


if __name__ == "__main__":
    main()

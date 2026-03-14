"""
Heat Pump Grid Load Prediction — Task 3
Predicts monthly average x2 for 600 heat pumps, May-Oct 2025.
Trained on Oct 2024 – Apr 2025 (winter → summer OOD extrapolation).

x2 is min-max normalized to [0,1].
Monthly means: Oct ~0.076, Jan ~0.215, Feb ~0.235, Mar ~0.138, Apr ~0.077.
Summer: below April, approaching DHW-only baseload (~0.02-0.04).

Architecture:
  A. Per-device physics model: x2 = HTC * HDD + DHW_baseload  (primary, 70%)
  B. Global monthly LightGBM on monthly aggregates (secondary, 30%)
  Final = 0.7 * physics + 0.3 * lgb with post-processing guardrails.

Key insights used:
  - validation/test periods have t1/x1/x3/deviceType — use them for HDD proxy
  - October 2025 ≈ October 2024 (same season)
  - Summer predictions floored at per-device DHW baseload
  - No pandas required
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "out"
WEATHER_CACHE = DATA_DIR / "weather_cache.parquet"
WEATHER_PARTIAL = DATA_DIR / "weather_partial.parquet"
SUBMISSION_FILE = OUT_DIR / "submission.csv"
BATCH_SIZE = 50


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and clean data.csv + devices.csv using Polars lazy mode."""
    print("Loading devices.csv …")
    devices = pl.read_csv(DATA_DIR / "devices.csv")

    print("Scanning data.csv (lazy, 10 GB) …")
    needed_cols = [
        "deviceId",
        "timedate",
        "period",
        "t1",
        "t2",
        "t7",
        "x1",
        "x2",
        "x3",
        "deviceType",
    ]

    lazy = pl.scan_csv(DATA_DIR / "data.csv", try_parse_dates=False)
    lazy = lazy.select(needed_cols).with_columns(
        pl.col("timedate")
        .str.replace(r" UTC$", "")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timedate")
    )

    print("Collecting … (this may take a few minutes)")
    df = lazy.collect()
    print(f"Loaded {len(df):,} rows")

    # Remove negative x2 (sensor errors)
    df = df.filter(pl.col("x2").is_null() | (pl.col("x2") >= 0))

    # Winsorize x2 at 99.5th percentile per device (train only)
    p995 = (
        df.filter(pl.col("period") == "train")
        .group_by("deviceId")
        .agg(pl.col("x2").quantile(0.995).alias("x2_cap"))
    )
    df = df.join(p995, on="deviceId", how="left")
    df = df.with_columns(
        pl.when(pl.col("x2").is_not_null() & pl.col("x2_cap").is_not_null())
        .then(pl.col("x2").clip(upper_bound=pl.col("x2_cap")))
        .otherwise(pl.col("x2"))
        .alias("x2")
    ).drop("x2_cap")

    # Remove stuck meters: x2 constant for > 24 consecutive readings (2 hrs)
    df = df.sort(["deviceId", "timedate"])
    df = df.with_columns(
        (pl.col("x2") != pl.col("x2").shift(1).over("deviceId"))
        .fill_null(True)
        .cast(pl.Int32)
        .alias("x2_changed")
    )
    df = df.with_columns(pl.col("x2_changed").cum_sum().over("deviceId").alias("streak_id"))
    streak_lengths = df.group_by(["deviceId", "streak_id"]).agg(pl.len().alias("streak_len"))
    df = df.join(streak_lengths, on=["deviceId", "streak_id"], how="left")
    df = df.filter((pl.col("streak_len") <= 24) | (pl.col("period") != "train")).drop(
        ["x2_changed", "streak_id", "streak_len"]
    )

    print(f"After cleaning: {len(df):,} rows")
    return df, devices


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEATHER FETCHING (Open-Meteo ERA5) — optional, falls back to t1
# ─────────────────────────────────────────────────────────────────────────────


def fetch_weather(devices: pl.DataFrame) -> pl.DataFrame | None:
    """Fetch hourly weather for each unique lat/lon using batched requests. Cached to parquet."""
    if WEATHER_CACHE.exists():
        print("Loading cached weather …")
        return pl.read_parquet(WEATHER_CACHE)

    print("Fetching weather from Open-Meteo …")
    locations = (
        devices.with_columns([pl.col("latitude").round(1), pl.col("longitude").round(1)])
        .select(["latitude", "longitude"])
        .unique()
        .sort(["latitude", "longitude"])
    )
    all_locs = locations.to_dicts()
    n_total = len(all_locs)
    print(
        f"  {n_total} unique locations → {math.ceil(n_total / BATCH_SIZE)} batches of {BATCH_SIZE}"
    )

    already_done: set[tuple] = set()
    partial_frames: list[pl.DataFrame] = []
    if WEATHER_PARTIAL.exists():
        partial = pl.read_parquet(WEATHER_PARTIAL)
        partial_frames.append(partial)
        for row in partial.select(["latitude", "longitude"]).unique().iter_rows():
            already_done.add(row)
        print(f"  Resuming: {len(already_done)} locations already cached")

    remaining = [loc for loc in all_locs if (loc["latitude"], loc["longitude"]) not in already_done]
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    for batch_i, batch in enumerate(batches):
        lats = [loc["latitude"] for loc in batch]
        lons = [loc["longitude"] for loc in batch]
        print(f"  Batch {batch_i + 1}/{len(batches)}: {len(batch)} locations …")

        params = {
            "latitude": ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
            "start_date": "2024-10-01",
            "end_date": "2025-10-31",
            "hourly": "temperature_2m,shortwave_radiation,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC",
        }

        resp = None
        for attempt in range(10):
            try:
                resp = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=params,
                    timeout=120,
                )
            except requests.RequestException as e:
                print(f"    Request error: {e}, retry {attempt + 1}/10 …")
                time.sleep(min(10 * (2**attempt), 120))
                continue

            if resp.status_code == 200:
                break
            wait = min(10 * (2**attempt), 120)
            print(f"    HTTP {resp.status_code}, waiting {wait}s, retry {attempt + 1}/10 …")
            time.sleep(wait)

        if resp is None or resp.status_code != 200:
            print(f"  WARNING: batch {batch_i + 1} failed after retries, skipping")
            continue

        body = resp.json()
        location_results = body if isinstance(body, list) else [body]

        records: list[dict] = []
        for loc_data, lat, lon in zip(location_results, lats, lons):
            hourly = loc_data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [None] * len(times))
            solar = hourly.get("shortwave_radiation", [None] * len(times))
            humid = hourly.get("relative_humidity_2m", [None] * len(times))
            wind = hourly.get("wind_speed_10m", [None] * len(times))
            for t, te, so, hu, wi in zip(times, temps, solar, humid, wind):
                records.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "hour": t,
                        "temp_2m": te,
                        "solar_rad": so,
                        "humidity": hu,
                        "wind_speed": wi,
                    }
                )

        batch_df = pl.DataFrame(records).with_columns(
            pl.col("hour").str.to_datetime("%Y-%m-%dT%H:%M", strict=False)
        )
        partial_frames.append(batch_df)
        combined = pl.concat(partial_frames)
        DATA_DIR.mkdir(exist_ok=True)
        combined.write_parquet(WEATHER_PARTIAL)
        time.sleep(5)

    if not partial_frames:
        print("  WARNING: No weather data fetched — will use t1-based fallback")
        return None

    weather = pl.concat(partial_frames)
    weather.write_parquet(WEATHER_CACHE)
    if WEATHER_PARTIAL.exists():
        WEATHER_PARTIAL.unlink()
    print(f"  Cached {len(weather):,} weather rows")
    return weather


def join_weather(df: pl.DataFrame, devices: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Join weather to device telemetry by nearest lat/lon and hour."""
    devices_rounded = devices.with_columns(
        [
            pl.col("latitude").round(1),
            pl.col("longitude").round(1),
        ]
    )
    df = df.join(devices_rounded, on="deviceId", how="left")
    df = df.with_columns(pl.col("timedate").dt.truncate("1h").alias("hour"))
    df = df.join(weather, on=["latitude", "longitude", "hour"], how="left")
    return df.drop("hour")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING + DAILY AGGREGATION (ALL periods)
# ─────────────────────────────────────────────────────────────────────────────


def day_length(lat_deg: float, doy: int) -> float:
    """Astronomical day length in hours for a given latitude and day-of-year."""
    lat_rad = math.radians(lat_deg)
    decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))
    cos_ha = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(decl)))
    return (2 / 15) * math.degrees(math.acos(cos_ha))


def add_physics_features(df: pl.DataFrame, has_weather: bool) -> pl.DataFrame:
    """Add thermodynamic features. Uses t1-derived approximate temp for ALL periods."""
    T_base15 = 15.0
    T_base18 = 18.0
    T_cool22 = 22.0
    T_indoor = 20.0

    if has_weather:
        temp_col = pl.col("temp_2m")
    else:
        # t1 is min-max normalized outdoor temp, Polish range ~-15°C to +35°C → 50°C span
        # t1=0 ≈ -15°C, t1=1 ≈ +35°C → temp ≈ t1 * 50 - 15
        temp_col = pl.col("t1") * 50.0 - 15.0
        df = df.with_columns(
            [
                (pl.col("t1") * 50.0 - 15.0).alias("temp_2m"),
                pl.lit(None).cast(pl.Float64).alias("solar_rad"),
                pl.lit(None).cast(pl.Float64).alias("humidity"),
                pl.lit(None).cast(pl.Float64).alias("wind_speed"),
            ]
        )

    df = df.with_columns(
        [
            (pl.lit(T_base15) - temp_col).clip(lower_bound=0).alias("hdd15_5min"),
            (pl.lit(T_base18) - temp_col).clip(lower_bound=0).alias("hdd18_5min"),
            (temp_col - pl.lit(T_cool22)).clip(lower_bound=0).alias("cdd22_5min"),
            ((pl.lit(T_indoor) - temp_col) / pl.lit(T_indoor)).clip(lower_bound=0).alias("inv_cop"),
            pl.col("timedate").dt.date().alias("date"),
            pl.col("timedate").dt.year().alias("year"),
            pl.col("timedate").dt.month().alias("month"),
            pl.col("timedate").dt.ordinal_day().alias("doy"),
        ]
    )
    return df


def daily_aggregate(df: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Compress 5-minute rows to one row per (deviceId, date) for ALL periods."""
    df = df.with_columns((pl.col("hdd15_5min") * pl.col("inv_cop")).alias("load_proxy_5min"))

    if "latitude" not in df.columns:
        df = df.join(devices.select(["deviceId", "latitude"]), on="deviceId", how="left")

    daily = df.group_by(["deviceId", "date"]).agg(
        [
            pl.col("x2").mean().alias("x2_mean"),
            pl.col("period").first().alias("period"),
            pl.col("year").first().alias("year"),
            pl.col("month").first().alias("month"),
            pl.col("doy").first().alias("doy"),
            pl.col("latitude").first().alias("latitude"),
            pl.col("t1").mean().alias("t1_mean"),
            pl.col("t1").max().alias("t1_max"),
            pl.col("t1").min().alias("t1_min"),
            pl.col("t2").mean().alias("t2_mean"),
            pl.col("t7").mean().alias("t7_mean"),
            pl.col("x1").mean().alias("x1_mean"),
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("temp_2m").mean().alias("temp_mean"),
            pl.col("temp_2m").max().alias("temp_max"),
            pl.col("temp_2m").min().alias("temp_min"),
            pl.col("solar_rad").sum().alias("solar_sum"),
            pl.col("humidity").mean().alias("humidity_mean"),
            pl.col("wind_speed").mean().alias("wind_mean"),
            (pl.col("hdd15_5min").sum() / 288).alias("HDD_15"),
            (pl.col("hdd18_5min").sum() / 288).alias("HDD_18"),
            (pl.col("cdd22_5min").sum() / 288).alias("CDD_22"),
            pl.col("inv_cop").mean().alias("inv_cop"),
            (pl.col("load_proxy_5min").sum() / 288).alias("load_proxy"),
        ]
    )

    daily = daily.with_columns(
        pl.struct(["latitude", "doy"])
        .map_elements(
            lambda s: day_length(s["latitude"], s["doy"]),
            return_dtype=pl.Float64,
        )
        .alias("day_length")
    )

    daily = daily.sort(["deviceId", "date"])
    daily = daily.with_columns((pl.col("temp_mean") < 15.0).cast(pl.Int32).alias("cold_day"))
    daily = daily.with_columns(
        (
            pl.col("cold_day")
            + pl.col("cold_day").shift(1).over("deviceId").fill_null(0)
            + pl.col("cold_day").shift(2).over("deviceId").fill_null(0)
        ).alias("cold_streak")
    )
    daily = daily.with_columns(
        (pl.col("cold_streak") >= 3).cast(pl.Int32).alias("is_heating_season")
    ).drop(["cold_day", "cold_streak"])

    # Upweight warm training days 3× — closest proxy to summer conditions
    daily = daily.with_columns(
        pl.when((pl.col("period") == "train") & (pl.col("temp_mean") > 15.0))
        .then(3.0)
        .otherwise(1.0)
        .alias("sample_weight")
    )

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. MONTHLY AGGREGATION (ALL periods)
# ─────────────────────────────────────────────────────────────────────────────


def build_monthly_features(daily: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily → monthly per device. ALL periods included."""
    monthly = daily.group_by(["deviceId", "year", "month"]).agg(
        [
            pl.col("x2_mean").mean().alias("x2_monthly"),
            pl.col("period").first().alias("period"),
            pl.col("temp_mean").mean().alias("temp_mean_monthly"),
            pl.col("temp_mean").min().alias("temp_min_monthly"),
            pl.col("HDD_15").sum().alias("HDD_15_monthly"),
            pl.col("HDD_18").sum().alias("HDD_18_monthly"),
            pl.col("CDD_22").sum().alias("CDD_22_monthly"),
            pl.col("load_proxy").sum().alias("load_proxy_monthly"),
            pl.col("solar_sum").sum().alias("solar_monthly"),
            pl.col("day_length").mean().alias("day_length_monthly"),
            pl.col("inv_cop").mean().alias("inv_cop_monthly"),
            pl.col("t1_mean").mean().alias("t1_mean_monthly"),
            pl.col("t1_min").min().alias("t1_min_monthly"),
            pl.col("x1_mean").mean().alias("x1_mean_monthly"),
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("is_heating_season").mean().alias("frac_heating_season"),
            pl.col("sample_weight").mean().alias("sample_weight"),
        ]
    )
    return monthly.sort(["deviceId", "year", "month"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL A — PER-DEVICE PHYSICS LINEAR REGRESSION (PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────


def fit_physics_model(monthly: pl.DataFrame) -> pl.DataFrame:
    """
    Per-device OLS: x2_monthly = HTC * HDD_15_monthly + DHW_baseload
    Returns deviceId, htc, dhw_baseload, x2_train_mean, apr_mean, oct24_mean.
    Uses monthly data so we have 7 clean training points per device.
    """
    train = monthly.filter(pl.col("period") == "train")

    # Global fallback values
    global_dhw = float(train.select(pl.col("x2_monthly").quantile(0.10)).item() or 0.02)
    global_mean = float(train.select(pl.col("x2_monthly").mean()).item() or 0.10)

    results = []
    for device_grp in train.partition_by("deviceId"):
        device_id = device_grp["deviceId"][0]
        hdd = device_grp["HDD_15_monthly"].to_numpy()
        x2 = device_grp["x2_monthly"].to_numpy()
        months = device_grp["month"].to_numpy()

        mask = np.isfinite(x2) & np.isfinite(hdd)
        x2_train_mean = float(np.nanmean(x2)) if mask.any() else global_mean

        # April mean (best proxy for early summer)
        apr_mask = months == 4
        apr_mean = (
            float(np.mean(x2[apr_mask & mask])) if (apr_mask & mask).any() else x2_train_mean * 0.4
        )

        # October 2024 mean (anchor for October 2025)
        oct_mask = months == 10
        oct24_mean = float(np.mean(x2[oct_mask & mask])) if (oct_mask & mask).any() else apr_mean

        if mask.sum() >= 3:
            A = np.column_stack([hdd[mask], np.ones(mask.sum())])
            coef, _, _, _ = np.linalg.lstsq(A, x2[mask], rcond=None)
            htc = max(float(coef[0]), 0.0)
            dhw_baseload = max(float(coef[1]), 0.0)
            # Sanity: cap DHW at April mean (physics: summer can't exceed spring)
            dhw_baseload = min(dhw_baseload, apr_mean)
            # Floor: DHW must be at least 1% of normalized x2
            dhw_baseload = max(dhw_baseload, 0.005)
        else:
            htc = 0.0
            dhw_baseload = max(x2_train_mean * 0.3, global_dhw)

        results.append(
            {
                "deviceId": device_id,
                "htc": htc,
                "dhw_baseload": dhw_baseload,
                "x2_train_mean": x2_train_mean,
                "apr_mean": apr_mean,
                "oct24_mean": oct24_mean,
            }
        )

    return pl.DataFrame(results)


def predict_physics(monthly_pred: pl.DataFrame, physics_params: pl.DataFrame) -> pl.DataFrame:
    """
    Apply per-device physics model to prediction months.
    Uses actual HDD_15_monthly from validation/test period (computed from t1).
    prediction = clip(HTC * HDD_actual + DHW_baseload, DHW_baseload * 0.8, apr_mean * 1.05)
    """
    pred = monthly_pred.join(physics_params, on="deviceId", how="left")

    pred = pred.with_columns(
        [
            (
                pl.col("htc").fill_null(0.0) * pl.col("HDD_15_monthly").fill_null(0.0)
                + pl.col("dhw_baseload").fill_null(0.02)
            ).alias("phys_raw"),
        ]
    )

    # Oct 2025 anchor: blend 50/50 with Oct 2024 actual
    pred = pred.with_columns(
        pl.when(pl.col("month") == 10)
        .then(pl.col("phys_raw") * 0.5 + pl.col("oct24_mean").fill_null(pl.col("phys_raw")) * 0.5)
        .otherwise(pl.col("phys_raw"))
        .alias("phys_pred")
    )

    # Floor: at least 80% of DHW baseload (pump never goes fully off)
    pred = pred.with_columns(
        pl.col("phys_pred")
        .clip(lower_bound=pl.col("dhw_baseload").fill_null(0.005) * 0.8)
        .alias("phys_pred")
    )

    # Summer cap: Jun-Aug must be <= April level (no heating, DHW only)
    pred = pred.with_columns(
        pl.when(pl.col("month").is_in([6, 7, 8]))
        .then(
            pl.col("phys_pred").clip(upper_bound=pl.col("apr_mean").fill_null(pl.col("phys_pred")))
        )
        .otherwise(pl.col("phys_pred"))
        .alias("phys_pred")
    )

    return pred.select(["deviceId", "year", "month", "phys_pred"])


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL B — GLOBAL MONTHLY LIGHTGBM (SECONDARY)
# ─────────────────────────────────────────────────────────────────────────────


MONTHLY_FEATURE_COLS = [
    "month",
    "temp_mean_monthly",
    "temp_min_monthly",
    "HDD_15_monthly",
    "HDD_18_monthly",
    "CDD_22_monthly",
    "load_proxy_monthly",
    "solar_monthly",
    "day_length_monthly",
    "inv_cop_monthly",
    "t1_mean_monthly",
    "t1_min_monthly",
    "x1_mean_monthly",
    "frac_heating_season",
    "x3",
    "deviceType",
]

_MONOTONE = [0] * len(MONTHLY_FEATURE_COLS)
_MONOTONE[MONTHLY_FEATURE_COLS.index("temp_mean_monthly")] = -1
_MONOTONE[MONTHLY_FEATURE_COLS.index("HDD_15_monthly")] = 1
_MONOTONE[MONTHLY_FEATURE_COLS.index("load_proxy_monthly")] = 1

LGB_PARAMS = {
    "objective": "huber",
    "huber_delta": 0.5,
    "linear_tree": True,
    "monotone_constraints": _MONOTONE,
    "metric": "mae",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": -1,
}


def get_cv_folds(monthly_train: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Temporal forward-chaining CV: predict Feb, Mar, Apr from prior months."""
    month_col = monthly_train["month"].to_numpy()
    is_winter = (month_col >= 10) | (month_col <= 1)  # Oct, Nov, Dec, Jan
    folds = [
        (is_winter, month_col == 2),
        (is_winter | (month_col == 2), month_col == 3),
        (is_winter | (month_col == 2) | (month_col == 3), month_col == 4),
    ]
    result = []
    for train_mask, val_mask in folds:
        ti = np.where(train_mask)[0]
        vi = np.where(val_mask)[0]
        if len(ti) > 0 and len(vi) > 0:
            result.append((ti, vi))
    return result


def train_monthly_lgb(monthly: pl.DataFrame) -> list[lgb.Booster]:
    """Train LightGBM on monthly data for all 600 devices × 7 months = ~4200 rows."""
    train_monthly = monthly.filter(pl.col("period") == "train").fill_null(0)
    X = train_monthly.select(MONTHLY_FEATURE_COLS).to_numpy().astype(np.float32)
    y = train_monthly["x2_monthly"].to_numpy().astype(np.float64)
    w = train_monthly["sample_weight"].to_numpy().astype(np.float64)

    folds = get_cv_folds(train_monthly)
    print(f"  Monthly CV folds: {len(folds)}, train rows: {len(y)}")

    models = []
    for fold_i, (ti, vi) in enumerate(folds):
        dtrain = lgb.Dataset(
            X[ti], label=y[ti], weight=w[ti], feature_name=MONTHLY_FEATURE_COLS, free_raw_data=False
        )
        dval = lgb.Dataset(X[vi], label=y[vi], reference=dtrain)
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        val_mae = float(np.mean(np.abs(model.predict(X[vi]) - y[vi])))
        print(f"  Monthly LGB Fold {fold_i + 1} MAE={val_mae:.4f}")
        models.append(model)

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENSEMBLE + SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────


def build_submission(
    monthly: pl.DataFrame,
    monthly_lgb_models: list[lgb.Booster],
    physics_params: pl.DataFrame,
) -> pl.DataFrame:
    """
    Weighted ensemble: 0.7 * physics + 0.3 * monthly_lgb.
    Physics model uses actual HDD from validation/test periods (t1-derived).
    """
    pred_monthly = monthly.filter(pl.col("period").is_in(["validation", "test"]))

    # ── Model A: per-device physics prediction ──────────────────────────────
    print("  Model A: per-device physics prediction …")
    pred_a_df = predict_physics(pred_monthly, physics_params)

    # ── Model B: monthly LightGBM ────────────────────────────────────────────
    print("  Model B: monthly LightGBM inference …")
    X_pred = pred_monthly.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)

    if monthly_lgb_models:
        lgb_preds = np.stack([m.predict(X_pred) for m in monthly_lgb_models], axis=1)
        lgb_ensemble = np.median(lgb_preds, axis=1)
    else:
        lgb_ensemble = np.zeros(len(X_pred))

    pred_b_df = pred_monthly.select(["deviceId", "year", "month"]).with_columns(
        pl.Series("lgb_pred", np.maximum(lgb_ensemble, 0.0))
    )

    # ── Combine: 70% physics + 30% LGB ──────────────────────────────────────
    combined = pred_a_df.join(pred_b_df, on=["deviceId", "year", "month"], how="left")
    combined = combined.with_columns(
        (pl.col("phys_pred") * 0.7 + pl.col("lgb_pred").fill_null(pl.col("phys_pred")) * 0.3).alias(
            "prediction"
        )
    )

    # ── Final clip to [0, 1] ─────────────────────────────────────────────────
    combined = combined.with_columns(pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0))

    return combined.select(["deviceId", "year", "month", "prediction"]).sort(
        ["deviceId", "year", "month"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean
    df, devices = load_data()

    # 2. Fetch weather (optional — falls back to t1-based physics)
    weather = fetch_weather(devices)
    has_weather = weather is not None

    if has_weather:
        print("Joining weather data …")
        df = join_weather(df, devices, weather)
    else:
        print("No weather data — using t1-based physics fallback")
        df = df.join(
            devices.select(["deviceId", "latitude", "longitude"]), on="deviceId", how="left"
        )

    # 3. Feature engineering (ALL periods — validation/test t1 used for HDD)
    print("Computing physics features …")
    df = add_physics_features(df, has_weather)

    print("Aggregating to daily …")
    daily = daily_aggregate(df, devices)
    print(f"Daily rows: {len(daily):,}")

    # 4. Monthly aggregation (ALL periods)
    print("Building monthly features …")
    monthly = build_monthly_features(daily)
    n_train = monthly.filter(pl.col("period") == "train")["month"].n_unique()
    print(f"Monthly rows: {len(monthly):,}  (train months: {n_train})")

    # 5. Fit per-device physics model
    print("Fitting per-device physics model …")
    physics_params = fit_physics_model(monthly)
    print(f"  Devices fitted: {len(physics_params)}")
    dhw_vals = physics_params["dhw_baseload"].to_numpy()
    print(f"  DHW baseload: mean={dhw_vals.mean():.4f}, median={np.median(dhw_vals):.4f}")

    # 6. Train monthly LightGBM
    print("Training monthly LightGBM …")
    monthly_lgb_models = train_monthly_lgb(monthly)

    # 7. Build submission
    print("Building submission …")
    submission = build_submission(monthly, monthly_lgb_models, physics_params)
    n = len(submission)
    print(f"Submission rows: {n:,} (expected 3600 = 600 × 6)")
    if n != 3600:
        print(f"  WARNING: expected 3600 rows, got {n}")

    submission.write_csv(SUBMISSION_FILE)
    print(f"Saved → {SUBMISSION_FILE}")

    pred_vals = submission["prediction"].to_numpy()
    print(
        f"Prediction stats: mean={pred_vals.mean():.4f}, "
        f"median={np.median(pred_vals):.4f}, "
        f"min={pred_vals.min():.4f}, max={pred_vals.max():.4f}"
    )

    # 8. Optional: submit via API
    api_token = os.getenv("TEAM_TOKEN")
    server_url = os.getenv("SERVER_URL")
    if api_token and server_url:
        print("Submitting to API …")
        resp = requests.post(
            f"{server_url}/task3",
            files={"csv_file": open(SUBMISSION_FILE, "rb")},
            headers={"X-API-Token": api_token},
            timeout=120,
        )
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        print(f"API response: {resp.status_code} {data}")
    else:
        print("No TEAM_TOKEN/SERVER_URL in .env — skipping API submission")


if __name__ == "__main__":
    main()

"""
Heat Pump Grid Load Prediction — Task 3
Predicts monthly average x2 for 600 heat pumps, May-Oct 2025.
Trained on Oct 2024 – Apr 2025 (winter → summer OOD extrapolation).

x2 is min-max normalized to [0,1]. Target MAE <= 0.01.
Key insight: monthly avg x2 winter ~0.20, April ~0.08, summer ~0.02-0.04.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
        "deviceId", "timedate", "period",
        "t1", "t2", "t7",
        "x1", "x2", "x3", "deviceType",
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

    # Remove negative x2
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
    df = df.filter(
        (pl.col("streak_len") <= 24) | (pl.col("period") != "train")
    ).drop(["x2_changed", "streak_id", "streak_len"])

    print(f"After cleaning: {len(df):,} rows")
    return df, devices


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEATHER FETCHING (Open-Meteo ERA5) — optional, falls back to t1
# ─────────────────────────────────────────────────────────────────────────────


def fetch_weather(devices: pl.DataFrame) -> pl.DataFrame | None:
    """Fetch hourly weather for each unique lat/lon. Returns None if unavailable."""
    if WEATHER_CACHE.exists():
        print("Loading cached weather …")
        return pl.read_parquet(WEATHER_CACHE)

    print("Fetching weather from Open-Meteo …")
    locations = (
        devices
        .with_columns([pl.col("latitude").round(1), pl.col("longitude").round(1)])
        .select(["latitude", "longitude"])
        .unique()
        .sort(["latitude", "longitude"])
    )
    all_locs = locations.to_dicts()
    n_total = len(all_locs)
    print(f"  {n_total} unique locations → {math.ceil(n_total / BATCH_SIZE)} batches of {BATCH_SIZE}")

    already_done: set[tuple] = set()
    partial_frames: list[pl.DataFrame] = []
    if WEATHER_PARTIAL.exists():
        partial = pl.read_parquet(WEATHER_PARTIAL)
        partial_frames.append(partial)
        for row in partial.select(["latitude", "longitude"]).unique().iter_rows():
            already_done.add(row)
        print(f"  Resuming: {len(already_done)} locations already cached")

    remaining = [loc for loc in all_locs if (loc["latitude"], loc["longitude"]) not in already_done]
    batches = [remaining[i: i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

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
                time.sleep(min(10 * (2 ** attempt), 120))
                continue

            if resp.status_code == 200:
                break
            wait = min(10 * (2 ** attempt), 120)
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
                records.append({
                    "latitude": lat, "longitude": lon, "hour": t,
                    "temp_2m": te, "solar_rad": so, "humidity": hu, "wind_speed": wi,
                })

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
    devices_rounded = devices.with_columns([
        pl.col("latitude").round(1),
        pl.col("longitude").round(1),
    ])
    df = df.join(devices_rounded, on="deviceId", how="left")
    df = df.with_columns(pl.col("timedate").dt.truncate("1h").alias("hour"))
    df = df.join(weather, on=["latitude", "longitude", "hour"], how="left")
    return df.drop("hour")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHYSICS-INFORMED FEATURE ENGINEERING + DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────


def day_length(lat_deg: float, doy: int) -> float:
    """Astronomical day length in hours for a given latitude and day-of-year."""
    lat_rad = math.radians(lat_deg)
    decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))
    cos_ha = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(decl)))
    return (2 / 15) * math.degrees(math.acos(cos_ha))


def add_physics_features(df: pl.DataFrame, has_weather: bool) -> pl.DataFrame:
    """Add thermodynamic features. Falls back to t1-derived approx if no weather."""
    T_base15 = 15.0
    T_base18 = 18.0
    T_cool22 = 22.0
    T_cool24 = 24.0
    T_indoor = 20.0

    if has_weather:
        # Use absolute Open-Meteo temperatures
        temp_col = pl.col("temp_2m")
    else:
        # Fallback: rescale normalized t1 → approximate °C
        # Polish range roughly -10°C to 25°C → t1 in [0, 1]
        temp_col = pl.col("t1") * 35.0 - 10.0
        # Add dummy weather columns so downstream code doesn't break
        df = df.with_columns([
            (pl.col("t1") * 35.0 - 10.0).alias("temp_2m"),
            pl.lit(None).cast(pl.Float64).alias("solar_rad"),
            pl.lit(None).cast(pl.Float64).alias("humidity"),
            pl.lit(None).cast(pl.Float64).alias("wind_speed"),
        ])

    df = df.with_columns([
        (pl.lit(T_base15) - temp_col).clip(lower_bound=0).alias("hdd15_5min"),
        (pl.lit(T_base18) - temp_col).clip(lower_bound=0).alias("hdd18_5min"),
        (temp_col - pl.lit(T_cool22)).clip(lower_bound=0).alias("cdd22_5min"),
        (temp_col - pl.lit(T_cool24)).clip(lower_bound=0).alias("cdd24_5min"),
        ((pl.lit(T_indoor) - temp_col) / pl.lit(T_indoor)).clip(lower_bound=0).alias("inv_cop"),
        pl.col("timedate").dt.date().alias("date"),
        pl.col("timedate").dt.year().alias("year"),
        pl.col("timedate").dt.month().alias("month"),
        pl.col("timedate").dt.ordinal_day().alias("doy"),
    ])
    return df


def daily_aggregate(df: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Compress 5-minute rows to one row per (deviceId, date)."""
    df = df.with_columns(
        (pl.col("hdd15_5min") * pl.col("inv_cop")).alias("load_proxy_5min")
    )

    # Join lat if not already present
    if "latitude" not in df.columns:
        df = df.join(devices.select(["deviceId", "latitude"]), on="deviceId", how="left")

    daily = df.group_by(["deviceId", "date"]).agg([
        pl.col("x2").mean().alias("x2_mean"),
        pl.col("x2").median().alias("x2_median"),
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
        (pl.col("cdd24_5min").sum() / 288).alias("CDD_24"),
        pl.col("inv_cop").mean().alias("inv_cop"),
        (pl.col("load_proxy_5min").sum() / 288).alias("load_proxy"),
    ])

    # Astronomical day length
    daily = daily.with_columns(
        pl.struct(["latitude", "doy"])
        .map_elements(
            lambda s: day_length(s["latitude"], s["doy"]),
            return_dtype=pl.Float64,
        )
        .alias("day_length")
    )

    daily = daily.sort(["deviceId", "date"])
    daily = daily.with_columns(
        (pl.col("temp_mean") < 15.0).cast(pl.Int32).alias("cold_day")
    )
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

    # Upweight warm training days (pseudo-summer proxy, 3x weight)
    daily = daily.with_columns(
        pl.when((pl.col("period") == "train") & (pl.col("temp_mean") > 15.0))
        .then(3.0)
        .otherwise(1.0)
        .alias("sample_weight")
    )

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-DEVICE MONTHLY TRAINING PROFILE
# ─────────────────────────────────────────────────────────────────────────────


def compute_device_monthly_profile(daily: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-device monthly mean x2 from training data.
    This is the single most important anchor for summer predictions:
    - Oct 2025 ≈ Oct 2024 (same seasonal conditions)
    - May-Sep should trend down from April
    """
    train = daily.filter(pl.col("period") == "train")

    # Correct monthly average: mean over ALL 5-min readings in that month
    # We approximate this as mean of daily means (close enough, daily rows are equally sized)
    profile = (
        train.group_by(["deviceId", "month"])
        .agg(pl.col("x2_mean").mean().alias("x2_monthly_mean"))
        .sort(["deviceId", "month"])
    )

    # Pivot to wide format: one row per device, columns = month_1 ... month_12
    profile_wide = profile.pivot(
        on="month", index="deviceId", values="x2_monthly_mean", aggregate_function="mean"
    )
    # Rename columns to m_1, m_2 ... for clarity
    rename_map = {str(m): f"m_{m}" for m in range(1, 13)}
    for old, new in rename_map.items():
        if old in profile_wide.columns:
            profile_wide = profile_wide.rename({old: new})

    # Per-device device-type and x3 (static features)
    device_meta = (
        train.group_by("deviceId")
        .agg([
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("x2_mean").mean().alias("x2_train_mean"),
            pl.col("x2_mean").std().alias("x2_train_std"),
        ])
    )
    profile_wide = profile_wide.join(device_meta, on="deviceId", how="left")
    return profile_wide


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL A — PER-DEVICE TREND EXTRAPOLATION
# ─────────────────────────────────────────────────────────────────────────────


def predict_trend_extrapolation(
    profile: pl.DataFrame,
    target_months: list[int],
) -> pl.DataFrame:
    """
    Simple per-device trend-based extrapolation:
    - Oct 2025 = Oct 2024 (calibration anchor)
    - May-Sep = decaying fraction of April value, shaped by device group median
    Key insight: summer load ≈ DHW-only baseload, which is ~25-50% of April load.
    """
    rows = []
    profile_np = profile.to_pandas()

    # Group medians by (deviceType, x3) for robustness
    group_apr = {}
    for _, grp in profile_np.groupby(["deviceType", "x3"]):
        key = (grp["deviceType"].iloc[0], grp["x3"].iloc[0])
        apr_val = grp.get("m_4", None)
        if apr_val is not None:
            group_apr[key] = float(apr_val.median()) if hasattr(apr_val, "median") else float(apr_val.iloc[0])

    # Summer decay factors relative to April (physics-informed)
    # Based on typical Polish heat pump seasonal patterns
    decay = {5: 0.55, 6: 0.38, 7: 0.30, 8: 0.30, 9: 0.38, 10: 1.00}

    for _, row in profile_np.iterrows():
        device_id = row["deviceId"]
        apr = row.get("m_4", np.nan)
        oct24 = row.get("m_10", np.nan)
        dt = row.get("deviceType", -1)
        x3 = row.get("x3", -1)

        # Fallback if device has no April data
        group_key = (dt, x3)
        if np.isnan(apr) or apr == 0:
            apr = group_apr.get(group_key, 0.05)

        if np.isnan(oct24) or oct24 == 0:
            oct24 = group_apr.get(group_key, apr)

        for m in target_months:
            if m == 10:
                pred = oct24  # Direct anchor
            else:
                pred = apr * decay.get(m, 0.35)
            rows.append({"deviceId": device_id, "month": m, "pred_trend": max(pred, 0.0)})

    return pl.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL B — GLOBAL LIGHTGBM ON MONTHLY DATA
# ─────────────────────────────────────────────────────────────────────────────


def build_monthly_features(daily: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate daily data to monthly level per device.
    Training: ~600 devices x 7 months = ~4200 rows.
    This is the correct granularity for the target (monthly mean x2).
    """
    monthly = (
        daily.group_by(["deviceId", "year", "month"])
        .agg([
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
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("is_heating_season").mean().alias("frac_heating_season"),
            pl.col("sample_weight").mean().alias("sample_weight"),
        ])
    )
    return monthly.sort(["deviceId", "year", "month"])


MONTHLY_FEATURE_COLS = [
    "month", "temp_mean_monthly", "temp_min_monthly",
    "HDD_15_monthly", "HDD_18_monthly", "CDD_22_monthly",
    "load_proxy_monthly", "solar_monthly", "day_length_monthly",
    "inv_cop_monthly", "t1_mean_monthly", "frac_heating_season",
    "x3", "deviceType",
]

MONTHLY_MONOTONE = [0] * len(MONTHLY_FEATURE_COLS)
MONTHLY_MONOTONE[MONTHLY_FEATURE_COLS.index("temp_mean_monthly")] = -1
MONTHLY_MONOTONE[MONTHLY_FEATURE_COLS.index("HDD_15_monthly")] = 1
MONTHLY_MONOTONE[MONTHLY_FEATURE_COLS.index("load_proxy_monthly")] = 1

LGB_MONTHLY_PARAMS = {
    "objective": "huber",
    "huber_delta": 0.5,
    "linear_tree": True,
    "monotone_constraints": MONTHLY_MONOTONE,
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


def get_monthly_cv_folds(monthly: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Temporal forward-chaining CV at monthly granularity."""
    month_col = monthly["month"].to_numpy()
    is_winter = (month_col >= 10) | (month_col <= 1)
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
    """Train LightGBM on monthly aggregated data."""
    train_monthly = monthly.filter(pl.col("period") == "train").fill_null(0)
    X = train_monthly.select(MONTHLY_FEATURE_COLS).to_numpy().astype(np.float32)
    y = train_monthly["x2_monthly"].to_numpy().astype(np.float64)
    w = train_monthly["sample_weight"].to_numpy().astype(np.float64)

    folds = get_monthly_cv_folds(train_monthly)
    print(f"  Monthly CV folds: {len(folds)}, train size: {len(y)}")

    models = []
    for fold_i, (ti, vi) in enumerate(folds):
        dtrain = lgb.Dataset(X[ti], label=y[ti], weight=w[ti],
                             feature_name=MONTHLY_FEATURE_COLS, free_raw_data=False)
        dval = lgb.Dataset(X[vi], label=y[vi], reference=dtrain)
        model = lgb.train(
            LGB_MONTHLY_PARAMS,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        val_mae = np.mean(np.abs(model.predict(X[vi]) - y[vi]))
        print(f"  Monthly LGB Fold {fold_i + 1} MAE={val_mae:.4f}")
        models.append(model)

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL C — PER-DEVICE PHYSICS BASELINE (OLS)
# ─────────────────────────────────────────────────────────────────────────────


def fit_physics_baseline(monthly: pl.DataFrame) -> pl.DataFrame:
    """
    Per-device OLS on monthly data: monthly_x2 ~ HDD_15 + 1
    Returns deviceId, htc (HDD coefficient), dhw_baseload (intercept).
    """
    train_monthly = monthly.filter(pl.col("period") == "train").fill_null(0)
    results = []
    for device_id, grp in train_monthly.group_by("deviceId"):
        hdd = grp["HDD_15_monthly"].to_numpy()
        x2 = grp["x2_monthly"].to_numpy()
        mask = np.isfinite(x2) & np.isfinite(hdd)
        if mask.sum() < 3:
            results.append({
                "deviceId": device_id[0],
                "htc": 0.0,
                "dhw_baseload": float(np.nanmedian(x2)),
            })
            continue
        A = np.column_stack([hdd[mask], np.ones(mask.sum())])
        coef, _, _, _ = np.linalg.lstsq(A, x2[mask], rcond=None)
        results.append({
            "deviceId": device_id[0],
            "htc": max(float(coef[0]), 0.0),
            "dhw_baseload": max(float(coef[1]), 0.0),
        })
    return pl.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 8. DAILY-LEVEL LIGHTGBM (for devices with weather data)
# ─────────────────────────────────────────────────────────────────────────────

DAILY_FEATURE_COLS = [
    "temp_mean", "temp_max", "temp_min",
    "HDD_15", "HDD_18", "CDD_22", "CDD_24",
    "inv_cop", "load_proxy",
    "solar_sum", "humidity_mean", "wind_mean",
    "day_length", "doy", "month",
    "is_heating_season",
    "t1_mean", "t1_max", "t1_min", "t2_mean", "t7_mean",
    "x1_mean", "x3", "deviceType",
]

DAILY_MONOTONE = [0] * len(DAILY_FEATURE_COLS)
DAILY_MONOTONE[DAILY_FEATURE_COLS.index("temp_mean")] = -1
DAILY_MONOTONE[DAILY_FEATURE_COLS.index("load_proxy")] = 1
DAILY_MONOTONE[DAILY_FEATURE_COLS.index("HDD_15")] = 1
DAILY_MONOTONE[DAILY_FEATURE_COLS.index("HDD_18")] = 1

LGB_DAILY_PARAMS = {
    "objective": "huber",
    "huber_delta": 1.0,
    "linear_tree": True,
    "monotone_constraints": DAILY_MONOTONE,
    "metric": "mae",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
}

DAILY_PHYSICS_FEATURES = [
    "HDD_15", "HDD_18", "CDD_22", "CDD_24",
    "load_proxy", "day_length", "solar_sum",
    "temp_mean", "inv_cop", "x3", "deviceType", "month",
]


def get_daily_cv_folds(daily_train: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    month_col = daily_train["month"].to_numpy()
    is_winter = (month_col >= 10) | (month_col <= 1)
    folds = [
        (is_winter, month_col == 2),
        (is_winter | (month_col == 2), month_col == 3),
        (is_winter | (month_col == 2) | (month_col == 3), month_col == 4),
    ]
    result = []
    for train_mask, val_mask in folds:
        ti, vi = np.where(train_mask)[0], np.where(val_mask)[0]
        if len(ti) > 0 and len(vi) > 0:
            result.append((ti, vi))
    return result


def train_daily_models(daily: pl.DataFrame) -> dict:
    """Train daily-level LightGBM + Ridge models."""
    train_df = daily.filter(pl.col("period") == "train").fill_null(0)
    X = train_df.select(DAILY_FEATURE_COLS).to_numpy().astype(np.float32)
    y = train_df["x2_mean"].to_numpy().astype(np.float64)
    w = train_df["sample_weight"].to_numpy().astype(np.float64)

    folds = get_daily_cv_folds(train_df)
    print(f"  Daily CV folds: {len(folds)}, train size: {len(y)}")

    models_lgb, models_lgb_log, models_ridge = [], [], []
    phys_idx = [DAILY_FEATURE_COLS.index(f) for f in DAILY_PHYSICS_FEATURES]

    for fold_i, (ti, vi) in enumerate(folds):
        X_tr, y_tr, w_tr = X[ti], y[ti], w[ti]
        X_val, y_val = X[vi], y[vi]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                             feature_name=DAILY_FEATURE_COLS, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        m = lgb.train(LGB_DAILY_PARAMS, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        print(f"  Daily LGB raw  Fold {fold_i+1} MAE={np.mean(np.abs(m.predict(X_val)-y_val)):.4f}")
        models_lgb.append(m)

        y_log = np.log1p(np.maximum(y_tr, 0))
        dtl = lgb.Dataset(X_tr, label=y_log, weight=w_tr,
                          feature_name=DAILY_FEATURE_COLS, free_raw_data=False)
        dvl = lgb.Dataset(X_val, label=np.log1p(np.maximum(y_val, 0)), reference=dtl)
        ml = lgb.train(LGB_DAILY_PARAMS, dtl, num_boost_round=500,
                       valid_sets=[dvl],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p_log = np.expm1(np.maximum(ml.predict(X_val), 0))
        print(f"  Daily LGB log  Fold {fold_i+1} MAE={np.mean(np.abs(p_log-y_val)):.4f}")
        models_lgb_log.append(ml)

        scaler = StandardScaler()
        Xp_tr = scaler.fit_transform(X_tr[:, phys_idx])
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xp_tr, y_tr, sample_weight=w_tr)
        p_r = ridge.predict(scaler.transform(X_val[:, phys_idx]))
        print(f"  Ridge          Fold {fold_i+1} MAE={np.mean(np.abs(p_r-y_val)):.4f}")
        models_ridge.append((scaler, ridge))

    return {"lgb": models_lgb, "lgb_log": models_lgb_log, "ridge": models_ridge}


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUBMISSION — 3-MODEL ENSEMBLE + POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def build_submission(
    daily: pl.DataFrame,
    monthly: pl.DataFrame,
    daily_models: dict,
    monthly_lgb_models: list[lgb.Booster],
    physics_params: pl.DataFrame,
    profile: pl.DataFrame,
) -> pl.DataFrame:
    """
    3-model ensemble:
      A. Per-device trend extrapolation (Oct anchor + spring/summer decay)
      B. Monthly LightGBM (global, 4200-row training set)
      C. Daily LightGBM/Ridge → aggregated to monthly

    Final = median(A, B, C) with post-processing guardrails.
    """
    all_devices = daily["deviceId"].unique().to_list()
    target_months = [5, 6, 7, 8, 9, 10]
    year = 2025

    # ── Model A: trend extrapolation ────────────────────────────────────────
    print("  Model A: per-device trend extrapolation …")
    pred_a = predict_trend_extrapolation(profile, target_months)

    # ── Model B: monthly LightGBM ────────────────────────────────────────────
    print("  Model B: monthly LightGBM inference …")
    pred_monthly = monthly.filter(pl.col("period").is_in(["validation", "test"]))
    X_monthly = pred_monthly.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)

    b_preds = []
    for m in monthly_lgb_models:
        b_preds.append(m.predict(X_monthly))
    b_ensemble = np.median(np.stack(b_preds, axis=1), axis=1) if b_preds else np.zeros(len(X_monthly))

    pred_b = pred_monthly.select(["deviceId", "year", "month"]).with_columns(
        pl.Series("pred_monthly_lgb", np.maximum(b_ensemble, 0.0))
    )

    # ── Model C: daily LightGBM aggregated to monthly ───────────────────────
    print("  Model C: daily LightGBM → monthly …")
    pred_daily_df = daily.filter(pl.col("period").is_in(["validation", "test"]))
    X_daily = pred_daily_df.select(DAILY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
    phys_idx = [DAILY_FEATURE_COLS.index(f) for f in DAILY_PHYSICS_FEATURES]

    c_preds = []
    for m in daily_models["lgb"]:
        c_preds.append(m.predict(X_daily))
    for m in daily_models["lgb_log"]:
        c_preds.append(np.expm1(np.maximum(m.predict(X_daily), 0)))
    for scaler, ridge in daily_models["ridge"]:
        Xp = scaler.transform(X_daily[:, phys_idx])
        c_preds.append(np.maximum(ridge.predict(Xp), 0))

    c_daily_ensemble = np.median(np.stack(c_preds, axis=1), axis=1) if c_preds else np.zeros(len(X_daily))

    pred_c_daily = pred_daily_df.select(["deviceId", "year", "month"]).with_columns(
        pl.Series("pred_daily_val", np.maximum(c_daily_ensemble, 0.0))
    )
    pred_c = (
        pred_c_daily.group_by(["deviceId", "year", "month"])
        .agg(pl.col("pred_daily_val").mean().alias("pred_daily_lgb"))
    )

    # ── Combine all three models ─────────────────────────────────────────────
    # Start from a complete (deviceId, year, month) grid
    # Use pred_b as the base (has all devices × months from monthly df)
    combined = pred_b.rename({"pred_monthly_lgb": "pred_b"})

    # Join pred_a
    combined = combined.join(
        pred_a.rename({"pred_trend": "pred_a"}),
        on=["deviceId", "month"],
        how="left",
    )

    # Join pred_c
    combined = combined.join(
        pred_c.rename({"pred_daily_lgb": "pred_c"}),
        on=["deviceId", "year", "month"],
        how="left",
    )

    # Fill missing with pred_b (monthly LGB is most reliable fallback)
    combined = combined.with_columns([
        pl.col("pred_a").fill_null(pl.col("pred_b")),
        pl.col("pred_c").fill_null(pl.col("pred_b")),
    ])

    # Compute median of three models
    pA = combined["pred_a"].to_numpy()
    pB = combined["pred_b"].to_numpy()
    pC = combined["pred_c"].to_numpy()
    final_pred = np.median(np.stack([pA, pB, pC], axis=1), axis=1)
    final_pred = np.maximum(final_pred, 0.0)

    combined = combined.with_columns(pl.Series("prediction", final_pred))

    # ── Post-processing guardrails ───────────────────────────────────────────
    # 1. DHW floor: prediction >= 50% of per-device DHW baseload
    combined = combined.join(physics_params.select(["deviceId", "dhw_baseload"]),
                              on="deviceId", how="left")
    combined = combined.with_columns(
        pl.when(pl.col("dhw_baseload").is_not_null())
        .then(pl.col("prediction").clip(lower_bound=pl.col("dhw_baseload") * 0.5))
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    ).drop("dhw_baseload")

    # 2. Oct 2025 ≈ Oct 2024 anchor (blend 60/40)
    oct24 = (
        daily.filter((pl.col("period") == "train") & (pl.col("month") == 10))
        .group_by("deviceId")
        .agg(pl.col("x2_mean").mean().alias("oct24_mean"))
    )
    combined = combined.join(oct24, on="deviceId", how="left")
    combined = combined.with_columns(
        pl.when((pl.col("month") == 10) & pl.col("oct24_mean").is_not_null())
        .then(pl.col("prediction") * 0.6 + pl.col("oct24_mean") * 0.4)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    ).drop("oct24_mean")

    # 3. Jun-Aug <= Apr for heating-only devices
    apr_avg = (
        daily.filter((pl.col("period") == "train") & (pl.col("month") == 4))
        .group_by("deviceId")
        .agg(pl.col("x2_mean").mean().alias("apr_mean"))
    )
    combined = combined.join(apr_avg, on="deviceId", how="left")
    combined = combined.with_columns(
        pl.when(pl.col("month").is_in([6, 7, 8]) & pl.col("apr_mean").is_not_null())
        .then(pl.col("prediction").clip(upper_bound=pl.col("apr_mean")))
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    ).drop("apr_mean")

    # 4. Final clip to [0, 1]
    combined = combined.with_columns(
        pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0)
    )

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

    # 2. Fetch weather (optional)
    weather = fetch_weather(devices)
    has_weather = weather is not None

    if has_weather:
        print("Joining weather data …")
        df = join_weather(df, devices, weather)
    else:
        print("No weather data — using t1-based physics fallback")
        # Add lat/lon from devices for day_length calculation
        df = df.join(devices.select(["deviceId", "latitude", "longitude"]), on="deviceId", how="left")

    # 3. Feature engineering
    print("Computing physics features …")
    df = add_physics_features(df, has_weather)

    print("Aggregating to daily …")
    daily = daily_aggregate(df, devices)
    print(f"Daily rows: {len(daily):,}")

    # 4. Build monthly features
    print("Building monthly features …")
    monthly = build_monthly_features(daily)
    print(f"Monthly rows: {len(monthly):,}")

    # 5. Compute per-device monthly profile (for trend extrapolation)
    print("Computing device monthly profiles …")
    profile = compute_device_monthly_profile(daily)

    # 6. Fit physics baseline
    print("Fitting per-device physics baseline …")
    physics_params = fit_physics_baseline(monthly)

    # 7. Train models
    print("Training monthly LightGBM …")
    monthly_lgb_models = train_monthly_lgb(monthly)

    print("Training daily LightGBM + Ridge …")
    daily_models = train_daily_models(daily)

    # 8. Build submission
    print("Building submission …")
    submission = build_submission(
        daily, monthly, daily_models, monthly_lgb_models, physics_params, profile
    )
    n = len(submission)
    print(f"Submission rows: {n:,} (expected 3600 = 600 × 6)")
    if n != 3600:
        print(f"  WARNING: expected 3600 rows, got {n}")

    submission.write_csv(SUBMISSION_FILE)
    print(f"Saved → {SUBMISSION_FILE}")

    # Print quick summary stats
    pred_vals = submission["prediction"].to_numpy()
    print(f"Prediction stats: mean={pred_vals.mean():.4f}, "
          f"median={np.median(pred_vals):.4f}, "
          f"min={pred_vals.min():.4f}, max={pred_vals.max():.4f}")

    # 9. Submit via API
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

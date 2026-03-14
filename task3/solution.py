"""
Heat Pump Grid Load Prediction — Task 3
Predicts monthly average x2 for 600 heat pumps, May-Oct 2025.
Trained on Oct 2024 – Apr 2025 (winter → summer OOD extrapolation).
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
SUBMISSION_FILE = OUT_DIR / "submission.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and clean data.csv + devices.csv using Polars lazy mode."""
    print("Loading devices.csv …")
    devices = pl.read_csv(DATA_DIR / "devices.csv")

    print("Scanning data.csv (lazy, 10 GB) …")
    # Columns we actually need — avoids loading the full 10 GB into RAM
    needed_cols = [
        "deviceId",
        "timedate",
        "period",
        "t1",
        "t2",
        "t7",  # external, internal, DHW tank temps (normalized)
        "x1",
        "x2",
        "x3",
        "deviceType",
    ]

    lazy = pl.scan_csv(DATA_DIR / "data.csv", try_parse_dates=False)
    lazy = lazy.select(needed_cols)

    # Parse datetime — strip trailing " UTC" if present
    lazy = lazy.with_columns(
        pl.col("timedate")
        .str.replace(r" UTC$", "")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timedate")
    )

    print("Collecting … (this may take a few minutes)")
    df = lazy.collect()

    print(f"Loaded {len(df):,} rows")

    # --- Clean x2 --------------------------------------------------------
    # Remove negative readings
    df = df.filter(pl.col("x2").is_null() | (pl.col("x2") >= 0))

    # Winsorize x2 at 99.5th percentile per device (using train period only)
    train_mask = pl.col("period") == "train"
    p995 = (
        df.filter(train_mask).group_by("deviceId").agg(pl.col("x2").quantile(0.995).alias("x2_cap"))
    )
    df = df.join(p995, on="deviceId", how="left")
    df = df.with_columns(
        pl.when(pl.col("x2").is_not_null() & pl.col("x2_cap").is_not_null())
        .then(pl.col("x2").clip(upper_bound=pl.col("x2_cap")))
        .otherwise(pl.col("x2"))
        .alias("x2")
    ).drop("x2_cap")

    # Remove stuck meters: x2 constant for > 24 consecutive readings (2 hrs)
    # Use a difference flag: mark rows where x2 changes, then count streaks
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
    # Keep rows that are either:
    # - not in a stuck streak (streak_len <= 24)
    # - or validation/test period (x2 is already NaN, we still need the rows for features)
    df = df.filter((pl.col("streak_len") <= 24) | (pl.col("period") != "train")).drop(
        ["x2_changed", "streak_id", "streak_len"]
    )

    print(f"After cleaning: {len(df):,} rows")
    return df, devices


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEATHER FETCHING (Open-Meteo ERA5)
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_PARTIAL = DATA_DIR / "weather_partial.parquet"
BATCH_SIZE = 50  # Open-Meteo supports up to 50 locations per request


def fetch_weather(devices: pl.DataFrame) -> pl.DataFrame:
    """Fetch hourly weather for each unique lat/lon using batched requests. Cached to parquet."""
    if WEATHER_CACHE.exists():
        print("Loading cached weather …")
        return pl.read_parquet(WEATHER_CACHE)

    print("Fetching weather from Open-Meteo …")
    locations = (
        devices.with_columns(
            [
                pl.col("latitude").round(1),
                pl.col("longitude").round(1),
            ]
        )
        .select(["latitude", "longitude"])
        .unique()
        .sort(["latitude", "longitude"])
    )
    all_locs = locations.to_dicts()
    n_total = len(all_locs)
    print(
        f"  {n_total} unique locations → {math.ceil(n_total / BATCH_SIZE)} batches of {BATCH_SIZE}"
    )

    # Resume from partial cache if it exists
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
        # Multi-location response is a list; single location is a dict
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

        # Save incremental progress after each batch
        combined = pl.concat(partial_frames)
        DATA_DIR.mkdir(exist_ok=True)
        combined.write_parquet(WEATHER_PARTIAL)

        time.sleep(5)

    if not partial_frames:
        raise RuntimeError("No weather data fetched — all batches failed")

    weather = pl.concat(partial_frames)
    weather.write_parquet(WEATHER_CACHE)
    if WEATHER_PARTIAL.exists():
        WEATHER_PARTIAL.unlink()
    print(f"  Cached {len(weather):,} weather rows")
    return weather


def join_weather(df: pl.DataFrame, devices: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Join weather to device telemetry by nearest lat/lon and hour."""
    # Round device coords to match weather grid
    devices_rounded = devices.with_columns(
        [
            pl.col("latitude").round(1),
            pl.col("longitude").round(1),
        ]
    )

    df = df.join(devices_rounded, on="deviceId", how="left")

    # Truncate timedate to hour for join
    df = df.with_columns(pl.col("timedate").dt.truncate("1h").alias("hour"))

    df = df.join(
        weather,
        on=["latitude", "longitude", "hour"],
        how="left",
    )
    return df.drop("hour")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHYSICS-INFORMED FEATURE ENGINEERING + DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────


def day_length(lat_deg: float, doy: int) -> float:
    """Astronomical day length in hours for a given latitude and day-of-year."""
    lat_rad = math.radians(lat_deg)
    # Solar declination (radians)
    decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))
    cos_ha = -math.tan(lat_rad) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    return (2 / 15) * math.degrees(math.acos(cos_ha))


def add_physics_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add thermodynamic features using absolute Open-Meteo temperatures."""
    T_base_heat_15 = 15.0
    T_base_heat_18 = 18.0
    T_base_cool_22 = 22.0
    T_base_cool_24 = 24.0
    T_indoor_estimate = 20.0  # assumed indoor setpoint °C

    df = df.with_columns(
        [
            # Degree days (computed per 5-min row, will be summed daily)
            (pl.lit(T_base_heat_15) - pl.col("temp_2m")).clip(lower_bound=0).alias("hdd15_5min"),
            (pl.lit(T_base_heat_18) - pl.col("temp_2m")).clip(lower_bound=0).alias("hdd18_5min"),
            (pl.col("temp_2m") - pl.lit(T_base_cool_22)).clip(lower_bound=0).alias("cdd22_5min"),
            (pl.col("temp_2m") - pl.lit(T_base_cool_24)).clip(lower_bound=0).alias("cdd24_5min"),
            # Inverse COP proxy: approaches 0 as outdoor temp → indoor setpoint
            ((pl.lit(T_indoor_estimate) - pl.col("temp_2m")) / pl.lit(T_indoor_estimate))
            .clip(lower_bound=0)
            .alias("inv_cop"),
            # Date parts
            pl.col("timedate").dt.date().alias("date"),
            pl.col("timedate").dt.year().alias("year"),
            pl.col("timedate").dt.month().alias("month"),
            pl.col("timedate").dt.ordinal_day().alias("doy"),
        ]
    )
    return df


def daily_aggregate(df: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Compress 5-minute rows to one row per (deviceId, date)."""
    # Compute load proxy: HDD15 × inv_COP (the extrapolation secret weapon)
    df = df.with_columns((pl.col("hdd15_5min") * pl.col("inv_cop")).alias("load_proxy_5min"))

    daily = df.group_by(["deviceId", "date"]).agg(
        [
            # Target (NaN for validation/test)
            pl.col("x2").mean().alias("x2_mean"),
            pl.col("x2").median().alias("x2_median"),
            # Period
            pl.col("period").first().alias("period"),
            pl.col("year").first().alias("year"),
            pl.col("month").first().alias("month"),
            pl.col("doy").first().alias("doy"),
            pl.col("latitude").first().alias("latitude"),
            # Normalized temp stats (relative signals even though not absolute)
            pl.col("t1").mean().alias("t1_mean"),
            pl.col("t1").max().alias("t1_max"),
            pl.col("t1").min().alias("t1_min"),
            pl.col("t2").mean().alias("t2_mean"),
            pl.col("t7").mean().alias("t7_mean"),  # DHW tank
            pl.col("x1").mean().alias("x1_mean"),
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            # Absolute weather features
            pl.col("temp_2m").mean().alias("temp_mean"),
            pl.col("temp_2m").max().alias("temp_max"),
            pl.col("temp_2m").min().alias("temp_min"),
            pl.col("solar_rad").sum().alias("solar_sum"),
            pl.col("humidity").mean().alias("humidity_mean"),
            pl.col("wind_speed").mean().alias("wind_mean"),
            # Physics features (convert 5-min means → daily sums / 12 for degree-day equivalents)
            (pl.col("hdd15_5min").sum() / 288).alias("HDD_15"),  # 288 readings/day
            (pl.col("hdd18_5min").sum() / 288).alias("HDD_18"),
            (pl.col("cdd22_5min").sum() / 288).alias("CDD_22"),
            (pl.col("cdd24_5min").sum() / 288).alias("CDD_24"),
            pl.col("inv_cop").mean().alias("inv_cop"),
            (pl.col("load_proxy_5min").sum() / 288).alias("load_proxy"),  # HDD15 × inv_COP
        ]
    )

    # Day length per (latitude, doy)
    daily = daily.with_columns(
        pl.struct(["latitude", "doy"])
        .map_elements(
            lambda s: day_length(s["latitude"], s["doy"]),
            return_dtype=pl.Float64,
        )
        .alias("day_length")
    )

    # Heating season flag: temp_mean < 15°C for 3+ consecutive days per device
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

    # Sample weight: upweight warm training days (pseudo-summer proxy)
    daily = daily.with_columns(
        pl.when((pl.col("period") == "train") & (pl.col("temp_mean") > 15.0))
        .then(3.0)
        .otherwise(1.0)
        .alias("sample_weight")
    )

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "HDD_15",
    "HDD_18",
    "CDD_22",
    "CDD_24",
    "inv_cop",
    "load_proxy",
    "solar_sum",
    "humidity_mean",
    "wind_mean",
    "day_length",
    "doy",
    "month",
    "is_heating_season",
    "t1_mean",
    "t1_max",
    "t1_min",
    "t2_mean",
    "t7_mean",
    "x1_mean",
    "x3",
    "deviceType",
]

# monotone constraint: load_proxy (index ~8) must be >= 0 trend with load
# temp_mean (index 0): higher temp → lower heating load → constraint -1
# All others: 0 (unconstrained)
MONOTONE = [0] * len(FEATURE_COLS)
MONOTONE[FEATURE_COLS.index("temp_mean")] = -1  # higher outdoor temp → lower load
MONOTONE[FEATURE_COLS.index("load_proxy")] = 1  # higher load_proxy → higher load
MONOTONE[FEATURE_COLS.index("HDD_15")] = 1  # more heating days → higher load
MONOTONE[FEATURE_COLS.index("HDD_18")] = 1

LGB_PARAMS = {
    "objective": "huber",       # supports monotone_constraints; small delta ≈ L1/MAE
    "huber_delta": 1.0,
    "linear_tree": True,
    "monotone_constraints": MONOTONE,
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

PHYSICS_FEATURES = [
    "HDD_15",
    "HDD_18",
    "CDD_22",
    "CDD_24",
    "load_proxy",
    "day_length",
    "solar_sum",
    "temp_mean",
    "inv_cop",
    "x3",
    "deviceType",
    "month",
]


def get_cv_folds(daily_train: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Temporal forward-chaining CV on training data."""
    month_col = daily_train["month"].to_numpy()
    # Training spans Oct(10), Nov(11), Dec(12), Jan(1), Feb(2), Mar(3), Apr(4)
    # Must include Oct-Dec in every training fold
    is_winter = (month_col >= 10) | (month_col <= 1)  # Oct, Nov, Dec, Jan
    folds = [
        # Fold 1: Oct–Jan → Feb
        (is_winter, month_col == 2),
        # Fold 2: Oct–Feb → Mar
        (is_winter | (month_col == 2), month_col == 3),
        # Fold 3: Oct–Mar → Apr (most important: heating wind-down)
        (is_winter | (month_col == 2) | (month_col == 3), month_col == 4),
    ]
    result = []
    for train_mask, val_mask in folds:
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            result.append((train_idx, val_idx))
    return result


def train_models(daily: pl.DataFrame) -> dict:
    """Train LightGBM (raw + log) and Ridge models using temporal CV."""
    train_df = daily.filter(pl.col("period") == "train").fill_null(0)
    X = train_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = train_df["x2_mean"].to_numpy().astype(np.float64)
    w = train_df["sample_weight"].to_numpy().astype(np.float64)

    folds = get_cv_folds(train_df)
    print(f"CV folds: {len(folds)}")

    models_lgb = []
    models_lgb_log = []
    models_ridge = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr, w_tr = X[train_idx], y[train_idx], w[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Model A: LightGBM raw x2
        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=w_tr, feature_name=FEATURE_COLS, free_raw_data=False
        )
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        lgb_model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred = lgb_model.predict(X_val)
        val_mae = np.mean(np.abs(val_pred - y_val))
        print(f"  Fold {fold_i + 1} LGB raw MAE={val_mae:.4f} (n_val={len(y_val)})")
        models_lgb.append(lgb_model)

        # Model B: LightGBM on log1p(x2)
        y_tr_log = np.log1p(np.maximum(y_tr, 0))
        dtrain_log = lgb.Dataset(
            X_tr, label=y_tr_log, weight=w_tr, feature_name=FEATURE_COLS, free_raw_data=False
        )
        y_val_log = np.log1p(np.maximum(y_val, 0))
        dval_log = lgb.Dataset(X_val, label=y_val_log, reference=dtrain_log)
        lgb_log_model = lgb.train(
            LGB_PARAMS,
            dtrain_log,
            num_boost_round=500,
            valid_sets=[dval_log],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred_log = np.expm1(lgb_log_model.predict(X_val))
        val_mae_log = np.mean(np.abs(val_pred_log - y_val))
        print(f"  Fold {fold_i + 1} LGB log MAE={val_mae_log:.4f}")
        models_lgb_log.append(lgb_log_model)

        # Model C: Ridge on physics features (linear extrapolation backbone)
        phys_idx = [FEATURE_COLS.index(f) for f in PHYSICS_FEATURES]
        X_tr_ph = X_tr[:, phys_idx]
        X_val_ph = X_val[:, phys_idx]
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_ph)
        X_val_scaled = scaler.transform(X_val_ph)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_scaled, y_tr, sample_weight=w_tr)
        val_pred_ridge = ridge.predict(X_val_scaled)
        val_mae_ridge = np.mean(np.abs(val_pred_ridge - y_val))
        print(f"  Fold {fold_i + 1} Ridge MAE={val_mae_ridge:.4f}")
        models_ridge.append((scaler, ridge))

    return {
        "lgb": models_lgb,
        "lgb_log": models_lgb_log,
        "ridge": models_ridge,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PHYSICS BASELINE (per-device DHW + heating fit)
# ─────────────────────────────────────────────────────────────────────────────


def fit_physics_baseline(daily: pl.DataFrame) -> pl.DataFrame:
    """
    Per-device OLS: daily_x2 ~ HDD_15 + 1
    Estimates building_HTC (HDD coefficient) and DHW_baseload (intercept).
    Returns a DataFrame with deviceId, htc, dhw_baseload.
    """
    train_df = daily.filter(pl.col("period") == "train").fill_null(0)
    results = []
    for device_id, grp in train_df.group_by("deviceId"):
        hdd = grp["HDD_15"].to_numpy()
        x2 = grp["x2_mean"].to_numpy()
        mask = np.isfinite(x2) & np.isfinite(hdd)
        if mask.sum() < 10:
            results.append(
                {"deviceId": device_id[0], "htc": 0.0, "dhw_baseload": float(np.nanmedian(x2))}
            )
            continue
        A = np.column_stack([hdd[mask], np.ones(mask.sum())])
        coef, _, _, _ = np.linalg.lstsq(A, x2[mask], rcond=None)
        htc = max(coef[0], 0.0)
        dhw = max(coef[1], 0.0)
        results.append({"deviceId": device_id[0], "htc": htc, "dhw_baseload": dhw})

    return pl.DataFrame(results)


def physics_predict(daily: pl.DataFrame, params: pl.DataFrame) -> pl.Series:
    """Predict using per-device physics model: htc × HDD_15 + dhw_baseload."""
    daily_with_params = daily.join(params, on="deviceId", how="left")
    pred = (
        daily_with_params["htc"] * daily_with_params["HDD_15"] + daily_with_params["dhw_baseload"]
    )
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICTION & SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────


def predict_all(daily: pl.DataFrame, trained: dict) -> np.ndarray:
    """Run all models on the full daily DataFrame, return median ensemble."""
    X_all = daily.select(FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
    phys_idx = [FEATURE_COLS.index(f) for f in PHYSICS_FEATURES]

    all_preds = []

    for m in trained["lgb"]:
        all_preds.append(m.predict(X_all))

    for m in trained["lgb_log"]:
        all_preds.append(np.expm1(np.maximum(m.predict(X_all), 0)))

    for scaler, ridge in trained["ridge"]:
        X_ph = X_all[:, phys_idx]
        X_sc = scaler.transform(X_ph)
        all_preds.append(np.maximum(ridge.predict(X_sc), 0))

    # Median ensemble across all models & folds
    ensemble = np.median(np.stack(all_preds, axis=1), axis=1)
    return ensemble


def build_submission(
    daily: pl.DataFrame,
    trained: dict,
    physics_params: pl.DataFrame,
) -> pl.DataFrame:
    """
    Generate submission.csv:
    1. Predict daily x2 for val+test periods
    2. Post-process with physics floor and constraints
    3. Aggregate to monthly averages
    """
    predict_periods = ["validation", "test"]
    pred_df = daily.filter(pl.col("period").is_in(predict_periods))

    X_pred = pred_df.select(FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
    phys_idx = [FEATURE_COLS.index(f) for f in PHYSICS_FEATURES]

    all_preds = []
    for m in trained["lgb"]:
        all_preds.append(m.predict(X_pred))
    for m in trained["lgb_log"]:
        all_preds.append(np.expm1(np.maximum(m.predict(X_pred), 0)))
    for scaler, ridge in trained["ridge"]:
        X_ph = X_pred[:, phys_idx]
        X_sc = scaler.transform(X_ph)
        all_preds.append(np.maximum(ridge.predict(X_sc), 0))

    ensemble_pred = np.median(np.stack(all_preds, axis=1), axis=1)

    # Physics prediction for floor
    physics_pred = physics_predict(pred_df, physics_params).to_numpy()

    # DHW baseload floor: take maximum of ensemble and DHW_baseload (intercept)
    dhw_floors = pred_df.join(physics_params, on="deviceId", how="left")["dhw_baseload"].to_numpy()
    final_pred = np.maximum(ensemble_pred, dhw_floors * 0.5)  # floor at 50% of fitted DHW

    # Clip to non-negative
    final_pred = np.maximum(final_pred, 0.0)

    pred_df = pred_df.with_columns(pl.Series("pred_daily", final_pred))

    # Heating-only constraint: Jun-Aug ≤ Apr predictions per device
    # Estimate April average from training data
    apr_avg = (
        daily.filter((pl.col("period") == "train") & (pl.col("month") == 4))
        .group_by("deviceId")
        .agg(pl.col("x2_mean").mean().alias("apr_mean"))
    )

    # Aggregate daily → monthly
    monthly = (
        pred_df.group_by(["deviceId", "year", "month"])
        .agg(pl.col("pred_daily").mean().alias("prediction"))
        .sort(["deviceId", "year", "month"])
    )

    # Apply June-August heating cap for heating-only devices (those with small April load)
    monthly = monthly.join(apr_avg, on="deviceId", how="left")
    monthly = monthly.with_columns(
        pl.when(pl.col("month").is_in([6, 7, 8]) & pl.col("apr_mean").is_not_null())
        .then(pl.col("prediction").clip(upper_bound=pl.col("apr_mean")))
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    ).drop("apr_mean")

    # October 2025 sanity: compare with Oct 2024 actuals; cap upward outliers
    oct24_avg = (
        daily.filter((pl.col("period") == "train") & (pl.col("month") == 10))
        .group_by("deviceId")
        .agg(pl.col("x2_mean").mean().alias("oct24_mean"))
    )
    monthly = monthly.join(oct24_avg, on="deviceId", how="left")
    monthly = monthly.with_columns(
        pl.when((pl.col("month") == 10) & pl.col("oct24_mean").is_not_null())
        .then(
            # Blend prediction with Oct 2024 anchor (60/40)
            pl.col("prediction") * 0.6 + pl.col("oct24_mean") * 0.4
        )
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    ).drop("oct24_mean")

    # Select final columns
    result = monthly.select(["deviceId", "year", "month", "prediction"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean
    df, devices = load_data()

    # 2. Fetch weather & join
    weather = fetch_weather(devices)
    df = join_weather(df, devices, weather)

    # 3. Feature engineering
    print("Computing physics features …")
    df = add_physics_features(df)

    print("Aggregating to daily …")
    daily = daily_aggregate(df, devices)
    print(f"Daily rows: {len(daily):,}")

    # 4. Fit physics baseline
    print("Fitting per-device physics baseline …")
    physics_params = fit_physics_baseline(daily)

    # 5. Train ML models
    print("Training ML models …")
    trained = train_models(daily)

    # 6. Build submission
    print("Building submission …")
    submission = build_submission(daily, trained, physics_params)
    print(f"Submission rows: {len(submission):,} (expected 3600 = 600 × 6)")

    submission.write_csv(SUBMISSION_FILE)
    print(f"Saved → {SUBMISSION_FILE}")

    # 7. Submit via API
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

from __future__ import annotations
import numpy as np
import polars as pl

SUMMER_DECAY = {5: 0.80, 6: 0.52, 7: 0.45, 8: 0.48, 9: 0.65, 10: None}

def build_full_grid(devices: pl.DataFrame) -> pl.DataFrame:
    """Build a complete 600 devices × 6 months = 3600-row prediction grid."""
    device_ids = devices["deviceId"].to_list()
    rows = [
        {"deviceId": did, "year": 2025, "month": m}
        for did in device_ids
        for m in [5, 6, 7, 8, 9, 10]
    ]
    return pl.DataFrame(rows)


def fit_physics_model(monthly: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Per-device anchor model using April as the summer baseline."""
    train = monthly.filter(pl.col("period") == "train")

    all_x2 = train["x2_monthly"].drop_nulls().to_numpy()
    global_mean = float(np.median(all_x2)) if len(all_x2) > 0 else 0.10

    apr_train = train.filter(pl.col("month") == 4)
    group_apr_df = (
        apr_train
        .group_by(["deviceType", "x3"])
        .agg(pl.col("x2_monthly").median().alias("group_apr_median"))
    )

    results = []
    for device_grp in train.partition_by("deviceId"):
        device_id = device_grp["deviceId"][0]
        x2 = device_grp["x2_monthly"].to_numpy()
        months = device_grp["month"].to_numpy()
        dt = int(device_grp["deviceType"][0]) if "deviceType" in device_grp.columns else -1
        x3 = int(device_grp["x3"][0]) if "x3" in device_grp.columns else -1

        mask = np.isfinite(x2)
        x2_train_mean = float(np.mean(x2[mask])) if mask.any() else global_mean

        apr_mask = (months == 4) & mask
        if apr_mask.any():
            apr_mean = float(np.mean(x2[apr_mask]))
        else:
            grp_row = group_apr_df.filter((pl.col("deviceType") == dt) & (pl.col("x3") == x3))
            if len(grp_row) > 0:
                apr_mean = float(grp_row["group_apr_median"][0])
            else:
                apr_mean = x2_train_mean * 0.40

        oct_mask = (months == 10) & mask
        oct24_mean = float(np.mean(x2[oct_mask])) if oct_mask.any() else apr_mean
        dhw_baseload = max(apr_mean * 0.50, 0.005)

        results.append({
            "deviceId": device_id,
            "apr_mean": apr_mean,
            "oct24_mean": oct24_mean,
            "dhw_baseload": dhw_baseload,
            "x2_train_mean": x2_train_mean,
        })

    return pl.DataFrame(results)


def predict_physics(full_grid: pl.DataFrame, monthly: pl.DataFrame, physics_params: pl.DataFrame) -> pl.DataFrame:
    """Per-device physics prediction for all 3600 pairs."""
    pred = full_grid.join(physics_params, on="deviceId", how="left")
    monthly_pred = monthly.filter(pl.col("period").is_in(["validation", "test"])).select([
        "deviceId", "year", "month", "HDD_15_monthly", "t1_mean_monthly", "day_length_monthly",
    ])
    pred = pred.join(monthly_pred, on=["deviceId", "year", "month"], how="left")

    global_apr = float(physics_params["apr_mean"].median())
    global_oct = float(physics_params["oct24_mean"].median())
    pred = pred.with_columns([
        pl.col("apr_mean").fill_null(global_apr),
        pl.col("oct24_mean").fill_null(global_oct),
        pl.col("dhw_baseload").fill_null(global_apr * 0.50),
        pl.col("HDD_15_monthly").fill_null(0.0),
    ])

    pred = pred.with_columns(
        pl.when(pl.col("month") == 5).then(pl.col("apr_mean") * 0.80)
        .when(pl.col("month") == 6).then(pl.col("apr_mean") * 0.52)
        .when(pl.col("month") == 7).then(pl.col("apr_mean") * 0.45)
        .when(pl.col("month") == 8).then(pl.col("apr_mean") * 0.48)
        .when(pl.col("month") == 9).then(pl.col("apr_mean") * 0.65)
        .when(pl.col("month") == 10).then(pl.col("oct24_mean") * 0.6 + pl.col("apr_mean") * 0.90 * 0.4)
        .otherwise(pl.col("apr_mean"))
        .alias("phys_base")
    )

    pred = pred.with_columns(
        (pl.col("phys_base") + (pl.col("HDD_15_monthly") - 5.0).clip(lower_bound=0.0) * 0.0003).alias("phys_pred")
    )
    pred = pred.with_columns(pl.col("phys_pred").clip(lower_bound=pl.col("dhw_baseload")).alias("phys_pred"))
    pred = pred.with_columns(pl.col("phys_pred").clip(upper_bound=pl.col("oct24_mean")).alias("phys_pred"))

    return pred.select(["deviceId", "year", "month", "phys_pred"])


def validate_physics_on_train(monthly: pl.DataFrame, physics_params: pl.DataFrame) -> None:
    """Self-check: compute physics MAE on training months."""
    train = monthly.filter(pl.col("period") == "train")
    TRAIN_DECAY = {10: None, 11: 2.2, 12: 2.8, 1: 2.8, 2: 3.0, 3: 1.8, 4: 1.0}

    errors = []
    for device_grp in train.partition_by("deviceId"):
        device_id = device_grp["deviceId"][0]
        row = physics_params.filter(pl.col("deviceId") == device_id)
        if len(row) == 0:
            continue
        apr = float(row["apr_mean"][0])
        oct24 = float(row["oct24_mean"][0])

        months_arr = device_grp["month"].to_numpy()
        x2_arr = device_grp["x2_monthly"].to_numpy()
        mask = np.isfinite(x2_arr)

        for m, x2_actual in zip(months_arr, x2_arr):
            if not np.isfinite(x2_actual):
                continue
            if m == 10:
                pred = oct24
            elif m in TRAIN_DECAY:
                factor = TRAIN_DECAY[m]
                pred = apr * factor if factor is not None else apr
            else:
                continue
            errors.append(abs(x2_actual - pred))

    if errors:
        mae = np.mean(errors)
        print(f"  Physics self-check MAE on train: {mae:.4f}  (n={len(errors)})")
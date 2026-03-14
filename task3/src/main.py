from __future__ import annotations
import numpy as np
import polars as pl
import lightgbm as lgb

from config import OUT_DIR, SUBMISSION_FILE
from data import load_data, fetch_weather, join_weather
from features import add_physics_features, daily_aggregate, build_monthly_features
from physics import build_full_grid, fit_physics_model, predict_physics, validate_physics_on_train
from lgbm import train_monthly_lgb, MONTHLY_FEATURE_COLS


def build_submission(
    full_grid: pl.DataFrame,
    monthly: pl.DataFrame,
    monthly_lgb_models: list[lgb.Booster],
    physics_params: pl.DataFrame,
) -> pl.DataFrame:
    """Weighted ensemble: 0.7 * physics + 0.3 * monthly_lgb."""
    print("  Model A: per-device physics prediction …")
    pred_a_df = predict_physics(full_grid, monthly, physics_params)

    print("  Model B: monthly LightGBM inference …")
    pred_monthly = monthly.filter(pl.col("period").is_in(["validation", "test"]))

    if monthly_lgb_models and len(pred_monthly) > 0:
        X_pred = pred_monthly.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
        lgb_preds = np.stack([m.predict(X_pred) for m in monthly_lgb_models], axis=1)
        lgb_ensemble = np.median(lgb_preds, axis=1)
        pred_b_df = pred_monthly.select(["deviceId", "year", "month"]).with_columns(
            pl.Series("lgb_pred", np.maximum(lgb_ensemble, 0.0))
        )
    else:
        pred_b_df = pred_a_df.select(["deviceId", "year", "month"]).with_columns(
            pl.col("phys_pred").alias("lgb_pred")
        )

    combined = pred_a_df.join(pred_b_df, on=["deviceId", "year", "month"], how="left")
    combined = combined.with_columns(pl.col("lgb_pred").fill_null(pl.col("phys_pred")))
    combined = combined.with_columns((pl.col("phys_pred") * 0.7 + pl.col("lgb_pred") * 0.3).alias("prediction"))
    combined = combined.with_columns(pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0))

    return combined.select(["deviceId", "year", "month", "prediction"]).sort(["deviceId", "year", "month"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, devices = load_data()
    weather = fetch_weather(devices)
    has_weather = weather is not None

    if has_weather:
        print("Joining weather data …")
        df = join_weather(df, devices, weather)
    else:
        print("No weather data — using t1-based physics fallback")
        df = df.join(devices.select(["deviceId", "latitude", "longitude"]), on="deviceId", how="left")

    print("Computing physics features …")
    df = add_physics_features(df, has_weather)

    print("Aggregating to daily …")
    daily = daily_aggregate(df, devices)
    print(f"Daily rows: {len(daily):,}")

    print("Building monthly features …")
    monthly = build_monthly_features(daily)
    n_train = monthly.filter(pl.col("period") == "train")["month"].n_unique()
    print(f"Monthly rows: {len(monthly):,}  (train months: {n_train})")

    print("Fitting per-device physics model …")
    physics_params = fit_physics_model(monthly, devices)
    print(f"  Devices fitted: {len(physics_params)}")
    
    validate_physics_on_train(monthly, physics_params)

    print("Training monthly LightGBM …")
    monthly_lgb_models = train_monthly_lgb(monthly)

    print("Building submission …")
    full_grid = build_full_grid(devices)
    submission = build_submission(full_grid, monthly, monthly_lgb_models, physics_params)
    
    n = len(submission)
    print(f"Submission rows: {n:,} (expected 3600 = 600 × 6)")

    submission.write_csv(SUBMISSION_FILE)
    print(f"Saved → {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
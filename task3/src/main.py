from __future__ import annotations
import polars as pl

try:
    from src.config import OUT_DIR, SUBMISSION_FILE
    from src.data import load_raw, load_weather, join_weather
    from src.model import train_and_predict
except ModuleNotFoundError:
    from config import OUT_DIR, SUBMISSION_FILE
    from data import load_raw, load_weather, join_weather
    from model import train_and_predict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dane
    df, devices = load_raw()
    weather = load_weather()
    has_weather = weather is not None

    if has_weather:
        print("Joining weather …")
        df = join_weather(df, devices, weather)

    # 2. Trenuj i predykuj
    print("Training per-device Ridge model …")
    monthly_preds = train_and_predict(df, has_weather=has_weather)

    # 3. Pełna siatka 600 × 6 = 3600 wierszy
    device_ids = devices["deviceId"].to_list()
    full_grid = pl.DataFrame([
        {"deviceId": did, "year": 2025, "month": m}
        for did in device_ids
        for m in [5, 6, 7, 8, 9, 10]
    ])

    # Fallback dla brakujących urządzeń/miesięcy
    global_median = float(monthly_preds["prediction"].median())
    device_median = (
        monthly_preds
        .group_by("deviceId")
        .agg(pl.col("prediction").median().alias("fallback"))
    )

    submission = (
        full_grid
        .join(monthly_preds, on=["deviceId", "year", "month"], how="left")
        .join(device_median, on="deviceId", how="left")
        .with_columns(
            pl.when(pl.col("prediction").is_null())
            .then(pl.col("fallback").fill_null(global_median))
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        .select(["deviceId", "year", "month", "prediction"])
        .sort(["deviceId", "year", "month"])
    )

    print(f"\nSubmission rows: {len(submission)} (expected 3600)")
    print(f"Null predictions: {submission['prediction'].null_count()}")
    print(f"Stats:\n{submission['prediction'].describe()}")

    submission.write_csv(SUBMISSION_FILE)
    print(f"\n✅ Saved → {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
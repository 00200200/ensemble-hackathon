"""
eval.py — lokalny holdout, proxy leaderboardu.
Uruchom z roota: uv run src/eval.py
"""
from __future__ import annotations
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "src")
from data import load_raw, load_weather, join_weather
from model import build_features, SENSOR_FEATS
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HOLDOUT_YEAR  = 2025
HOLDOUT_MONTH = 4


def run_eval():
    df, devices = load_raw()
    weather = load_weather()
    has_weather = weather is not None
    if has_weather:
        df = join_weather(df, devices, weather)

    df = df.with_columns([
        pl.col("timedate").dt.year().alias("year"),
        pl.col("timedate").dt.month().alias("month"),
    ])

    train_all = df.filter(pl.col("period") == "train").drop_nulls(SENSOR_FEATS + ["x2"])
    context      = train_all.filter(~((pl.col("year") == HOLDOUT_YEAR) & (pl.col("month") == HOLDOUT_MONTH)))
    holdout_5min = train_all.filter((pl.col("year") == HOLDOUT_YEAR) & (pl.col("month") == HOLDOUT_MONTH))

    holdout_actual = holdout_5min.group_by("deviceId").agg(pl.col("x2").mean().alias("x2_actual"))

    months_ctx = (
        context.with_columns(pl.col("timedate").dt.strftime("%Y-%m").alias("ym"))
        ["ym"].unique().sort().to_list()
    )
    print(f"Kontekst: {months_ctx}")
    print(f"Holdout:  {HOLDOUT_YEAR}-{HOLDOUT_MONTH:02d}  ({len(holdout_actual)} urzadzen)")

    sc_global = StandardScaler()
    global_model = Ridge(alpha=1.0).fit(
        sc_global.fit_transform(build_features(context, has_weather)),
        context["x2"].to_numpy()
    )

    all_preds = []
    for (did,), dev_ctx in context.group_by("deviceId"):
        dev_ho = holdout_5min.filter(pl.col("deviceId") == did)
        if len(dev_ho) == 0:
            continue
        X_te = build_features(dev_ho, has_weather)
        if len(dev_ctx) >= 200:
            sc = StandardScaler()
            preds = np.maximum(
                Ridge(alpha=0.1)
                .fit(sc.fit_transform(build_features(dev_ctx, has_weather)), dev_ctx["x2"].to_numpy())
                .predict(sc.transform(X_te)),
                0.0
            )
        else:
            preds = np.maximum(global_model.predict(sc_global.transform(X_te)), 0.0)
        all_preds.append({"deviceId": did, "x2_pred": float(np.mean(preds))})

    eval_df = holdout_actual.join(pl.DataFrame(all_preds), on="deviceId", how="inner")

    mae    = float((eval_df["x2_actual"] - eval_df["x2_pred"]).abs().mean())
    mse    = float(((eval_df["x2_actual"] - eval_df["x2_pred"]) ** 2).mean())
    bias   = float((eval_df["x2_pred"] - eval_df["x2_actual"]).mean())
    errors = (eval_df["x2_actual"] - eval_df["x2_pred"]).abs()

    baseline = context.group_by("deviceId").agg(pl.col("x2").mean().alias("b"))
    mae_base = float(
        holdout_actual.join(baseline, on="deviceId", how="inner")
        .select((pl.col("x2_actual") - pl.col("b")).abs().mean()).item()
    )

    print(f"\n{'='*55}")
    print(f"  Holdout Apr 2025  (proxy leaderboard May-Jun)")
    print(f"  MAE  : {mae:.5f}  (topka: ~0.0017)")
    print(f"  MSE  : {mse:.5f}")
    print(f"  RMSE : {mse**0.5:.5f}")
    print(f"  Bias : {bias:+.5f}  ({'zawyza' if bias > 0 else 'zaniza'})")
    print(f"  n    : {len(eval_df)} urzadzen")
    print(f"{'='*55}")
    print(f"  Baseline (device mean): MAE={mae_base:.5f}")
    print(f"  Delta vs baseline     : {mae - mae_base:+.5f}")
    print(f"\n  Rozklad bledow:")
    print(f"    p25={errors.quantile(0.25):.4f}  "
          f"p50={errors.quantile(0.50):.4f}  "
          f"p75={errors.quantile(0.75):.4f}  "
          f"p95={errors.quantile(0.95):.4f}")
    return mae


if __name__ == "__main__":
    run_eval()
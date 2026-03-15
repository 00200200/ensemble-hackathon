"""
test_seasonal_split.py
Testuje czy osobny model dla ciepłych miesięcy poprawia MAE
dla urządzeń z wysoką wariancją x2.
Uruchom z roota: uv run test_seasonal_split.py
"""
import sys
sys.path.insert(0, "src")
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.data import load_raw, load_weather, join_weather
from src.model import build_features, SENSOR_FEATS

df, devices = load_raw()
weather = load_weather()
df = join_weather(df, devices, weather)
df = df.with_columns([
    pl.col("timedate").dt.year().alias("year"),
    pl.col("timedate").dt.month().alias("month"),
])

train = df.filter(pl.col("period") == "train").drop_nulls(SENSOR_FEATS + ["x2"])
holdout = train.filter((pl.col("year") == 2025) & (pl.col("month") == 4))
context = train.filter(~((pl.col("year") == 2025) & (pl.col("month") == 4)))

WARM_MONTHS = [3, 4, 10]
STD_THRESHOLD = 0.25

errors_baseline = []
errors_split    = []

for (did,), grp_ctx in context.group_by("deviceId"):
    grp_ho = holdout.filter(pl.col("deviceId") == did)
    if len(grp_ho) < 10 or len(grp_ctx) < 200:
        continue

    x2_mean = float(grp_ctx["x2"].mean())
    x2_std  = float(grp_ctx["x2"].std())
    mar = grp_ctx.filter(pl.col("month") == 3)["x2"]
    lag1 = float(mar.mean()) if len(mar) > 0 else x2_mean
    feb = grp_ctx.filter(pl.col("month") == 2)["x2"]
    lag2 = float(feb.mean()) if len(feb) > 0 else lag1

    grp_ctx2 = grp_ctx.with_columns([
        pl.col("x2_lag1").fill_null(x2_mean),
        pl.col("x2_lag2").fill_null(x2_mean),
    ])
    grp_ho2 = grp_ho.with_columns([
        pl.lit(lag1).alias("x2_lag1"),
        pl.lit(lag2).alias("x2_lag2"),
    ])

    y_te = float(grp_ho2["x2"].mean())
    X_te = build_features(grp_ho2)

    # Baseline: jeden model na wszystkich danych
    sc = StandardScaler()
    X_tr = sc.fit_transform(build_features(grp_ctx2))
    model = Ridge(alpha=0.1).fit(X_tr, grp_ctx2["x2"].to_numpy())
    pred_base = max(float(model.predict(sc.transform(X_te)).mean()), 0)
    errors_baseline.append(abs(pred_base - y_te))

    # Split: dla urządzeń z wysoką wariancją — model tylko na ciepłych miesiącach
    if x2_std > STD_THRESHOLD:
        dev_warm = grp_ctx2.filter(pl.col("month").is_in(WARM_MONTHS))
        if len(dev_warm) >= 100:
            sc_w = StandardScaler()
            X_w = sc_w.fit_transform(build_features(dev_warm))
            model_w = Ridge(alpha=0.1).fit(X_w, dev_warm["x2"].to_numpy())
            pred_split = max(float(model_w.predict(sc_w.transform(X_te)).mean()), 0)
        else:
            pred_split = pred_base
    else:
        pred_split = pred_base

    errors_split.append(abs(pred_split - y_te))

a = np.array(errors_baseline)
b = np.array(errors_split)
print(f"Baseline (jeden model):  MAE={a.mean():.5f}")
print(f"Seasonal split:          MAE={b.mean():.5f}")
print(f"Delta:                   {b.mean()-a.mean():+.5f}")
print()

# Analiza tylko dla urządzeń z wysoką wariancją
high_std_mask = []
i = 0
for (did,), grp_ctx in context.group_by("deviceId"):
    grp_ho = holdout.filter(pl.col("deviceId") == did)
    if len(grp_ho) < 10 or len(grp_ctx) < 200:
        continue
    high_std_mask.append(float(grp_ctx["x2"].std()) > STD_THRESHOLD)

mask = np.array(high_std_mask)
print(f"Tylko urzadzenia z x2_std > {STD_THRESHOLD} ({mask.sum()} urzadzen):")
print(f"  Baseline: MAE={a[mask].mean():.5f}")
print(f"  Split:    MAE={b[mask].mean():.5f}")
print(f"  Delta:    {(b[mask].mean()-a[mask].mean()):+.5f}")
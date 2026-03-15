from __future__ import annotations
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

SENSOR_FEATS = ["t4", "t5", "t8", "t13", "t6", "t12", "t1", "t10", "t11"]
DELTA_FEATS  = ["dt4", "dt8"]


def build_features(df: pl.DataFrame, has_weather: bool = False) -> np.ndarray:
    parts = [df[SENSOR_FEATS].fill_null(0).to_numpy()]
    for col in DELTA_FEATS:
        if col in df.columns:
            parts.append(df[col].fill_null(0).to_numpy().reshape(-1, 1))
    month = df["month"].to_numpy()
    parts.append(np.sin(2 * np.pi * month / 12).reshape(-1, 1))
    parts.append(np.cos(2 * np.pi * month / 12).reshape(-1, 1))
    if has_weather and "solar_rad" in df.columns:
        solar = df["solar_rad"].fill_null(0).to_numpy().reshape(-1, 1)
        parts.append(solar / (solar.max() + 1e-6))
    return np.hstack(parts).astype(np.float32)


def _fit_two_stage(dev_train: pl.DataFrame, has_weather: bool):
    """
    Two-stage per-device:
      Stage 1: LogisticRegression → P(pompa ON)
      Stage 2: Ridge              → E[x2 | pompa ON]
    Fallback: Ridge na wszystkich danych gdy brak klas.
    """
    X = build_features(dev_train, has_weather)
    y = dev_train["x2"].to_numpy()
    y_binary = (y > 0.001).astype(int)
    n_on  = int(y_binary.sum())
    n_off = int(len(y_binary) - n_on)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    if n_on < 50 or n_off < 50:
        return {"type": "ridge", "model": Ridge(alpha=0.1).fit(X_s, y), "sc": sc}
    clf = LogisticRegression(max_iter=300, C=1.0).fit(X_s, y_binary)
    on_mask = y_binary == 1
    sc2 = StandardScaler()
    X_on_s = sc2.fit_transform(X[on_mask])
    reg = Ridge(alpha=0.1).fit(X_on_s, y[on_mask])
    return {"type": "two_stage", "clf": clf, "reg": reg, "sc": sc, "sc2": sc2}


def _predict_two_stage(fitted: dict, X_te_raw: np.ndarray) -> np.ndarray:
    if fitted["type"] == "ridge":
        return np.maximum(fitted["model"].predict(fitted["sc"].transform(X_te_raw)), 0.0)
    X_s  = fitted["sc"].transform(X_te_raw)
    X_s2 = fitted["sc2"].transform(X_te_raw)
    prob_on      = np.array(fitted["clf"].predict_proba(X_s)[:, 1])
    pred_when_on = np.maximum(np.array(fitted["reg"].predict(X_s2)), 0.0)
    return prob_on * pred_when_on


def _bar(current: int, total: int, width: int = 35) -> str:
    pct    = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█'*filled}{'░'*(width-filled)}] {current}/{total} ({pct*100:.0f}%)"


def train_and_predict(df: pl.DataFrame, has_weather: bool = False) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("timedate").dt.year().alias("year"),
        pl.col("timedate").dt.month().alias("month"),
    ])

    train_df = df.filter(pl.col("period") == "train").drop_nulls(SENSOR_FEATS + ["x2"])
    pred_df  = df.filter(pl.col("period").is_in(["valid", "test"])).drop_nulls(SENSOR_FEATS)

    print(f"  Train : {len(train_df):>12,} wierszy")
    print(f"  Pred  : {len(pred_df):>12,} wierszy")

    print("Trenuję globalny fallback …")
    global_fitted = _fit_two_stage(
        train_df.sample(min(50_000, len(train_df)), seed=42), has_weather
    )
    print("  ✓ fallback gotowy")

    device_ids = train_df["deviceId"].unique().to_list()
    total = len(device_ids)
    print(f"Trenuję per-device two-stage ({total} urządzeń) …")

    all_preds: list[pl.DataFrame] = []
    n_two_stage = 0
    n_fallback  = 0

    for i, device_id in enumerate(device_ids):
        dev_train = train_df.filter(pl.col("deviceId") == device_id)
        dev_pred  = pred_df.filter(pl.col("deviceId") == device_id)
        if len(dev_pred) == 0:
            continue
        X_te = build_features(dev_pred, has_weather)
        if len(dev_train) >= 200:
            fitted = _fit_two_stage(dev_train, has_weather)
            if fitted["type"] == "two_stage":
                n_two_stage += 1
            else:
                n_fallback += 1
        else:
            fitted = global_fitted
            n_fallback += 1
        preds = _predict_two_stage(fitted, X_te)
        all_preds.append(
            dev_pred.select(["deviceId", "year", "month"])
            .with_columns(pl.Series("x2_pred", np.array(preds).astype(np.float64)))
        )
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"\r  {_bar(i+1, total)}", end="", flush=True)

    print(f"\n  ✓ two-stage: {n_two_stage}  |  fallback: {n_fallback}")

    result = (
        pl.concat(all_preds)
        .group_by(["deviceId", "year", "month"])
        .agg(pl.col("x2_pred").mean().alias("prediction"))
        .with_columns(pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0))
        .sort(["deviceId", "year", "month"])
    )
    print(f"  Predykcji: {len(result):,}  |  null: {result['prediction'].null_count()}")
    return result
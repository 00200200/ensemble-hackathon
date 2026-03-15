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


THRESHOLDS = [0.0001, 0.001, 0.005, 0.01, 0.05]


def _fit_two_stage(dev_train: pl.DataFrame, has_weather: bool, threshold: float = 0.001):
    """
    Two-stage model:
    Stage 1: LogisticRegression → P(pompa pracuje)
    Stage 2: Ridge → E[x2 | pompa pracuje]
    """
    X = build_features(dev_train, has_weather)
    y = dev_train["x2"].to_numpy()
    y_binary = (y > threshold).astype(int)
    n_on  = y_binary.sum()
    n_off = len(y_binary) - n_on

    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    # Jeśli za mało danych dla dwóch klas — fallback do Ridge
    if n_on < 50 or n_off < 50:
        model = Ridge(alpha=0.1).fit(X_s, y)
        return {"type": "ridge", "model": model, "sc": sc}

    # Stage 1
    clf = LogisticRegression(max_iter=300, C=1.0).fit(X_s, y_binary)

    # Stage 2 — tylko na wierszach gdy pompa pracuje
    on_mask = y_binary == 1
    sc2 = StandardScaler()
    X_on_s = sc2.fit_transform(X[on_mask])
    reg = Ridge(alpha=0.1).fit(X_on_s, y[on_mask])

    return {"type": "two_stage", "clf": clf, "reg": reg, "sc": sc, "sc2": sc2}


def _find_best_threshold(dev_train: pl.DataFrame, has_weather: bool) -> float:
    """Znajdź próg który minimalizuje błąd na ostatnim miesiącu trainu (marzec)."""
    mar = dev_train.filter(pl.col("month") == 3)
    if len(mar) < 50:
        return 0.001

    # Trenuj na wszystkim poza marcem, testuj na marcu
    ctx = dev_train.filter(pl.col("month") != 3)
    if len(ctx) < 200:
        return 0.001

    # Subsample dla szybkości
    if len(ctx) > 10000:
        ctx = ctx.sample(10000, seed=42)

    X_ctx = build_features(ctx, has_weather)
    y_ctx = ctx["x2"].to_numpy()
    X_mar = build_features(mar, has_weather)
    y_mar = float(mar["x2"].mean())

    best_t, best_err = 0.001, float("inf")
    for t in THRESHOLDS:
        fitted = _fit_two_stage(ctx.with_columns(pl.col("month")), has_weather, threshold=t)
        # Użyj już obliczonych X
        y_bin = (y_ctx > t).astype(int)
        n_on = y_bin.sum()
        n_off = len(y_bin) - n_on
        if n_on < 30 or n_off < 30:
            continue
        pred = float(np.mean(_predict_two_stage(fitted, X_mar)))
        err = abs(pred - y_mar)
        if err < best_err:
            best_err, best_t = err, t

    return best_t


def _predict_two_stage(fitted: dict, X_te_raw: np.ndarray) -> np.ndarray:
    if fitted["type"] == "ridge":
        X_s = fitted["sc"].transform(X_te_raw)
        return np.maximum(fitted["model"].predict(X_s), 0.0)

    X_s  = fitted["sc"].transform(X_te_raw)
    X_s2 = fitted["sc2"].transform(X_te_raw)
    prob_on      = fitted["clf"].predict_proba(X_s)[:, 1]
    pred_when_on = np.maximum(fitted["reg"].predict(X_s2), 0.0)
    return prob_on * pred_when_on


def train_and_predict(df: pl.DataFrame, has_weather: bool = False) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("timedate").dt.year().alias("year"),
        pl.col("timedate").dt.month().alias("month"),
    ])

    train_df = df.filter(pl.col("period") == "train").drop_nulls(SENSOR_FEATS + ["x2"])
    pred_df  = df.filter(pl.col("period").is_in(["valid", "test"])).drop_nulls(SENSOR_FEATS)

    print(f"Train 5-min rows: {len(train_df):,}")
    print(f"Pred  5-min rows: {len(pred_df):,}")

    # Globalny fallback
    print("Trenuję globalny model fallback …")
    global_fitted = _fit_two_stage(train_df, has_weather)

    device_ids = train_df["deviceId"].unique().to_list()
    print(f"Trenuję per-device two-stage model dla {len(device_ids)} urządzeń …")

    all_preds: list[pl.DataFrame] = []
    n_two_stage = 0
    n_fallback  = 0

    for device_id in device_ids:
        dev_train = train_df.filter(pl.col("deviceId") == device_id)
        dev_pred  = pred_df.filter(pl.col("deviceId") == device_id)

        if len(dev_pred) == 0:
            continue

        X_te = build_features(dev_pred, has_weather)

        if len(dev_train) >= 200:
            best_t = _find_best_threshold(dev_train, has_weather)
            fitted = _fit_two_stage(dev_train, has_weather, threshold=best_t)
            preds  = _predict_two_stage(fitted, X_te)
            if fitted["type"] == "two_stage":
                n_two_stage += 1
            else:
                n_fallback += 1
        else:
            preds = _predict_two_stage(global_fitted, X_te)
            n_fallback += 1

        all_preds.append(
            dev_pred.select(["deviceId", "year", "month"])
            .with_columns(pl.Series("x2_pred", preds))
        )

    print(f"  Two-stage: {n_two_stage}  |  Ridge fallback: {n_fallback}")

    return (
        pl.concat(all_preds)
        .group_by(["deviceId", "year", "month"])
        .agg(pl.col("x2_pred").mean().alias("prediction"))
        .with_columns(pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0))
        .sort(["deviceId", "year", "month"])
    )
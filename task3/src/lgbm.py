from __future__ import annotations
import numpy as np
import polars as pl
import lightgbm as lgb

MONTHLY_FEATURE_COLS = [
    "month", "temp_mean_monthly", "temp_min_monthly", "HDD_15_monthly", 
    "HDD_18_monthly", "CDD_22_monthly", "load_proxy_monthly", "solar_monthly", 
    "day_length_monthly", "inv_cop_monthly", "t1_mean_monthly", "t1_min_monthly", 
    "x1_mean_monthly", "frac_heating_season", "x3", "deviceType",
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
    """Temporal forward-chaining CV."""
    month_col = monthly_train["month"].to_numpy()
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
    """Train LightGBM on monthly data."""
    train_monthly = monthly.filter(pl.col("period") == "train").fill_null(0)
    X = train_monthly.select(MONTHLY_FEATURE_COLS).to_numpy().astype(np.float32)
    y = train_monthly["x2_monthly"].to_numpy().astype(np.float64)
    w = train_monthly["sample_weight"].to_numpy().astype(np.float64)

    folds = get_cv_folds(train_monthly)
    print(f"  Monthly CV folds: {len(folds)}, train rows: {len(y)}")

    models = []
    for fold_i, (ti, vi) in enumerate(folds):
        dtrain = lgb.Dataset(X[ti], label=y[ti], weight=w[ti], feature_name=MONTHLY_FEATURE_COLS, free_raw_data=False)
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
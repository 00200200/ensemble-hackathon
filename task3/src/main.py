from __future__ import annotations
import numpy as np
import polars as pl
import lightgbm as lgb
import hydra
from omegaconf import DictConfig

from config import OUT_DIR, SUBMISSION_FILE
from data import load_data, fetch_weather, join_weather
from features import add_physics_features, daily_aggregate, build_monthly_features
from physics import build_full_grid, fit_physics_model, predict_physics, validate_physics_on_train
from lgbm import train_monthly_lgb, MONTHLY_FEATURE_COLS
from chronos_model import predict_chronos, predict_chronos_april_holdout


def _normalize_weights(phys: float, lgbm: float, chronos: float) -> tuple[float, float, float]:
    vals = np.array([phys, lgbm, chronos], dtype=np.float64)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= 0:
        return 0.50, 0.30, 0.20
    vals /= s
    return float(vals[0]), float(vals[1]), float(vals[2])


def build_submission(
    full_grid: pl.DataFrame,
    monthly: pl.DataFrame,
    monthly_lgb_models: list[lgb.Booster],
    physics_params: pl.DataFrame,
    chronos_preds: pl.DataFrame | None = None,
    weights_abc: tuple[float, float, float] = (0.50, 0.30, 0.20),
    weights_ab: tuple[float, float] = (0.67, 0.33),
) -> pl.DataFrame:
    """
    Ensemble with configurable weights.
    If chronos_preds is None, falls back to AB weights.
    """
    w_phys_abc, w_lgb_abc, w_chr_abc = weights_abc
    w_phys_ab, w_lgb_ab = weights_ab

    # --- MODEL A: Physics ---
    print("  Model A: per-device physics prediction …")
    pred_a_df = predict_physics(full_grid, monthly, physics_params)

    # --- MODEL B: LightGBM ---
    print("  Model B: monthly LightGBM inference …")
    pred_monthly = monthly.filter(pl.col("period").is_in(["validation", "test"]))

    if monthly_lgb_models and len(pred_monthly) > 0:
        X_pred = pred_monthly.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
        lgb_preds_log = np.stack([m.predict(X_pred) for m in monthly_lgb_models], axis=1)
        lgb_preds_real = np.expm1(lgb_preds_log)
        lgb_ensemble = np.median(lgb_preds_real, axis=1)
        
        pred_b_df = pred_monthly.select(["deviceId", "year", "month"]).with_columns(
            pl.Series("lgb_pred", np.maximum(lgb_ensemble, 0.0))
        )
    else:
        pred_b_df = pred_a_df.select(["deviceId", "year", "month"]).with_columns(
            pl.col("phys_pred").alias("lgb_pred")
        )

    # --- Połączenie A i B ---
    combined = pred_a_df.join(pred_b_df, on=["deviceId", "year", "month"], how="left")
    combined = combined.with_columns(pl.col("lgb_pred").fill_null(pl.col("phys_pred")))

    # --- MODEL C: Chronos ---
    if chronos_preds is not None and len(chronos_preds) > 0:
        combined = combined.join(chronos_preds, on=["deviceId", "year", "month"], how="left")
        combined = combined.with_columns(
            pl.col("chronos_pred").fill_null(pl.col("phys_pred"))
        )
        combined = combined.with_columns(
            (
                pl.col("phys_pred") * w_phys_abc +
                pl.col("lgb_pred")  * w_lgb_abc +
                pl.col("chronos_pred") * w_chr_abc
            ).alias("prediction")
        )
    else:
        # Fallback AB
        combined = combined.with_columns(
            (
                pl.col("phys_pred") * w_phys_ab +
                pl.col("lgb_pred") * w_lgb_ab
            ).alias("prediction")
        )

    # Finalny szlif: clip do [0, 1]
    combined = combined.with_columns(pl.col("prediction").clip(lower_bound=0.0, upper_bound=1.0))

    return combined.select(["deviceId", "year", "month", "prediction"]).sort(["deviceId", "year", "month"])


def evaluate_on_train(
    monthly: pl.DataFrame,
    monthly_lgb_models: list[lgb.Booster],
    physics_params: pl.DataFrame,
    chronos_cv_mae: float | None = None,
    chronos_apr_holdout: pl.DataFrame | None = None,
    weights_abc: tuple[float, float, float] = (0.50, 0.30, 0.20),
    weights_ab: tuple[float, float] = (0.67, 0.33),
) -> None:
    """Liczy MAE na danych treningowych dla Physics, LightGBM i ensemble A+B+C."""
    w_phys_abc, w_lgb_abc, w_chr_abc = weights_abc
    w_phys_ab, w_lgb_ab = weights_ab

    train = monthly.filter(pl.col("period") == "train").filter(pl.col("x2_monthly").is_not_null())
    if len(train) == 0:
        print("  No training ground truth available for MAE evaluation.")
        return

    y_true = train["x2_monthly"].to_numpy()

    # --- Model A: Physics ---
    from physics import predict_physics
    train_grid = train.select(["deviceId", "year", "month"])
    phys_preds = predict_physics(train_grid, monthly, physics_params)
    phys_joined = train.join(phys_preds, on=["deviceId", "year", "month"], how="left")
    mae_a = float(np.mean(np.abs(phys_joined["x2_monthly"].to_numpy() - phys_joined["phys_pred"].fill_null(0).to_numpy())))

    # --- Model B: LightGBM ---
    from lgbm import MONTHLY_FEATURE_COLS
    if monthly_lgb_models:
        X_train = train.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
        lgb_preds_stack = np.stack([m.predict(X_train) for m in monthly_lgb_models], axis=1)
        lgb_preds = np.maximum(np.expm1(np.median(lgb_preds_stack, axis=1)), 0.0)
        mae_b = float(np.mean(np.abs(y_true - lgb_preds)))
    else:
        lgb_preds = phys_joined["phys_pred"].fill_null(0).to_numpy()
        mae_b = mae_a

    # --- Ensemble A+B (bez N-BEATSx) ---
    phys_arr = phys_joined["phys_pred"].fill_null(0).to_numpy()
    ensemble_preds = np.clip(phys_arr * w_phys_ab + lgb_preds * w_lgb_ab, 0.0, 1.0)
    mae_ens = float(np.mean(np.abs(y_true - ensemble_preds)))

    chronos_line = f"  Chronos hold-out Apr MAE: {chronos_cv_mae:.4f}" if chronos_cv_mae is not None else ""

    # --- Minimal check: czy Chronos psuje wynik na ostatnim miesiącu treningu (April 2025) ---
    ablation_line = ""
    apr = train.filter((pl.col("year") == 2025) & (pl.col("month") == 4))
    if len(apr) > 0:
        apr_grid = apr.select(["deviceId", "year", "month"])
        apr_phys = predict_physics(apr_grid, monthly, physics_params).select(["deviceId", "year", "month", "phys_pred"])
        apr_joined = apr.join(apr_phys, on=["deviceId", "year", "month"], how="left")

        if monthly_lgb_models:
            X_apr = apr_joined.select(MONTHLY_FEATURE_COLS).fill_null(0).to_numpy().astype(np.float32)
            lgb_apr_stack = np.stack([m.predict(X_apr) for m in monthly_lgb_models], axis=1)
            lgb_apr = np.maximum(np.expm1(np.median(lgb_apr_stack, axis=1)), 0.0)
        else:
            lgb_apr = apr_joined["phys_pred"].fill_null(0).to_numpy()

        y_apr = apr_joined["x2_monthly"].to_numpy()
        phys_apr = apr_joined["phys_pred"].fill_null(0).to_numpy()

        # Baseline bez Chronos
        pred_ab = np.clip(phys_apr * w_phys_ab + lgb_apr * w_lgb_ab, 0.0, 1.0)
        mae_ab = float(np.mean(np.abs(y_apr - pred_ab)))

        # Z Chronos, tylko gdy mamy hold-out predykcje April
        if chronos_apr_holdout is not None and len(chronos_apr_holdout) > 0:
            apr_with_chronos = apr_joined.join(
                chronos_apr_holdout.select(["deviceId", "year", "month", "chronos_pred"]),
                on=["deviceId", "year", "month"],
                how="left",
            ).with_columns(pl.col("chronos_pred").fill_null(pl.col("phys_pred")))

            chronos_apr = apr_with_chronos["chronos_pred"].to_numpy()
            pred_abc = np.clip(phys_apr * w_phys_abc + lgb_apr * w_lgb_abc + chronos_apr * w_chr_abc, 0.0, 1.0)
            mae_abc = float(np.mean(np.abs(y_apr - pred_abc)))
            delta = mae_abc - mae_ab
            sign = "+" if delta >= 0 else ""
            ablation_line = (
                f"  April check  │  MAE AB: {mae_ab:.4f}  │  MAE ABC: {mae_abc:.4f}  │  Δ(ABC-AB): {sign}{delta:.4f}"
            )
        else:
            ablation_line = f"  April check  │  MAE AB (bez Chronos): {mae_ab:.4f}"

    print(f"\n{'─'*60}")
    print(f"  Train MAE  │  Physics: {mae_a:.4f}  │  LightGBM: {mae_b:.4f}  │  Ens(A+B): {mae_ens:.4f}")
    if chronos_line:
        print(chronos_line)
    if ablation_line:
        print(ablation_line)
    print(f"{'─'*60}\n")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    w_phys_abc, w_lgb_abc, w_chr_abc = _normalize_weights(
        float(cfg.ensemble.weights.physics),
        float(cfg.ensemble.weights.lgbm),
        float(cfg.ensemble.weights.chronos),
    )
    w_phys_ab, w_lgb_ab, _ = _normalize_weights(
        float(cfg.ensemble.fallback_weights.physics),
        float(cfg.ensemble.fallback_weights.lgbm),
        0.0,
    )
    use_chronos = bool(cfg.chronos.enabled)
    print(
        f"Ensemble weights: ABC=({w_phys_abc:.2f}, {w_lgb_abc:.2f}, {w_chr_abc:.2f}), "
        f"AB=({w_phys_ab:.2f}, {w_lgb_ab:.2f}), Chronos enabled={use_chronos}"
    )

    # 1. Dane i Pogoda
    df, devices = load_data()
    weather = fetch_weather(devices)
    has_weather = weather is not None

    if has_weather:
        print("Joining weather data …")
        df = join_weather(df, devices, weather)
    else:
        print("No weather data — using t1-based physics fallback")
        df = df.join(devices.select(["deviceId", "latitude", "longitude"]), on="deviceId", how="left")

    # 2. Feature Engineering i Agregacja
    print("Computing physics features …")
    df = add_physics_features(df, has_weather)

    print("Aggregating to daily …")
    daily = daily_aggregate(df, devices)
    
    print("Building monthly features …")
    monthly = build_monthly_features(daily)

    # 3. Trening Modelu A (Physics)
    print("Fitting per-device physics model …")
    physics_params = fit_physics_model(monthly, devices)
    validate_physics_on_train(monthly, physics_params)

    # 4. Trening Modelu B (LightGBM)
    print("Training monthly LightGBM …")
    monthly_lgb_models = train_monthly_lgb(monthly)

    # 5. Model C: Chronos
    print("Running Chronos forecasting (Model C) …")
    if use_chronos:
        try:
            chronos_preds = predict_chronos(monthly)
            chronos_apr_holdout = predict_chronos_april_holdout(monthly)
            if chronos_apr_holdout is not None and len(chronos_apr_holdout) > 0:
                chronos_cv_mae = float(
                    np.mean(
                        np.abs(
                            chronos_apr_holdout["chronos_pred"].to_numpy()
                            - chronos_apr_holdout["x2_monthly"].to_numpy()
                        )
                    )
                )
            else:
                chronos_cv_mae = None
            print(f"  Chronos predictions: {len(chronos_preds)} rows")
        except Exception as e:
            print(f"  ⚠ Chronos failed ({e}), skipping Model C")
            chronos_preds = None
            chronos_cv_mae = None
            chronos_apr_holdout = None
    else:
        print("  Chronos disabled in config")
        chronos_preds = None
        chronos_cv_mae = None
        chronos_apr_holdout = None

    # 6. Ensemble i Submission
    print("Building final submission ensemble …")
    full_grid = build_full_grid(devices)
    submission = build_submission(
        full_grid,
        monthly,
        monthly_lgb_models,
        physics_params,
        chronos_preds,
        weights_abc=(w_phys_abc, w_lgb_abc, w_chr_abc),
        weights_ab=(w_phys_ab, w_lgb_ab),
    )

    # 7. MAE na danych treningowych
    print("Evaluating models on training data …")
    evaluate_on_train(
        monthly,
        monthly_lgb_models,
        physics_params,
        chronos_cv_mae,
        chronos_apr_holdout,
        weights_abc=(w_phys_abc, w_lgb_abc, w_chr_abc),
        weights_ab=(w_phys_ab, w_lgb_ab),
    )

    print(f"Submission rows: {len(submission)} (expected 3600)")
    submission.write_csv(SUBMISSION_FILE)
    print(f"✅ Saved → {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
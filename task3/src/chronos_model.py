"""
Model C: Amazon Chronos-T5 (zero-shot + fine-tuned forecasting).

Strategy:
- Per device, build a monthly time-series from training x2 values (Oct 2024–Apr 2025 = 7 points).
- Use ChronosPipeline to predict h=6 future months (May–Oct 2025).
- Median across prediction samples used as point forecast.
- For devices with missing data, fall back to the global median Chronos prediction.

Notes:
- Uses 'amazon/chronos-t5-small' by default (~250MB, fast CPU/GPU inference).
- All 600 devices processed in batches to avoid OOM.
- Context: 7 months. Horizon: 6 months (May–Oct).
"""
from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import torch

# Suppress transformers verbose output
warnings.filterwarnings("ignore", category=UserWarning)

# Expected training months in calendar order
TRAIN_MONTHS = [10, 11, 12, 1, 2, 3, 4]  # Oct 2024 – Apr 2025
PRED_MONTHS = [5, 6, 7, 8, 9, 10]        # May–Oct 2025
PRED_YEAR = 2025
HORIZON = len(PRED_MONTHS)                # 6
CONTEXT_LEN = len(TRAIN_MONTHS)           # 7

_CHRONOS_MODEL = "amazon/chronos-t5-small"
_TRAIN_SEQUENCE = [(2024, 10), (2024, 11), (2024, 12), (2025, 1), (2025, 2), (2025, 3), (2025, 4)]


def _load_pipeline(preferred_device: str | None = None):
    """Load ChronosPipeline once; reuse across calls."""
    try:
        from chronos import ChronosPipeline  # type: ignore
    except ImportError as e:
        raise ImportError(
            "chronos-forecasting is required. Install with: uv add chronos-forecasting"
        ) from e

    device = preferred_device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Loading {_CHRONOS_MODEL} on {device} …")
    pipeline = ChronosPipeline.from_pretrained(
        _CHRONOS_MODEL,
        device_map=device,
        dtype=dtype,
    )
    return pipeline


def _is_cuda_oom(err: Exception) -> bool:
    txt = str(err).lower()
    return "cuda out of memory" in txt or "cublas" in txt and "alloc" in txt


def _build_context_matrix(monthly: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build a (N_devices, CONTEXT_LEN) matrix of monthly x2 values.
    Months are aligned to TRAIN_MONTHS order.
    Returns (matrix, list_of_device_ids).
    Missing months filled with np.nan.
    """
    train = monthly.filter(pl.col("period") == "train")

    # Identify the year for each training month
    # Oct/Nov/Dec → year 2024, Jan–Apr → year 2025
    def month_year(m: int) -> int:
        return 2024 if m >= 10 else 2025

    device_ids = train["deviceId"].unique(maintain_order=True).to_list()
    n = len(device_ids)
    matrix = np.full((n, CONTEXT_LEN), np.nan, dtype=np.float64)

    for di, device_id in enumerate(device_ids):
        dev_rows = train.filter(pl.col("deviceId") == device_id)
        for mi, m in enumerate(TRAIN_MONTHS):
            yr = month_year(m)
            row = dev_rows.filter((pl.col("month") == m) & (pl.col("year") == yr))
            if len(row) > 0:
                val = row["x2_monthly"][0]
                if val is not None and np.isfinite(float(val)):
                    matrix[di, mi] = float(val)

    return matrix, device_ids


def _impute_context(context: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in context with:
    1. Linear interpolation where neighbours exist.
    2. Forward/backward fill at edges.
    3. Column (month) median for remaining NaN.
    """
    out = context.copy()
    col_medians = np.nanmedian(out, axis=0)

    for i in range(len(out)):
        row = out[i]
        nan_mask = np.isnan(row)
        if not nan_mask.any():
            continue
        # Fill edges and interpolate
        x = row.copy()
        nans = np.isnan(x)
        inds = np.arange(len(x))
        valid = ~nans
        if valid.sum() == 0:
            x[:] = col_medians
        elif valid.sum() == 1:
            x[:] = x[valid][0]
        else:
            x[nans] = np.interp(inds[nans], inds[valid], x[valid])
        out[i] = x

    return out


def predict_chronos(monthly: pl.DataFrame, batch_size: int = 64) -> pl.DataFrame:
    """
    Run Chronos inference on all training devices and return predictions
    for May–Oct 2025 as a Polars DataFrame with columns:
      deviceId, year, month, chronos_pred

    Parameters
    ----------
    monthly : pl.DataFrame
        Output of build_monthly_features() — must contain 'period', 'x2_monthly', 'month', 'year'.
    batch_size : int
        How many device time series to process per forward pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _load_pipeline(device)

    print("  Building Chronos context matrix …")
    context_matrix, device_ids = _build_context_matrix(monthly)
    context_matrix = _impute_context(context_matrix)

    n = len(device_ids)
    print(f"  Running Chronos inference: {n} devices, horizon={HORIZON}, context={CONTEXT_LEN} …")

    all_preds = np.zeros((n, HORIZON), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = context_matrix[start:end]  # (B, T)

        # Chronos expects list of 1-D tensors (variable-length OK)
        tensors = [torch.tensor(row, dtype=torch.float32) for row in batch]

        try:
            with torch.no_grad():
                forecast = pipeline.predict(
                    inputs=tensors,
                    prediction_length=HORIZON,
                    num_samples=40,
                    limit_prediction_length=False,
                )
        except RuntimeError as e:
            if device == "cuda" and _is_cuda_oom(e):
                print("  CUDA OOM in Chronos inference — retrying on CPU …")
                del pipeline
                torch.cuda.empty_cache()
                device = "cpu"
                pipeline = _load_pipeline("cpu")
                with torch.no_grad():
                    forecast = pipeline.predict(
                        inputs=tensors,
                        prediction_length=HORIZON,
                        num_samples=40,
                        limit_prediction_length=False,
                    )
            else:
                raise

        # forecast shape: (B, num_samples, HORIZON)
        preds = forecast.numpy()                      # (B, num_samples, HORIZON)
        median_preds = np.median(preds, axis=1)      # (B, HORIZON)
        all_preds[start:end] = np.maximum(median_preds, 0.0)

        if start == 0 or (start // batch_size) % 5 == 0:
            pct = 100 * end / n
            print(f"    … {end}/{n} devices ({pct:.0f}%)")

    rows = []
    for di, device_id in enumerate(device_ids):
        for hi, month in enumerate(PRED_MONTHS):
            rows.append(
                {
                    "deviceId": device_id,
                    "year": PRED_YEAR,
                    "month": month,
                    "chronos_pred": float(all_preds[di, hi]),
                }
            )

    return pl.DataFrame(rows)


def predict_chronos_april_holdout(monthly: pl.DataFrame) -> pl.DataFrame | None:
    return predict_chronos_holdout(monthly, holdout_month=4)


def predict_chronos_holdout(monthly: pl.DataFrame, holdout_month: int) -> pl.DataFrame | None:
    """
    Hold-out prediction: train Chronos on all prior train months, predict selected month.
    Returns per-device rows with: deviceId, year, month, chronos_pred, x2_monthly.
    """
    holdout_candidates = [m for _, m in _TRAIN_SEQUENCE]
    if holdout_month not in holdout_candidates:
        raise ValueError(f"Unsupported holdout_month={holdout_month}. Expected one of {holdout_candidates}.")

    holdout_idx = holdout_candidates.index(holdout_month)
    if holdout_idx <= 0:
        raise ValueError("Holdout month must have at least one earlier month in context.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _load_pipeline(device)

    observed = monthly.filter(pl.col("x2_monthly").is_not_null())

    context_pairs = _TRAIN_SEQUENCE[:holdout_idx]
    pred_year_h, pred_month = _TRAIN_SEQUENCE[holdout_idx]

    device_ids = observed["deviceId"].unique(maintain_order=True).to_list()
    n = len(device_ids)
    context_matrix = np.full((n, len(context_pairs)), np.nan)
    y_true = np.full(n, np.nan)

    for di, device_id in enumerate(device_ids):
        dev = observed.filter(pl.col("deviceId") == device_id)
        for mi, (yr, m) in enumerate(context_pairs):
            row = dev.filter((pl.col("month") == m) & (pl.col("year") == yr))
            if len(row) > 0:
                v = row["x2_monthly"][0]
                if v is not None and np.isfinite(float(v)):
                    context_matrix[di, mi] = float(v)

        holdout_row = dev.filter((pl.col("month") == pred_month) & (pl.col("year") == pred_year_h))
        if len(holdout_row) > 0:
            v = holdout_row["x2_monthly"][0]
            if v is not None and np.isfinite(float(v)):
                y_true[di] = float(v)

    context_matrix = _impute_context(context_matrix)
    valid_mask = np.isfinite(y_true)

    if valid_mask.sum() == 0:
        print("  Chronos hold-out: no April actuals found.")
        return None

    tensors = [torch.tensor(row, dtype=torch.float32) for row in context_matrix]
    try:
        with torch.no_grad():
            forecast = pipeline.predict(
                inputs=tensors,
                prediction_length=1,
                num_samples=40,
                limit_prediction_length=False,
            )
    except RuntimeError as e:
        if device == "cuda" and _is_cuda_oom(e):
            print("  CUDA OOM in Chronos hold-out — retrying on CPU …")
            del pipeline
            torch.cuda.empty_cache()
            pipeline = _load_pipeline("cpu")
            with torch.no_grad():
                forecast = pipeline.predict(
                    inputs=tensors,
                    prediction_length=1,
                    num_samples=40,
                    limit_prediction_length=False,
                )
        else:
            raise

    preds = np.maximum(np.median(forecast.numpy()[:, :, 0], axis=1), 0.0)

    rows = []
    for i, device_id in enumerate(device_ids):
        if not np.isfinite(y_true[i]):
            continue
        rows.append(
            {
                "deviceId": device_id,
                "year": pred_year_h,
                "month": pred_month,
                "chronos_pred": float(preds[i]),
                "x2_monthly": float(y_true[i]),
            }
        )

    if not rows:
        return None

    return pl.DataFrame(rows)


def evaluate_chronos_on_train(monthly: pl.DataFrame) -> float | None:
    """
    Hold-out evaluation: train Chronos on Oct–Mar, predict Apr, compare to actual.
    Returns MAE on April.
    """
    holdout = predict_chronos_april_holdout(monthly)
    if holdout is None or len(holdout) == 0:
        return None

    mae = float(
        np.mean(
            np.abs(
                holdout["chronos_pred"].to_numpy()
                - holdout["x2_monthly"].to_numpy()
            )
        )
    )
    print(f"  Chronos hold-out MAE (April, n={len(holdout)}): {mae:.4f}")
    return mae

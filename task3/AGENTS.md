# Technical Specification: Heat Pump Grid Load Prediction

## Objective
Predict monthly average `x2` (grid load indicator) for 600 heat pumps, months May–October 2025.
Trained on Oct 2024 – Apr 2025. Evaluated on MAE (lower = better, 0 = perfect).

---

## Core Problem: OOD Seasonal Extrapolation

**This is not forecasting — it is regime-change extrapolation.** Winter: heat pumps run heating mode (warm = lower load). Summer: most units enter standby/DHW-only mode where load drops to ~10–20% of winter peak.

**CRITICAL: Tree models (LightGBM/XGBoost) cannot extrapolate** beyond the range of target values seen in training. If the lowest monthly average in training is April at ~X, a tree model cannot predict the lower summer baseload. This makes `linear_tree=True` and a physics baseline **existentially important**, not optional.

---

## Data

### Files
- `data/data.csv` — 10.4 GB, 64.5M rows, 5-minute telemetry per device
  - `deviceId`, `timedate`, `period` (train/validation/test)
  - `t1`–`t13`: temperatures — **all min-max normalized to [0,1]** — cannot use for absolute physics calculations
  - `x1`: compressor frequency, `x2`: grid load (target), `x3`: heating curve type, `deviceType`
  - `x2` is withheld for `period = validation` and `period = test`
- `data/devices.csv` — 600 rows, deviceId + latitude + longitude

### Data Splits
| Period      | Months           | Notes            |
|-------------|------------------|------------------|
| train       | Oct 2024 – Apr 2025 | Full feature set |
| validation  | May–Jun 2025     | x2 withheld      |
| test        | Jul–Oct 2025     | x2 withheld      |

---

## Feature Engineering

### Key rule: use Open-Meteo absolute temperatures for physics features
Since t1–t13 are normalized, all thermodynamic features (HDD, COP, degree days) must use absolute temperatures from Open-Meteo API, not device sensor data.

### Weather (Open-Meteo ERA5 reanalysis)
Fetch hourly from `archive-api.open-meteo.com/v1/archive` per unique lat/lon pair.
Variables: `temperature_2m`, `shortwave_radiation`, `relative_humidity_2m`, `wind_speed_10m`
Period: 2024-10-01 to 2025-10-31. Cache to `data/weather_cache.parquet`.

### Physics features (from absolute temp, daily aggregated)
- `HDD_15`, `HDD_18` = `max(T_base - T_daily_mean, 0)` — heating degree days at base 15°C and 18°C
- `CDD_22`, `CDD_24` = `max(T_daily_mean - T_base, 0)` — cooling degree days
- `inverse_COP_proxy` = `(T_indoor_estimate - T_outdoor) / T_indoor_estimate` — approaches 0 as heating demand vanishes
- `electrical_load_proxy` = `HDD_15 × inverse_COP_proxy` — **the primary extrapolation feature**: naturally goes to 0 in summer because HDD=0
- `solar_radiation_sum` — daily total solar gain (reduces heating demand)
- `day_length` = `(2/15) × arccos(-tan(lat_rad) × tan(solar_declination))` — deterministic seasonal signal

### Device/sensor features (daily aggregated from normalized data)
- `t1_mean`, `t1_max`, `t1_min` — normalized external temp stats (useful as relative signal even if not absolute)
- `t2_mean` — internal temp
- `t7_mean` — DHW tank temperature (proxy for DHW demand)
- `x1_mean` — compressor frequency (proxy for operating mode)
- `x3` — heating curve type: encodes building insulation quality; low slope = well insulated; **valuable summer feature because it predicts DHW baseload**
- `deviceType` — device category (air-source, ground-source, DHW-only)
- `month`, `day_of_year`, `is_heating_season` (1 if T_mean < 15°C for 3+ consecutive days)

### Upweight pseudo-summer samples
Identify training days where daily mean outdoor temp > 15°C (March/April warm days). These are the closest proxy to summer conditions and should receive **higher sample weight** during model training.

---

## Model Architecture

### Model A — Physics Baseline (per-device regression)
Fit per device from training data: `daily_load ≈ HDD × building_HTC / COP_estimate + DHW_baseload`
- Estimate `building_HTC` and `DHW_baseload` per device via OLS on training data
- Provides extrapolation backbone: correctly predicts near-zero heating in summer
- Use as a base predictor + as a post-processing floor

### Model B — Global LightGBM with linear trees
```python
params = {
    "objective": "regression_l1",   # MAE-optimal
    "linear_tree": True,             # enables extrapolation beyond training range
    "monotone_constraints": [...],   # load must decrease as outdoor temp rises (heating mode)
    "metric": "mae",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}
```
Train globally on all 600 devices simultaneously — cross-device learning gives "summer-like" data from warmer regions.

### Model C — LightGBM on log1p(x2) target
Diversity via target transformation. Right-skewed energy data benefits from log-normalization.
Ensemble result: `expm1(predicted_log)`.

### Model D — Ridge regression on physics features
Linear model that naturally extrapolates. Use `electrical_load_proxy`, `HDD_15`, `solar_radiation_sum`, `day_length`, `deviceType`, `x3`.

### Temporal Cross-Validation (forward-chaining only, no K-fold)
| Fold | Train      | Validate | Notes                                      |
|------|------------|----------|--------------------------------------------|
| 1    | Oct–Jan    | Feb      |                                            |
| 2    | Oct–Feb    | Mar      |                                            |
| 3    | Oct–Mar    | Apr      | Most important: simulates heating wind-down |

**DO NOT use standard K-fold.** The leaderboard shows May–Jun only, but the final score weights Jul–Oct at 4/6. Overfitting to May destroys the final score.

---

## Ensemble & Post-processing

1. **Median ensemble** across all models and folds — median is MAE-optimal for skewed distributions
2. Aggregate daily predictions to monthly averages per device (mean of daily values within month)
3. **DHW baseload floor**: summer predictions must be ≥ estimated DHW baseload per device (~60–85% of x2 equivalent). Use physics model's `DHW_baseload` estimate.
4. **Heating-only constraint**: Jun–Aug ≤ Apr predictions for devices identified as heating-only
5. **October calibration anchor**: Oct 2025 predictions should be consistent with Oct 2024 actuals from training data
6. Clip to `[0, device_99th_percentile_x2]`

---

## Implementation Notes

- Load `data.csv` with **Polars `scan_csv()` lazy mode** — never load 10 GB into memory at once
- Remove stuck meters: constant `x2` for > 24 consecutive readings (2 hours)
- Remove negative `x2`; winsorize at 99.5th percentile per device
- Use `data/out/` for intermediate outputs and final submission
- Submission format: `deviceId,year,month,prediction` — one row per device per month (600 × 6 = 3600 rows)

---

## DO NOT
- Use standard K-fold CV
- Use MSE loss — always use L1 (`regression_l1`) or quantile α=0.5
- Rely solely on tree models without physics baseline or `linear_tree=True`
- Use normalized t1 values for computing absolute HDD or COP (use Open-Meteo temps instead)
- Ignore `x3` and `deviceType` — critical for summer DHW baseload prediction

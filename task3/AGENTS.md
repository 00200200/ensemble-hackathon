# Agentic Workflow & Architecture Specification: Euros Energy Challenge

## 1. Project Overview & Objective
[cite_start]**Goal:** Develop a highly precise predictive system to forecast the monthly average electrical grid load indicator ($x2$) for heat pumps during the summer-autumn period (May-October 2025)[cite: 8].
[cite_start]**Challenge Type:** Out-of-Distribution (OOD) Seasonal Extrapolation (Winter training $\rightarrow$ Summer testing)[cite: 11].
[cite_start]**Metric:** Mean Absolute Error (MAE)[cite: 39]. [cite_start]A score of 0 corresponds to a perfect prediction[cite: 42].
**Core Strategy:** Physics-informed feature engineering combined with gradient boosting (LightGBM with linear trees) and external data enrichment (Open-Meteo), shifting the paradigm from temporal auto-regression to causal thermodynamic mapping.

---

## 2. Global Data Structures (Schemas)

Agents must strictly adhere to these data schemas during data transformation and feature engineering pipelines.

### 2.1. Raw Input Data
* **`devices.csv`** (Static Metadata)
    * [cite_start]`deviceId` (String): Unique identifier (join key)[cite: 26].
    * [cite_start]`latitude` (Float): Geographic coordinate[cite: 26].
    * [cite_start]`longitude` (Float): Geographic coordinate[cite: 26].
* [cite_start]**`data.csv`** (Time-Series Telemetry - 5-minute intervals) [cite: 15, 24]
    * [cite_start]`deviceId` (String)[cite: 21].
    * [cite_start]`Timedate` (Datetime UTC)[cite: 21].
    * [cite_start]`t1` (Float): External temperature (min-max normalized)[cite: 21].
    * [cite_start]`t2` (Float): Internal temperature[cite: 21].
    * [cite_start]`t3` to `t13` (Float): Various internal sensors (HEX, DHW tank, buffers)[cite: 21].
    * [cite_start]`x1` (Float): Compressor operating frequency[cite: 21].
    * [cite_start]`x2` (Float): Grid load indicator (Prediction Target)[cite: 21].
    * [cite_start]`x3` (Categorical): Heating curve type[cite: 21].
    * [cite_start]`deviceType` (Integer): Encoded device category[cite: 21].

### 2.2. External Enrichment Data (Open-Meteo API)
Fetched dynamically based on `latitude` and `longitude`.
* **`weather_era5_hourly`**
    * `temperature_2m` (Float): Absolute external temperature (°C).
    * `shortwave_radiation` (Float): Solar irradiance (W/m²).
    * `relative_humidity_2m` (Float): Humidity (%).
    * `wind_speed_10m` (Float): Wind speed (m/s).

### 2.3. Daily Aggregated Master Table
To minimize noise and optimize for MAE, 5-minute data is compressed to daily granularity.
* **`daily_features`**
    * `date` (Date): Aggregation key.
    * `deviceId` (String).
    * `target_x2_daily_median` (Float): Daily median of $x2$.
    * `t1_mean`, `t1_max`, `t1_min` (Float): Daily stats for external temp.
    * `solar_radiation_sum` (Float): Total daily solar energy.
    * `HDD_15`, `HDD_18` (Float): Heating Degree Days (base 15°C, 18°C).
    * `inverse_COP_proxy` (Float): Thermodynamic efficiency proxy.
    * `electrical_load_proxy` (Float): $HDD \times inverse\_COP\_proxy$.
    * `is_heating_season` (Boolean): 1 if `t1_mean` < 15°C for 3 days, else 0.

### 2.4. Submission Schema
* **`submission.csv`**
    * [cite_start]`deviceId` (String)[cite: 37].
    * [cite_start]`year` (Integer): Forecast year[cite: 37].
    * `month` (Integer): Forecast month (5-10)[cite: 37].
    * [cite_start]`prediction` (Float): Predicted average $x2$[cite: 37].
    * [cite_start]*Constraint:* $target_{d,m} = \frac{1}{N_{d,m}} \sum x_{2}^{(d,m,l)}$[cite: 29].

---

## 3. Agent Architecture & Execution Pipeline

### Agent 1: Data Integrator & Cleaner
**Role:** Ingest raw data, handle missing values, and fetch external weather context.
**Tasks:**
1.  Load `data.csv` and `devices.csv` using `Polars` for multithreaded performance.
2.  Identify "stuck meters" (constant $x2$ values for > 2 hours) and remove them.
3.  Impute short gaps (< 2 hours) in $t1-t13$ using linear interpolation.
4.  **API Call:** Query `archive-api.open-meteo.com/v1/archive` for each unique `lat/lon` pair to fetch hourly `temperature_2m` and `shortwave_radiation` for Oct 2024 - Oct 2025.
5.  Join external weather data with the primary telemetry using timezone-aware rounding.

### Agent 2: Physics-Informed Feature Engineer
**Role:** Transform raw telemetry into thermodynamic drivers that enable out-of-distribution extrapolation.
**Tasks:**
1.  **Astronomical Features:** Calculate `day_length` using PySolar or standard trigonometric formulas based on `latitude` and day of the year.
2.  **Thermodynamic Gradients:**
    * `delta_T_heating` = $max(t2_{mean} - t1_{mean}, 0)$
    * `COP_proxy` = $t5 / (t5 - t3)$ (using appropriate offset to avoid division by zero).
    * `inverse_COP` = $(t2 - t1) / t2$
3.  **Degree Days:** Calculate `HDD` (Heating Degree Days) for multiple base temperatures (15°C, 18°C).
4.  **The "Extrapolation Secret Weapon":** Create the combined feature `load_proxy_feature` = $HDD_{15} \times inverse\_COP$. *Rationale: When summer arrives, HDD becomes 0, mathematically forcing the feature contribution to zero, perfectly simulating the shutdown of space heating.*
5.  **Aggregation:** Downsample all 5-minute features to daily granularity (medians for temperatures, sums for radiation/HDD).

### Agent 3: ML Modeler (The Extrapolator)
**Role:** Train robust gradient boosting models optimized for MAE and capable of linear extrapolation.
**Tasks:**
1.  **Validation Setup:** Strictly use **Temporal Forward-Chaining**.
    * *Fold 1:* Train (Oct-Jan) $\rightarrow$ Val (Feb)
    * *Fold 2:* Train (Oct-Feb) $\rightarrow$ Val (Mar)
    * *Fold 3 (Crucial):* Train (Oct-Mar) $\rightarrow$ Val (Apr) - *April simulates the winter-to-summer transition.*
2.  **Model Configuration (LightGBM):**
    * `objective`: `'regression_l1'` (optimizes for MAE median, not mean).
    * `linear_tree`: `True` (allows trees to extrapolate trends beyond training data boundaries).
    * `monotone_constraints`: Impose negative constraint on $t1$ (as outdoor temp goes up, heating load must go down).
3.  **Target Transformation:** Train one model on raw $x2$ and a secondary model on $\log_{1p}(x2)$ to handle right-skewed energy consumption distributions.

### Agent 4: Meta-Ensembler & Submitter
**Role:** Aggregate daily predictions to the required monthly targets and format the submission.
**Tasks:**
1.  Run inference using the trained models (Agent 3) on the daily weather forecasts/data for May-Oct 2025.
2.  Calculate the median prediction across all trained folds/models (Median Ensembling).
3.  **Temporal Reconciliation:** Aggregate the daily predicted values into monthly averages per `deviceId` to construct $target_{d,m}$[cite: 29].
4.  **Post-Processing Guardrails:**
    * Ensure August load $\le$ April load for devices identified as heating-only.
    * Apply minimum floor threshold based on the calculated DHW (Domestic Hot Water) baseload.
5.  Generate `submission.csv` strictly adhering to the required format[cite: 31, 32].

---

## 4. Critical Constraints & "Bait" Warnings
* [cite_start]**Do NOT use standard K-Fold CV.** The public leaderboard is only May-Jun (validation)[cite: 44]. [cite_start]Overfitting to May will destroy the Final Score (which includes the peak summer months up to October)[cite: 44].
* **Do NOT use MSE.** Evaluation is strictly MAE[cite: 39]. Use `L1Loss` or Quantile Regression ($\alpha=0.5$).
* **Do NOT ignore `deviceType` and `x3` (heating curve).** Even in summer, `x3` is a powerful proxy for building insulation quality and dictates the DHW baseload.
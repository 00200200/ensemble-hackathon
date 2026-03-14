from __future__ import annotations
import math
import polars as pl


def day_length(lat_deg: float, doy: int) -> float:
    """Astronomical day length in hours for a given latitude and day-of-year."""
    lat_rad = math.radians(lat_deg)
    decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))
    cos_ha = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(decl)))
    return (2 / 15) * math.degrees(math.acos(cos_ha))


def add_physics_features(df: pl.DataFrame, has_weather: bool) -> pl.DataFrame:
    """Add thermodynamic features. Uses t1-derived approximate temp for ALL periods."""
    T_base15 = 15.0
    T_base18 = 18.0
    T_cool22 = 22.0
    T_indoor = 20.0

    if has_weather:
        temp_col = pl.col("temp_2m")
    else:
        temp_col = pl.col("t1") * 50.0 - 15.0
        df = df.with_columns(
            [
                (pl.col("t1") * 50.0 - 15.0).alias("temp_2m"),
                pl.lit(None).cast(pl.Float64).alias("solar_rad"),
                pl.lit(None).cast(pl.Float64).alias("humidity"),
                pl.lit(None).cast(pl.Float64).alias("wind_speed"),
            ]
        )

    df = df.with_columns(
        [
            (pl.lit(T_base15) - temp_col).clip(lower_bound=0).alias("hdd15_5min"),
            (pl.lit(T_base18) - temp_col).clip(lower_bound=0).alias("hdd18_5min"),
            (temp_col - pl.lit(T_cool22)).clip(lower_bound=0).alias("cdd22_5min"),
            ((pl.lit(T_indoor) - temp_col) / pl.lit(T_indoor)).clip(lower_bound=0).alias("inv_cop"),
            pl.col("timedate").dt.date().alias("date"),
            pl.col("timedate").dt.year().alias("year"),
            pl.col("timedate").dt.month().alias("month"),
            pl.col("timedate").dt.ordinal_day().alias("doy"),
        ]
    )
    return df


def daily_aggregate(df: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Compress 5-minute rows to one row per (deviceId, date) for ALL periods."""
    df = df.with_columns((pl.col("hdd15_5min") * pl.col("inv_cop")).alias("load_proxy_5min"))

    if "latitude" not in df.columns:
        df = df.join(devices.select(["deviceId", "latitude"]), on="deviceId", how="left")

    daily = df.group_by(["deviceId", "date"]).agg(
        [
            pl.col("x2").mean().alias("x2_mean"),
            pl.col("period").first().alias("period"),
            pl.col("year").first().alias("year"),
            pl.col("month").first().alias("month"),
            pl.col("doy").first().alias("doy"),
            pl.col("latitude").first().alias("latitude"),
            pl.col("t1").mean().alias("t1_mean"),
            pl.col("t1").max().alias("t1_max"),
            pl.col("t1").min().alias("t1_min"),
            pl.col("t2").mean().alias("t2_mean"),
            pl.col("t7").mean().alias("t7_mean"),
            pl.col("x1").mean().alias("x1_mean"),
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("temp_2m").mean().alias("temp_mean"),
            pl.col("temp_2m").max().alias("temp_max"),
            pl.col("temp_2m").min().alias("temp_min"),
            pl.col("solar_rad").sum().alias("solar_sum"),
            pl.col("humidity").mean().alias("humidity_mean"),
            pl.col("wind_speed").mean().alias("wind_mean"),
            (pl.col("hdd15_5min").sum() / 288).alias("HDD_15"),
            (pl.col("hdd18_5min").sum() / 288).alias("HDD_18"),
            (pl.col("cdd22_5min").sum() / 288).alias("CDD_22"),
            pl.col("inv_cop").mean().alias("inv_cop"),
            (pl.col("load_proxy_5min").sum() / 288).alias("load_proxy"),
        ]
    )

    daily = daily.with_columns(
        pl.struct(["latitude", "doy"])
        .map_elements(
            lambda s: day_length(s["latitude"], s["doy"]),
            return_dtype=pl.Float64,
        )
        .alias("day_length")
    )

    daily = daily.sort(["deviceId", "date"])
    daily = daily.with_columns((pl.col("temp_mean") < 15.0).cast(pl.Int32).alias("cold_day"))
    daily = daily.with_columns(
        (
            pl.col("cold_day")
            + pl.col("cold_day").shift(1).over("deviceId").fill_null(0)
            + pl.col("cold_day").shift(2).over("deviceId").fill_null(0)
        ).alias("cold_streak")
    )
    daily = daily.with_columns(
        (pl.col("cold_streak") >= 3).cast(pl.Int32).alias("is_heating_season")
    ).drop(["cold_day", "cold_streak"])

    # Upweight warm training days 3×
    daily = daily.with_columns(
        pl.when((pl.col("period") == "train") & (pl.col("temp_mean") > 15.0))
        .then(3.0)
        .otherwise(1.0)
        .alias("sample_weight")
    )

    return daily


def build_monthly_features(daily: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily → monthly per device."""
    monthly = daily.group_by(["deviceId", "year", "month"]).agg(
        [
            pl.col("x2_mean").mean().alias("x2_monthly"),
            pl.col("period").first().alias("period"),
            pl.col("temp_mean").mean().alias("temp_mean_monthly"),
            pl.col("temp_mean").min().alias("temp_min_monthly"),
            pl.col("HDD_15").sum().alias("HDD_15_monthly"),
            pl.col("HDD_18").sum().alias("HDD_18_monthly"),
            pl.col("CDD_22").sum().alias("CDD_22_monthly"),
            pl.col("load_proxy").sum().alias("load_proxy_monthly"),
            pl.col("solar_sum").sum().alias("solar_monthly"),
            pl.col("day_length").mean().alias("day_length_monthly"),
            pl.col("inv_cop").mean().alias("inv_cop_monthly"),
            pl.col("t1_mean").mean().alias("t1_mean_monthly"),
            pl.col("t1_min").min().alias("t1_min_monthly"),
            pl.col("x1_mean").mean().alias("x1_mean_monthly"),
            pl.col("x3").first().alias("x3"),
            pl.col("deviceType").first().alias("deviceType"),
            pl.col("is_heating_season").mean().alias("frac_heating_season"),
            pl.col("sample_weight").mean().alias("sample_weight"),
        ]
    )

    # Uporządkowanie czasowe jest wymagane dla lagów miesięcznych
    monthly = monthly.sort(["deviceId", "year", "month"])

    # Master features z analizy XAI
    monthly = monthly.with_columns(
        [
            (pl.col("temp_mean_monthly") * pl.col("x3")).alias("temp_weighted_x3"),
            (pl.col("temp_mean_monthly") * pl.col("solar_monthly")).alias("temp_solar_interaction"),
            pl.col("temp_mean_monthly").shift(1).over("deviceId").alias("lag_temp_1_month"),
        ]
    ).with_columns(
        [
            pl.col("lag_temp_1_month").fill_null(pl.col("temp_mean_monthly")).alias("lag_temp_1_month"),
        ]
    )

    return monthly
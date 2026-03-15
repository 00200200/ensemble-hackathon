from __future__ import annotations
import polars as pl
import sys
from config import DATA_DIR, WEATHER_CACHE

SENSOR_COLS = [f"t{i}" for i in range(1, 14)]
NEEDED_COLS = ["deviceId", "timedate", "period", "x2", "x3", "deviceType"] + SENSOR_COLS


def load_raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    devices_path = DATA_DIR / "devices.csv"
    data_path    = DATA_DIR / "data.csv"

    for p in [devices_path, data_path]:
        if not p.exists():
            print(f"❌ Nie znaleziono: {p}")
            sys.exit(1)

    print(f"Loading {devices_path} …")
    devices = pl.read_csv(devices_path)

    print(f"Scanning {data_path} (lazy) …")
    df = (
        pl.scan_csv(data_path)
        .select(NEEDED_COLS)
        .with_columns(
            pl.col("timedate")
            .str.replace(r" UTC$", "")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
            .alias("timedate")
        )
        .collect()
    )
    print(f"Loaded {len(df):,} rows | periods: {df['period'].unique().to_list()}")

    # Usuń ujemne x2
    df = df.filter(pl.col("x2").is_null() | (pl.col("x2") >= 0))

    # Winsorize x2 @ 99.5 percentile per device (tylko train)
    p995 = (
        df.filter(pl.col("period") == "train")
        .group_by("deviceId")
        .agg(pl.col("x2").quantile(0.995).alias("x2_cap"))
    )
    df = (
        df.join(p995, on="deviceId", how="left")
        .with_columns(
            pl.when(pl.col("x2").is_not_null() & pl.col("x2_cap").is_not_null())
            .then(pl.col("x2").clip(upper_bound=pl.col("x2_cap")))
            .otherwise(pl.col("x2"))
            .alias("x2")
        )
        .drop("x2_cap")
    )

    # Cechy różnicowe (delta t4, t8) — dostępne we wszystkich periodach
    df = df.sort(["deviceId", "timedate"]).with_columns([
        (pl.col("t4") - pl.col("t4").shift(1).over("deviceId")).alias("dt4"),
        (pl.col("t8") - pl.col("t8").shift(1).over("deviceId")).alias("dt8"),
    ])

    # Lagi x2 tylko w trainie (w valid/test x2=null, więc lag też null — OK)
    df = df.with_columns([
        pl.col("x2").shift(1).over("deviceId").alias("x2_lag1"),
        pl.col("x2").shift(2).over("deviceId").alias("x2_lag2"),
    ])

    print(f"After cleaning: {len(df):,} rows")
    return df, devices


def load_weather() -> pl.DataFrame | None:
    if not WEATHER_CACHE.exists():
        print("⚠ Brak weather_cache.parquet — pomijam pogodę")
        return None
    print("Loading cached weather …")
    return pl.read_parquet(WEATHER_CACHE)


def join_weather(df: pl.DataFrame, devices: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    devices_r = devices.with_columns([
        pl.col("latitude").round(1),
        pl.col("longitude").round(1),
    ])
    df = df.join(devices_r.select(["deviceId", "latitude", "longitude"]), on="deviceId", how="left")
    df = df.with_columns(pl.col("timedate").dt.truncate("1h").alias("hour"))
    df = df.join(weather, on=["latitude", "longitude", "hour"], how="left")
    return df.drop("hour")
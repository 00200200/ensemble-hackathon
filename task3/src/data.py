from __future__ import annotations

import math
import time
import requests
import polars as pl
import sys

from config import DATA_DIR, WEATHER_CACHE, WEATHER_PARTIAL, BATCH_SIZE


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and clean data.csv + devices.csv using Polars lazy mode."""
    
    # Zabezpieczenie ścieżek z jasnym komunikatem błędu
    devices_path = DATA_DIR / "devices.csv"
    data_path = DATA_DIR / "data.csv"
    
    if not devices_path.exists():
        print(f"❌ BŁĄD: Nie znaleziono pliku {devices_path}")
        print("💡 Upewnij się, że w pliku 'src/config.py' zmienna DATA_DIR jest ustawiona na Path('data') a nie '../data', jeśli odpalasz skrypt z głównego folderu.")
        sys.exit(1)
        
    if not data_path.exists():
        print(f"❌ BŁĄD: Nie znaleziono pliku {data_path}")
        sys.exit(1)

    print(f"Loading {devices_path} …")
    devices = pl.read_csv(devices_path)

    print(f"Scanning {data_path} (lazy, 10 GB) …")
    needed_cols = [
        "deviceId", "timedate", "period", "t1", "t2", "t7", 
        "x1", "x2", "x3", "deviceType",
    ]

    # Usunięto try_parse_dates=False (deprecated w nowych wersjach Polars)
    lazy = pl.scan_csv(data_path)
    
    # Rzutowanie daty i czasu
    lazy = lazy.select(needed_cols).with_columns(
        pl.col("timedate")
        .str.replace(r" UTC$", "")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timedate")
    )

    print("Collecting … (this may take a few minutes)")
    df = lazy.collect()
    print(f"Loaded {len(df):,} rows")

    # Remove negative x2 (sensor errors)
    df = df.filter(pl.col("x2").is_null() | (pl.col("x2") >= 0))

    # Winsorize x2 at 99.5th percentile per device (train only)
    p995 = (
        df.filter(pl.col("period") == "train")
        .group_by("deviceId")
        .agg(pl.col("x2").quantile(0.995).alias("x2_cap"))
    )
    df = df.join(p995, on="deviceId", how="left")
    df = df.with_columns(
        pl.when(pl.col("x2").is_not_null() & pl.col("x2_cap").is_not_null())
        .then(pl.col("x2").clip(upper_bound=pl.col("x2_cap")))
        .otherwise(pl.col("x2"))
        .alias("x2")
    ).drop("x2_cap")

    # Remove stuck meters: x2 constant for > 24 consecutive readings (2 hrs)
    df = df.sort(["deviceId", "timedate"])
    df = df.with_columns(
        (pl.col("x2") != pl.col("x2").shift(1).over("deviceId"))
        .fill_null(True)
        .cast(pl.Int32)
        .alias("x2_changed")
    )
    df = df.with_columns(pl.col("x2_changed").cum_sum().over("deviceId").alias("streak_id"))
    streak_lengths = df.group_by(["deviceId", "streak_id"]).agg(pl.len().alias("streak_len"))
    df = df.join(streak_lengths, on=["deviceId", "streak_id"], how="left")
    df = df.filter((pl.col("streak_len") <= 24) | (pl.col("period") != "train")).drop(
        ["x2_changed", "streak_id", "streak_len"]
    )

    print(f"After cleaning: {len(df):,} rows")
    return df, devices


def fetch_weather(devices: pl.DataFrame) -> pl.DataFrame | None:
    """Fetch hourly weather for each unique lat/lon using batched requests."""
    if WEATHER_CACHE.exists():
        print("Loading cached weather …")
        return pl.read_parquet(WEATHER_CACHE)

    print("Fetching weather from Open-Meteo API …")
    locations = (
        devices.with_columns([pl.col("latitude").round(1), pl.col("longitude").round(1)])
        .select(["latitude", "longitude"])
        .unique()
        .sort(["latitude", "longitude"])
    )
    all_locs = locations.to_dicts()
    n_total = len(all_locs)
    print(f"  {n_total} unique locations → {math.ceil(n_total / BATCH_SIZE)} batches of {BATCH_SIZE}")

    already_done: set[tuple] = set()
    partial_frames: list[pl.DataFrame] = []
    
    if WEATHER_PARTIAL.exists():
        partial = pl.read_parquet(WEATHER_PARTIAL)
        partial_frames.append(partial)
        for row in partial.select(["latitude", "longitude"]).unique().iter_rows():
            already_done.add(row)
        print(f"  Resuming: {len(already_done)} locations already cached")

    remaining = [loc for loc in all_locs if (loc["latitude"], loc["longitude"]) not in already_done]
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    for batch_i, batch in enumerate(batches):
        lats = [loc["latitude"] for loc in batch]
        lons = [loc["longitude"] for loc in batch]
        print(f"  Batch {batch_i + 1}/{len(batches)}: {len(batch)} locations …")

        params = {
            "latitude": ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
            "start_date": "2024-10-01",
            "end_date": "2025-10-31",
            "hourly": "temperature_2m,shortwave_radiation,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC",
        }

        resp = None
        for attempt in range(10):
            try:
                resp = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=params,
                    timeout=120,
                )
            except requests.RequestException as e:
                print(f"    Request error: {e}, retry {attempt + 1}/10 …")
                time.sleep(min(10 * (2**attempt), 120))
                continue

            if resp.status_code == 200:
                break
            wait = min(10 * (2**attempt), 120)
            print(f"    HTTP {resp.status_code}, waiting {wait}s, retry {attempt + 1}/10 …")
            time.sleep(wait)

        if resp is None or resp.status_code != 200:
            print(f"  WARNING: batch {batch_i + 1} failed after retries, skipping")
            continue

        body = resp.json()
        location_results = body if isinstance(body, list) else [body]

        records: list[dict] = []
        for loc_data, lat, lon in zip(location_results, lats, lons):
            hourly = loc_data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [None] * len(times))
            solar = hourly.get("shortwave_radiation", [None] * len(times))
            humid = hourly.get("relative_humidity_2m", [None] * len(times))
            wind = hourly.get("wind_speed_10m", [None] * len(times))
            for t, te, so, hu, wi in zip(times, temps, solar, humid, wind):
                records.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "hour": t,
                        "temp_2m": te,
                        "solar_rad": so,
                        "humidity": hu,
                        "wind_speed": wi,
                    }
                )

        batch_df = pl.DataFrame(records).with_columns(
            pl.col("hour").str.to_datetime("%Y-%m-%dT%H:%M", strict=False)
        )
        partial_frames.append(batch_df)
        combined = pl.concat(partial_frames)
        
        # Bezpieczne tworzenie folderu data
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(WEATHER_PARTIAL)
        time.sleep(5)

    if not partial_frames:
        print("  WARNING: No weather data fetched — will use t1-based fallback")
        return None

    weather = pl.concat(partial_frames)
    weather.write_parquet(WEATHER_CACHE)
    
    if WEATHER_PARTIAL.exists():
        WEATHER_PARTIAL.unlink()
        
    print(f"  Cached {len(weather):,} weather rows")
    return weather


def join_weather(df: pl.DataFrame, devices: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Join weather to device telemetry by nearest lat/lon and hour."""
    devices_rounded = devices.with_columns(
        [
            pl.col("latitude").round(1),
            pl.col("longitude").round(1),
        ]
    )
    df = df.join(devices_rounded, on="deviceId", how="left")
    df = df.with_columns(pl.col("timedate").dt.truncate("1h").alias("hour"))
    df = df.join(weather, on=["latitude", "longitude", "hour"], how="left")
    return df.drop("hour")
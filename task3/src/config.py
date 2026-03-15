from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "out"
WEATHER_CACHE = DATA_DIR / "weather_cache.parquet"
SUBMISSION_FILE = OUT_DIR / "submission.csv"
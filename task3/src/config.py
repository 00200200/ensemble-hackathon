from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ścieżki
DATA_DIR = Path("data") # Zakładam, że folder data jest poziom wyżej niż src/
OUT_DIR = DATA_DIR / "out"
WEATHER_CACHE = DATA_DIR / "weather_cache.parquet"
WEATHER_PARTIAL = DATA_DIR / "weather_partial.parquet"
SUBMISSION_FILE = OUT_DIR / "submission.csv"

# Parametry API
BATCH_SIZE = 50
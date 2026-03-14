import io
import os

import pandas as pd
import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task1"


API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
PARQUET_FILE = "chebi_submission_example.parquet"

def main():
    if not API_TOKEN:
        raise ValueError(
            "TEAM_TOKEN not provided. Define TEAM_TOKEN in .env"
        )

    if not SERVER_URL:
        raise ValueError(
            "SERVER_URL not defined. Define SERVER_URL in .env"
        )

    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        raise FileExistsError(f"Parquet file did not load properly, error {e}")
    
    # show example parquet data and df structure
    print(df)

    headers = {
        "X-API-Token": API_TOKEN
    }

    # load dataframe into buffer
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    # Important, the name of key in files - "parquet_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"parquet_file": buffer},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)


if __name__ == "__main__":
    main()
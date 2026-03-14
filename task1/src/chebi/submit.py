import io
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "task1"


def save_submission(
    test_df: pd.DataFrame, predictions, class_cols: list[str], path: str
) -> pd.DataFrame:
    sub = test_df[["mol_id", "SMILES"]].copy()
    for i, col in enumerate(class_cols):
        sub[col] = predictions[:, i].astype(int)
    sub.to_parquet(path, index=False)
    print(f"  Saved: {path}")
    return sub


def upload(sub: pd.DataFrame) -> None:
    api_token = os.getenv("TEAM_TOKEN")
    server_url = os.getenv("SERVER_URL")

    if not api_token or not server_url:
        print("No TEAM_TOKEN / SERVER_URL in .env — skipping upload.")
        return

    print(f"Uploading to {server_url}/{ENDPOINT} ...")
    buf = io.BytesIO()
    sub.to_parquet(buf, index=False)
    buf.seek(0)

    resp = requests.post(
        f"{server_url}/{ENDPOINT}",
        files={"parquet_file": buf},
        headers={"X-API-Token": api_token},
    )
    try:
        print(f"  Response: {resp.status_code} {resp.json()}")
    except Exception:
        print(f"  Response: {resp.status_code} {resp.text}")

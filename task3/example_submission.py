import os
import csv
import random

import numpy as np
import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task3"

API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

CSV_FILE="data/out/load_example_submission.csv"


def generate_mock_submission():
    predictions = [("deviceId", "year", "month", "prediction")]

    devices = (
        "000cc3cb7f030c3d0c481bd0e7cf42ee283012c3cb5bfc93ae46eecc5798f0fe",
        "005767201ec5d7c3336b3b4d1ffa8a72e7ca1ecdaac30fe5a99d7a76b53f9fc9"
        )

    months = (5, 6, 7, 8, 9, 10)

    for device in devices:
        for month in months:
            predictions.append((device, 2025, month, random.random()))

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(predictions)

def main():
    generate_mock_submission()

    if not API_TOKEN:
        raise ValueError(
            "TEAM_TOKEN not provided. Define TEAM_TOKEN in .env"
        )

    if not SERVER_URL:
        raise ValueError(
            "SERVER_URL not defined. Define SERVER_URL in .env"
        )

    headers = {
        "X-API-Token": API_TOKEN
    }

    # Important, the name of key in files - "csv_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"csv_file": open(CSV_FILE, "rb")},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)

if __name__ == "__main__":
    main()

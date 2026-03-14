import io
import os

import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task2"


API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
# Change accordingly
JSONL_FILE = "path/to/context_file.jsonl"
STAGE = "practice"

def main():
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

    data = {
        "stage": STAGE,
    }

    # Important, the name of key in files - "jsonl_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"jsonl_file": open(JSONL_FILE, "rb")},
        data=data,
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)


if __name__ == "__main__":
    main()
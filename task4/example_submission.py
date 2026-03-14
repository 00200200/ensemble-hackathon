import os

import numpy as np
import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task4"


API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

NPZ_FILE="data/out/ecg_example_submission.npz"

def generate_mock_submission():
    submission_dict = {}

    # Processing 'ecg_test_0001'
    record_name = "ecg_test_0001"
    standard_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for lead in standard_leads:
        # Generate a random mock signal of 1250 samples (2.5 seconds at 500Hz)
        mock_signal = np.random.randn(1250)

        # Format the key exactly as required: e.g., "ecg_test_0001_V1"
        flat_key = f"{record_name}_{lead}"

        # Cast to float16 to save space and add to dictionary
        submission_dict[flat_key] = mock_signal.astype(np.float16)

    # Save to the final compressed archive
    os.makedirs(os.path.dirname(NPZ_FILE), exist_ok=True)
    np.savez_compressed(NPZ_FILE, **submission_dict, )

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

    # Important, the name of key in files - "npz_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"npz_file": open(NPZ_FILE, "rb")},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)

if __name__ == "__main__":
    main()

import os
import requests
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

ENDPOINT = "task3"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

# Podmień na dokładną nazwę swojego wygenerowanego pliku
CSV_FILE = "submission.csv"

def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Brak pliku {CSV_FILE}! Upewnij się, że jest w tym samym folderze.")

    if not API_TOKEN or not SERVER_URL:
        raise ValueError("Brak TEAM_TOKEN lub SERVER_URL. Sprawdź plik .env!")

    headers = {
        "X-API-Token": API_TOKEN
    }

    print(f"Wysyłam {CSV_FILE} do {SERVER_URL}/{ENDPOINT} ...")

    # Important, the name of key in files - "csv_file" must be exact
    with open(CSV_FILE, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/{ENDPOINT}",
            files={"csv_file": f},
            headers=headers
        )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("Response:", response.status_code, data)

if __name__ == "__main__":
    main()
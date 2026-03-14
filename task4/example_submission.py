import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task4"


API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


def _import_from_src(module_name: str, file_name: str):
    module_path = SRC_DIR / file_name
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module file: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ecg_digitizer_module = _import_from_src("ecg_digitizer", "ecg_digitizer.py")
digitize_ecg_file = ecg_digitizer_module.digitize_ecg_file

try:
    import torch
    signal_corrector_module = _import_from_src("signal_corrector", "signal_corrector.py")
    ResidualCorrector = signal_corrector_module.ResidualCorrector
except Exception:
    torch = None
    ResidualCorrector = None


NPZ_FILE = BASE_DIR / "data" / "out" / "task4_submission.npz"
TEST_IMAGES_DIR = SRC_DIR / "data" / "test"
CORRECTOR_PATH = BASE_DIR / "data" / "out" / "signal_corrector_recordsplit.pt"
STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _load_corrector(model_path: Path):
    if torch is None or ResidualCorrector is None or not model_path.exists():
        return None, 0.0, 1.0

    ckpt = torch.load(model_path, map_location="cpu")
    model = ResidualCorrector(
        channels=int(ckpt.get("channels", 64)),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    if std < 1e-6:
        std = 1.0
    return model, mean, std


def _correct_signal(model, mean: float, std: float, signal: np.ndarray) -> np.ndarray:
    if model is None or torch is None:
        return signal
    with torch.no_grad():
        norm = (signal.astype(np.float32) - mean) / std
        x = torch.from_numpy(norm[None, None, :])
        pred = model(x).numpy()[0, 0, :]
    corrected = pred * std + mean
    return corrected.astype(np.float32)


def generate_submission_npz() -> None:
    image_paths = sorted(TEST_IMAGES_DIR.glob("ecg_test_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES_DIR}")

    model, mean, std = _load_corrector(CORRECTOR_PATH)
    if model is None:
        print("Corrector model unavailable, using raw digitizer output")
    else:
        print(f"Using corrector model: {CORRECTOR_PATH}")

    submission_dict = {}
    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        record_name = image_path.stem
        signals, _ = digitize_ecg_file(image_path)
        for lead in STANDARD_LEADS:
            key = f"{record_name}_{lead}"
            signal = signals.get(lead)
            if signal is None:
                signal = np.zeros(1250, dtype=np.float32)
            signal = _correct_signal(model, mean, std, signal)
            submission_dict[key] = signal.astype(np.float16)

        if idx % 25 == 0 or idx == total:
            print(f"Processed {idx}/{total} records")

    os.makedirs(NPZ_FILE.parent, exist_ok=True)
    np.savez_compressed(NPZ_FILE, **submission_dict)
    print(f"Saved submission file to {NPZ_FILE}")


def main():
    generate_submission_npz()

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

    with open(NPZ_FILE, "rb") as handle:
        response = requests.post(
            f"{SERVER_URL}/{ENDPOINT}",
            files={"npz_file": handle},
            headers=headers,
        )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

# Add the current directory (src) to sys.path so we can import local modules
# This allows running from project root as `python src/generate_submission.py`
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import numpy as np
import torch

from ecg_digitizer import digitize_ecg_file
from signal_corrector import ResidualCorrector

STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _load_corrector(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Corrector model file not found: {model_path}")

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
    with torch.no_grad():
        norm = (signal.astype(np.float32) - mean) / std
        x = torch.from_numpy(norm[None, None, :])
        pred = model(x).numpy()[0, 0, :]
    corrected = pred * std + mean
    return corrected.astype(np.float32)


def generate_submission(images_dir: Path, model_path: Path, output_file: Path) -> None:
    image_paths = sorted(images_dir.glob("ecg_test_*.png"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Loading model from {model_path}...")
    model, mean, std = _load_corrector(model_path)
    
    submission_dict = {}
    total = len(image_paths)
    print(f"Processing {total} images from {images_dir}...")

    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        record_name = image_path.stem
        try:
            signals, _ = digitize_ecg_file(image_path)
            
            for lead in STANDARD_LEADS:
                key = f"{record_name}_{lead}"
                signal = signals.get(lead)
                
                # Handle missing leads or digitization failures gracefully
                if signal is None:
                    # Fallback: zeros
                    signal = np.zeros(1250, dtype=np.float32)
                
                # Apply correction
                corrected_signal = _correct_signal(model, mean, std, signal)
                
                # Ensure float16 for submission
                submission_dict[key] = corrected_signal.astype(np.float16)

            success_count += 1
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")
            fail_count += 1
            # Still fill with zeros to avoid partial submission failure?
            # The instructions say "missing records... will be scored with 0".
            # Better to not crash.
            for lead in STANDARD_LEADS:
                 key = f"{record_name}_{lead}"
                 submission_dict[key] = np.zeros(1250, dtype=np.float16)


        if idx % 10 == 0 or idx == total:
             print(f"Processed {idx}/{total} ({success_count} ok, {fail_count} failed)")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **submission_dict)
    print(f"Successfully saved submission to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate ECG submission NPZ")
    parser.add_argument("--images-dir", type=Path, default=Path("data/test"), help="Directory containing test images")
    parser.add_argument("--model-path", type=Path, default=Path("src/signal_corrector_recordsplit.pt"), help="Path to trained corrector model")
    parser.add_argument("--output-file", type=Path, default=Path("submission.npz"), help="Path to output NPZ file")
    
    args = parser.parse_args()
    
    # Resolve paths relative to CWD if they are relative
    if not args.images_dir.is_absolute():
        args.images_dir = Path.cwd() / args.images_dir
    if not args.model_path.is_absolute():
        args.model_path = Path.cwd() / args.model_path
    if not args.output_file.is_absolute():
        args.output_file = Path.cwd() / args.output_file

    generate_submission(args.images_dir, args.model_path, args.output_file)


if __name__ == "__main__":
    sys.exit(main())

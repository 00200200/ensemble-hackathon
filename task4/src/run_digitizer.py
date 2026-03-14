from pathlib import Path
import sys

import cv2
import numpy as np

from ecg_digitizer import digitize_ecg_file


def _save_debug_images(output_dir: Path, record_name: str, debug: dict) -> None:
	for name, value in debug.items():
		if isinstance(value, np.ndarray) and value.ndim in (2, 3):
			cv2.imwrite(str(output_dir / f"{record_name}_{name}.png"), value)
		elif isinstance(value, dict):
			for sub_name, sub_value in value.items():
				if not isinstance(sub_value, dict):
					continue
				for image_name, image_value in sub_value.items():
					if isinstance(image_value, np.ndarray) and image_value.ndim in (2, 3):
						cv2.imwrite(str(output_dir / f"{record_name}_{sub_name}_{image_name}.png"), image_value)


def main() -> int:
	default_image = Path(r"C:\Users\rondo\Desktop\ensemble\task4\src\data\train\ecg_train_0001.png")
	image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_image
	record_name = image_path.stem
	signals, debug = digitize_ecg_file(image_path)

	output_dir = Path(r"C:\Users\rondo\Desktop\ensemble\task4\data\out")
	output_dir.mkdir(parents=True, exist_ok=True)

	npz_payload = {f"{record_name}_{lead}": wave.astype(np.float16) for lead, wave in signals.items()}
	npz_path = output_dir / f"{record_name}_digitized.npz"
	np.savez_compressed(npz_path, **npz_payload)

	_save_debug_images(output_dir, record_name, debug)

	print(f"Saved digitized signals to {npz_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
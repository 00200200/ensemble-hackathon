import argparse
from pathlib import Path

import numpy as np

from ecg_digitizer import digitize_ecg_file


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Batch digitize ECG train images to NPZ")
	parser.add_argument("--images-dir", default="task4/src/data/train")
	parser.add_argument("--out-dir", default="task4/data/out")
	parser.add_argument("--glob", default="ecg_train_*.png")
	parser.add_argument("--max-images", type=int, default=0, help="0 means all")
	parser.add_argument("--overwrite", action="store_true")
	return parser


def main() -> int:
	args = _build_parser().parse_args()
	images_dir = Path(args.images_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	image_paths = sorted(images_dir.glob(args.glob))
	if args.max_images > 0:
		image_paths = image_paths[: args.max_images]
	if not image_paths:
		print(f"No images matched in {images_dir} with pattern {args.glob}")
		return 1

	ok = 0
	fail = 0
	for idx, image_path in enumerate(image_paths, start=1):
		record_name = image_path.stem
		npz_path = out_dir / f"{record_name}_digitized.npz"
		if npz_path.exists() and not args.overwrite:
			print(f"[{idx}/{len(image_paths)}] skip existing: {npz_path.name}")
			continue
		try:
			signals, _ = digitize_ecg_file(image_path)
			npz_payload = {f"{record_name}_{lead}": wave.astype(np.float16) for lead, wave in signals.items()}
			np.savez_compressed(npz_path, **npz_payload)
			ok += 1
			print(f"[{idx}/{len(image_paths)}] saved: {npz_path.name}")
		except Exception as exc:
			fail += 1
			print(f"[{idx}/{len(image_paths)}] failed: {image_path.name} -> {exc}")

	print()
	print(f"Done. created={ok}, failed={fail}, total={len(image_paths)}")
	return 0 if fail == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())

import csv
import os
import sys
from pathlib import Path

import cv2


def _collect_pngs(root_dirs: list[Path]) -> list[Path]:
	images: list[Path] = []
	for root in root_dirs:
		if not root.exists():
			continue
		for dirpath, _, filenames in os.walk(root):
			for name in filenames:
				if name.lower().endswith(".png"):
					images.append(Path(dirpath) / name)
	return sorted(images)


def _load_existing(csv_path: Path) -> dict[str, str]:
	if not csv_path.exists():
		return {}
	labels: dict[str, str] = {}
	with csv_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		next(reader, None)
		for row in reader:
			if len(row) >= 2:
				labels[row[0]] = row[1]
	return labels


def main() -> int:
	base_dir = Path(__file__).resolve().parents[1]
	train_dir = base_dir / "src" / "data" / "train"
	test_dir = base_dir / "src" / "data" / "test"

	output_csv = base_dir / "labels.csv"
	existing = _load_existing(output_csv)

	images = _collect_pngs([train_dir, test_dir])
	if not images:
		print("No PNG images found.")
		return 1

	window_name = "Labeling (b=easy, n=medium, m=hard, s=skip, q=quit)"
	max_width = 1600
	max_height = 900
	rows: list[tuple[str, str]] = []

	for image_path in images:
		rel_name = image_path.name
		if rel_name in existing:
			continue

		image = cv2.imread(str(image_path))
		if image is None:
			print(f"Failed to read: {image_path}")
			continue

		h, w = image.shape[:2]
		scale = min(max_width / w, max_height / h, 1.0)
		if scale < 1.0:
			new_w = int(w * scale)
			new_h = int(h * scale)
			display = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		else:
			display = image
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, display)
		while True:
			key = cv2.waitKey(0) & 0xFF
			if key == ord("q"):
				cv2.destroyAllWindows()
				break
			if key == ord("s"):
				break
			if key == ord("b"):
				rows.append((rel_name, "easy"))
				break
			if key == ord("n"):
				rows.append((rel_name, "medium"))
				break
			if key == ord("m"):
				rows.append((rel_name, "hard"))
				break

		if key == ord("q"):
			break

	cv2.destroyAllWindows()

	if rows:
		write_header = not output_csv.exists()
		with output_csv.open("a", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle)
			if write_header:
				writer.writerow(["image", "label"])
			writer.writerows(rows)

	print(f"Saved {len(rows)} labels to {output_csv}")
	return 0


if __name__ == "__main__":
	sys.exit(main())

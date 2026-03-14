import csv
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Metrics:
	sharpness: float
	contrast: float
	brightness: float
	shadow: float
	edge_density: float


def _compute_metrics(image: np.ndarray) -> Metrics:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
	contrast = float(gray.std())
	brightness = float(gray.mean())
	shadow = float(np.percentile(gray, 10))
	edges = cv2.Canny(gray, 50, 150)
	edge_density = float(edges.mean() / 255.0)
	return Metrics(sharpness, contrast, brightness, shadow, edge_density)


def _metrics_to_vec(metrics: Metrics) -> np.ndarray:
	return np.array(
		[
			metrics.sharpness,
			metrics.contrast,
			metrics.brightness,
			metrics.shadow,
			metrics.edge_density,
		],
		dtype=np.float32,
	)


def _collect_pngs(root_dirs: list[Path]) -> dict[str, Path]:
	images: dict[str, Path] = {}
	for root in root_dirs:
		if not root.exists():
			continue
		for dirpath, _, filenames in os.walk(root):
			for name in filenames:
				if name.lower().endswith(".png"):
					images[name] = Path(dirpath) / name
	return images


def _read_labels(csv_path: Path) -> list[tuple[str, str]]:
	with csv_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		next(reader, None)
		return [(row[0], row[1]) for row in reader if len(row) >= 2]


def train_centroid_model(labels_csv: Path, train_roots: list[Path]) -> tuple[np.ndarray, list[str]]:
	images = _collect_pngs(train_roots)
	labels = _read_labels(labels_csv)

	vectors: list[np.ndarray] = []
	classes: list[str] = []
	for name, label in labels:
		path = images.get(name)
		if path is None:
			continue
		image = cv2.imread(str(path))
		if image is None:
			continue
		metrics = _compute_metrics(image)
		vectors.append(_metrics_to_vec(metrics))
		classes.append(label)

	if not vectors:
		raise ValueError("No training data found")

	# Normalize features for stability.
	data = np.vstack(vectors)
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	std[std < 1e-6] = 1.0
	data = (data - mean) / std

	labels_sorted = sorted(set(classes))
	centroids = []
	for label in labels_sorted:
		idx = [i for i, c in enumerate(classes) if c == label]
		centroids.append(data[idx].mean(axis=0))

	model = np.vstack(centroids)
	return (model, labels_sorted, mean, std)


def predict_image(image_path: Path, model: np.ndarray, labels: list[str], mean: np.ndarray, std: np.ndarray) -> str:
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Failed to read {image_path}")
	vec = _metrics_to_vec(_compute_metrics(image))
	vec = (vec - mean) / std
	dists = np.linalg.norm(model - vec, axis=1)
	return labels[int(np.argmin(dists))]


def main() -> int:
	base_dir = Path(__file__).resolve().parents[1]
	labels_csv = base_dir / "labels.csv"
	train_dir = base_dir / "src" / "data" / "train"
	test_dir = base_dir / "src" / "data" / "test"

	if not labels_csv.exists():
		print("labels.csv not found")
		return 1

	model, labels, mean, std = train_centroid_model(labels_csv, [train_dir, test_dir])
	print(f"Trained centroid model on {len(labels)} classes: {labels}")

	# Example usage: predict a single image.
	# Replace with any file path.
	example = next(iter(_collect_pngs([train_dir, test_dir]).values()), None)
	if example is None:
		print("No images found for demo")
		return 0
	pred = predict_image(example, model, labels, mean, std)
	print(f"Example: {example.name} -> {pred}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


@dataclass
class Sample:
	path: Path
	label: int


class EkgQualityDataset(Dataset):
	def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
		self.samples = samples
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		sample = self.samples[idx]
		image = Image.open(sample.path).convert("RGB")
		image = self.transform(image)
		return image, sample.label


def _read_labels(labels_csv: Path) -> dict[str, str]:
	labels: dict[str, str] = {}
	with labels_csv.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		next(reader, None)
		for row in reader:
			if len(row) >= 2:
				labels[row[0]] = row[1]
	return labels


def _collect_samples(root: Path, labels: dict[str, str], class_to_idx: dict[str, int]) -> list[Sample]:
	samples: list[Sample] = []
	for path in root.rglob("*.png"):
		label_name = labels.get(path.name)
		if label_name is None:
			continue
		samples.append(Sample(path=path, label=class_to_idx[label_name]))
	return samples


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
	indices = list(range(n))
	random.Random(seed).shuffle(indices)
	val_count = int(n * val_ratio)
	val_idx = indices[:val_count]
	train_idx = indices[val_count:]
	return train_idx, val_idx


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			logits = model(images)
			preds = torch.argmax(logits, dim=1)
			correct += int((preds == labels).sum().item())
			total += int(labels.size(0))
	return correct / max(total, 1)


def main() -> int:
	base_dir = Path(__file__).resolve().parents[1]
	labels_csv = base_dir / "labels.csv"
	train_root = base_dir / "src" / "data" / "train"
	test_root = base_dir / "src" / "data" / "test"

	if not labels_csv.exists():
		print("labels.csv not found")
		return 1

	labels = _read_labels(labels_csv)
	classes = sorted({label for label in labels.values()})
	class_to_idx = {name: idx for idx, name in enumerate(classes)}

	train_samples = _collect_samples(train_root, labels, class_to_idx)
	test_samples = _collect_samples(test_root, labels, class_to_idx)
	if not train_samples:
		print("No training samples found")
		return 1

	seed = 42
	train_idx, val_idx = _split_indices(len(train_samples), val_ratio=0.1, seed=seed)

	train_tf = transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(p=0.2),
			transforms.RandomRotation(3),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	val_tf = transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	train_ds = EkgQualityDataset(train_samples, train_tf)
	val_ds = EkgQualityDataset(train_samples, val_tf)
	train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=16, shuffle=True, num_workers=2)
	val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=16, shuffle=False, num_workers=2)

	if test_samples:
		test_ds = EkgQualityDataset(test_samples, val_tf)
		test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
	else:
		test_loader = None

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	model.fc = nn.Linear(model.fc.in_features, len(classes))
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	epochs = 8
	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		for images, labels_t in train_loader:
			images = images.to(device)
			labels_t = labels_t.to(device)
			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, labels_t)
			loss.backward()
			optimizer.step()
			running_loss += float(loss.item())

		val_acc = _accuracy(model, val_loader, device)
		print(f"Epoch {epoch}/{epochs} loss={running_loss:.4f} val_acc={val_acc:.3f}")

	if test_loader is not None:
		test_acc = _accuracy(model, test_loader, device)
		print(f"Test accuracy: {test_acc:.3f}")

	out_path = base_dir / "quality_cnn.pt"
	torch.save(
		{
			"state_dict": model.state_dict(),
			"classes": classes,
		},
		out_path,
	)
	print(f"Saved model to {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

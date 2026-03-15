import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from compare_npz_to_ground_truth import LeadPair, collect_pairs_from_npz


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class SignalSample:
	record: str
	x: np.ndarray
	y: np.ndarray


class ResidualBlock(nn.Module):
	def __init__(self, channels: int, dilation: int, dropout: float) -> None:
		super().__init__()
		padding = dilation * 3
		self.block = nn.Sequential(
			nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=padding, bias=False),
			nn.BatchNorm1d(channels),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Conv1d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm1d(channels),
			nn.GELU(),
			nn.Dropout(dropout),
		)
		self.se = nn.Sequential(
			nn.AdaptiveAvgPool1d(1),
			nn.Conv1d(channels, channels // 8, 1),
			nn.ReLU(),
			nn.Conv1d(channels // 8, channels, 1),
			nn.Sigmoid(),
		)
		self.act = nn.GELU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.block(x)
		# Squeeze-and-Excitation
		w = self.se(out)
		out = out * w
		return self.act(x + out)


class ResidualCorrector(nn.Module):
	def __init__(self, channels: int = 128, dropout: float = 0.2, dilations: Iterable[int] = None) -> None:
		super().__init__()
		if dilations is None:
			# 16 residual blocks with exponentially increasing dilation
			dilations = [2 ** i for i in range(8)] * 2
		self.stem = nn.Sequential(
			nn.Conv1d(1, channels, kernel_size=15, padding=7, bias=False),
			nn.BatchNorm1d(channels),
			nn.GELU(),
		)
		self.body = nn.Sequential(*[ResidualBlock(channels, dilation=d, dropout=dropout) for d in dilations])
		self.head = nn.Sequential(
			nn.Conv1d(channels, channels, kernel_size=7, padding=3, bias=False),
			nn.BatchNorm1d(channels),
			nn.GELU(),
			nn.Conv1d(channels, 1, kernel_size=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.body(self.stem(x))
		return x + self.head(features)


def _is_train_key(key: str) -> bool:
	if not key.startswith("ecg_train_"):
		return False
	return any(key.endswith(f"_{lead}") for lead in LEADS)


def _corrcoef_np(a: np.ndarray, b: np.ndarray) -> float:
	if a.size < 2 or b.size < 2:
		return 0.0
	if np.std(a) < 1e-8 or np.std(b) < 1e-8:
		return 0.0
	return float(np.corrcoef(a, b)[0, 1])


def _collect_training_samples(npz_paths: list[Path], train_dir: Path) -> list[SignalSample]:
	all_pairs: list[LeadPair] = []
	for npz_path in npz_paths:
		all_pairs.extend(collect_pairs_from_npz(npz_path, train_dir))
	if not all_pairs:
		raise ValueError("No training pairs found. Generate ecg_train_*_digitized.npz files first.")
	return [SignalSample(record=pair.record, x=pair.pred.astype(np.float32), y=pair.gt.astype(np.float32)) for pair in all_pairs]


def _samples_to_arrays(samples: list[SignalSample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
	if not samples:
		raise ValueError("No samples to convert")
	x = np.stack([sample.x for sample in samples], axis=0)
	y = np.stack([sample.y for sample in samples], axis=0)
	records = [sample.record for sample in samples]
	return x, y, records


def _split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	n = x.shape[0]
	idx = np.arange(n)
	rng = np.random.default_rng(42)
	rng.shuffle(idx)
	val_n = max(1, int(round(n * val_ratio)))
	val_idx = idx[:val_n]
	train_idx = idx[val_n:]
	if train_idx.size == 0:
		train_idx = val_idx
	return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _split_train_val_by_record(samples: list[SignalSample], val_ratio: float, seed: int) -> tuple[list[SignalSample], list[SignalSample]]:
	unique_records = sorted({sample.record for sample in samples})
	if len(unique_records) < 2:
		return samples, samples
	rng = np.random.default_rng(seed)
	perm = np.array(unique_records, dtype=object)
	rng.shuffle(perm)
	val_n = max(1, int(round(len(perm) * val_ratio)))
	val_records = set(perm[:val_n].tolist())
	train_records = set(perm[val_n:].tolist())
	if not train_records:
		train_records = set([next(iter(val_records))])
	train_samples = [sample for sample in samples if sample.record in train_records]
	val_samples = [sample for sample in samples if sample.record in val_records]
	if not train_samples:
		train_samples = val_samples
	if not val_samples:
		val_samples = train_samples
	return train_samples, val_samples


def _to_tensor3d(arr: np.ndarray) -> torch.Tensor:
	return torch.from_numpy(arr[:, None, :])


class AugmentedSignalDataset(Dataset):
	def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool, noise_std: float, amplitude_jitter: float, baseline_jitter: float) -> None:
		self.x = _to_tensor3d(x).float()
		self.y = _to_tensor3d(y).float()
		self.augment = augment
		self.noise_std = float(max(0.0, noise_std))
		self.amplitude_jitter = float(max(0.0, amplitude_jitter))
		self.baseline_jitter = float(max(0.0, baseline_jitter))

	def __len__(self) -> int:
		return int(self.x.shape[0])

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		x = self.x[idx].clone()
		y = self.y[idx]
		if self.augment:
			if self.amplitude_jitter > 0.0:
				scale = 1.0 + torch.empty(1).uniform_(-self.amplitude_jitter, self.amplitude_jitter)
				x = x * scale
			if self.baseline_jitter > 0.0:
				offset = torch.empty(1).uniform_(-self.baseline_jitter, self.baseline_jitter)
				x = x + offset
			if self.noise_std > 0.0:
				x = x + torch.randn_like(x) * self.noise_std
		return x, y


def _evaluate_numpy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, float, float]:
	model.eval()
	with torch.no_grad():
		pred = model(torch.from_numpy(x[:, None, :]).to(device)).cpu().numpy()[:, 0, :]
	mae = float(np.mean(np.abs(pred - y)))
	rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
	corrs = [_corrcoef_np(pred[i], y[i]) for i in range(pred.shape[0])]
	corr = float(np.mean(corrs))
	return mae, rmse, corr


def _batch_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	pred_centered = pred - pred.mean(dim=-1, keepdim=True)
	target_centered = target - target.mean(dim=-1, keepdim=True)
	num = (pred_centered * target_centered).mean(dim=-1)
	den = torch.sqrt((pred_centered.pow(2).mean(dim=-1) + eps) * (target_centered.pow(2).mean(dim=-1) + eps))
	corr = num / (den + eps)
	return 1.0 - corr.mean()


def _composite_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
	mse = torch.mean((pred - target) ** 2)
	l1 = torch.mean(torch.abs(pred - target))
	pred_diff = pred[..., 1:] - pred[..., :-1]
	target_diff = target[..., 1:] - target[..., :-1]
	deriv = torch.mean((pred_diff - target_diff) ** 2)
	corr = _batch_corr_loss(pred, target)
	return mse + args.l1_weight * l1 + args.derivative_weight * deriv + args.corr_weight * corr


def train_corrector(args: argparse.Namespace) -> int:
	base_dir = Path(__file__).resolve().parents[1]
	train_dir = base_dir / "src" / "data" / "train"
	npz_paths = sorted(Path().glob(args.npz_glob))
	if not npz_paths:
		print(f"No NPZ files matched: {args.npz_glob}")
		return 1

	samples = _collect_training_samples(npz_paths, train_dir)
	if args.split_by_record:
		train_samples, val_samples = _split_train_val_by_record(samples, val_ratio=args.val_ratio, seed=args.seed)
		if len({sample.record for sample in samples}) < 2:
			print("WARN: fewer than 2 unique records, using same data for train/val")
		x_train, y_train, train_records = _samples_to_arrays(train_samples)
		x_val, y_val, val_records = _samples_to_arrays(val_samples)
		print(f"Record split: train_records={len(set(train_records))} val_records={len(set(val_records))}")
	else:
		x, y, _ = _samples_to_arrays(samples)
		x_train, y_train, x_val, y_val = _split_train_val(x, y, val_ratio=args.val_ratio)

	mean = float(np.mean(x_train))
	std = float(np.std(x_train))
	if std < 1e-6:
		std = 1.0
	x_train_n = (x_train - mean) / std
	x_val_n = (x_val - mean) / std
	y_train_n = (y_train - mean) / std
	y_val_n = (y_val - mean) / std

	train_ds = AugmentedSignalDataset(
		x_train_n,
		y_train_n,
		augment=not args.no_augment,
		noise_std=args.aug_noise_std,
		amplitude_jitter=args.aug_amplitude_jitter,
		baseline_jitter=args.aug_baseline_jitter,
	)
	val_ds = TensorDataset(_to_tensor3d(x_val_n).float(), _to_tensor3d(y_val_n).float())
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ResidualCorrector(channels=args.channels, dropout=args.dropout).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

	best_val = float("inf")
	best_state = None
	best_epoch = 0
	no_improve_epochs = 0

	for epoch in range(1, args.epochs + 1):
		model.train()
		run_loss = 0.0
		for xb, yb in train_loader:
			xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
			yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
			opt.zero_grad()
			pred = model(xb)
			loss = _composite_loss(pred, yb, args)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
			opt.step()
			run_loss += float(loss.item())

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
				yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
				pred = model(xb)
				val_loss += float(_composite_loss(pred, yb, args).item())

		scheduler.step(epoch)

		if val_loss < best_val:
			best_val = val_loss
			best_epoch = epoch
			no_improve_epochs = 0
			best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
		else:
			no_improve_epochs += 1

		current_lr = opt.param_groups[0]["lr"]
		print(f"Epoch {epoch}/{args.epochs} train_loss={run_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.2e}")

		if no_improve_epochs >= args.patience:
			print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
			break

	if best_state is None:
		print("Training failed: no best checkpoint")
		return 1

	model.load_state_dict(best_state)
	mae_before = float(np.mean(np.abs(x_val - y_val)))
	rmse_before = float(np.sqrt(np.mean((x_val - y_val) ** 2)))
	corr_before = float(np.mean([_corrcoef_np(x_val[i], y_val[i]) for i in range(x_val.shape[0])]))
	mae_after, rmse_after, corr_after = _evaluate_numpy(model, x_val_n, y_val_n, device)
	mae_after *= std
	rmse_after *= std

	print("Validation before correction:")
	print(f"  MAE={mae_before:.4f} RMSE={rmse_before:.4f} Corr={corr_before:.4f}")
	print("Validation after correction:")
	print(f"  MAE={mae_after:.4f} RMSE={rmse_after:.4f} Corr={corr_after:.4f}")

	out_path = Path(args.model_out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"state_dict": best_state,
			"mean": mean,
			"std": std,
			"channels": args.channels,
			"dropout": args.dropout,
		},
		out_path,
	)
	print(f"Saved corrector model to {out_path}")
	return 0


def apply_corrector(args: argparse.Namespace) -> int:
	ckpt = torch.load(args.model_path, map_location="cpu")
	model = ResidualCorrector(
		channels=int(ckpt.get("channels", 64)),
		dropout=float(ckpt.get("dropout", 0.1)),
	)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()

	mean = float(ckpt["mean"])
	std = float(ckpt["std"])
	if std < 1e-6:
		std = 1.0

	inp = np.load(args.input_npz)
	out: dict[str, np.ndarray] = {}
	with torch.no_grad():
		for key in inp.files:
			arr = inp[key].astype(np.float32)
			if arr.ndim != 1 or arr.shape[0] != 1250 or not _is_train_key(key.replace("ecg_test_", "ecg_train_")):
				out[key] = arr.astype(np.float16)
				continue
			norm = (arr - mean) / std
			x = torch.from_numpy(norm[None, None, :])
			pred = model(x).numpy()[0, 0, :]
			corrected = pred * std + mean
			out[key] = corrected.astype(np.float16)

	output_path = Path(args.output_npz)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(output_path, **out)
	print(f"Saved corrected NPZ to {output_path}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train/apply a neural signal corrector for digitized ECG NPZ files")
	sub = parser.add_subparsers(dest="command", required=True)

	train = sub.add_parser("train", help="Train corrector from train NPZ vs ground truth")
	train.add_argument("--npz-glob", default="task4/data/out/ecg_train_*_digitized.npz")
	train.add_argument("--model-out", default="task4/data/out/signal_corrector.pt")
	train.add_argument("--epochs", type=int, default=200)
	train.add_argument("--batch-size", type=int, default=128)
	train.add_argument("--lr", type=float, default=2e-3)
	train.add_argument("--channels", type=int, default=128)
	train.add_argument("--dropout", type=float, default=0.2)
	train.add_argument("--weight-decay", type=float, default=5e-4)
	train.add_argument("--grad-clip", type=float, default=1.0)
	train.add_argument("--patience", type=int, default=10)
	train.add_argument("--l1-weight", type=float, default=0.05)
	train.add_argument("--derivative-weight", type=float, default=0.2)
	train.add_argument("--corr-weight", type=float, default=0.15)
	train.add_argument("--val-ratio", type=float, default=0.1)
	train.add_argument("--seed", type=int, default=42)
	train.add_argument("--split-by-record", action="store_true", default=True)
	train.add_argument("--no-split-by-record", dest="split_by_record", action="store_false")
	train.add_argument("--no-augment", action="store_true")
	train.add_argument("--aug-noise-std", type=float, default=0.02)
	train.add_argument("--aug-amplitude-jitter", type=float, default=0.08)
	train.add_argument("--aug-baseline-jitter", type=float, default=0.04)
	train.set_defaults(func=train_corrector)

	apply = sub.add_parser("apply", help="Apply trained corrector to NPZ")
	apply.add_argument("--model-path", required=True)
	apply.add_argument("--input-npz", required=True)
	apply.add_argument("--output-npz", required=True)
	apply.set_defaults(func=apply_corrector)

	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return int(args.func(args))


if __name__ == "__main__":
	raise SystemExit(main())

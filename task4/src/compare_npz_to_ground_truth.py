import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ALIASES = {
	"aVR": "AVR",
	"aVL": "AVL",
	"aVF": "AVF",
	"AVR": "AVR",
	"AVL": "AVL",
	"AVF": "AVF",
}


@dataclass
class LeadMetrics:
	record: str
	lead: str
	mae: float
	rmse: float
	corr: float
	pred_min: float
	pred_max: float
	gt_min: float
	gt_max: float


@dataclass
class LeadPair:
	record: str
	lead: str
	pred: np.ndarray
	gt: np.ndarray


@dataclass
class HeaderLead:
	gain: float
	baseline: float
	name: str


@dataclass
class HeaderInfo:
	record_name: str
	n_leads: int
	fs: float
	n_samples: int
	leads: list[HeaderLead]


def _normalize_lead_name(name: str) -> str:
	return LEAD_ALIASES.get(name, name)


def _split_npz_key(key: str) -> tuple[str, str] | tuple[None, None]:
	for lead in STANDARD_LEADS:
		suffix = f"_{lead}"
		if key.endswith(suffix):
			return key[: -len(suffix)], lead
	return None, None


def _read_header(header_path: Path) -> HeaderInfo:
	lines = header_path.read_text(encoding="utf-8").splitlines()
	first = lines[0].split()
	record_name = first[0]
	n_leads = int(first[1])
	fs = float(first[2])
	n_samples = int(first[3])
	leads: list[HeaderLead] = []
	for line in lines[1 : 1 + n_leads]:
		parts = line.split()
		gain_baseline_unit = parts[2]
		gain_str, rest = gain_baseline_unit.split("(")
		baseline_str = rest.split(")", maxsplit=1)[0]
		lead_name = _normalize_lead_name(parts[-1])
		leads.append(HeaderLead(gain=float(gain_str), baseline=float(baseline_str), name=lead_name))
	return HeaderInfo(record_name=record_name, n_leads=n_leads, fs=fs, n_samples=n_samples, leads=leads)


def _read_dat_signal(dat_path: Path, header: HeaderInfo) -> dict[str, np.ndarray]:
	raw = np.fromfile(dat_path, dtype="<i2")
	expected = header.n_samples * header.n_leads
	if raw.size != expected:
		raise ValueError(f"Unexpected DAT size for {dat_path.name}: got {raw.size}, expected {expected}")
	raw = raw.reshape(header.n_samples, header.n_leads)
	signals: dict[str, np.ndarray] = {}
	for idx, lead in enumerate(header.leads):
		physical = (raw[:, idx].astype(np.float32) - lead.baseline) / lead.gain
		signals[lead.name] = physical
	return signals


def _read_segments(json_path: Path) -> dict[str, tuple[int, int]]:
	payload = json.loads(json_path.read_text(encoding="utf-8"))
	segments: dict[str, tuple[int, int]] = {}
	for lead in payload.get("leads", []):
		lead_name = _normalize_lead_name(lead["lead_name"])
		start = int(round(lead["start_sample"]))
		end = int(round(lead["end_sample"]))
		if lead_name not in segments:
			segments[lead_name] = (start, end)
	return segments


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
	if a.size < 2 or b.size < 2:
		return 0.0
	if np.std(a) < 1e-8 or np.std(b) < 1e-8:
		return 0.0
	return float(np.corrcoef(a, b)[0, 1])


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
	if signal.shape[0] == target_len:
		return signal.astype(np.float32)
	x_old = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
	x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
	return np.interp(x_new, x_old, signal).astype(np.float32)


def _load_npz_predictions(npz_path: Path) -> dict[str, dict[str, np.ndarray]]:
	data = np.load(npz_path)
	predictions: dict[str, dict[str, np.ndarray]] = {}
	for key in data.files:
		record, lead = _split_npz_key(key)
		if record is None:
			continue
		predictions.setdefault(record, {})[lead] = data[key].astype(np.float32)
	return predictions


def align_record_pairs(record: str, predicted: dict[str, np.ndarray], train_dir: Path) -> list[LeadPair]:
	header_path = train_dir / f"{record}.hea"
	dat_path = train_dir / f"{record}.dat"
	json_path = train_dir / f"{record}.json"
	if not header_path.exists() or not dat_path.exists() or not json_path.exists():
		raise FileNotFoundError(f"Missing ground truth files for {record}")

	header = _read_header(header_path)
	gt_full = _read_dat_signal(dat_path, header)
	segments = _read_segments(json_path)
	pairs: list[LeadPair] = []

	for lead in STANDARD_LEADS:
		if lead not in predicted:
			continue
		if lead not in gt_full:
			continue
		start, end = segments.get(lead, (0, min(1250, gt_full[lead].shape[0])))
		gt_segment = gt_full[lead][start:end]
		pred_signal = predicted[lead]
		gt_segment = _resample_1d(gt_segment, pred_signal.shape[0])
		pred_signal = _resample_1d(pred_signal, gt_segment.shape[0])

		pairs.append(LeadPair(record=record, lead=lead, pred=pred_signal, gt=gt_segment))
	return pairs


def compute_metrics(pairs: list[LeadPair]) -> list[LeadMetrics]:
	metrics: list[LeadMetrics] = []
	for pair in pairs:
		mae = float(np.mean(np.abs(pair.pred - pair.gt)))
		rmse = float(np.sqrt(np.mean((pair.pred - pair.gt) ** 2)))
		corr = _safe_corr(pair.pred, pair.gt)
		metrics.append(
			LeadMetrics(
				record=pair.record,
				lead=pair.lead,
				mae=mae,
				rmse=rmse,
				corr=corr,
				pred_min=float(np.min(pair.pred)),
				pred_max=float(np.max(pair.pred)),
				gt_min=float(np.min(pair.gt)),
				gt_max=float(np.max(pair.gt)),
			)
		)
	return metrics


def save_comparison_plots(
	pairs: list[LeadPair],
	metrics: list[LeadMetrics],
	plot_dir: Path,
	max_plots: int,
) -> None:
	plot_dir.mkdir(parents=True, exist_ok=True)
	metric_map = {(item.record, item.lead): item for item in metrics}
	for idx, pair in enumerate(pairs[:max_plots]):
		metric = metric_map[(pair.record, pair.lead)]
		time = np.linspace(0.0, 2.5, num=pair.pred.shape[0], endpoint=False)
		plt.figure(figsize=(10, 4))
		plt.plot(time, pair.gt, label="Ground Truth", linewidth=1.4)
		plt.plot(time, pair.pred, label="Prediction", linewidth=1.0, alpha=0.9)
		title = (
			f"{pair.record}_{pair.lead} | MAE={metric.mae:.3f} mV, "
			f"RMSE={metric.rmse:.3f} mV, Corr={metric.corr:.3f}"
		)
		plt.title(title)
		plt.xlabel("Time [s]")
		plt.ylabel("Amplitude [mV]")
		plt.grid(alpha=0.3)
		plt.legend(loc="upper right")
		plt.tight_layout()
		plot_path = plot_dir / f"{pair.record}_{pair.lead}_compare.png"
		plt.savefig(plot_path, dpi=140)
		plt.close()


def collect_pairs_from_npz(npz_path: Path, train_dir: Path) -> list[LeadPair]:
	predictions = _load_npz_predictions(npz_path)
	if not predictions:
		return []
	all_pairs: list[LeadPair] = []
	for record, predicted in sorted(predictions.items()):
		if not record.startswith("ecg_train_"):
			continue
		try:
			all_pairs.extend(align_record_pairs(record, predicted, train_dir))
		except FileNotFoundError:
			continue
	return all_pairs


def summarize(metrics: list[LeadMetrics]) -> None:
	if not metrics:
		print("No comparable leads found")
		return
	print("Per-lead metrics:")
	for item in metrics:
		print(
			f"{item.record}_{item.lead}: "
			f"MAE={item.mae:.4f} mV, RMSE={item.rmse:.4f} mV, Corr={item.corr:.4f}, "
			f"pred=[{item.pred_min:.3f}, {item.pred_max:.3f}], "
			f"gt=[{item.gt_min:.3f}, {item.gt_max:.3f}]"
		)
	mean_mae = float(np.mean([item.mae for item in metrics]))
	mean_rmse = float(np.mean([item.rmse for item in metrics]))
	mean_corr = float(np.mean([item.corr for item in metrics]))
	print()
	print(f"Summary: mean MAE={mean_mae:.4f} mV, mean RMSE={mean_rmse:.4f} mV, mean Corr={mean_corr:.4f}")


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Compare digitized NPZ signals with WFDB ground truth")
	parser.add_argument("npz_path", nargs="?", default=None, help="Path to NPZ file with ecg_train_* keys")
	parser.add_argument("--plot-dir", default=None, help="Directory to save prediction-vs-ground-truth plots")
	parser.add_argument("--max-plots", type=int, default=24, help="Maximum number of lead plots to save")
	return parser


def main() -> int:
	parser = _build_arg_parser()
	args = parser.parse_args()
	base_dir = Path(__file__).resolve().parents[1]
	default_npz = base_dir / "data" / "out" / "ecg_train_0001_digitized.npz"
	npz_path = Path(args.npz_path) if args.npz_path is not None else default_npz
	train_dir = base_dir / "src" / "data" / "train"

	if not npz_path.exists():
		print(f"NPZ file not found: {npz_path}")
		print("Usage: python compare_npz_to_ground_truth.py <path_to_train_npz>")
		return 1

	pairs = collect_pairs_from_npz(npz_path, train_dir)
	if not pairs:
		print(f"No valid predictions found in {npz_path}")
		return 1
	all_metrics = compute_metrics(pairs)
	summarize(all_metrics)

	if args.plot_dir is not None:
		plot_dir = Path(args.plot_dir)
		save_comparison_plots(pairs, all_metrics, plot_dir=plot_dir, max_plots=max(1, args.max_plots))
		print(f"Saved comparison plots to {plot_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

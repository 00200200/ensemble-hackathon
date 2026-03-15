from pathlib import Path

import numpy as np


EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ALLOWED_DTYPES = {np.float16, np.float32, np.float64}


def _split_key(key: str) -> tuple[str, str] | tuple[None, None]:
	for lead in EXPECTED_LEADS:
		suffix = f"_{lead}"
		if key.endswith(suffix):
			return key[: -len(suffix)], lead
	return None, None


def validate_npz(npz_path: Path) -> int:
	if not npz_path.exists():
		print(f"ERROR: file not found: {npz_path}")
		return 1

	data = np.load(npz_path)
	keys = sorted(data.files)
	if not keys:
		print("ERROR: NPZ is empty")
		return 1

	records: dict[str, dict[str, np.ndarray]] = {}
	unknown_keys: list[str] = []
	for key in keys:
		record, lead = _split_key(key)
		if record is None:
			unknown_keys.append(key)
			continue
		records.setdefault(record, {})[lead] = data[key]

	error_count = 0
	if unknown_keys:
		error_count += len(unknown_keys)
		print("ERROR: keys with invalid lead suffix:")
		for key in unknown_keys[:20]:
			print(f"  - {key}")
		if len(unknown_keys) > 20:
			print(f"  ... and {len(unknown_keys) - 20} more")

	for record, lead_map in sorted(records.items()):
		missing = [lead for lead in EXPECTED_LEADS if lead not in lead_map]
		extra = [lead for lead in lead_map if lead not in EXPECTED_LEADS]
		if missing:
			error_count += len(missing)
			print(f"ERROR: {record} missing leads: {missing}")
		if extra:
			error_count += len(extra)
			print(f"ERROR: {record} unknown leads: {extra}")

		for lead, arr in lead_map.items():
			if arr.ndim != 1:
				error_count += 1
				print(f"ERROR: {record}_{lead} is not 1D, shape={arr.shape}")
				continue
			if arr.shape[0] != 1250:
				error_count += 1
				print(f"ERROR: {record}_{lead} length={arr.shape[0]} (expected 1250)")
			if arr.dtype.type not in ALLOWED_DTYPES:
				error_count += 1
				print(f"ERROR: {record}_{lead} dtype={arr.dtype} (expected float16/float32/float64)")
			if not np.isfinite(arr).all():
				error_count += 1
				print(f"ERROR: {record}_{lead} has NaN/Inf")

			peak = float(np.max(np.abs(arr))) if arr.size else 0.0
			if peak > 10.0:
				print(f"WARN: {record}_{lead} peak={peak:.2f} mV seems unusually high")

	if error_count == 0:
		n_records = len(records)
		n_signals = sum(len(v) for v in records.values())
		print(f"OK: NPZ looks valid. records={n_records}, signals={n_signals}")
		return 0

	print(f"FAILED: found {error_count} validation issues")
	return 2


def main() -> int:
	default_npz = Path(r"C:\Users\rondo\Desktop\ensemble\task4\data\out\ecg_train_0001_digitized.npz")
	return validate_npz(default_npz)


if __name__ == "__main__":
	raise SystemExit(main())

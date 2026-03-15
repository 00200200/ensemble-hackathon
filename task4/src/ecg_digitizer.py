from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d


LEAD_LAYOUT = [
	["I", "AVR", "V1", "V4"],
	["II", "AVL", "V2", "V5"],
	["III", "AVF", "V3", "V6"],
]


@dataclass
class DigitizerConfig:
	grid_rows: int = 3
	grid_cols: int = 4
	samples_per_lead: int = 1250
	lead_duration_sec: float = 2.5
	adaptive_block_size: int = 41
	adaptive_c: int = 8
	min_component_area: int = 20
	bbox_padding: int = 12
	# Set horizontal trim to 0 to preserve time alignment (Time Calibration score)
	lead_left_trim_ratio: float = 0.0
	lead_right_trim_ratio: float = 0.0
	# Minimal vertical trim to avoid border noise, but kept small
	lead_top_trim_ratio: float = 0.02
	lead_bottom_trim_ratio: float = 0.02
	bottom_row_extra_trim_ratio: float = 0.0
	text_component_max_area_ratio: float = 0.02
	text_component_max_width_ratio: float = 0.18
	text_component_max_height_ratio: float = 0.35
	trace_max_jump_ratio: float = 0.12
	trace_smooth_window: int = 7
	trace_clip_millivolts: float = 6.0
	vertical_component_min_height_ratio: float = 0.45
	vertical_component_max_width_ratio: float = 0.06
	lead_min_foreground_ratio: float = 0.003


def _ensure_odd(value: int) -> int:
	return value if value % 2 == 1 else value + 1


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	if num_labels <= 1:
		return np.zeros_like(mask)
	keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
	keep[0] = False
	clean = np.where(keep[labels], 255, 0).astype(np.uint8)
	return clean


def _extract_colored_grid(image: np.ndarray) -> np.ndarray:
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	red1 = cv2.inRange(hsv, (0, 10, 70), (20, 255, 255))
	red2 = cv2.inRange(hsv, (160, 10, 70), (180, 255, 255))
	blue = cv2.inRange(hsv, (85, 10, 60), (140, 255, 255))
	grid = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel, iterations=1)
	return cv2.dilate(grid, kernel, iterations=1)


def _extract_line_grid(binary_mask: np.ndarray) -> np.ndarray:
	kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
	kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
	grid_h = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_h, iterations=1)
	grid_v = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
	grid = cv2.bitwise_or(grid_h, grid_v)
	return _remove_small_components(grid, min_area=50)


def build_signal_mask(image: np.ndarray, config: DigitizerConfig = DigitizerConfig()) -> tuple[np.ndarray, np.ndarray, dict]:
	green = image[:, :, 1]
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	green_eq = clahe.apply(green)
	block_size = _ensure_odd(config.adaptive_block_size)
	binary = cv2.adaptiveThreshold(
		green_eq,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		block_size,
		config.adaptive_c,
	)
	colored_grid = _extract_colored_grid(image)
	fallback_grid = _extract_line_grid(binary)
	grid_mask = colored_grid if cv2.countNonZero(colored_grid) > 0 else fallback_grid
	signal_mask = cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))
	signal_mask = _remove_small_components(signal_mask, min_area=config.min_component_area)
	close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
	signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
	debug = {
		"green": green_eq,
		"binary": binary,
		"grid_mask": grid_mask,
	}
	return signal_mask, grid_mask, debug


def _find_active_bands(profile: np.ndarray, min_size: int) -> list[tuple[int, int]]:
	if profile.size == 0 or profile.max() <= 0:
		return []
	window = max(7, (profile.size // 50) | 1)
	kernel = np.ones(window, dtype=np.float32) / window
	smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
	threshold = max(1.0, smooth.max() * 0.2)
	active = smooth >= threshold
	bands: list[tuple[int, int]] = []
	start = None
	for idx, flag in enumerate(active):
		if flag and start is None:
			start = idx
		elif not flag and start is not None:
			if idx - start >= min_size:
				bands.append((start, idx - 1))
			start = None
	if start is not None and len(active) - start >= min_size:
		bands.append((start, len(active) - 1))
	return bands


def _split_band_evenly(start: int, end: int, count: int) -> list[tuple[int, int]]:
	edges = np.linspace(start, end + 1, count + 1, dtype=int)
	return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(count)]


def _find_row_bands(signal_mask: np.ndarray, config: DigitizerConfig) -> list[tuple[int, int]]:
	profile = signal_mask.sum(axis=1).astype(np.float32) / 255.0
	bands = _find_active_bands(profile, min_size=max(8, signal_mask.shape[0] // 50))
	if len(bands) >= config.grid_rows:
		bands = sorted(bands, key=lambda item: item[1] - item[0], reverse=True)[: config.grid_rows]
		return sorted(bands, key=lambda item: item[0])
	return _split_band_evenly(0, signal_mask.shape[0] - 1, config.grid_rows)


def _find_valleys(profile: np.ndarray, expected_positions: list[int], radius: int) -> list[int]:
	valleys: list[int] = []
	for expected in expected_positions:
		left = max(0, expected - radius)
		right = min(len(profile), expected + radius + 1)
		window = profile[left:right]
		if window.size == 0:
			valleys.append(expected)
			continue
		valleys.append(left + int(np.argmin(window)))
	return valleys


def _find_column_bands(mask: np.ndarray, config: DigitizerConfig) -> list[tuple[int, int]]:
	h, w = mask.shape
	profile = mask.sum(axis=0).astype(np.float32) / 255.0
	expected = [int(round(w * frac / config.grid_cols)) for frac in range(1, config.grid_cols)]
	
	# Tighten search radius to avoid large time shifts (Time Calibration penalty)
	# Trust geometric structure for "Hard" (crumpled/rotated) images.
	valleys = _find_valleys(profile, expected, radius=max(5, w // 60))
	
	edges = [0] + valleys + [w]
	bands: list[tuple[int, int]] = []
	for i in range(config.grid_cols):
		start = int(edges[i])
		end = int(max(start + 1, edges[i + 1]))
		bands.append((start, end - 1))
	return bands


def _remove_text_components(mask: np.ndarray, config: DigitizerConfig) -> np.ndarray:
	h, w = mask.shape
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	clean = np.zeros_like(mask)
	max_area = max(8, int(h * w * config.text_component_max_area_ratio))
	max_width = max(8, int(w * config.text_component_max_width_ratio))
	max_height = max(8, int(h * config.text_component_max_height_ratio))
	for label_idx in range(1, num_labels):
		x = stats[label_idx, cv2.CC_STAT_LEFT]
		y = stats[label_idx, cv2.CC_STAT_TOP]
		cw = stats[label_idx, cv2.CC_STAT_WIDTH]
		ch = stats[label_idx, cv2.CC_STAT_HEIGHT]
		area = stats[label_idx, cv2.CC_STAT_AREA]
		is_text_like = area <= max_area and cw <= max_width and ch <= max_height and x > 0 and y > 0
		if not is_text_like:
			clean[labels == label_idx] = 255
	return clean


def _remove_vertical_artifacts(mask: np.ndarray, config: DigitizerConfig) -> np.ndarray:
	h, w = mask.shape
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	clean = np.zeros_like(mask)
	min_height = max(8, int(round(h * config.vertical_component_min_height_ratio)))
	max_width = max(2, int(round(w * config.vertical_component_max_width_ratio)))
	for label_idx in range(1, num_labels):
		cw = stats[label_idx, cv2.CC_STAT_WIDTH]
		ch = stats[label_idx, cv2.CC_STAT_HEIGHT]
		is_vertical_artifact = ch >= min_height and cw <= max_width
		if not is_vertical_artifact:
			clean[labels == label_idx] = 255
	return clean


def _fallback_signal_mask_from_block(block_image: np.ndarray, config: DigitizerConfig) -> np.ndarray:
	if block_image is None or block_image.size == 0:
		return np.zeros((0, 0), dtype=np.uint8)
	if block_image.ndim == 3:
		gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
	else:
		gray = block_image
	if gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
		return np.zeros((0, 0), dtype=np.uint8)
	if gray.dtype != np.uint8:
		gray = np.clip(gray, 0, 255).astype(np.uint8)
	gray = np.ascontiguousarray(gray)
	h, w = gray.shape[:2]
	if h < 2 or w < 2:
		return np.zeros((h, w), dtype=np.uint8)
	kx = max(1, min(5, w if w % 2 == 1 else w - 1))
	ky = max(1, min(5, h if h % 2 == 1 else h - 1))
	if kx >= 3 and ky >= 3:
		try:
			gray = cv2.GaussianBlur(gray, (kx, ky), 0)
		except cv2.error:
			pass
	mask = cv2.adaptiveThreshold(
		gray,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		_ensure_odd(config.adaptive_block_size),
		config.adaptive_c,
	)
	grid_like = _extract_line_grid(mask)
	mask = cv2.bitwise_and(mask, cv2.bitwise_not(grid_like))
	mask = _remove_small_components(mask, min_area=max(8, config.min_component_area // 2))
	mask = _remove_text_components(mask, config)
	mask = _remove_vertical_artifacts(mask, config)
	return mask


def crop_signal_region(
	image: np.ndarray,
	signal_mask: np.ndarray,
	config: DigitizerConfig = DigitizerConfig(),
	reference_mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
	# Use union of signal and reference (grid) for bounding box to preserve context
	if reference_mask is not None and reference_mask.shape == signal_mask.shape:
		combined = cv2.bitwise_or(signal_mask, reference_mask)
	else:
		combined = signal_mask

	points = cv2.findNonZero(combined)
	if points is None:
		return image, signal_mask, (0, 0, image.shape[1], image.shape[0])
	
	x, y, w, h = cv2.boundingRect(points)
	pad = config.bbox_padding
	
	# Horizontal cropping logic: Align to Grid (Reference) if possible
	x0, x1 = 0, image.shape[1]
	
	if reference_mask is not None:
		ref_pts = cv2.findNonZero(reference_mask)
		if ref_pts is not None:
			rx, ry, rw, rh = cv2.boundingRect(ref_pts)
			# Grid check: Should cover substantial width to be valid time axis
			if rw > image.shape[1] * 0.5:
				x0 = rx
				x1 = rx + rw

	# Vertical cropping logic: Align to Content (Combined) with padding
	y0 = max(0, y - pad)
	y1 = min(image.shape[0], y + h + pad)
	
	cropped_image = image[y0:y1, x0:x1]
	cropped_mask = signal_mask[y0:y1, x0:x1]

	row_profile = cropped_mask.sum(axis=1).astype(np.float32) / 255.0
	bands = _find_active_bands(row_profile, min_size=max(8, cropped_mask.shape[0] // 40))
	if len(bands) >= config.grid_rows:
		selected = bands[: config.grid_rows]
		top = max(0, selected[0][0] - pad)
		bottom = min(cropped_image.shape[0], selected[-1][1] + pad)
		
		cropped_image = cropped_image[top:bottom, :]
		cropped_mask = cropped_mask[top:bottom, :]
		
		y0 = y0 + top
		y1 = y0 + cropped_image.shape[0]
		
		return cropped_image, cropped_mask, (x0, y0, x1, y1)
		
	return cropped_image, cropped_mask, (x0, y0, x1, y1)


def split_into_leads(image: np.ndarray, signal_mask: np.ndarray, config: DigitizerConfig = DigitizerConfig()) -> list[tuple[str, np.ndarray, np.ndarray]]:
	blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
	row_bands = _find_row_bands(signal_mask, config)
	for row_idx, (y0, y1_inclusive) in enumerate(row_bands):
		row_image = image[y0 : y1_inclusive + 1, :]
		row_mask = signal_mask[y0 : y1_inclusive + 1, :]
		col_bands = _find_column_bands(row_mask, config)
		for col_idx, (x0, x1_inclusive) in enumerate(col_bands):
			lead_name = LEAD_LAYOUT[row_idx][col_idx]
			blocks.append(
				(
					lead_name,
					row_image[:, x0 : x1_inclusive + 1],
					row_mask[:, x0 : x1_inclusive + 1],
				)
			)
	return blocks


def _trim_lead_block(
	lead_image: np.ndarray,
	lead_mask: np.ndarray,
	row_idx: int,
	config: DigitizerConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
	h, w = lead_mask.shape
	left = min(w - 1, max(0, int(round(w * config.lead_left_trim_ratio))))
	right = max(left + 1, min(w, w - int(round(w * config.lead_right_trim_ratio))))
	top = min(h - 1, max(0, int(round(h * config.lead_top_trim_ratio))))
	bottom_trim = config.lead_bottom_trim_ratio + (config.bottom_row_extra_trim_ratio if row_idx == config.grid_rows - 1 else 0.0)
	bottom = max(top + 1, min(h, h - int(round(h * bottom_trim))))
	return lead_image[top:bottom, left:right], lead_mask[top:bottom, left:right], (left, top, right, bottom)


def _overlay_waveform(lead_image: np.ndarray, y_values: np.ndarray, analysis_box: tuple[int, int, int, int]) -> np.ndarray:
	overlay = lead_image.copy()
	left, top, right, _ = analysis_box
	if overlay.ndim == 2:
		overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
	for x_idx, y_value in enumerate(y_values):
		if np.isnan(y_value):
			continue
		x = left + x_idx
		y = top + int(round(float(y_value)))
		if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
			cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)
	cv2.rectangle(overlay, (left, top), (right - 1, analysis_box[3] - 1), (0, 255, 0), 1)
	return overlay


def _projection_spacing(projection: np.ndarray) -> float:
	if projection.size == 0 or projection.max() <= 0:
		return 0.0
	window = max(5, (projection.size // 120) | 1)
	kernel = np.ones(window, dtype=np.float32) / window
	smooth = np.convolve(projection.astype(np.float32), kernel, mode="same")
	threshold = max(1.0, smooth.max() * 0.45)
	peaks = []
	for idx in range(1, len(smooth) - 1):
		if smooth[idx] >= threshold and smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]:
			peaks.append(idx)
	if len(peaks) < 2:
		return 0.0
	diffs = np.diff(peaks)
	diffs = diffs[(diffs >= 4) & (diffs <= 80)]
	if diffs.size == 0:
		return 0.0
	return float(np.median(diffs))


def estimate_pixels_per_mm(grid_mask: np.ndarray, roi_width: int) -> float:
	# Robust heuristic: The standard ECG 12-lead image width corresponds to 10 seconds.
	# At 25 mm/s paper speed, 10s = 250 mm.
	# Ideally, pixels_per_mm = width_pixels / 250.0
	width_based_ppm = roi_width / 250.0

	# Ensure we have enough grid points to trust the local grid estimation
	# For "Hard" images (photocopies, crumpled), local grid is unreliable.
	# We require a decent density of grid signal.
	total_pixels = grid_mask.size
	if total_pixels == 0:
		return width_based_ppm
	
	grid_density = cv2.countNonZero(grid_mask) / total_pixels
	
	# If grid is very sparse (likely just noise or bad detection), trust the document width
	if grid_density < 0.005:
		return width_based_ppm

	x_spacing = _projection_spacing(grid_mask.sum(axis=0) / 255.0)
	y_spacing = _projection_spacing(grid_mask.sum(axis=1) / 255.0)
	
	spacings = [value for value in [x_spacing, y_spacing] if value > 0]
	if spacings:
		local_ppm = float(np.median(spacings))
		# Sanity check: local_ppm should be within reasonable bounds of width_based (e.g. +/- 20%)
		# Smartphone photos might scale things, but extreme deviations imply detection error
		if 0.8 * width_based_ppm < local_ppm < 1.2 * width_based_ppm:
			return local_ppm

	return width_based_ppm


def _fast_viterbi_trace(mask: np.ndarray, config: DigitizerConfig) -> np.ndarray:
	"""
	Traces the ECG signal using a Viterbi-like dynamic programming algorithm
	to find the path of maximum likelihood (pixel intensity) that satisfies
	continuity constraints (minimized vertical jumps).
	"""
	h, w = mask.shape
	# Energy: +1.0 for likely signal (value=255), -1.0 for background
	# We normalize to 0..1 then map
	energy = (mask.astype(np.float32) / 255.0) * 2.0 - 0.5
	
	dp = np.zeros_like(energy)
	dp[:, 0] = energy[:, 0]
	backtrack = np.zeros((h, w), dtype=np.int32)
	
	# Pre-allocate indices
	rows = np.arange(h)
	
	# Max jump to check (vectorized optimization)
	# Typically ECG doesn't jump more than a few pixels per column unless QRS is very steep
	# We allow larger jumps but penalize them
	jump_limit = 5
	jump_penalty_base = 0.05
	
	for x in range(1, w):
		prev = dp[:, x-1]
		
		# Start with 0 jump (stay same row)
		best_val = prev.copy()
		best_src = rows.copy()
		
		# Vectorized check of neighbors [-jump_limit ... +jump_limit]
		for jump in range(1, jump_limit + 1):
			penalty = jump * jump_penalty_base
			
			# Down shift: current y comes from y-jump (prev[y-jump])
			# roll is not correct for boundary, use slicing
			val_down = np.full(h, -1e9, dtype=np.float32)
			val_down[jump:] = prev[:-jump] - penalty
			
			mask_down = val_down > best_val
			best_val[mask_down] = val_down[mask_down]
			best_src[mask_down] = rows[mask_down] - jump
			
			# Up shift: current y comes from y+jump (prev[y+jump])
			val_up = np.full(h, -1e9, dtype=np.float32)
			val_up[:-jump] = prev[jump:] - penalty
			
			mask_up = val_up > best_val
			best_val[mask_up] = val_up[mask_up]
			best_src[mask_up] = rows[mask_up] + jump
			
		dp[:, x] = best_val + energy[:, x]
		backtrack[:, x] = best_src

	# Backtrack finding max path
	path = np.zeros(w, dtype=np.float32)
	curr = np.argmax(dp[:, -1])
	path[-1] = float(curr)
	for x in range(w - 2, -1, -1):
		curr = backtrack[curr, x+1]
		path[x] = float(curr)
		
	return path


def extract_waveform(
	block_mask: np.ndarray,
	pixels_per_mm: float,
	config: DigitizerConfig = DigitizerConfig(),
) -> tuple[np.ndarray, np.ndarray]:
	h, w = block_mask.shape
	if h == 0 or w == 0:
		return np.zeros(config.samples_per_lead, dtype=np.float32), np.zeros(0, dtype=np.float32)

	# Use SOTA Viterbi tracing instead of greedy column search
	y_values = _fast_viterbi_trace(block_mask, config)

	# Optional: Refine y_values where signal is completely missing (mask is all 0)
	# The Viterbi might stick to one edge or drift if energy is all -0.5.
	# We can mask out parts where the underlying pixel value is 0 on the path?
	# For now, we trust the DP to find the "least bad" path through noise or silence.

	if w >= 3:
		win = max(3, _ensure_odd(config.trace_smooth_window))
		if win > w:
			win = w if w % 2 == 1 else w - 1
		if win >= 3:
			kernel = np.ones(win, dtype=np.float32) / float(win)
			y_values = np.convolve(y_values, kernel, mode="same").astype(np.float32)

	baseline = float(np.median(y_values))
	wave_pixels = -(y_values - baseline)
	wave_millivolts = wave_pixels / max(pixels_per_mm * 10.0, 1e-6)
	wave_millivolts = np.clip(
		wave_millivolts,
		-config.trace_clip_millivolts,
		config.trace_clip_millivolts,
	)

	x_old = np.linspace(0.0, config.lead_duration_sec, num=w, endpoint=False)
	x_new = np.linspace(0.0, config.lead_duration_sec, num=config.samples_per_lead, endpoint=False)
	resampler = interp1d(x_old, wave_millivolts, kind="linear", bounds_error=False, fill_value="extrapolate")
	return resampler(x_new).astype(np.float32), y_values


def digitize_ecg_image(image: np.ndarray, config: DigitizerConfig = DigitizerConfig()) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	if image is None:
		raise ValueError("Input image is None")

	signal_mask, grid_mask, debug = build_signal_mask(image, config)
	roi_image, roi_mask, (x0, y0, x1, y1) = crop_signal_region(image, signal_mask, config, reference_mask=grid_mask)
	roi_grid = grid_mask[y0:y1, x0:x1]
	pixels_per_mm = estimate_pixels_per_mm(roi_grid, roi_width=roi_image.shape[1])
	lead_blocks = split_into_leads(roi_image, roi_mask, config)

	signals: dict[str, np.ndarray] = {}
	lead_debug: dict[str, dict[str, np.ndarray]] = {}
	for idx, (lead_name, block_image, block_mask) in enumerate(lead_blocks):
		row_idx = idx // config.grid_cols
		analysis_image, analysis_mask, analysis_box = _trim_lead_block(block_image, block_mask, row_idx, config)
		if analysis_mask.size == 0:
			signals[lead_name] = np.zeros(config.samples_per_lead, dtype=np.float32)
			lead_debug[lead_name] = {
				"block_image": block_image,
				"block_mask": block_mask,
				"analysis_image": analysis_image,
				"analysis_mask": analysis_mask,
				"overlay": block_image,
			}
			continue
		analysis_mask = _remove_text_components(analysis_mask, config)
		analysis_mask = _remove_vertical_artifacts(analysis_mask, config)
		h, w = analysis_mask.shape
		min_foreground = int(round(h * w * config.lead_min_foreground_ratio))
		if cv2.countNonZero(analysis_mask) < max(4, min_foreground):
			analysis_mask = _fallback_signal_mask_from_block(analysis_image, config)
		signal, y_values = extract_waveform(analysis_mask, pixels_per_mm, config)
		signals[lead_name] = signal
		lead_debug[lead_name] = {
			"block_image": block_image,
			"block_mask": block_mask,
			"analysis_image": analysis_image,
			"analysis_mask": analysis_mask,
			"overlay": _overlay_waveform(block_image, y_values, analysis_box),
		}

	debug["signal_mask"] = signal_mask
	debug["roi_image"] = roi_image
	debug["roi_mask"] = roi_mask
	debug["pixels_per_mm"] = np.array([pixels_per_mm], dtype=np.float32)
	debug["lead_debug"] = lead_debug
	return signals, debug


def digitize_ecg_file(image_path: str | Path, config: DigitizerConfig = DigitizerConfig()) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Failed to read image: {image_path}")
	return digitize_ecg_image(image, config)
import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
	pts = pts.reshape(4, 2).astype(np.float32)
	s = pts.sum(axis=1)
	diff = np.diff(pts, axis=1).reshape(-1)
	ordered = np.zeros((4, 2), dtype=np.float32)
	ordered[0] = pts[np.argmin(s)]
	ordered[2] = pts[np.argmax(s)]
	ordered[1] = pts[np.argmin(diff)]
	ordered[3] = pts[np.argmax(diff)]
	return ordered


def _four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
	rect = _order_points(pts)
	(tl, tr, br, bl) = rect
	width_a = np.linalg.norm(br - bl)
	width_b = np.linalg.norm(tr - tl)
	height_a = np.linalg.norm(tr - br)
	height_b = np.linalg.norm(tl - bl)
	max_width = int(max(width_a, width_b))
	max_height = int(max(height_a, height_b))
	dst = np.array(
		[[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
		dtype=np.float32,
	)
	matrix = cv2.getPerspectiveTransform(rect, dst)
	return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _detect_document(image: np.ndarray) -> np.ndarray | None:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blur, 50, 150)
	edges = cv2.dilate(edges, None, iterations=1)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	for cnt in contours[:5]:
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		if len(approx) == 4:
			return approx
	return None


def _estimate_skew_angle(image: np.ndarray, mask: np.ndarray | None = None) -> float:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if mask is not None:
		gray = cv2.bitwise_and(gray, gray, mask=mask)
	edges = cv2.Canny(gray, 50, 150)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
	if lines is None:
		return 0.0
	angles = []
	for rho_theta in lines[:100]:
		_, theta = rho_theta[0]
		angle = (theta * 180 / np.pi) - 90
		if -20 <= angle <= 20:
			angles.append(angle)
	if not angles:
		return 0.0
	return float(np.median(angles))


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
	if abs(angle) < 0.01:
		return image
	h, w = image.shape[:2]
	center = (w // 2, h // 2)
	matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def _estimate_white_lab(image: np.ndarray) -> np.ndarray:
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l = lab[:, :, 0].astype(np.float32)
	thresh = np.percentile(l, 95)
	mask = l >= thresh
	if mask.sum() == 0:
		return np.array([255, 128, 128], dtype=np.float32)
	return lab[mask].mean(axis=0)


def _separate_grid_signal(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
	white_lab = _estimate_white_lab(image)
	delta = np.linalg.norm(lab - white_lab, axis=2)
	l_chan = lab[:, :, 0]

	ink_thresh = np.percentile(delta, 80)
	ink_mask = delta > ink_thresh

	dark_thresh = np.percentile(l_chan, 35)
	signal_mask = (l_chan < dark_thresh) & ink_mask

	color_delta = np.linalg.norm(lab[:, :, 1:] - white_lab[1:], axis=2)
	color_thresh = np.percentile(color_delta, 70)
	grid_mask = (color_delta > color_thresh) & ~signal_mask

	grid_mask = grid_mask.astype(np.uint8) * 255
	signal_mask = signal_mask.astype(np.uint8) * 255

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel, iterations=1)
	signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
	return grid_mask, signal_mask


def _crop_to_content(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return image
	x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
	return image[y : y + h, x : x + w]


def clean_single_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	if image is None:
		raise ValueError("Input image is None")

	debug: dict[str, np.ndarray] = {}
	orig = image.copy()

	doc = _detect_document(image)
	if doc is not None:
		image = _four_point_warp(image, doc)

	grid_mask, signal_mask = _separate_grid_signal(image)
	angle = _estimate_skew_angle(image, mask=grid_mask)
	image = _rotate(image, angle)

	grid_mask, signal_mask = _separate_grid_signal(image)
	ink_mask = cv2.bitwise_or(grid_mask, signal_mask)
	ink_mask = cv2.dilate(ink_mask, None, iterations=1)

	cropped = _crop_to_content(image, ink_mask)

	cleaned = np.full_like(image, 255)
	cleaned[signal_mask > 0] = (0, 0, 0)

	debug["original"] = orig
	debug["warped"] = image
	debug["grid_mask"] = grid_mask
	debug["signal_mask"] = signal_mask
	debug["ink_mask"] = ink_mask

	return cleaned, cropped, grid_mask, debug

"""Image cropping utilities for generating ordered patches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2

from app.config_loader import LayoutConfig


@dataclass
class CropResult:
    index: int
    image: np.ndarray
    bbox: tuple[int, int, int, int]


class CircleDetectionError(ValueError):
    """Raised when circle detection does not match the expected layout."""

    def __init__(self, message: str, detected: int, expected: int) -> None:
        super().__init__(message)
        self.detected = detected
        self.expected = expected


class CircleCropper:
    """Cropper that detects circular parts via watershed splitting and crops around them.

    The detection runs as grayscale, blur, erode, threshold, dilate, then
    watershed-based splitting and circle fitting. Circles are expanded by
    `radius_expand` and cropped into rectangular patches with masked
    background outside the circle.
    """

    def __init__(self, layout: LayoutConfig) -> None:
        self.layout = layout

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        k = max(1, int(self.layout.circle_blur_kernel) // 2 * 2 + 1)
        gray = cv2.GaussianBlur(gray, (k, k), 0)
        if self.layout.circle_erode_iter > 0:
            ker = np.ones((9, 9), np.uint8)
            gray = cv2.erode(gray, ker, iterations=int(self.layout.circle_erode_iter))
        _, thresh = cv2.threshold(gray, int(self.layout.circle_threshold), 255, 0)
        if self.layout.circle_dilate_iter > 0:
            kernel1 = np.ones((4, 4), np.uint8)
            thresh = cv2.dilate(thresh, kernel1, iterations=int(self.layout.circle_dilate_iter))
        return thresh

    def _resize_binary(self, binary_255: np.ndarray, scale: float) -> np.ndarray:
        h, w = binary_255.shape[:2]
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        return cv2.resize(binary_255, (nw, nh), interpolation=cv2.INTER_NEAREST)

    def _split_by_dist_watershed(self, binary_255: np.ndarray) -> np.ndarray:
        bin_img = binary_255
        if bin_img.dtype != np.uint8:
            bin_img = bin_img.astype(np.uint8)
        if bin_img.max() == 1:
            bin_img = (bin_img * 255).astype(np.uint8)

        if self.layout.ws_open_ksize > 1 and self.layout.ws_open_iter > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.layout.ws_open_ksize, self.layout.ws_open_ksize)
            )
            bin_clean = cv2.morphologyEx(
                bin_img, cv2.MORPH_OPEN, k, iterations=int(self.layout.ws_open_iter)
            )
        else:
            bin_clean = bin_img

        dist = cv2.distanceTransform(bin_clean, cv2.DIST_L2, int(self.layout.ws_dist_mask_size))
        dist_max = float(dist.max()) if dist.size else 0.0
        if dist_max <= 1e-6:
            return np.zeros_like(bin_clean, dtype=np.int32)

        _, sure_fg = cv2.threshold(dist, float(self.layout.ws_dist_ratio) * dist_max, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(bin_clean, kbg, iterations=int(self.layout.ws_bg_dilate_iter))
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown > 0] = 0

        img3 = cv2.cvtColor(bin_clean, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img3, markers)
        return markers

    @staticmethod
    def _fit_circle_lsq(points_xy: np.ndarray) -> tuple[float, float, float]:
        pts = points_xy.astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = sol
        cx = -D / 2.0
        cy = -E / 2.0
        r2 = cx * cx + cy * cy - F
        r = np.sqrt(max(r2, 0.0))
        return float(cx), float(cy), float(r)

    def _fit_outer_from_mask(self, mask_255: np.ndarray) -> tuple[float, float, float] | None:
        cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c_out = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c_out) < 10:
            return None

        pts = c_out.reshape(-1, 2)
        if pts.shape[0] > self.layout.ws_max_contour_pts:
            step = int(np.ceil(pts.shape[0] / self.layout.ws_max_contour_pts))
            pts = pts[::step]
        return self._fit_circle_lsq(pts)

    @staticmethod
    def _build_outer_disk_mask(shape_hw: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
        h, w = shape_hw
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
        return m

    @staticmethod
    def _crop_by_mask(
        original_bgr: np.ndarray, mask_255: np.ndarray, pad: int
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        ys, xs = np.where(mask_255 > 0)
        if len(xs) == 0:
            return None, None

        h, w = mask_255.shape
        x1 = max(int(xs.min()) - pad, 0)
        x2 = min(int(xs.max()) + pad + 1, w)
        y1 = max(int(ys.min()) - pad, 0)
        y2 = min(int(ys.max()) + pad + 1, h)

        roi = original_bgr[y1:y2, x1:x2].copy()
        mroi = mask_255[y1:y2, x1:x2]
        if roi.ndim == 2:
            roi[mroi == 0] = 0
        else:
            for c in range(roi.shape[2]):
                channel = roi[:, :, c]
                channel[mroi == 0] = 0
                roi[:, :, c] = channel
        return roi, (x1, y1, x2, y2)

    @staticmethod
    def _sort_row_major(circles: np.ndarray, rows: int, cols: int) -> np.ndarray:
        # Sort by y then x (row-major)
        arr = circles.reshape(-1, 3)
        if rows > 0 and cols > 0 and len(arr) == rows * cols:
            y_sorted = arr[np.argsort(arr[:, 1])]
            grouped: list[np.ndarray] = []
            for row_idx in range(rows):
                start = row_idx * cols
                end = start + cols
                row = y_sorted[start:end]
                row = row[np.argsort(row[:, 0])]
                grouped.append(row)
            return np.vstack(grouped)
        order = np.lexsort((arr[:, 0], arr[:, 1]))
        return arr[order]

    def crop_with_count(self, image: np.ndarray) -> tuple[List[CropResult], int]:
        """Return cropped patches and the number of detected circles.

        Unlike `crop`, this method does not enforce the expected circle count.
        """
        if image.ndim not in (2, 3):
            raise ValueError("Unsupported image format")

        h, w = image.shape[:2]
        bin_img = self._preprocess(image)

        ws_scale = float(self.layout.ws_scale)
        if 0.0 < ws_scale < 1.0:
            bin_small = self._resize_binary(bin_img, ws_scale)
            inv_scale = 1.0 / ws_scale
        else:
            bin_small = bin_img
            inv_scale = 1.0

        markers = self._split_by_dist_watershed(bin_small)
        labels = np.unique(markers)
        labels = labels[labels > 1]

        circles: List[tuple[float, float, float]] = []
        for lb in labels:
            obj_mask_s = (markers == lb).astype(np.uint8) * 255
            area_s = int(cv2.countNonZero(obj_mask_s))
            area_est = area_s * (inv_scale * inv_scale) if inv_scale != 1.0 else area_s
            if area_est < self.layout.ws_min_area_px:
                continue

            fit = self._fit_outer_from_mask(obj_mask_s)
            if fit is None:
                continue
            cx_s, cy_s, r_s = fit

            cx = cx_s * inv_scale
            cy = cy_s * inv_scale
            r = r_s * inv_scale

            if not (self.layout.circle_min_radius <= r <= self.layout.circle_max_radius):
                continue
            circles.append((cx, cy, r))

        if not circles:
            return [], 0

        arr = self._sort_row_major(np.array(circles, dtype=np.float32), self.layout.rows, self.layout.cols)
        detected = len(arr)

        patches: List[CropResult] = []
        radius_expand = float(self.layout.circle_radius_expand)
        pad = int(self.layout.patch_padding)
        for idx, (cx, cy, r) in enumerate(arr, start=1):
            radius = r + radius_expand
            mask = self._build_outer_disk_mask((h, w), cx, cy, radius)
            cropped_roi, bbox = self._crop_by_mask(image, mask, pad)
            if cropped_roi is None or bbox is None:
                continue
            patches.append(CropResult(index=idx, image=cropped_roi, bbox=bbox))

        return patches, detected

    def crop(self, image: np.ndarray) -> List[CropResult]:
        if image.ndim not in (2, 3):
            raise ValueError("Unsupported image format")

        expected = self.layout.count
        patches, detected = self.crop_with_count(image)
        if detected == 0:
            raise CircleDetectionError("No circles detected for cropping", detected=0, expected=expected)
        if detected != expected:
            raise CircleDetectionError(
                f"Detected {detected} circles, expected {expected}",
                detected=detected,
                expected=expected,
            )
        return patches


__all__ = ["CircleCropper", "CropResult", "CircleDetectionError"]

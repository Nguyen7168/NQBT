"""Image cropping utilities for generating ordered patches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
    """Cropper that detects circular parts via HoughCircles and crops around them.

    The detection runs roughly as in crop_hc_23_240.py: grayscale, blur,
    erode, threshold, dilate, then HoughCircles. Circles are expanded by
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

    def _detect_circles(self, bin_img: np.ndarray) -> np.ndarray | None:
        circles = cv2.HoughCircles(
            bin_img,
            cv2.HOUGH_GRADIENT,
            dp=float(self.layout.circle_dp),
            minDist=float(self.layout.circle_minDist),
            param1=float(self.layout.circle_param1),
            param2=float(self.layout.circle_param2),
            minRadius=int(self.layout.circle_min_radius),
            maxRadius=int(self.layout.circle_max_radius),
        )
        return circles

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
        expected = self.layout.count
        bin_img = self._preprocess(image)
        circles = self._detect_circles(bin_img)
        if circles is None or len(circles[0]) == 0:
            return [], 0

        circles = np.uint16(np.around(circles))
        arr = self._sort_row_major(circles[0], self.layout.rows, self.layout.cols)
        detected = len(arr)

        patches: List[CropResult] = []
        radius_expand = int(self.layout.circle_radius_expand)
        for idx, (cx, cy, r) in enumerate(arr, start=1):
            radius = int(r) + radius_expand
            x1 = max(int(cx) - radius, 0)
            x2 = min(int(cx) + radius, w)
            y1 = max(int(cy) - radius, 0)
            y2 = min(int(cy) + radius, h)

            # Create circular mask and apply
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
            roi_mask = mask[y1:y2, x1:x2]
            cropped_roi = image[y1:y2, x1:x2].copy()
            if cropped_roi.ndim == 2:
                cropped_roi[roi_mask == 0] = 0
            else:
                for c in range(cropped_roi.shape[2]):
                    channel = cropped_roi[:, :, c]
                    channel[roi_mask == 0] = 0
                    cropped_roi[:, :, c] = channel

            patches.append(CropResult(index=idx, image=cropped_roi, bbox=(x1, y1, x2, y2)))

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

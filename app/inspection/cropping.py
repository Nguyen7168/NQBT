"""Image cropping utilities for generating ordered patches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from app.config_loader import LayoutConfig


@dataclass
class CropResult:
    index: int
    image: np.ndarray
    bbox: tuple[int, int, int, int]


class GridCropper:
    """Cropper that splits the input image into a regular grid."""

    def __init__(self, layout: LayoutConfig) -> None:
        if layout.rows * layout.cols != layout.count:
            raise ValueError(
                "Layout rows*cols must match count (got "
                f"{layout.rows}x{layout.cols} for {layout.count})"
            )
        self.layout = layout

    def crop(self, image: np.ndarray) -> List[CropResult]:
        if image.ndim != 3:
            raise ValueError("Expected color image with shape (H, W, C)")
        height, width, _ = image.shape
        cell_h = height // self.layout.rows
        cell_w = width // self.layout.cols
        padding = self.layout.patch_padding

        patches: List[CropResult] = []
        idx = 0
        for row in range(self.layout.rows):
            for col in range(self.layout.cols):
                idx += 1
                y1 = max(row * cell_h - padding, 0)
                y2 = min((row + 1) * cell_h + padding, height)
                x1 = max(col * cell_w - padding, 0)
                x2 = min((col + 1) * cell_w + padding, width)
                patch = image[y1:y2, x1:x2]
                patches.append(CropResult(index=idx, image=patch, bbox=(x1, y1, x2, y2)))
        return patches


__all__ = ["GridCropper", "CropResult"]

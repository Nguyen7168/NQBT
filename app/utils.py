"""Miscellaneous utility helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PyQt5.QtGui import QImage

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return qimage.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.copy()
    raise ValueError("Unsupported image format for conversion to QImage")


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image to disk.
    Assumes image is in OpenCV BGR or grayscale format (as used across the app).
    Writes directly without color conversion to avoid unintended shifts.
    """
    ensure_dir(Path(path).parent)
    cv2.imwrite(str(path), image)


__all__ = ["ensure_dir", "numpy_to_qimage", "save_image"]

#!/usr/bin/env python3
"""
GUI to inspect the numbered concentric circles on the mirror target.

Pipeline:
  Original -> BrightDiskMask -> TargetROI -> GreenScore -> Profile -> Overlay

The detector first finds the large bright disk, crops a centered ROI around the
10 numbered circles, builds a radial profile from the green print, then selects
the requested circle index from the detected peaks.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

STAGES: List[str] = [
    "Original",
    "BrightDiskMask",
    "TargetROI",
    "GreenScore",
    "Profile",
    "Overlay",
]


def _odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return (k // 2) * 2 + 1


def cv_to_qpixmap(img: Optional[np.ndarray]) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


@dataclass
class BrightDiskResult:
    center: Tuple[float, float]
    radius: float
    contour_area: float


@dataclass
class RingDetectionResult:
    selected_ring: int
    center: Tuple[float, float]
    radius: float
    bright_disk: BrightDiskResult
    ring_radii: List[float]
    roi_rect: Tuple[int, int, int, int]
    score_peaks: List[Tuple[float, float]]


def detect_bright_disk(
    gray: np.ndarray,
    bright_thresh: int = 160,
    min_contour_area: float = 1_000_000.0,
) -> Tuple[BrightDiskResult, np.ndarray]:
    _, mask = cv2.threshold(gray, int(bright_thresh), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Khong tim thay dia sang lon. Hay giam Bright threshold.")

    filtered = [c for c in contours if cv2.contourArea(c) >= float(min_contour_area)]
    if not filtered:
        filtered = contours

    cnt = max(filtered, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    (cx, cy), r = cv2.minEnclosingCircle(cnt)

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(dbg, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 3)
    cv2.circle(dbg, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)
    cv2.putText(
        dbg,
        f"bright_r={r:.1f}px | area={area:.0f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return BrightDiskResult(center=(float(cx), float(cy)), radius=float(r), contour_area=area), dbg


def crop_roi_around_center(
    img: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    crop_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cx, cy = center
    half = max(120, int(round(radius * float(crop_ratio))))
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(img.shape[1], int(round(cx)) + half)
    y2 = min(img.shape[0], int(round(cy)) + half)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("ROI khong hop le. Hay tang crop ratio.")
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def make_green_score_image(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b, g, r = cv2.split(roi_bgr)
    score = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
    score = cv2.GaussianBlur(score, (0, 0), 1.0)
    score_vis = np.clip((score - score.min()) * 255.0 / max(1.0, score.max() - score.min()), 0, 255).astype(np.uint8)
    score_vis = cv2.applyColorMap(score_vis, cv2.COLORMAP_VIRIDIS)
    return score, score_vis


def radial_profile_from_score(
    score: np.ndarray,
    center: Tuple[float, float],
    min_radius: int,
    max_radius: int,
    band_half_width: int,
) -> np.ndarray:
    yy, xx = np.indices(score.shape[:2])
    dist_idx = np.rint(np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)).astype(np.int32)
    max_bin = max_radius + band_half_width + 2
    weighted = np.bincount(dist_idx.ravel(), weights=score.ravel(), minlength=max_bin + 1)
    counts = np.bincount(dist_idx.ravel(), minlength=max_bin + 1).astype(np.float32)
    base_profile = weighted[: max_bin + 1] / np.maximum(1.0, counts[: max_bin + 1])

    kernel_size = _odd(2 * band_half_width + 1)
    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    smoothed = np.convolve(base_profile, kernel, mode="same")
    return np.asarray(smoothed[min_radius : max_radius + 1], dtype=np.float32)


def find_profile_peaks(
    profile: np.ndarray,
    radius_offset: int,
    min_peak_value: float,
    min_peak_distance: int,
) -> List[Tuple[float, float]]:
    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(profile) - 1):
        if profile[i] < float(min_peak_value):
            continue
        if not (profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]):
            continue
        rad = float(i + radius_offset)
        val = float(profile[i])
        if peaks and (rad - peaks[-1][0]) < float(min_peak_distance):
            if val > peaks[-1][1]:
                peaks[-1] = (rad, val)
        else:
            peaks.append((rad, val))
    return peaks


def refine_center_and_radius(
    score: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    search_radius: int,
    radius_search: int,
    band_half_width: int,
) -> Tuple[Tuple[float, float], float]:
    if search_radius <= 0:
        return center, radius

    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    best_value = -1e18
    best_center = center
    best_radius = radius
    cx0, cy0 = center
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            cx = cx0 + dx
            cy = cy0 + dy
            for dr in range(-radius_search, radius_search + 1):
                rad = float(radius + dr)
                if rad <= 1:
                    continue
                samples: List[np.ndarray] = []
                for band in range(-band_half_width, band_half_width + 1):
                    rr = rad + float(band)
                    xs = np.clip(np.rint(cx + rr * cos_a).astype(np.int32), 0, score.shape[1] - 1)
                    ys = np.clip(np.rint(cy + rr * sin_a).astype(np.int32), 0, score.shape[0] - 1)
                    samples.append(score[ys, xs])
                value = float(np.mean(samples))
                if value > best_value:
                    best_value = value
                    best_center = (float(cx), float(cy))
                    best_radius = rad
    return best_center, best_radius


def build_profile_view(
    profile: np.ndarray,
    peaks: Sequence[Tuple[float, float]],
    selected_ring: int,
    radius_offset: int,
    selected_radius: float,
) -> np.ndarray:
    h, w = 420, 980
    canvas = np.full((h, w, 3), 250, dtype=np.uint8)
    if profile.size == 0:
        return canvas

    left, right, top, bottom = 70, 30, 30, 55
    plot_w = w - left - right
    plot_h = h - top - bottom
    p_min = float(np.min(profile))
    p_max = float(np.max(profile))
    scale = max(1e-6, p_max - p_min)

    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (70, 70, 70), 1)

    pts = []
    for idx, val in enumerate(profile):
        x = left + int(round(idx * (plot_w - 1) / max(1, profile.size - 1)))
        y = top + plot_h - int(round((val - p_min) * (plot_h - 1) / scale))
        pts.append([x, y])
    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (30, 120, 220), 2, cv2.LINE_AA)

    for idx, (rad, val) in enumerate(peaks, start=1):
        x_idx = int(round(rad - radius_offset))
        x = left + int(round(x_idx * (plot_w - 1) / max(1, profile.size - 1)))
        y = top + plot_h - int(round((val - p_min) * (plot_h - 1) / scale))
        color = (0, 0, 255) if idx == selected_ring else (0, 150, 0)
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    label = f"Selected ring {selected_ring}: r={selected_radius:.1f}px"
    cv2.putText(canvas, label, (left, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def draw_overlay(
    original: np.ndarray,
    result: RingDetectionResult,
) -> np.ndarray:
    out = original.copy()
    cx, cy = result.center
    x1, y1, x2, y2 = result.roi_rect

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 170, 0), 2)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(result.radius)), (0, 0, 255), 3)

    for idx, ring_radius in enumerate(result.ring_radii, start=1):
        color = (0, 0, 255) if idx == result.selected_ring else (0, 200, 0)
        thickness = 3 if idx == result.selected_ring else 1
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(ring_radius)), color, thickness)
        cv2.putText(
            out,
            str(idx),
            (int(round(cx + ring_radius + 8)), int(round(cy))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"Ring {result.selected_ring}: r={result.radius:.1f}px | bright_r={result.bright_disk.radius:.1f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Ring {result.selected_ring}: r={result.radius:.1f}px | bright_r={result.bright_disk.radius:.1f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def detect_numbered_ring(
    original: np.ndarray,
    target_ring: int,
    bright_thresh: int,
    crop_ratio: float,
    band_half_width: int,
    smooth_kernel: int,
    min_peak_value: float,
    min_peak_distance: int,
    max_ring_ratio: float,
    center_refine: int,
) -> Tuple[RingDetectionResult, Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    bright_disk, bright_dbg = detect_bright_disk(gray, bright_thresh=bright_thresh)

    roi_bgr, roi_rect = crop_roi_around_center(
        original,
        center=bright_disk.center,
        radius=bright_disk.radius,
        crop_ratio=crop_ratio,
    )
    roi_center = (bright_disk.center[0] - roi_rect[0], bright_disk.center[1] - roi_rect[1])
    score, score_vis = make_green_score_image(roi_bgr)

    min_radius = 20
    max_radius = int(round(bright_disk.radius * float(max_ring_ratio)))
    profile = radial_profile_from_score(
        score,
        center=roi_center,
        min_radius=min_radius,
        max_radius=max_radius,
        band_half_width=band_half_width,
    )
    smooth_kernel = _odd(max(3, smooth_kernel))
    profile_smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (1, smooth_kernel), 0).ravel()
    peaks = find_profile_peaks(
        profile_smoothed,
        radius_offset=min_radius,
        min_peak_value=min_peak_value,
        min_peak_distance=min_peak_distance,
    )

    if len(peaks) < target_ring:
        raise RuntimeError(
            f"Chi tim duoc {len(peaks)} vong. Hay giam min peak value, giam min peak distance, hoac tang crop ratio."
        )

    ring_radii = [float(rad) for rad, _ in peaks[:10]]
    selected_radius_roi = float(ring_radii[target_ring - 1])
    refined_center_roi, refined_radius = refine_center_and_radius(
        score,
        center=roi_center,
        radius=selected_radius_roi,
        search_radius=center_refine,
        radius_search=6,
        band_half_width=band_half_width,
    )
    center_abs = (
        float(refined_center_roi[0] + roi_rect[0]),
        float(refined_center_roi[1] + roi_rect[1]),
    )

    result = RingDetectionResult(
        selected_ring=int(target_ring),
        center=center_abs,
        radius=float(refined_radius),
        bright_disk=bright_disk,
        ring_radii=ring_radii,
        roi_rect=roi_rect,
        score_peaks=peaks,
    )

    profile_dbg = build_profile_view(
        profile_smoothed,
        peaks,
        selected_ring=target_ring,
        radius_offset=min_radius,
        selected_radius=refined_radius,
    )

    roi_dbg = roi_bgr.copy()
    for idx, ring_radius in enumerate(ring_radii, start=1):
        color = (0, 0, 255) if idx == target_ring else (0, 200, 0)
        thickness = 3 if idx == target_ring else 1
        cv2.circle(
            roi_dbg,
            (int(round(refined_center_roi[0])), int(round(refined_center_roi[1]))),
            int(round(ring_radius if idx != target_ring else refined_radius)),
            color,
            thickness,
        )
        cv2.putText(
            roi_dbg,
            str(idx),
            (int(round(refined_center_roi[0] + ring_radius + 6)), int(round(refined_center_roi[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    overlay = draw_overlay(original, result)
    stages: Dict[str, np.ndarray] = {
        "Original": original.copy(),
        "BrightDiskMask": bright_dbg,
        "TargetROI": roi_dbg,
        "GreenScore": score_vis,
        "Profile": profile_dbg,
        "Overlay": overlay,
    }
    return result, stages


class RingViewer(QWidget):
    def __init__(self, image_path: Optional[str]) -> None:
        super().__init__()
        self.setWindowTitle("Mirror Ring Inspector")
        self.resize(1450, 860)

        self.original: Optional[np.ndarray] = None
        self.stages: Dict[str, np.ndarray] = {}
        self.image_paths: List[Path] = []
        self.image_index: Optional[int] = None

        self.image_label = QLabel("Load an image...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #999; background: #111; color: #eee;")
        self.image_label.setMinimumWidth(900)

        self.stage_combo = QComboBox()
        self.stage_combo.addItems(STAGES)
        self.stage_combo.currentTextChanged.connect(self.update_preview)

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(lambda _=False: self.load_image())

        btn_prev = QPushButton("Previous")
        btn_prev.clicked.connect(lambda _=False: self.prev_image())

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(lambda _=False: self.next_image())

        btn_save = QPushButton("Save Current Stage...")
        btn_save.clicked.connect(self.save_current_stage)

        self.auto_fit = QPushButton("Auto fit: ON")
        self._auto_fit_on = True
        self.auto_fit.clicked.connect(self.toggle_autofit)

        self.status_label = QLabel("Status: ready")
        self.status_label.setStyleSheet("color: #111; font-weight: 700;")

        top_bar = QHBoxLayout()
        top_bar.addWidget(btn_load)
        top_bar.addWidget(btn_prev)
        top_bar.addWidget(btn_next)
        top_bar.addWidget(QLabel("View stage:"))
        top_bar.addWidget(self.stage_combo, 1)
        top_bar.addWidget(btn_save)
        top_bar.addWidget(self.auto_fit)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_bar)
        left_layout.addWidget(self.image_label, 1)
        left_layout.addWidget(self.status_label)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self._build_detect_group())
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        control_container = QWidget()
        control_container.setLayout(controls_layout)
        scroll.setWidget(control_container)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(scroll, 2)
        self.setLayout(main_layout)

        if image_path:
            self.load_image(path=image_path)

    def toggle_autofit(self) -> None:
        self._auto_fit_on = not self._auto_fit_on
        self.auto_fit.setText(f"Auto fit: {'ON' if self._auto_fit_on else 'OFF'}")
        self.update_preview()

    def _build_detect_group(self) -> QGroupBox:
        gb = QGroupBox("Target ring detection")
        f = QFormLayout()

        self.target_ring = QSpinBox()
        self.target_ring.setRange(1, 10)
        self.target_ring.setValue(10)
        self.target_ring.valueChanged.connect(self.on_param_change)

        self.bright_thresh = QSpinBox()
        self.bright_thresh.setRange(80, 255)
        self.bright_thresh.setValue(160)
        self.bright_thresh.valueChanged.connect(self.on_param_change)

        self.crop_ratio = QDoubleSpinBox()
        self.crop_ratio.setRange(0.10, 0.80)
        self.crop_ratio.setSingleStep(0.02)
        self.crop_ratio.setDecimals(2)
        self.crop_ratio.setValue(0.44)
        self.crop_ratio.valueChanged.connect(self.on_param_change)

        self.band_half_width = QSpinBox()
        self.band_half_width.setRange(1, 12)
        self.band_half_width.setValue(2)
        self.band_half_width.valueChanged.connect(self.on_param_change)

        self.smooth_kernel = QSpinBox()
        self.smooth_kernel.setRange(3, 51)
        self.smooth_kernel.setSingleStep(2)
        self.smooth_kernel.setValue(9)
        self.smooth_kernel.valueChanged.connect(self.on_param_change)

        self.min_peak_value = QDoubleSpinBox()
        self.min_peak_value.setRange(0.0, 50.0)
        self.min_peak_value.setSingleStep(0.5)
        self.min_peak_value.setDecimals(1)
        self.min_peak_value.setValue(6.0)
        self.min_peak_value.valueChanged.connect(self.on_param_change)

        self.min_peak_distance = QSpinBox()
        self.min_peak_distance.setRange(5, 100)
        self.min_peak_distance.setValue(28)
        self.min_peak_distance.valueChanged.connect(self.on_param_change)

        self.max_ring_ratio = QDoubleSpinBox()
        self.max_ring_ratio.setRange(0.20, 0.80)
        self.max_ring_ratio.setSingleStep(0.01)
        self.max_ring_ratio.setDecimals(2)
        self.max_ring_ratio.setValue(0.48)
        self.max_ring_ratio.valueChanged.connect(self.on_param_change)

        self.center_refine = QSpinBox()
        self.center_refine.setRange(0, 40)
        self.center_refine.setValue(10)
        self.center_refine.valueChanged.connect(self.on_param_change)

        f.addRow("Target ring (1-10):", self.target_ring)
        f.addRow("Bright threshold:", self.bright_thresh)
        f.addRow("Crop ratio:", self.crop_ratio)
        f.addRow("Band half width:", self.band_half_width)
        f.addRow("Profile smooth:", self.smooth_kernel)
        f.addRow("Min peak value:", self.min_peak_value)
        f.addRow("Min peak distance:", self.min_peak_distance)
        f.addRow("Max ring ratio:", self.max_ring_ratio)
        f.addRow("Center refine:", self.center_refine)

        gb.setLayout(f)
        return gb

    def load_image(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self._open_image_dialog()
        if not path:
            return

        self._set_image_list(Path(path))
        if self.image_index is None:
            return
        self._load_index(self.image_index)

    def _open_image_dialog(self) -> str:
        dialog = QFileDialog(self, "Select image")
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOptions(QFileDialog.DontUseNativeDialog)
        if not dialog.exec_():
            return ""
        files = dialog.selectedFiles()
        return files[0] if files else ""

    def _set_image_list(self, path: Path) -> None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not (path.exists() and path.is_file()):
            self.image_paths = []
            self.image_index = None
            return
        folder = path.parent
        images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
        self.image_paths = images
        self.image_index = images.index(path) if path in images else (0 if images else None)

    def _load_index(self, new_index: int) -> None:
        if not self.image_paths or not (0 <= new_index < len(self.image_paths)):
            return
        self.image_index = new_index
        img_path = self.image_paths[self.image_index]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot read image: {img_path.name}")
            return
        self.original = img
        self._safe_recompute_and_show()

    def prev_image(self) -> None:
        if self.image_index is None:
            return
        self._load_index(self.image_index - 1)

    def next_image(self) -> None:
        if self.image_index is None:
            return
        self._load_index(self.image_index + 1)

    def save_current_stage(self) -> None:
        if not self.stages:
            return
        stage = self.stage_combo.currentText()
        img = self.stages.get(stage)
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save stage image",
            f"{stage}.png",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        if not cv2.imwrite(path, img):
            QMessageBox.warning(self, "Save failed", "cv2.imwrite failed.")

    def on_param_change(self, *_: object) -> None:
        if self.original is None:
            return
        self._safe_recompute_and_show()

    def _safe_recompute_and_show(self) -> None:
        try:
            self.recompute_pipeline()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to process image: {exc}")
            if self.original is not None:
                self.stages = {"Original": self.original.copy()}
                self.stage_combo.setCurrentText("Original")
                self.update_preview()
            return
        self.update_preview()

    def recompute_pipeline(self) -> None:
        if self.original is None:
            self.stages = {}
            return

        start = time.perf_counter()
        result, stages = detect_numbered_ring(
            self.original,
            target_ring=int(self.target_ring.value()),
            bright_thresh=int(self.bright_thresh.value()),
            crop_ratio=float(self.crop_ratio.value()),
            band_half_width=int(self.band_half_width.value()),
            smooth_kernel=int(self.smooth_kernel.value()),
            min_peak_value=float(self.min_peak_value.value()),
            min_peak_distance=int(self.min_peak_distance.value()),
            max_ring_ratio=float(self.max_ring_ratio.value()),
            center_refine=int(self.center_refine.value()),
        )
        self.stages = stages

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        cx, cy = result.center
        self.status_label.setText(
            f"Status: ring={result.selected_ring} | center=({cx:.1f},{cy:.1f}) | r={result.radius:.1f}px | "
            f"found={len(result.score_peaks)} peaks | bright_r={result.bright_disk.radius:.1f}px | time={elapsed_ms:.1f} ms"
        )

    def update_preview(self) -> None:
        if not self.stages:
            return
        stage = self.stage_combo.currentText()
        img = self.stages.get(stage)
        if img is None:
            return
        pix = cv_to_qpixmap(img)
        if self._auto_fit_on:
            target = self.image_label.size()
            if target.width() > 1 and target.height() > 1:
                pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit_on:
            self.update_preview()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect the selected numbered circle on the mirror target")
    p.add_argument("--image", default=None, help="Optional image path to load")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = RingViewer(args.image)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

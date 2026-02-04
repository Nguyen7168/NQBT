"""
GUI tool to debug circle crop pipeline stages.

Usage:
  python test_crop.py --config config.yaml --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config_loader import load_config  # noqa: E402
from app.inspection.cropping import CircleCropper  # noqa: E402


STAGES: List[str] = [
    "Original",
    "Gray",
    "Blur",
    "Erode",
    "Threshold",
    "Dilate",
    "Watershed",
    "Markers",
    "CircleMask",
    "Overlay",
]


def cv_to_qpixmap(img: Optional[np.ndarray]) -> QPixmap:
    """Convert OpenCV image (BGR/Gray) to QPixmap."""
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


def normalize_markers(markers: np.ndarray) -> np.ndarray:
    """Normalize int32 watershed markers into uint8 [0..255] for display."""
    markers_i32 = markers.astype(np.int32, copy=False)
    if markers_i32.size == 0:
        return np.zeros_like(markers_i32, dtype=np.uint8)

    m_min = int(markers_i32.min())
    m_max = int(markers_i32.max())
    if m_max == m_min:
        return np.zeros_like(markers_i32, dtype=np.uint8)

    scaled = (markers_i32 - m_min) / float(m_max - m_min) * 255.0
    return scaled.astype(np.uint8)


class PipelineViewer(QWidget):
    def __init__(self, config_path: str, image_path: Optional[str]) -> None:
        super().__init__()
        self.setWindowTitle("Circle Crop Debugger")
        self.resize(1500, 820)

        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.layout = self.cfg.layout
        self.cropper = CircleCropper(self.layout)

        self.original: Optional[np.ndarray] = None
        self.stages: Dict[str, np.ndarray] = {}
        self.detected: Optional[int] = None
        self.expected: Optional[int] = None
        self.image_paths: List[Path] = []
        self.image_index: Optional[int] = None

        # ===== Left (preview) =====
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

        self.auto_fit = QCheckBox("Auto fit")
        self.auto_fit.setChecked(True)
        self.auto_fit.stateChanged.connect(self.update_preview)

        self.status_label = QLabel("Status: ready")
        self.status_label.setStyleSheet("color: #111; font-weight: 600;")

        self.radii_label = QLabel("Radii: -")
        self.radii_label.setStyleSheet("color: #222;")

        self.stats_label = QLabel("Stats: -")
        self.metrics_help_label = QLabel(
            "Help: std = độ lệch chuẩn bán kính (px) — càng nhỏ càng đồng đều | "
            "cv = std/mean — càng nhỏ càng ổn định | "
            "%detect = detected/expected — càng cao càng tốt"
        )
        self.metrics_help_label.setWordWrap(True)
        self.metrics_help_label.setStyleSheet("color: #666; font-size: 10pt;")
        self.stats_label.setStyleSheet("color: #222;")

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
        left_layout.addWidget(self.radii_label)
        left_layout.addWidget(self.stats_label)
        left_layout.addWidget(self.metrics_help_label)

        # ===== Right (controls) =====
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self._build_circle_group())
        controls_layout.addWidget(self._build_watershed_group())
        controls_layout.addWidget(self._build_layout_group())
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

    # =========================
    # UI groups
    # =========================
    def _build_circle_group(self) -> QGroupBox:
        gb = QGroupBox("Circle preprocess")
        f = QFormLayout()

        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(1, 99)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(int(self.layout.circle_blur_kernel))
        self.blur_kernel.valueChanged.connect(self.on_param_change)

        self.erode_iter = QSpinBox()
        self.erode_iter.setRange(0, 50)
        self.erode_iter.setValue(int(self.layout.circle_erode_iter))
        self.erode_iter.valueChanged.connect(self.on_param_change)

        self.dilate_iter = QSpinBox()
        self.dilate_iter.setRange(0, 50)
        self.dilate_iter.setValue(int(self.layout.circle_dilate_iter))
        self.dilate_iter.valueChanged.connect(self.on_param_change)

        self.threshold_val = QSpinBox()
        self.threshold_val.setRange(0, 255)
        self.threshold_val.setValue(int(self.layout.circle_threshold))
        self.threshold_val.valueChanged.connect(self.on_param_change)

        self.use_otsu = QCheckBox("Use OTSU")
        self.use_otsu.setChecked(bool(self.layout.circle_use_otsu))
        self.use_otsu.stateChanged.connect(self.on_param_change)

        self.invert = QCheckBox("Invert")
        self.invert.setChecked(bool(self.layout.circle_invert))
        self.invert.stateChanged.connect(self.on_param_change)

        self.min_radius = QSpinBox()
        self.min_radius.setRange(1, 2000)
        self.min_radius.setValue(int(self.layout.circle_min_radius))
        self.min_radius.valueChanged.connect(self.on_param_change)

        self.max_radius = QSpinBox()
        self.max_radius.setRange(1, 2000)
        self.max_radius.setValue(int(self.layout.circle_max_radius))
        self.max_radius.valueChanged.connect(self.on_param_change)

        self.radius_expand = QSpinBox()
        self.radius_expand.setRange(0, 200)
        self.radius_expand.setValue(int(self.layout.circle_radius_expand))
        self.radius_expand.valueChanged.connect(self.on_param_change)

        f.addRow("Blur kernel:", self.blur_kernel)
        f.addRow("Erode iter:", self.erode_iter)
        f.addRow("Dilate iter:", self.dilate_iter)
        f.addRow("Threshold:", self.threshold_val)
        f.addRow("Min radius:", self.min_radius)
        f.addRow("Max radius:", self.max_radius)
        f.addRow("Radius expand:", self.radius_expand)
        f.addRow(self.use_otsu)
        f.addRow(self.invert)

        gb.setLayout(f)
        return gb

    def _build_watershed_group(self) -> QGroupBox:
        gb = QGroupBox("Watershed split")
        f = QFormLayout()

        self.ws_scale = QDoubleSpinBox()
        self.ws_scale.setRange(0.1, 1.0)
        self.ws_scale.setSingleStep(0.1)
        self.ws_scale.setDecimals(2)
        self.ws_scale.setValue(float(self.layout.ws_scale))
        self.ws_scale.valueChanged.connect(self.on_param_change)

        self.ws_dist_ratio = QDoubleSpinBox()
        self.ws_dist_ratio.setRange(0.05, 1.0)
        self.ws_dist_ratio.setSingleStep(0.05)
        self.ws_dist_ratio.setDecimals(2)
        self.ws_dist_ratio.setValue(float(self.layout.ws_dist_ratio))
        self.ws_dist_ratio.valueChanged.connect(self.on_param_change)

        self.ws_open_ksize = QSpinBox()
        self.ws_open_ksize.setRange(1, 99)
        self.ws_open_ksize.setSingleStep(2)
        self.ws_open_ksize.setValue(int(self.layout.ws_open_ksize))
        self.ws_open_ksize.valueChanged.connect(self.on_param_change)

        self.ws_open_iter = QSpinBox()
        self.ws_open_iter.setRange(0, 20)
        self.ws_open_iter.setValue(int(self.layout.ws_open_iter))
        self.ws_open_iter.valueChanged.connect(self.on_param_change)

        self.ws_bg_dilate = QSpinBox()
        self.ws_bg_dilate.setRange(0, 20)
        self.ws_bg_dilate.setValue(int(self.layout.ws_bg_dilate_iter))
        self.ws_bg_dilate.valueChanged.connect(self.on_param_change)

        self.ws_dist_mask = QSpinBox()
        self.ws_dist_mask.setRange(3, 9)
        self.ws_dist_mask.setSingleStep(2)
        self.ws_dist_mask.setValue(int(self.layout.ws_dist_mask_size))
        self.ws_dist_mask.valueChanged.connect(self.on_param_change)

        self.ws_min_area = QSpinBox()
        self.ws_min_area.setRange(1, 1_000_000)
        self.ws_min_area.setValue(int(self.layout.ws_min_area_px))
        self.ws_min_area.valueChanged.connect(self.on_param_change)

        self.ws_max_pts = QSpinBox()
        self.ws_max_pts.setRange(100, 10_000)
        self.ws_max_pts.setValue(int(self.layout.ws_max_contour_pts))
        self.ws_max_pts.valueChanged.connect(self.on_param_change)

        f.addRow("Scale:", self.ws_scale)
        f.addRow("Dist ratio:", self.ws_dist_ratio)
        f.addRow("Open ksize:", self.ws_open_ksize)
        f.addRow("Open iter:", self.ws_open_iter)
        f.addRow("BG dilate:", self.ws_bg_dilate)
        f.addRow("Dist mask:", self.ws_dist_mask)
        f.addRow("Min area:", self.ws_min_area)
        f.addRow("Max contour pts:", self.ws_max_pts)

        gb.setLayout(f)
        return gb

    def _build_layout_group(self) -> QGroupBox:
        gb = QGroupBox("Layout")
        f = QFormLayout()

        self.rows = QSpinBox()
        self.rows.setRange(1, 20)
        self.rows.setValue(int(self.layout.rows))
        self.rows.valueChanged.connect(self.on_param_change)

        self.cols = QSpinBox()
        self.cols.setRange(1, 30)
        self.cols.setValue(int(self.layout.cols))
        self.cols.valueChanged.connect(self.on_param_change)

        self.expected_count = QSpinBox()
        self.expected_count.setRange(1, 100)
        self.expected_count.setValue(int(self.layout.count))
        self.expected_count.valueChanged.connect(self.on_param_change)

        self.patch_padding = QSpinBox()
        self.patch_padding.setRange(0, 200)
        self.patch_padding.setValue(int(self.layout.patch_padding))
        self.patch_padding.valueChanged.connect(self.on_param_change)

        f.addRow("Rows:", self.rows)
        f.addRow("Cols:", self.cols)
        f.addRow("Expected count:", self.expected_count)
        f.addRow("Patch padding:", self.patch_padding)

        gb.setLayout(f)
        return gb

    # =========================
    # Image loading/navigation
    # =========================
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

    # =========================
    # Parameters & pipeline
    # =========================
    def on_param_change(self, *_: object) -> None:
        if self.original is None:
            return
        self._sync_layout_from_ui()
        self._safe_recompute_and_show()

    def _sync_layout_from_ui(self) -> None:
        self.layout.circle_blur_kernel = int(self.blur_kernel.value())
        self.layout.circle_erode_iter = int(self.erode_iter.value())
        self.layout.circle_dilate_iter = int(self.dilate_iter.value())
        self.layout.circle_threshold = int(self.threshold_val.value())
        self.layout.circle_use_otsu = bool(self.use_otsu.isChecked())
        self.layout.circle_invert = bool(self.invert.isChecked())
        self.layout.circle_min_radius = int(self.min_radius.value())
        self.layout.circle_max_radius = int(self.max_radius.value())
        self.layout.circle_radius_expand = int(self.radius_expand.value())

        self.layout.ws_scale = float(self.ws_scale.value())
        self.layout.ws_dist_ratio = float(self.ws_dist_ratio.value())
        self.layout.ws_open_ksize = int(self.ws_open_ksize.value())
        self.layout.ws_open_iter = int(self.ws_open_iter.value())
        self.layout.ws_bg_dilate_iter = int(self.ws_bg_dilate.value())
        self.layout.ws_dist_mask_size = int(self.ws_dist_mask.value())
        self.layout.ws_min_area_px = int(self.ws_min_area.value())
        self.layout.ws_max_contour_pts = int(self.ws_max_pts.value())

        self.layout.rows = int(self.rows.value())
        self.layout.cols = int(self.cols.value())
        self.layout.count = int(self.expected_count.value())
        self.layout.patch_padding = int(self.patch_padding.value())

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

        self.stage_combo.setCurrentText("Overlay")
        self.update_preview()

    def _compute_threshold(self, img: np.ndarray) -> np.ndarray:
        if self.layout.circle_use_otsu:
            _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, out = cv2.threshold(img, int(self.layout.circle_threshold), 255, cv2.THRESH_BINARY)
        if self.layout.circle_invert:
            out = cv2.bitwise_not(out)
        return out

    def _fit_circles_from_markers(
        self, markers: np.ndarray, inv_scale: float
    ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        circles: List[Tuple[float, float, float]] = []
        radii: List[float] = []

        labels = np.unique(markers)
        labels = labels[labels > 1]  # ignore background and watershed boundary

        for lb in labels:
            obj_mask_s = (markers == lb).astype(np.uint8) * 255
            area_s = int(cv2.countNonZero(obj_mask_s))
            area_est = area_s * (inv_scale * inv_scale) if inv_scale != 1.0 else area_s
            if area_est < int(self.layout.ws_min_area_px):
                continue

            fit = self.cropper._fit_outer_from_mask(obj_mask_s)
            if fit is None:
                continue

            cx_s, cy_s, r_s = fit
            cx = float(cx_s) * inv_scale
            cy = float(cy_s) * inv_scale
            r = float(r_s) * inv_scale

            if not (self.layout.circle_min_radius <= r <= self.layout.circle_max_radius):
                continue

            circles.append((cx, cy, r))
            radii.append(r)

        return circles, radii

    def _draw_overlay(
        self, base_bgr: np.ndarray, circles: List[Tuple[float, float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        overlay = base_bgr.copy()
        mask = np.zeros(base_bgr.shape[:2], dtype=np.uint8)

        for cx, cy, r in circles:
            center = (int(round(cx)), int(round(cy)))
            radius = int(round(r))

            cv2.circle(mask, center, radius, 255, -1)
            cv2.circle(overlay, center, radius, (0, 255, 0), 6)

            label = f"{r:.1f}"
            base_size, _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
            base_h = max(base_size[1], 1)

            target_h = max(r / 3.0, 8.0)
            font_scale = max(0.3, target_h / base_h)
            thickness = max(1, int(round(font_scale * 1.2)))

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_org = (center[0] - text_w // 2, center[1] + text_h // 2)

            cv2.putText(
                overlay,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return overlay, mask

    def _update_status_labels(self, elapsed_ms: float, radii: List[float]) -> None:
        self.detected = int(len(radii))
        self.expected = int(self.layout.count)

        self.status_label.setText(
            "Status: detected "
            f"{self.detected}/{self.expected} circles"
            f" | time {elapsed_ms:.1f} ms"
            f" | config {Path(self.config_path).name}"
        )

        if not radii:
            self.radii_label.setText("Radii: -")
            self.stats_label.setText("Stats: -")
            return

        self.radii_label.setText("Radii: " + ", ".join(f"{r:.1f}" for r in radii))

        radii_arr = np.asarray(radii, dtype=np.float32)
        mean_radius = float(radii_arr.mean())
        mean_diameter = mean_radius * 2.0
        radius_range = float(radii_arr.max() - radii_arr.min())
        std_radius = float(radii_arr.std())
        cv_radius = (std_radius / mean_radius) if mean_radius > 1e-6 else 0.0
        pct_detect = (self.detected / self.expected * 100.0) if self.expected else 0.0

        self.stats_label.setText(
            "Stats: "
            f"mean_radius {mean_radius:.1f}"
            f" | range {radius_range:.1f}"
            f" | std {std_radius:.2f}"
            f" | cv {cv_radius:.3f}"
            f" | % detect {pct_detect:.1f}%"
        )

    def recompute_pipeline(self) -> None:
        if self.original is None:
            self.stages = {}
            return

        start = time.perf_counter()
        stages: Dict[str, np.ndarray] = {"Original": self.original.copy()}

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        stages["Gray"] = gray.copy()

        k = max(1, int(self.layout.circle_blur_kernel) // 2 * 2 + 1)  # enforce odd
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        stages["Blur"] = blur.copy()

        eroded = blur
        if int(self.layout.circle_erode_iter) > 0:
            ker = np.ones((9, 9), np.uint8)
            eroded = cv2.erode(blur, ker, iterations=int(self.layout.circle_erode_iter))
        stages["Erode"] = eroded.copy()

        thresh = self._compute_threshold(eroded)
        stages["Threshold"] = thresh.copy()

        dil = thresh
        if int(self.layout.circle_dilate_iter) > 0:
            ker = np.ones((4, 4), np.uint8)
            dil = cv2.dilate(thresh, ker, iterations=int(self.layout.circle_dilate_iter))
        stages["Dilate"] = dil.copy()

        ws_scale = float(self.layout.ws_scale)
        if 0.0 < ws_scale < 1.0:
            bin_small = self.cropper._resize_binary(dil, ws_scale)
            inv_scale = 1.0 / ws_scale
        else:
            bin_small = dil
            inv_scale = 1.0

        markers = self.cropper._split_by_dist_watershed(bin_small)
        stages["Watershed"] = bin_small.copy()
        stages["Markers"] = normalize_markers(markers)

        circles, radii = self._fit_circles_from_markers(markers, inv_scale)
        overlay, mask = self._draw_overlay(self.original, circles)

        stages["CircleMask"] = mask
        stages["Overlay"] = overlay
        self.stages = stages

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._update_status_labels(elapsed_ms, radii)

    # =========================
    # Preview rendering
    # =========================
    def update_preview(self) -> None:
        if not self.stages:
            return

        stage = self.stage_combo.currentText()
        img = self.stages.get(stage)
        if img is None:
            return

        pix = cv_to_qpixmap(img)
        if self.auto_fit.isChecked():
            target = self.image_label.size()
            if target.width() > 1 and target.height() > 1:
                pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.auto_fit.isChecked():
            self.update_preview()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI debugger for circle crop pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    parser.add_argument("--image", default=None, help="Optional image path to load")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = PipelineViewer(args.config, args.image)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

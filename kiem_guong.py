#!/usr/bin/env python3
"""
Minimal GUI to detect ONLY outer circle (center + radius) on a white-mirror ring target.

Usage:
  python test_outercircle.py --image path/to/image.jpg

Pipeline:
  Original -> Gray -> Blur -> Edges(Canny) -> Largest Contour -> minEnclosingCircle
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

STAGES: List[str] = [
    "Original",
    "Gray",
    "Blur",
    "Edges",
    "OuterCircle",
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
class OuterCircleResult:
    center: Tuple[float, float]
    radius: float
    contour_area: float


def find_outer_circle_from_edges(
    edges: np.ndarray,
    min_contour_area: float = 5000.0,
) -> Tuple[OuterCircleResult, np.ndarray]:
    """
    Find outer circle via largest contour -> minEnclosingCircle.

    Returns:
      result, debug_bgr
    """
    dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found. Try adjusting blur/canny.")

    # Filter too-small contours first (noise)
    filtered = [c for c in contours if cv2.contourArea(c) >= float(min_contour_area)]
    if not filtered:
        filtered = contours

    cnt = max(filtered, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    (cx, cy), r = cv2.minEnclosingCircle(cnt)

    cv2.circle(dbg, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 3)
    cv2.circle(dbg, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)
    cv2.putText(
        dbg,
        f"outer_r={r:.2f}px | area={area:.0f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )

    return OuterCircleResult(center=(float(cx), float(cy)), radius=float(r), contour_area=area), dbg


def draw_overlay(original: np.ndarray, center: Tuple[float, float], radius: float) -> np.ndarray:
    out = original.copy()
    cx, cy = center
    cv2.circle(out, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 255, 0), 3)
    cv2.putText(
        out,
        f"Outer D = {2.0*radius:.2f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Outer D = {2.0*radius:.2f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


class OuterCircleViewer(QWidget):
    def __init__(self, image_path: Optional[str]) -> None:
        super().__init__()
        self.setWindowTitle("OuterCircle Inspector (minimal)")
        self.resize(1450, 820)

        self.original: Optional[np.ndarray] = None
        self.stages: Dict[str, np.ndarray] = {}
        self.image_paths: List[Path] = []
        self.image_index: Optional[int] = None

        # ===== Left preview =====
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

        # ===== Right controls =====
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
        gb = QGroupBox("Outer circle detection")
        f = QFormLayout()

        # Defaults match your screenshot: blur=3, canny1=222, canny2=100
        self.blur_k = QSpinBox()
        self.blur_k.setRange(1, 99)
        self.blur_k.setSingleStep(2)
        self.blur_k.setValue(3)
        self.blur_k.valueChanged.connect(self.on_param_change)

        self.canny1 = QSpinBox()
        self.canny1.setRange(0, 2000)
        self.canny1.setValue(222)
        self.canny1.valueChanged.connect(self.on_param_change)

        self.canny2 = QSpinBox()
        self.canny2.setRange(0, 4000)
        self.canny2.setValue(100)
        self.canny2.valueChanged.connect(self.on_param_change)

        self.min_area = QSpinBox()
        self.min_area.setRange(0, 50_000_000)
        self.min_area.setSingleStep(5000)
        self.min_area.setValue(5000)
        self.min_area.valueChanged.connect(self.on_param_change)

        f.addRow("Blur kernel:", self.blur_k)
        f.addRow("Canny th1:", self.canny1)
        f.addRow("Canny th2:", self.canny2)
        f.addRow("Min contour area:", self.min_area)

        gb.setLayout(f)
        return gb

    # ---------- image navigation ----------
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

    # ---------- pipeline ----------
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
        stages: Dict[str, np.ndarray] = {"Original": self.original.copy()}

        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        stages["Gray"] = gray.copy()

        k = _odd(int(self.blur_k.value()))
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        stages["Blur"] = blur.copy()

        e1 = int(self.canny1.value())
        e2 = int(self.canny2.value())
        edges = cv2.Canny(blur, e1, e2)
        stages["Edges"] = edges.copy()

        result, outer_dbg = find_outer_circle_from_edges(
            edges,
            min_contour_area=float(self.min_area.value()),
        )
        stages["OuterCircle"] = outer_dbg

        overlay = draw_overlay(self.original, result.center, result.radius)
        stages["Overlay"] = overlay

        self.stages = stages

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        cx, cy = result.center
        self.status_label.setText(
            f"Status: center=({cx:.1f},{cy:.1f}) | outer_r={result.radius:.2f}px | "
            f"outer_D={2.0*result.radius:.2f}px | area={result.contour_area:.0f} | time={elapsed_ms:.1f} ms"
        )

    # ---------- preview ----------
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
    p = argparse.ArgumentParser(description="Minimal GUI to detect ONLY outer circle")
    p.add_argument("--image", default=None, help="Optional image path to load")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = OuterCircleViewer(args.image)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

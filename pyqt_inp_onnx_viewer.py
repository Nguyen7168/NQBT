"""PyQt app to inspect cropped images with INP ONNX inference (Original/Heatmap/Binary)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import ConfigError, load_config
from app.models.anomaly import AnomalyDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ItemResult:
    path: Path
    score: float
    status: str
    heat: np.ndarray
    binary: np.ndarray
    original: np.ndarray


def np_to_pix(img_bgr_or_gray: np.ndarray) -> QtGui.QPixmap:
    if img_bgr_or_gray.ndim == 2:
        qimg = QtGui.QImage(
            img_bgr_or_gray.data,
            img_bgr_or_gray.shape[1],
            img_bgr_or_gray.shape[0],
            img_bgr_or_gray.strides[0],
            QtGui.QImage.Format_Grayscale8,
        )
    else:
        rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        )
    return QtGui.QPixmap.fromImage(qimg.copy())


class InpOnnxViewer(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("INP ONNX Crop Viewer")
        self.resize(1280, 760)

        self.config_path = QtWidgets.QLineEdit("config.yaml")
        self.model_path = QtWidgets.QLineEdit()
        self.input_dir = QtWidgets.QLineEdit()
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(-1e9, 1e9)
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(28)
        self.bin_threshold = QtWidgets.QDoubleSpinBox()
        self.bin_threshold.setDecimals(6)
        self.bin_threshold.setRange(0, 1)
        self.bin_threshold.setSingleStep(0.01)
        self.bin_threshold.setValue(0.2)
        self.btn_run = QtWidgets.QPushButton("Run INP ONNX")
        self.btn_run.clicked.connect(self.run_inference)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select_row)

        self.lbl_original = QtWidgets.QLabel("Original")
        self.lbl_heat = QtWidgets.QLabel("Heatmap")
        self.lbl_binary = QtWidgets.QLabel("Binary")
        for lbl in (self.lbl_original, self.lbl_heat, self.lbl_binary):
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumSize(300, 300)
            lbl.setStyleSheet("border:1px solid #666;")

        self.info = QtWidgets.QLabel("Score: -")
        self.info.setStyleSheet("font-size:16px;font-weight:600;")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(self._row("Config", self.config_path, self._btn(self.pick_config)))
        root.addLayout(self._row("Model ONNX", self.model_path, self._btn(self.pick_model)))
        root.addLayout(self._row("Input Folder", self.input_dir, self._btn(self.pick_input_dir)))
        root.addLayout(self._params_row())
        root.addWidget(self.btn_run)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.list_widget, 2)
        right = QtWidgets.QVBoxLayout()
        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.lbl_original, 1)
        imgs.addWidget(self.lbl_heat, 1)
        imgs.addWidget(self.lbl_binary, 1)
        right.addLayout(imgs, 4)
        right.addWidget(self.info)
        right.addWidget(self.log, 2)
        body.addLayout(right, 5)
        root.addLayout(body, 1)

        self.results: List[ItemResult] = []

    def _btn(self, slot):
        b = QtWidgets.QPushButton("Browse")
        b.clicked.connect(slot)
        return b

    def _row(self, label: str, edit: QtWidgets.QLineEdit, btn: QtWidgets.QPushButton):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        return row

    def _params_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Threshold"))
        row.addWidget(self.threshold)
        row.addWidget(QtWidgets.QLabel("Batch"))
        row.addWidget(self.batch_size)
        row.addWidget(QtWidgets.QLabel("Binary Thres"))
        row.addWidget(self.bin_threshold)
        row.addStretch(1)
        return row

    def _collect_images(self, root: Path) -> List[Path]:
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)
        QtWidgets.QApplication.processEvents()

    def pick_config(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Config", self.config_path.text(), "YAML (*.yaml *.yml)")
        if p:
            self.config_path.setText(p)

    def pick_model(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Model", self.model_path.text(), "ONNX (*.onnx)")
        if p:
            self.model_path.setText(p)

    def pick_input_dir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Input folder", self.input_dir.text())
        if p:
            self.input_dir.setText(p)

    def run_inference(self):
        self.results.clear()
        self.list_widget.clear()
        self.log.clear()
        self.btn_run.setEnabled(False)
        try:
            cfg = load_config(self.config_path.text().strip())
            cfg.models.algo = "INP"
            cfg.models.inp.path = self.model_path.text().strip()
            cfg.models.inp.inp_mode = "default"
            self.threshold.setValue(float(cfg.models.inp.inp_threshold))
            self.bin_threshold.setValue(float(cfg.models.inp.inp_bin_thresh))
            detector = AnomalyDetector(cfg.models)

            paths = self._collect_images(Path(self.input_dir.text().strip()))
            if not paths:
                raise RuntimeError("No images found.")
            self._log(f"Images: {len(paths)}")
            self._log(f"Threshold: {self.threshold.value():.6f}  Binary: {self.bin_threshold.value():.6f}")

            for start in range(0, len(paths), int(self.batch_size.value())):
                batch_paths = paths[start : start + int(self.batch_size.value())]
                imgs, ok_paths = [], []
                for p in batch_paths:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    imgs.append(img)
                    ok_paths.append(p)
                if not imgs:
                    continue
                res = detector.infer(imgs)
                for p, img, score, amap in zip(ok_paths, imgs, res.scores, res.maps or []):
                    gray = np.clip(amap * 255.0, 0, 255).astype(np.uint8)
                    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                    binary = ((amap >= float(self.bin_threshold.value())).astype(np.uint8) * 255)
                    status = "NG" if float(score) >= float(self.threshold.value()) else "OK"
                    self.results.append(
                        ItemResult(
                            path=p,
                            score=float(score),
                            status=status,
                            heat=heat,
                            binary=binary,
                            original=img,
                        )
                    )
                self._log(f"Processed: {min(start + len(batch_paths), len(paths))}/{len(paths)}")

            for it in self.results:
                self.list_widget.addItem(f"{it.path.name} | {it.score:.6f} | {it.status}")
            if self.results:
                self.list_widget.setCurrentRow(0)
            self._log("Done.")
        except (RuntimeError, ConfigError) as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            self._log(f"[ERROR] {exc}")
        finally:
            self.btn_run.setEnabled(True)

    def on_select_row(self, row: int):
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]
        self.lbl_original.setPixmap(np_to_pix(r.original).scaled(self.lbl_original.size(), QtCore.Qt.KeepAspectRatio))
        self.lbl_heat.setPixmap(np_to_pix(r.heat).scaled(self.lbl_heat.size(), QtCore.Qt.KeepAspectRatio))
        self.lbl_binary.setPixmap(np_to_pix(r.binary).scaled(self.lbl_binary.size(), QtCore.Qt.KeepAspectRatio))
        self.info.setText(f"File: {r.path.name} | Score: {r.score:.6f} | Status: {r.status}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = InpOnnxViewer()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

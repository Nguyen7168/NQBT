"""Simple PyQt5 app to run anomaly inference on pre-cropped images and split by threshold."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List

import cv2

from app.config_loader import ConfigError, load_config
from app.models.anomaly import AnomalyDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ThresholdSorterApp:
    def __init__(self, QtWidgets) -> None:
        self.QtWidgets = QtWidgets
        self.widget = QtWidgets.QWidget()
        self.widget.setWindowTitle("Anomaly Threshold Sorter")
        self.widget.resize(760, 460)

        self.config_path = QtWidgets.QLineEdit("config.yaml")
        self.model_path = QtWidgets.QLineEdit()
        self.input_dir = QtWidgets.QLineEdit()
        self.output_dir = QtWidgets.QLineEdit()
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setRange(-1e9, 1e9)
        self.threshold_spin.setValue(0.15)

        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(["Auto", "INP", "GLASS"])

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)

        self.btn_run = QtWidgets.QPushButton("Infer & Split")
        self.btn_run.clicked.connect(self.run_inference)

        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.addLayout(self._row("Config", self.config_path, self._btn("Browse", self.pick_config)))
        layout.addLayout(self._row("Model", self.model_path, self._btn("Browse", self.pick_model)))
        layout.addLayout(self._row("Input Folder", self.input_dir, self._btn("Browse", self.pick_input_dir)))
        layout.addLayout(self._row("Output Root", self.output_dir, self._btn("Browse", self.pick_output_dir)))
        layout.addLayout(self._algo_row())
        layout.addLayout(self._threshold_row())
        layout.addWidget(self.btn_run)
        layout.addWidget(self.log_box)

    def show(self):
        self.widget.show()

    def _btn(self, title: str, slot):
        btn = self.QtWidgets.QPushButton(title)
        btn.clicked.connect(slot)
        return btn

    def _row(self, label: str, editor, btn):
        row = self.QtWidgets.QHBoxLayout()
        row.addWidget(self.QtWidgets.QLabel(label))
        row.addWidget(editor, 1)
        row.addWidget(btn)
        return row

    def _algo_row(self):
        row = self.QtWidgets.QHBoxLayout()
        row.addWidget(self.QtWidgets.QLabel("Algo"))
        row.addWidget(self.algo_combo, 1)
        return row

    def _threshold_row(self):
        row = self.QtWidgets.QHBoxLayout()
        row.addWidget(self.QtWidgets.QLabel("Threshold (>= upper)"))
        row.addWidget(self.threshold_spin, 1)
        return row

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(text)
        self.QtWidgets.QApplication.processEvents()

    def pick_config(self):
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.widget, "Choose config", self.config_path.text(), "YAML (*.yaml *.yml)")
        if path:
            self.config_path.setText(path)

    def pick_model(self):
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.widget, "Choose model", self.model_path.text(), "Model Files (*.onnx *.pth *.pt);;All Files (*)")
        if path:
            self.model_path.setText(path)

    def pick_input_dir(self):
        path = self.QtWidgets.QFileDialog.getExistingDirectory(self.widget, "Choose input folder", self.input_dir.text())
        if path:
            self.input_dir.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(path)

    def pick_output_dir(self):
        path = self.QtWidgets.QFileDialog.getExistingDirectory(self.widget, "Choose output folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def _collect_images(self, root: Path) -> List[Path]:
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    def _resolve_algo(self, model_path: str, config_algo: str) -> str:
        ui_algo = self.algo_combo.currentText().strip().upper()
        if ui_algo in {"INP", "GLASS"}:
            return ui_algo
        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            return "INP"
        if suffix in {".pth", ".pt"}:
            return "GLASS"
        return config_algo

    def run_inference(self):
        self.log_box.clear()
        self.btn_run.setEnabled(False)
        try:
            cfg_path = self.config_path.text().strip()
            model_path = self.model_path.text().strip()
            in_dir = Path(self.input_dir.text().strip())
            out_root = Path(self.output_dir.text().strip())
            threshold = float(self.threshold_spin.value())

            if not cfg_path or not Path(cfg_path).exists():
                raise RuntimeError("Config path is invalid")
            if not model_path or not Path(model_path).exists():
                raise RuntimeError("Model path is invalid")
            if not in_dir.exists() or not in_dir.is_dir():
                raise RuntimeError("Input folder is invalid")
            if not out_root.exists() or not out_root.is_dir():
                raise RuntimeError("Output folder is invalid")

            config = load_config(cfg_path)
            config_algo = str(config.models.algo or "INP").strip().upper()
            algo = self._resolve_algo(model_path, config_algo)
            config.models.algo = algo
            if algo == "INP":
                config.models.inp.path = model_path
            elif algo == "GLASS":
                config.models.glass.path = model_path
            else:
                raise RuntimeError(f"Unsupported algo: {algo}")

            detector = AnomalyDetector(config.models)
            images = self._collect_images(in_dir)
            if not images:
                raise RuntimeError("No images found in input folder")

            upper_dir = out_root / "upper threshold"
            under_dir = out_root / "under threshold"
            upper_dir.mkdir(parents=True, exist_ok=True)
            under_dir.mkdir(parents=True, exist_ok=True)

            upper_count = 0
            under_count = 0
            scores = []
            self.log(f"Algo: {algo}")
            self.log(f"Input images: {len(images)}")
            self.log(f"Threshold: {threshold:.6f}")

            for idx, img_path in enumerate(images, 1):
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    self.log(f"[WARN] Skip unreadable image: {img_path}")
                    continue

                result = detector.infer([image])
                score = float(result.scores[0])
                scores.append(score)
                if score >= threshold:
                    dst = upper_dir / img_path.name
                    upper_count += 1
                else:
                    dst = under_dir / img_path.name
                    under_count += 1

                dst = self._unique_path(dst)
                shutil.copy2(img_path, dst)
                self.log(f"[{idx}/{len(images)}] {img_path.name} -> score={score:.6f} => {dst.parent.name}")

            self.log("---- DONE ----")
            if scores:
                self.log(f"Score min/max/mean: {min(scores):.6f} / {max(scores):.6f} / {sum(scores)/len(scores):.6f}")
            self.log(f"upper threshold: {upper_count}")
            self.log(f"under threshold: {under_count}")
            self.QtWidgets.QMessageBox.information(self.widget, "Finished", "Inference and split completed.")
        except (RuntimeError, ConfigError) as exc:
            self.QtWidgets.QMessageBox.critical(self.widget, "Error", str(exc))
            self.log(f"[ERROR] {exc}")
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.widget, "Unexpected Error", str(exc))
            self.log(f"[ERROR] Unexpected: {exc}")
        finally:
            self.btn_run.setEnabled(True)

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        i = 1
        while True:
            candidate = parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1


def main() -> int:
    try:
        from PyQt5 import QtWidgets
    except Exception as exc:
        print(f"PyQt5 import failed: {exc}")
        return 1

    app = QtWidgets.QApplication(sys.argv)
    win = ThresholdSorterApp(QtWidgets)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

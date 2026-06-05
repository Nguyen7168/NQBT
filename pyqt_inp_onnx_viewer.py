"""PyQt app to inspect cropped images with INP ONNX inference (Original/Heatmap/Binary)."""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List

# ── DLL path setup for NVIDIA libraries (must happen before any imports) ──
_ENV_ROOT = os.path.dirname(os.path.dirname(sys.executable))
_SITE_PACKAGES = os.path.join(_ENV_ROOT, "Lib", "site-packages")
for _nv_dir in (
    os.path.join(_SITE_PACKAGES, "nvidia", "cudnn", "bin"),
    os.path.join(_SITE_PACKAGES, "nvidia", "cublas", "bin"),
    os.path.join(_SITE_PACKAGES, "nvidia", "cuda_runtime", "bin"),
):
    if os.path.isdir(_nv_dir):
        os.environ["PATH"] = _nv_dir + os.pathsep + os.environ["PATH"]

# ⚠  IMPORTANT: onnxruntime must be imported BEFORE PyQt5.
# PyQt5 loads Qt DLLs that can interfere with ORT's DLL initialization.
# Ensure onnxruntime is fully loaded first so its DLLs resolve correctly.
import onnxruntime  # noqa: E402

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import ConfigError, load_config
from app.models.anomaly import AnomalyDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _detect_ort_info() -> dict:
    info = {
        "python_exe": sys.executable,
        "python_version": sys.version.split()[0],
        "ort_version": "N/A",
        "ort_type": "N/A",
        "ort_file": "N/A",
        "providers": "N/A",
        "import_ok": False,
        "import_error": "",
    }
    try:
        import onnxruntime as ort
        info["ort_version"] = ort.__version__
        info["ort_file"] = getattr(ort, "__file__", "N/A")
        raw_providers = ort.get_available_providers()
        info["providers"] = ", ".join(raw_providers)
        info["import_ok"] = True
        if "CUDAExecutionProvider" in raw_providers:
            info["ort_type"] = "onnxruntime-gpu"
        elif "CPUExecutionProvider" in raw_providers:
            info["ort_type"] = "onnxruntime (CPU)"
    except Exception as exc:
        info["import_error"] = f"{type(exc).__name__}: {exc}"
        info["ort_type"] = "FAILED TO IMPORT"
        try:
            import pkg_resources
            for dist in pkg_resources.working_set:
                name = dist.project_name.lower()
                if "onnxruntime" in name:
                    info["ort_type"] = f"INSTALLED: {dist.project_name}=={dist.version}"
                    info["ort_version"] = str(dist.version)
                    info["ort_file"] = str(dist.location)
                    break
        except Exception:
            pass
    return info


def _collect_cuda_diagnostic() -> dict:
    diag = {"cuda_path_env": os.environ.get("CUDA_PATH", "not set"),
             "cuda_118_bin": os.path.isdir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"),
             "cuda_12x_dirs": []}
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        for entry in os.listdir(base):
            ver_path = os.path.join(base, entry)
            if os.path.isdir(ver_path) and os.path.isdir(os.path.join(ver_path, "bin")):
                diag["cuda_12x_dirs"].append(entry)
    return diag


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
        self.resize(1400, 820)

        self.config_path = QtWidgets.QLineEdit("config.yaml")
        self.model_path = QtWidgets.QLineEdit(
            "D:/DataMea14/NGUYEN/NQBT/app/models/anomaly.onnx"
        )
        self.input_dir = QtWidgets.QLineEdit(
            "D:/DataMea14/NGUYEN/INP/mvtec_anomaly_detection/nqbt/test/bad"
        )
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems(["CPU", "CUDA", "Auto"])
        self.provider_combo.setCurrentText("CPU")
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(-1e9, 1e9)
        self.threshold.setValue(0.056)
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(28)
        self.bin_threshold = QtWidgets.QDoubleSpinBox()
        self.bin_threshold.setDecimals(6)
        self.bin_threshold.setRange(0, 1)
        self.bin_threshold.setSingleStep(0.01)
        self.bin_threshold.setValue(0.8)
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

        self.debug_label = QtWidgets.QLabel("Diagnostics")
        self.debug_label.setStyleSheet("font-weight:700; color:#0a0;")
        self.debug_text = QtWidgets.QPlainTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumBlockCount(100)
        self.debug_text.setStyleSheet("font-family:Consolas; font-size:11px;")

        self.btn_refresh_diag = QtWidgets.QPushButton("Refresh Diagnostics")
        self.btn_refresh_diag.clicked.connect(self._refresh_diagnostics)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(self._row("Config", self.config_path, self._btn(self.pick_config)))
        root.addLayout(self._row("Model ONNX", self.model_path, self._btn(self.pick_model)))
        root.addLayout(self._row("Input Folder", self.input_dir, self._btn(self.pick_input_dir)))
        root.addLayout(self._params_row())
        root.addWidget(self.btn_run)

        body = QtWidgets.QHBoxLayout()
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(self.list_widget, 3)

        diag_header = QtWidgets.QHBoxLayout()
        diag_header.addWidget(self.debug_label, 1)
        diag_header.addWidget(self.btn_refresh_diag)
        left_panel.addLayout(diag_header)
        left_panel.addWidget(self.debug_text, 1)
        body.addLayout(left_panel, 2)

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
        self._refresh_diagnostics()

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
        row.addWidget(QtWidgets.QLabel("Provider"))
        row.addWidget(self.provider_combo)
        row.addStretch(1)
        return row

    def _refresh_diagnostics(self):
        self.debug_text.clear()
        ort_info = _detect_ort_info()
        cuda_diag = _collect_cuda_diagnostic()

        lines = [
            "=== SYSTEM ===",
            f"Python executable : {ort_info['python_exe']}",
            f"Python version    : {ort_info['python_version']}",
            "",
            "=== ONNXRUNTIME ===",
            f"Type              : {ort_info['ort_type']}",
            f"Version           : {ort_info['ort_version']}",
            f"Location          : {ort_info['ort_file']}",
            f"Available providers: {ort_info['providers']}",
            f"Import OK         : {ort_info['import_ok']}",
        ]
        if ort_info["import_error"]:
            lines.append(f"Import error      : {ort_info['import_error']}")

        lines.extend([
            "",
            "=== CUDA ENVIRONMENT ===",
            f"CUDA_PATH env      : {cuda_diag['cuda_path_env']}",
            f"CUDA v11.8 bin     : {'FOUND' if cuda_diag['cuda_118_bin'] else 'NOT FOUND'}",
            f"CUDA 12.x dirs     : {', '.join(cuda_diag['cuda_12x_dirs']) if cuda_diag['cuda_12x_dirs'] else 'none'}",
            "",
            f"Provider selected  : {self.provider_combo.currentText()}",
            f"Site-packages      : {_SITE_PACKAGES}",
        ])

        for line in lines:
            self.debug_text.appendPlainText(line)
        self.debug_text.verticalScrollBar().setValue(0)

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
            provider_choice = self.provider_combo.currentText()
            cfg = load_config(self.config_path.text().strip())
            cfg.models.algo = "INP"
            cfg.models.inp.path = self.model_path.text().strip()
            cfg.models.inp.inp_mode = cfg.models.inp.inp_mode or "legacy_script"

            self._log(f"Provider mode: {provider_choice}")
            if provider_choice == "CPU":
                cfg.models.inp.provider = "cpu"
            elif provider_choice == "CUDA":
                cfg.models.inp.provider = "cuda"
            else:
                cfg.models.inp.provider = "cuda"

            self._log(f"Loading model: {cfg.models.inp.path}")
            detector = self._create_detector(cfg)

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
            ng_count = sum(1 for r in self.results if r.status == "NG")
            ok_count = len(self.results) - ng_count
            self._log(f"Done.  OK: {ok_count}  |  NG: {ng_count}  (threshold={self.threshold.value():.6f})")
        except (RuntimeError, ConfigError) as exc:
            detail_msg = str(exc)
            if "onnxruntime" in detail_msg.lower() or "dll" in detail_msg.lower():
                detail_msg += f"\n\nCurrent provider: {self.provider_combo.currentText()}\nTry switching to 'CPU' provider."
            QtWidgets.QMessageBox.critical(self, "Error", detail_msg)
            self._log(f"[ERROR] {exc}")
        except Exception as exc:
            detail_msg = f"{type(exc).__name__}: {exc}"
            self._log(f"[UNEXPECTED ERROR] {detail_msg}")
            self._log(traceback.format_exc())
        finally:
            self.btn_run.setEnabled(True)

    def _create_detector(self, cfg):
        try:
            return AnomalyDetector(cfg.models)
        except RuntimeError as exc:
            detail = str(exc)
            provider = self.provider_combo.currentText()
            if provider in ("Auto", "CUDA") and ("cuda" in detail.lower() or "dll" in detail.lower() or "onnxruntime" in detail.lower()):
                self._log("[WARN] CUDA failed, retrying with CPU provider...")
                self._log(f"  {detail}")
                cfg.models.inp.provider = "cpu"
                self.provider_combo.setCurrentText("CPU")
                self._refresh_diagnostics()
                return AnomalyDetector(cfg.models)
            raise

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

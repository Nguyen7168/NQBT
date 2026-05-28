"""Simple PyQt5 app to run anomaly inference on pre-cropped images and split by threshold."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

import cv2
import numpy as np
from PyQt5 import QtWidgets

from app.config_loader import ConfigError, load_config
from app.models.anomaly import AnomalyDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ThresholdSorterApp(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Anomaly Threshold Sorter")
        self.resize(760, 460)

        self.config_path = QtWidgets.QLineEdit("config.yaml")
        self.model_path = QtWidgets.QLineEdit("D:/DataMea14/NGUYEN/INP/ckpt_epoch_0214.onnx")
        self.input_dir = QtWidgets.QLineEdit("D:/DataMea14/NGUYEN/INP/CROP/CROP_OUT")
        self.output_dir = QtWidgets.QLineEdit("D:/DataMea14/NGUYEN/INP/CROP/CROP_OUT")

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setRange(-1e9, 1e9)
        self.threshold_spin.setValue(0.062)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(48)
        self.gpu_strict_chk = QtWidgets.QCheckBox("INP GPU strict (no CPU fallback)")
        self.gpu_strict_chk.setChecked(True)

        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(["Auto", "INP", "GLASS"])

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)

        self.btn_run = QtWidgets.QPushButton("Infer & Split")
        self.btn_run.clicked.connect(self.run_inference)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self._row("Config", self.config_path, self._btn("Browse", self.pick_config)))
        layout.addLayout(self._row("Model", self.model_path, self._btn("Browse", self.pick_model)))
        layout.addLayout(self._row("Input Folder", self.input_dir, self._btn("Browse", self.pick_input_dir)))
        layout.addLayout(self._row("Output Root", self.output_dir, self._btn("Browse", self.pick_output_dir)))
        layout.addLayout(self._algo_row())
        layout.addLayout(self._threshold_row())
        layout.addLayout(self._batch_row())
        layout.addWidget(self.gpu_strict_chk)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.log_box)

    def _btn(self, title: str, slot):
        btn = QtWidgets.QPushButton(title)
        btn.clicked.connect(slot)
        return btn

    def _row(
        self,
        label: str,
        editor: QtWidgets.QLineEdit,
        btn: QtWidgets.QPushButton,
    ):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addWidget(editor, 1)
        row.addWidget(btn)
        return row

    def _algo_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Algo"))
        row.addWidget(self.algo_combo, 1)
        return row

    def _threshold_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Threshold (>= upper)"))
        row.addWidget(self.threshold_spin, 1)
        return row

    def _batch_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Batch Size"))
        row.addWidget(self.batch_spin, 1)
        return row

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(text)
        QtWidgets.QApplication.processEvents()

    def pick_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose config",
            self.config_path.text(),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.config_path.setText(path)

    def pick_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose model",
            self.model_path.text(),
            "Model Files (*.onnx *.pth *.pt);;All Files (*)",
        )
        if path:
            self.model_path.setText(path)

    def pick_input_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose input folder",
            self.input_dir.text(),
        )
        if path:
            self.input_dir.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(path)

    def pick_output_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            self.output_dir.text(),
        )
        if path:
            self.output_dir.setText(path)

    def _collect_images(self, root: Path) -> List[Path]:
        return sorted(
            [
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        )

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
            batch_size = int(self.batch_spin.value())
            gpu_strict = bool(self.gpu_strict_chk.isChecked())

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

            detector = None if (algo == "INP" and gpu_strict) else AnomalyDetector(config.models)
            strict_ctx = None
            if algo == "INP" and gpu_strict:
                strict_ctx = self._build_onnx_gpu_strict_ctx(config.models.inp)

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
            self.log(f"Batch size: {batch_size}")
            self.log(f"GPU strict (INP): {'ON' if gpu_strict else 'OFF'}")

            total = len(images)
            processed = 0
            begin = perf_counter()
            for start in range(0, total, batch_size):
                batch_paths = images[start : start + batch_size]
                valid_paths: List[Path] = []
                batch_images: List = []
                for img_path in batch_paths:
                    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if image is None:
                        self.log(f"[WARN] Skip unreadable image: {img_path}")
                        continue
                    valid_paths.append(img_path)
                    batch_images.append(image)

                if not batch_images:
                    continue

                if algo == "INP" and gpu_strict:
                    assert strict_ctx is not None
                    batch_scores = self._infer_scores_onnx_gpu_strict_ctx(strict_ctx, batch_images)
                else:
                    assert detector is not None
                    batch_scores = self._infer_scores_fast(detector, batch_images)
                for img_path, score in zip(valid_paths, batch_scores):
                    score = float(score)
                    scores.append(score)
                    if score >= threshold:
                        dst = upper_dir / img_path.name
                        upper_count += 1
                    else:
                        dst = under_dir / img_path.name
                        under_count += 1
                    dst = self._unique_path(dst)
                    shutil.copy2(img_path, dst)
                    processed += 1

                self.log(
                    f"[{processed}/{total}] batch={len(batch_images)} "
                    f"last={valid_paths[-1].name}"
                )

            elapsed_s = perf_counter() - begin
            if processed > 0:
                self.log(f"Throughput: {processed / max(elapsed_s, 1e-8):.2f} img/s")

            self.log("---- DONE ----")

            if scores:
                mean_score = sum(scores) / len(scores)
                self.log(
                    f"Score min/max/mean: "
                    f"{min(scores):.6f} / {max(scores):.6f} / {mean_score:.6f}"
                )

            self.log(f"upper threshold: {upper_count}")
            self.log(f"under threshold: {under_count}")

            QtWidgets.QMessageBox.information(
                self,
                "Finished",
                "Inference and split completed.",
            )

        except (RuntimeError, ConfigError) as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            self.log(f"[ERROR] {exc}")

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Unexpected Error", str(exc))
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

    @staticmethod
    def _infer_scores_fast(detector: AnomalyDetector, patches: Sequence) -> List[float]:
        impl = getattr(detector, "_impl", None)
        session = getattr(impl, "_session", None)
        input_name = getattr(impl, "_input_name", None)
        output_names = getattr(impl, "_output_names", None)
        input_hw = getattr(impl, "_input_hw", None)
        mode = str(getattr(impl, "_mode", "")).strip().lower()

        can_use_local_batch = (
            session is not None
            and input_name
            and output_names
            and input_hw
            and mode == "default"
            and hasattr(impl, "_preprocess")
        )
        if not can_use_local_batch:
            result = detector.infer(patches)
            return [float(s) for s in result.scores]

        try:
            blobs = [impl._preprocess(patch)[0] for patch in patches]
            batch_blob = np.stack(blobs, axis=0).astype(np.float32)
            outputs = session.run(output_names, {input_name: batch_blob})
            if len(outputs) < 4:
                result = detector.infer(patches)
                return [float(s) for s in result.scores]
            fs_list = [np.asarray(outputs[0]), np.asarray(outputs[1])]
            ft_list = [np.asarray(outputs[2]), np.asarray(outputs[3])]
            h, w = input_hw
            blur_k = int(getattr(impl, "_blur_k", 5))
            blur_sigma = float(getattr(impl, "_blur_sigma", 4.0))
            max_ratio = float(getattr(impl, "_max_ratio", 0.01))
            scores: List[float] = []
            for i in range(batch_blob.shape[0]):
                a_maps = []
                for fs, ft in zip(fs_list, ft_list):
                    fs0, ft0 = fs[i], ft[i]
                    c, fh, fw = fs0.shape
                    fs_flat = fs0.reshape(c, -1).T
                    ft_flat = ft0.reshape(c, -1).T
                    eps = 1e-8
                    fs_norm = np.linalg.norm(fs_flat, axis=1, keepdims=True).clip(min=eps)
                    ft_norm = np.linalg.norm(ft_flat, axis=1, keepdims=True).clip(min=eps)
                    sim = np.sum(fs_flat * ft_flat, axis=1, keepdims=True) / (fs_norm * ft_norm)
                    dist = (1.0 - sim).reshape(fh, fw)
                    dist_resized = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)
                    a_maps.append(dist_resized[None, None, ...])
                concat_maps = np.concatenate(a_maps, axis=1)
                anomaly_map = np.mean(concat_maps, axis=1, keepdims=True)
                amap = anomaly_map[0, 0]
                amap = cv2.GaussianBlur(amap, (blur_k, blur_k), blur_sigma)
                a_min, a_max = float(np.min(amap)), float(np.max(amap))
                norm = (amap - a_min) / (a_max - a_min + 1e-8)
                if max_ratio and max_ratio > 0.0:
                    flat = np.sort(norm.ravel())
                    k = max(1, int(flat.shape[0] * max_ratio))
                    score = float(np.mean(flat[-k:]))
                else:
                    score = float(np.max(norm))
                scores.append(score)
            return scores
        except Exception:
            result = detector.infer(patches)
            return [float(s) for s in result.scores]

    @staticmethod
    def _build_onnx_gpu_strict_ctx(inp_cfg):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(f"onnxruntime import failed in GPU strict mode: {type(exc).__name__}: {exc}") from exc

        providers = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"})]
        session = ort.InferenceSession(inp_cfg.path, providers=providers)
        active = session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"GPU strict mode requires CUDAExecutionProvider, got providers={active}")
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        ishape = session.get_inputs()[0].shape
        h = ishape[-2] if ishape and len(ishape) >= 4 and isinstance(ishape[-2], int) else int(inp_cfg.input_size)
        w = ishape[-1] if ishape and len(ishape) >= 4 and isinstance(ishape[-1], int) else int(inp_cfg.input_size)
        return {
            "session": session,
            "input_name": input_name,
            "output_names": output_names,
            "h": int(h),
            "w": int(w),
            "inp_cfg": inp_cfg,
            "mode": str(getattr(inp_cfg, "inp_mode", "default") or "default").strip().lower(),
        }

    @staticmethod
    def _infer_scores_onnx_gpu_strict_ctx(ctx, patches: Sequence[np.ndarray]) -> List[float]:
        session = ctx["session"]
        input_name = ctx["input_name"]
        output_names = ctx["output_names"]
        h = int(ctx["h"])
        w = int(ctx["w"])
        inp_cfg = ctx["inp_cfg"]
        mode = str(ctx["mode"])

        if mode == "legacy_script":
            return ThresholdSorterApp._infer_scores_onnx_gpu_strict_legacy(
                session=session,
                input_name=input_name,
                output_names=output_names,
                h=h,
                w=w,
                inp_cfg=inp_cfg,
                patches=patches,
            )
        return ThresholdSorterApp._infer_scores_onnx_gpu_strict_default(
            session=session,
            input_name=input_name,
            output_names=output_names,
            h=h,
            w=w,
            inp_cfg=inp_cfg,
            patches=patches,
        )

    @staticmethod
    def _infer_scores_onnx_gpu_strict_default(*, session, input_name, output_names, h: int, w: int, inp_cfg, patches: Sequence[np.ndarray]) -> List[float]:
        
        def _pre(patch: np.ndarray) -> np.ndarray:
            resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                resized = np.expand_dims(resized, axis=-1)
            if resized.shape[2] == 1:
                resized = np.repeat(resized, 3, axis=2)
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            return np.transpose(normalized, (2, 0, 1))

        batch_blob = np.stack([_pre(p) for p in patches], axis=0).astype(np.float32)
        outputs = session.run(output_names, {input_name: batch_blob})
        if len(outputs) < 4:
            raise RuntimeError("INP model must have >=4 outputs (encoder/decoder features)")
        fs_list = [np.asarray(outputs[0]), np.asarray(outputs[1])]
        ft_list = [np.asarray(outputs[2]), np.asarray(outputs[3])]
        blur_k = int(inp_cfg.inp_blur_kernel)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_sigma = float(inp_cfg.inp_blur_sigma)
        max_ratio = float(inp_cfg.inp_max_ratio)
        scores: List[float] = []
        for i in range(batch_blob.shape[0]):
            a_maps = []
            for fs, ft in zip(fs_list, ft_list):
                fs0, ft0 = fs[i], ft[i]
                c, fh, fw = fs0.shape
                fs_flat = fs0.reshape(c, -1).T
                ft_flat = ft0.reshape(c, -1).T
                eps = 1e-8
                sim = np.sum(fs_flat * ft_flat, axis=1, keepdims=True) / (
                    np.linalg.norm(fs_flat, axis=1, keepdims=True).clip(min=eps)
                    * np.linalg.norm(ft_flat, axis=1, keepdims=True).clip(min=eps)
                )
                dist = (1.0 - sim).reshape(fh, fw)
                a_maps.append(cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)[None, None, ...])
            amap = np.mean(np.concatenate(a_maps, axis=1), axis=1, keepdims=True)[0, 0]
            amap = cv2.GaussianBlur(amap, (blur_k, blur_k), blur_sigma)
            a_min, a_max = float(np.min(amap)), float(np.max(amap))
            norm = (amap - a_min) / (a_max - a_min + 1e-8)
            if max_ratio > 0.0:
                flat = np.sort(norm.ravel())
                k = max(1, int(flat.shape[0] * max_ratio))
                scores.append(float(np.mean(flat[-k:])))
            else:
                scores.append(float(np.max(norm)))
        return scores

    @staticmethod
    def _infer_scores_onnx_gpu_strict_legacy(*, session, input_name, output_names, h: int, w: int, inp_cfg, patches: Sequence[np.ndarray]) -> List[float]:
        def _pre_legacy(patch: np.ndarray) -> np.ndarray:
            bgr = patch
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            elif bgr.shape[2] == 1:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            return np.transpose(normalized, (2, 0, 1))

        blur_k = int(inp_cfg.inp_blur_kernel)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_sigma = float(inp_cfg.inp_blur_sigma)
        max_ratio = float(inp_cfg.inp_max_ratio)
        out_size = (h, w)
        legacy_map_size = 256
        # kernel equivalent to app.models.anomaly _legacy_gaussian_kernel
        x_coord = np.arange(blur_k)
        x_grid = np.repeat(x_coord, blur_k).reshape(blur_k, blur_k)
        y_grid = x_grid.T
        xy_grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
        mean_xy = (blur_k - 1) / 2.0
        variance = blur_sigma ** 2
        legacy_kernel = (1.0 / (2.0 * np.pi * variance)) * np.exp(
            -np.sum((xy_grid - mean_xy) ** 2.0, axis=-1) / (2.0 * variance)
        )
        legacy_kernel = legacy_kernel / np.sum(legacy_kernel)

        batch_blob = np.stack([_pre_legacy(p) for p in patches], axis=0).astype(np.float32)
        outputs = session.run(output_names, {input_name: batch_blob})
        if len(outputs) < 4:
            raise RuntimeError("INP model must have >=4 outputs (encoder/decoder features)")
        fs_list = [np.asarray(outputs[0], dtype=np.float32), np.asarray(outputs[1], dtype=np.float32)]
        ft_list = [np.asarray(outputs[2], dtype=np.float32), np.asarray(outputs[3], dtype=np.float32)]
        scores: List[float] = []
        for i in range(batch_blob.shape[0]):
            a_maps = []
            for fs, ft in zip(fs_list, ft_list):
                fs_arr = fs[i : i + 1]
                ft_arr = ft[i : i + 1]
                fs_norm = np.linalg.norm(fs_arr, axis=1, keepdims=True).clip(min=1e-8)
                ft_norm = np.linalg.norm(ft_arr, axis=1, keepdims=True).clip(min=1e-8)
                sim = np.sum(fs_arr * ft_arr, axis=1, keepdims=True) / (fs_norm * ft_norm)
                a_map = np.round(1.0 - sim, decimals=4)
                a_map = np.squeeze(a_map)
                in_h, in_w = a_map.shape[-2:]
                x_indices = np.linspace(0, in_w - 1, out_size[1]).astype(np.float32)
                y_indices = np.linspace(0, in_h - 1, out_size[0]).astype(np.float32)
                map_x, map_y = np.meshgrid(x_indices, y_indices)
                a_map = cv2.remap(
                    a_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                a_maps.append(np.expand_dims(np.expand_dims(a_map, axis=0), axis=0))
            anomaly_map = np.round(np.mean(np.concatenate(a_maps, axis=1), axis=1, keepdims=True), decimals=4)
            resized_images = np.zeros((1, 1, legacy_map_size, legacy_map_size), dtype=anomaly_map.dtype)
            resized_images[0, 0] = cv2.resize(
                anomaly_map[0, 0], (legacy_map_size, legacy_map_size), interpolation=cv2.INTER_LINEAR
            )
            amap_raw = resized_images[0, 0]
            amap_raw = cv2.filter2D(amap_raw, -1, legacy_kernel, borderType=cv2.BORDER_CONSTANT)
            amap_raw = np.round(amap_raw, decimals=4)
            if max_ratio == 0:
                score = float(np.max(amap_raw.ravel()))
            else:
                flat = amap_raw.ravel()
                k = max(1, int(flat.shape[0] * max_ratio))
                score = float(np.sort(flat)[-k:].mean())
            scores.append(score)
        return scores


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = ThresholdSorterApp()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

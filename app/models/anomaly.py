"""ONNX based anomaly detection wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore

from app.config_loader import ModelConfig, InpModelConfig, GlassModelConfig
from app.models.glass_torch import GLASSInfer, preprocess_patch

LOGGER = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    scores: List[float]
    inference_ms: float
    maps: List[np.ndarray] | None = None  # Optional per-patch normalized maps (0..1)


class _BaseDetector:
    def __init__(self, config):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self._config = config
        providers = self._build_providers(config.provider)
        LOGGER.info("Loading anomaly model %s with providers %s", config.path, providers)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(config.path, sess_options=session_options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    @staticmethod
    def _build_providers(provider_name: str) -> Sequence[str | tuple[str, dict[str, str]]]:
        provider = provider_name.lower()
        if provider == "cuda":
            return [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]


class _InpOnnxDetector(_BaseDetector):
    def __init__(self, config: InpModelConfig):
        super().__init__(config)
        # Infer expected HxW from model; fallback to config
        try:
            ishape = self._session.get_inputs()[0].shape  # e.g. [1,3,H,W]
            h, w = None, None
            if ishape and len(ishape) >= 4:
                h = ishape[-2] if isinstance(ishape[-2], int) else None
                w = ishape[-1] if isinstance(ishape[-1], int) else None
            self._input_hw = (
                (int(h), int(w)) if (h is not None and w is not None) else (config.input_size, config.input_size)
            )
        except Exception:
            self._input_hw = (config.input_size, config.input_size)
        # Output name: prefer first
        self._output_name = self._output_names[0]
        # INP-specific params
        self._blur_k = int(config.inp_blur_kernel)
        if self._blur_k % 2 == 0:
            self._blur_k += 1
        self._blur_sigma = float(config.inp_blur_sigma)
        self._max_ratio = float(config.inp_max_ratio)
        self._bin_thresh = float(config.inp_bin_thresh)
        self._mode = str(getattr(config, "inp_mode", "default") or "default").strip().lower()
        self._legacy_map_size = 256

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        if self._mode == "legacy_script":
            return self._infer_legacy_script(patches)
        return self._infer_default(patches)

    def _infer_default(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        start = perf_counter()
        scores: List[float] = []
        maps: List[np.ndarray] = []
        for patch in patches:
            blob = self._preprocess(patch)
            # Expect multiple outputs (encoder/decoder features) for INP map computation
            outputs = self._session.run(self._output_names, {self._input_name: blob})
            if len(outputs) < 4:
                raise RuntimeError("INP model must have >=4 outputs (encoder/decoder features)")
            fs_list = outputs[0:2]
            ft_list = outputs[2:4]
            a_maps = []
            for fs, ft in zip(fs_list, ft_list):
                fs = np.asarray(fs)
                ft = np.asarray(ft)
                if fs.ndim != 4 or ft.ndim != 4:
                    raise RuntimeError("INP feature tensors must be 4D (B,C,H,W)")
                fs0, ft0 = fs[0], ft[0]  # (C,H,W)
                C, H, W = fs0.shape
                fs_flat = fs0.reshape(C, -1).T  # (H*W, C)
                ft_flat = ft0.reshape(C, -1).T
                eps = 1e-8
                fs_norm = np.linalg.norm(fs_flat, axis=1, keepdims=True).clip(min=eps)
                ft_norm = np.linalg.norm(ft_flat, axis=1, keepdims=True).clip(min=eps)
                sim = np.sum(fs_flat * ft_flat, axis=1, keepdims=True) / (fs_norm * ft_norm)
                dist = (1.0 - sim).reshape(H, W)
                h, w = self._input_hw
                dist_resized = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)
                a_maps.append(dist_resized[None, None, ...])
            concat_maps = np.concatenate(a_maps, axis=1)  # (1,N,H,W)
            anomaly_map = np.mean(concat_maps, axis=1, keepdims=True)  # (1,1,H,W)
            amap = anomaly_map[0, 0]
            # Blur and normalize
            amap = cv2.GaussianBlur(amap, (self._blur_k, self._blur_k), self._blur_sigma)
            a_min, a_max = float(np.min(amap)), float(np.max(amap))
            norm = (amap - a_min) / (a_max - a_min + 1e-8)
            # Score aggregation
            if self._max_ratio and self._max_ratio > 0.0:
                flat = np.sort(norm.ravel())
                k = max(1, int(flat.shape[0] * self._max_ratio))
                score = float(np.mean(flat[-k:]))
            else:
                score = float(np.max(norm))
            scores.append(score)
            maps.append(norm.astype(np.float32))
        elapsed = (perf_counter() - start) * 1000.0
        LOGGER.debug("Anomaly (INP) inference finished in %.2f ms", elapsed)
        return AnomalyResult(scores=scores, inference_ms=elapsed, maps=maps)

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        h, w = self._input_hw
        resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        if resized.shape[2] == 1:
            resized = np.repeat(resized, 3, axis=2)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        tensor = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def _preprocess_legacy(self, patch: np.ndarray) -> np.ndarray:
        h, w = self._input_hw
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        elif patch.shape[2] == 1:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        tensor = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    @staticmethod
    def _legacy_resize_with_align_corners(image: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        in_height, in_width = image.shape[-2:]
        out_height, out_width = out_size
        x_indices = np.linspace(0, in_width - 1, out_width).astype(np.float32)
        y_indices = np.linspace(0, in_height - 1, out_height).astype(np.float32)
        map_x, map_y = np.meshgrid(x_indices, y_indices)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def _legacy_resize_without_align_corners(image: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        batch_size, channels, _, _ = image.shape
        out_height, out_width = out_size
        resized_images = np.zeros((batch_size, channels, out_height, out_width), dtype=image.dtype)
        for b in range(batch_size):
            for c in range(channels):
                resized_images[b, c] = cv2.resize(image[b, c], (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        return resized_images

    @staticmethod
    def _legacy_gaussian_kernel(kernel_size: int = 5, sigma: float = 4.0) -> np.ndarray:
        x_coord = np.arange(kernel_size)
        x_grid = np.repeat(x_coord, kernel_size).reshape(kernel_size, kernel_size)
        y_grid = x_grid.T
        xy_grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2
        kernel = (1.0 / (2.0 * np.pi * variance)) * np.exp(-np.sum((xy_grid - mean) ** 2.0, axis=-1) / (2.0 * variance))
        kernel = kernel / np.sum(kernel)
        return kernel.reshape(kernel_size, kernel_size)

    def _infer_legacy_script(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        start = perf_counter()
        scores: List[float] = []
        maps: List[np.ndarray] = []
        out_size = self._input_hw
        legacy_kernel = self._legacy_gaussian_kernel(kernel_size=self._blur_k, sigma=self._blur_sigma)
        for patch in patches:
            blob = self._preprocess_legacy(patch)
            outputs = self._session.run(self._output_names, {self._input_name: blob})
            if len(outputs) < 4:
                raise RuntimeError("INP model must have >=4 outputs (encoder/decoder features)")
            fs_list = outputs[0:2]
            ft_list = outputs[2:4]
            a_maps: List[np.ndarray] = []
            for fs, ft in zip(fs_list, ft_list):
                fs_arr = np.asarray(fs, dtype=np.float32)
                ft_arr = np.asarray(ft, dtype=np.float32)
                fs_norm = np.linalg.norm(fs_arr, axis=1, keepdims=True).clip(min=1e-8)
                ft_norm = np.linalg.norm(ft_arr, axis=1, keepdims=True).clip(min=1e-8)
                sim = np.sum(fs_arr * ft_arr, axis=1, keepdims=True) / (fs_norm * ft_norm)
                a_map = np.round(1.0 - sim, decimals=4)
                a_map = np.squeeze(a_map)
                a_map = self._legacy_resize_with_align_corners(a_map, out_size)
                a_maps.append(np.expand_dims(np.expand_dims(a_map, axis=0), axis=0))
            anomaly_map = np.round(np.mean(np.concatenate(a_maps, axis=1), axis=1, keepdims=True), decimals=4)
            anomaly_map = self._legacy_resize_without_align_corners(anomaly_map, (self._legacy_map_size, self._legacy_map_size))
            amap_raw = anomaly_map[0, 0]
            amap_raw = cv2.filter2D(amap_raw, -1, legacy_kernel, borderType=cv2.BORDER_CONSTANT)
            amap_raw = np.round(amap_raw, decimals=4)
            if self._max_ratio == 0:
                score = float(np.max(amap_raw.ravel()))
            else:
                flat = amap_raw.ravel()
                k = max(1, int(flat.shape[0] * self._max_ratio))
                score = float(np.sort(flat)[-k:].mean())
            # Keep app map contract as normalized float32 for UI saving/visualization.
            a_min, a_max = float(np.min(amap_raw)), float(np.max(amap_raw))
            norm = (amap_raw - a_min) / (a_max - a_min + 1e-8)
            scores.append(score)
            maps.append(norm.astype(np.float32))
        elapsed = (perf_counter() - start) * 1000.0
        LOGGER.debug("Anomaly (INP legacy_script) inference finished in %.2f ms", elapsed)
        return AnomalyResult(scores=scores, inference_ms=elapsed, maps=maps)


class _GlassTorchDetector:
    def __init__(self, config: GlassModelConfig):
        provider = (config.provider or "cuda").lower()
        if provider.startswith("cuda") and torch.cuda.is_available():
            device = provider
        else:
            device = "cpu"
        self._device = torch.device(device)
        self._size = int(config.input_size)
        self._model = GLASSInfer(
            device=device,
            backbone_name="wideresnet50",
            layers_to_extract_from=("layer2", "layer3"),
            input_shape=(3, self._size, self._size),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            patchstride=1,
            dsc_layers=2,
            dsc_hidden=1024,
            pre_proj=1,
        )
        self._model.load_checkpoint(config.path)

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        if not patches:
            return AnomalyResult(scores=[], inference_ms=0.0, maps=[])
        tensors = [preprocess_patch(patch, self._size) for patch in patches]
        batch = torch.stack(tensors).to(self._device)
        scores, masks, elapsed_ms = self._model.infer_batch(batch)
        maps = [mask.astype(np.float32) for mask in masks]
        return AnomalyResult(scores=scores, inference_ms=elapsed_ms, maps=maps)


class AnomalyDetector:
    def __init__(self, models: ModelConfig):
        algo = str(models.algo or "INP").strip().upper()
        if algo == "GLASS":
            self._impl = _GlassTorchDetector(models.glass)
        elif algo == "INP":
            self._impl = _InpOnnxDetector(models.inp)
        else:
            raise RuntimeError(f"Unsupported models.algo: {algo}")

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        return self._impl.infer(patches)

"""ONNX based anomaly detection wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore

from app.config_loader import AnomalyModelConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    scores: List[float]
    inference_ms: float
    maps: List[np.ndarray] | None = None  # Optional per-patch normalized maps (0..1)


class _BaseDetector:
    def __init__(self, config: AnomalyModelConfig):
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
    def __init__(self, config: AnomalyModelConfig):
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
        self._blur_k = int(getattr(config, "inp_blur_kernel", 5))
        if self._blur_k % 2 == 0:
            self._blur_k += 1
        self._blur_sigma = float(getattr(config, "inp_blur_sigma", 4.0))
        self._max_ratio = float(getattr(config, "inp_max_ratio", 0.0))
        self._bin_thresh = float(getattr(config, "inp_bin_thresh", 0.2))

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
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


class _GlassOnnxDetector(_BaseDetector):
    def __init__(self, config: AnomalyModelConfig):
        super().__init__(config)
        self._size = int(config.input_size)
        self._batch = max(1, int(getattr(config, "glass_batch", 8)))
        self._blur_k = int(getattr(config, "glass_blur_kernel", 33))
        # kernel must be odd
        if self._blur_k % 2 == 0:
            self._blur_k += 1
        self._blur_sigma = float(getattr(config, "glass_blur_sigma", 4.0))
        self._norm_eps = float(getattr(config, "glass_norm_eps", 1e-8))

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        start = perf_counter()
        scores: List[float] = []
        maps: List[np.ndarray] = []
        for patch in patches:
            inp = self._preprocess(patch)  # (1,3,H,W)
            inp = np.repeat(inp, self._batch, axis=0)  # (B,3,H,W)
            outputs = self._session.run(None, {self._input_name: inp})
            amap = np.asarray(outputs[0])  # expect (B,H,W)
            if amap.ndim == 4:  # some models might return (B,1,H,W)
                amap = amap[:, 0]
            amap0 = amap[0]
            amap0 = cv2.resize(amap0, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
            amap0 = cv2.GaussianBlur(amap0, (self._blur_k, self._blur_k), self._blur_sigma)
            # min-max norm
            a_min, a_max = float(np.min(amap0)), float(np.max(amap0))
            norm = (amap0 - a_min) / (a_max - a_min + self._norm_eps)
            score = float(np.max(norm))
            scores.append(score)
            maps.append(norm.astype(np.float32))
        elapsed = (perf_counter() - start) * 1000.0
        LOGGER.debug("Anomaly (GLASS) inference finished in %.2f ms", elapsed)
        return AnomalyResult(scores=scores, inference_ms=elapsed, maps=maps)

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        img = patch
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        chw = np.transpose(img, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)


class AnomalyDetector:
    def __init__(self, config: AnomalyModelConfig):
        algo = (config.algo or "INP").upper()
        if algo == "GLASS":
            self._impl = _GlassOnnxDetector(config)
        else:
            self._impl = _InpOnnxDetector(config)

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        return self._impl.infer(patches)

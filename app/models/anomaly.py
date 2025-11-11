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


class AnomalyDetector:
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
        self._output_name = self._session.get_outputs()[0].name
        # Try to infer expected input HxW from the model; fall back to config
        try:
            ishape = self._session.get_inputs()[0].shape  # e.g. [1,3,H,W]
            h, w = None, None
            if ishape and len(ishape) >= 4:
                h = ishape[-2] if isinstance(ishape[-2], int) else None
                w = ishape[-1] if isinstance(ishape[-1], int) else None
            if h is not None and w is not None:
                self._input_hw = (int(h), int(w))
            else:
                self._input_hw = (config.input_size, config.input_size)
        except Exception:
            self._input_hw = (config.input_size, config.input_size)

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

    def infer(self, patches: Sequence[np.ndarray]) -> AnomalyResult:
        start = perf_counter()
        scores: List[float] = []
        for patch in patches:
            blob = self._preprocess(patch)
            outputs = self._session.run([self._output_name], {self._input_name: blob})
            arr = np.asarray(outputs[0]).squeeze()
            # Accept scalar, vector, or map outputs; reduce to a single score.
            if arr.ndim == 0:
                score = float(arr)
            else:
                score = float(np.max(arr))
            scores.append(score)
        elapsed = (perf_counter() - start) * 1000.0
        LOGGER.debug("Anomaly inference finished in %.2f ms", elapsed)
        return AnomalyResult(scores=scores, inference_ms=elapsed)

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

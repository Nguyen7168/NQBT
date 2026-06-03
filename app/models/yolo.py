"""Optional YOLO detector used for visual overlays only."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import List, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore


@dataclass
class YoloResult:
    boxes: List[np.ndarray]
    scores: List[float]
    class_ids: List[int]
    inference_ms: float


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int | None = None,
        device: str | None = None,
        classes: Sequence[int] | None = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self._model = YOLO(model_path)
        self._conf = conf_thres
        self._iou = iou_thres
        self._imgsz = imgsz
        self._device = (device or "").strip() or None
        self._classes = list(classes) if classes is not None else None
        self._device_failed_once = False

    def _predict(self, source: object):
        predict_kwargs = {
            "source": source,
            "conf": self._conf,
            "iou": self._iou,
            "verbose": False,
        }
        if self._imgsz:
            predict_kwargs["imgsz"] = int(self._imgsz)
        if self._classes is not None:
            predict_kwargs["classes"] = self._classes
        if self._device and not self._device_failed_once:
            predict_kwargs["device"] = self._device
        try:
            return self._model.predict(**predict_kwargs)
        except Exception:
            if self._device and not self._device_failed_once:
                self._device_failed_once = True
                LOGGER.warning("YOLO predict failed with device=%s, retrying with default device", self._device)
                predict_kwargs.pop("device", None)
                return self._model.predict(**predict_kwargs)
            raise

    @staticmethod
    def _result_from_prediction(pred, inference_ms: float) -> YoloResult:
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        class_ids: List[int] = []
        if pred.boxes is not None:
            boxes.extend(pred.boxes.xyxy.cpu().numpy())
            scores.extend(pred.boxes.conf.cpu().numpy())
            class_ids.extend(pred.boxes.cls.cpu().numpy().astype(int))
        return YoloResult(boxes=boxes, scores=scores, class_ids=class_ids, inference_ms=inference_ms)

    def detect(self, image: np.ndarray) -> YoloResult:
        start = perf_counter()
        predictions = self._predict(image)
        elapsed = (perf_counter() - start) * 1000.0
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        class_ids: List[int] = []
        for pred in predictions:
            result = self._result_from_prediction(pred, elapsed)
            boxes.extend(result.boxes)
            scores.extend(result.scores)
            class_ids.extend(result.class_ids)
        return YoloResult(boxes=boxes, scores=scores, class_ids=class_ids, inference_ms=elapsed)

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[YoloResult]:
        if not images:
            return []
        start = perf_counter()
        predictions = self._predict(list(images))
        elapsed = (perf_counter() - start) * 1000.0
        per_image_ms = elapsed / max(len(images), 1)
        results = [self._result_from_prediction(pred, per_image_ms) for pred in predictions]
        if len(results) != len(images):
            LOGGER.warning("YOLO batch returned %d predictions for %d images", len(results), len(images))
        while len(results) < len(images):
            results.append(YoloResult(boxes=[], scores=[], class_ids=[], inference_ms=0.0))
        return results[: len(images)]

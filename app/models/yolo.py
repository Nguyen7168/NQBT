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
    def __init__(self, model_path: str, conf_thres: float, iou_thres: float) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self._model = YOLO(model_path)
        self._conf = conf_thres
        self._iou = iou_thres

    def detect(self, image: np.ndarray) -> YoloResult:
        start = perf_counter()
        predictions = self._model.predict(source=image, conf=self._conf, iou=self._iou, verbose=False)
        elapsed = (perf_counter() - start) * 1000.0
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        class_ids: List[int] = []
        for pred in predictions:
            if pred.boxes is None:
                continue
            boxes.extend(pred.boxes.xyxy.cpu().numpy())
            scores.extend(pred.boxes.conf.cpu().numpy())
            class_ids.extend(pred.boxes.cls.cpu().numpy().astype(int))
        return YoloResult(boxes=boxes, scores=scores, class_ids=class_ids, inference_ms=elapsed)

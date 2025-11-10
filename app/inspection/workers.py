"""Qt worker objects orchestrating camera capture, inference, and PLC handshake."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PyQt5 import QtCore

from app.config_loader import AppConfig
from app.inspection.camera import BaslerCamera, DummyCamera
from app.inspection.cropping import CropResult, GridCropper
from app.inspection.plc_client import PlcController
from app.models.anomaly import AnomalyDetector
from app.models.yolo import YoloDetector, YoloResult
from app.utils import ensure_dir, save_image

LOGGER = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    raw_image: np.ndarray
    overlay_image: np.ndarray
    patches: List[CropResult]
    anomaly_scores: List[float]
    statuses: List[str]
    ng_total: int
    anomaly_inference_ms: float
    yolo_result: Optional[YoloResult]
    timestamp: float
    model_path: str
    threshold: float

    def to_json(self) -> Dict[str, object]:
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return {
            "timestamp": ts,
            "model_anomaly": self.model_path,
            "threshold": self.threshold,
            "count": len(self.patches),
            "per_part": [
                {"idx": patch.index, "score": float(score), "status": status}
                for patch, score, status in zip(self.patches, self.anomaly_scores, self.statuses)
            ],
            "ng_total": self.ng_total,
            "inference_ms": self.anomaly_inference_ms,
            "yolo": None
            if self.yolo_result is None
            else {
                "boxes": [list(map(float, box)) for box in self.yolo_result.boxes],
                "scores": [float(s) for s in self.yolo_result.scores],
                "class_ids": [int(c) for c in self.yolo_result.class_ids],
                "inference_ms": self.yolo_result.inference_ms,
            },
        }


class InspectionWorker(QtCore.QObject):
    cycle_started = QtCore.pyqtSignal()
    cycle_completed = QtCore.pyqtSignal(InspectionResult)
    cycle_failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        config: AppConfig,
        plc: PlcController,
        parent: Optional[QtCore.QObject] = None,
        use_dummy_camera: bool = False,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.plc = plc
        self.cropper = GridCropper(config.layout)
        self.camera = DummyCamera(config.camera) if use_dummy_camera else BaslerCamera(config.camera)
        self.anomaly = AnomalyDetector(config.models.anomaly)
        self.yolo = None
        if config.models.yolo.enabled and config.models.yolo.path:
            try:
                self.yolo = YoloDetector(
                    config.models.yolo.path,
                    config.models.yolo.conf_thres,
                    config.models.yolo.iou_thres,
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialise YOLO detector: %s", exc)
        self._lock = threading.Lock()
        self._camera_ready = False

    @QtCore.pyqtSlot()
    def run_cycle(self) -> None:
        with self._lock:
            try:
                self.cycle_started.emit()
                self.plc.set_busy(True)
                if not self._camera_ready:
                    self.camera.connect()
                    self._camera_ready = True
                capture = self.camera.capture()
                LOGGER.debug("Captured image with shape %s", capture.image.shape)
                patches = self.cropper.crop(capture.image)
                anomaly = self.anomaly.infer([p.image for p in patches])
                threshold = self.config.models.anomaly.threshold
                statuses = ["OK" if score <= threshold else "NG" for score in anomaly.scores]
                ng_total = sum(1 for status in statuses if status == "NG")

                yolo_result = None
                if self.yolo is not None:
                    try:
                        yolo_result = self.yolo.detect(capture.image)
                    except Exception as exc:
                        LOGGER.error("YOLO inference failed: %s", exc)

                overlay = self._build_overlay(capture.image, patches, statuses, yolo_result)
                result = InspectionResult(
                    raw_image=capture.image,
                    overlay_image=overlay,
                    patches=patches,
                    anomaly_scores=anomaly.scores,
                    statuses=statuses,
                    ng_total=ng_total,
                    anomaly_inference_ms=anomaly.inference_ms,
                    yolo_result=yolo_result,
                    timestamp=time.time(),
                    model_path=self.config.models.anomaly.path,
                    threshold=threshold,
                )

                self.plc.write_results([status == "OK" for status in statuses])
                self.plc.set_error(False)
                self.plc.set_done(True)
                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Inspection cycle failed: %s", exc)
                try:
                    self.plc.write_results([False] * self.config.layout.count)
                    self.plc.set_error(True)
                    self.plc.set_done(True)
                finally:
                    self.cycle_failed.emit(str(exc))
            finally:
                self.plc.finalize_cycle()

    @QtCore.pyqtSlot()
    def shutdown(self) -> None:  # pragma: no cover - cleanup
        with self._lock:
            try:
                self.camera.disconnect()
            except Exception as exc:
                LOGGER.debug("Camera disconnect failed: %s", exc)

    @QtCore.pyqtSlot(str)
    def reload_anomaly_model(self, model_path: str) -> None:
        with self._lock:
            try:
                self.config.models.anomaly.path = model_path
                self.anomaly = AnomalyDetector(self.config.models.anomaly)
                LOGGER.info("Reloaded anomaly model from %s", model_path)
            except Exception as exc:
                LOGGER.error("Failed to reload anomaly model: %s", exc)
                raise

    def _build_overlay(
        self,
        image: np.ndarray,
        patches: Sequence[CropResult],
        statuses: Sequence[str],
        yolo_result: Optional[YoloResult],
    ) -> np.ndarray:
        overlay = image.copy()
        for patch, status in zip(patches, statuses):
            x1, y1, x2, y2 = patch.bbox
            color = (0, 255, 0) if status == "OK" else (0, 0, 255)
            overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        if yolo_result:
            for box, score, cls in zip(yolo_result.boxes, yolo_result.scores, yolo_result.class_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 215, 0), 2)
                cv2.putText(
                    overlay,
                    f"{cls}:{score:.2f}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 215, 0),
                    1,
                    cv2.LINE_AA,
                )
        return overlay


class SaveWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: AppConfig, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.config = config

    @QtCore.pyqtSlot(object)
    def save(self, result: object) -> None:
        try:
            assert isinstance(result, InspectionResult)
            ensure_dir(self.config.io.output_dir)
            output_dir = Path(self.config.io.output_dir)
            raw_path = output_dir / "raw.png"
            overlay_path = output_dir / "overlay.png"
            json_path = output_dir / "results.json"
            save_image(raw_path, result.raw_image)
            save_image(overlay_path, result.overlay_image)
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(result.to_json(), fh, indent=2)
            self.finished.emit(str(output_dir))
        except Exception as exc:
            LOGGER.exception("Failed to save inspection artefacts: %s", exc)
            self.failed.emit(str(exc))


class PlcTriggerWorker(QtCore.QThread):
    triggered = QtCore.pyqtSignal()

    def __init__(self, plc: PlcController, poll_interval: float = 0.05, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._plc = plc
        self._poll_interval = poll_interval
        self._stopping = threading.Event()

    def run(self) -> None:  # pragma: no cover - thread logic
        while not self._stopping.is_set():
            try:
                if self._plc.client.read_bit(self._plc.config.addr.trigger):
                    self.triggered.emit()
                    while self._plc.client.read_bit(self._plc.config.addr.trigger) and not self._stopping.is_set():
                        self.msleep(int(self._poll_interval * 1000))
                self.msleep(int(self._poll_interval * 1000))
            except Exception as exc:
                LOGGER.error("PLC trigger polling failed: %s", exc)
                self.msleep(1000)

    def stop(self) -> None:  # pragma: no cover
        self._stopping.set()
        self.wait(1000)

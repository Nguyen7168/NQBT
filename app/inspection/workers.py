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
from app.inspection.cropping import CropResult, CircleCropper, CircleDetectionError
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
    anomaly_maps: Optional[List[np.ndarray]] = None  # Optional per-patch normalized maps
    detected_circles: Optional[int] = None
    expected_circles: Optional[int] = None

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
    camera_ready = QtCore.pyqtSignal()
    camera_failed = QtCore.pyqtSignal(str)

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
        # Always use circle-based cropper per current requirements
        self.cropper = CircleCropper(config.layout)
        self.camera = DummyCamera(config.camera) if use_dummy_camera else BaslerCamera(config.camera)
        self.anomaly: Optional[AnomalyDetector]
        self._anomaly_error: Optional[str] = None
        try:
            self.anomaly = AnomalyDetector(config.models)
        except Exception as exc:
            LOGGER.warning("Failed to initialise anomaly detector: %s", exc)
            self.anomaly = None
            self._anomaly_error = str(exc)
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
    def connect_camera(self) -> None:
        with self._lock:
            try:
                if not self._camera_ready:
                    self.camera.connect()
                    self._camera_ready = True
                self.camera_ready.emit()
            except Exception as exc:
                self._camera_ready = False
                LOGGER.error("Camera connection failed: %s", exc)
                self.camera_failed.emit(str(exc))

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
                patches, detected = self.cropper.crop_with_count(capture.image)
                expected = self.config.layout.count
                mismatch = detected != expected
                if self.anomaly is None:
                    raise RuntimeError(
                        f"Anomaly model not available: {self._anomaly_error or 'unknown error'}"
                    )
                anomaly = None
                if patches:
                    anomaly = self.anomaly.infer([p.image for p in patches])
                algo = (self.config.models.algo or "INP").upper()
                if algo == "GLASS":
                    threshold = float(self.config.models.glass.glass_threshold)
                    model_path = self.config.models.glass.path
                else:
                    threshold = float(self.config.models.inp.inp_threshold)
                    model_path = self.config.models.inp.path
                statuses = (
                    ["OK" if score <= threshold else "NG" for score in anomaly.scores]
                    if anomaly is not None
                    else []
                )
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
                    anomaly_scores=anomaly.scores if anomaly is not None else [],
                    statuses=statuses,
                    ng_total=ng_total,
                    anomaly_inference_ms=anomaly.inference_ms if anomaly is not None else 0.0,
                    yolo_result=yolo_result,
                    timestamp=time.time(),
                    model_path=model_path,
                    threshold=threshold,
                    anomaly_maps=anomaly.maps if anomaly is not None else None,
                    detected_circles=detected,
                    expected_circles=expected,
                )

                if mismatch:
                    LOGGER.warning("Circle detection mismatch: detected %s, expected %s", detected, expected)
                    self.plc.write_results([False] * expected)
                    self.plc.set_error(True)
                else:
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
                # Update the current algorithm's model path
                if (self.config.models.algo or "INP").upper() == "GLASS":
                    self.config.models.glass.path = model_path
                else:
                    self.config.models.inp.path = model_path
                # Recreate detector with updated models config
                self.anomaly = AnomalyDetector(self.config.models)
                self._anomaly_error = None
                LOGGER.info("Reloaded anomaly model from %s", model_path)
            except Exception as exc:
                self.anomaly = None
                self._anomaly_error = str(exc)
                LOGGER.error("Failed to reload anomaly model: %s", exc)

    @QtCore.pyqtSlot(object)
    def run_on_image(self, image_obj: object) -> None:
        """Run anomaly (and optional YOLO) on a provided image.

        This does not interact with the PLC or camera; it reuses the
        cropping + inference pipeline and emits cycle_completed/failed
        for the UI to update as usual.
        """
        with self._lock:
            try:
                assert isinstance(image_obj, np.ndarray), "Expected numpy image"
                image = image_obj
                self.cycle_started.emit()

                patches, detected = self.cropper.crop_with_count(image)
                expected = self.config.layout.count
                if self.anomaly is None:
                    raise RuntimeError(
                        f"Anomaly model not available: {self._anomaly_error or 'unknown error'}"
                    )
                anomaly = None
                if patches:
                    anomaly = self.anomaly.infer([p.image for p in patches])
                algo = (self.config.models.algo or "INP").upper()
                if algo == "GLASS":
                    threshold = float(self.config.models.glass.glass_threshold)
                    model_path = self.config.models.glass.path
                else:
                    threshold = float(self.config.models.inp.inp_threshold)
                    model_path = self.config.models.inp.path
                statuses = (
                    ["OK" if score <= threshold else "NG" for score in anomaly.scores]
                    if anomaly is not None
                    else []
                )
                ng_total = sum(1 for status in statuses if status == "NG")

                yolo_result = None
                if self.yolo is not None:
                    try:
                        yolo_result = self.yolo.detect(image)
                    except Exception as exc:
                        LOGGER.error("YOLO inference failed: %s", exc)

                overlay = self._build_overlay(image, patches, statuses, yolo_result)
                result = InspectionResult(
                    raw_image=image,
                    overlay_image=overlay,
                    patches=patches,
                    anomaly_scores=anomaly.scores if anomaly is not None else [],
                    statuses=statuses,
                    ng_total=ng_total,
                    anomaly_inference_ms=anomaly.inference_ms if anomaly is not None else 0.0,
                    yolo_result=yolo_result,
                    timestamp=time.time(),
                    model_path=model_path,
                    threshold=threshold,
                    anomaly_maps=anomaly.maps if anomaly is not None else None,
                    detected_circles=detected,
                    expected_circles=expected,
                )

                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Manual image inference failed: %s", exc)
                self.cycle_failed.emit(str(exc))

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
            label = str(patch.index)
            box_width = max(x2 - x1, 1)
            font_scale = max(0.5, min(1.2, box_width / 180))
            thickness = max(1, int(round(font_scale * 2)))
            text_origin = (x1 + 4, y1 + int(20 * font_scale))
            cv2.putText(
                overlay,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
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
            overlay_path = output_dir / "overlay.png"
            json_path = output_dir / "results.json"
            ts_value = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            model_name = Path(result.model_path).stem or "model"
            filename_pattern = self.config.io.filename_pattern
            raw_name = filename_pattern.format(ts=ts_value, model=model_name, idx=0, cls="raw")
            raw_dir = ensure_dir(self.config.io.raw_dir)
            raw_path = raw_dir / raw_name
            save_image(raw_path, result.raw_image)
            save_image(overlay_path, result.overlay_image)
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(result.to_json(), fh, indent=2)
            if self.config.io.save_crops:
                crops_dir = ensure_dir(self.config.io.crops_dir)
                for patch, status in zip(result.patches, result.statuses):
                    crop_name = filename_pattern.format(
                        ts=ts_value,
                        model=model_name,
                        idx=patch.index,
                        cls=status.lower(),
                    )
                    crop_path = crops_dir / crop_name
                    save_image(crop_path, patch.image)
            # Optionally save per-patch heatmaps and binary masks if available
            if (self.config.io.save_heatmap or self.config.io.save_binary) and result.anomaly_maps:
                maps_dir = output_dir / "maps"
                ensure_dir(maps_dir)
                algo = (self.config.models.algo or "INP").upper()
                if algo == "GLASS":
                    bin_th = float(self.config.models.glass.glass_bin_thresh)
                else:
                    bin_th = float(self.config.models.inp.inp_bin_thresh)
                for patch, status, amap in zip(result.patches, result.statuses, result.anomaly_maps):
                    idx = patch.index
                    # Normalize to 0..255 uint8 for saving
                    gray = np.clip(amap * 255.0, 0, 255).astype(np.uint8)
                    if self.config.io.save_heatmap:
                        heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                        heat_path = maps_dir / f"heatmap_{idx:02d}.png"
                        cv2.imwrite(str(heat_path), heat)
                    if self.config.io.save_binary:
                        binary = (amap >= bin_th).astype(np.uint8) * 255
                        bin_path = maps_dir / f"binary_{idx:02d}.png"
                        cv2.imwrite(str(bin_path), binary)
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

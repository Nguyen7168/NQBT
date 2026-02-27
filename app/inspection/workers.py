"""Qt worker objects orchestrating camera capture, inference, and PLC handshake."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from enum import IntEnum
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
from kiem_guong import find_outer_circle_from_edges

LOGGER = logging.getLogger(__name__)


class OperatingMode(IntEnum):
    RUN = 1
    SAMPLE = 2
    MIRROR = 3


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
    model_reloaded = QtCore.pyqtSignal(str, float)
    model_reload_failed = QtCore.pyqtSignal(str)
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

    def _ensure_camera_ready(self) -> None:
        if not self._camera_ready:
            self.camera.connect()
            self._camera_ready = True

    def _build_result_from_image(self, image: np.ndarray) -> InspectionResult:
        patches, detected = self.cropper.crop_with_count(image)
        expected = self.config.layout.count
        if self.anomaly is None:
            raise RuntimeError(
                f"Anomaly model not available: {self._anomaly_error or 'unknown error'}"
            )
        anomaly = self.anomaly.infer([p.image for p in patches]) if patches else None
        algo = (self.config.models.algo or "INP").upper()
        if algo == "GLASS":
            threshold = float(self.config.models.glass.glass_threshold)
            model_path = self.config.models.glass.path
        else:
            threshold = float(self.config.models.inp.inp_threshold)
            model_path = self.config.models.inp.path
        statuses = (["OK" if score <= threshold else "NG" for score in anomaly.scores] if anomaly is not None else [])
        ng_total = sum(1 for status in statuses if status == "NG")

        yolo_result = None
        if self.yolo is not None:
            try:
                yolo_result = self.yolo.detect(image)
            except Exception as exc:
                LOGGER.error("YOLO inference failed: %s", exc)

        overlay = self._build_overlay(image, patches, statuses, yolo_result)
        return InspectionResult(
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

    def _run_standard_cycle(self, image: np.ndarray) -> InspectionResult:
        result = self._build_result_from_image(image)
        mismatch = result.detected_circles != result.expected_circles
        if mismatch:
            expected = result.expected_circles or self.config.layout.count
            LOGGER.warning("Circle detection mismatch: detected %s, expected %s", result.detected_circles, expected)
            self.plc.write_results([False] * expected)
        else:
            self.plc.write_results([status == "OK" for status in result.statuses])
        self.plc.set_error(False)
        return result

    @QtCore.pyqtSlot()
    def run_cycle(self) -> None:
        with self._lock:
            try:
                self.cycle_started.emit()
                self.plc.set_busy(True)
                self._ensure_camera_ready()
                self.plc.set_run(True)
                capture = self.camera.capture()
                LOGGER.debug("Captured image with shape %s", capture.image.shape)
                result = self._run_standard_cycle(capture.image)
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
                try:
                    self.plc.finalize_cycle()
                except Exception:
                    LOGGER.exception("Finalize cycle failed")

    @QtCore.pyqtSlot(str)
    def run_sample_cycle(self, sample_image_path: str) -> None:
        with self._lock:
            try:
                self.cycle_started.emit()
                self.plc.set_busy(True)
                image = cv2.imread(sample_image_path)
                if image is None:
                    raise RuntimeError(f"Cannot read sample image: {sample_image_path}")
                LOGGER.info("Running sample cycle using image: %s", sample_image_path)
                result = self._run_standard_cycle(image)
                self.plc.set_done(True)
                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Sample cycle failed: %s", exc)
                try:
                    self.plc.write_results([False] * self.config.layout.count)
                    self.plc.set_error(True)
                    self.plc.set_done(True)
                finally:
                    self.cycle_failed.emit(str(exc))
            finally:
                try:
                    self.plc.finalize_cycle()
                except Exception:
                    LOGGER.exception("Finalize cycle failed")

    @QtCore.pyqtSlot()
    def run_mirror_cycle(self) -> None:
        with self._lock:
            try:
                self.cycle_started.emit()
                self.plc.set_busy(True)
                self._ensure_camera_ready()
                capture = self.camera.capture()
                image = capture.image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_kernel = max(1, int(self.config.mirror_blur_kernel))
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
                edges = cv2.Canny(blur, int(self.config.mirror_canny_threshold1), int(self.config.mirror_canny_threshold2))
                result_circle, _ = find_outer_circle_from_edges(edges, min_contour_area=float(self.config.mirror_min_contour_area))
                diameter = float(2.0 * result_circle.radius)
                is_ok = float(self.config.mirror_diameter_min) <= diameter <= float(self.config.mirror_diameter_max)
                out_addr = self.config.plc.addr.mirror_result_word
                if out_addr:
                    self.plc.write_word(out_addr, 1 if is_ok else 0)
                self.plc.set_error(False)
                self.plc.set_done(True)
                LOGGER.info("Mirror mode diameter=%.3f, limits=[%.3f, %.3f], result=%s", diameter, self.config.mirror_diameter_min, self.config.mirror_diameter_max, "OK" if is_ok else "NG")
            except Exception as exc:
                LOGGER.warning("Mirror cycle failed or no circle detected: %s", exc)
                out_addr = self.config.plc.addr.mirror_result_word
                try:
                    if out_addr:
                        self.plc.write_word(out_addr, 0)
                    self.plc.set_error(False)
                    self.plc.set_done(True)
                finally:
                    self.cycle_failed.emit(str(exc))
            finally:
                try:
                    self.plc.finalize_cycle()
                except Exception:
                    LOGGER.exception("Finalize cycle failed")

    @QtCore.pyqtSlot()
    def shutdown(self) -> None:  # pragma: no cover - cleanup
        with self._lock:
            try:
                self.camera.disconnect()
            except Exception as exc:
                LOGGER.debug("Camera disconnect failed: %s", exc)

    @QtCore.pyqtSlot(str)
    def reload_anomaly_model(self, model_path: str) -> None:
        self.reload_anomaly_model_with_threshold(model_path, self.config.models.glass.glass_threshold if (self.config.models.algo or "INP").upper() == "GLASS" else self.config.models.inp.inp_threshold)

    @QtCore.pyqtSlot(str, float)
    def reload_anomaly_model_with_threshold(self, model_path: str, threshold: float) -> None:
        with self._lock:
            try:
                # Update the current algorithm's model path
                if (self.config.models.algo or "INP").upper() == "GLASS":
                    self.config.models.glass.path = model_path
                    self.config.models.glass.glass_threshold = float(threshold)
                    effective_threshold = float(self.config.models.glass.glass_threshold)
                else:
                    self.config.models.inp.path = model_path
                    self.config.models.inp.inp_threshold = float(threshold)
                    effective_threshold = float(self.config.models.inp.inp_threshold)
                # Recreate detector with updated models config
                self.anomaly = AnomalyDetector(self.config.models)
                self._anomaly_error = None
                LOGGER.info("Reloaded anomaly model from %s", model_path)
                self.model_reloaded.emit(model_path, effective_threshold)
            except Exception as exc:
                self.anomaly = None
                self._anomaly_error = str(exc)
                LOGGER.error("Failed to reload anomaly model: %s", exc)
                self.model_reload_failed.emit(str(exc))

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




class PlcModelSelectWorker(QtCore.QThread):
    model_code_changed = QtCore.pyqtSignal(int)

    def __init__(
        self,
        plc: PlcController,
        address: str,
        poll_interval: float = 0.2,
        stable_ms: int = 200,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plc = plc
        self._address = address
        self._poll_interval = poll_interval
        self._stable_ms = max(int(stable_ms), 0)
        self._stopping = threading.Event()
        self._last_emitted_value: Optional[int] = None
        self._candidate_value: Optional[int] = None
        self._candidate_since: float = 0.0

    def run(self) -> None:  # pragma: no cover - thread logic
        while not self._stopping.is_set():
            try:
                value = int(self._plc.read_word(self._address))
                now = time.time()
                if self._candidate_value != value:
                    self._candidate_value = value
                    self._candidate_since = now
                stable_ms = (now - self._candidate_since) * 1000.0
                if self._last_emitted_value is None:
                    self._last_emitted_value = value
                    self.model_code_changed.emit(value)
                elif self._candidate_value != self._last_emitted_value and stable_ms >= self._stable_ms:
                    self._last_emitted_value = self._candidate_value
                    self.model_code_changed.emit(self._candidate_value)
                self.msleep(int(self._poll_interval * 1000))
            except Exception as exc:
                LOGGER.error("PLC model select polling failed: %s", exc)
                self.msleep(1000)

    def stop(self) -> None:  # pragma: no cover
        self._stopping.set()
        self.wait(1000)


class PlcModeSelectWorker(QtCore.QThread):
    mode_changed = QtCore.pyqtSignal(int)

    def __init__(
        self,
        plc: PlcController,
        address: str,
        poll_interval: float = 0.2,
        stable_ms: int = 200,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plc = plc
        self._address = address
        self._poll_interval = poll_interval
        self._stable_ms = max(int(stable_ms), 0)
        self._stopping = threading.Event()
        self._last_emitted_value: Optional[int] = None
        self._candidate_value: Optional[int] = None
        self._candidate_since: float = 0.0

    def run(self) -> None:  # pragma: no cover - thread logic
        while not self._stopping.is_set():
            try:
                value = int(self._plc.read_word(self._address))
                now = time.time()
                if self._candidate_value != value:
                    self._candidate_value = value
                    self._candidate_since = now
                stable_ms = (now - self._candidate_since) * 1000.0
                if self._last_emitted_value is None:
                    self._last_emitted_value = value
                    self.mode_changed.emit(value)
                elif self._candidate_value != self._last_emitted_value and stable_ms >= self._stable_ms:
                    self._last_emitted_value = self._candidate_value
                    self.mode_changed.emit(self._candidate_value)
                self.msleep(int(self._poll_interval * 1000))
            except Exception as exc:
                LOGGER.error("PLC mode polling failed: %s", exc)
                self.msleep(1000)

    def stop(self) -> None:  # pragma: no cover
        self._stopping.set()
        self.wait(1000)


class PlcTriggerWorker(QtCore.QThread):
    triggered = QtCore.pyqtSignal()

    def __init__(
        self,
        plc: PlcController,
        trigger_address: Optional[str] = None,
        poll_interval: float = 0.05,
        min_interval_ms: int = 100,
        high_stable_ms: int = 80,
        low_stable_ms: int = 80,
        cooldown_ms: int = 300,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plc = plc
        self._trigger_address = trigger_address or plc.config.addr.trigger
        self._poll_interval = poll_interval
        self._min_interval_ms = max(int(min_interval_ms), 0)
        self._high_stable_ms = max(int(high_stable_ms), 0)
        self._low_stable_ms = max(int(low_stable_ms), 0)
        self._cooldown_ms = max(int(cooldown_ms), 0)
        self._stopping = threading.Event()
        self._last_trigger_ts: float = 0.0
        self._armed = True
        self._high_since: Optional[float] = None
        self._low_since: Optional[float] = time.time()

    def _can_emit(self, now: float) -> bool:
        elapsed_ms = (now - self._last_trigger_ts) * 1000.0
        required_ms = max(self._min_interval_ms, self._cooldown_ms)
        if self._last_trigger_ts == 0.0 or elapsed_ms >= required_ms:
            return True
        LOGGER.debug(
            "Ignore trigger due to cooldown (%d ms < %d ms)",
            int(elapsed_ms),
            required_ms,
        )
        return False

    def run(self) -> None:  # pragma: no cover - thread logic
        while not self._stopping.is_set():
            try:
                state = self._plc.client.read_bit(self._trigger_address)
                now = time.time()

                if state:
                    self._low_since = None
                    if self._high_since is None:
                        self._high_since = now
                    high_ms = (now - self._high_since) * 1000.0
                    if self._armed and high_ms >= self._high_stable_ms and self._can_emit(now):
                        self._last_trigger_ts = now
                        self._armed = False
                        self.triggered.emit()
                else:
                    self._high_since = None
                    if self._low_since is None:
                        self._low_since = now
                    low_ms = (now - self._low_since) * 1000.0
                    if not self._armed and low_ms >= self._low_stable_ms:
                        self._armed = True

                self.msleep(int(self._poll_interval * 1000))
            except Exception as exc:
                LOGGER.error("PLC trigger polling failed: %s", exc)
                self.msleep(1000)

    def stop(self) -> None:  # pragma: no cover
        self._stopping.set()
        self.wait(1000)

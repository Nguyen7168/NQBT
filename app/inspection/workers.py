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
from typing import Dict, List, MutableMapping, Optional, Sequence

import cv2
import numpy as np

from app.config_loader import AppConfig
from app.inspection.camera import BaslerCamera, DummyCamera
from app.inspection.cropping import CropResult, CircleDetectionError
from app.inspection.inference_pipeline import InferencePipeline, InferenceResultPayload
from app.inspection.inference_service import RemoteInferenceClient
from app.inspection.plc_client import PlcController
from app.models.yolo import YoloResult
from PyQt5 import QtCore
from app.utils import ensure_dir, save_image
try:
    from kiem_guong import draw_overlay, find_outer_circle_from_edges
except Exception:  # pragma: no cover - optional mirror module
    draw_overlay = None  # type: ignore
    find_outer_circle_from_edges = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class OperatingMode(IntEnum):
    RUN = 3
    SAMPLE = 2
    MIRROR = 1


@dataclass
class InspectionResult:
    raw_image: np.ndarray
    overlay_image: np.ndarray
    patches: List[CropResult]
    anomaly_scores: List[float]
    statuses: List[str]
    ng_total: int
    anomaly_inference_ms: float
    cycle_elapsed_ms: float
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
            "cycle_elapsed_ms": self.cycle_elapsed_ms,
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
    mirror_test_completed = QtCore.pyqtSignal(object, float, bool)

    def __init__(
        self,
        config: AppConfig,
        plc: PlcController,
        parent: Optional[QtCore.QObject] = None,
        use_dummy_camera: bool = False,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        self.plc = plc
        self.camera = DummyCamera(config.camera) if use_dummy_camera else BaslerCamera(config.camera)
        self.pipeline = InferencePipeline(config)
        self.remote_client: Optional[RemoteInferenceClient] = None
        if bool(getattr(config.inference_service, "enabled", False)):
            self.remote_client = RemoteInferenceClient(
                config.inference_service.host,
                config.inference_service.port,
                timeout_ms=config.inference_service.timeout_ms,
            )
        self._lock = threading.Lock()
        self._camera_ready = False
        self._camera_fail_streak = 0
        self._camera_recover_stable_success = 0

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

    def _capture_with_reconnect(self) -> object:
        try:
            self._ensure_camera_ready()
            capture = self.camera.capture()
            self._camera_fail_streak = 0
            self._camera_recover_stable_success += 1
            return capture
        except Exception as exc:
            self._camera_fail_streak += 1
            self._camera_recover_stable_success = 0
            threshold = max(1, int(getattr(self.config.camera, "reconnect_fail_streak_threshold", 3)))
            if not bool(getattr(self.config.camera, "reconnect_enabled", True)) or self._camera_fail_streak < threshold:
                raise
            LOGGER.warning("Camera capture failed %d times; trying reconnect", self._camera_fail_streak)
            try:
                self.camera.disconnect()
            except Exception:
                pass
            self._camera_ready = False
            interval_ms = max(0, int(getattr(self.config.camera, "reconnect_interval_ms", 2000)))
            if interval_ms > 0:
                time.sleep(interval_ms / 1000.0)
            self._ensure_camera_ready()
            capture = self.camera.capture()
            self._camera_fail_streak = 0
            self._camera_recover_stable_success = 1
            return capture

    @staticmethod
    def _timed_call(timings: MutableMapping[str, float], key: str, action) -> object:
        start = time.perf_counter()
        result = action()
        timings[key] = (time.perf_counter() - start) * 1000.0
        return result

    def _log_cycle_timing(
        self,
        cycle_type: str,
        timings: MutableMapping[str, float],
        result: Optional[InspectionResult] = None,
    ) -> None:
        if not bool(getattr(self.config, "timing_log_enabled", False)):
            return
        fields = [
            f"camera={timings.get('camera', 0.0):.1f}ms",
            f"crop={timings.get('crop', 0.0):.1f}ms",
            f"crop_yolo_detect_ms={timings.get('crop_yolo_detect_ms', 0.0):.1f}ms",
            f"crop_box_to_circle_ms={timings.get('crop_box_to_circle_ms', 0.0):.1f}ms",
            f"crop_mask_and_cut_ms={timings.get('crop_mask_and_cut_ms', 0.0):.1f}ms",
            f"crop_sort_ms={timings.get('crop_sort_ms', 0.0):.1f}ms",
            f"anomaly={timings.get('anomaly', 0.0):.1f}ms",
            f"anomaly_model={timings.get('anomaly_model', 0.0):.1f}ms",
            f"plc_busy={timings.get('plc_busy', 0.0):.1f}ms",
            f"plc_run={timings.get('plc_run', 0.0):.1f}ms",
            f"plc_write_results={timings.get('plc_write_results', 0.0):.1f}ms",
            f"plc_write_detected_count={timings.get('plc_write_detected_count', 0.0):.1f}ms",
            f"plc_error={timings.get('plc_error', 0.0):.1f}ms",
            f"plc_done={timings.get('plc_done', 0.0):.1f}ms",
            f"total={timings.get('cycle_total', 0.0):.1f}ms",
        ]
        if result is not None:
            fields.extend(
                [
                    f"detected={result.detected_circles}",
                    f"expected={result.expected_circles}",
                    f"patches={len(result.patches)}",
                    f"ng={result.ng_total}",
                ]
            )
        LOGGER.info("Timing[%s] %s", cycle_type, " | ".join(fields))

    @staticmethod
    def _payload_to_result(payload: InferenceResultPayload | Dict[str, object]) -> InspectionResult:
        data = payload.__dict__ if isinstance(payload, InferenceResultPayload) else payload
        return InspectionResult(
            raw_image=data["raw_image"],  # type: ignore[arg-type]
            overlay_image=data["overlay_image"],  # type: ignore[arg-type]
            patches=data["patches"],  # type: ignore[arg-type]
            anomaly_scores=data["anomaly_scores"],  # type: ignore[arg-type]
            statuses=data["statuses"],  # type: ignore[arg-type]
            ng_total=int(data["ng_total"]),
            anomaly_inference_ms=float(data["anomaly_inference_ms"]),
            cycle_elapsed_ms=0.0,
            yolo_result=data.get("yolo_result"),  # type: ignore[arg-type]
            timestamp=float(data["timestamp"]),
            model_path=str(data["model_path"]),
            threshold=float(data["threshold"]),
            anomaly_maps=data.get("anomaly_maps"),  # type: ignore[arg-type]
            detected_circles=(None if data.get("detected_circles") is None else int(data["detected_circles"])),
            expected_circles=(None if data.get("expected_circles") is None else int(data["expected_circles"])),
        )

    @staticmethod
    def _merge_remote_timings(timings: MutableMapping[str, float], remote_data: Dict[str, object]) -> None:
        remote_timings = remote_data.get("_remote_timings")
        if not isinstance(remote_timings, dict):
            return
        for key, value in remote_timings.items():
            try:
                timings[str(key)] = float(value)
            except Exception:
                continue

    def _run_inference_pipeline(
        self,
        image: np.ndarray,
        timings: Optional[MutableMapping[str, float]] = None,
    ) -> InspectionResult:
        if self.remote_client is not None:
            if timings is not None:
                remote_result = self._timed_call(
                    timings,
                    "remote_inference_total",
                    lambda: self.remote_client.infer(image, self.config_path, self.config),
                )
                assert isinstance(remote_result, dict)
                self._merge_remote_timings(timings, remote_result)
            else:
                remote_result = self.remote_client.infer(image, self.config_path, self.config)
            return self._payload_to_result(remote_result)
        payload = self.pipeline.run_on_image(image, timings=timings)
        return self._payload_to_result(payload)

    def _write_detected_count(self, count: int) -> None:
        addr = getattr(self.plc.config.addr, "detected_count_word", None)
        if not addr:
            return
        self.plc.write_word(addr, max(0, int(count)))

    def _write_detected_count(self, count: int) -> None:
        addr = getattr(self.plc.config.addr, "detected_count_word", None)
        if not addr:
            return
        self.plc.write_word(addr, max(0, int(count)))

    def _run_standard_cycle(self, image: np.ndarray, timings: Optional[MutableMapping[str, float]] = None) -> InspectionResult:
        result = self._run_inference_pipeline(image, timings=timings)
        detected_count = result.detected_circles if result.detected_circles is not None else 0
        if timings is not None:
            self._timed_call(timings, "plc_write_detected_count", lambda: self._write_detected_count(detected_count))
        else:
            self._write_detected_count(detected_count)
        mismatch = result.detected_circles != result.expected_circles
        if mismatch:
            expected = result.expected_circles or self.config.layout.count
            LOGGER.warning("Circle detection mismatch: detected %s, expected %s", result.detected_circles, expected)
            if timings is not None:
                self._timed_call(timings, "plc_write_results", lambda: self.plc.write_results([False] * expected))
            else:
                self.plc.write_results([False] * expected)
        else:
            if timings is not None:
                self._timed_call(timings, "plc_write_results", lambda: self.plc.write_results([status == "OK" for status in result.statuses]))
            else:
                self.plc.write_results([status == "OK" for status in result.statuses])
        if timings is not None:
            self._timed_call(timings, "plc_error", lambda: self.plc.set_error(False))
        else:
            self.plc.set_error(False)
        return result

    @QtCore.pyqtSlot()
    def run_cycle(self) -> None:
        with self._lock:
            cycle_start = time.perf_counter()
            timings: Dict[str, float] = {}
            result: Optional[InspectionResult] = None
            try:
                self.cycle_started.emit()
                self._timed_call(timings, "plc_busy", lambda: self.plc.set_busy(True, reason="run_cycle"))
                self._ensure_camera_ready()
                self._timed_call(timings, "plc_run", lambda: self.plc.set_run(True))
                capture = self._timed_call(timings, "camera", self._capture_with_reconnect)
                LOGGER.debug("Captured image with shape %s", capture.image.shape)
                result = self._run_standard_cycle(capture.image, timings=timings)
                self._timed_call(timings, "plc_done", lambda: self.plc.set_done(True))
                result.cycle_elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
                timings["cycle_total"] = result.cycle_elapsed_ms
                self._log_cycle_timing("run", timings, result)
                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Inspection cycle failed: %s", exc)
                try:
                    self.plc.write_results([False] * self.config.layout.count)
                    try:
                        self._write_detected_count(0)
                    except Exception:
                        LOGGER.warning("Failed to write detected_count_word=0 on run cycle failure", exc_info=True)
                    self.plc.set_error(True)
                    self.plc.set_done(True)
                finally:
                    timings["cycle_total"] = (time.perf_counter() - cycle_start) * 1000.0
                    self._log_cycle_timing("run_failed", timings, result)
                    self.cycle_failed.emit(str(exc))
            finally:
                try:
                    self.plc.finalize_cycle()
                except Exception:
                    LOGGER.exception("Finalize cycle failed")

    @QtCore.pyqtSlot(str)
    def run_sample_cycle(self, sample_image_path: str) -> None:
        with self._lock:
            cycle_start = time.perf_counter()
            timings: Dict[str, float] = {}
            result: Optional[InspectionResult] = None
            try:
                self.cycle_started.emit()
                self._timed_call(timings, "plc_busy", lambda: self.plc.set_busy(True, reason="sample_cycle"))
                image = self._timed_call(timings, "camera", lambda: cv2.imread(sample_image_path))
                if image is None:
                    raise RuntimeError(f"Cannot read sample image: {sample_image_path}")
                LOGGER.info("Running sample cycle using image: %s", sample_image_path)
                result = self._run_standard_cycle(image, timings=timings)
                self._timed_call(timings, "plc_done", lambda: self.plc.set_done(True))
                result.cycle_elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
                timings["cycle_total"] = result.cycle_elapsed_ms
                self._log_cycle_timing("sample", timings, result)
                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Sample cycle failed: %s", exc)
                try:
                    self.plc.write_results([False] * self.config.layout.count)
                    try:
                        self._write_detected_count(0)
                    except Exception:
                        LOGGER.warning("Failed to write detected_count_word=0 on sample cycle failure", exc_info=True)
                    self.plc.set_error(True)
                    self.plc.set_done(True)
                finally:
                    timings["cycle_total"] = (time.perf_counter() - cycle_start) * 1000.0
                    self._log_cycle_timing("sample_failed", timings, result)
                    self.cycle_failed.emit(str(exc))
            finally:
                try:
                    self.plc.finalize_cycle()
                except Exception:
                    LOGGER.exception("Finalize cycle failed")

    def _measure_mirror(self, image: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if find_outer_circle_from_edges is None:
            raise RuntimeError("Mirror module unavailable: kiem_guong.py not found")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_kernel = max(1, int(self.config.mirror_blur_kernel))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blur, int(self.config.mirror_canny_threshold1), int(self.config.mirror_canny_threshold2))
        result_circle, _ = find_outer_circle_from_edges(edges, min_contour_area=float(self.config.mirror_min_contour_area))
        diameter = float(2.0 * result_circle.radius)
        is_ok = float(self.config.mirror_diameter_min) <= diameter <= float(self.config.mirror_diameter_max)
        if draw_overlay is not None:
            overlay = draw_overlay(image, result_circle.center, result_circle.radius)
        else:
            overlay = image.copy()
            cx, cy = map(int, map(round, result_circle.center))
            cv2.circle(overlay, (cx, cy), int(round(result_circle.radius)), (0, 255, 0), 3)
            cv2.putText(overlay, f"Outer D={diameter:.2f}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        color = (0, 255, 0) if is_ok else (0, 0, 255)
        cv2.putText(
            overlay,
            f"Mirror {'OK' if is_ok else 'NG'} | D={diameter:.2f}px | lim=[{self.config.mirror_diameter_min:.2f},{self.config.mirror_diameter_max:.2f}]",
            (20, max(70, overlay.shape[0] - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return overlay, diameter, is_ok

    def _write_mirror_result(self, is_ok: bool) -> None:
        out_addr = self.config.plc.addr.mirror_result_word
        if out_addr:
            self.plc.write_word(out_addr, 1 if is_ok else 0)

    @QtCore.pyqtSlot(str)
    def run_mirror_test_on_image(self, image_path: str) -> None:
        with self._lock:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Cannot read mirror test image: {image_path}")
                overlay, diameter, is_ok = self._measure_mirror(image)
                self._write_mirror_result(is_ok)
                self.mirror_test_completed.emit(overlay, diameter, is_ok)
            except Exception as exc:
                LOGGER.warning("Mirror test from image failed: %s", exc)
                try:
                    self._write_mirror_result(False)
                except Exception:
                    pass
                self.cycle_failed.emit(f"MIRROR: {exc}")

    @QtCore.pyqtSlot()
    def run_mirror_cycle(self) -> None:
        with self._lock:
            try:
                self.cycle_started.emit()
                self.plc.set_busy(True, reason="mirror_cycle")
                try:
                    self._ensure_camera_ready()
                    capture = self._capture_with_reconnect()
                except Exception as exc:
                    raise RuntimeError(f"MIRROR_CAMERA: {exc}") from exc

                image = capture.image
                try:
                    overlay, diameter, is_ok = self._measure_mirror(image)
                except Exception as exc:
                    raise RuntimeError(f"MIRROR_DETECT: {exc}") from exc

                self._write_mirror_result(is_ok)
                self.plc.set_error(False)
                self.plc.set_done(True)
                self.mirror_test_completed.emit(overlay, diameter, is_ok)
                LOGGER.info("Mirror mode diameter=%.3f, limits=[%.3f, %.3f], result=%s", diameter, self.config.mirror_diameter_min, self.config.mirror_diameter_max, "OK" if is_ok else "NG")
            except Exception as exc:
                LOGGER.warning("Mirror cycle failed or no circle detected: %s", exc)
                try:
                    self._write_mirror_result(False)
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
        self.reload_anomaly_model_with_threshold(
            model_path,
            self.pipeline.active_glass_threshold if (self.config.models.algo or "INP").upper() == "GLASS" else self.config.models.inp.inp_threshold,
        )

    @QtCore.pyqtSlot(str, float)
    def reload_anomaly_model_with_threshold(self, model_path: str, threshold: float) -> None:
        self.reload_recipe_models(model_path, threshold, "")

    @QtCore.pyqtSlot(str, float, str)
    def reload_recipe_models(self, model_path: str, threshold: float, crop_yolo_path: str = "") -> None:
        with self._lock:
            try:
                _, effective_threshold = self.pipeline.reload_recipe_models(model_path, threshold, crop_yolo_path)
                self.model_reloaded.emit(model_path, effective_threshold)
            except Exception as exc:
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
                result = self._run_inference_pipeline(image)
                self.cycle_completed.emit(result)
            except Exception as exc:
                LOGGER.exception("Manual image inference failed: %s", exc)
                self.cycle_failed.emit(str(exc))


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
        error_backoff_ms: int = 1000,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plc = plc
        self._address = address
        self._poll_interval = poll_interval
        self._stable_ms = max(int(stable_ms), 0)
        self._stopping = threading.Event()
        self._error_backoff_ms = max(int(error_backoff_ms), 0)
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
                self.msleep(self._error_backoff_ms)

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
        error_backoff_ms: int = 1000,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plc = plc
        self._address = address
        self._poll_interval = poll_interval
        self._stable_ms = max(int(stable_ms), 0)
        self._stopping = threading.Event()
        self._error_backoff_ms = max(int(error_backoff_ms), 0)
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
                self.msleep(self._error_backoff_ms)

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
        error_backoff_ms: int = 1000,
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
        self._error_backoff_ms = max(int(error_backoff_ms), 0)
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
                if not self._plc.can_process_plc_trigger():
                    self.msleep(int(self._poll_interval * 1000))
                    continue
                state = self._plc.read_bit(self._trigger_address)
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
                self.msleep(self._error_backoff_ms)

    def stop(self) -> None:  # pragma: no cover
        self._stopping.set()
        self.wait(1000)

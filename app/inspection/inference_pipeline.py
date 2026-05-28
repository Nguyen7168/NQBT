"""Reusable crop + anomaly inference pipeline for local workers and remote service."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, MutableMapping, Optional, Sequence

import cv2
import numpy as np

from app.config_loader import AppConfig
from app.inspection.cropping import CropResult, CircleCropper, YoloCircleCropper
from app.models.anomaly import AnomalyDetector
from app.models.yolo import YoloDetector, YoloResult

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceResultPayload:
    raw_image: np.ndarray
    overlay_image: np.ndarray
    patches: List[CropResult]
    anomaly_scores: List[float]
    diameters_mm: List[Optional[float]]
    statuses: List[str]
    ng_total: int
    anomaly_inference_ms: float
    yolo_result: Optional[YoloResult]
    timestamp: float
    model_path: str
    threshold: float
    diameter_min_mm: float
    diameter_max_mm: float
    anomaly_maps: Optional[List[np.ndarray]] = None
    detected_circles: Optional[int] = None
    expected_circles: Optional[int] = None


class InferencePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cropper = CircleCropper(config.layout)
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
        self.crop_yolo: Optional[YoloDetector] = None
        self._configure_cropper(crop_yolo_path=getattr(config.models.yolo, "crop_path", None))
        self._active_recipe_threshold = self._resolve_active_recipe_threshold()

    @property
    def anomaly_error(self) -> Optional[str]:
        return self._anomaly_error

    @property
    def active_recipe_threshold(self) -> float:
        return float(self._active_recipe_threshold)

    def _resolve_active_recipe_threshold(self) -> float:
        active_code = getattr(self.config.models, "active_recipe_code", None)
        recipes = list(getattr(self.config.models, "recipes", []))
        if active_code is not None:
            for recipe in recipes:
                if int(getattr(recipe, "code", -1)) == int(active_code):
                    return float(recipe.threshold)
        if recipes:
            return float(recipes[0].threshold)
        algo = (self.config.models.algo or "INP").upper()
        if algo == "GLASS":
            return float(self.config.models.glass.glass_threshold)
        return float(self.config.models.inp.inp_threshold)

    def _resolve_active_diameter_thresholds(self) -> tuple[float, float]:
        active_code = getattr(self.config.models, "active_recipe_code", None)
        recipes = list(getattr(self.config.models, "recipes", []))
        if active_code is not None:
            for recipe in recipes:
                if int(getattr(recipe, "code", -1)) == int(active_code):
                    return (
                        float(getattr(recipe, "diameter_min_mm", 0.0)),
                        float(getattr(recipe, "diameter_max_mm", 99999.0)),
                    )
        if recipes:
            recipe = recipes[0]
            return (
                float(getattr(recipe, "diameter_min_mm", 0.0)),
                float(getattr(recipe, "diameter_max_mm", 99999.0)),
            )
        algo = (self.config.models.algo or "INP").upper()
        if algo != "GLASS":
            return (
                float(getattr(self.config.models.inp, "diameter_min_mm", 0.0)),
                float(getattr(self.config.models.inp, "diameter_max_mm", 99999.0)),
            )
        return (
            float(getattr(self.config.models.glass, "diameter_min_mm", 0.0)),
            float(getattr(self.config.models.glass, "diameter_max_mm", 99999.0)),
        )

    def _init_crop_yolo_detector(self, model_path: Optional[str] = None) -> Optional[YoloDetector]:
        yolo_cfg = self.config.models.yolo
        if not bool(getattr(yolo_cfg, "crop_enabled", False)):
            return None
        path = (model_path or yolo_cfg.crop_path or "").strip()
        if not path:
            return None
        return YoloDetector(
            path,
            float(getattr(yolo_cfg, "crop_conf_thres", 0.5)),
            float(getattr(yolo_cfg, "crop_iou_thres", 0.7)),
            imgsz=int(getattr(yolo_cfg, "crop_imgsz", 960)),
            device=str(getattr(yolo_cfg, "crop_device", "cuda:0")),
            classes=getattr(yolo_cfg, "crop_classes", None),
        )

    def _configure_cropper(self, crop_yolo_path: Optional[str] = None) -> None:
        method = (getattr(self.config.layout, "crop_method", "circle") or "circle").strip().lower()
        if method != "yolo_circle":
            self.crop_yolo = None
            self.cropper = CircleCropper(self.config.layout)
            return
        try:
            self.crop_yolo = self._init_crop_yolo_detector(model_path=crop_yolo_path)
            if self.crop_yolo is None:
                raise RuntimeError("YOLO crop model is not configured")
            self.cropper = YoloCircleCropper(self.config.layout, self.crop_yolo)
            LOGGER.info("Using YOLO circle cropper")
        except Exception as exc:
            if bool(getattr(self.config.layout, "yolo_crop_fallback_to_circle", True)):
                LOGGER.warning("YOLO crop initialisation failed (%s); fallback to circle cropper", exc)
                self.crop_yolo = None
                self.cropper = CircleCropper(self.config.layout)
            else:
                raise

    def reload_recipe_models(self, model_path: str, threshold: float, crop_yolo_path: str = "") -> tuple[str, float]:
        if (self.config.models.algo or "INP").upper() == "GLASS":
            self.config.models.glass.path = model_path
            self._active_recipe_threshold = float(threshold)
            effective_threshold = float(self._active_recipe_threshold)
        else:
            self.config.models.inp.path = model_path
            self._active_recipe_threshold = float(threshold)
            effective_threshold = float(self._active_recipe_threshold)
        self.anomaly = AnomalyDetector(self.config.models)
        resolved_crop_path = (crop_yolo_path or "").strip() or getattr(self.config.models.yolo, "crop_path", None)
        if resolved_crop_path:
            self.config.models.yolo.crop_path = resolved_crop_path
        self._configure_cropper(crop_yolo_path=resolved_crop_path)
        self._anomaly_error = None
        LOGGER.info("Reloaded anomaly model from %s", model_path)
        return model_path, effective_threshold

    @staticmethod
    def _timed_call(timings: MutableMapping[str, float], key: str, action) -> object:
        start = time.perf_counter()
        result = action()
        timings[key] = (time.perf_counter() - start) * 1000.0
        return result

    def run_on_image(
        self,
        image: np.ndarray,
        timings: Optional[MutableMapping[str, float]] = None,
    ) -> InferenceResultPayload:
        if timings is not None:
            patches, detected = self._timed_call(timings, "crop", lambda: self.cropper.crop_with_count(image))
            if isinstance(self.cropper, YoloCircleCropper):
                for key in (
                    "crop_yolo_detect_ms",
                    "crop_box_to_circle_ms",
                    "crop_mask_and_cut_ms",
                    "crop_sort_ms",
                ):
                    timings[key] = float(self.cropper.last_timing_ms.get(key, 0.0))
        else:
            patches, detected = self.cropper.crop_with_count(image)
        expected = self.config.layout.count
        if self.anomaly is None:
            raise RuntimeError(f"Anomaly model not available: {self._anomaly_error or 'unknown error'}")

        anomaly = None
        if patches:
            if timings is not None:
                anomaly = self._timed_call(timings, "anomaly", lambda: self.anomaly.infer([p.image for p in patches]))
            else:
                anomaly = self.anomaly.infer([p.image for p in patches])
        LOGGER.info(
            "Inference cycle summary: algo=%s detected=%s expected=%s patches=%d anomaly_ms=%.2f",
            (self.config.models.algo or "INP").upper(),
            detected,
            expected,
            len(patches),
            float(anomaly.inference_ms) if anomaly is not None else 0.0,
        )

        algo = (self.config.models.algo or "INP").upper()
        threshold = float(self._active_recipe_threshold)
        if algo == "GLASS":
            model_path = self.config.models.glass.path
        else:
            model_path = self.config.models.inp.path

        if timings is not None:
            timings["anomaly_model"] = float(anomaly.inference_ms) if anomaly is not None else 0.0

        mm_per_pixel = max(float(getattr(self.config.measurement, "mm_per_pixel", 0.01)), 1e-9)
        diameters_mm = [
            (None if patch.diameter_px is None else float(patch.diameter_px) * mm_per_pixel)
            for patch in patches
        ]
        diameter_min_mm, diameter_max_mm = self._resolve_active_diameter_thresholds()

        statuses: List[str] = []
        if anomaly is not None:
            for patch, score, diameter_mm in zip(patches, anomaly.scores, diameters_mm):
                score_ok = float(score) <= threshold
                diameter_ok = (
                    diameter_mm is not None
                    and float(diameter_min_mm) <= float(diameter_mm) <= float(diameter_max_mm)
                )
                statuses.append("OK" if score_ok and diameter_ok else "NG")
        ng_total = sum(1 for status in statuses if status == "NG")

        yolo_result = None
        if self.yolo is not None:
            try:
                yolo_result = self.yolo.detect(image)
            except Exception as exc:
                LOGGER.error("YOLO inference failed: %s", exc)

        overlay = self.build_overlay(image, patches, statuses, yolo_result)
        return InferenceResultPayload(
            raw_image=image,
            overlay_image=overlay,
            patches=patches,
            anomaly_scores=anomaly.scores if anomaly is not None else [],
            diameters_mm=diameters_mm,
            statuses=statuses,
            ng_total=ng_total,
            anomaly_inference_ms=anomaly.inference_ms if anomaly is not None else 0.0,
            yolo_result=yolo_result,
            timestamp=time.time(),
            model_path=model_path,
            threshold=threshold,
            diameter_min_mm=diameter_min_mm,
            diameter_max_mm=diameter_max_mm,
            anomaly_maps=anomaly.maps if anomaly is not None else None,
            detected_circles=detected,
            expected_circles=expected,
        )

    def build_overlay(
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
            text_mode = str(getattr(self.config.layout, "overlay_index_text_mode", "auto") or "auto").strip().lower()
            if text_mode == "fixed":
                font_scale = max(0.1, float(getattr(self.config.layout, "overlay_index_font_scale", 1.0)))
                thickness = max(1, int(getattr(self.config.layout, "overlay_index_thickness", 2)))
            else:
                min_scale = max(0.1, float(getattr(self.config.layout, "overlay_index_min_scale", 0.5)))
                max_scale = max(min_scale, float(getattr(self.config.layout, "overlay_index_max_scale", 1.2)))
                divisor = max(1.0, float(getattr(self.config.layout, "overlay_index_scale_divisor", 180.0)))
                font_scale = max(min_scale, min(max_scale, box_width / divisor))
                thickness = max(1, int(round(font_scale * 2)))
            outline_extra = max(1, int(getattr(self.config.layout, "overlay_index_outline_extra", 2)))
            text_origin = (x1 + 4, y1 + int(20 * font_scale))
            cv2.putText(
                overlay,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + outline_extra,
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

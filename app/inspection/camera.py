"""Basler camera wrapper built on top of the Pylon SDK."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from app.config_loader import CameraConfig

try:
    from pypylon import pylon  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pylon = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera fails to operate."""


@dataclass
class CaptureResult:
    image: np.ndarray
    timestamp_ms: float


class BaslerCamera:
    """Thin wrapper managing a Basler camera lifecycle."""

    def __init__(self, config: CameraConfig):
        self._config = config
        self._camera = None

    def connect(self) -> None:
        if pylon is None:
            raise CameraError("pypylon is not installed; cannot connect to Basler camera")

        factory = pylon.TlFactory.GetInstance()
        if self._config.serial:
            devices = factory.EnumerateDevices()
            device = next((d for d in devices if d.GetSerialNumber() == self._config.serial), None)
            if device is None:
                raise CameraError(f"Camera with serial {self._config.serial} not found")
            self._camera = pylon.InstantCamera(factory.CreateDevice(device))
        else:
            self._camera = pylon.InstantCamera(factory.CreateFirstDevice())

        self._camera.Open()
        if self._config.exposure_us is not None:
            try:
                self._camera.ExposureTimeAbs.SetValue(float(self._config.exposure_us))  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - depends on model
                LOGGER.warning("Camera does not support ExposureTimeAbs; skipping")
        if self._config.gain is not None:
            try:
                self._camera.Gain.SetValue(float(self._config.gain))  # type: ignore[attr-defined]
            except Exception:
                LOGGER.warning("Camera does not support Gain; skipping")
        if self._config.pixel_format:
            try:
                self._camera.PixelFormat.SetValue(self._config.pixel_format)
            except Exception:
                LOGGER.warning("Unsupported pixel format %s", self._config.pixel_format)

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        LOGGER.info("Camera connected and grabbing")

    def disconnect(self) -> None:
        if self._camera is not None:
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
            finally:
                self._camera.Close()
                self._camera = None
                LOGGER.info("Camera disconnected")

    def capture(self, timeout_ms: int = 3000) -> CaptureResult:
        if self._camera is None:
            raise CameraError("Camera not connected")
        result = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        if not result.GrabSucceeded():
            raise CameraError(f"Failed to grab image: {result.ErrorDescription}")
        img = pylon.PylonImage()
        img.AttachGrabResultBuffer(result)
        converted = pylon.ImageFormatConverter()
        converted.OutputPixelFormat = pylon.PixelType_RGB8packed
        converted.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        image = converted.Convert(result)
        array = image.GetArray()
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        img.Release()
        result.Release()
        return CaptureResult(image=array, timestamp_ms=result.TimeStamp)

    def __enter__(self) -> "BaslerCamera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - resource cleanup
        self.disconnect()


class DummyCamera(BaslerCamera):
    """Camera implementation for development without hardware."""

    def __init__(self, config: CameraConfig, image_shape: tuple[int, int, int] = (1080, 1920, 3)):
        super().__init__(config)
        self._image_shape = image_shape
        self._next_value = 0

    def connect(self) -> None:  # pragma: no cover - simple stub
        LOGGER.info("Dummy camera ready with shape %s", self._image_shape)

    def disconnect(self) -> None:  # pragma: no cover
        LOGGER.info("Dummy camera disconnected")

    def capture(self, timeout_ms: int = 0) -> CaptureResult:  # pragma: no cover
        height, width, channels = self._image_shape
        x = np.linspace(0, 255, num=width, dtype=np.uint8)
        gradient = np.tile(x, (height, 1))
        image = np.stack([gradient] * channels, axis=-1)
        image = np.roll(image, self._next_value, axis=1)
        self._next_value = (self._next_value + 5) % width
        return CaptureResult(image=image, timestamp_ms=0.0)

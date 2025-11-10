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
        self._device_class: Optional[str] = None

    def connect(self) -> None:
        if pylon is None:
            raise CameraError("pypylon is not installed; cannot connect to Basler camera")

        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise CameraError("No Basler camera detected")

        selected = None
        if self._config.serial:
            selected = next((d for d in devices if d.GetSerialNumber() == self._config.serial), None)
            if selected is None:
                raise CameraError(f"Camera with serial {self._config.serial} not found")
        else:
            # Prefer USB, then GigE, else first available
            selected = next((d for d in devices if getattr(d, "GetDeviceClass", lambda: "")() == "BaslerUsb"), None)
            if selected is None:
                selected = next((d for d in devices if getattr(d, "GetDeviceClass", lambda: "")() == "BaslerGigE"), None)
            if selected is None:
                selected = devices[0]

        self._device_class = getattr(selected, "GetDeviceClass", lambda: "")( ) if selected else None
        try:
            self._camera = pylon.InstantCamera(factory.CreateDevice(selected))
        except Exception as exc:
            raise CameraError(f"Failed to create camera instance: {exc}")

        # Open and apply configuration
        self._camera.Open()

        # Load default user set if available
        try:
            if hasattr(self._camera, "UserSetSelector") and hasattr(self._camera, "UserSetLoad"):
                self._camera.UserSetSelector.SetValue("Default")  # type: ignore[attr-defined]
                self._camera.UserSetLoad.Execute()  # type: ignore[attr-defined]
        except Exception as exc:
            LOGGER.debug("Skipping user set load: %s", exc)

        # Turn off autos if present
        for feat, value in (("BalanceWhiteAuto", "Off"), ("ExposureAuto", "Off"), ("GainAuto", "Off")):
            try:
                if hasattr(self._camera, feat):
                    getattr(self._camera, feat).SetValue(value)  # type: ignore[attr-defined]
            except Exception as exc:
                LOGGER.debug("Feature %s not applied: %s", feat, exc)

        # Exposure time from config; try both APIs
        if self._config.exposure_us is not None:
            exp_val = float(self._config.exposure_us)
            applied = False
            try:
                self._camera.ExposureTimeAbs.SetValue(exp_val)  # type: ignore[attr-defined]
                applied = True
            except Exception:
                try:
                    # Some models expose ExposureTime instead
                    self._camera.ExposureTime.SetValue(exp_val)  # type: ignore[attr-defined]
                    applied = True
                except Exception:
                    LOGGER.warning("Camera does not support setting exposure time")
            if applied:
                LOGGER.info("Set exposure to %.1f us", exp_val)

        # Gain from config
        if self._config.gain is not None:
            try:
                self._camera.Gain.SetValue(float(self._config.gain))  # type: ignore[attr-defined]
            except Exception:
                LOGGER.warning("Camera does not support Gain; skipping")

        # Pixel format: prefer config; otherwise pick BayerRG8 if available
        try:
            if self._config.pixel_format:
                self._camera.PixelFormat.SetValue(self._config.pixel_format)
            else:
                symbolics = getattr(self._camera.PixelFormat, "Symbolics", [])  # type: ignore[attr-defined]
                if symbolics and "BayerRG8" in symbolics:
                    self._camera.PixelFormat.SetValue("BayerRG8")
        except Exception as exc:
            LOGGER.warning("Could not apply pixel format: %s", exc)

        # GigE-specific tuning (best-effort)
        try:
            if self._device_class == "BaslerGigE":
                if hasattr(self._camera, "MaxNumBuffer"):
                    self._camera.MaxNumBuffer = 50  # type: ignore[attr-defined]
                if hasattr(self._camera, "OutputQueueSize"):
                    self._camera.OutputQueueSize = 50  # type: ignore[attr-defined]
                if hasattr(self._camera, "GevSCPD"):
                    self._camera.GevSCPD.SetValue(10000)  # type: ignore[attr-defined]
                if hasattr(self._camera, "GevSCPSPacketSize"):
                    self._camera.GevSCPSPacketSize.SetValue(1500)  # type: ignore[attr-defined]
                if hasattr(self._camera, "AcquisitionFrameRateEnable"):
                    self._camera.AcquisitionFrameRateEnable.SetValue(True)  # type: ignore[attr-defined]
                    try:
                        # Some cameras expose AcquisitionFrameRate or ResultingFrameRateAbs
                        if hasattr(self._camera, "AcquisitionFrameRate"):
                            self._camera.AcquisitionFrameRate.SetValue(15)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception as exc:
            LOGGER.debug("GigE tuning skipped: %s", exc)

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        LOGGER.info("Camera connected (%s) and grabbing", self._device_class or "Unknown")

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
        try:
            if not result.GrabSucceeded():
                # Ensure we can read a textual description if available
                try:
                    desc = result.ErrorDescription
                except Exception:
                    desc = "Unknown camera grab error"
                raise CameraError(f"Failed to grab image: {desc}")

            # Read timestamp before releasing the grab result
            timestamp = 0.0
            try:
                timestamp = float(result.TimeStamp)
            except Exception:
                pass

            # Convert to RGB8 then to BGR for OpenCV
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_RGB8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            converted = converter.Convert(result)
            array = converted.GetArray()
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            return CaptureResult(image=array, timestamp_ms=timestamp)
        finally:
            try:
                result.Release()
            except Exception:
                pass

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

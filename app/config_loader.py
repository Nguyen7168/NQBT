"""Configuration loader and dataclass definitions for the inspection application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CameraConfig:
    serial: Optional[str] = None
    exposure_us: Optional[int] = None
    gain: Optional[float] = None
    pixel_format: Optional[str] = None


@dataclass
class PlcAddressConfig:
    busy: str
    done: str
    error: str
    trigger: str
    ack: str
    result_bits_start_word: str


@dataclass
class PlcTimeoutConfig:
    connect_ms: int = 3000
    cycle_ms: int = 5000


@dataclass
class PlcConfig:
    protocol: str
    ip: str
    port: int
    addr: PlcAddressConfig
    timeouts: PlcTimeoutConfig = field(default_factory=PlcTimeoutConfig)


@dataclass
class AnomalyModelConfig:
    path: str
    provider: str = "cuda"
    threshold: float = 0.15
    input_size: int = 256


@dataclass
class YoloModelConfig:
    enabled: bool = False
    path: Optional[str] = None
    conf_thres: float = 0.25
    iou_thres: float = 0.45


@dataclass
class ModelConfig:
    anomaly: AnomalyModelConfig
    yolo: YoloModelConfig = field(default_factory=YoloModelConfig)


@dataclass
class IOConfig:
    save_images: bool = True
    output_dir: str = "runs"
    filename_pattern: str = "{ts}_{model}_{idx:02d}_{cls}.png"


@dataclass
class LayoutConfig:
    count: int
    order: str = "row_major"
    rows: int = 4
    cols: int = 7
    patch_padding: int = 0
    # Deprecated: was used to select cropper. Grid cropper removed; always circle-based.
    crop_method: str = "circle"
    # Circle crop parameters (defaults inspired by crop_hc_23_240.py)
    circle_min_radius: int = 300
    circle_max_radius: int = 340
    circle_dp: float = 1.0
    circle_minDist: float = 600.0
    circle_param1: float = 200.0
    circle_param2: float = 9.0
    circle_radius_expand: int = 16
    circle_erode_iter: int = 5
    circle_dilate_iter: int = 5
    circle_blur_kernel: int = 5
    circle_threshold: int = 50


@dataclass
class AppConfig:
    camera: CameraConfig
    plc: PlcConfig
    models: ModelConfig
    io: IOConfig
    layout: LayoutConfig


class ConfigError(RuntimeError):
    """Raised when configuration loading fails."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse configuration file {path}: {exc}") from exc


def _require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file."""

    config_path = Path(path)
    raw = _load_yaml(config_path)

    camera = CameraConfig(**_require(raw, "camera"))

    plc_raw = _require(raw, "plc")
    addr = PlcAddressConfig(**_require(plc_raw, "addr"))
    timeouts = PlcTimeoutConfig(**plc_raw.get("timeouts", {}))
    plc = PlcConfig(
        protocol=plc_raw.get("protocol", "FINS_TCP"),
        ip=_require(plc_raw, "ip"),
        port=int(_require(plc_raw, "port")),
        addr=addr,
        timeouts=timeouts,
    )

    models_raw = _require(raw, "models")
    anomaly = AnomalyModelConfig(**_require(models_raw, "anomaly"))
    yolo = YoloModelConfig(**models_raw.get("yolo", {}))
    models = ModelConfig(anomaly=anomaly, yolo=yolo)

    io_cfg = IOConfig(**raw.get("io", {}))
    layout = LayoutConfig(**_require(raw, "layout"))

    return AppConfig(camera=camera, plc=plc, models=models, io=io_cfg, layout=layout)

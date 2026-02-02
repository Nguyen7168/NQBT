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
    ready: str
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
    log_raw_response: bool = False
    trigger_poll_interval_ms: int = 50


@dataclass
class InpModelConfig:
    path: str
    provider: str = "cuda"
    input_size: int = 256
    inp_threshold: float = 0.15
    inp_blur_kernel: int = 5
    inp_blur_sigma: float = 4.0
    inp_max_ratio: float = 0.0  # 0 -> use max; otherwise mean of top-k ratio
    inp_bin_thresh: float = 0.2


@dataclass
class GlassModelConfig:
    path: str
    provider: str = "cuda"
    input_size: int = 288
    glass_threshold: float = 0.5
    glass_batch: int = 8
    glass_blur_kernel: int = 33
    glass_blur_sigma: float = 4.0
    glass_norm_eps: float = 1e-8
    glass_bin_thresh: float = 0.8


@dataclass
class YoloModelConfig:
    enabled: bool = False
    path: Optional[str] = None
    conf_thres: float = 0.25
    iou_thres: float = 0.45


@dataclass
class ModelConfig:
    algo: str = "INP"  # "INP" or "GLASS"
    inp: InpModelConfig = field(default_factory=lambda: InpModelConfig(path=""))
    glass: GlassModelConfig = field(default_factory=lambda: GlassModelConfig(path=""))
    yolo: YoloModelConfig = field(default_factory=YoloModelConfig)


@dataclass
class IOConfig:
    save_images: bool = True
    output_dir: str = "runs"
    raw_dir: str = "runs/raw"
    crops_dir: str = "runs/crops"
    filename_pattern: str = "{ts}_{model}_{idx:02d}_{cls}.png"
    save_heatmap: bool = False
    save_binary: bool = False
    save_crops: bool = False


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
    circle_radius_expand: int = 16
    circle_erode_iter: int = 0
    circle_dilate_iter: int = 0
    circle_blur_kernel: int = 5
    circle_threshold: int = 50
    circle_use_otsu: bool = False
    circle_invert: bool = False
    # Watershed-based split parameters
    ws_scale: float = 0.5
    ws_dist_ratio: float = 0.4
    ws_open_ksize: int = 3
    ws_open_iter: int = 1
    ws_bg_dilate_iter: int = 2
    ws_dist_mask_size: int = 5
    ws_min_area_px: int = 2000
    ws_max_contour_pts: int = 800


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
        log_raw_response=bool(plc_raw.get("log_raw_response", False)),
        trigger_poll_interval_ms=int(plc_raw.get("trigger_poll_interval_ms", 50)),
    )

    models_raw = _require(raw, "models")
    algo = models_raw.get("algo", "INP")
    # Support both new separated config and legacy single anomaly block
    if "inp" in models_raw and "glass" in models_raw:
        inp = InpModelConfig(**_require(models_raw, "inp"))
        glass = GlassModelConfig(**_require(models_raw, "glass"))
    elif "anomaly" in models_raw:
        # Legacy: map common fields into INP; provide a minimal GLASS with same path
        legacy = _require(models_raw, "anomaly")
        inp = InpModelConfig(
            path=_require(legacy, "path"),
            provider=legacy.get("provider", "cuda"),
            input_size=int(legacy.get("input_size", 256)),
            inp_threshold=float(legacy.get("inp_threshold", legacy.get("threshold", 0.15))),
            inp_blur_kernel=int(legacy.get("inp_blur_kernel", 5)),
            inp_blur_sigma=float(legacy.get("inp_blur_sigma", 4.0)),
            inp_max_ratio=float(legacy.get("inp_max_ratio", 0.0)),
            inp_bin_thresh=float(legacy.get("inp_bin_thresh", 0.2)),
        )
        glass = GlassModelConfig(
            path=_require(legacy, "path"),
            provider=legacy.get("provider", "cuda"),
            input_size=int(legacy.get("input_size", 288)),
            glass_threshold=float(legacy.get("glass_threshold", 0.5)),
            glass_batch=int(legacy.get("glass_batch", 8)),
            glass_blur_kernel=int(legacy.get("glass_blur_kernel", 33)),
            glass_blur_sigma=float(legacy.get("glass_blur_sigma", 4.0)),
            glass_norm_eps=float(legacy.get("glass_norm_eps", 1e-8)),
            glass_bin_thresh=float(legacy.get("glass_bin_thresh", 0.8)),
        )
    else:
        raise ConfigError("Missing models.inp and models.glass sections")
    yolo = YoloModelConfig(**models_raw.get("yolo", {}))
    models = ModelConfig(algo=algo, inp=inp, glass=glass, yolo=yolo)

    io_cfg = IOConfig(**raw.get("io", {}))
    layout = LayoutConfig(**_require(raw, "layout"))

    return AppConfig(camera=camera, plc=plc, models=models, io=io_cfg, layout=layout)

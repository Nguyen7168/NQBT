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
    reverse_x: bool = False
    reverse_y: bool = False
    reconnect_enabled: bool = True
    reconnect_interval_ms: int = 2000
    reconnect_backoff_multiplier: float = 1.5
    reconnect_max_interval_ms: int = 10000
    reconnect_fail_streak_threshold: int = 3
    reconnect_recover_stable_success_count: int = 2


@dataclass
class PlcAddressConfig:
    busy: str
    done: str
    error: str
    ready: str
    run: str
    trigger: str
    ack: str
    result_bits_start_word: str
    model_select_word: Optional[str] = None
    model_current_word: Optional[str] = None
    sample_trigger: Optional[str] = None
    mode_request_word: Optional[str] = None
    mode_current_word: Optional[str] = None
    mirror_result_word: Optional[str] = None
    detected_count_word: Optional[str] = None
    camera_capture_done: Optional[str] = None


@dataclass
class PlcTimeoutConfig:
    connect_ms: int = 3000
    response_ms: int = 1000
    cycle_ms: int = 5000
    ack_clear_ms: int = 2000


@dataclass
class PlcConfig:
    protocol: str
    ip: str
    port: int
    addr: PlcAddressConfig
    timeouts: PlcTimeoutConfig = field(default_factory=PlcTimeoutConfig)
    log_raw_request: bool = False
    log_raw_response: bool = False
    trigger_poll_interval_ms: int = 50
    trigger_min_interval_ms: int = 100
    trigger_high_stable_ms: int = 80
    trigger_low_stable_ms: int = 80
    trigger_cooldown_ms: int = 300
    enable_plc_trigger: bool = True
    model_poll_interval_ms: int = 200
    model_stable_ms: int = 200
    sample_trigger_poll_interval_ms: int = 50
    poll_error_backoff_ms: int = 1000
    reconnect_enabled: bool = True
    reconnect_interval_ms: int = 3000
    reconnect_backoff_multiplier: float = 1.5
    reconnect_max_interval_ms: int = 30000
    reconnect_consecutive_fail_to_degraded: int = 2
    reconnect_recover_stable_success_count: int = 2
    reconnect_switch_to_mock_on_startup_fail: bool = True
    reconnect_keep_mock_until_real_stable: bool = True
    reconnect_allow_plc_trigger_in_mock: bool = False
    done_hold_ms: int = 0


@dataclass
class RecipeConfig:
    code: int
    inp_model_path: str
    glass_model_path: str
    name: Optional[str] = None
    threshold: float = 0.5
    crop_yolo_path: Optional[str] = None
    diameter_min_mm: float = 0.0
    diameter_max_mm: float = 99999.0


@dataclass
class InpModelConfig:
    path: str
    provider: str = "cuda"
    input_size: int = 256
    inp_mode: str = "default"  # "default" or "legacy_script"
    inp_threshold: float = 0.15
    inp_blur_kernel: int = 5
    inp_blur_sigma: float = 4.0
    inp_max_ratio: float = 0.0  # 0 -> use max; otherwise mean of top-k ratio
    inp_bin_thresh: float = 0.2
    diameter_min_mm: float = 0.0
    diameter_max_mm: float = 99999.0


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
    diameter_min_mm: float = 0.0
    diameter_max_mm: float = 99999.0


@dataclass
class YoloModelConfig:
    enabled: bool = False
    path: Optional[str] = None
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    crop_enabled: bool = False
    crop_path: Optional[str] = None
    crop_conf_thres: float = 0.5
    crop_iou_thres: float = 0.7
    crop_imgsz: int = 960
    crop_device: str = "cuda:0"
    crop_classes: Optional[List[int]] = None


@dataclass
class ModelConfig:
    algo: str = "INP"  # "INP" or "GLASS"
    inp: InpModelConfig = field(default_factory=lambda: InpModelConfig(path=""))
    glass: GlassModelConfig = field(default_factory=lambda: GlassModelConfig(path=""))
    yolo: YoloModelConfig = field(default_factory=YoloModelConfig)
    active_recipe_code: Optional[int] = None
    recipes: List[RecipeConfig] = field(default_factory=list)


@dataclass
class IOConfig:
    save_images: bool = True
    save_only_ng: bool = False
    output_dir: str = "runs"
    raw_dir: str = "runs/raw"
    crops_dir: str = "runs/crops"
    filename_pattern: str = "{ts}_{model}_{idx:02d}_{cls}.png"
    save_heatmap: bool = False
    save_binary: bool = False
    save_crops: bool = False
    save_ng_crops_only: bool = False


@dataclass
class LayoutConfig:
    count: int
    order: str = "row_major"
    rows: int = 4
    cols: int = 7
    patch_padding: int = 0
    # Crop strategy: "circle" (legacy watershed) or "yolo_circle" (YOLO bbox -> inscribed circle).
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
    yolo_circle_radius_expand: int = 0
    yolo_crop_fallback_to_circle: bool = True
    # Watershed-based split parameters
    ws_scale: float = 0.5
    ws_dist_ratio: float = 0.4
    ws_open_ksize: int = 3
    ws_open_iter: int = 1
    ws_bg_dilate_iter: int = 2
    ws_dist_mask_size: int = 5
    ws_min_area_px: int = 2000
    ws_max_contour_pts: int = 800
    # Overlay index text (number at top-left of each detected part)
    # mode: "auto" derives scale from ROI width; "fixed" uses fixed scale/thickness.
    overlay_index_text_mode: str = "auto"
    overlay_index_font_scale: float = 1.0
    overlay_index_thickness: int = 2
    overlay_index_outline_extra: int = 2
    overlay_index_min_scale: float = 0.5
    overlay_index_max_scale: float = 1.2
    overlay_index_scale_divisor: float = 180.0


@dataclass
class WindowConfig:
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class InferenceServiceConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    timeout_ms: int = 30000


@dataclass
class MeasurementConfig:
    mm_per_pixel: float = 0.01


@dataclass
class MirrorConfig:
    target_ring: int = 10
    bright_thresh: int = 160
    crop_ratio: float = 0.44
    band_half_width: int = 2
    profile_smooth: int = 9
    min_peak_value: float = 6.0
    min_peak_distance: int = 28
    max_ring_ratio: float = 0.48
    center_refine: int = 10
    diameter_min: float = 0.0
    diameter_max: float = 99999.0


@dataclass
class AppConfig:
    camera: CameraConfig
    plc: PlcConfig
    models: ModelConfig
    io: IOConfig
    layout: LayoutConfig
    inference_service: InferenceServiceConfig = field(default_factory=InferenceServiceConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
    app_title: Optional[str] = None
    window: Optional[WindowConfig] = None
    sample_image_root: str = "samples"
    timing_log_enabled: bool = False


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

    camera_raw = _require(raw, "camera")
    camera = CameraConfig(
        serial=camera_raw.get("serial"),
        exposure_us=(int(camera_raw["exposure_us"]) if camera_raw.get("exposure_us") is not None else None),
        gain=(float(camera_raw["gain"]) if camera_raw.get("gain") is not None else None),
        pixel_format=camera_raw.get("pixel_format"),
        reverse_x=bool(camera_raw.get("reverse_x", False)),
        reverse_y=bool(camera_raw.get("reverse_y", False)),
        reconnect_enabled=bool(camera_raw.get("reconnect_enabled", True)),
        reconnect_interval_ms=int(camera_raw.get("reconnect_interval_ms", 2000)),
        reconnect_backoff_multiplier=float(camera_raw.get("reconnect_backoff_multiplier", 1.5)),
        reconnect_max_interval_ms=int(camera_raw.get("reconnect_max_interval_ms", 10000)),
        reconnect_fail_streak_threshold=int(camera_raw.get("reconnect_fail_streak_threshold", 3)),
        reconnect_recover_stable_success_count=int(camera_raw.get("reconnect_recover_stable_success_count", 2)),
    )

    plc_raw = _require(raw, "plc")
    addr = PlcAddressConfig(**_require(plc_raw, "addr"))
    timeouts = PlcTimeoutConfig(**plc_raw.get("timeouts", {}))
    plc = PlcConfig(
        protocol=plc_raw.get("protocol", "FINS_TCP"),
        ip=_require(plc_raw, "ip"),
        port=int(_require(plc_raw, "port")),
        addr=addr,
        timeouts=timeouts,
        log_raw_request=bool(plc_raw.get("log_raw_request", False)),
        log_raw_response=bool(plc_raw.get("log_raw_response", False)),
        trigger_poll_interval_ms=int(plc_raw.get("trigger_poll_interval_ms", 50)),
        trigger_min_interval_ms=int(plc_raw.get("trigger_min_interval_ms", 100)),
        trigger_high_stable_ms=int(plc_raw.get("trigger_high_stable_ms", 80)),
        trigger_low_stable_ms=int(plc_raw.get("trigger_low_stable_ms", 80)),
        trigger_cooldown_ms=int(plc_raw.get("trigger_cooldown_ms", 300)),
        enable_plc_trigger=bool(plc_raw.get("enable_plc_trigger", True)),
        model_poll_interval_ms=int(plc_raw.get("model_poll_interval_ms", 200)),
        model_stable_ms=int(plc_raw.get("model_stable_ms", 200)),
        sample_trigger_poll_interval_ms=int(plc_raw.get("sample_trigger_poll_interval_ms", 50)),
        poll_error_backoff_ms=int(plc_raw.get("poll_error_backoff_ms", 1000)),
        reconnect_enabled=bool(plc_raw.get("reconnect_enabled", True)),
        reconnect_interval_ms=int(plc_raw.get("reconnect_interval_ms", 3000)),
        reconnect_backoff_multiplier=float(plc_raw.get("reconnect_backoff_multiplier", 1.5)),
        reconnect_max_interval_ms=int(plc_raw.get("reconnect_max_interval_ms", 30000)),
        reconnect_consecutive_fail_to_degraded=int(plc_raw.get("reconnect_consecutive_fail_to_degraded", 2)),
        reconnect_recover_stable_success_count=int(plc_raw.get("reconnect_recover_stable_success_count", 2)),
        reconnect_switch_to_mock_on_startup_fail=bool(plc_raw.get("reconnect_switch_to_mock_on_startup_fail", True)),
        reconnect_keep_mock_until_real_stable=bool(plc_raw.get("reconnect_keep_mock_until_real_stable", True)),
        reconnect_allow_plc_trigger_in_mock=bool(plc_raw.get("reconnect_allow_plc_trigger_in_mock", False)),
        done_hold_ms=int(plc_raw.get("done_hold_ms", 0)),
    )

    models_raw = _require(raw, "models")
    algo = str(models_raw.get("algo", "INP")).strip().upper()
    if algo not in {"INP", "GLASS"}:
        raise ConfigError(f"Unsupported models.algo: {algo}. Expected 'INP' or 'GLASS'.")
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
            diameter_min_mm=float(legacy.get("diameter_min_mm", 0.0)),
            diameter_max_mm=float(legacy.get("diameter_max_mm", 99999.0)),
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
            diameter_min_mm=float(legacy.get("diameter_min_mm", 0.0)),
            diameter_max_mm=float(legacy.get("diameter_max_mm", 99999.0)),
        )
    else:
        raise ConfigError("Missing models.inp and models.glass sections")
    yolo = YoloModelConfig(**models_raw.get("yolo", {}))
    recipes = []
    recipes_raw = models_raw.get("recipes", [])
    if not isinstance(recipes_raw, list) or not recipes_raw:
        raise ConfigError("Missing required models.recipes (list)")
    for entry in recipes_raw:
        if entry.get("threshold") is None:
            code = entry.get("code", "?")
            raise ConfigError(f"Missing required threshold for recipe code: {code}")
        inp_model_path = str(entry.get("inp_model_path", "")).strip()
        glass_model_path = str(entry.get("glass_model_path", "")).strip()
        if not inp_model_path or not glass_model_path:
            code = entry.get("code", "?")
            raise ConfigError(f"Recipe code {code} must define both inp_model_path and glass_model_path")
        recipe_entry = dict(entry)
        recipe_entry["threshold"] = float(recipe_entry["threshold"])
        recipe_entry["inp_model_path"] = inp_model_path
        recipe_entry["glass_model_path"] = glass_model_path
        recipe_entry["diameter_min_mm"] = float(recipe_entry.get("diameter_min_mm", 0.0))
        recipe_entry["diameter_max_mm"] = float(recipe_entry.get("diameter_max_mm", 99999.0))
        recipes.append(RecipeConfig(**recipe_entry))
    models = ModelConfig(
        algo=algo,
        inp=inp,
        glass=glass,
        yolo=yolo,
        active_recipe_code=(int(models_raw["active_recipe_code"]) if models_raw.get("active_recipe_code") is not None else None),
        recipes=recipes,
    )

    io_cfg = IOConfig(**raw.get("io", {}))
    layout = LayoutConfig(**_require(raw, "layout"))
    window_raw = raw.get("window")
    window_cfg = WindowConfig(**window_raw) if isinstance(window_raw, dict) else None
    inference_service_raw = raw.get("inference_service", {})
    inference_service_cfg = (
        InferenceServiceConfig(**inference_service_raw)
        if isinstance(inference_service_raw, dict)
        else InferenceServiceConfig()
    )
    measurement_raw = raw.get("measurement", {})
    measurement_cfg = MeasurementConfig(**measurement_raw) if isinstance(measurement_raw, dict) else MeasurementConfig()
    mirror_raw = raw.get("mirror", {})
    mirror_cfg = MirrorConfig(**mirror_raw) if isinstance(mirror_raw, dict) else MirrorConfig()

    return AppConfig(
        camera=camera,
        plc=plc,
        models=models,
        io=io_cfg,
        layout=layout,
        inference_service=inference_service_cfg,
        measurement=measurement_cfg,
        mirror=mirror_cfg,
        app_title=(str(raw.get("app_title")).strip() if raw.get("app_title") is not None else None),
        window=window_cfg,
        sample_image_root=str(raw.get("sample_image_root", "samples")),
        timing_log_enabled=bool(raw.get("timing_log_enabled", False)),
    )

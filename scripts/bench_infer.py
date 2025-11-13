r"""Benchmark anomaly pipeline (crop -> infer) with detailed timings.

Usage examples:
  python scripts/bench_infer.py --config config.yaml --images "D:\AI\NQBT\app\models\20250510081249.jpg" --algo GLASS
  python scripts/bench_infer.py --config config.yaml --images patch1.png --single-patch --algo GLASS
  python scripts/bench_infer.py --config config.yaml --images app\models\20250510081249.jpg --algo GLASS

Outputs per image include timings for load, crop, infer (wall/model), overlay, total.
Optional `--out` to save raw/overlay per image for inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config_loader import load_config
from app.inspection.cropping import CircleCropper, CropResult
from app.models.anomaly import AnomalyDetector
try:
    import onnxruntime as ort
except Exception:
    ort = None
from app.utils import ensure_dir, save_image


def build_overlay(image: np.ndarray, patches: list[CropResult], statuses: list[str]) -> np.ndarray:
    if not patches:
        return image.copy()
    overlay = image.copy()
    for patch, status in zip(patches, statuses):
        x1, y1, x2, y2 = patch.bbox
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
    return overlay


def bench_image(
    path: Path,
    cropper: CircleCropper | None,
    detector: AnomalyDetector,
    threshold: float,
    save_dir: Path | None,
) -> dict[str, float | int | str]:
    timings: dict[str, float | int | str] = {"image": str(path)}

    t0 = perf_counter()
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    timings["load_ms"] = (perf_counter() - t0) * 1000.0

    if cropper is None:
        patches = [CropResult(index=1, image=image, bbox=(0, 0, image.shape[1], image.shape[0]))]
        timings["crop_ms"] = 0.0
    else:
        t_crop = perf_counter()
        patches = cropper.crop(image)
        timings["crop_ms"] = (perf_counter() - t_crop) * 1000.0

    t_infer = perf_counter()
    anomaly = detector.infer([p.image for p in patches])
    timings["infer_wall_ms"] = (perf_counter() - t_infer) * 1000.0
    timings["infer_model_ms"] = float(anomaly.inference_ms)

    statuses = ["OK" if score <= threshold else "NG" for score in anomaly.scores]
    timings["patches"] = len(patches)
    timings["ng"] = sum(1 for s in statuses if s == "NG")

    t_overlay = perf_counter()
    overlay = build_overlay(image, patches, statuses)
    timings["overlay_ms"] = (perf_counter() - t_overlay) * 1000.0
    timings["total_ms"] = timings["load_ms"] + timings["crop_ms"] + timings["infer_wall_ms"] + timings["overlay_ms"]

    if save_dir is not None:
        ensure_dir(save_dir)
        save_image(save_dir / "raw.png", image)
        save_image(save_dir / "overlay.png", overlay)
        meta = {
            "scores": [float(s) for s in anomaly.scores],
            "statuses": statuses,
            "threshold": threshold,
            "timings": {k: v for k, v in timings.items() if k.endswith("_ms")},
        }
        (save_dir / "results.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark anomaly inference pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--images", nargs="+", help="Paths to images (full frame or patches)")
    parser.add_argument("--single-patch", action="store_true", help="Treat each image as a single cropped patch (skip CircleCropper)")
    parser.add_argument("--algo", choices=["INP", "GLASS"], default=None, help="Override model algo (default from config)")
    parser.add_argument("--out", default=None, help="Directory to save raw/overlay outputs per image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.algo:
        cfg.models.algo = args.algo.upper()

    detector = AnomalyDetector(cfg.models)
    cropper = None if args.single_patch else CircleCropper(cfg.layout)
    if (cfg.models.algo or "INP").upper() == "GLASS":
        threshold = float(cfg.models.glass.glass_threshold)
        model_path = cfg.models.glass.path
    else:
        threshold = float(cfg.models.inp.inp_threshold)
        model_path = cfg.models.inp.path

    provider_info = "unknown"
    if ort is not None:
        try:
            providers = ort.get_available_providers()
            provider_info = ",".join(providers)
        except Exception:
            provider_info = "n/a"
    print(
        f"Algo={cfg.models.algo} | Model={model_path} | Threshold={threshold} | AvailableProviders={provider_info}"
    )
    summary: list[dict[str, float | int | str]] = []
    out_root = Path(args.out) if args.out else None
    if out_root:
        ensure_dir(out_root)

    for img_path in args.images:
        img_path = Path(img_path)
        save_dir = out_root / img_path.stem if out_root else None
        timings = bench_image(img_path, cropper, detector, threshold, save_dir)
        summary.append(timings)
        print(
            f"{img_path.name}: load={timings['load_ms']:.1f}ms | crop={timings['crop_ms']:.1f}ms | "
            f"infer_wall={timings['infer_wall_ms']:.1f}ms (model={timings['infer_model_ms']:.1f}ms) | "
            f"overlay={timings['overlay_ms']:.1f}ms | patches={timings['patches']} | NG={timings['ng']}"
        )

    total_wall = sum(t.get("total_ms", 0.0) for t in summary)
    print(f"\nProcessed {len(summary)} image(s), total_wall={total_wall:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

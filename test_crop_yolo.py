"""Quick utility to test YOLO-based circle cropping on a folder of images.

Usage:
  python test_crop_yolo.py --config config.yaml --source samples/23-233HA-E --output runs/test_crop_yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO circle cropping")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--output", default="runs/test_crop_yolo", help="Output directory")
    parser.add_argument("--force-circle", action="store_true", help="Force legacy circle cropper")
    return parser.parse_args()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    args = parse_args()

    import cv2

    from app.config_loader import load_config
    from app.inspection.cropping import CircleCropper, YoloCircleCropper
    from app.models.yolo import YoloDetector

    cfg = load_config(args.config)
    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    use_yolo = (cfg.layout.crop_method or "circle").lower() == "yolo_circle" and not args.force_circle

    if use_yolo:
        path = (cfg.models.yolo.crop_path or "").strip()
        if not path:
            print("[ERROR] models.yolo.crop_path is empty")
            return 1
        detector = YoloDetector(
            path,
            cfg.models.yolo.crop_conf_thres,
            cfg.models.yolo.crop_iou_thres,
            imgsz=cfg.models.yolo.crop_imgsz,
        )
        cropper = YoloCircleCropper(cfg.layout, detector)
        print(f"[INFO] Using YOLO crop model: {path}")
    else:
        cropper = CircleCropper(cfg.layout)
        print("[INFO] Using legacy circle cropper")

    images = _iter_images(src)
    if not images:
        print(f"[ERROR] No images found at: {src}")
        return 1

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        patches, detected = cropper.crop_with_count(image)
        expected = int(cfg.layout.count)
        print(f"[INFO] {img_path.name}: detected={detected}, expected={expected}, saved={len(patches)}")

        stem_out = out / img_path.stem
        stem_out.mkdir(parents=True, exist_ok=True)
        for patch in patches:
            save_path = stem_out / f"{patch.index:02d}.png"
            cv2.imwrite(str(save_path), patch.image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

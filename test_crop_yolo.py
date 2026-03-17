"""Quick utility to test YOLO-based circle cropping on a folder of images.

Usage:
  python test_crop_yolo.py --config config.yaml --source samples/23-233HA-E --output runs/test_crop_yolo
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO circle cropping")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--output", default="runs/test_crop_yolo", help="Output directory")
    parser.add_argument("--recipe-code", type=int, default=None, help="Recipe code to pick crop_yolo_path")
    parser.add_argument("--recipe-name", default=None, help="Recipe name to pick crop_yolo_path")
    parser.add_argument("--debug-boxes", action="store_true", help="Enable per-box debug logs (w,h,r)")
    parser.add_argument(
        "--save-mode",
        choices=("per_image", "flat"),
        default="per_image",
        help="Save crops per input image folder or flatten all crops into one output folder",
    )
    parser.add_argument("--force-circle", action="store_true", help="Force legacy circle cropper")
    return parser.parse_args()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _resolve_recipe_crop_path(cfg, src: Path, recipe_code: int | None, recipe_name: str | None) -> str | None:
    recipes = list(getattr(cfg.models, "glass_recipes", []))
    if not recipes:
        return None

    target = None
    if recipe_code is not None:
        target = next((r for r in recipes if int(r.code) == int(recipe_code)), None)
    elif recipe_name:
        name = recipe_name.strip().lower()
        target = next((r for r in recipes if (r.name or "").strip().lower() == name), None)
    else:
        # Auto by source folder name (common test flow: samples/<recipe_name>)
        src_name = (src.parent.name if src.is_file() else src.name).strip().lower()
        target = next((r for r in recipes if (r.name or "").strip().lower() == src_name), None)
        # Fallback by active recipe code
        if target is None and cfg.models.active_recipe_code is not None:
            target = next((r for r in recipes if int(r.code) == int(cfg.models.active_recipe_code)), None)

    if target is None:
        return None
    return (target.crop_yolo_path or "").strip() or None


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug_boxes else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
        recipe_path = _resolve_recipe_crop_path(cfg, src, args.recipe_code, args.recipe_name)
        path = recipe_path or (cfg.models.yolo.crop_path or "").strip()
        if not path:
            print("[ERROR] Cannot resolve crop_yolo_path from recipe or models.yolo.crop_path")
            return 1
        detector = YoloDetector(
            path,
            cfg.models.yolo.crop_conf_thres,
            cfg.models.yolo.crop_iou_thres,
            imgsz=cfg.models.yolo.crop_imgsz,
            device=getattr(cfg.models.yolo, "crop_device", "cuda:0"),
            classes=getattr(cfg.models.yolo, "crop_classes", None),
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

        if args.save_mode == "per_image":
            stem_out = out / img_path.stem
            stem_out.mkdir(parents=True, exist_ok=True)
            for patch in patches:
                save_path = stem_out / f"{patch.index:02d}.png"
                cv2.imwrite(str(save_path), patch.image)
        else:
            # Flat save mode for training-data collection.
            for patch in patches:
                save_path = out / f"{img_path.stem}_{patch.index:02d}.png"
                cv2.imwrite(str(save_path), patch.image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

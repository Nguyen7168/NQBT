r"""Standalone test: crop bearings via CircleCropper and save overlays/patches.

Usage:
  python scripts/test_crop.py --config config.yaml --images path1 path2 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.config_loader import load_config
from app.inspection.cropping import CircleCropper
from app.utils import ensure_dir, save_image


def draw_bboxes(image: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = image.copy()
    for x1, y1, x2, y2 in bboxes:
        overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 3)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Test circle-based cropping")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--images", nargs="+", help="Paths to input images")
    parser.add_argument("--out", default=None, help="Optional override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cropper = CircleCropper(cfg.layout)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out or cfg.io.output_dir) / f"test_crop_{ts}"
    ensure_dir(base_out)

    for img_path in args.images:
        p = Path(img_path)
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skip unreadable image: {p}")
            continue
        patches = cropper.crop(img)
        bboxes = [c.bbox for c in patches]
        overlay = draw_bboxes(img, bboxes)

        stem = p.stem
        out_dir = base_out / stem
        ensure_dir(out_dir)
        save_image(out_dir / "overlay.png", overlay)
        save_image(out_dir / "raw.png", img)
        # Save individual patches
        patches_dir = out_dir / "patches"
        ensure_dir(patches_dir)
        for pr in patches:
            save_image(patches_dir / f"patch_{pr.index:02d}.png", pr.image)
        print(f"Saved crop results for {p} -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

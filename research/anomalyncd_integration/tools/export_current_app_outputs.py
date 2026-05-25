#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.config_loader import load_config
from app.inspection.inference_pipeline import InferencePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def save_map(map_arr: np.ndarray, out_path: Path) -> None:
    norm = np.clip(map_arr, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), u8)


def save_binary(map_arr: np.ndarray, out_path: Path, threshold: float = 0.5) -> None:
    m = (map_arr >= threshold).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), m)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline export from current app pipeline (no PLC/camera).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-root", required=True, help="Folder containing images; subfolder name can be used as true_label")
    parser.add_argument("--output-root", default="research/anomalyncd_integration/outputs")
    parser.add_argument("--csv-path", default="research/anomalyncd_integration/reports/current_app_export.csv")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = InferencePipeline(cfg)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    crops_root = output_root / "crops"
    maps_root = output_root / "maps"
    masks_root = output_root / "masks"
    for d in (crops_root, maps_root, masks_root, Path(args.csv_path).parent):
        d.mkdir(parents=True, exist_ok=True)

    images = iter_images(input_root)
    if not images:
        raise RuntimeError(f"No images found under: {input_root}")

    with Path(args.csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "crop_path", "map_path", "mask_path", "patch_index", "anomaly_score", "status", "true_label"])

        for image_path in images:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            true_label = image_path.parent.name
            payload = pipeline.run_on_image(img)
            map_list = payload.anomaly_maps if payload.anomaly_maps is not None else [None] * len(payload.patches)

            for patch, score, status, amap in zip(payload.patches, payload.anomaly_scores, payload.statuses, map_list):
                stem = f"{image_path.stem}_p{patch.index:02d}"
                crop_path = crops_root / f"{stem}.png"
                cv2.imwrite(str(crop_path), patch.image)

                map_path = ""
                mask_path = ""
                if amap is not None:
                    map_file = maps_root / f"{stem}_map.png"
                    save_map(amap, map_file)
                    map_path = str(map_file)
                    mask_file = masks_root / f"{stem}_mask.png"
                    save_binary(amap, mask_file, threshold=args.mask_threshold)
                    mask_path = str(mask_file)

                writer.writerow([
                    str(image_path),
                    str(crop_path),
                    map_path,
                    mask_path,
                    patch.index,
                    float(score),
                    status,
                    true_label,
                ])

    print(f"Export finished. CSV: {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

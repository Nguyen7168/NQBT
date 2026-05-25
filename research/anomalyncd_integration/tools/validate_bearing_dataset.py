#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2

CLASS_DIRS = [
    "OK",
    "01_thieu_lieu",
    "02_loi_khac",
    "03_nut_me_vo",
    "04_dinh_di_vat",
    "05_tray_can",
    "06_bien_mau_ri_set",
    "07_sai_di_dang",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def validate_split(split_root: Path, min_images: int) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    for class_dir in CLASS_DIRS:
        class_path = split_root / class_dir
        if not class_path.exists() or not class_path.is_dir():
            rows.append({"class": class_dir, "count": 0, "readable": 0, "unreadable": 0, "missing_dir": 1})
            warnings.append(f"Missing class folder: {class_path}")
            continue

        count = readable = unreadable = 0
        for img_path in iter_images(class_path):
            count += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                unreadable += 1
            else:
                readable += 1
        if readable < min_images:
            warnings.append(f"Class '{class_dir}' has only {readable} readable images (<{min_images}).")
        rows.append({"class": class_dir, "count": count, "readable": readable, "unreadable": unreadable, "missing_dir": 0})
    return rows, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate bearing dataset structure and image readability.")
    parser.add_argument("--dataset-root", type=Path, default=Path("research/anomalyncd_integration/dataset_bac_truc"))
    parser.add_argument("--split", choices=["raw_full_images", "crops_labeled"], default="crops_labeled")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--reports-dir", type=Path, default=Path("research/anomalyncd_integration/reports"))
    args = parser.parse_args()

    split_root = args.dataset_root / args.split
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_dir / "dataset_summary.csv"
    md_path = args.reports_dir / "dataset_validation_report.md"

    rows, warnings = validate_split(split_root, args.min_images)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "count", "readable", "unreadable", "missing_dir"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": args.split, **row})

    ts = datetime.now(timezone.utc).isoformat()
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Dataset Validation Report\n\nGenerated: {ts}\n\n")
        f.write(f"- Dataset root: `{args.dataset_root}`\n")
        f.write(f"- Split: `{args.split}`\n")
        f.write(f"- Min readable images warning threshold: `{args.min_images}`\n\n")
        f.write("## Summary\n\n")
        f.write("| Class | Files | Readable | Unreadable | Missing dir |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['class']} | {row['count']} | {row['readable']} | {row['unreadable']} | {row['missing_dir']} |\n")
        f.write("\n## Warnings\n\n")
        if warnings:
            for w in warnings:
                f.write(f"- ⚠️ {w}\n")
        else:
            f.write("- ✅ No warnings.\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Standalone test: GLASS anomaly inference on single-part images (no crop).

Usage:
  python scripts/test_infer_glass.py --config config.yaml --images path1 path2 ...
  python scripts/test_infer_glass.py --config config.yaml --images "D:\AI\NQBT\runs\latest\test_crop_20251111_151825\20250510081249\patches\ngoaile.jpg"
Each image is assumed to be a single cropped bearing part.
Saves score, heatmap, and binary mask per image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.config_loader import load_config
from app.models.anomaly import AnomalyDetector
from app.utils import ensure_dir, save_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Test GLASS anomaly inference")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--images", nargs="+", help="Paths to input images")
    parser.add_argument("--out", default=None, help="Optional override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Force GLASS
    cfg.models.algo = "GLASS"
    detector = AnomalyDetector(cfg.models)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out or cfg.io.output_dir) / f"test_glass_{ts}"
    ensure_dir(base_out)

    thres = float(cfg.models.glass.glass_threshold)
    bin_th = float(cfg.models.glass.glass_bin_thresh)

    for img_path in args.images:
        p = Path(img_path)
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skip unreadable image: {p}")
            continue
        # Run inference directly on the single-part image
        anomaly = detector.infer([img])
        score = float(anomaly.scores[0])
        status = "OK" if score <= thres else "NG"

        stem = p.stem
        out_dir = base_out / stem
        ensure_dir(out_dir)
        save_image(out_dir / "raw.png", img)
        # Save JSON summary
        result = {
            "algo": "GLASS",
            "model": cfg.models.glass.path,
            "threshold": thres,
            "score": score,
            "status": status,
        }
        (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        # Always save heatmap and binary mask if map available
        if anomaly.maps:
            amap = anomaly.maps[0]
            gray = np.clip(amap * 255.0, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            cv2.imwrite(str(out_dir / "heatmap.png"), heat)
            binary = (amap >= bin_th).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / "binary.png"), binary)

        print(f"Saved GLASS results for {p} -> {out_dir} | score={score:.4f} ({status})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

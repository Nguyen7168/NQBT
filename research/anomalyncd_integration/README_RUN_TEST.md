# README_RUN_TEST

## 1) Tạo môi trường
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install onnxruntime  # nếu chạy INP ONNX CPU
```

## 2) Validate dataset
```bash
python research/anomalyncd_integration/tools/validate_bearing_dataset.py \
  --dataset-root research/anomalyncd_integration/dataset_bac_truc \
  --split crops_labeled --min-images 20
```

## 3) Export output từ app hiện tại (offline, không camera/PLC)
```bash
python research/anomalyncd_integration/tools/export_current_app_outputs.py \
  --config config.yaml \
  --input-root research/anomalyncd_integration/dataset_bac_truc/raw_full_images \
  --output-root research/anomalyncd_integration/outputs \
  --csv-path research/anomalyncd_integration/reports/current_app_export.csv
```

## 4) Smoke test
```bash
python research/anomalyncd_integration/tools/validate_bearing_dataset.py --help
python research/anomalyncd_integration/tools/export_current_app_outputs.py --help
```

## 5) Codex Cloud chạy được gì
- Chạy validate dataset (nếu có ảnh trong workspace).
- Sinh report CSV/Markdown.
- Soạn script export offline.

## 6) Chỉ chạy local Windows/GPU/camera
- Basler camera live capture.
- PLC handshake/run trigger thật.
- Model INP/GLASS với đường dẫn weights nội bộ chưa có trong cloud.
- Luồng tốc độ thật (cycle time) cần benchmark trên máy production.

## 7) AnomalyNCD
Do môi trường cloud hiện tại bị chặn clone GitHub (403), cần clone AnomalyNCD ở local theo `external/AnomalyNCD/README_FETCH.md`.

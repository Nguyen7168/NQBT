# AnomalyNCD Fetch Notes

Codex Cloud hiện tại không clone được GitHub URL này (HTTP 403 tunnel), nên chưa thể đồng bộ source trực tiếp vào workspace này.

## Clone local (khuyến nghị chạy trên máy local)
```bash
git clone https://github.com/HUST-SLOW/AnomalyNCD.git research/anomalyncd_integration/external/AnomalyNCD
```

## Các mục cần đọc sau khi clone
- `README.md`
- `requirements.txt`
- `scripts/anomalyncd.sh`
- `scripts/anomalyncd_test.sh`
- `configs/`
- `datasets/*_preprocess.py`

## Snapshot thông tin đã xác minh qua index công khai
- Repo có các thư mục: `assets`, `configs`, `datasets`, `examples`, `models`, `scripts`, `utils`.
- README nêu quy trình train bằng `scripts/anomalyncd.sh`, inference bằng `scripts/anomalyncd_test.sh`.
- README mô tả cần anomaly map và có tham số: `dataset_path`, `anomaly_map_path`, `binary_data_path`, `crop_data_path`, `base_data_path`.

> Lưu ý: dataset format chi tiết cho case custom bạc trục cần xác nhận trực tiếp từ source script/config sau khi clone local.

# Shared Inference Service Guide

Tài liệu này mô tả kiến trúc **Phương án B**: 2 app camera vẫn giữ camera/UI/PLC local, nhưng phần crop + anomaly inference được gom vào một service chung để tránh 2 process cùng tranh GPU.

---

## 1) Mô hình runtime

- `run.py --config config.yaml`: app camera 1
- `run.py --config config1.yaml`: app camera 2
- `run_inference_service.py --host 127.0.0.1 --port 8765`: service inference chung

Luồng xử lý:

1. App nhận trigger PLC.
2. App set `BUSY=1`, chụp ảnh camera local.
3. App gửi ảnh + runtime state sang inference service.
4. Service xếp lịch request theo một hàng đợi chung (TCP server xử lý tuần tự theo thứ tự nhận).
5. Service chạy crop/anomaly và trả kết quả.
6. App ghi kết quả về PLC, set `DONE=1`, rồi `finalize_cycle()`.

---

## 2) Phần nào ở local app, phần nào ở service

### Giữ local trong app camera
- Camera capture
- PLC handshake (`busy/done/error/ready/run/result bits/...`)
- UI / monitor / trigger worker / mode worker
- SAMPLE image picker
- MIRROR mode

### Chạy trong service chung
- Crop pipeline (`circle` / `yolo_circle`)
- Crop YOLO detect
- Anomaly inference
- Overlay build
- Trả về patch results + timing inference

---

## 3) Cấu hình

Mỗi app camera có block:

```yaml
inference_service:
  enabled: true
  host: "127.0.0.1"
  port: 8765
  timeout_ms: 30000
```

- Khi `enabled: true`, `InspectionWorker` sẽ gửi request sang service thay vì chạy inference local.
- Khi `enabled: false`, app quay về pipeline local như cũ.

---

## 4) Startup khuyến nghị

Mở service trước:

```bash
python run_inference_service.py --host 127.0.0.1 --port 8765
```

Sau đó mở 2 app:

```bash
python run.py --config config.yaml
python run.py --config config1.yaml
```

---

## 5) Ghi chú vận hành

- Service hiện tại ưu tiên **ổn định và tránh contention**: request được xử lý tuần tự trong một process chung.
- Điều này có thể làm request thứ hai phải chờ queue ngắn, nhưng đổi lại tránh việc 2 app cùng gọi YOLO crop trên `cuda:0` một lúc.
- Vì PLC handshake vẫn nằm ở local app, logic ladder hiện tại không cần đổi kiến trúc lớn.

---

## 6) Hướng mở rộng sau này

- Thêm `queue_wait_ms` riêng để monitor scheduler.
- Thêm policy fairness theo `camera_id`.
- Thêm multi-GPU routing (`cuda:0`, `cuda:1`).
- Nếu cần throughput cao hơn mới cân nhắc micro-batching.

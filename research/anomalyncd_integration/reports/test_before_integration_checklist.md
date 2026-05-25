# Test Before Integration Checklist

- [ ] Clone và chạy được AnomalyNCD độc lập trên local GPU.
- [ ] Xác nhận dataset format cuối cùng từ source AnomalyNCD (không suy đoán).
- [ ] Validate đủ dữ liệu 7 lớp NG (đạt ngưỡng tối thiểu).
- [ ] Export được crop/map/mask từ pipeline hiện tại ở chế độ offline.
- [ ] Đo latency tầng 1 (INP) riêng và tầng 2 (classifier) riêng.
- [ ] Xác nhận tầng 2 không ảnh hưởng PLC OK/NG handshake.
- [ ] Test fallback: nếu tầng 2 fail thì hệ thống vẫn trả OK/NG như cũ.
- [ ] Thiết kế schema output mới (defect_class_id/name/confidence/source) và backward compatible.

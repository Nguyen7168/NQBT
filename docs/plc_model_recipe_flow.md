# PLC Recipe/Model Selection Flow (DM1530 ↔ DM1524)

> Xem thêm tài liệu vận hành tổng hợp: `docs/plc_operation_guide.md`


## Mục tiêu
- PLC ghi mã hàng vào `DM1530` (model_select_word).
- App tự chuyển model GLASS tương ứng theo mapping cấu hình.
- App ghi lại mã đang chạy vào `DM1524` (model_current_word) khi load thành công.

## Cấu hình
Trong `config.yaml`:
- `plc.addr.model_select_word`: thanh ghi PLC gửi mã hàng (ví dụ DM1530).
- `plc.addr.model_current_word`: thanh ghi App phản hồi model hiện hành (ví dụ DM1524).
- `plc.model_poll_interval_ms`: chu kỳ poll model code.
- `models.glass_recipes`: danh sách mapping `code -> path (+ threshold)`.

## Quy trình runtime
1. `PlcModelSelectWorker` poll `model_select_word`.
2. Khi giá trị đổi và app không BUSY, UI yêu cầu worker reload model.
3. `InspectionWorker.reload_anomaly_model_with_threshold(...)` reload model an toàn trong worker thread.
4. Nếu load OK:
   - cập nhật model đang dùng,
   - ghi `model_current_word = code`,
   - clear `error` bit.
5. Nếu load fail hoặc code không hợp lệ:
   - giữ model cũ,
   - set `error` bit để PLC xử lý.

## Lưu ý an toàn
- Nếu đang chạy cycle (`busy=1`), app sẽ queue yêu cầu đổi model và áp dụng sau khi cycle kết thúc.
- PLC nên chỉ phát trigger inspection khi đã thấy `DM1524 == DM1530`.


## ACK cho đổi model
- Flow đổi model bằng DM1530/DM1524 **không dùng ACK bit**.
- ACK/DONE hiện chỉ dùng cho chu kỳ inspection (trigger chụp + xử lý).

## Trường hợp PLC reset DM1530
- Quy ước hiện tại: `DM1530 = 0` được hiểu là **không yêu cầu đổi model**.
- App giữ model hiện tại, không set error, và tiếp tục phản hồi `DM1524` = mã model đang chạy.


## App chủ động làm gì khi PLC đổi code?
- Khi thấy DM1530 đổi (khác code hiện tại), app tự chạy luồng đổi model.
- Trong lúc đổi model, app bật `BUSY=1`, tạm bỏ qua trigger inspection mới.
- Khi đổi model xong (thành công/thất bại), app trả `BUSY=0`.
- Thành công: ghi `DM1524` = code mới và clear `ERROR`.
- Thất bại: giữ model cũ và set `ERROR`.

## Chống tín hiệu PLC rung/nhiễu liên tục
- Trigger inspection có thêm `trigger_min_interval_ms` để bỏ các cạnh trigger quá sát nhau.
- Trigger mới sẽ bị bỏ qua nếu app đang đổi model hoặc đã có cycle đang chạy/chờ chạy.
- Poll DM1530 có thêm `model_stable_ms`: chỉ đổi model khi mã ổn định đủ thời gian.
- Chờ ACK clear sau cycle có timeout `ack_clear_ms` để tránh chờ vô hạn nếu ACK bị kẹt ON.


## Triển khai 2 app
- App 1: chạy với `config.yaml` (DM1530 ↔ DM1524).
- App 2: chạy với `config1.yaml` (DM1630 ↔ DM1624).
- PLC chỉ phát trigger inspection khi model_current == model_select tương ứng từng app.

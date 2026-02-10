# PLC Recipe/Model Selection Flow (DM1530 ↔ DM1524)

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

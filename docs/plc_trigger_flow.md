Hệ thống gồm 3 phần chính:
Phát hiện PLC gửi trigger
Chuyển trigger vào luồng xử lý ảnh (inspection cycle)
Chạy vòng kiểm tra và trả kết quả lại PLC
Ngoài ra còn có phần handshake để PLC biết khi nào hệ thống bận, khi nào xong.
1️⃣ PLC GỬI TÍN HIỆU TRIGGER → App phát hiện
🔹 File & hàm:
PlcTriggerWorker.run — app/inspection/workers.py
🔹 Điều gì xảy ra?
Đây là một thread chạy nền, luôn luôn kiểm tra bit trigger của PLC (plc.config.addr.trigger).
Khi thấy trigger = True, nó:
Gửi Qt signal: triggered
Đợi cho bit trigger quay về False (PLC reset) mới tiếp tục vòng lặp.
👉 Bạn có thể hiểu: “PLC ấn nút → thread phát hiện → phát tín hiệu lên UI".
2️⃣ App nhận trigger → đưa vào vòng xử lý ảnh
🔹 File & hàm:
MainWindow._init_workers — nơi kết nối signal
MainWindow._handle_trigger — nơi xử lý trigger
🔹 Cơ chế:
Cả 2 nguồn trigger:
Từ PLC (PlcTriggerWorker.triggered)
Từ nút test trên UI (trigger_manual)
đều dẫn về 1 hàm chung: _handle_trigger()
_handle_trigger() sẽ:
Gửi lệnh chạy InspectionWorker.run_cycle
Nhưng quan trọng: nó gọi bằng
QtCore.QMetaObject.invokeMethod(..., Qt.QueuedConnection)
để đảm bảo toàn bộ việc nặng (AI + camera) chạy trong thread worker, không chạy trong UI thread.
👉 UI chỉ nhận trigger → đẩy task vào queue để worker thực thi.
3️⃣ Worker chạy vòng kiểm tra ảnh (inspection cycle)
🔹 File & hàm:
InspectionWorker.run_cycle — app/inspection/workers.py
🔹 Bên trong run_cycle:
Toàn bộ quá trình được khóa bằng mutex → không bao giờ chạy 2 cycle cùng lúc.
Trình tự:
Bật cờ “busy” lên PLC
→ báo cho PLC biết hệ thống đang xử lý.
Kết nối camera (nếu chưa kết nối)
→ chụp ảnh
→ crop ROI
→ chạy AI (anomaly + YOLO nếu bật)
→ hợp nhất kết quả thành OK/NG.
Ghi kết quả về PLC
Ghi bit OK/NG
Xóa lỗi cũ
Bật cờ done
Gửi signal cycle_completed
(nếu lỗi: bật tất cả NG + bật error + gửi cycle_failed)
finally
Gọi plc.finalize_cycle() để làm bước "handshake cuối"
(chi tiết bước 4 ở dưới)
👉 Đây là trung tâm của hệ thống – toàn bộ inference chạy tại đây.
4️⃣ Handshake PLC: chờ PLC xác nhận đã nhận kết quả
🔹 File & hàm:
PlcController.finalize_cycle — app/inspection/plc_client.py
🔹 Mục đích:
Đảm bảo PLC và App kết thúc cycle đúng chuẩn.
Trình tự:
Chờ PLC bật ack bit trong thời gian cycle_ms.
Khi PLC đã ack:
Clear cờ done
Clear busy
Clear error nếu có.
Chờ PLC tắt ack
→ xác nhận PLC đã sẵn sàng cho chu kỳ tiếp theo.
👉 Bạn có thể hiểu:
App chờ PLC báo “OK tao nhận rồi”.
PLC xong → App dọn các bit về trạng thái ban đầu.
Hai bên sẵn sàng bắt đầu chu kỳ mới.
🏁 KẾT LUẬN (HIỂU NHANH)
Luồng tổng thể:
PLC bật trigger
Thread phát hiện → gửi signal
UI chuyển sang worker
Worker chạy AI + camera → ghi kết quả về PLC
Worker chờ PLC ack để kết thúc sạch sẽ
Tắt busy/done/error → sẵn sàng chu kỳ tiếp theo


# Chu kỳ kiểm tra PLC – App

## 1️⃣ Rung 1 – Gửi Trigger
Khi PLC muốn yêu cầu 1 chu kỳ kiểm tra:
- **|----[ X0 : Nút/Điều kiện Trigger ]---------------------( TRIG )----|
**TRIG** giữ mức **ON** cho đến khi App đọc và xử lý xong.

---

## 2️⃣ Rung 2 – Chờ App nhận Trigger và bật Busy
App khi bắt đầu `run_cycle` sẽ bật `BUSY = 1`.  
PLC chỉ chờ, không tác động.
- **|----[ TRIG ]--------------------------------------------(  )--------|
(App sẽ bật Y0 = BUSY)
---

## 3️⃣ Rung 3 – Chờ Done để bật ACK
App khi hoàn thành xử lý → bật `DONE = 1`.  
PLC sau đó bật `ACK = 1` để báo “đã nhận kết quả”.
- **|----[ DONE ]-------------------------------------------( M100 )----|
|                                           |
Y1 = DONE                               M100 = ACK

---

## 4️⃣ Rung 4 – Reset Trigger sau khi ACK
PLC chỉ tắt `TRIG` khi đã `ACK` để chuẩn bị chu kỳ kế tiếp.
|----[ M100 ]-------------------------------------------[RST TRIG]---|
---

## 5️⃣ Rung 5 – Chờ App reset Busy/Done/Error
App gọi `finalize_cycle()` và reset:
- BUSY = 0
- DONE = 0
- ERROR = 0

Khi App đã reset xong (tức là `Y0 = BUSY OFF` & `Y1 = DONE OFF`),  
PLC sẽ tự reset `ACK`.
|----[ /BUSY ]---[ /DONE ]-----------------------------[RST M100]----|
**/BUSY = BUSY OFF**  
**/DONE = DONE OFF**  
→ Nếu cả hai bit đều OFF, PLC reset ACK → kết thúc cycle.

---

## 6️⃣ Rung 6 – Xử lý kết quả OK/NG
App đặt OK hoặc NG:

**OK result:**
|----[ Y10 ]----------------------------------------------------------|

**NG result:**

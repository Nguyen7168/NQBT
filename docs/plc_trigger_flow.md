# Hệ Thống Inspection -- PLC ↔ App

Hệ thống gồm 3 phần chính:

1.  Phát hiện PLC gửi trigger\
2.  Chuyển trigger vào luồng xử lý ảnh (inspection cycle)\
3.  Chạy vòng kiểm tra và trả kết quả lại PLC

Ngoài ra còn có phần *handshake* để PLC biết khi nào hệ thống bận, khi
nào xong.

------------------------------------------------------------------------

## 1️⃣ PLC gửi tín hiệu Trigger → App phát hiện

**File & hàm:**\
`app/inspection/workers.py` → `PlcTriggerWorker.run`

**Điều gì xảy ra?**

-   Thread chạy nền kiểm tra liên tục bit trigger:
    `plc.config.addr.trigger`
-   Khi trigger = True:
    -   Gửi signal: `triggered`
    -   Chờ PLC reset trigger về False

**Hiểu nhanh:**\
"PLC ấn nút → thread phát hiện → emit signal lên UI".

------------------------------------------------------------------------

## 2️⃣ App nhận trigger → đưa vào vòng xử lý ảnh

**File & hàm:**

-   `MainWindow._init_workers`
-   `MainWindow._handle_trigger`

**Cơ chế:**

Hai nguồn trigger: - Từ PLC: `PlcTriggerWorker.triggered` - Từ nút test
UI: `trigger_manual`

Cả hai đều gọi chung hàm: `_handle_trigger()`

**Bên trong `_handle_trigger()`:**

-   Gọi `InspectionWorker.run_cycle`

-   Nhưng gọi qua:

    ``` python
    QtCore.QMetaObject.invokeMethod(worker, "run_cycle", Qt.QueuedConnection)
    ```

    → đảm bảo AI + camera chạy trong thread worker, không chạy trong UI
    thread.

**Hiểu nhanh:**\
UI chỉ nhận trigger → đẩy task vào queue để worker chạy.

------------------------------------------------------------------------

## 3️⃣ Worker chạy vòng kiểm tra ảnh (Inspection Cycle)

**File & hàm:**\
`app/inspection/workers.py` → `InspectionWorker.run_cycle`

**Bên trong run_cycle:**

Có mutex đảm bảo **chỉ 1 cycle** chạy tại một thời điểm.

**Trình tự:**

1.  Bật cờ `busy` lên PLC\
2.  Kết nối camera\
3.  Chụp ảnh\
4.  Crop ROI\
5.  Chạy AI (anomaly + YOLO nếu bật)\
6.  Tổng hợp OK/NG\
7.  Ghi kết quả về PLC:
    -   bit OK/NG\
    -   clear lỗi cũ\
    -   bật bit `done`\
8.  Gửi signal `cycle_completed`
    -   Nếu lỗi → bật NG, bật error, gửi `cycle_failed`\
9.  Cuối cùng gọi `plc.finalize_cycle()` (handshake)

------------------------------------------------------------------------

## 4️⃣ Handshake PLC: chờ PLC xác nhận đã nhận kết quả

**File & hàm:**\
`app/inspection/plc_client.py` → `PlcController.finalize_cycle`

**Mục đích:**\
Đảm bảo App và PLC kết thúc cycle đúng chuẩn.

**Trình tự:**

1.  Chờ PLC bật ack trong `cycle_ms`\
2.  Khi PLC đã ack:
    -   Clear `done`
    -   Clear `busy`
    -   Clear `error`\
3.  Chờ PLC tắt ack → cycle sẵn sàng chạy tiếp


# Chu kỳ kiểm tra PLC – App

## 1️⃣ Rung 1 – Gửi Trigger
Khi PLC muốn yêu cầu 1 chu kỳ kiểm tra:
- |----[ X0 : Nút/Điều kiện Trigger ]---------------------( TRIG )----|
**TRIG** giữ mức **ON** cho đến khi App đọc và xử lý xong.

---

## 2️⃣ Rung 2 – Chờ App nhận Trigger và bật Busy
App khi bắt đầu `run_cycle` sẽ bật `BUSY = 1`.  
PLC chỉ chờ, không tác động.
- |----[ TRIG ]--------------------------------------------(  )--------|
(App sẽ bật Y0 = BUSY)
---

## 3️⃣ Rung 3 – Chờ Done để bật ACK
App khi hoàn thành xử lý → bật `DONE = 1`.  
PLC sau đó bật `ACK = 1` để báo “đã nhận kết quả”.
- |----[ DONE ]-------------------------------------------( M100 )----|
|                                           |
Y1 = DONE                               M100 = ACK

---

## 4️⃣ Rung 4 – Reset Trigger sau khi ACK
PLC chỉ tắt `TRIG` khi đã `ACK` để chuẩn bị chu kỳ kế tiếp.
- |----[ M100 ]-------------------------------------------[RST TRIG]---|
---

## 5️⃣ Rung 5 – Chờ App reset Busy/Done/Error
App gọi `finalize_cycle()` và reset:
- BUSY = 0
- DONE = 0
- ERROR = 0

Khi App đã reset xong (tức là `Y0 = BUSY OFF` & `Y1 = DONE OFF`),  
PLC sẽ tự reset `ACK`.
- |----[ /BUSY ]---[ /DONE ]-----------------------------[RST M100]----|
**/BUSY = BUSY OFF**  
**/DONE = DONE OFF**  
→ Nếu cả hai bit đều OFF, PLC reset ACK → kết thúc cycle.
## 6️⃣ Rung 6 – Xử lý kết quả OK/NG → Output kết quả OK NG cho PLC

# Tài liệu giải thích chức năng các hàm PLC – Tiếng Việt

> Xem thêm tài liệu vận hành tổng hợp: `docs/plc_operation_guide.md`


Tài liệu này mô tả **mục đích và cách sử dụng (use-case)** của từng hàm liên quan đến PLC trong hệ thống,
bao gồm cả **hàm low-level (giao tiếp PLC)** và **hàm high-level trong PlcController**.

---

# 1. Khái niệm cơ bản

Hệ thống PLC trong dự án được chia thành 2 lớp:

## **1. Lớp Low-level PLC Client (FINS / ASCII / Mock)**
Chức năng:
- Gửi lệnh trực tiếp tới PLC  
- Đọc / ghi từng bit  
- Ghi nhiều bit liên tục theo dạng word  

Các hàm này **không** biết về logic inspection (OK/NG, handshake).  
Chúng chỉ làm nhiệm vụ giao tiếp cấp thấp.

---

## **2. Lớp High-level PlcController**
Chức năng:
- Quản lý handshake BUSY / DONE / ERROR / ACK  
- Ghi kết quả OK/NG từ AI xuống PLC  
- Đọc trigger  
- Tổ chức logic kiểm soát chu kỳ kiểm tra  

Đây là lớp “thông minh”, xây dựng dựa trên client low-level.

---

# 2. Các hàm Low-Level PLC Client

## 2.1 `write_bit(address: str, value: bool)`
**Mục đích:**  
Ghi **1 bit đơn** xuống PLC.

**Dùng trong:**  
- BUSY  
- DONE  
- ERROR  
- ACK  
- Các cờ điều khiển 1-bit khác

**Ví dụ:**  
```python
self.client.write_bit("W150.00", True)
```

---

## 2.2 `read_bit(address: str) -> bool`
**Mục đích:**  
Đọc 1 bit từ PLC.

**Dùng để đọc:**  
- TRIGGER  
- ACK  
- Các tín hiệu trạng thái khác

**Ví dụ:**  
```python
if self.client.read_bit("W100.00"):
    print("PLC trigger bật")
```

---

## 2.3 `write_result_bits(start_word: str, bits: Sequence[bool])`
**Mục đích:**  
Ghi mảng kết quả OK/NG theo quy tắc **pack bit vào word 16-bit**, bắt đầu từ `start_word`.

**Quy tắc mapping (bit order):**
- Cờ thứ `i` (đánh số từ 1) → nằm ở:
  - word offset: `(i - 1) // 16`
  - bit offset: `(i - 1) % 16`
- Cờ đầu tiên map vào bit thấp `.00`.

Ví dụ: Start word `EM7901`
- Cờ 1..16  → `EM7901.00 .. EM7901.15`
- Cờ 17..32 → `EM7902.00 .. EM7902.15`

Khi số cờ không chia hết cho 16, các bit dư ở word cuối sẽ được ghi `0` để tránh giữ rác từ chu kỳ trước.

---

# 3. Các hàm trong lớp High-Level `PlcController`

## 3.1 `set_busy(value: bool)`
**Mục đích:**  
Thông báo **bắt đầu chu kỳ kiểm tra**.

- `BUSY = 1` khi bắt đầu  
- `BUSY = 0` khi chu kỳ kết thúc (sau khi PLC ACK)

---

## 3.2 `set_done(value: bool)`
**Mục đích:**  
Thông báo **đã hoàn thành chu kỳ**.

- `DONE = 1` sau khi ghi kết quả  
- `DONE = 0` reset trong finalize_cycle  

---

## 3.3 `set_error(value: bool)`
**Mục đích:**  
Gửi báo lỗi kiểm tra.

- `ERROR = 1` nếu cycle_failed  
- `ERROR = 0` reset cuối chu kỳ  

---

## 3.4 `write_results(results: Sequence[bool])`
**Mục đích:**  
Ghi toàn bộ kết quả OK/NG xuống PLC.

Hàm này là wrapper cho:

```python
self.client.write_result_bits(...)
```

**Ví dụ input:**

```python
[True, False, True, True]
```

→ PLC ghi OK/NG tương ứng từng vị trí ROI.

---

## 3.5 `wait_for_trigger(...)`
**Mục đích:**  
Dùng khi muốn chờ trigger theo dạng blocking-loop.

- Kiểm tra trigger bit nhiều lần  
- Trả về `True` khi trigger bật  
- Trả về `False` nếu timeout  

(Trong hệ thống hiện tại, QThread được dùng thay thế.)

---

## 3.6 `wait_for_ack_clear(...)`
**Mục đích:**  
Chờ PLC tắt ACK, đảm bảo PLC đã sẵn sàng cho chu kỳ tiếp theo.

---

## 3.7 `finalize_cycle()`
**Mục đích:**  
Thực hiện handshake cuối cùng với PLC.

Quy trình:

1. Chờ PLC bật `ACK = 1`  
2. Reset các bit:  
   - DONE = 0  
   - BUSY = 0  
   - ERROR = 0 (nếu có)  
3. Chờ PLC tắt ACK  

→ Đảm bảo hệ thống trở về trạng thái ban đầu, sẵn sàng nhận trigger mới.

---

# 4. Tóm tắt chu kỳ kiểm tra

### Khi bắt đầu:
- `set_busy(True)`

### Khi AI xử lý xong:
- `write_results([...])`
- `set_error(False)`
- `set_done(True)`

### Khi lỗi:
- Ghi tất cả NG  
- `set_error(True)`  
- `set_done(True)`

### Khi kết thúc chu kỳ:
- `finalize_cycle()` thực hiện reset handshake

---

# 5. Sơ đồ luồng tổng quát

```
TRIGGER → BUSY → XỬ LÝ AI → GHI KẾT QUẢ → DONE → PLC ACK → RESET
```

---

# 6. Khi nào dùng hàm nào?

| Hàm | Mục đích |
|-----|----------|
| `write_bit` | Ghi từng bit BUSY / DONE / ERROR |
| `read_bit` | Đọc trigger hoặc ACK |
| `write_result_bits` | Ghi nhiều bit OK/NG |
| `write_results` | Hàm high-level ghi ALL kết quả học máy |
| `set_busy` | Bắt đầu chu kỳ |
| `set_done` | Kết thúc chu kỳ |
| `set_error` | Thông báo lỗi |
| `finalize_cycle` | Hoàn tất handshake và reset trạng thái |

---

# 7. Giải thích tham số cấu hình PLC trong `config.yaml`

Mục này dùng để đội lập trình PLC và đội app thống nhất ý nghĩa các tham số timing/debounce.

```yaml
trigger_poll_interval_ms: 50
trigger_min_interval_ms: 100
trigger_high_stable_ms: 80
trigger_low_stable_ms: 80
trigger_cooldown_ms: 300
enable_plc_trigger: true
model_poll_interval_ms: 200
model_stable_ms: 200
sample_trigger_poll_interval_ms: 50
poll_error_backoff_ms: 1000
timeouts:
  connect_ms: 3000
  response_ms: 1000
  cycle_ms: 5000
  ack_clear_ms: 2000
```

## 7.1 Nhóm trigger inspection

### `trigger_poll_interval_ms`
- Chu kỳ app đọc bit Trigger từ PLC (đơn vị ms).
- Ví dụ `50` nghĩa là đọc khoảng **20 lần/giây**.
- Giảm giá trị này sẽ bắt trigger nhanh hơn nhưng tăng tải giao tiếp PLC.

### `trigger_min_interval_ms`
- Khoảng cách tối thiểu giữa hai trigger hợp lệ liên tiếp.
- Dùng để chặn trigger lặp quá sát nhau.
- Trong app, tham số này kết hợp với cooldown; ngưỡng chặn thực tế lấy giá trị lớn hơn.

### `trigger_high_stable_ms`
- Bit Trigger phải giữ mức ON liên tục ít nhất thời gian này thì mới được coi là trigger hợp lệ.
- Dùng để lọc nhiễu/xung ON quá ngắn (debounce mức cao).

### `trigger_low_stable_ms`
- Sau khi đã nhận 1 trigger, bit Trigger phải giữ OFF ổn định ít nhất thời gian này thì app mới “re-arm” để nhận trigger mới.
- Dùng để tránh rung ON/OFF gây bắn nhiều chu kỳ.

### `trigger_cooldown_ms`
- Thời gian nghỉ tối thiểu sau 1 trigger hợp lệ trước khi cho phép trigger kế tiếp.
- Đây là lớp bảo vệ mạnh chống auto-trigger liên tục khi tín hiệu không sạch.

### `enable_plc_trigger`
- Bật/tắt hoàn toàn luồng nhận trigger từ PLC.
- `true`: app nhận trigger tự động từ PLC, nhưng chỉ poll `trigger` chính ở mode `RUN`/`MIRROR`; khi vào `SAMPLE` thì main trigger worker tự dừng.
- `false`: app không nhận trigger PLC (hữu ích khi debug/manual test).

## 7.2 Nhóm chọn model (recipe)

### `model_poll_interval_ms`
- Chu kỳ đọc thanh ghi model code từ PLC (word chọn recipe).
- Ví dụ `200` nghĩa là đọc mỗi 200 ms.

### `model_stable_ms`
- Giá trị model code phải ổn định liên tục trong thời gian này trước khi app xác nhận đổi model.
- Mục tiêu là tránh đổi model do nhiễu đọc nhất thời.

## 7.3 Trigger SAMPLE riêng

### `sample_trigger_poll_interval_ms`
- Chu kỳ app đọc bit `sample_trigger` từ PLC (đơn vị ms).
- Worker `sample_trigger` chỉ chạy khi app đang ở mode `SAMPLE`; khi mode khác, worker sẽ dừng để giảm tải poll PLC.
- Đồng thời ở mode `SAMPLE`, worker `trigger` chính cũng dừng poll nên app chỉ nhận trigger từ `sample_trigger`.

### `poll_error_backoff_ms`
- Thời gian sleep sau mỗi lần poll lỗi (áp dụng chung cho trigger/model/mode/sample-trigger worker).
- Trước đây hard-code 1000 ms; hiện đã config được để giảm thời gian “đứng chờ” sau lỗi.
- Chu kỳ lặp log lỗi gần đúng: `response_ms + poll_error_backoff_ms` (cộng thêm overhead nhỏ).

## 7.4 Nhóm timeout handshake

### `timeouts.connect_ms`
- Timeout kết nối TCP tới PLC khi khởi động.

### `timeouts.response_ms`
- Timeout chờ phản hồi cho mỗi lệnh đọc/ghi PLC qua socket (ví dụ `RD MR62500`).
- Mặc định `1000` ms.
- Nếu PLC không trả lời, thời gian chờ này cộng với `poll_error_backoff_ms` quyết định tốc độ retry/log lỗi.
- Có thể chỉnh trong `config.yaml`/`config1.yaml` theo chất lượng mạng/PLC thực tế.

### `timeouts.cycle_ms`
- Timeout tối đa cho pha chờ ACK/cycle handshake.
- Quá thời gian này app sẽ ghi warning/timeout để tránh treo chu kỳ.

### `timeouts.ack_clear_ms`
- Timeout chờ PLC hạ ACK về OFF sau khi app finalize cycle.
- Dùng để đảm bảo hệ thống quay về trạng thái sẵn sàng cho chu kỳ tiếp theo.

## 7.5 Khuyến nghị nhanh cho đội PLC

- Trigger nên là bit sạch, có xung ON đủ dài hơn `trigger_high_stable_ms`.
- Sau khi app nhận trigger, PLC nên hạ Trigger rõ ràng và giữ OFF đủ lâu (`trigger_low_stable_ms`).
- Nếu gặp auto-trigger ngoài ý muốn, tăng `trigger_cooldown_ms` trước rồi mới tinh chỉnh stable time.
- Khi test không muốn chạy tự động theo PLC, đặt `enable_plc_trigger: false`.

---

# ✔ Kết thúc tài liệu

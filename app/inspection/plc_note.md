# Tài liệu giải thích chức năng các hàm PLC – Tiếng Việt

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
Ghi **nhiều bit liên tục** (thường là OK/NG cho từng ROI) bắt đầu từ 1 word.

Ví dụ:  
Start word: `W160`  
Input bits: `[True, False, True]`

PLC sẽ ghi:

```
W160.00 = 1  
W160.01 = 0  
W160.02 = 1
```

**Đây là cách hiệu quả nhất để ghi nhiều kết quả OK/NG.**

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

# ✔ Kết thúc tài liệu

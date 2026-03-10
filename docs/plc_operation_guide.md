# PLC Operation Guide (RUN / SAMPLE / MIRROR)

Tài liệu này là bản vận hành chuẩn giữa **PLC ↔ App** để tránh lỗi handshake, sai mode, hoặc mất trigger.

---

## 1) I/O Mapping Master Table

> Địa chỉ thực tế đọc từ `config.yaml`.

| Nhóm | Key config | Ví dụ | Hướng | Ý nghĩa |
|---|---|---|---|---|
| Trigger chính | `plc.addr.trigger` | `R000` | PLC → App | Yêu cầu chạy cycle theo mode hiện tại |
| Trigger SAMPLE riêng | `plc.addr.sample_trigger` | `MR87900` | PLC → App | Yêu cầu chạy thử ảnh mẫu (chỉ hiệu lực khi mode=SAMPLE) |
| ACK | `plc.addr.ack` | `EM7002` | PLC → App | PLC xác nhận đã nhận kết quả cycle |
| BUSY | `plc.addr.busy` | `DM1522` | App → PLC | App đang xử lý cycle |
| DONE | `plc.addr.done` | `DM1523` | App → PLC | Cycle đã xong, kết quả đã ghi |
| ERROR | `plc.addr.error` | `R1002` | App → PLC | Lỗi cycle/model |
| RUN cờ app | `plc.addr.run` | `R1004` | App → PLC | App running state |
| Kết quả OK/NG mảng | `plc.addr.result_bits_start_word` | `EM7901` | App → PLC | Mảng kết quả ROI (RUN/SAMPLE) |
| Model request | `plc.addr.model_select_word` | `DM1530` | PLC → App | Mã model PLC yêu cầu |
| Model current | `plc.addr.model_current_word` | `DM1524` | App → PLC | Mã model app đang dùng |
| Mode request | `plc.addr.mode_request_word` | `DM1533` | PLC → App | Mode PLC yêu cầu: RUN=3, SAMPLE=2, MIRROR=1 |
| Mode current | `plc.addr.mode_current_word` | `DM1525` | App → PLC | Mode app đang chạy: RUN=3, SAMPLE=2, MIRROR=1 |
| Mirror result | `plc.addr.mirror_result_word` | `EM1332` | App → PLC | Kết quả mirror: OK=1, NG=0 |

### Mapping mảng kết quả `result_bits_start_word`
- App ghi kết quả theo dạng **pack 16 bit/word**.
- Quy tắc: cờ thứ `i` (1-based) map vào bit `((i-1) % 16)` của word `start + ((i-1) // 16)`.
- Bit order: cờ đầu tiên nằm ở bit thấp `.00`.
- Ví dụ `start=EM7901`:
  - Cờ 1..16  → `EM7901.00 .. EM7901.15`
  - Cờ 17..32 → `EM7902.00 .. EM7902.15`
- Bit dư ở word cuối (nếu tổng cờ không chia hết 16) luôn được ghi `0`.

---

## 2) Mode State Machine

## 2.1 Mode code
- `3 = RUN`
- `2 = SAMPLE`
- `1 = MIRROR`

## 2.2 Nguồn đổi mode
- **Manual trên app**: người vận hành chọn từ menu/toolbar `Mode`.
- **PLC mode request**: ghi vào `mode_request_word`.

## 2.3 Rule ưu tiên
- **PLC luôn ưu tiên cao hơn manual**.
- Nếu PLC đổi mode khi app đang idle, app đổi ngay.
- Nếu app đang xử lý cycle (`busy=1`), mode mới được queue và áp sau khi cycle kết thúc.

## 2.4 Ảnh hưởng vận hành
- Không cắt ngang cycle đang chạy → an toàn dữ liệu kết quả.
- PLC cần đọc `mode_current_word` để biết mode đã áp thực tế.

---

## 3) Trigger Matrix theo mode

| Mode hiện tại | Trigger chính (`trigger`) | Trigger SAMPLE (`sample_trigger`) | Hành vi |
|---|---:|---:|---|
| RUN | Có hiệu lực | Bỏ qua | Chụp camera + infer model + ghi mảng kết quả |
| SAMPLE | Không poll (main trigger worker dừng) | Có hiệu lực | Load ảnh mẫu `sample_image_root/<ten_recipe>/*.png` + infer + ghi mảng kết quả |
| MIRROR | Có hiệu lực | Bỏ qua | Chụp camera + đo đường kính + ghi `mirror_result_word` |

### Lưu ý quan trọng cho SAMPLE
- `sample_trigger` chỉ được poll khi app đang ở mode `SAMPLE`; mode `RUN/MIRROR` sẽ dừng worker sample trigger để giảm tải PLC.
- Ở mode `SAMPLE`, main trigger worker cũng dừng nên trigger chính không được xử lý trong mode này.
- Nếu không tìm thấy ảnh mẫu theo mã hàng: app cảnh báo, không crash.
- PLC nên giám sát timeout tại tầng ladder để tránh chờ vô hạn.
- App hiện tại tìm ảnh theo thứ tự:
  1. `sample_image_root/<recipe.name>/*.png`
  2. nếu không có `recipe.name` thì fallback `sample_image_root/<active_recipe_code>/*.png`
- Chỉ nhận file đuôi `*.png` (không tự lấy jpg/jpeg).
- Tên thư mục phải trùng **chính xác** với `models.glass_recipes[].name` (phân biệt hoa/thường trên Linux).

---

## 3.1) Chuẩn đặt thư mục ảnh mẫu

Ví dụ:

```text
samples/
  23-233HA-E/
    sample_001.png
    sample_002.png
  ITEM_2/
    sample_001.png
```

Trong đó:
- `samples` là giá trị `sample_image_root` trong file config.
- `23-233HA-E`, `ITEM_2` phải đúng với tên recipe trong `models.glass_recipes`.

---

## 4) ACK Handshake chuẩn (inspection cycle)

1. PLC bật trigger.
2. App nhận trigger → `BUSY=1`.
3. App xử lý xong → ghi kết quả + `DONE=1`.
4. PLC đọc kết quả, sau đó bật `ACK=1`.
5. App `finalize_cycle()`:
   - clear `DONE`
   - clear `BUSY`
   - clear `ERROR` (nếu có)
6. PLC hạ `ACK=0` sau khi thấy `BUSY=0` và `DONE=0`.

### Khuyến nghị ladder
- Không giữ ACK ON liên tục nhiều chu kỳ.
- Nếu ACK bị kẹt ON, app có timeout `ack_clear_ms` và log cảnh báo.

---

## 5) Error Handling Matrix

| Tình huống | BUSY | DONE | ERROR | Mirror word |
|---|---:|---:|---:|---:|
| RUN/SAMPLE thành công | 1→0 | xung 1 rồi 0 | 0 | giữ nguyên |
| RUN/SAMPLE lỗi cycle | 1→0 | xung 1 rồi 0 | 1 (sau đó clear khi finalize) | giữ nguyên |
| MIRROR thành công | 1→0 | xung 1 rồi 0 | 0 | OK=1 / NG=0 theo ngưỡng |
| MIRROR lỗi (không thấy vòng tròn, lỗi module...) | 1→0 | xung 1 rồi 0 | 0 hoặc theo logic app | ghi NG=0 |

---

## 6) Commissioning Checklist (trước chạy thật)

## 6.1 Kiểm tra cấu hình
- [ ] IP/port PLC đúng.
- [ ] Tất cả địa chỉ bit/word map đúng như bảng I/O.
- [ ] `mode_request_word` / `mode_current_word` map đúng RUN=3, SAMPLE=2, MIRROR=1.
- [ ] `mirror_result_word` map đúng vùng PLC đọc.
- [ ] `sample_image_root` tồn tại và có thư mục theo mã hàng.

## 6.2 Kiểm tra timing/debounce
- [ ] Trigger ON đủ dài hơn `trigger_high_stable_ms`.
- [ ] Trigger OFF đủ dài hơn `trigger_low_stable_ms`.
- [ ] `trigger_cooldown_ms` đủ để chặn rung xung.
- [ ] `model_stable_ms` phù hợp chống nhiễu thanh ghi mode/model.
- [ ] `timeouts.response_ms` phù hợp với độ trễ mạng/PLC (mặc định 1000 ms).
- [ ] `poll_error_backoff_ms` phù hợp để retry nhanh/chậm sau lỗi poll (mặc định 1000 ms).

## 6.3 Test tuần tự bắt buộc
1. Test RUN: 3 cycle liên tiếp, xác nhận ACK clear ổn định.
2. Test SAMPLE: gửi `sample_trigger`, xác nhận output map giống RUN.
3. Test MIRROR: test pass/fail ngưỡng, xác nhận `mirror_result_word` 1/0.
4. Test PLC override manual: chọn mode manual rồi đổi bằng PLC, xác nhận app theo PLC.

## 6.4 Khi có lỗi treo chu kỳ
- [ ] Kiểm tra ACK có kẹt ON không.
- [ ] Kiểm tra trigger có rung liên tục không.
- [ ] Kiểm tra `BUSY`/`DONE` có về 0 sau mỗi cycle không.
- [ ] Kiểm tra log timeout `cycle_ms` / `ack_clear_ms`.
- [ ] Nếu gặp `PLC trigger polling failed: Timeout waiting for PLC response...`, tinh chỉnh `timeouts.response_ms` và/hoặc `poll_error_backoff_ms` để cân bằng tốc độ retry và tải log.

## 6.5 Khi app không chuyển mode theo yêu cầu PLC (dù monitor thấy PLC ghi đúng)
- [ ] Xác nhận PLC ghi đúng mã mode mới: `RUN=3`, `SAMPLE=2`, `MIRROR=1`. Giá trị khác sẽ bị app bỏ qua.
- [ ] Kiểm tra app có đang bận chu kỳ không (`BUSY=1` hoặc đang inflight). Khi bận, mode sẽ được queue và chỉ áp sau khi cycle kết thúc.
- [ ] Kiểm tra timeout/handshake ACK: nếu ACK không lên/xuống đúng nhịp, app có thể chờ đến timeout rồi mới clear `BUSY` và áp mode mới.
- [ ] Kiểm tra polling mode có lỗi truyền thông không (log `PLC mode polling failed`). Khi lỗi, worker sẽ backoff theo `poll_error_backoff_ms` nên nhìn như phản ứng chậm.
- [ ] Kiểm tra nhiễu mode request: mode mới phải giữ ổn định đủ `model_stable_ms` để worker phát hiện thay đổi.

## 6.6 Khuyến nghị tinh chỉnh tránh timeout (ưu tiên cho Keyence KV-8000)

> Mục tiêu: giảm timeout giả trước, sau đó tối ưu tốc độ poll.

### Bước 1 — Ổn định truyền thông
- Tăng `timeouts.response_ms` lên **1500–2000 ms** (khởi điểm đề xuất: `1500`).
- Đặt `poll_error_backoff_ms` khoảng **300–500 ms** (khởi điểm: `500`).

### Bước 2 — Giảm tải polling
- Tăng `trigger_poll_interval_ms` từ 10 ms lên **20–30 ms**.
- Giữ `sample_trigger_poll_interval_ms` ở **50 ms** (hoặc 80 ms nếu vẫn quá tải).

### Bước 3 — Nới timeout handshake nếu ladder nhiều bước
- `timeouts.cycle_ms`: **7000–10000 ms**.
- `timeouts.ack_clear_ms`: **3000–5000 ms**.

### Bộ giá trị khởi điểm tham khảo
```yaml
plc:
  trigger_poll_interval_ms: 20
  sample_trigger_poll_interval_ms: 50
  poll_error_backoff_ms: 500
  timeouts:
    response_ms: 1500
    cycle_ms: 7000
    ack_clear_ms: 3000
```

### Trình tự tuning khuyến nghị
1. Bật `log_raw_response: true` để kiểm tra phản hồi thô của PLC.
2. Chỉnh `response_ms` + `poll_error_backoff_ms` trước.
3. Nếu vẫn timeout, mới tăng `trigger_poll_interval_ms`.
4. Cuối cùng mới nới `cycle_ms` / `ack_clear_ms` theo thực tế ladder.

---

## 7) Triển khai 2 app (config.yaml + config1.yaml)

- App #1 dùng `config.yaml`.
- App #2 dùng `config1.yaml`.
- Khuyến nghị chạy bằng tham số:

```bash
python run.py --config config.yaml
python run.py --config config1.yaml
```

Lưu ý bắt buộc khi chạy song song:
- Không để trùng các word/bit output điều khiển giữa 2 app (`busy/done/mode_current/result_bits_start_word/mirror_result_word/...`).
- Nếu 2 app cùng PLC, nên xác nhận lại các bit ACK/ERROR có tách riêng hay không trước khi chạy production.

---

## 8) Liên kết tài liệu liên quan
- Trigger/handshake chi tiết: `docs/plc_trigger_flow.md`
- Recipe/model flow: `docs/plc_model_recipe_flow.md`
- API PLC + timing notes: `app/inspection/plc_note.md`

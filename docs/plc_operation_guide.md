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
| Mode request | `plc.addr.mode_request_word` | `DM1533` | PLC → App | Mode PLC yêu cầu: 1/2/3 |
| Mode current | `plc.addr.mode_current_word` | `DM1525` | App → PLC | Mode app đang chạy: 1/2/3 |
| Mirror result | `plc.addr.mirror_result_word` | `EM1332` | App → PLC | Kết quả mirror: OK=1, NG=0 |

---

## 2) Mode State Machine

## 2.1 Mode code
- `1 = RUN`
- `2 = SAMPLE`
- `3 = MIRROR`

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
<<<<<<< codex/add-image-sampling-functionality-with-plc-signal-43h886
| SAMPLE | Có hiệu lực | Có hiệu lực | Load ảnh mẫu `sample_image_root/<ten_recipe>/*.png` + infer + ghi mảng kết quả |
=======
| SAMPLE | Có hiệu lực | Có hiệu lực | Load ảnh mẫu `sample_root/<ma_hang>/*.png` + infer + ghi mảng kết quả |
>>>>>>> main
| MIRROR | Có hiệu lực | Bỏ qua | Chụp camera + đo đường kính + ghi `mirror_result_word` |

### Lưu ý quan trọng cho SAMPLE
- Nếu không tìm thấy ảnh mẫu theo mã hàng: app cảnh báo, không crash.
- PLC nên giám sát timeout tại tầng ladder để tránh chờ vô hạn.
<<<<<<< codex/add-image-sampling-functionality-with-plc-signal-43h886
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
=======
>>>>>>> main

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
- [ ] `mode_request_word` / `mode_current_word` map đúng 1/2/3.
- [ ] `mirror_result_word` map đúng vùng PLC đọc.
- [ ] `sample_image_root` tồn tại và có thư mục theo mã hàng.

## 6.2 Kiểm tra timing/debounce
- [ ] Trigger ON đủ dài hơn `trigger_high_stable_ms`.
- [ ] Trigger OFF đủ dài hơn `trigger_low_stable_ms`.
- [ ] `trigger_cooldown_ms` đủ để chặn rung xung.
- [ ] `model_stable_ms` phù hợp chống nhiễu thanh ghi mode/model.

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

---

<<<<<<< codex/add-image-sampling-functionality-with-plc-signal-43h886
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
=======
## 7) Liên kết tài liệu liên quan
>>>>>>> main
- Trigger/handshake chi tiết: `docs/plc_trigger_flow.md`
- Recipe/model flow: `docs/plc_model_recipe_flow.md`
- API PLC + timing notes: `app/inspection/plc_note.md`

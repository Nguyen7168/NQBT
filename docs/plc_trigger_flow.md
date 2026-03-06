# PLC Trigger Flow (Theo code hiện tại)

> Tài liệu ngắn gọn cho PLC team: tập trung vào trigger, ACK handshake, timeout và checklist triển khai nhanh.

## 1) Trạng thái mode và nguồn trigger

| Mode | Main trigger (`plc.addr.trigger`) | Sample trigger (`plc.addr.sample_trigger`) | Hành vi cycle |
|---|---|---|---|
| RUN (1) | ✅ Có hiệu lực | ❌ Không chạy worker sample | Chụp camera + infer + ghi mảng OK/NG |
| SAMPLE (2) | ❌ Worker main trigger tạm dừng | ✅ Có hiệu lực | Dùng ảnh mẫu + infer + ghi mảng OK/NG |
| MIRROR (3) | ✅ Có hiệu lực | ❌ Không chạy worker sample | Chụp camera + đo mirror + ghi `mirror_result_word` |

**Điểm quan trọng:**
- Ở mode `SAMPLE`, app chỉ nhận trigger từ `sample_trigger`.
- Main trigger worker được start/stop theo mode để tránh poll thừa khi chạy SAMPLE.

---

## 2) Sơ đồ trạng thái inspection cycle (PLC ↔ App)

```text
[IDLE / READY=1]
      |
      | Trigger hợp lệ theo mode
      v
[START CYCLE]
  App: BUSY=1, READY=0
      |
      v
[PROCESSING]
  - RUN/SAMPLE: capture(or sample image) + inference + write results
  - MIRROR: mirror measurement + write mirror_result
      |
      +--> Success: DONE=1, ERROR=0
      |
      +--> Fail:    DONE=1, ERROR=1, results=NG all (RUN/SAMPLE)
      v
[WAIT ACK ON]
  App chờ PLC bật ACK=1 (timeout: cycle_ms)
      |
      v
[FINALIZE FLAGS]
  App clear DONE=0, BUSY=0, ERROR=0 (nếu đang ON)
      |
      v
[WAIT ACK OFF]
  App chờ PLC hạ ACK=0 (timeout: ack_clear_ms)
      |
      v
[READY AGAIN]
  App set READY=1 -> chu kỳ mới
```

---

## 3) Trình tự handshake chuẩn (đề xuất ladder)

1. PLC bật trigger theo mode đang chạy.
2. App nhận trigger và bật `BUSY=1`.
3. App xử lý xong, ghi output, bật `DONE=1` (và `ERROR=1` nếu cycle fail).
4. PLC đọc output xong thì bật `ACK=1`.
5. App vào `finalize_cycle()`:
   - clear `DONE`
   - clear `BUSY`
   - clear `ERROR` (nếu có)
6. PLC thấy `BUSY=0` và `DONE=0` thì hạ `ACK=0`.
7. App chờ ACK OFF xong sẽ set `READY=1`.

---

## 4) Timeout behavior (quan trọng khi debug)

- **Chờ ACK ON timeout (`cycle_ms`)**:
  - App ghi warning log, không treo vô hạn.
  - Vẫn tiếp tục clear cờ và đi tiếp finalize.

- **Chờ ACK OFF timeout (`ack_clear_ms`)**:
  - App ghi warning log.
  - Vẫn set `READY=1` để hệ thống quay lại trạng thái sẵn sàng.

- **Polling trigger/model/mode lỗi truyền thông**:
  - Worker sleep `poll_error_backoff_ms` rồi retry.

### 4.1 Gợi ý tuning nhanh để giảm timeout
- Nếu thường xuyên thấy `Timeout waiting for PLC response...`:
  1. Tăng `timeouts.response_ms` lên 1500 trước.
  2. Đặt `poll_error_backoff_ms` 500 để retry ổn định hơn.
  3. Tăng `trigger_poll_interval_ms` lên 20 ms nếu PLC còn quá tải.
- Nếu timeout ở pha ACK:
  - tăng `timeouts.cycle_ms` (ví dụ 7000),
  - tăng `timeouts.ack_clear_ms` (ví dụ 3000).
- Chỉ tối ưu lại xuống thấp hơn sau khi đã chạy ổn định nhiều chu kỳ liên tiếp.

---

## 5) Checklist nhanh cho PLC team

### A. Mapping & mode
- [ ] `trigger`, `sample_trigger`, `ack`, `busy`, `done`, `error`, `ready` map đúng địa chỉ.
- [ ] Mode code thống nhất: `1=RUN`, `2=SAMPLE`, `3=MIRROR`.
- [ ] PLC đọc `mode_current_word` để biết mode app đang áp thực tế.

### B. Trigger theo mode
- [ ] RUN/MIRROR: dùng main trigger.
- [ ] SAMPLE: dùng sample trigger; không kỳ vọng main trigger chạy.
- [ ] Trigger ON đủ lâu hơn debounce (`trigger_high_stable_ms`).

### C. ACK handshake
- [ ] Sau khi thấy `DONE=1`, PLC bật `ACK=1`.
- [ ] Khi thấy `BUSY=0` và `DONE=0`, PLC hạ `ACK=0`.
- [ ] Không giữ ACK ON qua nhiều chu kỳ liên tiếp.

### D. Khi có timeout
- [ ] Kiểm tra PLC có thực sự bật ACK sau DONE chưa.
- [ ] Kiểm tra ACK có bị kẹt ON không hạ về OFF.
- [ ] Kiểm tra `timeouts.response_ms`, `cycle_ms`, `ack_clear_ms` phù hợp thực tế mạng.
- [ ] Kiểm tra log `Timeout waiting for PLC response...` để khoanh vùng lệnh bị treo.

---


## 6) Quy tắc ACK khuyến nghị khi lập trình PLC

- ✅ **Khuyến nghị chuẩn:** bật `ACK=1` khi thấy `DONE=1` (sau khi PLC đã đọc xong output).
- ⚠️ **Không nên** dùng điều kiện `DONE=1 OR ERROR=1` làm trigger ACK chính.

### Vì sao nên bám theo DONE?

1. Ở RUN/SAMPLE, cả thành công và lỗi cycle đều đi qua `DONE=1`:
   - Thành công: `DONE=True`, `ERROR=False`.
   - Lỗi: `ERROR=True` rồi cũng `DONE=True`.
2. Ở MIRROR lỗi, code hiện tại vẫn có thể `DONE=True` nhưng `ERROR=False`.
   - Nếu PLC ACK dựa vào `ERROR`, có thể bỏ lỡ ACK dù cycle đã hoàn tất.
3. `finalize_cycle()` của app được thiết kế theo state machine: chờ ACK ON → clear cờ → chờ ACK OFF → set READY lại.
   - Vì vậy ACK nên bám tín hiệu “chu kỳ hoàn tất” (`DONE`) để đồng bộ chắc chắn.

### Mẫu logic ladder đề xuất

1. Chờ `DONE=1`.
2. Đọc output cần thiết (result bits / mirror result / error bit để phân loại).
3. Bật `ACK=1`.
4. Khi thấy `BUSY=0` và `DONE=0`, hạ `ACK=0`.

---

## 7) Tham chiếu
- Luồng handshake thực thi: `app/inspection/plc_client.py` (`finalize_cycle`, `wait_for_ack_clear`).
- Luồng cycle & xử lý lỗi: `app/inspection/workers.py` (`run_cycle`, `run_sample_cycle`).
- Tài liệu vận hành tổng hợp: `docs/plc_operation_guide.md`.

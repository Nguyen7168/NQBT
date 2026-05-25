# AnomalyNCD Integration Research Plan

## Phạm vi
- Không sửa production workflow OK/NG.
- Không đụng PLC output.
- Chỉ nghiên cứu tầng 2 phân loại 7 lớp NG.

## Khảo sát app hiện tại
- Tầng 1 đã có `InferencePipeline` trả `statuses` theo patch (`OK/NG`) + score/map.
- `InspectionResult` + `to_json()` hiện chưa có field defect class.
- UI table hiện có 4 cột: Index/Score/D(mm)/Status.

## Thiết kế tầng 2 đề xuất (chưa implement production)
Chỉ với patch `status == "NG"`, gọi module phân loại và trả:
- `defect_class_id`
- `defect_class_name`
- `defect_confidence`
- `defect_source` (anomalyncd|supervised|rule)

## Điểm cần chỉnh sau này (đề xuất)
- `InferenceResultPayload`: thêm list kết quả class theo patch.
- `InspectionResult`: mirror fields mới.
- `to_json()`: thêm payload defect cho từng part.
- UI: thêm cột Defect Class + Confidence, overlay text class ngắn.
- PLC: giữ nguyên bit OK/NG giai đoạn đầu.

## So sánh A/B/C
### A) Dùng AnomalyNCD trực tiếp
- Dễ tích hợp: thấp-trung bình.
- Ổn định production: trung bình-thấp (phụ thuộc env riêng, GPU).
- Dataset: cần đúng format và anomaly map pipeline của repo.
- Inference: có thể nặng hơn classifier nhỏ.
- Rủi ro dependency: cao.
- Rủi ro PLC cycle time: trung bình-cao nếu đồng bộ inline.

### B) Dùng AnomalyNCD để tạo feature/cluster rồi map 7 class
- Dễ tích hợp: trung bình.
- Ổn định production: trung bình (offline mining rồi deploy model nhỏ).
- Dataset: vẫn cần dữ liệu NG có chất lượng.
- Inference: nhanh nếu chỉ deploy head/classifier.
- Rủi ro dependency: trung bình (research env tách biệt).
- Rủi ro PLC cycle time: thấp-trung bình.

### C) Dùng supervised classifier 7 lớp sau INP
- Dễ tích hợp: cao nhất.
- Ổn định production: cao nhất (mô hình gọn, dễ kiểm soát).
- Dataset: cần nhãn chuẩn 7 lớp đủ lớn.
- Inference: nhanh nhất nếu ONNX/TensorRT head nhỏ.
- Rủi ro dependency: thấp-trung bình.
- Rủi ro PLC cycle time: thấp nhất.

## Kết luận đề xuất
- **Ưu tiên hướng C** cho production roadmap.
- Song song: dùng **B** ở pha R&D để khám phá feature/cluster hỗ trợ làm sạch nhãn/khám phá subclass.
- Hướng A chỉ nên làm benchmark đối chứng, không nên đưa thẳng vào app hiện tại.

## Trạng thái khảo sát AnomalyNCD trong cloud
- Đã xác nhận metadata README/public index và flow mức cao.
- Chưa đọc source chi tiết trong workspace do clone GitHub bị chặn 403.
- Cần bước local để chốt format/config cuối cùng.

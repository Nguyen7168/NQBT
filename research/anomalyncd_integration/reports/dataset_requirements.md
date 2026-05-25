# Dataset Requirements - Bạc trục

- Cấu trúc chuẩn:
  - `raw_full_images/{OK,01..07_*}`
  - `crops_labeled/{OK,01..07_*}`
- Mỗi class tối thiểu 20 ảnh đọc được (mức cảnh báo), khuyến nghị >=200/class cho classifier supervised ổn định.
- Với tầng 2 chỉ xử lý NG: ưu tiên tăng dữ liệu 7 class NG; class OK vẫn cần để kiểm soát false-positive khi thử mô hình phụ.
- Ảnh crop cần nhất quán kích thước hoặc policy resize cố định.
- Nên lưu metadata: recipe code, camera id, timestamp, operator, lighting version.

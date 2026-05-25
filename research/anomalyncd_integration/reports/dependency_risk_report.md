# Dependency Risk Report

## App hiện tại
- `requirements.txt` đang có `torch`, `torchvision`, `numpy==2.1.2`, `opencv-python`, `PyQt5`, `pypylon`; `onnxruntime` đang comment.
- `app/models/anomaly.py` import `onnxruntime` tùy chọn, thiếu sẽ lỗi runtime khi tạo detector ONNX.

## Rủi ro chính
1. **torch/torchvision/CUDA**: AnomalyNCD nhiều khả năng khóa version cụ thể; dễ xung đột với stack hiện tại.
2. **numpy/opencv ABI**: chênh version có thể làm lỗi binary wheels.
3. **PyQt5**: không nên trộn dependency nghiên cứu vào env UI production.
4. **onnxruntime + torch CUDA**: cạnh tranh provider/GPU memory khi chạy chung process.

## Kết luận
- **Không khuyến nghị import trực tiếp AnomalyNCD vào process PyQt production ngay**.
- Khuyến nghị chạy **sidecar env** riêng cho nghiên cứu (venv/conda tách biệt), giao tiếp qua file/CSV hoặc RPC nhẹ.

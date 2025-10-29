import cv2
import numpy as np
import os
from glob import glob
import time

# Tạo thư mục lưu các ảnh cắt
output_dir = r"D:\07. NQbactruc\Images sample\QA\23-233\NG_Crop"
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn tới thư mục chứa ảnh
input_folder = r"D:\07. NQbactruc\Images sample\QA\23-233\NG"
image_paths = glob(os.path.join(input_folder, "*.*"))

# Kiểm tra nếu không có ảnh trong thư mục
if not image_paths:
    print("Không tìm thấy ảnh trong thư mục!")
else:
    print(f"Tìm thấy {len(image_paths)} ảnh trong thư mục.")

# Xử lý từng ảnh trong thư mục
for img_index, image_path in enumerate(image_paths):
    print(f"Đang xử lý ảnh {img_index + 1}/{len(image_paths)}: {image_path}")

    start_time = time.time()
    original_image = cv2.imread(image_path)  # Đọc ảnh gốc
    if original_image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        continue

    # Tính chiều dài và chiều rộng ảnh gốc
    image_height, image_width = original_image.shape[:2]
    print(f"Chiều dài ảnh gốc: {image_width}, Chiều rộng ảnh gốc: {image_height}")

    # Chuyển ảnh sang ảnh xám
    imgray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)

    # Thu nhỏ lại bằng erode
    ker = np.ones((9, 9), np.uint8)
    imgray = cv2.erode(imgray, ker, iterations=5)

    # Áp dụng ngưỡng hóa (thresholding) để tạo ảnh nhị phân
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    # Giãn nở lại để khôi phục hình dạng
    kernel1 = np.ones((4, 4), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations=5)

    # Phát hiện hình tròn bằng HoughCircles
    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=600,
        param1=200,
        param2=9,
        minRadius=300,  # Bán kính tối thiểu 23_240
        maxRadius=340   # Bán kính tối đa 23_240
        # minRadius=200,  # Bán kính tối thiểu 23_222
        # maxRadius=270   # Bán kính tối đa 23_222
    )

    time_xla = time.time() - start_time
    print(f"Thời gian xử lý tìm vòng tròn: {time_xla:.2f} giây")

    # Biến đếm để đặt tên file
    circle_index = 0

    # Xử lý các hình tròn được phát hiện
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # Tọa độ tâm
            radius = i[2] + 16     # Tăng bán kính một chút

            # Tạo mask hình tròn
            mask = np.zeros_like(original_image)
            cv2.circle(mask, center, radius, (255, 255, 255), -1)  # Vẽ hình tròn trắng
            mask = mask[:, :, 0] > 0  # Chuyển thành mask nhị phân

            # Tạo ảnh nền đen
            cropped_image = np.zeros_like(original_image)
            # Áp dụng mask để giữ lại vùng hình tròn
            cropped_image[mask] = original_image[mask]

            # Cắt vùng bao quanh hình tròn để giảm kích thước ảnh
            crop_x1 = max(center[0] - radius, 0)
            crop_x2 = min(center[0] + radius, image_width)
            crop_y1 = max(center[1] - radius, 0)
            crop_y2 = min(center[1] + radius, image_height)
            cropped_image = cropped_image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Tính chiều dài và chiều rộng của vùng crop
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            print(f"Vòng tròn {circle_index}: Chiều rộng = {crop_width}, Chiều cao = {crop_height}")

            # Lưu ảnh cắt
            filename = os.path.basename(image_path).split('.')[0]
            cropped_image_path = os.path.join(output_dir, f"{filename}_circle_{circle_index}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)
            circle_index += 1

    print(f"Đã xử lý xong ảnh {img_index + 1}/{len(image_paths)}!\n")

print("Hoàn thành xử lý tất cả các ảnh!")
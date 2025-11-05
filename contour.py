
import cv2
import time
import numpy as np

start_time = time.time()
# Đọc ảnh từ đường dẫn
img = cv2.imread(r"D:\07. NQbactruc\Images sample\20250510081249.jpg")

# Kiểm tra nếu ảnh được tải thành công
if img is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    # Chuyển ảnh sang ảnh xám
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    
    # Thu nhỏ lại
    ker = np.ones((9, 9), np.uint8)
    imgray = cv2.erode(imgray, ker, iterations=5)

    # Áp dụng ngưỡng hóa (thresholding) để tạo ảnh nhị phân
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    # Giãn nở lại để khôi phục hình dạng ban đầu
    kernel1 = np.ones((4, 4), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations=5)

    # Phát hiện đường tròn bằng HoughCircles
    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1,           # Độ phân giải accumulator
        minDist=600,    # Khoảng cách tối thiểu giữa các tâm
        param1=200,     # Ngưỡng gradient
        param2=9,       # Ngưỡng tích lũy
        minRadius=300,  # Bán kính tối thiểu
        maxRadius=340   # Bán kính tối đa
    )
    
    time_xla = time.time() - start_time
    print("timexla_timcircle: ", time_xla)

    # Phát hiện cạnh bằng Canny
    param1 = 100  # Ngưỡng trên cho Canny Edge Detection
    param2 = param1 // 2  # Ngưỡng dưới
    edges = cv2.Canny(thresh, param2, param1)
    
    # Chồng các cạnh lên hình ảnh gốc
    img_with_edges = thresh
    img_with_edges = cv2.cvtColor(img_with_edges, cv2.COLOR_GRAY2RGB)
    img_with_edges[edges > 0] = [0, 255, 0]  # Tô màu xanh lá cho các cạnh

    # Vẽ các hình tròn được phát hiện và hiển thị đường kính
    circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = len(circles[0, :])  # Đếm số lượng đường tròn
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            diameter = 2 * radius  # Tính đường kính
            
            # Vẽ tâm và viền đường tròn
            cv2.circle(img, center, 1, (0, 255, 0), 9)  # Tâm: xanh lá
            cv2.circle(img, center, radius + 15, (0, 255, 0), 9)  # Viền: xanh lá
            
            # Hiển thị đường kính trên ảnh
            text = f"D={round(diameter * 0.00854, 2)}mm"
            text_position = (center[0] - 50-200, center[1] - radius - 30+400)  # Vị trí văn bản (phía trên đường tròn)
            cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)

    # Hiển thị số lượng đường tròn phát hiện được
    count_text = f"SO LUONG BAC TRUC: {circle_count}"
    cv2.putText(img, count_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3, cv2.LINE_AA)

    # Lưu ảnh kết quả

    cv2.imwrite('contour.jpg', img)

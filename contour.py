#param1 càng cao càng ít cạnh tìm thấy →tốc độ xử lý nhanh hơn
#param2 cao: Ít hình tròn được phát hiện → kết quả chính xác hơn, ít phát hiện sai, nhưng có thể bỏ sót hình tròn không hoàn hảo.
#param2 thấp: Nhiều hình tròn được phát hiện → tăng khả năng phát hiện hình tròn không hoàn hảo, nhưng dễ phát hiện sai.

# import cv2
# # Đọc ảnh từ đường dẫn
# img = cv2.imread(r'D:\07. NQbactruc\Images sample\dataset_3\20250415082359.jpg')

# # Kiểm tra nếu ảnh được tải thành công
# if img is None:
#     print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
# else:
#     # Chuyển ảnh sang ảnh xám
#     imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgray = cv2.GaussianBlur(imgray, (3,3 ), 0)
#     # Áp dụng ngưỡng hóa (thresholding) để tạo ảnh nhị phân
#     ret, thresh = cv2.threshold(imgray, 60, 255, 0)

#     # Tìm contours trong ảnh nhị phân
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     # In thông tin số lượng contours và thông tin của contour đầu tiên
#     print("Number of contours = " + str(len(contours)))
#     if len(contours) > 0:
#         print("First contour points:", contours[0])

#     # Vẽ tất cả contours lên ảnh gốc
#     cv2.drawContours(img, contours, -1, (0, 255, 0), 13)

#     cv2.imwrite('contour.jpg', img)


# import cv2 # tim được hết nhưng bị nhiễu bỏi phần nối
# import time# và vẽ đường tròn ngoại tiếp
# import numpy as np
# start_time=time.time()
# # Đọc ảnh từ đường dẫn
# img = cv2.imread(r"D:\07. NQbactruc\Images sample\anomally\prepare\20250417103526.jpg")

# # Kiểm tra nếu ảnh được tải thành công
# if img is None:
#     print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
# else:
#     # Chuyển ảnh sang ảnh xám
#     imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
#     cv2.imwrite('contour0.jpg', imgray)
    
#     # Thu nhỏ lại
#     ker = np.ones((9, 9), np.uint8)
#     imgray = cv2.erode(imgray, ker, iterations=5)
#     cv2.imwrite('contour1.jpg', imgray)

#     # Áp dụng ngưỡng hóa (thresholding) để tạo ảnh nhị phân
#     ret, thresh = cv2.threshold(imgray, 50, 255, 0)
#     cv2.imwrite('contour2.jpg', thresh)

#     # Giãn nở lại để khôi phục hình dạng ban đầu
#     kernel1 = np.ones((4, 4), np.uint8)
#     thresh = cv2.dilate(thresh, kernel1, iterations=5)
#     cv2.imwrite('contour3.jpg', thresh)

#     # Thực hiện phép mở (morphological opening) để loại bỏ phần nối
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (110, 110))  # Kernel hình elip
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     cv2.imwrite('contour4.jpg', thresh)

#     # # test fillup hole
#     # kerne = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # # Áp dụng morphological closing
#     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kerne)
#     # cv2.imwrite('contour3-3.jpg', thresh)

#     # # test fillup hole
#     # kerne = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     # # Áp dụng morphological closing
#     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kerne)
#     # cv2.imwrite('contourhole.jpg', thresh)

#     # # Tạo kernel (ma trận 3x3)
#     # kernel = np.ones((7, 7), np.uint8)
#     # # Áp dụng Erosion
#     # eroded = cv2.erode(thresh, kernel, iterations=11)
#     # # Giãn nở lại để khôi phục hình dạng ban đầu
#     # eroded = cv2.dilate(eroded, kernel, iterations=5)
#     # cv2.imwrite('contour1.jpg', thresh)
#     # cv2.imwrite('contour2.jpg', eroded)

#     # Tìm contours trong ảnh nhị phân
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     time_timcontour = time.time() - start_time
#     print("timxla: ",time_timcontour)
#     # In thông tin số lượng contours ban đầu
#     # print("Tổng số contours = " + str(len(contours)))

#     # Danh sách lưu các contour có bán kính lớn hơn 300
#     filtered_contours = []

#     # Lọc các contour có bán kính lớn hơn 300
#     for contour in contours:
#         # Tính hình tròn ngoại tiếp cho contour
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         if radius > 280:
#             filtered_contours.append(contour)
#              # Vẽ đường tròn ngoại tiếp lên ảnh gốc
#             center = (int(x), int(y))  # Tọa độ tâm hình tròn
#             radius = int(radius+15)      # Bán kính hình tròn
#             cv2.circle(img, center, radius, (0, 255, 0), 2)  # Màu xanh lá, độ dày 2

#     # # In số lượng contours sau khi lọc
#     print("Số contours có bán kính > 300 = " + str(len(filtered_contours)))
#     img1 = cv2.imread(r"D:\07. NQbactruc\Images sample\anomally\prepare\20250417103526.jpg")
#     # # Vẽ chỉ các contours thỏa mãn điều kiện lên ảnh gốc
#     cv2.drawContours(img1, filtered_contours, -1, (0, 255, 0), 5)
#     cv2.imwrite('contourk.jpg', img1)
#     time_ve_contour = time.time() - start_time
#     print(time_ve_contour)
#     # Lưu ảnh kết quả
#     cv2.imwrite('contourt.jpg', img)

# import cv2 # xu ly 0.8s(cần cải thiện bằng cách dùng resize) phát hiện ok với hough circle
# import time# và vẽ đường tròn ngoại tiếp
# import numpy as np
# start_time=time.time()
# # Đọc ảnh từ đường dẫn
# start_time = time.time()
# img = cv2.imread(r"D:\07. NQbactruc\Images sample\20250510081249.jpg")

# # Kiểm tra nếu ảnh được tải thành công
# if img is None:
#     print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
# else:
#     # Chuyển ảnh sang ảnh xám
#     imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
#     # cv2.imwrite('contour0.jpg', imgray)
    
#     # Thu nhỏ lại
#     ker = np.ones((9, 9), np.uint8)
#     imgray = cv2.erode(imgray, ker, iterations=5)
#     # cv2.imwrite('contour1.jpg', imgray)

#     # Áp dụng ngưỡng hóa (thresholding) để tạo ảnh nhị phân
#     ret, thresh = cv2.threshold(imgray, 50, 255, 0)
#     # cv2.imwrite('contour2.jpg', thresh)

#     # Giãn nở lại để khôi phục hình dạng ban đầu
#     kernel1 = np.ones((4, 4), np.uint8)
#     thresh = cv2.dilate(thresh, kernel1, iterations=5)
#     # cv2.imwrite('contour3.jpg', thresh)

#     # # Thực hiện phép mở (morphological opening) để loại bỏ phần nối(tin nhieu time)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (110, 110))  # Kernel hình elip
#     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     # cv2.imwrite('contour4.jpg', thresh)

#     circles = cv2.HoughCircles(
#         # imgray,
#         thresh,
#         cv2.HOUGH_GRADIENT,
#         dp=1,           # Độ phân giải accumulator
#         minDist=600,    # Khoảng cách tối thiểu giữa các tâm (2 lần bán kính)
#         param1=200,      # Ngưỡng gradient (giảm vì ảnh nhị phân)
#         param2=9,      # Ngưỡng tích lũy (giảm để tăng độ nhạy)
#         minRadius=300,  # Bán kính tối thiểu
#         maxRadius=340
# )
    
#     time_xla=time.time() - start_time
#     print("timexla_timcircle: ", time_xla)

#     # Thử nghiệm với giá trị param1
#     param1 = 100  # Ngưỡng trên cho Canny Edge Detection
#     param2 = param1 // 2  # Ngưỡng dưới (một nửa giá trị của param1)
#     # Phát hiện cạnh bằng Canny
#     im=thresh
#     edges = cv2.Canny(im, param2, param1)
#     # Chồng các cạnh lên hình ảnh gốc (tô màu cạnh)
#     img_with_edges = im
#     img_with_edges = cv2.cvtColor(img_with_edges, cv2.COLOR_BGR2RGB)
#     img_with_edges[edges > 0] = [0, 255, 0]  # Tô màu xanh lá cho các cạnh
#     # cv2.imwrite('contourcanny.jpg', img_with_edges)

#     # Vẽ các hình tròn được phát hiện
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             center = (i[0], i[1])
#             cv2.circle(img, center, 1, (0, 255, 0), 9)  # Tâm: xanh lá
#             radius = i[2]
#             cv2.circle(img, center, radius+15, (0, 255, 0), 9)  # Viền: đỏ
#     cv2.imwrite('contour.jpg', img)


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
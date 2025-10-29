
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer
from ultralytics import YOLO
from pypylon import pylon
import numpy as np
import cv2, random, socket, inspect
import os, datetime, time
from threading import Thread

class Ui_MainWindow(object):

    # def setupUi(self, MainWindow):
    #     MainWindow.setObjectName("MainWindow")
    #     MainWindow.resize(1300, 750)

    #     # self.soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #     self.soc = ""

    #     self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
    #     self.centralwidget.setObjectName("centralwidget")
    #     self.error_label = QtWidgets.QLabel(parent=self.centralwidget)
    #     self.error_label.setStyleSheet("background-color: white;")
    #     self.error_label.setGeometry(QtCore.QRect(1050, 360, 200, 200))
    #     self.error_label.setObjectName("error_label")
    #     self.error_label.setWordWrap(True)

    #     self.textEdit = QtWidgets.QTextEdit(parent=self.centralwidget)
    #     self.textEdit.move(25, 0)
    #     self.textEdit.setObjectName("textEdit")
    #     font = QtGui.QFont()
    #     font.setPointSize(72)
    #     self.textEdit.setFont(font)
    #     self.textEdit.setStyleSheet("border: none; background-color: transparent; selection-background-color: transparent;")

    #     self.image_label = QtWidgets.QLabel(parent=self.centralwidget)
    #     self.image_label.setGeometry(QtCore.QRect(10, 140, 800, 550))
    #     self.image_label.setObjectName("label")
    #     self.Loadmodelbt = QtWidgets.QPushButton('Load model',parent=self.centralwidget)
    #     self.Loadmodelbt.setGeometry(QtCore.QRect(1050, 20, 191, 61))
    #     self.Loadmodelbt.setObjectName("Loadmodelbt")

    #     self.open_button = QtWidgets.QPushButton('Open Image Folder',parent=self.centralwidget)
    #     self.open_button.setGeometry(QtCore.QRect(1050, 90, 191, 61))
    #     self.open_button.setObjectName("Open_Folder")
    
    #     self.next_button = QtWidgets.QPushButton('Next',parent=self.centralwidget)
    #     self.next_button.setGeometry(QtCore.QRect(1150, 160, 93, 28))
    #     self.next_button.setObjectName("next_button")
    #     self.prev_button = QtWidgets.QPushButton('Previous',parent=self.centralwidget)
    #     self.prev_button.setGeometry(QtCore.QRect(1050, 160, 93, 28))
    #     self.prev_button.setObjectName("prev_button")
    #     self.measure_button = QtWidgets.QPushButton('Measure',parent=self.centralwidget)
    #     self.measure_button.setGeometry(QtCore.QRect(1050, 260, 93, 40))
    #     self.measure_button.setObjectName("measure_button")
    #     self.SaveImgDirBT = QtWidgets.QPushButton('. . .',parent=self.centralwidget)
    #     self.SaveImgDirBT.setGeometry(QtCore.QRect(1245, 300, 30, 20))
    #     self.SaveImgDirBT.setObjectName("SaveImgDir")
    #     self.stop_continuous = QtWidgets.QPushButton('Stop continuous',parent=self.centralwidget)
    #     self.stop_continuous.setGeometry(QtCore.QRect(1050, 260, 93, 40))
    #     self.stop_continuous.setObjectName("Stop_continuous")
    #     self.stop_continuous.setVisible(False)
    #     self.reconnect_button = QtWidgets.QPushButton('Reconnect',parent=self.centralwidget)
    #     self.reconnect_button.setGeometry(QtCore.QRect(1050, 580, 93, 40))
    #     self.reconnect_button.setObjectName("reconnect_button")
    #     self.checkbox = QtWidgets.QCheckBox("Chế độ continuous", parent=self.centralwidget)
    #     self.checkbox.move(1150, 260)
    #     self.checkbox.setObjectName("checkbox")
    #     self.checkbox_bbox = QtWidgets.QCheckBox("Vẽ bbox", parent=self.centralwidget)
    #     self.checkbox_bbox.move(1150, 280)
    #     self.checkbox_bbox.setObjectName("checkbox_bbox")
    #     self.checkbox_saveim = QtWidgets.QCheckBox("Save image as", parent=self.centralwidget)
    #     self.checkbox_saveim.move(1150, 300)
    #     self.checkbox_saveim.setObjectName("checkbox_saveim")

    #     MainWindow.setCentralWidget(self.centralwidget)
    #     self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
    #     self.menubar.setGeometry(QtCore.QRect(0, 0, 1107, 10))
    #     self.menubar.setObjectName("menubar")
    #     MainWindow.setMenuBar(self.menubar)
    #     self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
    #     self.statusbar.setObjectName("statusbar")
    #     MainWindow.setStatusBar(self.statusbar)

    #     self.timer = QTimer()
    #     self.timer.timeout.connect(self.update_frame)

    #     self.model = None
    #     self.image_list = []
    #     self.current_index = 0
    #     self.SaveImgPath = "D:\\AI\\yolov8"

    #     self.next_button.clicked.connect(self.show_next_image)
    #     self.stop_continuous.clicked.connect(self.stopcontinuous)
    #     self.prev_button.clicked.connect(self.show_prev_image)
    #     self.open_button.clicked.connect(self.open_folder)
    #     self.Loadmodelbt.clicked.connect(self.load_model)
    #     self.measure_button.clicked.connect(self.measure)
    #     self.SaveImgDirBT.clicked.connect(self.Save_Img_Dir)
    #     self.reconnect_button.clicked.connect(self.reconnect)
    #     self.errors = []
    #     self.create()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 750)
        MainWindow.setMinimumSize(1300, 750)

        # Central widget
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)

        # Left side: Image and error display
        left_layout = QtWidgets.QVBoxLayout()

        # QTextEdit để hiển thị trạng thái OK/NG
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setReadOnly(True)  # Chỉ cho phép đọc để hiển thị trạng thái
        self.textEdit.setFixedSize(130, 100)  # Đặt kích thước cố định
        font = QtGui.QFont()
        font.setPointSize(60)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet(
            "border: none; background-color: transparent; selection-background-color: transparent;"
        )
        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_layout.addWidget(self.textEdit)
        left_layout.addStretch(1)

        # Image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setObjectName("image_label")
        self.image_label.setMinimumSize(800, 550)
        self.image_label.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.image_label)
        left_layout.addStretch(1)

        # Error display (QScrollArea containing QLabel)
        self.error_scroll_area = QtWidgets.QScrollArea()
        # self.error_scroll_area.setStyleSheet("background-color: white; border: 1px solid black;")
        self.error_scroll_area.setWidgetResizable(True)
        self.error_scroll_area.setMinimumSize(200, 50)

        self.error_label = QtWidgets.QLabel()
        self.error_label.setObjectName("error_label")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.error_scroll_area.setWidget(self.error_label)

        left_layout.addWidget(self.error_scroll_area, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, stretch=3)

        # Right side: Controls
        right_layout = QtWidgets.QVBoxLayout()

        # File controls
        file_group = QtWidgets.QGroupBox("File Controls")
        file_layout = QtWidgets.QVBoxLayout()
        self.Loadmodelbt = QtWidgets.QPushButton("Load Model")
        self.Loadmodelbt.setObjectName("Loadmodelbt")
        file_layout.addWidget(self.Loadmodelbt)

        self.open_button = QtWidgets.QPushButton("Open Image Folder")
        self.open_button.setObjectName("Open_Folder")
        file_layout.addWidget(self.open_button)

        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)

        # Navigation controls
        nav_group = QtWidgets.QGroupBox("Navigation")
        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.setObjectName("prev_button")
        nav_layout.addWidget(self.prev_button)

        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.setObjectName("next_button")
        nav_layout.addWidget(self.next_button)

        nav_group.setLayout(nav_layout)
        right_layout.addWidget(nav_group)

        # Measurement controls
        measure_group = QtWidgets.QGroupBox("Measurement")
        measure_layout = QtWidgets.QVBoxLayout()
        self.measure_button = QtWidgets.QPushButton("Measure")
        self.measure_button.setObjectName("measure_button")
        measure_layout.addWidget(self.measure_button)

        # Thêm QSpinBox để nhập thời gian timer
        self.timer_interval_spinbox = QtWidgets.QSpinBox()
        self.timer_interval_spinbox.setObjectName("timer_interval_spinbox")
        self.timer_interval_spinbox.setRange(50, 10000)  # Giới hạn từ 50ms đến 10s
        self.timer_interval_spinbox.setValue(200)  # Giá trị mặc định là 200ms
        self.timer_interval_spinbox.setSuffix(" ms")  # Hiển thị đơn vị ms
        self.timer_interval_spinbox.setSingleStep(50)  # Bước nhảy 50ms
        measure_layout.addWidget(self.timer_interval_spinbox)

        self.stop_continuous = QtWidgets.QPushButton("Stop Continuous")
        self.stop_continuous.setObjectName("Stop_continuous")
        self.stop_continuous.setVisible(False)
        measure_layout.addWidget(self.stop_continuous)

        # Add start measure button
        self.start_measure_button = QtWidgets.QPushButton("RUN")
        self.start_measure_button.setObjectName("start_measure_button")
        measure_layout.addWidget(self.start_measure_button)

        self.checkbox = QtWidgets.QCheckBox("continuous")
        self.checkbox.setObjectName("checkbox")
        measure_layout.addWidget(self.checkbox)

        self.checkbox_bbox = QtWidgets.QCheckBox("Predict")
        self.checkbox_bbox.setObjectName("checkbox_bbox")
        measure_layout.addWidget(self.checkbox_bbox)

        self.checkbox_saveim = QtWidgets.QCheckBox("Save image as")
        self.checkbox_saveim.setObjectName("checkbox_saveim")
        measure_layout.addWidget(self.checkbox_saveim)

        measure_group.setLayout(measure_layout)
        right_layout.addWidget(measure_group)

        # Save controls
        save_group = QtWidgets.QGroupBox("Save Settings")
        save_layout = QtWidgets.QHBoxLayout()
        self.SaveImgDirBT = QtWidgets.QPushButton(". . .")
        self.SaveImgDirBT.setObjectName("SaveImgDir")
        save_layout.addWidget(self.SaveImgDirBT)
        save_group.setLayout(save_layout)
        right_layout.addWidget(save_group)

        # Reconnection controls
        reconnect_group = QtWidgets.QGroupBox("Reconnect")
        reconnect_layout = QtWidgets.QVBoxLayout()
        self.reconnect_button = QtWidgets.QPushButton("Reconnect")
        self.reconnect_button.setObjectName("reconnect_button")
        reconnect_layout.addWidget(self.reconnect_button)
        reconnect_group.setLayout(reconnect_layout)
        right_layout.addWidget(reconnect_group)

        # Add right layout to main layout
        main_layout.addLayout(right_layout)

        # Set central widget and menu bar
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Timer and variables
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.measure)  # Call self.measure every timeout

        self.model = None
        self.image_list = []
        self.current_index = 0
        self.SaveImgPath = "D:\\AI\\yolov8"

        # Connect buttons to functions
        self.next_button.clicked.connect(self.show_next_image)
        self.stop_continuous.clicked.connect(self.stopcontinuous)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.open_button.clicked.connect(self.open_folder)
        self.Loadmodelbt.clicked.connect(self.load_model)
        self.measure_button.clicked.connect(self.measure)
        self.SaveImgDirBT.clicked.connect(self.Save_Img_Dir)
        self.reconnect_button.clicked.connect(self.reconnect)

        # Connect start_measure_button to start/stop timer
        self.start_measure_button.clicked.connect(self.toggle_measure_timer)

        # Errors
        self.errors = []
        self.create()

    def toggle_measure_timer(self):
        """Start/stop the measure timer."""
        if not self.timer1.isActive():
            interval = self.timer_interval_spinbox.value()  # Lấy giá trị từ QSpinBox
            self.timer1.start(interval)  # Sử dụng giá trị interval
            self.start_measure_button.setText("STOP Measure . . .")
        else:
            self.timer1.stop()
            self.start_measure_button.setText("RUN")

    def create(self):
        self.errors = []
        ip = "192.168.0.10"
        port = 8501
        try: #PLC
            self.soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.soc.connect((ip,port))
        except OSError :
            self.errors.append("Can't not connect to PLC"+ str(inspect.currentframe().f_lineno))
        self.writedata("R1000", 1)

        # try: #Camera USB
        #     self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # except Exception as e:
        #     self.writedata("R1000", 0)
        #     print("115: ",self.readdata("R1000"))  
        #     self.errors.append("Can't connect to Camera: "+ str(inspect.currentframe().f_lineno) + str(e))
        # else:
        #     if not self.camera.IsOpen():
        #         self.camera.Open()
        #         self.camera.PixelFormat.value = "BGR8"
        #         self.camera.ExposureTime.value = 5000 

        # try: #Camera ETHERNET
        #     tlf = pylon.TlFactory.GetInstance()
        #     tl = tlf.CreateTl('BaslerGigE')
        #     cam_info = tl.CreateDeviceInfo()
        #     cam_info.SetIpAddress('192.168.1.124')
        #     self.camera = pylon.InstantCamera(tlf.CreateDevice(cam_info))
        # except Exception as e:
        #     self.writedata("R1000", 0)
        #     print("115: ",self.readdata("R1000"))  
        #     self.errors.append("Can't connect to Camera: "+ str(inspect.currentframe().f_lineno) + str(e))
        # else:
        #     if not self.camera.IsOpen():
        #         self.camera.Open()
        #         self.camera.PixelFormat.value = "BGR8"
        #         self.camera.ExposureTime.value = 5000 
        # print(str(inspect.currentframe().f_lineno),self.readdata("R1000")) 
        # self.error_label.setText("\n\n".join(self.errors))


        # try:  # Camera ETHERNET
        #     tlf = pylon.TlFactory.GetInstance()
        #     tl = tlf.CreateTl('BaslerGigE')
        #     cam_info = tl.CreateDeviceInfo()
        #     cam_info.SetIpAddress('192.168.1.124')
        #     self.camera = pylon.InstantCamera(tlf.CreateDevice(cam_info))
        #     # Kiểm tra nếu camera chưa mở
        #     if not self.camera.IsOpen():
        #         self.camera.Open()
        #     # Cấu hình thông số camera
        #     if "BGR8" in self.camera.PixelFormat.Symbolics:
        #         self.camera.PixelFormat.value = "BGR8"
        #         print("BGR8 IN")
        #     else:
        #         raise Exception("PixelFormat 'BGR8' không được hỗ trợ bởi camera này!")
        #     # Kiểm tra giá trị ExposureTime
        #     exposure_time = 5000
        #     if self.camera.ExposureTime.Min <= exposure_time <= self.camera.ExposureTime.Max:
        #         self.camera.ExposureTime.value = exposure_time
        #     else:
        #         raise Exception(
        #             f"ExposureTime {exposure_time} nằm ngoài giới hạn cho phép: "
        #             f"[{self.camera.ExposureTime.Min}, {self.camera.ExposureTime.Max}]"
        #         )
        # except Exception as e:
        #     # Xử lý lỗi
        #     self.writedata("R1000", 0)
        #     print("115: ", self.readdata("R1000"))
        #     self.errors.append(f"Can't connect to Camera: {str(inspect.currentframe().f_lineno)} {str(e)}")
        # else:
        #     print("Camera đã kết nối và cấu hình thành công!")



        try:  # Kết nối camera Ethernet

            # # Tạo kết nối với camera
            # tlf = pylon.TlFactory.GetInstance()
            # tl = tlf.CreateTl('BaslerGigE')
            # cam_info = tl.CreateDeviceInfo()
            # cam_info.SetIpAddress('192.168.1.124')  # Địa chỉ IP của camera
            # self.camera = pylon.InstantCamera(tlf.CreateDevice(cam_info))

            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            print(len(devices))
            if len(devices) != 0:
                for device in devices:

                    if device.GetDeviceClass() == "BaslerUsb":
                        self.camera = pylon.InstantCamera(tlf.CreateDevice(device))
                        break  # Nếu bạn chỉ muốn kết nối với thiết bị đầu tiên phù hợp
                    elif device.GetDeviceClass() == "BaslerGigE":
                        tl = tlf.CreateTl('BaslerGigE')
                        cam_info = tl.CreateDeviceInfo()
                        cam_info.SetIpAddress('192.168.1.124')  # Địa chỉ IP của camera
                        self.camera = pylon.InstantCamera(tlf.CreateDevice(cam_info))
                        break  # Nếu bạn chỉ muốn kết nối với thiết bị đầu tiên phù hợp

            # Mở camera
            if not self.camera.IsOpen():
                self.camera.Open()

            # # **Tải cấu hình mặc định của camera**
            if hasattr(self.camera, "UserSetSelector") and hasattr(self.camera, "UserSetLoad"):
                print("Đang tải cấu hình mặc định từ camera...")
                self.camera.UserSetSelector.SetValue("Default")  # Chọn cấu hình mặc định (Default User Set)
                ## self.camera.UserSetSelector.SetValue("UserSet1")
                self.camera.UserSetLoad.Execute()  # Tải cấu hình
                if "BayerRG8" in self.camera.PixelFormat.Symbolics:
                    self.camera.PixelFormat.SetValue("BayerRG8")
                    print("Pixel Format đã được đặt thành Bayer RG8.")
                else:
                    print("Camera không hỗ trợ định dạng Bayer RG8.")
                print("Cấu hình mặc định đã được tải thành công.")
                if device.GetDeviceClass() == "BaslerGigE":
                    ## Tối ưu hóa buffer
                    self.camera.MaxNumBuffer = 50
                    self.camera.OutputQueueSize = 50
                    # Tăng khoảng cách giữa các packet (nếu hỗ trợ)
                    if hasattr(self.camera, "GevSCPD"):
                        self.camera.GevSCPD.SetValue(10000)  # 10 micro giây
                    # Tăng kích thước packet (nếu hỗ trợ)
                    if hasattr(self.camera, "GevSCPSPacketSize"):
                        self.camera.GevSCPSPacketSize.SetValue(1500)
                    # Giảm tốc độ khung hình
                    if hasattr(self.camera, "AcquisitionFrameRateEnable"):
                        self.camera.AcquisitionFrameRateEnable.SetValue(True)
                        self.camera.AcquisitionFrameRate.Value = 15
                    # khắc phục các ảnh chụp không đồng nhất (đậm, nhạt)
                if hasattr(self.camera, "BalanceWhiteAuto"):
                    self.camera.BalanceWhiteAuto.SetValue("Off")  # Tắt Auto White Balance
                if hasattr(self.camera, "ExposureAuto"):
                    self.camera.ExposureAuto.SetValue("Off")  # Tắt Auto Exposure
                    self.camera.ExposureTime.SetValue(4000)# cam2 30000 len mca25, cam1 12000 len tcm03-150, nqpssf266: C2 8000, C1 4000
                if hasattr(self.camera, "GainAuto"):
                    self.camera.GainAuto.SetValue("Off")  # Tắt Auto Gain
            else:
                print("Camera không hỗ trợ User Set hoặc không thể tải cấu hình mặc định.")


        except Exception as e:
            self.writedata("R1000", 0)
            print("115: ", self.readdata("R1000"))
            print(e)
            self.errors.append(f"Can't connect to Camera: {str(e)}")
        else:
            print("Kết nối và cấu hình camera thành công!")



    def reconnect(self):
        self.create()
        self.start_plc_monitoring()
        
    def load_model(self):
            model_path, _ = QFileDialog.getOpenFileName()
            if model_path:
                self.model = YOLO(model_path)
                self.model.fuse()

    def predict(self, frame):
            results = self.model(frame, conf=0.4)
            return results
    
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(MainWindow, "Select Folder", os.path.expanduser("~"))
        if folder_path:
            self.load_images_from_folder(folder_path)

    def Save_Img_Dir(self):
        self.SaveImgPath = QFileDialog.getExistingDirectory(MainWindow, "Select Folder", os.path.expanduser("~"))

    def load_images_from_folder(self, folder_path):
        self.image_list = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        self.show_image()

    def stopcontinuous(self):
        self.timer.stop()
        self.camera.StopGrabbing()
        self.stop_continuous.setVisible(False) 

    def closeEvent(self, event):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()
        if self.soc:
            self.soc.close()
            print("Closing socket connection")
        event.accept() 

    def update_frame(self):
        self.grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if self.grabResult.GrabSucceeded():
                image = self.grabResult.Array
                if self.checkbox_bbox.isChecked():
                    image = self.plot_bbox(image)
        if image is not None:  # Đảm bảo rằng `image` có giá trị trước khi sử dụng
            self.show_image_main(image)

    def measure(self): #new
        try:
            if self.checkbox.isChecked():
                self.stop_continuous.setVisible(True)
                self.camera.AcquisitionMode.Value = "Continuous"
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.timer.start(50)
            else:
                # Bắt đầu đo thời gian
                start_time = time.time()
                self.camera.AcquisitionMode.Value = "SingleFrame"
                # Đảm bảo camera đã sẵn sàng
                if not self.camera.IsOpen():
                    self.camera.Open()
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                # Bắt đầu chụp
                self.camera.StartGrabbing()
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Lấy dữ liệu ảnh
                    image = grabResult.Array
                    print("Kích thước ảnh gốc: ", image.shape)
                    # Kiểm tra và chuyển đổi màu
                    if len(image.shape) == 2:  # Ảnh grayscale (có thể là Bayer RG 8)
                        # Kiểm tra và chuyển đổi từ Bayer RG 8 sang RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)

                    if len(image.shape) == 3:  # Ảnh có 3 kênh
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Hiển thị ảnh
                    if self.checkbox_bbox.isChecked():
                        image = self.plot_bbox(image)
                    self.show_image_main(image)
                    # Lưu ảnh nếu cần
                    if self.checkbox_saveim.isChecked():
                        if self.SaveImgPath:
                            now = datetime.datetime.now()
                            filename = now.strftime("%Y%m%d%H%M%S") + ".jpg"
                            file_path = self.SaveImgPath + "/" + filename
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(file_path, image)
                            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100]) # Giữ lại 95% chất lượng
                            print("Ảnh đã được lưu tại: ", file_path)
                        else:
                            print("Chọn đường dẫn lưu")
                else:
                    print("Lỗi khi chụp hình: ", grabResult.ErrorDescription)
                    grabResult.Release()
                    return  # Thoát khỏi hàm nếu không lấy được hình ảnh
                grabResult.Release()
                self.camera.StopGrabbing()
                # Kết thúc đo thời gian
                end_time = time.time()
                # Tính và in thời gian xử lý
                processing_time = end_time - start_time
                print("Thoi gian xử lý: ", processing_time, " giây")
        except Exception as e:
            print("Lỗi trong quá trình chụp hình: ", str(e)+ str(inspect.currentframe().f_lineno))

    def show_image_main(self,arr):
        # height, width, _ = arr.shape
        # bytesPerLine = 3 * width
        # qimage = QImage(arr.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        # qpixmap = QPixmap.fromImage(qimage)
        # self.image_label.setPixmap(qpixmap.scaled(self.image_label.size(), aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    #sửa lỗi 2 or 3 kênh màu(fixed):
        if len(arr.shape) == 3:  # Ảnh màu (RGB hoặc BGR)
            height, width, _ = arr.shape
            bytesPerLine = 3 * width
            qimage = QImage(arr.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        elif len(arr.shape) == 2:  # Ảnh grayscale
            height, width = arr.shape
            bytesPerLine = width
            # Chuyển ảnh grayscale thành RGB để QImage xử lý
            arr_rgb = np.stack((arr, arr, arr), axis=-1)  # Sao chép kênh để tạo ảnh RGB
            qimage = QImage(arr_rgb.data, width, height, bytesPerLine * 3, QImage.Format.Format_RGB888)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")
        # Chuyển đổi thành QPixmap và hiển thị
        qpixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(qpixmap.scaled(self.image_label.size(), aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio))


    def judge(self,results):
        class_names = []
        for r in results:
            class_ids = r.boxes.cls
            for class_id in class_ids:
                class_name = self.model.names[int(class_id)]
                # if class_name not in class_names:
                class_names.append(class_name)
        print(class_names)
        try:
            jud = 1 if not class_names else 0
            if jud == 1:
                self.textEdit.setPlainText("OK")
                self.textEdit.setStyleSheet("color: green; border: none; background-color: transparent;") 
            else:
                self.textEdit.setPlainText("NG")
                self.textEdit.setStyleSheet("color: red; border: none; background-color: transparent;")
        except ValueError:
            self.textEdit.setPlainText("Invalid input")
            self.textEdit.setStyleSheet("color: black; border: none; background-color: transparent;")

    def plot_bbox(self,image):
        results = self.predict(image)
        self.judge(results)
        DP = results[0].cpu().numpy()
        class_list = list(self.model.names.values())
        detection_colors = [255, 0, 0]
        # for i in range(len(class_list)):
        #     r = random.randint(0, 255)
        #     g = random.randint(0, 255)
        #     b = random.randint(0, 255)
            # detection_colors.append((b, g, r))
            # detection_colors.append((128, 128, 128))
        # print(DP)
        if len(DP) != 0:
            print(len(results[0]))
            for i in range(len(results[0])):
                boxes = results[0].boxes.cpu()
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                cv2.rectangle(image,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),detection_colors[int(clsID)],3,)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(image,class_list[int(clsID)] + " " + (str(round(conf*100, 2))) + "%",(int(bb[0]), int(bb[1]) - 10),font,1,(255, 255, 255),2,)
        return image  
      
    def show_image(self):
        if not self.image_list:
            print("Danh sách hình ảnh rỗng, không thể hiển thị hình ảnh.")
            return
        # Kiểm tra chỉ số hiện tại có nằm trong phạm vi hợp lệ
        if not (0 <= self.current_index < len(self.image_list)):
            print("Chỉ số không hợp lệ. Đặt lại chỉ số về 0.")
            self.current_index = 0
        # Thử lấy hình ảnh từ danh sách
        try:
            pixmap = QPixmap(self.image_list[self.current_index])
        except Exception as e:
            print(f"Lỗi khi tải hình ảnh: {e}")
            return

        qimage = pixmap.toImage()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        arr = arr[:, :, :3]
        arr = arr.astype(np.uint8)
        results = self.predict(arr)
        self.judge(results)
        DP = results[0].cpu().numpy()
        class_list = list(self.model.names.values())
        detection_colors = []
        for i in range(len(class_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            detection_colors.append((b, g, r))
        if len(DP) != 0:
            for i in range(len(results[0])):
                boxes = results[0].boxes.cpu()
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                cv2.rectangle(arr,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),detection_colors[int(clsID)],3,)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(arr,class_list[int(clsID)] + " " + (str(round(conf*100, 2))) + "%",(int(bb[0]), int(bb[1]) - 10),font,1,(255, 255, 255),2,)
        height, width, channel = arr.shape
        bytesPerLine = 3 * width
        qimage = QImage(arr.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(qpixmap.scaled(self.image_label.size(), aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.show_image()

    def show_prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.show_image()

    def readdata(self, data):
        """
        # Thực hiện đọc dữ liệu từ PLC 
        data : Thanh ghi bên PLC. Vd : DM1
        """
        a = 'RD '
        c = '\x0D'
        d = a + data + c
        datasend = d.encode("UTF-8")
        try:
            self.soc.sendall(datasend)
            response = self.soc.recv(1024)
            dataFromPLC = response.decode("UTF-8")
            # print(dataFromPLC)
            data1 = int(dataFromPLC)
            return data1
        except OSError:
            self.errors.append("Can't not read data"+ str(inspect.currentframe().f_lineno))
        # print(data1)
    
    def writedata(self, register, data):
        """
        Ghi dữ liệu vào PLC 
        register : Thanh ghi cần ghi dữ liệu bên PLC
        data : Dữ liệu cần truyền là
        """
        a = 'WR '
        b = ' '
        c = '\x0D'
        d = a+ register + b + str(data) + c
        datasend  = d.encode("UTF-8")
        try:
            self.soc.sendall(datasend)
            response = self.soc.recv(1024)
            print(response)
        except OSError:
            self.errors.append("Can't not write data"+ str(inspect.currentframe().f_lineno))

    def start_plc_monitoring(self):
        self.plc_thread = Thread(target=self.monitor_plc)
        if self.readdata(data="R1000") == 1:
            self.running = True
            self.plc_thread.start()
        else:
            print(self.readdata(data="R1000"))
            self.errors.append("Can't not measure"+ str(inspect.currentframe().f_lineno))
            self.error_label.setText("\n\n".join(self.errors))

    def stop_plc_monitoring(self):
            self.running = False
            if self.plc_thread.is_alive():
                self.plc_thread.join()

    def monitor_plc(self, data="M200"):
        while self.running:
            data1 = self.readdata(data)
            if data1 == 1:
                self.writedata("R1000", 0)
                self.measure()
                self.writedata("R1000", 1)
                print(data1)
            time.sleep(0.5)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.start_plc_monitoring()  # Start PLC monitoring
    app.aboutToQuit.connect(ui.stop_plc_monitoring)
    sys.exit(app.exec())
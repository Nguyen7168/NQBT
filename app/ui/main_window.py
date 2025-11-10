"""PyQt5 main window controlling the inspection workflow."""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import AppConfig
from app.inspection.plc_client import PlcController
from app.inspection.workers import InspectionResult, InspectionWorker, PlcTriggerWorker, SaveWorker
from app.utils import numpy_to_qimage

LOGGER = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    trigger_manual = QtCore.pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        plc: PlcController,
        parent: Optional[QtWidgets.QWidget] = None,
        use_dummy_camera: bool = False,
        plc_status: str = "Disconnected",
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.plc = plc
        self._use_dummy_camera = use_dummy_camera
        self._manual_images: List[np.ndarray] = []
        self._manual_index = 0
        self.setWindowTitle("Bearing Inspection")
        self.resize(1400, 900)

        self._init_ui()
        self._init_workers()

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        content_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content_layout)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        content_layout.addWidget(self.image_label, stretch=2)

        right_panel = QtWidgets.QVBoxLayout()
        content_layout.addLayout(right_panel, stretch=1)

        self.result_table = QtWidgets.QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Index", "Score", "Status"])
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        right_panel.addWidget(self.result_table)

        info_group = QtWidgets.QGroupBox("Summary")
        form = QtWidgets.QFormLayout(info_group)
        self.model_label = QtWidgets.QLabel(self.config.models.anomaly.path)
        self.threshold_label = QtWidgets.QLabel(f"{self.config.models.anomaly.threshold:.2f}")
        self.ng_label = QtWidgets.QLabel("0")
        self.inference_label = QtWidgets.QLabel("0 ms")
        form.addRow("Model", self.model_label)
        form.addRow("Threshold", self.threshold_label)
        form.addRow("NG total", self.ng_label)
        form.addRow("Inference", self.inference_label)
        right_panel.addWidget(info_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.load_model_button = QtWidgets.QPushButton("Load model")
        self.open_image_button = QtWidgets.QPushButton("Open image")
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.open_image_button)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        right_panel.addLayout(button_layout)

        main_layout.addStretch(1)

        options_menu = self.menuBar().addMenu("File")
        self.save_images_action = QtWidgets.QAction("Save processed images", self)
        self.save_images_action.setCheckable(True)
        self.save_images_action.setChecked(self.config.io.save_images)
        options_menu.addAction(self.save_images_action)

        self.enable_yolo_action = QtWidgets.QAction("Enable YOLO", self)
        self.enable_yolo_action.setCheckable(True)
        self.enable_yolo_action.setChecked(self.config.models.yolo.enabled)
        options_menu.addAction(self.enable_yolo_action)

        select_output_action = QtWidgets.QAction("Select output folder", self)
        options_menu.addAction(select_output_action)

        self.status_camera = QtWidgets.QLabel("Camera: Idle")
        self.status_plc = QtWidgets.QLabel(f"PLC: {plc_status}")
        self.statusBar().addWidget(self.status_camera)
        self.statusBar().addPermanentWidget(self.status_plc)

        # Connect UI actions
        self.capture_button.clicked.connect(self.trigger_manual.emit)
        self.load_model_button.clicked.connect(self._select_model)
        self.open_image_button.clicked.connect(self._open_image)
        self.prev_button.clicked.connect(self._previous_image)
        self.next_button.clicked.connect(self._next_image)
        select_output_action.triggered.connect(self._select_output_dir)
        self.save_images_action.toggled.connect(self._toggle_save_images)
        self.enable_yolo_action.toggled.connect(self._toggle_yolo)

    def _init_workers(self) -> None:
        self.inspection_thread = QtCore.QThread(self)
        self.worker = InspectionWorker(self.config, self.plc, use_dummy_camera=self._use_dummy_camera)
        self.worker.moveToThread(self.inspection_thread)
        self.inspection_thread.start()

        self.save_thread = QtCore.QThread(self)
        self.save_worker = SaveWorker(self.config)
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.start()

        self.trigger_worker = PlcTriggerWorker(self.plc)
        self.trigger_worker.triggered.connect(self._handle_trigger)
        self.trigger_worker.start()

        self.trigger_manual.connect(self._handle_trigger)
        self.worker.cycle_started.connect(lambda: self.status_camera.setText("Camera: Busy"))
        self.worker.cycle_completed.connect(self._update_ui)
        self.worker.cycle_failed.connect(self._handle_failure)
        self.save_worker.finished.connect(lambda path: self.statusBar().showMessage(f"Saved results to {path}", 3000))
        self.save_worker.failed.connect(lambda msg: self.statusBar().showMessage(f"Save failed: {msg}", 5000))
        self.status_camera.setText("Camera: Ready")

    @QtCore.pyqtSlot()
    def _handle_trigger(self) -> None:
        QtCore.QMetaObject.invokeMethod(self.worker, "run_cycle", QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot(InspectionResult)
    def _update_ui(self, result: InspectionResult) -> None:
        pixmap = QtGui.QPixmap.fromImage(numpy_to_qimage(result.overlay_image))
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        self.result_table.setRowCount(len(result.patches))
        for row, (patch, score, status) in enumerate(zip(result.patches, result.anomaly_scores, result.statuses)):
            self.result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(patch.index)))
            self.result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{score:.3f}"))
            self.result_table.setItem(row, 2, QtWidgets.QTableWidgetItem(status))
        self.ng_label.setText(str(result.ng_total))
        self.inference_label.setText(f"{result.anomaly_inference_ms:.1f} ms")
        self.status_camera.setText("Camera: Ready")

        if self.save_images_action.isChecked():
            QtCore.QMetaObject.invokeMethod(
                self.save_worker,
                "save",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, result),
            )

    @QtCore.pyqtSlot(str)
    def _handle_failure(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Inspection failed", message)
        self.status_camera.setText("Camera: Error")

    def _select_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.config.io.output_dir)
        if directory:
            self.config.io.output_dir = directory

    def _toggle_save_images(self, enabled: bool) -> None:
        self.config.io.save_images = enabled

    def _toggle_yolo(self, enabled: bool) -> None:
        self.config.models.yolo.enabled = enabled

    def _select_model(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ONNX model", str(self.config.models.anomaly.path), "ONNX files (*.onnx);;All files (*)")
        if file_path:
            self.config.models.anomaly.path = file_path
            self.model_label.setText(file_path)
            QtCore.QMetaObject.invokeMethod(
                self.worker,
                "reload_anomaly_model",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, file_path),
            )

    def _open_image(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open images", "", "Images (*.png *.jpg *.bmp)")
        if paths:
            images = []
            for path in paths:
                if not path:
                    continue
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
            if not images:
                QtWidgets.QMessageBox.warning(self, "Open image", "No readable images selected")
                return
            self._manual_images = images
            self._manual_index = 0
            self._show_manual_image()

    def _show_manual_image(self) -> None:
        if not getattr(self, "_manual_images", None):
            return
        image = self._manual_images[self._manual_index]
        pixmap = QtGui.QPixmap.fromImage(numpy_to_qimage(image))
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _next_image(self) -> None:
        if getattr(self, "_manual_images", None):
            self._manual_index = (self._manual_index + 1) % len(self._manual_images)
            self._show_manual_image()

    def _previous_image(self) -> None:
        if getattr(self, "_manual_images", None):
            self._manual_index = (self._manual_index - 1) % len(self._manual_images)
            self._show_manual_image()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI cleanup
        try:
            QtCore.QMetaObject.invokeMethod(self.worker, "shutdown", QtCore.Qt.BlockingQueuedConnection)
            self.trigger_worker.stop()
            self.inspection_thread.quit()
            self.inspection_thread.wait(2000)
            self.save_thread.quit()
            self.save_thread.wait(2000)
            self.plc.close()
        finally:
            super().closeEvent(event)

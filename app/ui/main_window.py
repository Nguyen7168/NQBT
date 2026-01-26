"""PyQt5 main window controlling the inspection workflow."""
from __future__ import annotations

import logging
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import AppConfig
from app.inspection.plc_client import PlcController, PLCError
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
        # Store initial PLC status string for UI
        self._plc_status = plc_status
        self._manual_images: List[np.ndarray] = []
        self._manual_index = 0
        self.setWindowTitle("Bearing Inspection")
        self.resize(1400, 900)

        self._init_ui()
        self._init_workers()
        self._show_startup_health()

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

        right_tabs = QtWidgets.QTabWidget()
        content_layout.addWidget(right_tabs, stretch=1)

        inspection_tab = QtWidgets.QWidget()
        right_tabs.addTab(inspection_tab, "Inspection")
        inspection_layout = QtWidgets.QVBoxLayout(inspection_tab)

        self.result_table = QtWidgets.QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Index", "Score", "Status"])
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        inspection_layout.addWidget(self.result_table)

        info_group = QtWidgets.QGroupBox("Summary")
        form = QtWidgets.QFormLayout(info_group)
        self.model_label = QtWidgets.QLabel(self._current_model_path())
        self.threshold_label = QtWidgets.QLabel(f"{self._current_threshold():.2f}")
        self.ng_label = QtWidgets.QLabel("0")
        self.inference_label = QtWidgets.QLabel("0 ms")
        form.addRow("Model", self.model_label)
        form.addRow("Threshold", self.threshold_label)
        form.addRow("NG total", self.ng_label)
        form.addRow("Inference", self.inference_label)
        inspection_layout.addWidget(info_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.load_model_button = QtWidgets.QPushButton("Load model")
        self.open_image_button = QtWidgets.QPushButton("Open image")
        self.run_anomaly_button = QtWidgets.QPushButton("Run anomaly")
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.open_image_button)
        button_layout.addWidget(self.run_anomaly_button)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        inspection_layout.addLayout(button_layout)

        plc_tab = QtWidgets.QWidget()
        right_tabs.addTab(plc_tab, "PLC Monitor")
        plc_layout = QtWidgets.QVBoxLayout(plc_tab)

        self.plc_monitor_toggle = QtWidgets.QCheckBox("Enable PLC Monitor")
        self.plc_monitor_status = QtWidgets.QLabel("Monitor: Off")
        plc_layout.addWidget(self.plc_monitor_toggle)
        plc_layout.addWidget(self.plc_monitor_status)

        tx_group = QtWidgets.QGroupBox("TX (App → PLC)")
        tx_form = QtWidgets.QFormLayout(tx_group)
        self.tx_busy_label = QtWidgets.QLabel("-")
        self.tx_done_label = QtWidgets.QLabel("-")
        self.tx_error_label = QtWidgets.QLabel("-")
        self.tx_ready_label = QtWidgets.QLabel("-")
        tx_form.addRow("Busy", self.tx_busy_label)
        tx_form.addRow("Done", self.tx_done_label)
        tx_form.addRow("Error", self.tx_error_label)
        tx_form.addRow("Ready", self.tx_ready_label)
        plc_layout.addWidget(tx_group)

        rx_group = QtWidgets.QGroupBox("RX (PLC → App)")
        rx_form = QtWidgets.QFormLayout(rx_group)
        self.rx_trigger_label = QtWidgets.QLabel("-")
        self.rx_ack_label = QtWidgets.QLabel("-")
        rx_form.addRow("Trigger", self.rx_trigger_label)
        rx_form.addRow("ACK", self.rx_ack_label)
        plc_layout.addWidget(rx_group)

        self.plc_results_table = QtWidgets.QTableWidget(self.config.layout.count, 2)
        self.plc_results_table.setHorizontalHeaderLabels(["Index", "Result"])
        self.plc_results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for row in range(self.config.layout.count):
            self.plc_results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
            self.plc_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
        plc_layout.addWidget(self.plc_results_table)

        self.plc_monitor_error = QtWidgets.QLabel("")
        self.plc_monitor_error.setStyleSheet("color: red;")
        plc_layout.addWidget(self.plc_monitor_error)

        main_layout.addStretch(1)

        options_menu = self.menuBar().addMenu("File")
        self.save_images_action = QtWidgets.QAction("Save processed images", self)
        self.save_images_action.setCheckable(True)
        self.save_images_action.setChecked(self.config.io.save_images)
        options_menu.addAction(self.save_images_action)

        self.save_heatmap_action = QtWidgets.QAction("Save heatmaps (per patch)", self)
        self.save_heatmap_action.setCheckable(True)
        self.save_heatmap_action.setChecked(getattr(self.config.io, "save_heatmap", False))
        options_menu.addAction(self.save_heatmap_action)

        self.save_binary_action = QtWidgets.QAction("Save binary masks (per patch)", self)
        self.save_binary_action.setCheckable(True)
        self.save_binary_action.setChecked(getattr(self.config.io, "save_binary", False))
        options_menu.addAction(self.save_binary_action)

        self.enable_yolo_action = QtWidgets.QAction("Enable YOLO", self)
        self.enable_yolo_action.setCheckable(True)
        self.enable_yolo_action.setChecked(self.config.models.yolo.enabled)
        options_menu.addAction(self.enable_yolo_action)

        select_output_action = QtWidgets.QAction("Select output folder", self)
        options_menu.addAction(select_output_action)

        self.status_camera = QtWidgets.QLabel("Camera: Idle")
        self.status_plc = QtWidgets.QLabel(f"PLC: {self._plc_status}")
        self.statusBar().addWidget(self.status_camera)
        self.statusBar().addPermanentWidget(self.status_plc)

        # Connect UI actions
        self.capture_button.clicked.connect(self.trigger_manual.emit)
        self.load_model_button.clicked.connect(self._select_model)
        self.open_image_button.clicked.connect(self._open_image)
        self.run_anomaly_button.clicked.connect(self._run_anomaly_on_manual)
        self.prev_button.clicked.connect(self._previous_image)
        self.next_button.clicked.connect(self._next_image)
        select_output_action.triggered.connect(self._select_output_dir)
        self.save_images_action.toggled.connect(self._toggle_save_images)
        self.enable_yolo_action.toggled.connect(self._toggle_yolo)
        self.save_heatmap_action.toggled.connect(self._toggle_save_heatmap)
        self.save_binary_action.toggled.connect(self._toggle_save_binary)
        self.plc_monitor_toggle.toggled.connect(self._toggle_plc_monitor)
        self.run_anomaly_button.setEnabled(False)

        self.plc_monitor_timer = QtCore.QTimer(self)
        self.plc_monitor_timer.setInterval(200)
        self.plc_monitor_timer.timeout.connect(self._poll_plc_monitor)

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
        self.worker.camera_ready.connect(lambda: self.status_camera.setText("Camera: Ready"))
        self.worker.camera_failed.connect(lambda msg: (self.status_camera.setText("Camera: Error"), QtWidgets.QMessageBox.critical(self, "Camera", msg)))
        self.save_worker.finished.connect(lambda path: self.statusBar().showMessage(f"Saved results to {path}", 3000))
        self.save_worker.failed.connect(lambda msg: self.statusBar().showMessage(f"Save failed: {msg}", 5000))
        # Keep initial status as Idle until a successful cycle completes
        self.status_camera.setText("Camera: Idle")

    def _show_startup_health(self) -> None:
        messages: List[str] = []

        # PLC status coloring
        if "Disconnected" in self.status_plc.text():
            self.status_plc.setStyleSheet("color: red;")

        # Camera status on startup
        if self._use_dummy_camera:
            self.status_camera.setText("Camera: Dummy")
        else:
            # Best-effort check: if pypylon is missing, we can report immediately.
            try:
                from app.inspection.camera import pylon  # type: ignore
            except Exception:
                pylon = None  # type: ignore
            if pylon is None:
                self.status_camera.setText("Camera: Not available (pypylon missing)")
                messages.append("Camera not available: pypylon is not installed")
            else:
                # Try to proactively connect camera via worker thread
                QtCore.QMetaObject.invokeMethod(self.worker, "connect_camera", QtCore.Qt.QueuedConnection)
                self.status_camera.setText("Camera: Connecting...")

        # Model availability
        model_path = Path(self._current_model_path())
        if not model_path.exists():
            self.model_label.setText(f"{model_path} (missing)")
            self.model_label.setStyleSheet("color: red;")
            messages.append(f"Anomaly model not found: {model_path}")

        if messages:
            QtWidgets.QMessageBox.warning(self, "Startup issues", "\n".join(messages))
            self.statusBar().showMessage("; ".join(messages), 5000)

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
        if result.detected_circles is not None and result.expected_circles is not None:
            self.status_camera.setText(
                f"Camera: Ready (circles {result.detected_circles}/{result.expected_circles})"
            )
        else:
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

    def _toggle_plc_monitor(self, enabled: bool) -> None:
        if enabled:
            self.plc_monitor_status.setText("Monitor: On")
            self.plc_monitor_error.setText("")
            self.plc_monitor_timer.start()
            self._poll_plc_monitor()
        else:
            self.plc_monitor_timer.stop()
            self.plc_monitor_status.setText("Monitor: Off")
            self.plc_monitor_error.setText("")

    def _poll_plc_monitor(self) -> None:
        try:
            trigger = self.plc.client.read_bit(self.plc.config.addr.trigger)
            ack = self.plc.client.read_bit(self.plc.config.addr.ack)
            self.rx_trigger_label.setText("ON" if trigger else "OFF")
            self.rx_ack_label.setText("ON" if ack else "OFF")
            self.tx_busy_label.setText("ON" if self.plc.state.busy else "OFF")
            self.tx_done_label.setText("ON" if self.plc.state.done else "OFF")
            self.tx_error_label.setText("ON" if self.plc.state.error else "OFF")
            self.tx_ready_label.setText("ON" if self.plc.state.ready else "OFF")
            results = self.plc.state.last_results
            total = self.config.layout.count
            for row in range(total):
                if results is None or row >= len(results):
                    text = "-"
                else:
                    text = "OK" if results[row] else "NG"
                self.plc_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(text))
            self.plc_monitor_error.setText("")
        except PLCError as exc:
            self.plc_monitor_error.setText(str(exc))
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self.plc_monitor_error.setText(str(exc))

    def _select_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.config.io.output_dir)
        if directory:
            self.config.io.output_dir = directory

    def _toggle_save_images(self, enabled: bool) -> None:
        self.config.io.save_images = enabled

    def _toggle_yolo(self, enabled: bool) -> None:
        self.config.models.yolo.enabled = enabled

    def _toggle_save_heatmap(self, enabled: bool) -> None:
        self.config.io.save_heatmap = enabled

    def _toggle_save_binary(self, enabled: bool) -> None:
        self.config.io.save_binary = enabled

    def _select_model(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ONNX model", str(self._current_model_path()), "ONNX files (*.onnx);;All files (*)"
        )
        if file_path:
            # Update config branch according to current algo
            if (self.config.models.algo or "INP").upper() == "GLASS":
                self.config.models.glass.path = file_path
            else:
                self.config.models.inp.path = file_path
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
            self.run_anomaly_button.setEnabled(True)

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

    def _run_anomaly_on_manual(self) -> None:
        if not getattr(self, "_manual_images", None):
            QtWidgets.QMessageBox.information(self, "Run anomaly", "Please open an image first.")
            return
        image = self._manual_images[self._manual_index]
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "run_on_image",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(object, image),
        )

    def _current_threshold(self) -> float:
        algo = (self.config.models.algo or "INP").upper()
        return (
            float(self.config.models.glass.glass_threshold)
            if algo == "GLASS"
            else float(self.config.models.inp.inp_threshold)
        )

    def _current_model_path(self) -> str:
        algo = (self.config.models.algo or "INP").upper()
        return self.config.models.glass.path if algo == "GLASS" else self.config.models.inp.path

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI cleanup
        try:
            if hasattr(self, "plc_monitor_timer"):
                self.plc_monitor_timer.stop()
            QtCore.QMetaObject.invokeMethod(self.worker, "shutdown", QtCore.Qt.BlockingQueuedConnection)
            self.trigger_worker.stop()
            self.inspection_thread.quit()
            self.inspection_thread.wait(2000)
            self.save_thread.quit()
            self.save_thread.wait(2000)
            self.plc.close()
        finally:
            super().closeEvent(event)

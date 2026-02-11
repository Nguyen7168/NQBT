"""PyQt5 main window controlling the inspection workflow."""
from __future__ import annotations

import logging
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import AppConfig, GlassRecipeConfig
from app.inspection.plc_client import PlcController, PLCError
from app.inspection.workers import (
    InspectionResult,
    InspectionWorker,
    PlcModelSelectWorker,
    PlcTriggerWorker,
    SaveWorker,
)
from app.utils import numpy_to_qimage

LOGGER = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    trigger_manual = QtCore.pyqtSignal()
    _table_row_height = 24

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
        self._recipe_by_code: dict[int, GlassRecipeConfig] = {
            int(recipe.code): recipe for recipe in getattr(self.config.models, "glass_recipes", [])
        }
        self._current_recipe_code: Optional[int] = getattr(self.config.models, "active_recipe_code", None)
        self._pending_recipe_code: Optional[int] = None
        self._recipe_switch_in_progress = False
        self._cycle_request_inflight = False
        self._last_status_message_ts: dict[str, float] = {}
        self._display_image: Optional[QtGui.QImage] = None
        self.setWindowTitle("Bearing Inspection")
        self._apply_window_geometry()

        self._init_ui()
        self._init_workers()
        self._show_startup_health()

    def _apply_window_geometry(self) -> None:
        window_cfg = getattr(self.config, "window", None)
        if window_cfg and window_cfg.width and window_cfg.height:
            self.resize(window_cfg.width, window_cfg.height)
        else:
            self.resize(1400, 900)
        if window_cfg and window_cfg.x is not None and window_cfg.y is not None:
            self.move(window_cfg.x, window_cfg.y)

    def _init_ui(self) -> None:
        window_cfg = getattr(self.config, "window", None)
        configured_width = window_cfg.width if window_cfg and window_cfg.width else None
        configured_height = window_cfg.height if window_cfg and window_cfg.height else None

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        right_tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(right_tabs)

        inspection_tab = QtWidgets.QWidget()
        right_tabs.addTab(inspection_tab, "Inspection")
        inspection_layout = QtWidgets.QVBoxLayout(inspection_tab)

        summary_bar = QtWidgets.QHBoxLayout()

        overall_widget = QtWidgets.QWidget()
        overall_layout = QtWidgets.QVBoxLayout(overall_widget)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        self.overall_label = QtWidgets.QLabel("-")
        self.overall_label.setMinimumWidth(170)
        self.overall_label.setAlignment(QtCore.Qt.AlignCenter)
        self._set_overall_status(None)
        overall_layout.addWidget(self.overall_label)
        summary_bar.addWidget(overall_widget, stretch=1)

        summary_widget = QtWidgets.QWidget()
        summary_form = QtWidgets.QFormLayout(summary_widget)
        self.ok_label = QtWidgets.QLabel("0")
        self.ng_label = QtWidgets.QLabel("0")
        summary_form.addRow("OK total", self.ok_label)
        summary_form.addRow("NG total", self.ng_label)
        summary_bar.addWidget(summary_widget, stretch=1)

        model_widget = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_widget)
        model_title = QtWidgets.QLabel("Model")
        self.model_label = QtWidgets.QLabel(self._current_model_display_name())
        speed_title = QtWidgets.QLabel("Speed")
        self.speed_label = QtWidgets.QLabel("0.0 ms")
        model_layout.addWidget(model_title)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(speed_title)
        model_layout.addWidget(self.speed_label)

        controls_layout = QtWidgets.QGridLayout()
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.load_model_button = QtWidgets.QPushButton("Load model")
        self.open_image_button = QtWidgets.QPushButton("Open image")
        self.run_anomaly_button = QtWidgets.QPushButton("Run anomaly")
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        controls_layout.addWidget(self.capture_button, 0, 0)
        controls_layout.addWidget(self.load_model_button, 0, 1)
        controls_layout.addWidget(self.open_image_button, 0, 2)
        controls_layout.addWidget(self.run_anomaly_button, 1, 0)
        controls_layout.addWidget(self.prev_button, 1, 1)
        controls_layout.addWidget(self.next_button, 1, 2)
        model_layout.addLayout(controls_layout)

        summary_bar.addWidget(model_widget, stretch=2)
        inspection_layout.addLayout(summary_bar)

        self.image_label = QtWidgets.QLabel()
        min_image_width = 960
        min_image_height = 540
        if configured_width:
            min_image_width = min(min_image_width, max(320, int(configured_width * 0.6)))
        if configured_height:
            min_image_height = min(min_image_height, max(240, int(configured_height * 0.35)))
        self.image_label.setMinimumSize(min_image_width, min_image_height)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        inspection_layout.addWidget(self.image_label, stretch=7)

        self.result_table = QtWidgets.QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Index", "Score", "Status"])
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.result_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.result_table.setWordWrap(False)
        self.result_table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._set_table_row_heights(self.result_table)
        header_height = self.result_table.horizontalHeader().height()
        table_height = header_height + (self._table_row_height * self.config.layout.count) + 4
        self.result_table.setFixedHeight(table_height)
        table_scroll = QtWidgets.QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        table_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        table_scroll.setFixedHeight(table_height + 2)
        table_scroll.setWidget(self.result_table)
        inspection_layout.addWidget(table_scroll, stretch=0)

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
        self.tx_run_label = QtWidgets.QLabel("-")
        tx_form.addRow("Busy", self.tx_busy_label)
        tx_form.addRow("Done", self.tx_done_label)
        tx_form.addRow("Error", self.tx_error_label)
        tx_form.addRow("Ready", self.tx_ready_label)
        tx_form.addRow("Run", self.tx_run_label)
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
        self.plc_results_table.verticalHeader().setVisible(False)
        self.plc_results_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.plc_results_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.plc_results_table.setWordWrap(False)
        for row in range(self.config.layout.count):
            self.plc_results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
            self.plc_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
        self._set_table_row_heights(self.plc_results_table)
        plc_layout.addWidget(self.plc_results_table)

        self.plc_monitor_error = QtWidgets.QLabel("")
        self.plc_monitor_error.setStyleSheet("color: red;")
        plc_layout.addWidget(self.plc_monitor_error)

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

        self.save_crops_action = QtWidgets.QAction("Save crop ROI images", self)
        self.save_crops_action.setCheckable(True)
        self.save_crops_action.setChecked(getattr(self.config.io, "save_crops", False))
        options_menu.addAction(self.save_crops_action)

        self.enable_yolo_action = QtWidgets.QAction("Enable YOLO", self)
        self.enable_yolo_action.setCheckable(True)
        self.enable_yolo_action.setChecked(self.config.models.yolo.enabled)
        options_menu.addAction(self.enable_yolo_action)

        select_output_action = QtWidgets.QAction("Select output folder", self)
        options_menu.addAction(select_output_action)

        self.status_camera = QtWidgets.QLabel("Camera: Idle")
        self.status_plc = QtWidgets.QLabel(f"PLC: {self._plc_status}")
        status_min_width = 260
        if configured_width and configured_width <= 800:
            status_min_width = 140
        self.status_camera.setMinimumWidth(status_min_width)
        self.status_plc.setMinimumWidth(status_min_width)
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
        self.save_crops_action.toggled.connect(self._toggle_save_crops)
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

        self.trigger_worker = None
        if self.config.plc.enable_plc_trigger:
            poll_interval = max(self.config.plc.trigger_poll_interval_ms, 1) / 1000.0
            self.trigger_worker = PlcTriggerWorker(
                self.plc,
                poll_interval=poll_interval,
                min_interval_ms=self.config.plc.trigger_min_interval_ms,
                high_stable_ms=self.config.plc.trigger_high_stable_ms,
                low_stable_ms=self.config.plc.trigger_low_stable_ms,
                cooldown_ms=self.config.plc.trigger_cooldown_ms,
            )
            self.trigger_worker.triggered.connect(self._handle_trigger)
            self.trigger_worker.start()

        self.model_select_worker = None
        model_select_addr = getattr(self.config.plc.addr, "model_select_word", None)
        if (self.config.models.algo or "INP").upper() == "GLASS" and model_select_addr and self._recipe_by_code:
            model_poll_interval = max(self.config.plc.model_poll_interval_ms, 10) / 1000.0
            self.model_select_worker = PlcModelSelectWorker(
                self.plc,
                model_select_addr,
                poll_interval=model_poll_interval,
                stable_ms=self.config.plc.model_stable_ms,
                parent=self,
            )
            self.model_select_worker.model_code_changed.connect(self._on_plc_model_code_changed)
            self.model_select_worker.start()

        self.trigger_manual.connect(self._handle_trigger)
        self.worker.cycle_started.connect(self._on_cycle_started)
        self.worker.cycle_completed.connect(self._update_ui)
        self.worker.cycle_failed.connect(self._handle_failure)
        self.worker.model_reloaded.connect(self._on_model_reloaded)
        self.worker.model_reload_failed.connect(self._on_model_reload_failed)
        self.worker.camera_ready.connect(self._on_camera_ready)
        self.worker.camera_failed.connect(self._on_camera_failed)
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
            self.model_label.setText(f"{self._current_model_display_name()} (missing)")
            self.model_label.setStyleSheet("color: red;")
            messages.append(f"Anomaly model not found: {model_path}")

        if self._current_recipe_code is not None:
            recipe = self._recipe_by_code.get(int(self._current_recipe_code))
            if recipe and recipe.name:
                self.model_label.setText(recipe.name)

        model_current_addr = getattr(self.config.plc.addr, "model_current_word", None)
        if model_current_addr and self._current_recipe_code is not None:
            try:
                self.plc.write_word(model_current_addr, int(self._current_recipe_code))
            except PLCError as exc:
                messages.append(f"Cannot write current recipe code at startup: {exc}")

        if messages:
            QtWidgets.QMessageBox.warning(self, "Startup issues", "\n".join(messages))
            self.statusBar().showMessage("; ".join(messages), 5000)

    @QtCore.pyqtSlot()
    def _on_camera_ready(self) -> None:
        self.status_camera.setText("Camera: Ready")
        try:
            self.plc.set_run(True)
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to set RUN ON: {exc}", 5000)

    @QtCore.pyqtSlot(str)
    def _on_camera_failed(self, message: str) -> None:
        self.status_camera.setText("Camera: Error")
        try:
            self.plc.set_run(False)
        except PLCError:
            pass
        QtWidgets.QMessageBox.critical(self, "Camera", message)

    @QtCore.pyqtSlot()
    def _handle_trigger(self) -> None:
        if self._recipe_switch_in_progress:
            self._show_rate_limited_status("switching_ignore", "Model is switching, trigger ignored", 1000)
            return
        if self._cycle_request_inflight:
            self._show_rate_limited_status("cycle_busy_ignore", "Inspection cycle already in progress, trigger ignored", 1000)
            return
        self._cycle_request_inflight = True
        QtCore.QMetaObject.invokeMethod(self.worker, "run_cycle", QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def _on_cycle_started(self) -> None:
        self.status_camera.setText("Camera: Busy")

    @QtCore.pyqtSlot(InspectionResult)
    def _update_ui(self, result: InspectionResult) -> None:
        self._cycle_request_inflight = False
        self._set_display_image(numpy_to_qimage(result.overlay_image))

        self.result_table.setRowCount(len(result.patches))
        self._set_table_row_heights(self.result_table)
        for row, (patch, score, status) in enumerate(zip(result.patches, result.anomaly_scores, result.statuses)):
            self.result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(patch.index)))
            self.result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{score:.3f}"))
            self.result_table.setItem(row, 2, QtWidgets.QTableWidgetItem(status))
        ok_total = sum(1 for status in result.statuses if status == "OK")
        self.ok_label.setText(str(ok_total))
        self.ng_label.setText(str(result.ng_total))
        expected_total = result.expected_circles if result.expected_circles is not None else self.config.layout.count
        overall = "OK" if expected_total > 0 and len(result.statuses) == expected_total and result.ng_total == 0 else "NG"
        self._set_overall_status(overall)
        self.speed_label.setText(f"{result.anomaly_inference_ms:.1f} ms")
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

        if self._pending_recipe_code is not None and not self._recipe_switch_in_progress and not self.plc.state.busy:
            self._apply_recipe_code(self._pending_recipe_code)

    @QtCore.pyqtSlot(str)
    def _handle_failure(self, message: str) -> None:
        self._cycle_request_inflight = False
        QtWidgets.QMessageBox.critical(self, "Inspection failed", message)
        self.status_camera.setText("Camera: Error")

    def _show_rate_limited_status(self, key: str, message: str, min_interval_ms: int = 1000) -> None:
        now = QtCore.QDateTime.currentMSecsSinceEpoch() / 1000.0
        last = self._last_status_message_ts.get(key, 0.0)
        if (now - last) * 1000.0 >= max(0, min_interval_ms):
            self._last_status_message_ts[key] = now
            self.statusBar().showMessage(message, 2000)

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
            self.tx_run_label.setText("ON" if self.plc.state.run else "OFF")
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

    def _set_overall_status(self, status: Optional[str]) -> None:
        if status == "OK":
            self.overall_label.setText("OK")
            self.overall_label.setStyleSheet(
                "background-color: #2f2f2f; color: #00E5EE; font-size: 54px; font-weight: 700; border: 1px solid #555; padding: 4px;"
            )
        elif status == "NG":
            self.overall_label.setText("NG")
            self.overall_label.setStyleSheet(
                "background-color: #2f2f2f; color: #FF2A2A; font-size: 54px; font-weight: 700; border: 1px solid #555; padding: 4px;"
            )
        else:
            self.overall_label.setText("-")
            self.overall_label.setStyleSheet(
                "background-color: #2f2f2f; color: #CCCCCC; font-size: 42px; font-weight: 600; border: 1px solid #555; padding: 4px;"
            )

    def _set_table_row_heights(self, table: QtWidgets.QTableWidget) -> None:
        for row in range(table.rowCount()):
            table.setRowHeight(row, self._table_row_height)

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

    def _toggle_save_crops(self, enabled: bool) -> None:
        self.config.io.save_crops = enabled

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
            self.model_label.setText(self._display_name_from_path(file_path))
            self.model_label.setStyleSheet("")
            QtCore.QMetaObject.invokeMethod(
                self.worker,
                "reload_anomaly_model",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, file_path),
            )


    @QtCore.pyqtSlot(int)
    def _on_plc_model_code_changed(self, model_code: int) -> None:
        # Convention: 0 means "no selection / reset request" from PLC.
        # Keep current model, do not raise error, and reflect current code to PLC.
        if int(model_code) == 0:
            self._pending_recipe_code = None
            model_current_addr = getattr(self.config.plc.addr, "model_current_word", None)
            if model_current_addr and self._current_recipe_code is not None:
                try:
                    self.plc.write_word(model_current_addr, int(self._current_recipe_code))
                    self.plc.set_error(False)
                except PLCError as exc:
                    self.statusBar().showMessage(f"Failed to keep current model code: {exc}", 5000)
            self.statusBar().showMessage("PLC model select reset (0): keep current model", 3000)
            return

        if model_code == self._current_recipe_code:
            return
        if self.plc.state.busy or self._recipe_switch_in_progress:
            self._pending_recipe_code = model_code
            self.statusBar().showMessage(f"Queued model code {model_code} until cycle complete", 3000)
            return
        self._apply_recipe_code(model_code)

    def _apply_recipe_code(self, model_code: int) -> None:
        recipe = self._recipe_by_code.get(int(model_code))
        if recipe is None:
            self.statusBar().showMessage(f"Unknown model code from PLC: {model_code}", 5000)
            try:
                self.plc.set_error(True)
            except PLCError:
                pass
            return
        threshold = (
            float(recipe.glass_threshold)
            if recipe.glass_threshold is not None
            else float(self.config.models.glass.glass_threshold)
        )
        self._pending_recipe_code = int(model_code)
        self._recipe_switch_in_progress = True
        try:
            self.plc.set_busy(True)
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to set BUSY for model switch: {exc}", 5000)
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "reload_anomaly_model_with_threshold",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, recipe.path),
            QtCore.Q_ARG(float, threshold),
        )

    @QtCore.pyqtSlot(str, float)
    def _on_model_reloaded(self, model_path: str, threshold: float) -> None:
        if self._pending_recipe_code is not None:
            self._current_recipe_code = self._pending_recipe_code
            self.config.models.active_recipe_code = self._current_recipe_code
        self._recipe_switch_in_progress = False
        recipe_name = self._display_name_from_path(model_path)
        if self._current_recipe_code is not None:
            recipe = self._recipe_by_code.get(int(self._current_recipe_code))
            if recipe and recipe.name:
                recipe_name = recipe.name
        self.model_label.setText(recipe_name)
        self.model_label.setStyleSheet("")
        self.statusBar().showMessage(
            f"Model switched: {recipe_name} (th={threshold:.3f})",
            3000,
        )
        model_current_addr = getattr(self.config.plc.addr, "model_current_word", None)
        if model_current_addr and self._current_recipe_code is not None:
            try:
                self.plc.write_word(model_current_addr, int(self._current_recipe_code))
                self.plc.set_error(False)
            except PLCError as exc:
                self.statusBar().showMessage(f"Failed to write current model code: {exc}", 5000)
        try:
            self.plc.set_busy(False)
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to clear BUSY after model switch: {exc}", 5000)
        self._pending_recipe_code = None

    @QtCore.pyqtSlot(str)
    def _on_model_reload_failed(self, message: str) -> None:
        self._recipe_switch_in_progress = False
        self.statusBar().showMessage(f"Model reload failed: {message}", 5000)
        try:
            self.plc.set_error(True)
        except PLCError:
            pass
        try:
            self.plc.set_busy(False)
        except PLCError:
            pass


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
        self._set_display_image(numpy_to_qimage(image))

    def _set_display_image(self, image: QtGui.QImage) -> None:
        self._display_image = image
        self._redraw_display_image()

    def _redraw_display_image(self) -> None:
        if self._display_image is None:
            return
        if self.image_label.width() <= 0 or self.image_label.height() <= 0:
            return
        pixmap = QtGui.QPixmap.fromImage(self._display_image)
        scaled = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

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

    def _current_model_display_name(self) -> str:
        return self._display_name_from_path(self._current_model_path())

    def _display_name_from_path(self, model_path: str) -> str:
        if not model_path:
            return "-"
        return Path(model_path).stem

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI cleanup
        try:
            if hasattr(self, "plc_monitor_timer"):
                self.plc_monitor_timer.stop()
            QtCore.QMetaObject.invokeMethod(self.worker, "shutdown", QtCore.Qt.BlockingQueuedConnection)
            if self.trigger_worker is not None:
                self.trigger_worker.stop()
            if getattr(self, "model_select_worker", None) is not None:
                self.model_select_worker.stop()
            self.inspection_thread.quit()
            self.inspection_thread.wait(2000)
            self.save_thread.quit()
            self.save_thread.wait(2000)
            try:
                self.plc.set_run(False)
            except Exception:
                pass
            self.plc.close()
        finally:
            super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._redraw_display_image()


"""PyQt5 main window controlling the inspection workflow."""
from __future__ import annotations

import logging
import re
import time
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np

from app.config_loader import AppConfig, RecipeConfig
from app.inspection.plc_client import PlcController, PLCError
from app.inspection.workers import (
    InspectionResult,
    InspectionWorker,
    OperatingMode,
    PlcModeSelectWorker,
    PlcModelSelectWorker,
    PlcTriggerWorker,
    SaveWorker,
)
from PyQt5 import QtCore, QtGui, QtWidgets
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
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        self.plc = plc
        self._use_dummy_camera = use_dummy_camera
        # Store initial PLC status string for UI
        self._plc_status = plc_status
        self._manual_images: List[np.ndarray] = []
        self._manual_index = 0
        self._recipe_by_code: dict[int, RecipeConfig] = {
            int(recipe.code): recipe for recipe in getattr(self.config.models, "recipes", [])
        }
        self._current_recipe_code: Optional[int] = getattr(self.config.models, "active_recipe_code", None)
        self._pending_recipe_code: Optional[int] = None
        self._recipe_switch_in_progress = False
        self._cycle_request_inflight = False
        self._current_mode = OperatingMode.RUN
        self._requested_mode = OperatingMode.RUN
        self._last_status_message_ts: dict[str, float] = {}
        self._model_mismatch_streak = 0
        self._last_model_sync_write_ts = 0.0
        self._display_image: Optional[QtGui.QImage] = None
        ui_cfg = getattr(self.config, "ui", None)
        self._show_notch_column = bool(getattr(ui_cfg, "show_notch_column", True))
        configured_title = (getattr(self.config, "app_title", None) or "").strip()
        self.setWindowTitle(configured_title or "Bearing Inspection")
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

    def _format_addr_label(self, name: str, address: Optional[str]) -> str:
        return f"{name} ({address})" if address else f"{name} (N/A)"

    def _result_index_with_address(self, index_1_based: int) -> str:
        start_word = (self.config.plc.addr.result_bits_start_word or "").strip()
        match = re.match(r"^([A-Za-z]+)(\d+)$", start_word)
        if not match:
            return str(index_1_based)
        prefix, base_text = match.groups()
        base = int(base_text)
        offset = max(0, index_1_based - 1)
        word = base + (offset // 16)
        bit = offset % 16
        return f"{index_1_based} ({prefix}{word}.{bit:02d})"

    @staticmethod
    def _is_plc_timeout_message(message: str) -> bool:
        text = (message or "").lower()
        return (
            "timeout waiting for plc response" in text
            or "timeout waiting for plc ack" in text
            or "plc ack wait failed" in text
            or "plc ack clear wait failed" in text
        )

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
        right_tabs.addTab(inspection_tab, "Main")
        inspection_layout = QtWidgets.QVBoxLayout(inspection_tab)

        inspection_content = QtWidgets.QWidget()
        inspection_content_layout = QtWidgets.QVBoxLayout(inspection_content)

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

        runtime_group = QtWidgets.QGroupBox("Runtime")
        runtime_form = QtWidgets.QFormLayout(runtime_group)
        self.model_label = QtWidgets.QLabel(self._current_model_display_name())
        self.speed_label = QtWidgets.QLabel("0.0 ms")
        self.runtime_mode_label = QtWidgets.QLabel(self._current_mode.name)
        runtime_form.addRow("Model", self.model_label)
        runtime_form.addRow("Speed", self.speed_label)
        runtime_form.addRow("Mode", self.runtime_mode_label)
        model_layout.addWidget(runtime_group)

        actions_group = QtWidgets.QGroupBox("Actions")
        controls_layout = QtWidgets.QGridLayout(actions_group)
        self.capture_button = QtWidgets.QPushButton("TRIGGER")
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
        model_layout.addWidget(actions_group)

        summary_bar.addWidget(model_widget, stretch=2)
        inspection_content_layout.addLayout(summary_bar)

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
        inspection_content_layout.addWidget(self.image_label, stretch=7)

        self.result_table = QtWidgets.QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["Index", "Score", "D(mm)", "Notch", "Status"])
        self.result_table.setColumnHidden(3, not self._show_notch_column)
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
        table_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        table_scroll.setWidget(self.result_table)
        inspection_content_layout.addWidget(table_scroll, stretch=0)

        diameter_min_mm, diameter_max_mm = self._current_diameter_thresholds()
        self.applied_threshold_label = QtWidgets.QLabel(
            f"Applied thresholds: score<={self._current_threshold():.3f} | diameter=[{diameter_min_mm:.3f}, {diameter_max_mm:.3f}] mm{self._current_notch_text()}"
        )
        self.applied_threshold_label.setAlignment(QtCore.Qt.AlignRight)
        inspection_content_layout.addWidget(self.applied_threshold_label, stretch=0)

        inspection_scroll = QtWidgets.QScrollArea()
        inspection_scroll.setWidgetResizable(True)
        inspection_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inspection_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        inspection_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        inspection_scroll.setWidget(inspection_content)
        inspection_layout.addWidget(inspection_scroll)

        custom_dialog_tab = QtWidgets.QWidget()
        right_tabs.addTab(custom_dialog_tab, "Custom Dialog")
        custom_dialog_layout = QtWidgets.QVBoxLayout(custom_dialog_tab)
        custom_dialog_layout.addWidget(self._build_custom_dialog_limits_table())
        custom_dialog_layout.addStretch(1)

        plc_tab = QtWidgets.QWidget()
        right_tabs.addTab(plc_tab, "PLC Monitor")
        plc_layout = QtWidgets.QVBoxLayout(plc_tab)

        plc_content = QtWidgets.QWidget()
        plc_content_layout = QtWidgets.QVBoxLayout(plc_content)

        self.plc_monitor_toggle = QtWidgets.QCheckBox("Enable PLC Monitor")
        self.plc_monitor_status = QtWidgets.QLabel("Monitor: Off")
        plc_content_layout.addWidget(self.plc_monitor_toggle)
        plc_content_layout.addWidget(self.plc_monitor_status)

        tx_group = QtWidgets.QGroupBox("TX (App → PLC)")
        tx_form = QtWidgets.QFormLayout(tx_group)
        self.tx_busy_label = QtWidgets.QLabel("-")
        self.tx_done_label = QtWidgets.QLabel("-")
        self.tx_error_label = QtWidgets.QLabel("-")
        self.tx_ready_label = QtWidgets.QLabel("-")
        self.tx_run_label = QtWidgets.QLabel("-")
        self.tx_camera_capture_done_label = QtWidgets.QLabel("-")
        self.tx_mode_current_label = QtWidgets.QLabel("-")
        self.tx_model_current_label = QtWidgets.QLabel("-")
        self.tx_mirror_result_label = QtWidgets.QLabel("-")
        self.tx_detected_count_label = QtWidgets.QLabel("-")
        addr = self.config.plc.addr
        tx_form.addRow(self._format_addr_label("Busy", addr.busy), self.tx_busy_label)
        tx_form.addRow(self._format_addr_label("Done", addr.done), self.tx_done_label)
        tx_form.addRow(self._format_addr_label("Error", addr.error), self.tx_error_label)
        tx_form.addRow(self._format_addr_label("Ready", addr.ready), self.tx_ready_label)
        tx_form.addRow(self._format_addr_label("Run", addr.run), self.tx_run_label)
        tx_form.addRow(
            self._format_addr_label("Camera capture done", addr.camera_capture_done),
            self.tx_camera_capture_done_label,
        )
        tx_form.addRow(self._format_addr_label("Mode current word", addr.mode_current_word), self.tx_mode_current_label)
        tx_form.addRow(self._format_addr_label("Model current word", addr.model_current_word), self.tx_model_current_label)
        tx_form.addRow(self._format_addr_label("Mirror result word", addr.mirror_result_word), self.tx_mirror_result_label)
        tx_form.addRow(self._format_addr_label("Detected count word", addr.detected_count_word), self.tx_detected_count_label)
        plc_content_layout.addWidget(tx_group)

        rx_group = QtWidgets.QGroupBox("RX (PLC → App)")
        rx_form = QtWidgets.QFormLayout(rx_group)
        self.rx_trigger_label = QtWidgets.QLabel("-")
        self.rx_sample_trigger_label = QtWidgets.QLabel("-")
        self.rx_ack_label = QtWidgets.QLabel("-")
        self.rx_mode_request_label = QtWidgets.QLabel("-")
        self.rx_model_select_label = QtWidgets.QLabel("-")
        rx_form.addRow(self._format_addr_label("Trigger", addr.trigger), self.rx_trigger_label)
        rx_form.addRow(self._format_addr_label("Sample trigger", addr.sample_trigger), self.rx_sample_trigger_label)
        rx_form.addRow(self._format_addr_label("ACK", addr.ack), self.rx_ack_label)
        rx_form.addRow(self._format_addr_label("Mode request word", addr.mode_request_word), self.rx_mode_request_label)
        rx_form.addRow(self._format_addr_label("Model select word", addr.model_select_word), self.rx_model_select_label)
        plc_content_layout.addWidget(rx_group)

        self.plc_results_table = QtWidgets.QTableWidget(self.config.layout.count, 2)
        self.plc_results_table.setHorizontalHeaderLabels(["Index", "Result"])
        self.plc_results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.plc_results_table.verticalHeader().setVisible(False)
        self.plc_results_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.plc_results_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.plc_results_table.setWordWrap(False)
        self.plc_results_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        for row in range(self.config.layout.count):
            self.plc_results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(self._result_index_with_address(row + 1)))
            self.plc_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
        self._set_table_row_heights(self.plc_results_table)
        plc_content_layout.addWidget(self.plc_results_table)

        self.plc_monitor_error = QtWidgets.QLabel("")
        self.plc_monitor_error.setStyleSheet("color: red;")
        plc_content_layout.addWidget(self.plc_monitor_error)
        plc_content_layout.addStretch(1)

        plc_scroll = QtWidgets.QScrollArea()
        plc_scroll.setWidgetResizable(True)
        plc_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        plc_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        plc_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        plc_scroll.setWidget(plc_content)
        plc_layout.addWidget(plc_scroll)

        options_menu = self.menuBar().addMenu("File")
        self.save_images_action = QtWidgets.QAction("Save processed images", self)
        self.save_images_action.setCheckable(True)
        self.save_images_action.setChecked(self.config.io.save_images)
        options_menu.addAction(self.save_images_action)

        self.save_only_ng_action = QtWidgets.QAction("Save only NG cycles", self)
        self.save_only_ng_action.setCheckable(True)
        self.save_only_ng_action.setChecked(getattr(self.config.io, "save_only_ng", False))
        options_menu.addAction(self.save_only_ng_action)

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
        self.save_ng_crops_only_action = QtWidgets.QAction("Save NG crops only", self)
        self.save_ng_crops_only_action.setCheckable(True)
        self.save_ng_crops_only_action.setChecked(getattr(self.config.io, "save_ng_crops_only", False))
        options_menu.addAction(self.save_ng_crops_only_action)

        self.enable_yolo_action = QtWidgets.QAction("Enable YOLO", self)
        self.enable_yolo_action.setCheckable(True)
        self.enable_yolo_action.setChecked(self.config.models.yolo.enabled)
        options_menu.addAction(self.enable_yolo_action)

        select_output_action = QtWidgets.QAction("Select output folder", self)
        options_menu.addAction(select_output_action)

        mode_menu = self.menuBar().addMenu("Mode")
        self.mode_action_group = QtWidgets.QActionGroup(self)
        self.mode_action_group.setExclusive(True)
        self.mode_run_action = QtWidgets.QAction("RUN", self, checkable=True)
        self.mode_sample_action = QtWidgets.QAction("SAMPLE", self, checkable=True)
        self.mode_mirror_action = QtWidgets.QAction("MIRROR", self, checkable=True)
        for action in (self.mode_run_action, self.mode_sample_action, self.mode_mirror_action):
            self.mode_action_group.addAction(action)
            mode_menu.addAction(action)

        tools_menu = self.menuBar().addMenu("Tools")
        self.test_mirror_from_image_action = QtWidgets.QAction("Test Mirror from Image", self)
        tools_menu.addAction(self.test_mirror_from_image_action)


        self.status_camera = QtWidgets.QLabel("Camera: Idle")
        self.status_plc = QtWidgets.QLabel(f"PLC: {self._plc_status}")
        self.status_mode = QtWidgets.QLabel("Mode: RUN")
        status_min_width = 260
        if configured_width and configured_width <= 800:
            status_min_width = 140
        self.status_camera.setMinimumWidth(status_min_width)
        self.status_plc.setMinimumWidth(status_min_width)
        self.statusBar().addWidget(self.status_camera)
        self.statusBar().addPermanentWidget(self.status_mode)
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
        self.save_only_ng_action.toggled.connect(self._toggle_save_only_ng)
        self.enable_yolo_action.toggled.connect(self._toggle_yolo)
        self.save_heatmap_action.toggled.connect(self._toggle_save_heatmap)
        self.save_binary_action.toggled.connect(self._toggle_save_binary)
        self.save_crops_action.toggled.connect(self._toggle_save_crops)
        self.save_ng_crops_only_action.toggled.connect(self._toggle_save_ng_crops_only)
        self.plc_monitor_toggle.toggled.connect(self._toggle_plc_monitor)
        self.mode_run_action.triggered.connect(lambda: self._on_manual_mode_selected(OperatingMode.RUN))
        self.mode_sample_action.triggered.connect(lambda: self._on_manual_mode_selected(OperatingMode.SAMPLE))
        self.mode_mirror_action.triggered.connect(lambda: self._on_manual_mode_selected(OperatingMode.MIRROR))
        self.test_mirror_from_image_action.triggered.connect(self._test_mirror_from_image)
        self.run_anomaly_button.setEnabled(False)

        self.plc_monitor_timer = QtCore.QTimer(self)
        self.plc_monitor_timer.setInterval(200)
        self.plc_monitor_timer.timeout.connect(self._poll_plc_monitor)

        self.model_sync_timer = QtCore.QTimer(self)
        self.model_sync_timer.setInterval(500)
        self.model_sync_timer.timeout.connect(self._model_sync_heartbeat)

        # Safety heartbeat: ensure queued mode requests are applied as soon as app becomes idle.
        self.mode_apply_timer = QtCore.QTimer(self)
        self.mode_apply_timer.setInterval(200)
        self.mode_apply_timer.timeout.connect(self._on_mode_apply_tick)
        self.mode_apply_timer.start()

        self._apply_machine_style()

    def _apply_machine_style(self) -> None:
        """Apply a dark machine-style theme without changing workflow behavior."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #000000;
                color: #ffffff;
                font-family: Helvetica, Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #000000;
            }
            QTabBar::tab {
                background-color: #202020;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px 18px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                border-bottom-color: #3a3a3a;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 3px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #1f1f1f;
                color: #ffffff;
                border: 1px solid #777777;
                border-radius: 3px;
                padding: 10px 14px;
                font-weight: bold;
                min-height: 34px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QPushButton:disabled {
                color: #777777;
                background-color: #151515;
                border-color: #333333;
            }
            QTableWidget {
                background-color: #101010;
                alternate-background-color: #1a1a1a;
                color: #ffffff;
                gridline-color: #555555;
                border: 1px solid #555555;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
            }
            QMenuBar, QMenu {
                background-color: #111111;
                color: #ffffff;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background-color: #333333;
            }
            QStatusBar {
                background-color: #111111;
                color: #ffffff;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            """
        )
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #777777;")

    def _on_mode_apply_tick(self) -> None:
        self._update_manual_mode_controls()
        self._apply_requested_mode_if_idle()

    def _update_manual_mode_controls(self) -> None:
        manual_allowed = self.plc.is_mock_mode()
        for action in (self.mode_run_action, self.mode_sample_action, self.mode_mirror_action):
            action.setEnabled(manual_allowed)

    def _init_workers(self) -> None:
        self.inspection_thread = QtCore.QThread(self)
        self.worker = InspectionWorker(
            self.config,
            self.plc,
            use_dummy_camera=self._use_dummy_camera,
            config_path=self.config_path,
        )
        self.worker.moveToThread(self.inspection_thread)
        self.inspection_thread.start()

        self.save_thread = QtCore.QThread(self)
        self.save_worker = SaveWorker(self.config)
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.start()

        self.trigger_worker = None
        self._refresh_main_trigger_worker()

        self.mode_select_worker = None
        mode_addr = getattr(self.config.plc.addr, "mode_request_word", None)
        if mode_addr:
            mode_poll_interval = max(self.config.plc.model_poll_interval_ms, 10) / 1000.0
            self.mode_select_worker = PlcModeSelectWorker(
                self.plc,
                mode_addr,
                poll_interval=mode_poll_interval,
                stable_ms=self.config.plc.model_stable_ms,
                error_backoff_ms=self.config.plc.poll_error_backoff_ms,
                parent=self,
            )
            self.mode_select_worker.mode_changed.connect(self._on_plc_mode_changed)
            self.mode_select_worker.start()

        self.sample_trigger_worker = None
        self._refresh_sample_trigger_worker()

        self.model_select_worker = None
        model_select_addr = getattr(self.config.plc.addr, "model_select_word", None)
        if model_select_addr and self._recipe_by_code:
            model_poll_interval = max(self.config.plc.model_poll_interval_ms, 10) / 1000.0
            self.model_select_worker = PlcModelSelectWorker(
                self.plc,
                model_select_addr,
                poll_interval=model_poll_interval,
                stable_ms=self.config.plc.model_stable_ms,
                error_backoff_ms=self.config.plc.poll_error_backoff_ms,
                parent=self,
            )
            self.model_select_worker.model_code_changed.connect(self._on_plc_model_code_changed)
            self.model_select_worker.start()

        if getattr(self.config.plc.addr, "model_select_word", None) and getattr(self.config.plc.addr, "model_current_word", None):
            self.model_sync_timer.start()

        self.trigger_manual.connect(self._handle_trigger)
        self.worker.cycle_started.connect(self._on_cycle_started)
        self.worker.cycle_completed.connect(self._update_ui)
        self.worker.cycle_failed.connect(self._handle_failure)
        self.worker.model_reloaded.connect(self._on_model_reloaded)
        self.worker.model_reload_failed.connect(self._on_model_reload_failed)
        self.worker.camera_ready.connect(self._on_camera_ready)
        self.worker.camera_failed.connect(self._on_camera_failed)
        self.worker.mirror_test_completed.connect(self._on_mirror_test_completed)
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

        self._sync_mode_actions()
        self._write_mode_current()

        if messages:
            non_timeout_messages = [m for m in messages if not self._is_plc_timeout_message(m)]
            timeout_messages = [m for m in messages if self._is_plc_timeout_message(m)]
            if non_timeout_messages:
                QtWidgets.QMessageBox.warning(self, "Startup issues", "\n".join(non_timeout_messages))
            if timeout_messages:
                self._show_rate_limited_status(
                    "startup_plc_timeout",
                    "; ".join(timeout_messages),
                    1000,
                )
            else:
                self.statusBar().showMessage("; ".join(non_timeout_messages), 5000)

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

    def _start_cycle_method(self, method_name: str, *args: object) -> None:
        if self._recipe_switch_in_progress:
            self._show_rate_limited_status("switching_ignore", "Model is switching, trigger ignored", 1000)
            return
        if self._cycle_request_inflight:
            self._show_rate_limited_status("cycle_busy_ignore", "Inspection cycle already in progress, trigger ignored", 1000)
            return
        self._cycle_request_inflight = True
        if args:
            QtCore.QMetaObject.invokeMethod(
                self.worker,
                method_name,
                QtCore.Qt.QueuedConnection,
                *[QtCore.Q_ARG(type(arg), arg) for arg in args],
            )
        else:
            QtCore.QMetaObject.invokeMethod(self.worker, method_name, QtCore.Qt.QueuedConnection)

    def _pick_sample_image(self) -> Optional[str]:
        recipe = self._recipe_by_code.get(int(self._current_recipe_code or 0))
        product_code = recipe.name if recipe and recipe.name else str(self._current_recipe_code or "")
        product_code = (product_code or "").strip()
        if not product_code:
            return None
        sample_dir = Path(self.config.sample_image_root) / product_code
        if not sample_dir.exists():
            return None
        images = sorted(sample_dir.glob("*.png"))
        if not images:
            return None
        idx = int(QtCore.QDateTime.currentMSecsSinceEpoch()) % len(images)
        return str(images[idx])

    @QtCore.pyqtSlot()
    def _handle_trigger(self) -> None:
        """Handle manual trigger requests."""
        if self._current_mode == OperatingMode.MIRROR:
            self._start_cycle_method("run_mirror_cycle")
            return
        if self._current_mode == OperatingMode.SAMPLE:
            sample_path = self._pick_sample_image()
            if not sample_path:
                self.statusBar().showMessage("Warning: không tìm thấy hình ảnh mẫu", 3000)
                return
            self._start_cycle_method("run_sample_cycle", sample_path)
            return
        self._start_cycle_method("run_cycle")

    @QtCore.pyqtSlot()
    def _handle_plc_trigger(self) -> None:
        """Handle the main PLC trigger bit.

        In SAMPLE mode, inspection must be started only by `sample_trigger`,
        so the main trigger is ignored.
        """
        if self._current_mode == OperatingMode.SAMPLE:
            self._show_rate_limited_status(
                "plc_trigger_ignored_in_sample",
                "Ignored main PLC trigger in SAMPLE mode",
                1000,
            )
            return
        self._handle_trigger()

    @QtCore.pyqtSlot()
    def _handle_sample_trigger(self) -> None:
        if self._current_mode != OperatingMode.SAMPLE:
            return
        sample_path = self._pick_sample_image()
        if not sample_path:
            self.statusBar().showMessage("Warning: không tìm thấy hình ảnh mẫu", 3000)
            return
        self._start_cycle_method("run_sample_cycle", sample_path)

    @QtCore.pyqtSlot()
    def _on_cycle_started(self) -> None:
        self.status_camera.setText(f"Camera: Busy ({self._current_mode.name})")

    @QtCore.pyqtSlot(InspectionResult)
    def _update_ui(self, result: InspectionResult) -> None:
        self._cycle_request_inflight = False
        self._set_display_image(numpy_to_qimage(result.overlay_image))

        self.result_table.setRowCount(len(result.patches))
        self._set_table_row_heights(self.result_table)
        for row, (patch, score, diameter_mm, notch_count, status) in enumerate(
            zip(result.patches, result.anomaly_scores, result.diameters_mm, result.notch_counts, result.statuses)
        ):
            self.result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(patch.index)))
            self.result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{score:.3f}"))
            self.result_table.setItem(row, 2, QtWidgets.QTableWidgetItem("-" if diameter_mm is None else f"{diameter_mm:.3f}"))
            self.result_table.setItem(row, 3, QtWidgets.QTableWidgetItem("-" if notch_count is None else str(int(notch_count))))
            self.result_table.setItem(row, 4, QtWidgets.QTableWidgetItem(status))
        notch_text = (
            f" | notch=[{result.notch_min_count},{result.notch_max_count}]"
            if result.notch_check_enabled
            else " | notch=OFF"
        )
        self.applied_threshold_label.setText(
            f"Applied thresholds: score<={result.threshold:.3f} | diameter=[{result.diameter_min_mm:.3f}, {result.diameter_max_mm:.3f}] mm{notch_text}"
        )
        ok_total = sum(1 for status in result.statuses if status == "OK")
        self.ok_label.setText(str(ok_total))
        self.ng_label.setText(str(result.ng_total))
        expected_total = result.expected_circles if result.expected_circles is not None else self.config.layout.count
        overall = "OK" if expected_total > 0 and len(result.statuses) == expected_total and result.ng_total == 0 else "NG"
        self._set_overall_status(overall)
        self.speed_label.setText(f"{result.cycle_elapsed_ms:.1f} ms")
        if result.detected_circles is not None and result.expected_circles is not None:
            self.status_camera.setText(
                f"Camera: Ready (circles {result.detected_circles}/{result.expected_circles})"
            )
        else:
            self.status_camera.setText("Camera: Ready")

        should_save = self.save_images_action.isChecked() and (
            not getattr(self.config.io, "save_only_ng", False) or result.ng_total > 0
        )
        if should_save:
            QtCore.QMetaObject.invokeMethod(
                self.save_worker,
                "save",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, result),
            )

        if self._pending_recipe_code is not None and not self._recipe_switch_in_progress and not self.plc.state.busy:
            self._apply_recipe_code(self._pending_recipe_code)
        self._apply_requested_mode_if_idle()

    @QtCore.pyqtSlot(str)
    def _handle_failure(self, message: str) -> None:
        self._cycle_request_inflight = False
        mirror_related = self._current_mode == OperatingMode.MIRROR or (message or "").startswith("MIRROR:")
        mirror_camera_error = (message or "").startswith("MIRROR_CAMERA:")
        mirror_detect_error = (message or "").startswith("MIRROR_DETECT:")
        if self._is_plc_timeout_message(message):
            self._show_rate_limited_status(
                "inspection_plc_timeout",
                f"Inspection failed (PLC timeout): {message}",
                1000,
            )
        elif mirror_camera_error:
            clean_message = (message or "").replace("MIRROR_CAMERA:", "", 1).strip()
            LOGGER.warning("Mirror camera error: %s", clean_message)
            self._show_rate_limited_status("mirror_camera_error", f"Mirror camera error: {clean_message}", 1000)
            self.status_camera.setText("Camera: Error")
            self.result_table.setRowCount(1)
            self._set_table_row_heights(self.result_table)
            self.result_table.setItem(0, 0, QtWidgets.QTableWidgetItem("MIRROR"))
            self.result_table.setItem(0, 1, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 2, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 3, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 4, QtWidgets.QTableWidgetItem("NG"))
            self.ok_label.setText("0")
            self.ng_label.setText("1")
            self._set_overall_status("NG")
        elif mirror_detect_error:
            clean_message = (message or "").replace("MIRROR_DETECT:", "", 1).strip()
            LOGGER.warning("Mirror detect error: %s", clean_message)
            self._show_rate_limited_status("mirror_detect_error", f"Mirror detect error: {clean_message}", 1000)
            self.status_camera.setText("Mirror: Detect Error")
            self.result_table.setRowCount(1)
            self._set_table_row_heights(self.result_table)
            self.result_table.setItem(0, 0, QtWidgets.QTableWidgetItem("MIRROR"))
            self.result_table.setItem(0, 1, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 2, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 3, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 4, QtWidgets.QTableWidgetItem("NG"))
            self.ok_label.setText("0")
            self.ng_label.setText("1")
            self._set_overall_status("NG")
        elif mirror_related:
            LOGGER.warning("Mirror warning (non-blocking): %s", message)
            self._show_rate_limited_status("mirror_warning", f"Mirror warning: {message}", 1000)
            self.status_camera.setText("Mirror: Warning")
            self.result_table.setRowCount(1)
            self._set_table_row_heights(self.result_table)
            self.result_table.setItem(0, 0, QtWidgets.QTableWidgetItem("MIRROR"))
            self.result_table.setItem(0, 1, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 2, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 3, QtWidgets.QTableWidgetItem("-"))
            self.result_table.setItem(0, 4, QtWidgets.QTableWidgetItem("NG"))
            self.ok_label.setText("0")
            self.ng_label.setText("1")
            self._set_overall_status("NG")
        else:
            QtWidgets.QMessageBox.critical(self, "Inspection failed", message)
            self.status_camera.setText("Camera: Error")
        self._apply_requested_mode_if_idle()

    def _show_rate_limited_status(self, key: str, message: str, min_interval_ms: int = 1000) -> None:
        now = QtCore.QDateTime.currentMSecsSinceEpoch() / 1000.0
        last = self._last_status_message_ts.get(key, 0.0)
        if (now - last) * 1000.0 >= max(0, min_interval_ms):
            self._last_status_message_ts[key] = now
            self.statusBar().showMessage(message, 2000)

    @QtCore.pyqtSlot()
    def _test_mirror_from_image(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select mirror test image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "run_mirror_test_on_image",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, file_path),
        )

    @QtCore.pyqtSlot(object, float, bool)
    def _on_mirror_test_completed(self, overlay_image: object, diameter: float, is_ok: bool) -> None:
        self._cycle_request_inflight = False
        if isinstance(overlay_image, np.ndarray):
            self._set_display_image(numpy_to_qimage(overlay_image))
        self.result_table.setRowCount(1)
        self._set_table_row_heights(self.result_table)
        self.result_table.setItem(0, 0, QtWidgets.QTableWidgetItem("MIRROR"))
        self.result_table.setItem(0, 1, QtWidgets.QTableWidgetItem("-"))
        self.result_table.setItem(0, 2, QtWidgets.QTableWidgetItem(f"{diameter:.3f}"))
        self.result_table.setItem(0, 3, QtWidgets.QTableWidgetItem("-"))
        self.result_table.setItem(0, 4, QtWidgets.QTableWidgetItem("OK" if is_ok else "NG"))
        self.ok_label.setText("1" if is_ok else "0")
        self.ng_label.setText("0" if is_ok else "1")
        self._set_overall_status("OK" if is_ok else "NG")
        self.status_camera.setText("Camera: Ready")
        self.applied_threshold_label.setText("Applied thresholds: -")
        self.statusBar().showMessage(
            f"Mirror test {'OK' if is_ok else 'NG'} | diameter={diameter:.2f}px",
            4000,
        )

    def _start_sample_trigger_worker(self) -> None:
        if self.sample_trigger_worker is not None:
            return
        sample_trigger_addr = getattr(self.config.plc.addr, "sample_trigger", None)
        if not sample_trigger_addr:
            return
        sample_poll_interval = max(self.config.plc.sample_trigger_poll_interval_ms, 1) / 1000.0
        worker = PlcTriggerWorker(
            self.plc,
            trigger_address=sample_trigger_addr,
            poll_interval=sample_poll_interval,
            min_interval_ms=self.config.plc.trigger_min_interval_ms,
            high_stable_ms=self.config.plc.trigger_high_stable_ms,
            low_stable_ms=self.config.plc.trigger_low_stable_ms,
            cooldown_ms=self.config.plc.trigger_cooldown_ms,
            error_backoff_ms=self.config.plc.poll_error_backoff_ms,
        )
        worker.triggered.connect(self._handle_sample_trigger)
        worker.start()
        self.sample_trigger_worker = worker

    def _start_main_trigger_worker(self) -> None:
        if not self.config.plc.enable_plc_trigger or self.trigger_worker is not None:
            return
        poll_interval = max(self.config.plc.trigger_poll_interval_ms, 1) / 1000.0
        worker = PlcTriggerWorker(
            self.plc,
            poll_interval=poll_interval,
            min_interval_ms=self.config.plc.trigger_min_interval_ms,
            high_stable_ms=self.config.plc.trigger_high_stable_ms,
            low_stable_ms=self.config.plc.trigger_low_stable_ms,
            cooldown_ms=self.config.plc.trigger_cooldown_ms,
            error_backoff_ms=self.config.plc.poll_error_backoff_ms,
        )
        worker.triggered.connect(self._handle_plc_trigger)
        worker.start()
        self.trigger_worker = worker

    def _stop_main_trigger_worker(self) -> None:
        if self.trigger_worker is None:
            return
        self.trigger_worker.stop()
        self.trigger_worker = None

    def _refresh_main_trigger_worker(self) -> None:
        if self._current_mode == OperatingMode.SAMPLE:
            self._stop_main_trigger_worker()
        else:
            self._start_main_trigger_worker()

    def _stop_sample_trigger_worker(self) -> None:
        if self.sample_trigger_worker is None:
            return
        self.sample_trigger_worker.stop()
        self.sample_trigger_worker = None

    def _refresh_sample_trigger_worker(self) -> None:
        if self._current_mode == OperatingMode.SAMPLE:
            self._start_sample_trigger_worker()
        else:
            self._stop_sample_trigger_worker()

    def _sync_mode_actions(self) -> None:
        mapping = {
            OperatingMode.RUN: self.mode_run_action,
            OperatingMode.SAMPLE: self.mode_sample_action,
            OperatingMode.MIRROR: self.mode_mirror_action,
        }
        action = mapping.get(self._current_mode)
        if action is not None:
            action.setChecked(True)

    def _on_manual_mode_selected(self, mode: OperatingMode) -> None:
        # Manual mode selection is allowed only when PLC is not connected to a real client.
        if not self.plc.is_mock_mode():
            self.statusBar().showMessage(
                "Manual mode is locked while PLC is connected (PLC has priority)",
                3000,
            )
            self._sync_mode_actions()
            return
        self._requested_mode = mode
        self.statusBar().showMessage(f"Manual mode requested: {mode.name}", 2000)
        self._apply_requested_mode_if_idle()

    def _write_mode_current(self) -> None:
        mode_current_addr = getattr(self.config.plc.addr, "mode_current_word", None)
        if not mode_current_addr:
            return
        try:
            self.plc.write_word(mode_current_addr, int(self._current_mode))
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to write mode current: {exc}", 5000)

    def _set_mode(self, mode: OperatingMode) -> None:
        self._current_mode = mode
        self.status_mode.setText(f"Mode: {mode.name}")
        self.runtime_mode_label.setText(mode.name)
        self._sync_mode_actions()
        self._refresh_main_trigger_worker()
        self._refresh_sample_trigger_worker()
        self._write_mode_current()

    def _apply_requested_mode_if_idle(self) -> None:
        if self._cycle_request_inflight:
            return
        if self.plc.state.busy:
            busy_addr = getattr(self.config.plc.addr, "busy", None)
            if busy_addr:
                try:
                    plc_busy = self.plc.read_bit(busy_addr)
                except Exception:
                    return
                if plc_busy:
                    return
            else:
                return
        if self._requested_mode != self._current_mode:
            self._set_mode(self._requested_mode)

    @QtCore.pyqtSlot(int)
    def _on_plc_mode_changed(self, mode_value: int) -> None:
        mode_map = {3: OperatingMode.RUN, 2: OperatingMode.SAMPLE, 1: OperatingMode.MIRROR}
        target = mode_map.get(int(mode_value))
        if target is None:
            self._show_rate_limited_status(
                "plc_mode_invalid",
                f"Ignored invalid PLC mode value: {mode_value}",
                1000,
            )
            return
        self._requested_mode = target
        if self._cycle_request_inflight or self.plc.state.busy:
            self.statusBar().showMessage(f"PLC mode {target.name} queued until cycle complete", 3000)
            return
        self.statusBar().showMessage(f"PLC override mode -> {target.name}", 2500)
        self._set_mode(target)

    def _model_sync_heartbeat(self) -> None:
        model_select_addr = getattr(self.config.plc.addr, "model_select_word", None)
        model_current_addr = getattr(self.config.plc.addr, "model_current_word", None)
        if not model_select_addr or not model_current_addr:
            return
        try:
            select_code = int(self.plc.read_word(model_select_addr))
            current_code = int(self.plc.read_word(model_current_addr))
        except PLCError as exc:
            self._show_rate_limited_status("model_sync_read_fail", f"Model sync monitor read failed: {exc}", 2000)
            return

        mismatch = select_code != current_code
        if mismatch:
            self._model_mismatch_streak += 1
            if self._model_mismatch_streak >= 3:
                self._show_rate_limited_status(
                    "model_sync_mismatch",
                    f"Model mismatch: select={select_code}, current={current_code}",
                    1000,
                )
        else:
            self._model_mismatch_streak = 0

        expected_code = self._current_recipe_code
        if expected_code is None:
            return
        expected = int(expected_code)

        # Conditional heartbeat: re-write model_current only when PLC request already matches app model,
        # but model_current word on PLC is still out-of-sync.
        if (
            current_code != expected
            and select_code == expected
            and not self._recipe_switch_in_progress
            and self._pending_recipe_code is None
            and not self.plc.state.busy
            and not self._cycle_request_inflight
        ):
            now = time.time()
            if (now - self._last_model_sync_write_ts) * 1000.0 < 1000.0:
                return
            try:
                self.plc.write_word(model_current_addr, expected)
                self._last_model_sync_write_ts = now
                self._show_rate_limited_status(
                    "model_sync_rewrite",
                    f"Model sync heartbeat rewrote current={expected}",
                    1000,
                )
            except PLCError as exc:
                self._show_rate_limited_status("model_sync_write_fail", f"Model sync rewrite failed: {exc}", 2000)

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

    def _read_plc_bit_text(self, address: Optional[str]) -> str:
        if not address:
            return "N/A"
        try:
            value = self.plc.read_bit(address)
            return "ON" if value else "OFF"
        except Exception:
            return "ERR"

    def _read_plc_word_text(self, address: Optional[str]) -> str:
        if not address:
            return "N/A"
        try:
            value = int(self.plc.read_word(address))
            return str(value)
        except Exception:
            return "ERR"

    def _poll_plc_monitor(self) -> None:
        try:
            trigger = self.plc.read_bit(self.plc.config.addr.trigger)
            ack = self.plc.read_bit(self.plc.config.addr.ack)
            sample_trigger_addr = getattr(self.plc.config.addr, "sample_trigger", None)
            mode_request_addr = getattr(self.plc.config.addr, "mode_request_word", None)
            model_select_addr = getattr(self.plc.config.addr, "model_select_word", None)
            mode_current_addr = getattr(self.plc.config.addr, "mode_current_word", None)
            model_current_addr = getattr(self.plc.config.addr, "model_current_word", None)
            mirror_result_addr = getattr(self.plc.config.addr, "mirror_result_word", None)
            detected_count_addr = getattr(self.plc.config.addr, "detected_count_word", None)

            self.rx_trigger_label.setText("ON" if trigger else "OFF")
            self.rx_sample_trigger_label.setText(self._read_plc_bit_text(sample_trigger_addr))
            self.rx_ack_label.setText("ON" if ack else "OFF")
            self.rx_mode_request_label.setText(self._read_plc_word_text(mode_request_addr))
            self.rx_model_select_label.setText(self._read_plc_word_text(model_select_addr))

            self.tx_busy_label.setText("ON" if self.plc.state.busy else "OFF")
            self.tx_done_label.setText("ON" if self.plc.state.done else "OFF")
            self.tx_error_label.setText("ON" if self.plc.state.error else "OFF")
            self.tx_ready_label.setText("ON" if self.plc.state.ready else "OFF")
            self.tx_run_label.setText("ON" if self.plc.state.run else "OFF")
            self.tx_camera_capture_done_label.setText("ON" if self.plc.state.camera_capture_done else "OFF")
            self.tx_mode_current_label.setText(self._read_plc_word_text(mode_current_addr))
            self.tx_model_current_label.setText(self._read_plc_word_text(model_current_addr))
            self.tx_mirror_result_label.setText(self._read_plc_word_text(mirror_result_addr))
            self.tx_detected_count_label.setText(self._read_plc_word_text(detected_count_addr))
            self.status_plc.setText(f"PLC: {self.plc.connection_state()}")
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

    def _build_custom_dialog_limits_table(self) -> QtWidgets.QTableWidget:
        limits = list(getattr(getattr(self.config, "custom_dialog", None), "limits", []) or [])
        table = QtWidgets.QTableWidget(len(limits), 3)
        table.setHorizontalHeaderLabels(["Feature", "Gioi han duoi", "Gioi han tren"])
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(QtCore.Qt.NoFocus)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        for row, limit in enumerate(limits):
            values = (
                str(getattr(limit, "feature", "")),
                self._format_custom_limit_value(getattr(limit, "lower", None)),
                self._format_custom_limit_value(getattr(limit, "upper", None)),
            )
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                alignment = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter if col == 0 else QtCore.Qt.AlignCenter
                item.setTextAlignment(alignment)
                table.setItem(row, col, item)

        self._set_table_row_heights(table)
        table_height = table.horizontalHeader().height() + (self._table_row_height * max(1, len(limits))) + 4
        table.setFixedHeight(table_height)
        return table

    @staticmethod
    def _format_custom_limit_value(value: Optional[float]) -> str:
        if value is None:
            return "-"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.3f}".rstrip("0").rstrip(".")

    def _select_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.config.io.output_dir)
        if directory:
            self.config.io.output_dir = directory

    def _toggle_save_images(self, enabled: bool) -> None:
        self.config.io.save_images = enabled

    def _toggle_save_only_ng(self, enabled: bool) -> None:
        self.config.io.save_only_ng = enabled

    def _toggle_yolo(self, enabled: bool) -> None:
        self.config.models.yolo.enabled = enabled

    def _toggle_save_heatmap(self, enabled: bool) -> None:
        self.config.io.save_heatmap = enabled

    def _toggle_save_binary(self, enabled: bool) -> None:
        self.config.io.save_binary = enabled

    def _toggle_save_crops(self, enabled: bool) -> None:
        self.config.io.save_crops = enabled

    def _toggle_save_ng_crops_only(self, enabled: bool) -> None:
        self.config.io.save_ng_crops_only = enabled

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
        valid_codes = set(self._recipe_by_code.keys())
        model_code = int(model_code)
        if model_code not in valid_codes:
            self._show_rate_limited_status(
                "plc_model_invalid",
                f"Ignored invalid PLC model code: {model_code}",
                1000,
            )
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
        threshold = float(recipe.threshold)
        algo = (self.config.models.algo or "INP").upper()
        model_path = recipe.glass_model_path if algo == "GLASS" else recipe.inp_model_path
        self._pending_recipe_code = int(model_code)
        self._recipe_switch_in_progress = True
        try:
            self.plc.set_busy(True, reason="model_switch")
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to set BUSY for model switch: {exc}", 5000)
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "reload_recipe_models",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, model_path),
            QtCore.Q_ARG(float, threshold),
            QtCore.Q_ARG(str, (recipe.crop_yolo_path or "")),
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
        diameter_min_mm, diameter_max_mm = self._current_diameter_thresholds()
        self.applied_threshold_label.setText(
            f"Applied thresholds: score<={threshold:.3f} | diameter=[{diameter_min_mm:.3f}, {diameter_max_mm:.3f}] mm{self._current_notch_text()}"
        )
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
            self.plc.set_busy(False, reason="model_switch_done")
            self.plc.set_ready(True, reason="model_switch_done")
        except PLCError as exc:
            self.statusBar().showMessage(f"Failed to clear BUSY/READY after model switch: {exc}", 5000)
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
            self.plc.set_busy(False, reason="model_switch_fail")
            self.plc.set_ready(True, reason="model_switch_fail")
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
        if self._current_recipe_code is not None:
            recipe = self._recipe_by_code.get(int(self._current_recipe_code))
            if recipe is not None:
                return float(recipe.threshold)
        # Safety fallback.
        algo = (self.config.models.algo or "INP").upper()
        return float(self.config.models.glass.glass_threshold if algo == "GLASS" else self.config.models.inp.inp_threshold)

    def _current_notch_text(self) -> str:
        if self._current_recipe_code is not None:
            recipe = self._recipe_by_code.get(int(self._current_recipe_code))
            if recipe is not None and bool(getattr(recipe, "notch_check_enabled", False)):
                return f" | notch=[{int(recipe.notch_min_count)},{int(recipe.notch_max_count)}]"
        return " | notch=OFF"

    def _current_diameter_thresholds(self) -> tuple[float, float]:
        if self._current_recipe_code is not None:
            recipe = self._recipe_by_code.get(int(self._current_recipe_code))
            if recipe is not None:
                return (
                    float(getattr(recipe, "diameter_min_mm", 0.0)),
                    float(getattr(recipe, "diameter_max_mm", 99999.0)),
                )
        algo = (self.config.models.algo or "INP").upper()
        if algo != "GLASS":
            return (
                float(getattr(self.config.models.inp, "diameter_min_mm", 0.0)),
                float(getattr(self.config.models.inp, "diameter_max_mm", 99999.0)),
            )
        return (
            float(getattr(self.config.models.glass, "diameter_min_mm", 0.0)),
            float(getattr(self.config.models.glass, "diameter_max_mm", 99999.0)),
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
            if hasattr(self, "model_sync_timer"):
                self.model_sync_timer.stop()
            if hasattr(self, "mode_apply_timer"):
                self.mode_apply_timer.stop()
            QtCore.QMetaObject.invokeMethod(self.worker, "shutdown", QtCore.Qt.BlockingQueuedConnection)
            if self.trigger_worker is not None:
                self.trigger_worker.stop()
            if getattr(self, "sample_trigger_worker", None) is not None:
                self.sample_trigger_worker.stop()
            if getattr(self, "mode_select_worker", None) is not None:
                self.mode_select_worker.stop()
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

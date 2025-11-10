"""Simple camera test UI (real hardware only).

Usage:
    python test_camera.py --config config.yaml [--log-level INFO]
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from app.config_loader import AppConfig, ConfigError, load_config
from app.inspection.camera import BaslerCamera, CameraError
from app.utils import numpy_to_qimage


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class CameraTestWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Camera Test")
        self.resize(1200, 800)

        # Camera instance
        self.camera = BaslerCamera(config.camera)
        self._connected = False

        # UI
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        vbox.addWidget(self.image_label)

        btns = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_preview_start = QtWidgets.QPushButton("Start Preview")
        self.btn_preview_stop = QtWidgets.QPushButton("Stop Preview")
        self.btn_capture = QtWidgets.QPushButton("Capture Once")
        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_disconnect)
        btns.addWidget(self.btn_preview_start)
        btns.addWidget(self.btn_preview_stop)
        btns.addWidget(self.btn_capture)
        vbox.addLayout(btns)

        self.status = QtWidgets.QLabel("Camera: Idle")
        self.statusBar().addWidget(self.status)

        # Timer for preview
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._grab_frame)

        # Signals
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_preview_start.clicked.connect(lambda: self._set_preview(True))
        self.btn_preview_stop.clicked.connect(lambda: self._set_preview(False))
        self.btn_capture.clicked.connect(self._grab_frame)

        # Initial hint
        self.status.setText("Camera: Not connected")

    def _connect(self) -> None:
        if self._connected:
            return
        try:
            self.camera.connect()
            self._connected = True
            self.status.setText("Camera: Connected")
        except Exception as exc:
            logging.getLogger(__name__).error("Camera connect failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Camera connect failed", str(exc))
            self.status.setText("Camera: Error")

    def _disconnect(self) -> None:
        try:
            self.timer.stop()
            self.camera.disconnect()
        except Exception:
            pass
        self._connected = False
        self.status.setText("Camera: Disconnected")

    def _set_preview(self, enabled: bool) -> None:
        if enabled and not self._connected:
            self._connect()
        if enabled and self._connected:
            self.timer.start()
            self.status.setText("Camera: Preview")
        else:
            self.timer.stop()
            if self._connected:
                self.status.setText("Camera: Connected")
            else:
                self.status.setText("Camera: Disconnected")

    def _grab_frame(self) -> None:
        try:
            if not self._connected:
                self._connect()
            capture = self.camera.capture()
            pixmap = QtGui.QPixmap.fromImage(numpy_to_qimage(capture.image))
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )
            self.statusBar().showMessage("Captured frame", 1000)
        except Exception as exc:
            logging.getLogger(__name__).exception("Capture failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Capture failed", str(exc))
            self.status.setText("Camera: Error")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera tester (hardware only)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    app = QtWidgets.QApplication(sys.argv)
    win = CameraTestWindow(config)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

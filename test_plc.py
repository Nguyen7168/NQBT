"""Simple PLC test UI (real hardware only) that shares the same config as the main app.

Usage:
    python test_plc.py --config config.yaml [--log-level INFO]
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from app.config_loader import AppConfig, ConfigError, load_config
from app.inspection.plc_client import PlcController


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def create_plc_controller(config: AppConfig) -> PlcController:
    controller = PlcController(config.plc)
    controller.connect()  # Let exceptions bubble up — no mock fallback
    controller.set_busy(False)
    controller.set_done(False)
    controller.set_error(False)
    return controller


class PlcTestWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.plc = create_plc_controller(config)
        self.setWindowTitle("PLC Test")
        self.resize(800, 500)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # Status labels for key bits
        self.lbl_trigger = QtWidgets.QLabel("Trigger: ?")
        self.lbl_ack = QtWidgets.QLabel("ACK: ?")
        self.lbl_busy = QtWidgets.QLabel("Busy: ?")
        self.lbl_done = QtWidgets.QLabel("Done: ?")
        self.lbl_error = QtWidgets.QLabel("Error: ?")

        grid.addWidget(self.lbl_trigger, 0, 0)
        grid.addWidget(self.lbl_ack, 0, 1)
        grid.addWidget(self.lbl_busy, 1, 0)
        grid.addWidget(self.lbl_done, 1, 1)
        grid.addWidget(self.lbl_error, 1, 2)

        # Control buttons
        self.btn_busy_on = QtWidgets.QPushButton("Set Busy ON")
        self.btn_busy_off = QtWidgets.QPushButton("Set Busy OFF")
        self.btn_done_on = QtWidgets.QPushButton("Set Done ON")
        self.btn_done_off = QtWidgets.QPushButton("Set Done OFF")
        self.btn_err_on = QtWidgets.QPushButton("Set Error ON")
        self.btn_err_off = QtWidgets.QPushButton("Set Error OFF")
        self.btn_results_ok = QtWidgets.QPushButton("Write Results: All OK")
        self.btn_results_ng = QtWidgets.QPushButton("Write Results: All NG")
        self.btn_poll = QtWidgets.QPushButton("Poll Now")

        grid.addWidget(self.btn_busy_on, 2, 0)
        grid.addWidget(self.btn_busy_off, 2, 1)
        grid.addWidget(self.btn_done_on, 3, 0)
        grid.addWidget(self.btn_done_off, 3, 1)
        grid.addWidget(self.btn_err_on, 4, 0)
        grid.addWidget(self.btn_err_off, 4, 1)
        grid.addWidget(self.btn_results_ok, 5, 0)
        grid.addWidget(self.btn_results_ng, 5, 1)
        grid.addWidget(self.btn_poll, 6, 0)

        # Status bar
        self.statusBar().showMessage("PLC: Connected")

        # Connect signals
        self.btn_busy_on.clicked.connect(lambda: self._safe(lambda: self.plc.set_busy(True)))
        self.btn_busy_off.clicked.connect(lambda: self._safe(lambda: self.plc.set_busy(False)))
        self.btn_done_on.clicked.connect(lambda: self._safe(lambda: self.plc.set_done(True)))
        self.btn_done_off.clicked.connect(lambda: self._safe(lambda: self.plc.set_done(False)))
        self.btn_err_on.clicked.connect(lambda: self._safe(lambda: self.plc.set_error(True)))
        self.btn_err_off.clicked.connect(lambda: self._safe(lambda: self.plc.set_error(False)))
        self.btn_results_ok.clicked.connect(self._write_all_ok)
        self.btn_results_ng.clicked.connect(self._write_all_ng)
        self.btn_poll.clicked.connect(self._poll)

        # Poll timer
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

        # Initial read
        self._poll()

    def _safe(self, action) -> None:
        try:
            action()
        except Exception as exc:
            logging.getLogger(__name__).error("PLC action failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "PLC action failed", str(exc))

    def _write_all_ok(self) -> None:
        try:
            self.plc.write_results([True] * self.config.layout.count)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Write results failed", str(exc))

    def _write_all_ng(self) -> None:
        try:
            self.plc.write_results([False] * self.config.layout.count)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Write results failed", str(exc))

    def _poll(self) -> None:
        try:
            c = self.plc.client
            addr = self.plc.config.addr
            trigger = c.read_bit(addr.trigger)
            ack = c.read_bit(addr.ack)
            busy = c.read_bit(addr.busy)
            done = c.read_bit(addr.done)
            err = c.read_bit(addr.error)
            self.lbl_trigger.setText(f"Trigger: {trigger}")
            self.lbl_ack.setText(f"ACK: {ack}")
            self.lbl_busy.setText(f"Busy: {busy}")
            self.lbl_done.setText(f"Done: {done}")
            self.lbl_error.setText(f"Error: {err}")
        except Exception as exc:
            logging.getLogger(__name__).error("Poll failed: %s", exc)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLC tester (hardware only)")
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
    try:
        win = PlcTestWindow(config)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "PLC connect failed", str(exc))
        return 1
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

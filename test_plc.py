"""Simple PLC test UI (real hardware only) that shares the same config as the main app.

Usage:
    python test_plc.py --config config.yaml [--log-level INFO]
"""
from __future__ import annotations

import argparse
import logging
import re
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
        self.lbl_app_results = QtWidgets.QLabel("App Results: -")
        self.lbl_plc_results = QtWidgets.QLabel("PLC Results: -")
        self.lbl_compare = QtWidgets.QLabel("Compare: -")

        grid.addWidget(self.lbl_trigger, 0, 0)
        grid.addWidget(self.lbl_ack, 0, 1)
        grid.addWidget(self.lbl_busy, 1, 0)
        grid.addWidget(self.lbl_done, 1, 1)
        grid.addWidget(self.lbl_error, 1, 2)
        grid.addWidget(self.lbl_app_results, 2, 0)
        grid.addWidget(self.lbl_plc_results, 2, 1)
        grid.addWidget(self.lbl_compare, 2, 2)

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

        grid.addWidget(self.btn_busy_on, 3, 0)
        grid.addWidget(self.btn_busy_off, 3, 1)
        grid.addWidget(self.btn_done_on, 4, 0)
        grid.addWidget(self.btn_done_off, 4, 1)
        grid.addWidget(self.btn_err_on, 5, 0)
        grid.addWidget(self.btn_err_off, 5, 1)
        grid.addWidget(self.btn_results_ok, 6, 0)
        grid.addWidget(self.btn_results_ng, 6, 1)
        grid.addWidget(self.btn_poll, 7, 0)

        self.results_table = QtWidgets.QTableWidget(self.config.layout.count, 5)
        self.results_table.setHorizontalHeaderLabels(["Index", "Address", "App", "PLC", "Match"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        for row in range(self.config.layout.count):
            self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
        grid.addWidget(self.results_table, 8, 0, 1, 3)

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

        self.expected_results: Optional[list[bool]] = None

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
            self.expected_results = [True] * self.config.layout.count
            self.plc.write_results(self.expected_results)
            self._update_results_view()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Write results failed", str(exc))

    def _write_all_ng(self) -> None:
        try:
            self.expected_results = [False] * self.config.layout.count
            self.plc.write_results(self.expected_results)
            self._update_results_view()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Write results failed", str(exc))

    def _result_addresses(self) -> list[str]:
        start = self.plc.config.addr.result_bits_start_word
        match = re.match(r"^([A-Za-z]+)(\d+)(?:\.(\d+))?$", start)
        if not match:
            raise ValueError(f"Unsupported result address format: {start}")
        prefix, base_str, bit_str = match.group(1), match.group(2), match.group(3)
        count = self.config.layout.count
        if bit_str is not None or prefix in {"W", "D"}:
            base = int(base_str)
            bit_start = int(bit_str or 0)
            return [
                f"{prefix}{base + (bit_start + idx) // 16}.{(bit_start + idx) % 16:02d}"
                for idx in range(count)
            ]
        base = int(base_str)
        return [f"{prefix}{base + idx}" for idx in range(count)]

    def _read_plc_results(self) -> Optional[list[bool]]:
        try:
            addresses = self._result_addresses()
            return [self.plc.client.read_bit(addr) for addr in addresses]
        except Exception as exc:
            logging.getLogger(__name__).error("Read PLC results failed: %s", exc)
            return None

    def _update_results_view(self) -> None:
        addresses = self._result_addresses()
        plc_results = self._read_plc_results()
        app_results = self.expected_results
        app_ok = app_ng = plc_ok = plc_ng = mismatches = 0

        for idx in range(self.config.layout.count):
            addr_text = addresses[idx] if idx < len(addresses) else "-"
            app_val = app_results[idx] if app_results is not None else None
            plc_val = plc_results[idx] if plc_results is not None else None
            app_text = "OK" if app_val is True else "NG" if app_val is False else "-"
            plc_text = "OK" if plc_val is True else "NG" if plc_val is False else "-"
            match = app_val is not None and plc_val is not None and app_val == plc_val
            match_text = "OK" if match else "NG" if app_val is not None and plc_val is not None else "-"

            if app_val is True:
                app_ok += 1
            elif app_val is False:
                app_ng += 1
            if plc_val is True:
                plc_ok += 1
            elif plc_val is False:
                plc_ng += 1
            if app_val is not None and plc_val is not None and not match:
                mismatches += 1

            self.results_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(addr_text))
            self.results_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(app_text))
            self.results_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(plc_text))
            self.results_table.setItem(idx, 4, QtWidgets.QTableWidgetItem(match_text))

        self.lbl_app_results.setText(f"App Results: OK={app_ok} NG={app_ng}")
        if plc_results is None:
            self.lbl_plc_results.setText("PLC Results: read failed")
        else:
            self.lbl_plc_results.setText(f"PLC Results: OK={plc_ok} NG={plc_ng}")
        if app_results is None or plc_results is None:
            self.lbl_compare.setText("Compare: -")
        else:
            self.lbl_compare.setText(f"Compare: mismatches={mismatches}")

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
            self._update_results_view()
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

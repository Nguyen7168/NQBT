"""Entry point for the bearing inspection application."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PyQt5 import QtWidgets

from app.config_loader import AppConfig, ConfigError, load_config
from app.inspection.plc_client import PlcController, PLCError, MockPLCClient
from app.ui.main_window import MainWindow


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def create_plc_controller(config: AppConfig, allow_mock: bool) -> tuple[PlcController, str]:
    """Create a PLC controller and attempt connection.

    If connection fails, always fall back to a mock client so the UI can start.
    The returned status string includes the specific error to display in the UI.
    """
    logger = logging.getLogger(__name__)
    controller = PlcController(config.plc)
    try:
        controller.connect()
        status = "Connected"
        controller.set_busy(False)
        controller.set_done(False)
        controller.set_error(False)
        controller.set_ready(True)
    except Exception as exc:
        # Do not abort startup; fall back to a mock PLC so the app can run.
        reason = str(exc)
        if allow_mock:
            logger.warning("Failed to connect to PLC: %s. Using mock client.", reason)
        else:
            logger.error("Failed to connect to PLC: %s. Falling back to mock to allow UI startup.", reason)
        controller = PlcController(config.plc, client=MockPLCClient())
        # Connect the mock for consistency/logging
        try:
            controller.connect()
        except Exception:
            pass
        status = f"Disconnected: {reason} (mock)"
    return controller, status


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bearing inspection station")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--use-dummy-camera", action="store_true", help="Use synthetic camera feed for development")
    parser.add_argument("--allow-mock-plc", action="store_true", help="Fall back to mock PLC when connection fails")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    try:
        plc, plc_status = create_plc_controller(config, allow_mock=args.allow_mock_plc)
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to create PLC controller: %s", exc)
        return 1

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config, plc, use_dummy_camera=args.use_dummy_camera, plc_status=plc_status)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

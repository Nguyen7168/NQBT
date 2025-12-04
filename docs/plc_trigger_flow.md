# PLC trigger handling flow

This document maps the PLC trigger path to concrete functions and files so you can follow the runtime call chain.

## Trigger detection
- **`PlcTriggerWorker.run`** (`app/inspection/workers.py`): background `QThread` that polls the PLC trigger bit (`plc.config.addr.trigger`). When it reads `True`, it emits the Qt signal `triggered` and waits for the bit to clear before continuing.

## Dispatch to the inspection cycle
- **`MainWindow._init_workers`** (`app/ui/main_window.py`): connects `PlcTriggerWorker.triggered` and the manual UI button signal `trigger_manual` to `_handle_trigger`.
- **`MainWindow._handle_trigger`** (`app/ui/main_window.py`): enqueues `InspectionWorker.run_cycle` on the worker thread via `QtCore.QMetaObject.invokeMethod`, ensuring the heavy work happens off the UI thread.

## Inspection cycle execution
- **`InspectionWorker.run_cycle`** (`app/inspection/workers.py`): guarded by a lock to serialize cycles.
  1. Sets PLC **busy** flag and establishes a camera connection if needed.
  2. Captures an image, crops ROIs, runs anomaly detection (and YOLO if enabled), and aggregates OK/NG statuses.
  3. Writes OK/NG bits back to the PLC, clears error state, raises **done** flag, and emits `cycle_completed` (or `cycle_failed` on exception after writing all NG and raising error).
  4. In `finally`, calls `plc.finalize_cycle()` to wait for PLC **ack**, then clears `done`/`busy` (and `error` if set).

## PLC handshake helpers
- **`PlcController.finalize_cycle`** (`app/inspection/plc_client.py`): waits for the PLC **ack** bit (within `cycle_ms` timeout), clears **done** and **busy**, resets **error** if it was set, then waits for the ack bit to drop before returning.

These functions form the end-to-end path from receiving a PLC trigger to completing the handshake after inference.

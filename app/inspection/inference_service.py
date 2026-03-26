"""Remote inference service and client for shared GPU scheduling."""
from __future__ import annotations

import json
import logging
import pickle
import socket
import socketserver
import struct
import threading
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from app.config_loader import AppConfig, load_config

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.inspection.inference_pipeline import InferencePipeline, InferenceResultPayload

_LENGTH_STRUCT = struct.Struct("!I")
_VALID_ALGOS = {"INP", "GLASS"}


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            raise ConnectionError("Connection closed while receiving payload")
        chunks.extend(block)
    return bytes(chunks)


def _send_message(sock: socket.socket, payload: Dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LENGTH_STRUCT.pack(len(data)))
    sock.sendall(data)


def _recv_message(sock: socket.socket) -> Dict[str, Any]:
    raw_size = _recv_exact(sock, _LENGTH_STRUCT.size)
    (size,) = _LENGTH_STRUCT.unpack(raw_size)
    data = _recv_exact(sock, size)
    message = pickle.loads(data)
    if not isinstance(message, dict):
        raise ValueError("Invalid message payload type")
    return message


def build_inference_runtime_state(config: AppConfig) -> Dict[str, Any]:
    algo = str(config.models.algo or "INP").strip().upper()
    if algo not in _VALID_ALGOS:
        algo = "INP"
    return {
        "algo": algo,
        "active_recipe_code": getattr(config.models, "active_recipe_code", None),
        "inp_path": config.models.inp.path,
        "inp_threshold": float(config.models.inp.inp_threshold),
        "glass_path": config.models.glass.path,
        "glass_threshold": float(config.models.glass.glass_threshold),
        "yolo_enabled": bool(config.models.yolo.enabled),
        "crop_path": getattr(config.models.yolo, "crop_path", None),
    }


def _apply_runtime_state(config: AppConfig, state: Dict[str, Any]) -> None:
    current_algo = str(config.models.algo or "INP").strip().upper()
    requested_algo_raw = state.get("algo", current_algo)
    requested_algo = str(requested_algo_raw).strip().upper() if requested_algo_raw is not None else current_algo
    if requested_algo not in _VALID_ALGOS:
        LOGGER.warning("Ignoring unsupported runtime algo '%s'; keeping %s", requested_algo_raw, current_algo)
        requested_algo = current_algo
    config.models.algo = requested_algo
    if state.get("active_recipe_code") is not None:
        config.models.active_recipe_code = int(state["active_recipe_code"])
    if state.get("inp_path"):
        config.models.inp.path = str(state["inp_path"])
    if state.get("inp_threshold") is not None:
        config.models.inp.inp_threshold = float(state["inp_threshold"])
    if state.get("glass_path"):
        config.models.glass.path = str(state["glass_path"])
    if state.get("glass_threshold") is not None:
        config.models.glass.glass_threshold = float(state["glass_threshold"])
    config.models.yolo.enabled = bool(state.get("yolo_enabled", config.models.yolo.enabled))
    crop_path = state.get("crop_path")
    if crop_path:
        config.models.yolo.crop_path = str(crop_path)


def _engine_cache_key(config_path: str, state: Dict[str, Any]) -> str:
    return json.dumps({"config_path": str(config_path), "state": state}, sort_keys=True, ensure_ascii=False)


def _result_to_dict(result: "InferenceResultPayload") -> Dict[str, Any]:
    return {
        "raw_image": result.raw_image,
        "overlay_image": result.overlay_image,
        "patches": result.patches,
        "anomaly_scores": result.anomaly_scores,
        "diameters_mm": result.diameters_mm,
        "statuses": result.statuses,
        "ng_total": result.ng_total,
        "anomaly_inference_ms": result.anomaly_inference_ms,
        "yolo_result": result.yolo_result,
        "timestamp": result.timestamp,
        "model_path": result.model_path,
        "threshold": result.threshold,
        "diameter_min_mm": result.diameter_min_mm,
        "diameter_max_mm": result.diameter_max_mm,
        "anomaly_maps": result.anomaly_maps,
        "detected_circles": result.detected_circles,
        "expected_circles": result.expected_circles,
    }


class RemoteInferenceClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 30000) -> None:
        self._host = host
        self._port = int(port)
        self._timeout_ms = max(1000, int(timeout_ms))

    def infer(self, image: np.ndarray, config_path: str, config: AppConfig) -> Dict[str, Any]:
        payload = {
            "action": "infer",
            "config_path": str(config_path),
            "runtime_state": build_inference_runtime_state(config),
            "image": image,
        }
        with socket.create_connection((self._host, self._port), timeout=self._timeout_ms / 1000.0) as sock:
            sock.settimeout(self._timeout_ms / 1000.0)
            _send_message(sock, payload)
            response = _recv_message(sock)
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error") or "Remote inference failed"))
        result = response.get("result")
        timings = response.get("timings", {})
        if not isinstance(result, dict):
            raise RuntimeError("Remote inference returned invalid result")
        if isinstance(timings, dict):
            result["_remote_timings"] = timings
        return result


class _InferenceEngineCache:
    def __init__(self) -> None:
        self._engines: dict[str, "InferencePipeline"] = {}
        self._lock = threading.Lock()

    def get_pipeline(self, config_path: str, state: Dict[str, Any]) -> "InferencePipeline":
        from app.inspection.inference_pipeline import InferencePipeline

        key = _engine_cache_key(config_path, state)
        with self._lock:
            cached = self._engines.get(key)
            if cached is not None:
                return cached
            config = load_config(config_path)
            _apply_runtime_state(config, state)
            pipeline = InferencePipeline(config)
            self._engines[key] = pipeline
            LOGGER.info("Inference service cached pipeline for %s", key)
            return pipeline


class InferenceServiceHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # pragma: no cover - exercised by runtime integration
        try:
            request = _recv_message(self.request)
            if request.get("action") != "infer":
                raise RuntimeError(f"Unsupported action: {request.get('action')}")
            config_path = str(request.get("config_path") or "").strip()
            if not config_path:
                raise RuntimeError("Missing config_path in inference request")
            image = request.get("image")
            if not isinstance(image, np.ndarray):
                raise RuntimeError("Inference request image must be a numpy array")
            runtime_state = request.get("runtime_state")
            if not isinstance(runtime_state, dict):
                raise RuntimeError("Inference request runtime_state must be a dict")
            pipeline = self.server.engine_cache.get_pipeline(config_path, runtime_state)  # type: ignore[attr-defined]
            timings: dict[str, float] = {}
            result = pipeline.run_on_image(image, timings=timings)
            _send_message(
                self.request,
                {
                    "status": "ok",
                    "result": _result_to_dict(result),
                    "timings": timings,
                },
            )
        except Exception as exc:
            LOGGER.exception("Inference service request failed: %s", exc)
            try:
                _send_message(self.request, {"status": "error", "error": str(exc)})
            except Exception:
                LOGGER.exception("Failed to return inference error to client")


class InferenceServiceServer(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, server_address: Tuple[str, int]):
        super().__init__(server_address, InferenceServiceHandler)
        self.engine_cache = _InferenceEngineCache()


def run_inference_service(host: str, port: int) -> None:
    with InferenceServiceServer((host, int(port))) as server:
        LOGGER.info("Inference service listening on %s:%d", host, int(port))
        server.serve_forever()

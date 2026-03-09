"""PLC communication helpers supporting Omron FINS/TCP."""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import re

from app.config_loader import PlcAddressConfig, PlcConfig

LOGGER = logging.getLogger(__name__)


class PLCError(RuntimeError):
    pass


@dataclass
class PlcHandshakeState:
    busy: bool = False
    done: bool = False
    error: bool = False
    ready: bool = False
    run: bool = False
    last_cycle_started: Optional[float] = None
    last_results: Optional[List[bool]] = None


class PlcConnectionState(str, Enum):
    REAL_CONNECTED = "REAL_CONNECTED"
    DEGRADED_MOCK = "DEGRADED_MOCK"
    RECONNECTING_REAL = "RECONNECTING_REAL"


class BasePLCClient:
    """Abstract PLC client interface."""

    def connect(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_bit(self, address: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def read_word(self, address: str) -> int:  # pragma: no cover
        raise NotImplementedError

    def write_bit(self, address: str, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_word(self, address: str, value: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:  # pragma: no cover
        raise NotImplementedError


class MockPLCClient(BasePLCClient):
    """In-memory PLC client useful for development."""

    def __init__(self) -> None:
        self._bits: dict[str, bool] = {}
        self._words: dict[str, int] = {}

    def connect(self) -> None:  # pragma: no cover
        LOGGER.info("Mock PLC connected")

    def close(self) -> None:  # pragma: no cover
        LOGGER.info("Mock PLC disconnected")

    def read_bit(self, address: str) -> bool:
        return self._bits.get(address, False)

    def write_bit(self, address: str, value: bool) -> None:
        LOGGER.debug("PLC bit %s <- %s", address, value)
        self._bits[address] = value

    def read_word(self, address: str) -> int:
        return int(self._words.get(address, 0))

    def write_word(self, address: str, value: int) -> None:
        LOGGER.debug("PLC word %s <- %s", address, value)
        self._words[address] = int(value)

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:
        LOGGER.debug("PLC write results from %s: %s", start_word, bits)
        base = start_word
        for idx, bit in enumerate(bits):
            self._bits[f"{base}.{idx:02d}"] = bit


class OmronFinsTcpClient(BasePLCClient):
    """Minimalistic implementation of FINS/TCP bit read/write."""

    _FINS_HEADER = b"\x46\x49\x4e\x53\x00\x00\x00\x0c\x00\x00\x00\x00"

    def __init__(self, config: PlcConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._sid = 0x10

    def connect(self) -> None:
        self._sock = socket.create_connection((self._config.ip, self._config.port), timeout=self._config.timeouts.connect_ms / 1000.0)
        self._sock.settimeout(max(0.001, self._config.timeouts.response_ms / 1000.0))
        LOGGER.info("Connected to PLC at %s:%s", self._config.ip, self._config.port)

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, fins_cmd: bytes) -> bytes:
        if self._sock is None:
            raise PLCError("PLC not connected")
        self._sid = (self._sid + 1) % 0xFF
        header = bytearray(self._FINS_HEADER)
        length = len(fins_cmd) + 8
        header[7] = length
        header.extend(b"\x00\x00\x00\x02\x00\x00")
        header.append(self._sid)
        header.extend(b"\x00\x00")
        packet = bytes(header) + fins_cmd
        self._sock.sendall(packet)
        resp = self._sock.recv(2048)
        if len(resp) < 30:
            raise PLCError("Invalid FINS response")
        return resp

    def _parse_address(self, address: str) -> tuple[int, int]:
        # Example address W150.00
        raw = address.strip().upper()
        if raw.startswith("DM"):
            area = "D"
            raw = "D" + raw[2:]
        area = raw[0]
        if area not in {"W", "D"}:
            raise PLCError(f"Unsupported memory area in address {address}")
        word = int(raw[1:].split(".")[0])
        bit = int(raw.split(".")[1]) if "." in raw else 0
        return word, bit

    def write_bit(self, address: str, value: bool) -> None:
        word, bit = self._parse_address(address)
        fins_cmd = bytes(
            [
                0x80,
                0x00,
                0x02,
                0x00,
                0x01,
                0x30,
                0x00,
                0x00,
                (word >> 8) & 0xFF,
                word & 0xFF,
                bit,
                0x00,
                0x01,
                0x01 if value else 0x00,
            ]
        )
        self._send(fins_cmd)

    def read_bit(self, address: str) -> bool:
        word, bit = self._parse_address(address)
        fins_cmd = bytes(
            [
                0x80,
                0x00,
                0x01,
                0x01,
                0x01,
                0x30,
                0x00,
                0x00,
                (word >> 8) & 0xFF,
                word & 0xFF,
                bit,
                0x00,
                0x00,
                0x01,
            ]
        )
        resp = self._send(fins_cmd)
        return bool(resp[-1] & 0x01)

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:
        word, _ = self._parse_address(start_word)
        word_count = (len(bits) + 15) // 16
        payload = bytearray()
        for word_idx in range(word_count):
            value = 0
            for bit_idx in range(16):
                global_idx = word_idx * 16 + bit_idx
                if global_idx < len(bits) and bits[global_idx]:
                    value |= 1 << bit_idx
            payload.extend(struct.pack(">H", value))
        fins_cmd = bytes(
            [
                0x80,
                0x00,
                0x02,
                0x00,
                0x01,
                0x30,
                0x00,
                0x00,
                (word >> 8) & 0xFF,
                word & 0xFF,
                0x00,
                0x00,
                word_count,
            ]
        ) + bytes(payload)
        self._send(fins_cmd)

    def read_word(self, address: str) -> int:
        word, _ = self._parse_address(address)
        fins_cmd = bytes(
            [
                0x80,
                0x00,
                0x01,
                0x01,
                0x01,
                0x30,
                0x00,
                0x00,
                (word >> 8) & 0xFF,
                word & 0xFF,
                0x00,
                0x00,
                0x00,
                0x01,
            ]
        )
        resp = self._send(fins_cmd)
        if len(resp) < 2:
            raise PLCError("Invalid word read response")
        return int(struct.unpack(">H", resp[-2:])[0])

    def write_word(self, address: str, value: int) -> None:
        word, _ = self._parse_address(address)
        payload = struct.pack(">H", int(value) & 0xFFFF)
        fins_cmd = bytes(
            [
                0x80,
                0x00,
                0x02,
                0x00,
                0x01,
                0x30,
                0x00,
                0x00,
                (word >> 8) & 0xFF,
                word & 0xFF,
                0x00,
                0x00,
                0x01,
            ]
        ) + payload
        self._send(fins_cmd)


class AsciiTcpClient(BasePLCClient):
    """Simple ASCII TCP client using 'RD <addr>\r' and 'WR <addr> <val>\r'.

    Interprets any non-zero read value as True. For writing result arrays,
    it packs flags into 16-bit words starting at the configured word.
    Example: start 'EM7901' -> flags 1..16 map to EM7901 bits 00..15.
    """

    def __init__(self, config: PlcConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._addr_re = re.compile(r"^([A-Za-z]+)(\d+)$")

    def connect(self) -> None:
        self._sock = socket.create_connection((self._config.ip, self._config.port), timeout=self._config.timeouts.connect_ms / 1000.0)
        self._sock.settimeout(max(0.001, self._config.timeouts.response_ms / 1000.0))
        LOGGER.info("Connected (ASCII TCP) to PLC at %s:%s", self._config.ip, self._config.port)

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send_cmd(self, cmd: str) -> str:
        if self._sock is None:
            raise PLCError("PLC not connected")
        data = (cmd + "\r").encode("utf-8")
        self._sock.sendall(data)
        try:
            resp = self._sock.recv(1024)
        except socket.timeout as exc:
            raise PLCError(f"Timeout waiting for PLC response to {cmd}") from exc
        try:
            decoded = resp.decode("utf-8").strip()
        except Exception as exc:
            raise PLCError(f"Invalid ASCII PLC response: {resp!r}") from exc
        if self._config.log_raw_response:
            LOGGER.info("PLC raw response for %s: %r", cmd, decoded)
        return decoded

    def read_bit(self, address: str) -> bool:
        resp = self._send_cmd(f"RD {address}")
        tokens = [token for token in resp.replace("\r", " ").split() if token]
        if not tokens:
            raise PLCError(f"Empty response from {address}")
        try:
            return int(tokens[0]) != 0
        except ValueError:
            upper = tokens[0].upper()
            if upper == "OK" and len(tokens) > 1:
                try:
                    return int(tokens[1]) != 0
                except ValueError as exc:
                    raise PLCError(f"Non-integer read from {address}: {resp}") from exc
            if upper == "OK":
                LOGGER.warning("PLC returned OK without data for %s: %s", address, resp)
                return False
            raise PLCError(f"Non-integer read from {address}: {resp}")

    def write_bit(self, address: str, value: bool) -> None:
        self._send_cmd(f"WR {address} {1 if value else 0}")

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:
        """Write result flags packed into 16-bit words.

        Mapping rule (little-endian bit order per word):
        - Flag i (1-based) maps to bit ((i-1) % 16) of word (base + (i-1)//16)

        Example with start_word EM7901:
        - flags 1..16  -> EM7901.00 .. EM7901.15
        - flags 17..32 -> EM7902.00 .. EM7902.15

        Unused bits in the final partial word are written as 0.
        """
        m = self._addr_re.match(start_word)
        if not m:
            raise PLCError(f"Unsupported ASCII start address: {start_word}")
        prefix, base = m.group(1), int(m.group(2))
        total = len(bits)
        if total == 0:
            return

        word_count = (total + 15) // 16
        for word_idx in range(word_count):
            packed = 0
            base_idx = word_idx * 16
            for bit_idx in range(16):
                src_idx = base_idx + bit_idx
                if src_idx >= total:
                    break
                if bits[src_idx]:
                    packed |= 1 << bit_idx
            addr = f"{prefix}{base + word_idx}"
            self.write_word(addr, packed)

    def read_word(self, address: str) -> int:
        resp = self._send_cmd(f"RD {address}")
        tokens = [token for token in resp.replace("\r", " ").split() if token]
        if not tokens:
            raise PLCError(f"Empty response from {address}")
        try:
            return int(tokens[0])
        except ValueError:
            upper = tokens[0].upper()
            if upper == "OK" and len(tokens) > 1:
                try:
                    return int(tokens[1])
                except ValueError as exc:
                    raise PLCError(f"Non-integer read from {address}: {resp}") from exc
            raise PLCError(f"Non-integer read from {address}: {resp}")

    def write_word(self, address: str, value: int) -> None:
        self._send_cmd(f"WR {address} {int(value)}")


class PlcController:
    """High level handshake manager."""

    def __init__(self, config: PlcConfig, client: Optional[BasePLCClient] = None) -> None:
        self.config = config
        if client is not None:
            self.client = client
        else:
            if config.protocol == "FINS_TCP":
                self.client = OmronFinsTcpClient(config)
            elif config.protocol == "ASCII_TCP":
                self.client = AsciiTcpClient(config)
            else:
                LOGGER.warning("Unknown PLC protocol %s; using mock", config.protocol)
                self.client = MockPLCClient()
        self.state = PlcHandshakeState()
        self._lock = threading.RLock()
        self._io_failures = 0
        self._stable_recover_success = 0
        self._connection_state = PlcConnectionState.DEGRADED_MOCK if isinstance(self.client, MockPLCClient) else PlcConnectionState.REAL_CONNECTED
        self._reconnect_stop = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        with self._lock:
            self.client.connect()
            self._io_failures = 0
            self._stable_recover_success = 0
            self._connection_state = PlcConnectionState.DEGRADED_MOCK if isinstance(self.client, MockPLCClient) else PlcConnectionState.REAL_CONNECTED

    def close(self) -> None:
        self.stop_auto_reconnect()
        with self._lock:
            self.client.close()

    def connection_state(self) -> str:
        with self._lock:
            return self._connection_state.value

    def is_mock_mode(self) -> bool:
        with self._lock:
            return isinstance(self.client, MockPLCClient)

    def can_process_plc_trigger(self) -> bool:
        with self._lock:
            if not isinstance(self.client, MockPLCClient):
                return True
            return bool(self.config.reconnect_allow_plc_trigger_in_mock)

    def _build_real_client_no_lock(self) -> BasePLCClient:
        if self.config.protocol == "FINS_TCP":
            return OmronFinsTcpClient(self.config)
        if self.config.protocol == "ASCII_TCP":
            return AsciiTcpClient(self.config)
        raise PLCError(f"Unsupported protocol for reconnect: {self.config.protocol}")

    def _switch_to_mock_no_lock(self, reason: str) -> None:
        if isinstance(self.client, MockPLCClient):
            self._connection_state = PlcConnectionState.DEGRADED_MOCK
            return
        old = self.client
        self.client = MockPLCClient()
        try:
            old.close()
        except Exception:
            pass
        try:
            self.client.connect()
        except Exception:
            pass
        self._connection_state = PlcConnectionState.DEGRADED_MOCK
        LOGGER.warning("PLC switched to mock mode: %s", reason)

    def swap_client(self, new_client: BasePLCClient, mark_state: PlcConnectionState = PlcConnectionState.REAL_CONNECTED) -> None:
        with self._lock:
            old = self.client
            self.client = new_client
            try:
                old.close()
            except Exception:
                pass
            self._connection_state = mark_state
            self._io_failures = 0

    def _handle_io_failure_no_lock(self, exc: Exception) -> None:
        self._io_failures += 1
        threshold = max(1, int(self.config.reconnect_consecutive_fail_to_degraded))
        if not isinstance(self.client, MockPLCClient) and self.config.reconnect_enabled and self._io_failures >= threshold:
            self._switch_to_mock_no_lock(str(exc))
            self._ensure_reconnect_thread_no_lock()

    def _run_with_io_recovery(self, action):
        with self._lock:
            try:
                result = action()
                self._io_failures = 0
                return result
            except Exception as exc:
                self._handle_io_failure_no_lock(exc)
                raise

    def _initialize_outputs_after_recover(self) -> None:
        self.set_busy(False)
        self.set_done(False)
        self.set_error(False)
        self.set_ready(True)
        self.set_run(False)

    def _ensure_reconnect_thread_no_lock(self) -> None:
        if not self.config.reconnect_enabled:
            return
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._reconnect_stop.clear()
        self._connection_state = PlcConnectionState.RECONNECTING_REAL
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True, name="plc-reconnect")
        self._reconnect_thread.start()

    def start_auto_reconnect(self) -> None:
        with self._lock:
            if isinstance(self.client, MockPLCClient):
                self._ensure_reconnect_thread_no_lock()

    def stop_auto_reconnect(self) -> None:
        self._reconnect_stop.set()
        thread = self._reconnect_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _reconnect_loop(self) -> None:
        interval_ms = max(200, int(self.config.reconnect_interval_ms))
        max_ms = max(interval_ms, int(self.config.reconnect_max_interval_ms))
        factor = max(1.0, float(self.config.reconnect_backoff_multiplier))
        stable_required = max(1, int(self.config.reconnect_recover_stable_success_count))
        keep_mock_until_stable = bool(self.config.reconnect_keep_mock_until_real_stable)
        while not self._reconnect_stop.wait(interval_ms / 1000.0):
            try:
                candidate = self._build_real_client_no_lock()
                candidate.connect()
                # Lightweight health check read before swapping in.
                candidate.read_bit(self.config.addr.trigger)
            except Exception as exc:
                LOGGER.warning("PLC reconnect attempt failed: %s", exc)
                interval_ms = min(max_ms, int(interval_ms * factor))
                with self._lock:
                    self._connection_state = PlcConnectionState.RECONNECTING_REAL
                    self._stable_recover_success = 0
                continue

            with self._lock:
                try:
                    if keep_mock_until_stable and self._stable_recover_success + 1 < stable_required:
                        self._stable_recover_success += 1
                        self._connection_state = PlcConnectionState.RECONNECTING_REAL
                        LOGGER.info(
                            "PLC reconnect health-check passed (%d/%d)",
                            self._stable_recover_success,
                            stable_required,
                        )
                        try:
                            candidate.close()
                        except Exception:
                            pass
                        interval_ms = max(200, int(self.config.reconnect_interval_ms))
                        continue
                    self.swap_client(candidate, mark_state=PlcConnectionState.RECONNECTING_REAL)
                    self._initialize_outputs_after_recover()
                    self._stable_recover_success += 1
                    if self._stable_recover_success >= stable_required:
                        self._connection_state = PlcConnectionState.REAL_CONNECTED
                        self._stable_recover_success = 0
                        LOGGER.info("PLC reconnect recovered to real client")
                        interval_ms = max(200, int(self.config.reconnect_interval_ms))
                    else:
                        self._connection_state = PlcConnectionState.RECONNECTING_REAL
                except Exception as exc:
                    LOGGER.warning("PLC recover handover failed: %s", exc)
                    self._switch_to_mock_no_lock(str(exc))
                    self._connection_state = PlcConnectionState.RECONNECTING_REAL
                    interval_ms = min(max_ms, int(interval_ms * factor))

    def set_busy(self, value: bool) -> None:
        with self._lock:
            self._run_with_io_recovery(lambda: self.client.write_bit(self.config.addr.busy, value))
            self.state.busy = value
            if value:
                self.state.last_cycle_started = time.time()
                if self.state.ready:
                    self._set_ready_no_lock(False)

    def set_done(self, value: bool) -> None:
        with self._lock:
            self._run_with_io_recovery(lambda: self.client.write_bit(self.config.addr.done, value))
            self.state.done = value

    def set_error(self, value: bool) -> None:
        with self._lock:
            self._run_with_io_recovery(lambda: self.client.write_bit(self.config.addr.error, value))
            self.state.error = value
            if value and self.state.ready:
                self._set_ready_no_lock(False)

    def _set_ready_no_lock(self, value: bool) -> None:
        self._run_with_io_recovery(lambda: self.client.write_bit(self.config.addr.ready, value))
        self.state.ready = value

    def set_ready(self, value: bool) -> None:
        with self._lock:
            self._set_ready_no_lock(value)

    def set_run(self, value: bool) -> None:
        with self._lock:
            self._run_with_io_recovery(lambda: self.client.write_bit(self.config.addr.run, value))
            self.state.run = value

    def write_results(self, results: Sequence[bool]) -> None:
        with self._lock:
<<<<<<< codex/find-automatic-reconnect-feature-cyekq5
            self._run_with_io_recovery(lambda: self.client.write_result_bits(self.config.addr.result_bits_start_word, results))
=======
            self.client.write_result_bits(self.config.addr.result_bits_start_word, results)
>>>>>>> main
            self.state.last_results = list(results)

    def read_bit(self, address: str) -> bool:
        with self._lock:
<<<<<<< codex/find-automatic-reconnect-feature-cyekq5
            return bool(self._run_with_io_recovery(lambda: self.client.read_bit(address)))

    def read_word(self, address: str) -> int:
        with self._lock:
            return int(self._run_with_io_recovery(lambda: self.client.read_word(address)))

    def write_word(self, address: str, value: int) -> None:
        with self._lock:
            self._run_with_io_recovery(lambda: self.client.write_word(address, int(value)))
=======
            return self.client.read_bit(address)

    def read_word(self, address: str) -> int:
        with self._lock:
            return self.client.read_word(address)

    def write_word(self, address: str, value: int) -> None:
        with self._lock:
            self.client.write_word(address, int(value))
>>>>>>> main

    def wait_for_trigger(self, poll_interval: float = 0.05) -> bool:
        start = time.time()
        while True:
            if self.read_bit(self.config.addr.trigger):
                LOGGER.debug("PLC trigger detected")
                return True
            if self.state.last_cycle_started and (time.time() - self.state.last_cycle_started) * 1000 > self.config.timeouts.cycle_ms:
                LOGGER.warning("Cycle timeout exceeded while waiting for trigger")
                return False
            time.sleep(poll_interval)

    def wait_for_ack_clear(self, poll_interval: float = 0.05, timeout_ms: Optional[int] = None) -> None:
        start = time.time()
        timeout = self.config.timeouts.ack_clear_ms if timeout_ms is None else int(timeout_ms)
        while self.read_bit(self.config.addr.ack):
            if timeout > 0 and (time.time() - start) * 1000 > timeout:
                raise PLCError("Timeout waiting for PLC ACK clear")
            time.sleep(poll_interval)

    def finalize_cycle(self) -> None:
        # Wait for PLC acknowledgement before clearing flags
        start = time.time()
        try:
            while not self.read_bit(self.config.addr.ack):
                if (time.time() - start) * 1000 > self.config.timeouts.cycle_ms:
                    LOGGER.warning("Timeout waiting for PLC ACK signal")
                    break
                time.sleep(0.05)
        except PLCError as exc:
            LOGGER.warning("PLC ACK wait failed: %s", exc)
        self.set_done(False)
        self.set_busy(False)
        if self.state.error:
            self.set_error(False)
        try:
            self.wait_for_ack_clear()
        except PLCError as exc:
            LOGGER.warning("PLC ACK clear wait failed: %s", exc)
        self.set_ready(True)

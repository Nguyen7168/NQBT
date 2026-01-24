"""PLC communication helpers supporting Omron FINS/TCP."""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
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
    last_cycle_started: Optional[float] = None


class BasePLCClient:
    """Abstract PLC client interface."""

    def connect(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_bit(self, address: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def write_bit(self, address: str, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:  # pragma: no cover
        raise NotImplementedError


class MockPLCClient(BasePLCClient):
    """In-memory PLC client useful for development."""

    def __init__(self) -> None:
        self._bits: dict[str, bool] = {}

    def connect(self) -> None:  # pragma: no cover
        LOGGER.info("Mock PLC connected")

    def close(self) -> None:  # pragma: no cover
        LOGGER.info("Mock PLC disconnected")

    def read_bit(self, address: str) -> bool:
        return self._bits.get(address, False)

    def write_bit(self, address: str, value: bool) -> None:
        LOGGER.debug("PLC bit %s <- %s", address, value)
        self._bits[address] = value

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
        self._sock.settimeout(1.0)
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
        area = address[0]
        if area not in {"W", "D"}:
            raise PLCError(f"Unsupported memory area in address {address}")
        word = int(address[1:].split(".")[0])
        bit = int(address.split(".")[1]) if "." in address else 0
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


class AsciiTcpClient(BasePLCClient):
    """Simple ASCII TCP client using 'RD <addr>\r' and 'WR <addr> <val>\r'.

    Interprets any non-zero read value as True. For writing result arrays,
    it writes sequential registers by incrementing the numeric suffix.
    Example: start 'R160' -> writes R160, R161, ... as 1/0.
    """

    def __init__(self, config: PlcConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._addr_re = re.compile(r"^([A-Za-z]+)(\d+)$")

    def connect(self) -> None:
        self._sock = socket.create_connection((self._config.ip, self._config.port), timeout=self._config.timeouts.connect_ms / 1000.0)
        self._sock.settimeout(1.0)
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
            return resp.decode("utf-8").strip()
        except Exception as exc:
            raise PLCError(f"Invalid ASCII PLC response: {resp!r}") from exc

    def read_bit(self, address: str) -> bool:
        resp = self._send_cmd(f"RD {address}")
        try:
            tokens = [token for token in resp.replace("\r", " ").split() if token]
            if not tokens:
                raise ValueError("empty response")
            return int(tokens[0]) != 0
        except ValueError as exc:
            raise PLCError(f"Non-integer read from {address}: {resp}") from exc

    def write_bit(self, address: str, value: bool) -> None:
        self._send_cmd(f"WR {address} {1 if value else 0}")

    def write_result_bits(self, start_word: str, bits: Sequence[bool]) -> None:
        m = self._addr_re.match(start_word)
        if not m:
            raise PLCError(f"Unsupported ASCII start address: {start_word}")
        prefix, base = m.group(1), int(m.group(2))
        for idx, bit in enumerate(bits):
            addr = f"{prefix}{base + idx}"
            self.write_bit(addr, bit)


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
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.client.connect()

    def close(self) -> None:
        self.client.close()

    def set_busy(self, value: bool) -> None:
        with self._lock:
            self.client.write_bit(self.config.addr.busy, value)
            self.state.busy = value
            if value:
                self.state.last_cycle_started = time.time()

    def set_done(self, value: bool) -> None:
        with self._lock:
            self.client.write_bit(self.config.addr.done, value)
            self.state.done = value

    def set_error(self, value: bool) -> None:
        with self._lock:
            self.client.write_bit(self.config.addr.error, value)
            self.state.error = value

    def write_results(self, results: Sequence[bool]) -> None:
        self.client.write_result_bits(self.config.addr.result_bits_start_word, results)

    def wait_for_trigger(self, poll_interval: float = 0.05) -> bool:
        start = time.time()
        while True:
            if self.client.read_bit(self.config.addr.trigger):
                LOGGER.debug("PLC trigger detected")
                return True
            if self.state.last_cycle_started and (time.time() - self.state.last_cycle_started) * 1000 > self.config.timeouts.cycle_ms:
                LOGGER.warning("Cycle timeout exceeded while waiting for trigger")
                return False
            time.sleep(poll_interval)

    def wait_for_ack_clear(self, poll_interval: float = 0.05) -> None:
        while self.client.read_bit(self.config.addr.ack):
            time.sleep(poll_interval)

    def finalize_cycle(self) -> None:
        # Wait for PLC acknowledgement before clearing flags
        start = time.time()
        try:
            while not self.client.read_bit(self.config.addr.ack):
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

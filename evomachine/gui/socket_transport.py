from __future__ import annotations

import json
import queue
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from evomachine.gui.protocol import GuiRequest, GuiResponse, response_from_exception


HEADER_SIZE = 4
MAX_PACKET_SIZE = 512 * 1024 * 1024


class GuiRequestHandler(Protocol):
    def handle(self, request: GuiRequest) -> GuiResponse:
        """Handle one parsed GUI request."""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send one length-prefixed JSON message."""
    payload = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(payload) > MAX_PACKET_SIZE:
        raise ValueError(f"packet is too large: {len(payload)} bytes")
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def receive_packet(sock: socket.socket) -> dict[str, Any]:
    """Receive one length-prefixed JSON message."""
    header = _recv_exact(sock, HEADER_SIZE)
    (payload_size,) = struct.unpack("!I", header)
    if payload_size > MAX_PACKET_SIZE:
        raise ValueError(f"packet is too large: {payload_size} bytes")
    payload = _recv_exact(sock, payload_size)
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"packet payload must decode to dict, received {type(data)}")
    return data


@dataclass
class RpcJob:
    request: GuiRequest
    response_queue: "queue.Queue[GuiResponse]" = field(default_factory=queue.Queue)


class GuiRpcServer:
    """TCP server that queues GUI requests for bounded automaton-thread handling."""

    def __init__(
            self,
            handler: GuiRequestHandler,
            host: str = "127.0.0.1",
            port: int = 0,
            accept_timeout: float = 0.1,
            response_timeout: float = 30.0,
    ):
        self.handler = handler
        self.host = host
        self.port = port
        self.accept_timeout = accept_timeout
        self.response_timeout = response_timeout
        self._jobs: "queue.Queue[RpcJob]" = queue.Queue()
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._client_threads: list[threading.Thread] = []

    @property
    def bound_address(self) -> tuple[str, int]:
        if self._server_socket is None:
            return self.host, self.port
        host, port = self._server_socket.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server_socket is not None:
            return self.bound_address
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        server_socket.settimeout(self.accept_timeout)
        self._server_socket = server_socket
        self._accept_thread = threading.Thread(target=self._accept_loop, name="GuiRpcServer", daemon=True)
        self._accept_thread.start()
        return self.bound_address

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
        for thread in self._client_threads:
            thread.join(timeout=0.2)

    def process_pending(self, max_jobs: int) -> int:
        if not isinstance(max_jobs, int) or max_jobs < 1:
            raise ValueError(f"GuiRpcServer.process_pending: max_jobs must be positive int, received {max_jobs}.")
        processed = 0
        for _ in range(max_jobs):
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            try:
                response = self.handler.handle(job.request)
            except Exception as error:
                response = response_from_exception(request_id=job.request.request_id, error=error)
            job.response_queue.put(response)
            processed += 1
        return processed

    def _accept_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                client_socket, _address = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._client_loop,
                args=(client_socket,),
                name="GuiRpcClient",
                daemon=True,
            )
            self._client_threads.append(thread)
            thread.start()

    def _client_loop(self, client_socket: socket.socket) -> None:
        with client_socket:
            while not self._stop_event.is_set():
                try:
                    request = GuiRequest.from_dict(receive_packet(client_socket))
                    job = RpcJob(request=request)
                    self._jobs.put(job)
                    response = job.response_queue.get(timeout=self.response_timeout)
                    send_packet(client_socket, response.to_dict())
                except (ConnectionError, socket.timeout):
                    break
                except Exception as error:
                    request_id = request.request_id if "request" in locals() else "unknown"
                    response = response_from_exception(request_id=request_id, error=error)
                    try:
                        send_packet(client_socket, response.to_dict())
                    except Exception:
                        break


class GuiSocketClient:
    """Small synchronous TCP client used by tests and the Qt worker."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def request(self, command, payload: dict[str, Any] | None = None) -> GuiResponse:
        return self.request_object(GuiRequest(command=command, payload={} if payload is None else payload))

    def request_object(self, request: GuiRequest) -> GuiResponse:
        self.connect()
        assert self._socket is not None
        self._socket.settimeout(self.timeout)
        try:
            send_packet(self._socket, request.to_dict())
            response = GuiResponse.from_dict(receive_packet(self._socket))
        except Exception:
            self.close()
            raise
        finally:
            if self._socket is not None:
                self._socket.settimeout(self.timeout)
        if response.request_id != request.request_id:
            raise RuntimeError(
                f"GuiSocketClient: response ID {response.request_id} does not match request ID {request.request_id}."
            )
        return response

    def __enter__(self) -> "GuiSocketClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

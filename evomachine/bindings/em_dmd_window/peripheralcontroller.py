from __future__ import annotations

import logging
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import numpy as np

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.peripherals.peripherals import SocketPeripheralController

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 12345
MAX_BYTE_SIZE = 65482
NUM_CHUNKS = 97
CHUNK_ROWS = int(DMD_WIDTH_HEIGHT[0] / NUM_CHUNKS)
EM_DMD_PROGRAM_PATH = Path(__file__).resolve().parents[4] / "em_dmd_window/Release/evomachine_dmd_window"


class FakeSocket:
    """Socket-like object that records bytes sent by DMD tests."""

    def __init__(self):
        """
        Initialise fake socket state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.closed = False
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        """Record bytes sent through the fake socket."""
        self.sent.append(data)

    def close(self) -> None:
        """Mark the fake socket as closed."""
        self.closed = True


class EmDmdWindowPeripheralController(SocketPeripheralController):
    """Peripheral controller for the socket-backed em_dmd_window DMD."""

    DEFAULT_NAME: str = "em_dmd_window DMD Peripheral Controller"

    def __init__(
            self,
            name: str = DEFAULT_NAME,
            close_on_shutdown: bool = True,
            debug_mode: bool = False,
            host: str = HOST,
            port: int = PORT,
            program_path: Path | None = None,
            socket_obj: Any | None = None,
            process: subprocess.Popen | None = None,
            launch_process: bool | None = None,
            connect_socket: bool | None = None,
    ):
        self.debug_mode: bool = debug_mode
        self.host: str = host
        self.port: int = port
        self.program_path: Path = program_path or EM_DMD_PROGRAM_PATH
        self.launch_process: bool = socket_obj is None if launch_process is None else launch_process
        self.connect_socket: bool = socket_obj is None if connect_socket is None else connect_socket
        self.s: Any = socket_obj or socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._process: subprocess.Popen | None = process
        self.errors: list[str] = []
        super().__init__(name=name, close_on_shutdown=close_on_shutdown)

    @classmethod
    def from_default(
            cls,
            name: str = DEFAULT_NAME,
            close_on_shutdown: bool = True,
            **dmd_options: Any,
    ) -> EmDmdWindowPeripheralController:
        """Create a socket-backed DMD peripheral controller."""
        return cls(name=name, close_on_shutdown=close_on_shutdown, **dmd_options)

    def send_image(self, img: np.ndarray) -> None:
        """Send a row-major DMD image to the C window process."""
        if self.debug_mode:
            return
        self.s.sendall(img.transpose().tobytes())

    def _get_socket_controller(self) -> EmDmdWindowPeripheralController:
        return self

    def _launch_dmd_window(self) -> None:
        if self.debug_mode or not self.launch_process:
            return
        self._process = subprocess.Popen(
            [str(self.program_path.resolve())],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(1)

    def _connect_socket(self) -> None:
        if self.debug_mode or not self.connect_socket:
            return
        try:
            self.s.connect((self.host, self.port))
        except OSError as error:
            logger.info(f"Received error {error} on opening socket. Retrying once.")
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.host, self.port))

    def _connection_test(self) -> bool:
        if self.debug_mode:
            return True
        try:
            test_arr = np.zeros(DMD_WIDTH_HEIGHT, dtype=np.uint8)
            for i in range(DMD_WIDTH_HEIGHT[0]):
                test_arr[i, :] = i % 255
            self.s.sendall(test_arr.tobytes())
            return True
        except ConnectionResetError as error:
            msg = f"Error connection test: {error}"
            logger.error(msg)
            self.errors.append(msg)
            return False

    def _initialise(self, force: bool = False) -> bool:
        logger.info("EmDmdWindowPeripheralController.initialise: initialising DMD socket backend.")
        if self.debug_mode:
            return True
        try:
            self._launch_dmd_window()
            self._connect_socket()
            return self._connection_test()
        except Exception as error:
            msg = f"Error initialising DMD socket backend: {error}"
            logger.error(msg)
            self.errors.append(msg)
            return False

    def _check_is_alive(self) -> bool:
        return self._is_initialised

    def _stop(self) -> None:
        return

    def _before_disconnect(self, force: bool = False) -> None:
        self._is_initialised = False

    def _disconnect(self) -> None:
        close = getattr(self.s, "close", None)
        if callable(close):
            close()
        time.sleep(1)
        if self._process is not None and self._process.poll() != 0:
            logger.warning("EmDmdWindowPeripheralController: Forcing C program shutdown.")
            self._process.terminate()

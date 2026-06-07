from __future__ import annotations

import os
from typing import Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from evomachine.gui.protocol import GUI_HOST_ENV as HOST_ENV
from evomachine.gui.protocol import GUI_PORT_ENV as PORT_ENV
from evomachine.gui.protocol import GuiCommandType, GuiRequest, GuiResponse
from evomachine.gui.socket_transport import GuiSocketClient


class RpcClientWorker(QObject):
    """Qt worker that performs blocking socket requests off the GUI thread."""

    response_ready = pyqtSignal(object)
    request_failed = pyqtSignal(str)

    def __init__(self, client: GuiSocketClient):
        super().__init__()
        self.client = client

    @pyqtSlot(object)
    def send_request(self, request: GuiRequest) -> None:
        try:
            self.response_ready.emit(self.client.request_object(request))
        except Exception as error:
            self.request_failed.emit(f"{type(error).__name__}: {error}")

    @pyqtSlot()
    def close(self) -> None:
        self.client.close()


class EvoMachineGuiController(QObject):
    """Qt-facing controller used by the Napari panels."""

    request_ready = pyqtSignal(object)
    response_error = pyqtSignal(str)
    stage_status_received = pyqtSignal(dict)
    stage_coordinates_received = pyqtSignal(dict)
    led_list_received = pyqtSignal(list)
    led_state_received = pyqtSignal(dict)
    lifecycle_status_received = pyqtSignal(dict)

    def __init__(
            self,
            host: str | None = None,
            port: int | None = None,
            client: GuiSocketClient | None = None,
            start_worker: bool = True,
    ):
        super().__init__()
        resolved_host = host or os.environ.get(HOST_ENV, "127.0.0.1")
        resolved_port = port if port is not None else int(os.environ.get(PORT_ENV, "0"))
        self.client = client if client is not None else GuiSocketClient(host=resolved_host, port=resolved_port)
        self._thread: QThread | None = None
        self._worker: RpcClientWorker | None = None
        if start_worker:
            self._thread = QThread()
            self._worker = RpcClientWorker(client=self.client)
            self._worker.moveToThread(self._thread)
            self.request_ready.connect(self._worker.send_request)
            self._worker.response_ready.connect(self._handle_response)
            self._worker.request_failed.connect(self.response_error.emit)
            self._thread.start()

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
        else:
            self.client.close()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(1000)

    def ping(self) -> None:
        self._send(GuiCommandType.PING)

    def initialise_devices(self) -> None:
        self._send(GuiCommandType.INITIALISE_DEVICES)

    def stop(self) -> None:
        self._send(GuiCommandType.STOP)

    def shutdown_automaton(self) -> None:
        self._send(GuiCommandType.SHUTDOWN)

    def refresh_stage(self) -> None:
        self._send(GuiCommandType.STAGE_GET_COORDINATES)

    def move_stage_absolute(self, x: float | None, y: float | None, z: float | None) -> None:
        self._send(GuiCommandType.STAGE_MOVE_ABSOLUTE, {"x": x, "y": y, "z": z})

    def move_stage_relative(self, dx: float | None, dy: float | None, dz: float | None) -> None:
        self._send(GuiCommandType.STAGE_MOVE_RELATIVE, {"dx": dx, "dy": dy, "dz": dz})

    def stop_stage(self) -> None:
        self._send(GuiCommandType.STAGE_STOP)

    def refresh_leds(self) -> None:
        self._send(GuiCommandType.LED_LIST)

    def set_led(self, led: str, brightness: float, duration: float | None = None) -> None:
        self._send(GuiCommandType.LED_SET, {"led": led, "brightness": brightness, "duration": duration})

    def disable_led(self, led: str) -> None:
        self._send(GuiCommandType.LED_DISABLE, {"led": led})

    def disable_all_leds(self) -> None:
        self._send(GuiCommandType.LED_DISABLE_ALL)

    def _send(self, command: GuiCommandType, payload: dict[str, Any] | None = None) -> None:
        request = GuiRequest(command=command, payload={} if payload is None else payload)
        if self._worker is None:
            self._handle_response(self.client.request_object(request))
            return
        self.request_ready.emit(request)

    @pyqtSlot(object)
    def _handle_response(self, response: GuiResponse) -> None:
        if not response.ok:
            self.response_error.emit(response.error or "Unknown automaton RPC error.")
            return
        payload = response.payload
        if "coordinate" in payload:
            self.stage_coordinates_received.emit(payload)
        if "stage" in payload and "coordinate" not in payload:
            self.stage_status_received.emit(payload["stage"])
        if "leds" in payload:
            self.led_list_received.emit(payload["leds"])
        if "state" in payload:
            self.led_state_received.emit(payload["state"])
        if "devices_initialised" in payload or "shutdown" in payload or "strategy_active" in payload:
            self.lifecycle_status_received.emit(payload)

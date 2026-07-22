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
    controller_status_received = pyqtSignal(dict)
    fovs_received = pyqtSignal(list)
    stage_status_received = pyqtSignal(dict)
    stage_coordinates_received = pyqtSignal(dict)
    camera_status_received = pyqtSignal(dict)
    frame_received = pyqtSignal(dict)
    filter_wheel_status_received = pyqtSignal(dict)
    led_list_received = pyqtSignal(list)
    led_state_received = pyqtSignal(dict)
    dmd_status_received = pyqtSignal(dict)
    autofocus_status_received = pyqtSignal(dict)
    software_focus_status_received = pyqtSignal(dict)
    strategies_received = pyqtSignal(list)
    strategy_status_received = pyqtSignal(dict)
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

    def refresh_controller_status(self) -> None:
        self._send(GuiCommandType.CONTROLLER_STATUS)

    def initialise_fovs(self, fovs: list[dict[str, Any]], use_autofocus: bool = False) -> None:
        self._send(GuiCommandType.FOV_INITIALISE, {"fovs": fovs, "use_autofocus": use_autofocus})

    def refresh_stage(self) -> None:
        self._send(GuiCommandType.STAGE_GET_COORDINATES)

    def move_stage_absolute(self, x: float | None, y: float | None, z: float | None) -> None:
        self._send(GuiCommandType.STAGE_MOVE_ABSOLUTE, {"x": x, "y": y, "z": z})

    def move_stage_relative(self, dx: float | None, dy: float | None, dz: float | None) -> None:
        self._send(GuiCommandType.STAGE_MOVE_RELATIVE, {"dx": dx, "dy": dy, "dz": dz})

    def stop_stage(self) -> None:
        self._send(GuiCommandType.STAGE_STOP)

    def refresh_camera(self) -> None:
        self._send(GuiCommandType.CAMERA_STATUS)

    def set_camera_exposure(self, exposure: float) -> None:
        self._send(GuiCommandType.CAMERA_SET_EXPOSURE, {"exposure": exposure})

    def acquire_frame(self, payload: dict[str, Any] | None = None) -> None:
        self._send(GuiCommandType.ACQUISITION_TAKE_FRAME, payload)

    def acquire_z_stack(self, payload: dict[str, Any] | None = None) -> None:
        self._send(GuiCommandType.ACQUISITION_TAKE_Z_STACK, payload)

    def refresh_filter_wheel(self) -> None:
        self._send(GuiCommandType.FILTER_WHEEL_STATUS)

    def set_filter_wheel(self, filter_wheel: str) -> None:
        self._send(GuiCommandType.FILTER_WHEEL_SET, {"filter_wheel": filter_wheel})

    def refresh_leds(self) -> None:
        self._send(GuiCommandType.LED_LIST)

    def set_led(self, led: str, brightness: float, duration: float | None = None) -> None:
        self._send(GuiCommandType.LED_SET, {"led": led, "brightness": brightness, "duration": duration})

    def disable_led(self, led: str) -> None:
        self._send(GuiCommandType.LED_DISABLE, {"led": led})

    def disable_all_leds(self) -> None:
        self._send(GuiCommandType.LED_DISABLE_ALL)

    def refresh_dmd(self) -> None:
        self._send(GuiCommandType.DMD_STATUS)

    def display_dmd_pattern(self, pattern: str) -> None:
        self._send(GuiCommandType.DMD_DISPLAY_PATTERN, {"pattern": pattern})

    def calibrate_dmd(self) -> None:
        self._send(GuiCommandType.DMD_CALIBRATE)

    def refresh_autofocus(self) -> None:
        self._send(GuiCommandType.AUTOFOCUS_STATUS)

    def configure_autofocus(self, config: dict[str, Any] | None = None) -> None:
        payload = {} if config is None else {"config": config}
        self._send(GuiCommandType.AUTOFOCUS_CONFIGURE, payload)

    def initialise_autofocus(
            self,
            lock_after_initialise: bool = False,
            config: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"lock_after_initialise": lock_after_initialise}
        if config is not None:
            payload["config"] = config
        self._send(GuiCommandType.AUTOFOCUS_INITIALISE, payload)

    def lock_autofocus(self) -> None:
        self._send(GuiCommandType.AUTOFOCUS_LOCK)

    def unlock_autofocus(self) -> None:
        self._send(GuiCommandType.AUTOFOCUS_UNLOCK)

    def disable_autofocus(self) -> None:
        self._send(GuiCommandType.AUTOFOCUS_DISABLE)

    def refresh_software_focus(self) -> None:
        self._send(GuiCommandType.SOFTWARE_FOCUS_STATUS)

    def run_software_focus(self) -> None:
        self._send(GuiCommandType.SOFTWARE_FOCUS_RUN)

    def refresh_strategy_status(self) -> None:
        self._send(GuiCommandType.STRATEGY_STATUS)

    def refresh_strategies(self) -> None:
        self._send(GuiCommandType.STRATEGY_LIST)

    def set_strategy(self, name: str, file_path: str | None = None) -> None:
        payload: dict[str, Any] = {"name": name}
        if file_path is not None:
            payload["file_path"] = file_path
        self._send(GuiCommandType.STRATEGY_SET, payload)

    def start_strategy(self) -> None:
        self._send(GuiCommandType.STRATEGY_START)

    def stop_strategy(self) -> None:
        self._send(GuiCommandType.STRATEGY_STOP)

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
        if "controllers" in payload:
            self.controller_status_received.emit(payload)
        if "fovs" in payload:
            self.fovs_received.emit(payload["fovs"])
        if "stage" in payload and "coordinate" not in payload:
            self.stage_status_received.emit(payload["stage"])
        if "camera" in payload:
            self.camera_status_received.emit(payload["camera"])
        if "frame" in payload:
            self.frame_received.emit(payload["frame"])
        if "filter_wheel" in payload:
            self.filter_wheel_status_received.emit(payload["filter_wheel"])
        if "leds" in payload:
            self.led_list_received.emit(payload["leds"])
        if "state" in payload:
            self.led_state_received.emit(payload["state"])
        if "dmd" in payload:
            self.dmd_status_received.emit(payload["dmd"])
        if "autofocus" in payload:
            self.autofocus_status_received.emit(payload["autofocus"])
        if "software_focus" in payload:
            self.software_focus_status_received.emit(payload["software_focus"])
        if "strategies" in payload:
            self.strategies_received.emit(payload["strategies"])
        if "strategy" in payload:
            self.strategy_status_received.emit(payload["strategy"])
        if "devices_initialised" in payload or "shutdown" in payload or "strategy_active" in payload:
            self.lifecycle_status_received.emit(payload)

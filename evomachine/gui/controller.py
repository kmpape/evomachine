from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from evomachine.gui.image_payloads import (
    IMAGE_TRANSPORT_AUTO,
    IMAGE_TRANSPORT_ENV,
    IMAGE_TRANSPORT_SOCKET_TIFF,
    IMAGE_TRANSPORT_TEMP_TIFF,
    normalise_image_transport,
)
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
    acquisition_files_received = pyqtSignal(list)
    acquisition_directory_received = pyqtSignal(dict)
    acquisition_experiments_received = pyqtSignal(dict)
    frame_received = pyqtSignal(dict)
    filter_wheel_status_received = pyqtSignal(dict)
    led_list_received = pyqtSignal(list)
    led_state_received = pyqtSignal(dict)
    dmd_status_received = pyqtSignal(dict)
    dmd_calibration_points_received = pyqtSignal(dict)
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
        self._requested_image_transport = normalise_image_transport(os.environ.get(IMAGE_TRANSPORT_ENV))
        self._image_transport = (
            IMAGE_TRANSPORT_SOCKET_TIFF
            if self._requested_image_transport == IMAGE_TRANSPORT_AUTO
            else self._requested_image_transport
        )
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
        self._send(GuiCommandType.STAGE_MOVE_ABSOLUTE, {"x": x, "y": y, "z": z, "block": False})

    def move_stage_relative(self, dx: float | None, dy: float | None, dz: float | None) -> None:
        self._send(GuiCommandType.STAGE_MOVE_RELATIVE, {"dx": dx, "dy": dy, "dz": dz, "block": False})

    def move_stage_fov(self, direction: str, multiplier: float = 1.0) -> None:
        self._send(
            GuiCommandType.STAGE_MOVE_FOV,
            {"direction": direction, "multiplier": multiplier, "block": False},
        )

    def stop_stage(self) -> None:
        self._send(GuiCommandType.STAGE_STOP)

    def zero_stage(self) -> None:
        self._send(GuiCommandType.STAGE_ZERO)

    def return_stage_to_origin(self) -> None:
        self._send(GuiCommandType.STAGE_RETURN_ORIGIN)

    def refresh_camera(self) -> None:
        self._send(GuiCommandType.CAMERA_STATUS)

    def probe_image_transport(self) -> None:
        if self._requested_image_transport == IMAGE_TRANSPORT_AUTO:
            self._send(GuiCommandType.IMAGE_TRANSPORT_PROBE)

    def set_camera_exposure(self, exposure: float) -> None:
        self._send(GuiCommandType.CAMERA_SET_EXPOSURE, {"exposure": exposure})

    def refresh_acquisition_files(self) -> None:
        self._send(GuiCommandType.ACQUISITION_LIST_FILES)

    def refresh_acquisition_experiments(self) -> None:
        self._send(GuiCommandType.ACQUISITION_LIST_EXPERIMENTS)

    def create_acquisition_experiment(self, name: str) -> None:
        self._send(GuiCommandType.ACQUISITION_CREATE_EXPERIMENT, {"name": name})

    def select_acquisition_experiment(self, name: str) -> None:
        self._send(GuiCommandType.ACQUISITION_SELECT_EXPERIMENT, {"name": name})

    def load_acquisition_frame(self, filename: str, image_transport: str | None = None) -> None:
        payload: dict[str, Any] = {"filename": filename}
        if image_transport is not None:
            payload["image_transport"] = normalise_image_transport(image_transport)
        self._send(GuiCommandType.ACQUISITION_LOAD_FRAME, self._with_image_transport(payload))

    def acquire_frame(self, payload: dict[str, Any] | None = None) -> None:
        self._send(GuiCommandType.ACQUISITION_TAKE_FRAME, self._with_image_transport(payload))

    def acquire_z_stack(self, payload: dict[str, Any] | None = None) -> None:
        self._send(GuiCommandType.ACQUISITION_TAKE_Z_STACK, self._with_image_transport(payload))

    def refresh_filter_wheel(self) -> None:
        self._send(GuiCommandType.FILTER_WHEEL_STATUS)

    def set_filter_wheel(self, filter_wheel: str) -> None:
        self._send(GuiCommandType.FILTER_WHEEL_SET, {"filter_wheel": filter_wheel})

    def refresh_leds(self) -> None:
        self._send(GuiCommandType.LED_LIST)

    def refresh_led_state(self, led: str) -> None:
        self._send(GuiCommandType.LED_GET_STATE, {"led": led})

    def set_led(self, led: str, brightness: float, duration: float | None = None) -> None:
        self._send(GuiCommandType.LED_SET, {"led": led, "brightness": brightness, "duration": duration})

    def disable_led(self, led: str) -> None:
        self._send(GuiCommandType.LED_DISABLE, {"led": led})

    def disable_all_leds(self) -> None:
        self._send(GuiCommandType.LED_DISABLE_ALL)

    def refresh_dmd(self) -> None:
        self._send(GuiCommandType.DMD_STATUS)

    def display_dmd_pattern(
            self,
            pattern: str,
            config: dict[str, Any] | None = None,
            warp: bool = True,
    ) -> None:
        payload: dict[str, Any] = {"pattern": pattern, "warp": warp}
        if config is not None:
            payload["config"] = config
        self._send(GuiCommandType.DMD_DISPLAY_PATTERN, payload)

    def load_dmd_pattern(self, filename: str) -> None:
        self._send(GuiCommandType.DMD_LOAD_PATTERN, {"filename": filename})

    def display_loaded_dmd_pattern(self) -> None:
        self._send(GuiCommandType.DMD_DISPLAY_LOADED_PATTERN)

    def calibrate_dmd(self) -> None:
        self._send(GuiCommandType.DMD_CALIBRATE)

    def load_dmd_calibration(self, filename: str) -> None:
        self._send(GuiCommandType.DMD_LOAD_CALIBRATION, {"filename": filename})

    def request_dmd_calibration_points(self) -> None:
        self._send(GuiCommandType.DMD_CALIBRATION_POINTS)

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

    def _with_image_transport(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        updated = {} if payload is None else dict(payload)
        updated.setdefault("image_transport", self._image_transport)
        return updated

    @pyqtSlot(object)
    def _handle_response(self, response: GuiResponse) -> None:
        if not response.ok:
            self.response_error.emit(response.error or "Unknown automaton RPC error.")
            return
        payload = response.payload
        if "image_transport_probe" in payload:
            self._handle_image_transport_probe(payload["image_transport_probe"])
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
        if "acquisition_files" in payload:
            self.acquisition_files_received.emit(payload["acquisition_files"])
        if "acquisition_directory" in payload:
            self.acquisition_directory_received.emit({
                "directory": payload["acquisition_directory"],
                "experiment_root": payload.get("experiment_root"),
                "experiment_name": payload.get("experiment_name"),
            })
        if "experiments" in payload:
            self.acquisition_experiments_received.emit({
                "experiments": payload["experiments"],
                "active_experiment": payload.get("active_experiment"),
                "experiment_root": payload.get("experiment_root"),
            })
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
        if "dmd_calibration_points" in payload:
            self.dmd_calibration_points_received.emit(payload["dmd_calibration_points"])
        if "autofocus" in payload:
            self.autofocus_status_received.emit(payload["autofocus"])
        if "software_focus" in payload:
            self.software_focus_status_received.emit(payload["software_focus"])
        if "strategies" in payload:
            self.strategies_received.emit(payload["strategies"])
        if "strategy" in payload:
            self.strategy_status_received.emit(payload["strategy"])
        if (
            "devices_initialised" in payload
            or "shutdown" in payload
            or "strategy_active" in payload
            or "stopped" in payload
        ):
            self.lifecycle_status_received.emit(payload)

    def _handle_image_transport_probe(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        path = payload.get("path")
        token = payload.get("token")
        if not isinstance(path, str) or not isinstance(token, str):
            self._image_transport = IMAGE_TRANSPORT_SOCKET_TIFF
            return
        try:
            probe_path = Path(path)
            can_read = probe_path.read_text(encoding="ascii") == token
            probe_path.unlink(missing_ok=True)
        except Exception:
            can_read = False
        self._image_transport = IMAGE_TRANSPORT_TEMP_TIFF if can_read else IMAGE_TRANSPORT_SOCKET_TIFF

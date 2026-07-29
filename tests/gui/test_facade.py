from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from evomachine.bindings.binding_types import BindingType
from evomachine.coordinates import Coordinate
from evomachine.filemanager import FileManager, FileNameConfig
from evomachine.frame import Frame
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.peripherals.camera import CameraConfig, ImageConfigType, ObjectiveConfigType
from evomachine.types import AutoFocusStatusType, FilterWheelType, FocusCurveType, FocusStatusType, FovDirectionType, LEDType
from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.image_payloads import IMAGE_TRANSPORT_DIR_ENV, IMAGE_TRANSPORT_RAW, array_from_preview_payload
from evomachine.gui.protocol import GuiCommandType, GuiRequest


@dataclass
class FakeLedState:
    led_type: LEDType
    brightness: float = 0.0
    is_on: bool = False
    stop_time: float | None = None


class FakePeripheralController:
    def __init__(self, name="Fake Controller", initialised=True, alive=True):
        self.name = name
        self.initialised = initialised
        self.alive = alive
        self.initialise_count = 0
        self.shutdown_count = 0

    def initialise(self):
        self.initialise_count += 1
        self.initialised = True
        self.alive = True

    def shutdown(self):
        self.shutdown_count += 1
        self.initialised = False
        self.alive = False

    def is_initialised(self):
        return self.initialised

    def is_alive(self):
        return self.alive


class FakeStage:
    def __init__(self):
        self.coordinate = Coordinate(1, 2, 3)
        self.stop_count = 0

    def is_initialised(self):
        return True

    def is_alive(self):
        return True

    def get_fov_id(self):
        return 0

    def get_fov_step_size(self):
        return 100.0

    def get_coordinates(self, query_hardware=True):
        return self.coordinate.copy()

    def move(self, target, block=True):
        if isinstance(target, tuple):
            direction, multiplier = target
            step = self.get_fov_step_size() * float(multiplier)
            deltas = {
                FovDirectionType.UP: (0.0, -step),
                FovDirectionType.DOWN: (0.0, step),
                FovDirectionType.LEFT: (-step, 0.0),
                FovDirectionType.RIGHT: (step, 0.0),
            }
            dx, dy = deltas[direction]
            self.coordinate = Coordinate(self.coordinate.x + dx, self.coordinate.y + dy, self.coordinate.z)
            return
        self.coordinate = self.coordinate.merge(target)

    def stop(self):
        self.stop_count += 1


class FakeLedManager:
    def __init__(self):
        self.states = {LEDType.LED_450_NM: FakeLedState(LEDType.LED_450_NM)}
        self.disable_all_count = 0

    def get_available_leds(self):
        return list(self.states)

    def set_led(self, led_type, brightness=100.0, duration=None):
        self.states[led_type] = FakeLedState(led_type=led_type, brightness=brightness, is_on=brightness > 0, stop_time=duration)

    def disable_led(self, led_type=None):
        if led_type is None:
            self.disable_all_count += 1
            for led in list(self.states):
                self.states[led] = FakeLedState(led)
            return
        self.states[led_type] = FakeLedState(led_type)

    def get_led_state(self, led_type):
        return self.states[led_type]


class FakeCamera:
    def __init__(self):
        self.name = "Fake Camera"
        self.image = SimpleNamespace(shape=(48, 64), pxl_dtype="uint16")
        self.config = CameraConfig(
            binding=BindingType.VIRTUAL,
            image=ImageConfigType(pxl_horiz=64, pxl_vert=48, pxl_dtype=np.dtype("uint16")),
            objective_config=ObjectiveConfigType(na=0.95, mag=40),
        )
        self.default_exposure_time = 200
        self.readout_mode = None
        self.exposure = 200
        self.stop_count = 0

    def is_initialised(self):
        return True

    def is_alive(self):
        return True

    def get_exposure(self):
        return self.exposure

    def fov_size(self):
        return 7.8

    def set_exposure(self, exposure_time):
        self.exposure = exposure_time

    def stop(self):
        self.stop_count += 1


class FakeDmd:
    def __init__(self):
        self.name = "Fake DMD"
        self.width_height_DMD = (20, 10)
        self.width_height_CAM = (30, 30)
        self.calls = []
        self.full_display = False
        self.calibrated = False
        self.calibration_filename = Path("loaded_calibration.pkl")
        self._calib_data = SimpleNamespace(
            dmd_points=[(1, 2), (3, 4)],
            cam_points=[(5, 6), (7, 8)],
            path=self.calibration_filename,
        )
        self.image = None

    def is_initialised(self):
        return True

    def is_alive(self):
        return True

    def is_full_display(self):
        return self.full_display

    def is_calibrated(self):
        return self.calibrated

    def get_calibration_filename(self):
        return self.calibration_filename

    def calibrate_from_path(self, path):
        self.calibrated = True
        self.calibration_filename = path

    def get_zero_array(self):
        return np.zeros(self.width_height_DMD, dtype=np.uint8)

    def get_one_array(self):
        return np.ones(self.width_height_DMD, dtype=np.uint8) * 255

    def get_checkerboard(self):
        image = self.get_zero_array()
        image[::2, ::2] = 255
        image[1::2, 1::2] = 255
        return image

    def get_calibration_image(self):
        image = self.get_zero_array()
        image[:, image.shape[1] // 2] = 255
        image[image.shape[0] // 2, :] = 255
        return image

    def get_pattern(self, pattern, config=None):
        if pattern in {"empty", "clear"}:
            return self.get_zero_array()
        if pattern == "full":
            return self.get_one_array()
        if pattern == "checkerboard":
            return self.get_checkerboard()
        image = self.get_zero_array()
        image[image.shape[0] // 2, image.shape[1] // 2] = 255
        return image

    def display_image(self, image, _is_full_display=False):
        self.calls.append("display_image")
        self.image = image
        self.full_display = _is_full_display


class FakeFilterWheel:
    def __init__(self):
        self.name = "Fake Filter Wheel"
        self.current_filter = FilterWheelType.NO_FILTER
        self.available_filters = [
            FilterWheelType.FILTER_465nm,
            FilterWheelType.FILTER_527nm,
            FilterWheelType.NO_FILTER,
            FilterWheelType.BLOCKING,
        ]
        self.calls = []

    def is_initialised(self):
        return True

    def is_alive(self):
        return True

    def get_filter_wheel(self):
        return self.current_filter

    def get_available_filters(self):
        return list(self.available_filters)

    def set_filter_wheel(self, filter_type):
        self.calls.append(filter_type)
        self.current_filter = filter_type


class FakeAutofocus:
    def __init__(self):
        self.name = "Fake Autofocus"
        self.status = AutoFocusStatusType.IDLE
        self.locked = False
        self.calls = []

    def is_initialised(self):
        return True

    def is_alive(self):
        return True

    def get_status(self):
        return self.status

    def is_locked(self):
        return self.locked

    def apply_config(self, config=None):
        self.calls.append(("apply_config", config))
        return True

    def configure(self, config=None):
        return self.apply_config(config=config)

    def run_calibration(self, config=None, lock_after_calibration=False):
        self.calls.append(("run_calibration", config, lock_after_calibration))
        self.status = AutoFocusStatusType.READY
        if lock_after_calibration:
            self.lock()
        return True

    def initialise_autofocus(self, config=None, lock_after_initialise=False):
        return self.run_calibration(config=config, lock_after_calibration=lock_after_initialise)

    def lock(self):
        self.calls.append("lock")
        self.locked = True
        self.status = AutoFocusStatusType.IN_FOCUS

    def unlock(self):
        self.calls.append("unlock")
        self.locked = False
        self.status = AutoFocusStatusType.READY

    def disable(self):
        self.calls.append("disable")
        self.locked = False
        self.status = AutoFocusStatusType.IDLE


class FakeSoftwareFocus:
    def __init__(self):
        self.default_config = SimpleNamespace(
            rel_range=15,
            step_size=5,
            algorithm=SimpleNamespace(name="STEEL"),
        )
        self.calls = []

    def run(self, fov_id=None):
        self.calls.append(("run", fov_id))
        return SimpleNamespace(
            focus_status=FocusStatusType.IN_FOCUS,
            curve_status=FocusCurveType.HAS_GLOBAL_MAXIMUM,
            best_coordinate=Coordinate(1, 2, 4),
            previous_coordinate=Coordinate(1, 2, 3),
            z_coordinates=np.asarray([-5, 0, 5]),
        )


class FakeAcquisitionManager:
    def __init__(self, led_manager, camera, filter_wheel, dmd, file_manager=None):
        self.led_manager = led_manager
        self.camera = camera
        self.filter_wheel = filter_wheel
        self.dmd = dmd
        self.file_manager = file_manager
        self.calls = []

    def take_frame(self, frame_metadata, settings=None):
        self.calls.append((frame_metadata, settings))
        image = np.arange(12, dtype=np.uint16).reshape(3, 4)
        saved_paths = [Path("frame_0.tiff")] if getattr(settings, "save", False) else [None]
        return Frame(frame_metadata=[frame_metadata], array=np.stack([image]), saved_paths=saved_paths)

    def take_z_stack(self, frame_metadata, z_coordinates, settings=None):
        self.calls.append((frame_metadata, z_coordinates, settings))
        images = [
            np.full((3, 4), index, dtype=np.uint16)
            for index, _coordinate in enumerate(z_coordinates)
        ]
        saved_paths = [
            Path(f"z_stack_{index}.tiff") if getattr(settings, "save", False) else None
            for index, _coordinate in enumerate(z_coordinates)
        ]
        return Frame(
            frame_metadata=[frame_metadata for _coordinate in z_coordinates],
            array=np.stack(images),
            saved_paths=saved_paths,
        )


class FakeAutomaton:
    def __init__(
            self,
            with_stage: bool = True,
            with_led_manager: bool = True,
            with_camera: bool = True,
            with_filter_wheel: bool = True,
            with_dmd: bool = True,
            with_autofocus: bool = True,
            with_software_focus: bool = True,
            devices_initialised: bool = True,
            file_manager=None,
    ):
        stage = FakeStage()
        led_manager = FakeLedManager()
        camera = FakeCamera()
        filter_wheel = FakeFilterWheel()
        dmd = FakeDmd()
        autofocus = FakeAutofocus()
        software_focus = FakeSoftwareFocus()
        shared_controller = FakePeripheralController("Shared Controller")
        dmd_controller = FakePeripheralController("DMD Controller")
        for device in (stage, led_manager, camera, filter_wheel, autofocus):
            device.peripheral_ctrl = shared_controller
        dmd.peripheral_ctrl = dmd_controller
        acq_mngr_attrs = {}
        if with_led_manager:
            acq_mngr_attrs["led_manager"] = led_manager
        if with_camera:
            acq_mngr_attrs["camera"] = camera
        if with_filter_wheel:
            acq_mngr_attrs["filter_wheel"] = filter_wheel
        if with_dmd:
            acq_mngr_attrs["dmd"] = dmd
        focus_nav_attrs = {}
        if with_stage:
            focus_nav_attrs["stage"] = stage
        if with_autofocus:
            focus_nav_attrs["autofocus"] = autofocus
        if with_software_focus:
            focus_nav_attrs["software_focus"] = software_focus
        self.focus_nav = SimpleNamespace(**focus_nav_attrs)
        self.acq_mngr = FakeAcquisitionManager(
            led_manager=acq_mngr_attrs.get("led_manager"),
            camera=acq_mngr_attrs.get("camera"),
            filter_wheel=acq_mngr_attrs.get("filter_wheel"),
            dmd=acq_mngr_attrs.get("dmd"),
            file_manager=file_manager,
        )
        for name in ("led_manager", "camera", "filter_wheel", "dmd"):
            if name not in acq_mngr_attrs:
                delattr(self.acq_mngr, name)
        self.shutdown_count = 0
        self.devices_initialised = devices_initialised
        self.dmd_calibration_calls = []
        self._cfg = ImageProcessorConfigFactory.default_config(channels=[LEDType.LED_450_NM], channels_seg=[LEDType.LED_450_NM])
        self._strategy = None
        self._strategy_is_initialised = False
        self._fov_list_is_initialised = False
        self._fovs = {}
        self.next_commands = []
        self.last_commands = []
        self.strategy_started = False
        self.strategy_stopped = False
        self.automaton_stopped = False

    def strategy_has_started(self):
        return self.strategy_started

    def strategy_has_stopped(self):
        return self.strategy_stopped

    def has_shutdown(self):
        return self.shutdown_count > 0

    def shutdown(self):
        self.shutdown_count += 1

    def stop(self):
        self.automaton_stopped = True

    def stopped(self):
        return self.automaton_stopped

    def initialise_devices(self):
        self.devices_initialised = True
        return None

    def initialise(self, fovs, cropping_boxes=None, use_autofocus=False):
        self.initialise_devices()
        self._fovs = {fov_id: coordinate.copy() for fov_id, coordinate in fovs.items()}
        self._fov_list_is_initialised = True
        if self._strategy is not None:
            self._strategy_is_initialised = True

    def devices_is_initialised(self):
        return self.devices_initialised

    def set_strategy(self, strategy):
        self._strategy = strategy
        self._strategy_is_initialised = self._fov_list_is_initialised
        self.next_commands = []
        self.last_commands = []

    def start_strategy(self):
        if self._strategy is None:
            raise RuntimeError("Automaton.start_strategy: strategy is required.")
        if not self._strategy_is_initialised:
            raise RuntimeError("Automaton.start_strategy: strategy is not initialised.")
        self.strategy_started = True
        self.strategy_stopped = False

    def stop_strategy(self):
        self.strategy_stopped = True

    def dmd_calibrate(self, cfg, filename=None):
        self.dmd_calibration_calls.append((cfg, filename))
        self.acq_mngr.dmd.calibrated = True
        return [("point",)], None, None, Path(filename or "calibration.pkl")


def test_facade_handles_stage_and_led_requests() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton())

    response = facade.handle(GuiRequest(command=GuiCommandType.PING))
    assert response.ok
    assert response.payload["devices_initialised"] is True
    assert response.payload["controllers"][0]["name"] == "Shared Controller"

    response = facade.handle(GuiRequest(command=GuiCommandType.STAGE_MOVE_ABSOLUTE, payload={"x": 5, "y": 6, "z": 7}))
    assert response.ok
    assert response.payload["coordinate"] == {"x": 5, "y": 6, "z": 7, "channel_id": 0}

    response = facade.handle(GuiRequest(command=GuiCommandType.STAGE_MOVE_FOV, payload={"direction": "RIGHT"}))
    assert response.ok
    assert response.payload["coordinate"] == {"x": 12.8, "y": 6.0, "z": 7, "channel_id": 0}
    assert response.payload["stage"]["fov_step_size"] == 100.0
    assert response.payload["stage"]["camera_fov_step_size"] == pytest.approx(7.8)

    response = facade.handle(GuiRequest(command=GuiCommandType.LED_SET, payload={"led": "LED_450_NM", "brightness": 22}))
    assert response.ok
    assert response.payload["state"]["brightness"] == 22

    response = facade.handle(GuiRequest(command=GuiCommandType.CAMERA_SET_EXPOSURE, payload={"exposure": 150}))
    assert response.ok
    assert response.payload["camera"]["exposure"] == 150

    response = facade.handle(GuiRequest(command=GuiCommandType.FILTER_WHEEL_SET, payload={"filter_wheel": "FILTER_527nm"}))
    assert response.ok
    assert response.payload["filter_wheel"]["current_filter"]["name"] == "FILTER_527nm"


def test_facade_handles_image_transport_probe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(IMAGE_TRANSPORT_DIR_ENV, str(tmp_path))
    facade = AutomatonGuiFacade(FakeAutomaton())

    response = facade.handle(GuiRequest(command=GuiCommandType.IMAGE_TRANSPORT_PROBE))

    assert response.ok
    probe = response.payload["image_transport_probe"]
    assert Path(probe["path"]).read_text(encoding="ascii") == probe["token"]


def test_facade_handles_dmd_requests() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_STATUS))
    assert response.ok
    assert response.payload["dmd"]["name"] == "Fake DMD"

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_DISPLAY_PATTERN, payload={"pattern": "checkerboard"}))
    assert response.ok
    assert automaton.acq_mngr.dmd.calls == ["display_image"]
    assert response.payload["dmd"]["last_pattern"] == "checkerboard"
    assert response.payload["dmd"]["preview"]["shape"] == [10, 20]

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_CALIBRATE))
    assert response.ok
    assert automaton.dmd_calibration_calls
    assert response.payload["dmd"]["calibration_file"] == "calibration.pkl"
    assert response.payload["dmd"]["calibration_points"] == 1

    response = facade.handle(
        GuiRequest(command=GuiCommandType.DMD_LOAD_CALIBRATION, payload={"filename": "selected_calibration.pkl"})
    )
    assert response.ok
    assert automaton.acq_mngr.dmd.is_calibrated()
    assert response.payload["dmd"]["calibration_file"] == "selected_calibration.pkl"

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_CALIBRATION_POINTS))
    assert response.ok
    assert response.payload["dmd_calibration_points"]["dmd_shape"] == [20, 10]
    assert response.payload["dmd_calibration_points"]["cam_shape"] == [30, 30]
    assert response.payload["dmd_calibration_points"]["dmd_points"] == [
        {"row": 1, "col": 2},
        {"row": 3, "col": 4},
    ]
    assert response.payload["dmd_calibration_points"]["cam_points"] == [
        {"row": 5, "col": 6},
        {"row": 7, "col": 8},
    ]


def test_facade_handles_manual_acquisition_request() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(
        GuiRequest(
            command=GuiCommandType.ACQUISITION_TAKE_FRAME,
            payload={"settings": {"normalise": True, "save": True}, "image_transport": IMAGE_TRANSPORT_RAW},
        )
    )

    assert response.ok
    assert response.payload["frame"]["kind"] == "frame"
    assert response.payload["frame"]["planes"] == 1
    assert response.payload["frame"]["image_shape"] == [3, 4]
    assert response.payload["frame"]["stack_shape"] == [1, 3, 4]
    assert response.payload["frame"]["dtype"] == "uint16"
    assert response.payload["frame"]["preview"]["shape"] == [3, 4]
    assert "stack_preview" not in response.payload["frame"]
    assert response.payload["frame"]["saved_paths"] == ["frame_0.tiff"]
    assert automaton.acq_mngr.calls[-1][1].normalise is True
    assert automaton.acq_mngr.calls[-1][1].save is True


def test_facade_handles_explicit_manual_acquisition_peripherals() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(
        GuiRequest(
            command=GuiCommandType.ACQUISITION_TAKE_FRAME,
            payload={
                "use_current_main_controls": False,
                "leds": {"LED_450_NM": 33},
                "filter_wheel": "BLOCKING",
                "dmd_pattern": "checkerboard",
                "settings": {"illuminate_dmd": True},
                "image_transport": IMAGE_TRANSPORT_RAW,
            },
        )
    )

    assert response.ok
    metadata, settings = automaton.acq_mngr.calls[-1]
    assert metadata.leds == {LEDType.LED_450_NM: 33.0}
    assert metadata.filter_wheel is FilterWheelType.BLOCKING
    assert metadata.dmd_pattern is not None
    assert settings.illuminate_dmd is True


def test_facade_lists_and_loads_saved_acquisition_tiff(tmp_path) -> None:
    image = np.arange(20, dtype=np.uint16).reshape(4, 5)
    image_path = tmp_path / "saved_frame.tiff"
    tifffile.imwrite(image_path, image)
    file_manager = FileManager(FileNameConfig(directory=tmp_path))
    facade = AutomatonGuiFacade(FakeAutomaton(file_manager=file_manager))

    response = facade.handle(GuiRequest(command=GuiCommandType.ACQUISITION_LIST_FILES))

    assert response.ok
    assert response.payload["acquisition_files"][0]["label"] == "saved_frame.tiff"
    assert response.payload["acquisition_files"][0]["path"] == str(image_path)

    response = facade.handle(
        GuiRequest(
            command=GuiCommandType.ACQUISITION_LOAD_FRAME,
            payload={"filename": str(image_path), "image_transport": IMAGE_TRANSPORT_RAW},
        )
    )

    assert response.ok
    payload = response.payload["frame"]
    assert payload["kind"] == "loaded_frame"
    assert payload["source"] == "file"
    assert payload["image_shape"] == [4, 5]
    assert np.array_equal(array_from_preview_payload(payload["preview"]), image)


def test_facade_handles_z_stack_acquisition_request() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(
        GuiRequest(
            command=GuiCommandType.ACQUISITION_TAKE_Z_STACK,
            payload={
                "start_z": -1,
                "end_z": 1,
                "step_z": 1,
                "settings": {"illuminate_dmd": False},
                "image_transport": IMAGE_TRANSPORT_RAW,
            },
        )
    )

    assert response.ok
    assert response.payload["frame"]["kind"] == "z_stack"
    assert response.payload["frame"]["planes"] == 3
    assert response.payload["frame"]["image_shape"] == [3, 4]
    assert response.payload["frame"]["stack_shape"] == [3, 3, 4]
    assert response.payload["frame"]["stack_preview"]["is_stack"] is True
    assert response.payload["frame"]["stack_preview"]["shape"] == [3, 3, 4]
    assert response.payload["frame"]["z_positions"] == [-1.0, 0.0, 1.0]
    _metadata, z_coordinates, settings = automaton.acq_mngr.calls[-1]
    assert [coordinate.z for coordinate in z_coordinates] == [-1.0, 0.0, 1.0]
    assert settings.illuminate_dmd is False


def test_facade_handles_controller_status_request() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton())

    response = facade.handle(GuiRequest(command=GuiCommandType.CONTROLLER_STATUS))

    assert response.ok
    controllers = response.payload["controllers"]
    assert [controller["name"] for controller in controllers] == ["Shared Controller", "DMD Controller"]
    shared = controllers[0]
    assert shared["connected"] is True
    assert "Fake Camera" in shared["owners"]
    assert "Fake Filter Wheel" in shared["owners"]


def test_facade_initialise_devices_initialises_controllers_first() -> None:
    automaton = FakeAutomaton(devices_initialised=False)
    shared_controller = automaton.acq_mngr.camera.peripheral_ctrl
    dmd_controller = automaton.acq_mngr.dmd.peripheral_ctrl
    shared_controller.initialised = False
    shared_controller.alive = False
    dmd_controller.initialised = False
    dmd_controller.alive = False
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.INITIALISE_DEVICES))

    assert response.ok
    assert automaton.devices_initialised is True
    assert shared_controller.initialise_count == 1
    assert dmd_controller.initialise_count == 1
    assert [controller["connected"] for controller in response.payload["controllers"]] == [True, True]


def test_facade_shutdown_closes_controllers_and_reports_status() -> None:
    automaton = FakeAutomaton()
    shared_controller = automaton.acq_mngr.camera.peripheral_ctrl
    dmd_controller = automaton.acq_mngr.dmd.peripheral_ctrl
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.SHUTDOWN))

    assert response.ok
    assert automaton.shutdown_count == 1
    assert shared_controller.shutdown_count == 1
    assert dmd_controller.shutdown_count == 1
    assert [controller["connected"] for controller in response.payload["controllers"]] == [False, False]


def test_facade_handles_filter_wheel_requests() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.FILTER_WHEEL_STATUS))
    assert response.ok
    assert response.payload["filter_wheel"]["current_filter"]["name"] == "NO_FILTER"

    response = facade.handle(GuiRequest(command=GuiCommandType.FILTER_WHEEL_SET, payload={"filter_wheel": "BLOCKING"}))
    assert response.ok
    assert automaton.acq_mngr.filter_wheel.calls == [FilterWheelType.BLOCKING]
    assert response.payload["filter_wheel"]["current_filter"]["name"] == "BLOCKING"


def test_facade_handles_autofocus_requests() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_STATUS))
    assert response.ok
    assert response.payload["autofocus"]["status"]["name"] == "IDLE"

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_CONFIGURE, payload={"config": {"preset": "oil"}}))
    assert response.ok
    assert automaton.focus_nav.autofocus.calls[0][0] == "apply_config"
    assert response.payload["autofocus"]["configured"] is True

    response = facade.handle(
        GuiRequest(command=GuiCommandType.AUTOFOCUS_INITIALISE, payload={"lock_after_initialise": True})
    )
    assert response.ok
    assert automaton.focus_nav.autofocus.calls[1][0] == "run_calibration"
    assert response.payload["autofocus"]["autofocus_initialised"] is True
    assert response.payload["autofocus"]["status"]["name"] == "IN_FOCUS"

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_LOCK))
    assert response.ok
    assert response.payload["autofocus"]["status"]["name"] == "IN_FOCUS"

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_UNLOCK))
    assert response.ok
    assert response.payload["autofocus"]["status"]["name"] == "READY"

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_DISABLE))
    assert response.ok
    assert response.payload["autofocus"]["status"]["name"] == "IDLE"


def test_facade_handles_software_focus_requests() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.SOFTWARE_FOCUS_STATUS))
    assert response.ok
    assert response.payload["software_focus"]["config"]["algorithm"] == "STEEL"
    assert response.payload["software_focus"]["last_result"] is None

    response = facade.handle(GuiRequest(command=GuiCommandType.SOFTWARE_FOCUS_RUN))
    assert response.ok
    assert automaton.focus_nav.software_focus.calls == [("run", None)]
    result = response.payload["software_focus"]["last_result"]
    assert result["focus_status"]["name"] == "IN_FOCUS"
    assert result["best_coordinate"]["z"] == 4
    assert result["z_points"] == 3


def test_facade_handles_strategy_lifecycle_requests() -> None:
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)

    response = facade.handle(GuiRequest(command=GuiCommandType.STRATEGY_LIST))

    assert response.ok
    strategy_names = [strategy["name"] for strategy in response.payload["strategies"]]
    assert "NoStrategy" in strategy_names
    assert "SimpleImagingStrategy" in strategy_names
    assert response.payload["strategy"]["name"] is None

    response = facade.handle(GuiRequest(command=GuiCommandType.STRATEGY_SET, payload={"name": "NoStrategy"}))
    assert response.ok
    assert response.payload["strategy"]["name"] == "NoStrategy"
    assert response.payload["strategy"]["is_initialised"] is False

    response = facade.handle(
        GuiRequest(
            command=GuiCommandType.FOV_INITIALISE,
            payload={"fovs": [{"fov_id": 0, "x": 1, "y": 2, "z": 3}]},
        )
    )
    assert response.ok
    assert response.payload["fovs"] == [{"fov_id": 0, "x": 1.0, "y": 2.0, "z": 3.0, "channel_id": 0}]
    assert response.payload["strategy"]["is_initialised"] is True

    response = facade.handle(GuiRequest(command=GuiCommandType.STRATEGY_START))
    assert response.ok
    assert response.payload["strategy_active"] is True
    assert response.payload["strategy"]["running"] is True

    response = facade.handle(GuiRequest(command=GuiCommandType.STRATEGY_STOP))
    assert response.ok
    assert response.payload["strategy_active"] is False
    assert response.payload["strategy"]["stopped"] is True


def test_facade_rejects_mutating_requests_during_strategy() -> None:
    automaton = FakeAutomaton()
    automaton.strategy_has_started = lambda: True
    facade = AutomatonGuiFacade(automaton)

    rejected = facade.handle(GuiRequest(command=GuiCommandType.LED_SET, payload={"led": "LED_450_NM", "brightness": 22}))
    allowed = facade.handle(GuiRequest(command=GuiCommandType.LED_DISABLE_ALL))

    assert not rejected.ok
    assert allowed.ok


def test_facade_stage_request_missing_stage_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_stage=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.STAGE_STATUS))

    assert not response.ok
    assert "no stage is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI stage request ignored because no stage is configured."]


def test_facade_led_request_missing_led_manager_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_led_manager=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.LED_LIST))

    assert not response.ok
    assert "no LED manager is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI LED request ignored because no LED manager is configured."]


def test_facade_camera_request_missing_camera_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_camera=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.CAMERA_STATUS))

    assert not response.ok
    assert "no camera is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI camera request ignored because no camera is configured."]


def test_facade_filter_wheel_request_missing_filter_wheel_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_filter_wheel=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.FILTER_WHEEL_STATUS))

    assert not response.ok
    assert "no filter wheel is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI filter wheel request ignored because no filter wheel is configured."]


def test_facade_dmd_request_missing_dmd_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_dmd=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_STATUS))

    assert not response.ok
    assert "no DMD is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI DMD request ignored because no DMD is configured."]


def test_facade_autofocus_request_missing_autofocus_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_autofocus=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_STATUS))

    assert not response.ok
    assert "no autofocus is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI autofocus request ignored because no autofocus is configured."]


def test_facade_software_focus_request_missing_software_focus_logs_and_returns_error(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr("evomachine.gui.facade.logger.warning", lambda message: warnings.append(message))
    facade = AutomatonGuiFacade(FakeAutomaton(with_software_focus=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.SOFTWARE_FOCUS_STATUS))

    assert not response.ok
    assert "no software focus is configured" in response.error
    assert warnings == ["AutomatonGuiFacade: GUI software focus request ignored because no software focus is configured."]


def test_facade_rejects_led_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.LED_SET, payload={"led": "LED_450_NM", "brightness": 22}))

    assert not response.ok
    assert "Initialise devices before using LED controls" in response.error


def test_facade_rejects_stage_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.STAGE_MOVE_ABSOLUTE, payload={"x": 1, "y": 2, "z": 3}))

    assert not response.ok
    assert "Initialise devices before using stage controls" in response.error


def test_facade_rejects_camera_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.CAMERA_SET_EXPOSURE, payload={"exposure": 150}))

    assert not response.ok
    assert "Initialise devices before using camera controls" in response.error


def test_facade_rejects_filter_wheel_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.FILTER_WHEEL_SET, payload={"filter_wheel": "BLOCKING"}))

    assert not response.ok
    assert "Initialise devices before using filter wheel controls" in response.error


def test_facade_rejects_dmd_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.DMD_DISPLAY_PATTERN, payload={"pattern": "full"}))

    assert not response.ok
    assert "Initialise devices before using DMD controls" in response.error


def test_facade_rejects_autofocus_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.AUTOFOCUS_LOCK))

    assert not response.ok
    assert "Initialise devices before using autofocus controls" in response.error


def test_facade_rejects_software_focus_request_before_devices_initialised() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton(devices_initialised=False))

    response = facade.handle(GuiRequest(command=GuiCommandType.SOFTWARE_FOCUS_RUN))

    assert not response.ok
    assert "Initialise devices before using software focus controls" in response.error

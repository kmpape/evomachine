from __future__ import annotations

import os

import pytest

if os.environ.get("EVOMACHINE_GUI_RUN_QT_TESTS") != "1":
    pytest.skip("Qt widget tests are opt-in in headless environments.", allow_module_level=True)

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from evomachine.gui.panels.acquisition import (
    FrameAcquisitionSettingsPanel,
    ManualAcquisitionPanel,
    SavedImageLoaderPanel,
    ZStackSettingsPanel,
)
from evomachine.gui.panels.autofocus import AutofocusPanel
from evomachine.gui.panels.camera import CameraPanel
from evomachine.gui.panels.dmd import DmdPanel
from evomachine.gui.panels.filterwheel import FilterWheelPanel
from evomachine.gui.panels.leds import LedManagerPanel
from evomachine.gui.panels.software_focus import SoftwareFocusPanel
from evomachine.gui.panels.stage import StagePanel
from evomachine.gui.panels.strategy import FovSetupPanel, StrategySetupPanel
from evomachine.types import FilterWheelType, LEDType


_QT_APP: QApplication | None = None


class FakeController(QObject):
    stage_coordinates_received = pyqtSignal(dict)
    stage_status_received = pyqtSignal(dict)
    camera_status_received = pyqtSignal(dict)
    acquisition_files_received = pyqtSignal(list)
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
    response_error = pyqtSignal(str)
    fovs_received = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.calls = []

    def refresh_stage(self):
        self.calls.append(("refresh_stage",))

    def initialise_fovs(self, fovs, use_autofocus=False):
        self.calls.append(("initialise_fovs", fovs, use_autofocus))

    def move_stage_absolute(self, x, y, z):
        self.calls.append(("move_stage_absolute", x, y, z))

    def move_stage_relative(self, dx, dy, dz):
        self.calls.append(("move_stage_relative", dx, dy, dz))

    def move_stage_fov(self, direction, multiplier=1.0):
        self.calls.append(("move_stage_fov", direction, multiplier))

    def stop_stage(self):
        self.calls.append(("stop_stage",))

    def zero_stage(self):
        self.calls.append(("zero_stage",))

    def refresh_camera(self):
        self.calls.append(("refresh_camera",))

    def set_camera_exposure(self, exposure):
        self.calls.append(("set_camera_exposure", exposure))

    def acquire_frame(self, payload=None):
        self.calls.append(("acquire_frame", payload))

    def acquire_z_stack(self, payload=None):
        self.calls.append(("acquire_z_stack", payload))

    def refresh_acquisition_files(self, directory=None):
        self.calls.append(("refresh_acquisition_files", directory))

    def load_acquisition_frame(self, filename, image_transport=None):
        self.calls.append(("load_acquisition_frame", filename, image_transport))

    def refresh_filter_wheel(self):
        self.calls.append(("refresh_filter_wheel",))

    def set_filter_wheel(self, filter_wheel):
        self.calls.append(("set_filter_wheel", filter_wheel))

    def refresh_leds(self):
        self.calls.append(("refresh_leds",))

    def refresh_led_state(self, led):
        self.calls.append(("refresh_led_state", led))

    def set_led(self, led, brightness, duration=None):
        self.calls.append(("set_led", led, brightness, duration))

    def disable_led(self, led):
        self.calls.append(("disable_led", led))

    def disable_all_leds(self):
        self.calls.append(("disable_all_leds",))

    def refresh_dmd(self):
        self.calls.append(("refresh_dmd",))

    def display_dmd_pattern(self, pattern, config=None):
        self.calls.append(("display_dmd_pattern", pattern, config))

    def calibrate_dmd(self):
        self.calls.append(("calibrate_dmd",))

    def load_dmd_calibration(self, filename):
        self.calls.append(("load_dmd_calibration", filename))

    def request_dmd_calibration_points(self):
        self.calls.append(("request_dmd_calibration_points",))

    def refresh_autofocus(self):
        self.calls.append(("refresh_autofocus",))

    def configure_autofocus(self, config=None):
        self.calls.append(("configure_autofocus", config))

    def initialise_autofocus(self, lock_after_initialise=False, config=None):
        self.calls.append(("initialise_autofocus", lock_after_initialise, config))

    def lock_autofocus(self):
        self.calls.append(("lock_autofocus",))

    def unlock_autofocus(self):
        self.calls.append(("unlock_autofocus",))

    def disable_autofocus(self):
        self.calls.append(("disable_autofocus",))

    def refresh_software_focus(self):
        self.calls.append(("refresh_software_focus",))

    def run_software_focus(self):
        self.calls.append(("run_software_focus",))

    def refresh_strategies(self):
        self.calls.append(("refresh_strategies",))

    def refresh_strategy_status(self):
        self.calls.append(("refresh_strategy_status",))

    def set_strategy(self, name, file_path=None):
        self.calls.append(("set_strategy", name, file_path))

    def start_strategy(self):
        self.calls.append(("start_strategy",))

    def stop_strategy(self):
        self.calls.append(("stop_strategy",))


def _app():
    global _QT_APP
    _QT_APP = QApplication.instance() or _QT_APP or QApplication([])
    return _QT_APP


def test_stage_panel_sends_relative_delta_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.x_input.setValue(1)
    panel.y_input.setValue(2)
    panel.z_input.setValue(3)

    panel._move_delta()

    assert controller.calls == [("move_stage_relative", 1.0, 2.0, 3.0)]


def test_stage_panel_sends_zero_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.zero_button.click()

    assert controller.calls == [("zero_stage",)]


def test_stage_panel_sends_camera_fov_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel._move_camera_fov("RIGHT")

    assert controller.calls == [("move_stage_fov", "RIGHT", 1.0)]


def test_camera_panel_sends_set_exposure_request() -> None:
    _app()
    controller = FakeController()
    panel = CameraPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.exposure_input.setValue(150)

    panel._set_exposure()

    assert controller.calls == [("set_camera_exposure", 150.0)]


def test_camera_panel_sends_acquire_frame_request() -> None:
    _app()
    controller = FakeController()
    panel = CameraPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_frame_button.click()

    assert controller.calls == [("acquire_frame", None)]


def test_manual_acquisition_panel_sends_settings_payload() -> None:
    _app()
    controller = FakeController()
    settings_panel = FrameAcquisitionSettingsPanel()
    settings_panel.config_values["normalise"] = True
    panel = ManualAcquisitionPanel(controller=controller, settings_provider=settings_panel.payload)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_button.click()

    assert controller.calls == [
        (
            "acquire_frame",
            {
                "settings": {
                    "save": False,
                    "normalise": True,
                    "illuminate_dmd": False,
                    "clear_dmd_after": False,
                    "restore_leds_after": True,
                    "disable_leds_after": False,
                },
                "use_current_main_controls": True,
            },
        )
    ]


def test_acquisition_config_does_not_offer_unknown_filter_wheel() -> None:
    _app()
    panel = FrameAcquisitionSettingsPanel()

    filter_field = next(field for field in panel._config_fields() if field.key == "filter_wheel")

    assert FilterWheelType.UNKNOWN.name not in filter_field.choices


def test_z_stack_panel_sends_request() -> None:
    _app()
    controller = FakeController()
    settings_panel = FrameAcquisitionSettingsPanel()
    settings_panel.config_values["start_z"] = -1.0
    settings_panel.config_values["end_z"] = 1.0
    settings_panel.config_values["step_z"] = 0.5
    panel = ZStackSettingsPanel(controller=controller, settings_provider=settings_panel.z_stack_payload)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_button.click()

    assert controller.calls == [
        (
            "acquire_z_stack",
            {
                "settings": {
                    "save": False,
                    "normalise": False,
                    "illuminate_dmd": False,
                    "clear_dmd_after": False,
                    "restore_leds_after": True,
                    "disable_leds_after": False,
                },
                "use_current_main_controls": True,
                "start_z": -1.0,
                "end_z": 1.0,
                "step_z": 0.5,
            },
        )
    ]


def test_saved_image_loader_panel_sends_load_requests() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    panel.update_file_list([{"label": "test.tiff", "path": "/tmp/test.tiff"}])

    panel.load_button.click()
    panel.force_socket_transport = True
    panel.load_button.click()

    assert controller.calls == [
        ("load_acquisition_frame", "/tmp/test.tiff", None),
        ("load_acquisition_frame", "/tmp/test.tiff", "socket_tiff"),
    ]


def test_saved_image_loader_panel_refreshes_configured_directory() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    panel.directory = "/tmp/example_output"

    panel.refresh_button.click()

    assert controller.calls == [("refresh_acquisition_files", "/tmp/example_output")]


def test_filter_wheel_panel_sends_set_request() -> None:
    _app()
    controller = FakeController()
    panel = FilterWheelPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_status(
        {
            "available_filters": [
                {"name": "NO_FILTER", "value": 4},
                {"name": "BLOCKING", "value": 5},
            ],
            "current_filter": {"name": "NO_FILTER", "value": 4},
            "is_initialised": True,
            "is_alive": True,
        }
    )
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("BLOCKING"))

    panel.set_button.click()

    assert controller.calls == [("set_filter_wheel", "BLOCKING")]


def test_led_panel_sends_set_request() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    panel.brightness_inputs[LEDType.LED_450_NM].setValue(12)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)

    assert controller.calls == [("set_led", "LED_450_NM", 12.0, None)]


def test_led_panel_allows_full_backend_brightness_range() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)

    assert panel.brightness_inputs[LEDType.LED_450_NM].maximum() == 100.0


def test_led_panel_toggles_one_led_off() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])

    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(False)

    assert controller.calls == [
        ("set_led", "LED_450_NM", 29.0, None),
        ("disable_led", "LED_450_NM"),
    ]


def test_led_panel_refreshes_timed_state_after_stop_time(monkeypatch) -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    callbacks = []
    monkeypatch.setattr("evomachine.gui.panels.leds.time.time", lambda: 1000.0)
    monkeypatch.setattr(
        "evomachine.gui.panels.leds.QTimer.singleShot",
        lambda _delay, callback: callbacks.append(callback),
    )

    panel.update_state({"led": "LED_450_NM", "brightness": 50, "is_on": True, "stop_time": 1003.0})
    callbacks[0]()

    assert controller.calls == [("refresh_led_state", "LED_450_NM")]


def test_dmd_panel_sends_pattern_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.pattern_buttons["checkerboard"].click()

    assert controller.calls == [
        ("display_dmd_pattern", "checkerboard", panel._shape_config_payload())
    ]


def test_dmd_panel_sends_load_calibration_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_status(
        {
            "is_initialised": True,
            "is_alive": True,
            "is_calibrated": True,
            "calibration_file": "/tmp/current_calibration.pkl",
            "calibration_files": [
                {
                    "label": "current_calibration.pkl",
                    "path": "/tmp/current_calibration.pkl",
                    "is_current": True,
                }
            ],
        }
    )

    panel.load_calibration_button.click()

    assert controller.calls == [("load_dmd_calibration", "/tmp/current_calibration.pkl")]


def test_dmd_panel_sends_calibration_plot_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.show_calibration_plot_button.click()

    assert controller.calls == [("request_dmd_calibration_points",)]


def test_autofocus_panel_sends_lock_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.lock_button.click()

    assert controller.calls == [("lock_autofocus",)]


def test_autofocus_panel_sends_config_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.config_values["led_intensity"] = 80
    panel.config_values["objective_na"] = 1.4

    panel._apply_config()

    assert controller.calls == [
        (
            "configure_autofocus",
            {
                "averaging": 5,
                "led_intensity": 80,
                "lock_range": 0.1,
                "loop_gain": 10,
                "update_rate": 10,
                "min_error": 100,
                "min_snr": 2.0,
            },
        ),
    ]


def test_autofocus_panel_sends_calibration_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.lock_after_calibration_checkbox.setChecked(True)

    panel.run_calibration_button.click()

    assert controller.calls == [
        (
            "initialise_autofocus",
            True,
            {
                "averaging": 5,
                "led_intensity": 70,
                "lock_range": 0.1,
                "loop_gain": 10,
                "update_rate": 10,
                "min_error": 100,
                "objective_na": 0.9,
                "min_snr": 2.0,
            },
        ),
    ]


def test_software_focus_panel_sends_run_request() -> None:
    _app()
    controller = FakeController()
    panel = SoftwareFocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.run_button.click()

    assert controller.calls == [("run_software_focus",)]


def test_strategy_panel_sends_lifecycle_requests() -> None:
    _app()
    controller = FakeController()
    panel = StrategySetupPanel(controller=controller)
    controller.calls.clear()
    panel.update_strategies(
        [
            {
                "name": "NoStrategy",
                "file_path": None,
                "commands": [],
                "built_in": True,
            },
            {
                "name": "SimpleImagingStrategy",
                "file_path": "/tmp/strategy_simple_imaging.py",
                "commands": ["IMAGE", "MOVE", "WAIT"],
                "built_in": False,
            },
        ]
    )
    panel.strategy_combo.setCurrentIndex(1)

    panel.set_button.click()
    panel.update_status(
        {
            "name": "SimpleImagingStrategy",
            "is_initialised": True,
            "running": False,
            "fovs_initialised": True,
        }
    )
    panel.start_button.click()
    panel.update_status(
        {
            "name": "SimpleImagingStrategy",
            "is_initialised": True,
            "running": True,
            "fovs_initialised": True,
        }
    )
    panel.stop_button.click()

    assert controller.calls == [
        ("set_strategy", "SimpleImagingStrategy", "/tmp/strategy_simple_imaging.py"),
        ("start_strategy",),
        ("stop_strategy",),
    ]


def test_fov_setup_panel_sends_initialise_request() -> None:
    _app()
    controller = FakeController()
    panel = FovSetupPanel(controller=controller)
    panel.fov_id_input.setValue(2)
    panel.x_input.setValue(10)
    panel.y_input.setValue(20)
    panel.z_input.setValue(30)

    panel.add_button.click()
    panel.initialise_button.click()

    assert controller.calls == [
        (
            "initialise_fovs",
            [{"fov_id": 2, "x": 10.0, "y": 20.0, "z": 30.0, "channel_id": 0}],
            False,
        )
    ]

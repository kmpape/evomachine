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
from evomachine.types import LEDType


class FakeController(QObject):
    stage_coordinates_received = pyqtSignal(dict)
    stage_status_received = pyqtSignal(dict)
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

    def stop_stage(self):
        self.calls.append(("stop_stage",))

    def refresh_camera(self):
        self.calls.append(("refresh_camera",))

    def set_camera_exposure(self, exposure):
        self.calls.append(("set_camera_exposure", exposure))

    def acquire_frame(self, payload=None):
        self.calls.append(("acquire_frame", payload))

    def acquire_z_stack(self, payload=None):
        self.calls.append(("acquire_z_stack", payload))

    def refresh_filter_wheel(self):
        self.calls.append(("refresh_filter_wheel",))

    def set_filter_wheel(self, filter_wheel):
        self.calls.append(("set_filter_wheel", filter_wheel))

    def refresh_leds(self):
        self.calls.append(("refresh_leds",))

    def set_led(self, led, brightness, duration=None):
        self.calls.append(("set_led", led, brightness, duration))

    def disable_led(self, led):
        self.calls.append(("disable_led", led))

    def disable_all_leds(self):
        self.calls.append(("disable_all_leds",))

    def refresh_dmd(self):
        self.calls.append(("refresh_dmd",))

    def display_dmd_pattern(self, pattern):
        self.calls.append(("display_dmd_pattern", pattern))

    def calibrate_dmd(self):
        self.calls.append(("calibrate_dmd",))

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
    return QApplication.instance() or QApplication([])


def test_stage_panel_sends_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.x_input.setValue(1)
    panel.y_input.setValue(2)
    panel.z_input.setValue(3)

    panel._move_absolute()

    assert controller.calls == [("move_stage_absolute", 1.0, 2.0, 3.0)]


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
    settings_panel.checkboxes["normalise"].setChecked(True)
    panel = ManualAcquisitionPanel(controller=controller, settings_provider=settings_panel.payload)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_button.click()

    assert controller.calls == [
        ("acquire_frame", {"settings": {
            "save": False,
            "normalise": True,
            "illuminate_dmd": True,
            "clear_dmd_after": False,
            "restore_leds_after": True,
            "disable_leds_after": False,
        }})
    ]


def test_z_stack_panel_sends_request() -> None:
    _app()
    controller = FakeController()
    settings_panel = FrameAcquisitionSettingsPanel()
    panel = ZStackSettingsPanel(controller=controller, settings_provider=settings_panel.payload)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.start_input.setValue(-1)
    panel.end_input.setValue(1)
    panel.step_input.setValue(0.5)

    panel.acquire_button.click()

    assert controller.calls == [
        ("acquire_z_stack", {
            "settings": {
                "save": False,
                "normalise": False,
                "illuminate_dmd": True,
                "clear_dmd_after": False,
                "restore_leds_after": True,
                "disable_leds_after": False,
            },
            "start_z": -1.0,
            "end_z": 1.0,
            "step_z": 0.5,
        })
    ]


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


def test_dmd_panel_sends_pattern_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.pattern_buttons["checkerboard"].click()

    assert controller.calls == [("display_dmd_pattern", "checkerboard")]


def test_autofocus_panel_sends_lock_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.lock_button.click()

    assert controller.calls == [("lock_autofocus",)]


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
    panel.update_strategies([
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
    ])
    panel.strategy_combo.setCurrentIndex(1)

    panel.set_button.click()
    panel.update_status({"name": "SimpleImagingStrategy", "is_initialised": True, "running": False, "fovs_initialised": True})
    panel.start_button.click()
    panel.update_status({"name": "SimpleImagingStrategy", "is_initialised": True, "running": True, "fovs_initialised": True})
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
        ("initialise_fovs", [{"fov_id": 2, "x": 10.0, "y": 20.0, "z": 30.0, "channel_id": 0}], False)
    ]

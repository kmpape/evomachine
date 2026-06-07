from __future__ import annotations

import os

import pytest

if os.environ.get("EVOMACHINE_GUI_RUN_QT_TESTS") != "1":
    pytest.skip("Qt widget tests are opt-in in headless environments.", allow_module_level=True)

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from evomachine.gui.panels.leds import LedManagerPanel
from evomachine.gui.panels.stage import StagePanel


class FakeController(QObject):
    stage_coordinates_received = pyqtSignal(dict)
    stage_status_received = pyqtSignal(dict)
    led_list_received = pyqtSignal(list)
    led_state_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.calls = []

    def refresh_stage(self):
        self.calls.append(("refresh_stage",))

    def move_stage_absolute(self, x, y, z):
        self.calls.append(("move_stage_absolute", x, y, z))

    def stop_stage(self):
        self.calls.append(("stop_stage",))

    def refresh_leds(self):
        self.calls.append(("refresh_leds",))

    def set_led(self, led, brightness, duration=None):
        self.calls.append(("set_led", led, brightness, duration))

    def disable_led(self, led):
        self.calls.append(("disable_led", led))

    def disable_all_leds(self):
        self.calls.append(("disable_all_leds",))


def _app():
    return QApplication.instance() or QApplication([])


def test_stage_panel_sends_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.x_input.setValue(1)
    panel.y_input.setValue(2)
    panel.z_input.setValue(3)

    panel._move_absolute()

    assert controller.calls == [("move_stage_absolute", 1.0, 2.0, 3.0)]


def test_led_panel_sends_set_request() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_leds(["LED_450_NM"])
    panel.brightness_input.setValue(12)

    panel._set_led()

    assert controller.calls == [("set_led", "LED_450_NM", 12.0, None)]

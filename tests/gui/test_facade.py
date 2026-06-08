from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from evomachine.coordinates import Coordinate
from evomachine.types import LEDType
from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.protocol import GuiCommandType, GuiRequest


@dataclass
class FakeLedState:
    led_type: LEDType
    brightness: float = 0.0
    is_on: bool = False
    stop_time: float | None = None


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


class FakeAutomaton:
    def __init__(self, with_stage: bool = True, with_led_manager: bool = True):
        stage = FakeStage()
        led_manager = FakeLedManager()
        self.focus_nav = SimpleNamespace(stage=stage) if with_stage else SimpleNamespace()
        self.acq_mngr = SimpleNamespace(led_manager=led_manager) if with_led_manager else SimpleNamespace()
        self.shutdown_count = 0

    def strategy_has_started(self):
        return False

    def strategy_has_stopped(self):
        return False

    def has_shutdown(self):
        return self.shutdown_count > 0

    def shutdown(self):
        self.shutdown_count += 1

    def stop(self):
        return None

    def initialise_devices(self):
        return None

    def devices_is_initialised(self):
        return True


def test_facade_handles_stage_and_led_requests() -> None:
    facade = AutomatonGuiFacade(FakeAutomaton())

    response = facade.handle(GuiRequest(command=GuiCommandType.STAGE_MOVE_ABSOLUTE, payload={"x": 5, "y": 6, "z": 7}))
    assert response.ok
    assert response.payload["coordinate"] == {"x": 5, "y": 6, "z": 7, "channel_id": 0}

    response = facade.handle(GuiRequest(command=GuiCommandType.LED_SET, payload={"led": "LED_450_NM", "brightness": 22}))
    assert response.ok
    assert response.payload["state"]["brightness"] == 22


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

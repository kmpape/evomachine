from typing import get_type_hints

import pytest

from evomachine.bindings.asitiger.leds import FakeTigerLedController, TigerLedSource
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.kwr103.leds import FakeKWR103, KWR103LedSource
from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController
from evomachine.bindings.syncboard.leds import FakeSyncBoardController, SyncBoardLedSource
from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.bindings.virtual.leds import VirtualLedSource
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager, LedSource
from evomachine.peripherals import PeripheralController
from evomachine.bindings.binding_types import BindingType
from evomachine.types import BrightnessType, LEDType


# TODO(CODEX): Make these Fake classes import dependent. If some global variable is true, the real classes are imported and the real bindings tested. For security reasons, we need test settings defined somewhere.
class FakeTimer:
    instances = []

    def __init__(self, interval, function, kwargs=None):
        self.interval = interval
        self.function = function
        self.kwargs = kwargs or {}
        self.daemon = False
        self.started = False
        self.cancelled = False
        FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function(**self.kwargs)


class CountingPeripheralController(PeripheralController):
    def __init__(self, alive_result=True):
        self.alive_result = alive_result
        self.alive_queries = 0
        super().__init__(name="counting")

    def _initialise(self, force=False):
        return True

    def _check_is_alive(self):
        self.alive_queries += 1
        return self.alive_result

    def _stop(self):
        return

    def _shutdown(self, force=False):
        return


def make_virtual_source() -> VirtualLedSource:
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    source = VirtualLedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.LED_450_NM],
    )
    source.initialise()
    return source


def test_led_config_rejects_missing_internal_mapping():
    with pytest.raises(TypeError):
        LedConfig(binding="virtual", available_leds=[LEDType.LED_450_NM])

    with pytest.raises(ValueError):
        LedConfig(binding=BindingType.VIRTUAL, available_leds=[LEDType.NO_LED])

    with pytest.raises(ValueError, match="missing mappings"):
        LedConfig(
            binding=BindingType.VIRTUAL,
            available_leds=[LEDType.LED_450_NM, LEDType.LED_515_NM],
            led_to_internal={LEDType.LED_450_NM: "450"},
        )


def test_led_config_allows_extra_internal_mappings():
    config = LedConfig(
        binding=BindingType.VIRTUAL,
        available_leds=[LEDType.LED_450_NM],
        led_to_internal={
            LEDType.LED_450_NM: "450",
            LEDType.LED_515_NM: "515",
        },
    )

    assert config.led_to_internal == {
        LEDType.LED_450_NM: "450",
        LEDType.LED_515_NM: "515",
    }


def test_led_source_rejects_invalid_brightness():
    source = make_virtual_source()

    with pytest.raises(ValueError):
        source.set_led(LEDType.LED_450_NM, brightness=-1)

    with pytest.raises(ValueError):
        source.set_led(LEDType.LED_450_NM, brightness=101)


def test_led_source_rejects_unavailable_led():
    source = make_virtual_source()

    with pytest.raises(ValueError):
        source.set_led(LEDType.LED_515_NM, brightness=50)


def test_base_timed_led_starts_timer_and_callback_disables(monkeypatch):
    FakeTimer.instances = []
    monkeypatch.setattr("evomachine.peripherals.leds.threading.Timer", FakeTimer)
    source = make_virtual_source()

    source.set_led(LEDType.LED_450_NM, brightness=25, duration=500)

    assert source.led_is_on(LEDType.LED_450_NM)
    assert FakeTimer.instances[0].interval == 0.5
    assert FakeTimer.instances[0].started

    FakeTimer.instances[0].fire()

    assert not source.led_is_on(LEDType.LED_450_NM)
    assert source.commands[-1] == (LEDType.LED_450_NM, 0.0)


def test_disabling_led_cancels_existing_timer(monkeypatch):
    FakeTimer.instances = []
    monkeypatch.setattr("evomachine.peripherals.leds.threading.Timer", FakeTimer)
    source = make_virtual_source()

    source.set_led(LEDType.LED_450_NM, brightness=25, duration=500)
    source.disable_led(LEDType.LED_450_NM)

    assert FakeTimer.instances[0].cancelled


def test_led_source_rejects_negative_duration_before_command():
    source = make_virtual_source()

    with pytest.raises(ValueError, match="duration"):
        source.set_led(LEDType.LED_450_NM, brightness=25, duration=-1)

    assert source.commands == []


def test_led_manager_routes_leds_and_no_led_disables_all():
    source_450 = make_virtual_source()
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    source_515 = VirtualLedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.LED_515_NM],
    )
    source_515.initialise()
    manager = LedManager([source_450, source_515])

    manager.set_led(LEDType.LED_515_NM, brightness=33)
    manager.set_led(LEDType.NO_LED)

    assert source_515.commands[-2] == (LEDType.LED_515_NM, 33.0)
    assert source_450.commands[-1] == (LEDType.LED_450_NM, 0.0)
    assert source_515.commands[-1] == (LEDType.LED_515_NM, 0.0)


def test_led_manager_rejects_empty_duplicate_and_unknown_sources():
    source = make_virtual_source()
    duplicate = make_virtual_source()

    with pytest.raises(ValueError):
        LedManager([])
    with pytest.raises(ValueError):
        LedManager([source, duplicate])
    with pytest.raises(ValueError):
        LedManager([source]).set_led(LEDType.LED_515_NM)


def test_led_source_is_alive_queries_peripheral_when_check_alive_is_false():
    peripheral_ctrl = CountingPeripheralController()
    peripheral_ctrl.initialise()
    queries_after_initialise = peripheral_ctrl.alive_queries
    source = VirtualLedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.LED_450_NM],
        check_alive=False,
    )

    assert source.is_alive()
    assert peripheral_ctrl.alive_queries == queries_after_initialise + 1


def test_led_manager_is_alive_queries_all_sources():
    first_ctrl = CountingPeripheralController(alive_result=True)
    second_ctrl = CountingPeripheralController(alive_result=True)
    first_ctrl.initialise()
    second_ctrl.initialise()
    first_source = VirtualLedSource(
        peripheral_ctrl=first_ctrl,
        available_leds=[LEDType.LED_450_NM],
        check_alive=False,
    )
    second_source = VirtualLedSource(
        peripheral_ctrl=second_ctrl,
        available_leds=[LEDType.LED_515_NM],
        check_alive=False,
    )
    first_ctrl.alive_result = False

    assert not LedManager([first_source, second_source]).is_alive()
    assert first_ctrl.alive_queries == 2
    assert second_ctrl.alive_queries == 2


def test_tiger_led_source_sends_full_brightness_mapping():
    tiger = FakeTigerLedController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger, card_address_led=7)
    peripheral_ctrl.initialise()
    source = TigerLedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.TIGER_LED_1, LEDType.TIGER_LED_2],
    )
    source.initialise()

    source.set_led(LEDType.TIGER_LED_2, brightness=42.8)

    assert tiger.led_calls == [({"X": 0, "Y": 42}, 7)]


def test_syncboard_led_source_uses_native_duration_and_intensity(monkeypatch):
    FakeTimer.instances = []
    monkeypatch.setattr("evomachine.peripherals.leds.threading.Timer", FakeTimer)
    syncboard = FakeSyncBoardController()
    peripheral_ctrl = SyncBoardPeripheralController(syncboard=syncboard)
    peripheral_ctrl.initialise()
    source = SyncBoardLedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.LED_450_NM],
    )
    source.initialise()

    source.set_led(LEDType.LED_450_NM, brightness=29, duration=120000)

    assert syncboard.enabled_leds == [(1, 0.29, 120000)]
    assert FakeTimer.instances == []


def test_kwr103_led_source_initialises_and_maps_brightness_to_voltage():
    kwr103 = FakeKWR103()
    peripheral_ctrl = KWR103PeripheralController(kwr103=kwr103)
    peripheral_ctrl.initialise()
    source = KWR103LedSource(
        peripheral_ctrl=peripheral_ctrl,
        available_leds=[LEDType.LED_OVERHEAD],
    )

    source.initialise()
    source.set_led(LEDType.LED_OVERHEAD, brightness=50)
    source.set_led(LEDType.LED_OVERHEAD, brightness=0)

    assert kwr103.current_calls == [0.1]
    assert kwr103.output_calls == [False, True, False]
    assert kwr103.voltage_calls == [8.0]


def test_led_factory_creates_binding_sources():
    virtual_ctrl = VirtualPeripheralController()
    virtual_ctrl.initialise()
    source = LedFactory.create(
        LedConfig(
            binding=BindingType.VIRTUAL,
            available_leds=[LEDType.LED_450_NM],
        ),
        peripheral_controllers=virtual_ctrl,
    )

    assert isinstance(source, VirtualLedSource)


def test_led_brightness_annotations_use_brightness_type():
    source_hints = get_type_hints(LedSource.set_led, include_extras=True)
    manager_hints = get_type_hints(LedManager.set_led, include_extras=True)

    assert source_hints["brightness"] == BrightnessType
    assert manager_hints["brightness"] == BrightnessType

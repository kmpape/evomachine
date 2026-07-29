"""Physical SyncBoard and ASI Tiger LED tests.

Check optical safety before enabling illumination. Tests use the brightness and
pulse duration configured through the EVOMACHINE_LED_TEST_* variables.

Run all LED cases:
EVOMACHINE_RUN_ILLUMINATION=1 uv run pytest tests/hardware/test_leds_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os

import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.types import LEDType


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_ILLUMINATION") != "1", reason="Set EVOMACHINE_RUN_ILLUMINATION=1 after checking optical safety.")]


@pytest.fixture(scope="module")
def physical_led_manager():
    syncboard = PeripheralControllerFactory.create(SerialPeripheralControllerConfig(binding=BindingType.SYNCBOARD, hwid=os.getenv("EVOMACHINE_SYNCBOARD_HWID", "USB VID:PID=16C0:0483 SER=14582700 LOCATION=7-2:1.0")))
    tiger = PeripheralControllerFactory.create(SerialPeripheralControllerConfig(binding=BindingType.ASI_TIGER, hwid=os.getenv("EVOMACHINE_TIGER_HWID", "USB VID:PID=10C4:EA60 SER=0001 LOCATION=5-3")))
    sources = [
        LedFactory.create(LedConfig(binding=BindingType.SYNCBOARD, available_leds=[LEDType.LED_450_NM]), peripheral_controllers=[syncboard, tiger]),
        LedFactory.create(LedConfig(binding=BindingType.ASI_TIGER, available_leds=[LEDType.LED_OVERHEAD_TIGER]), peripheral_controllers=[syncboard, tiger]),
    ]
    manager = LedManager(sources, name="Physical LED Manager")
    try:
        manager.initialise()
        manager.disable_led()
        yield manager
    finally:
        with suppress(Exception):
            manager.disable_led()
        with suppress(Exception):
            manager.finalise()
        with suppress(Exception):
            syncboard.shutdown()
        with suppress(Exception):
            tiger.shutdown()


def test_led_manager_initialises_both_sources(physical_led_manager) -> None:
    assert physical_led_manager.is_initialised()
    assert physical_led_manager.is_alive()
    assert set(physical_led_manager.get_available_leds()) == {LEDType.LED_450_NM, LEDType.LED_OVERHEAD_TIGER}
    for led_type in physical_led_manager.get_available_leds():
        state = physical_led_manager.get_led_state(led_type)
        assert not state.is_on
        assert state.brightness == 0.0


@pytest.mark.parametrize("led_type", [LEDType.LED_450_NM, LEDType.LED_OVERHEAD_TIGER])
def test_led_manager_routes_safe_pulse(physical_led_manager, led_type: LEDType) -> None:
    brightness = float(os.getenv("EVOMACHINE_LED_TEST_BRIGHTNESS", "5"))
    try:
        physical_led_manager.set_led(led_type=led_type, brightness=brightness, duration=100)
        state = physical_led_manager.get_led_state(led_type)
        assert state.is_on
        assert state.brightness == pytest.approx(brightness)
    finally:
        physical_led_manager.disable_led()
    assert not physical_led_manager.get_led_state(led_type).is_on


def test_led_manager_no_led_command_disables_both_sources(physical_led_manager) -> None:
    brightness = float(os.getenv("EVOMACHINE_LED_TEST_BRIGHTNESS", "5"))
    try:
        for led_type in physical_led_manager.get_available_leds():
            physical_led_manager.set_led(led_type=led_type, brightness=brightness)
        physical_led_manager.set_led(LEDType.NO_LED)
        assert all(
            not physical_led_manager.get_led_state(led_type).is_on
            for led_type in physical_led_manager.get_available_leds()
        )
    finally:
        physical_led_manager.disable_led()

"""Physical ASI Tiger filter-wheel tests.

Ensure the wheel can rotate freely before running.

Run all filter-wheel cases:
EVOMACHINE_RUN_FILTERWHEEL=1 uv run pytest tests/hardware/test_filterwheel_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os
import time

import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.filterwheel import FilterWheelConfig, FilterWheelFactory
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.types import FilterWheelType


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_FILTERWHEEL") != "1", reason="Set EVOMACHINE_RUN_FILTERWHEEL=1.")]


def wait_for_filter_wheel(physical_filter_wheel) -> None:
    """Wait for Tiger to report idle, tolerating transient malformed replies."""
    timeout_s = float(os.getenv("EVOMACHINE_FILTER_TIMEOUT_S", "10"))
    settle_s = float(os.getenv("EVOMACHINE_FILTER_SETTLE_S", "1"))
    poll_s = 0.05
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            is_busy = physical_filter_wheel.tiger.is_busy()
        except ValueError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "ASI Tiger repeatedly returned malformed filter-wheel "
                    f"status replies for more than {timeout_s} seconds."
                ) from error
            time.sleep(poll_s)
            continue
        if not is_busy:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ASI Tiger filter wheel remained busy for more than {timeout_s} seconds.")
        time.sleep(poll_s)
    time.sleep(settle_s)


@pytest.fixture(scope="module")
def physical_filter_wheel():
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            hwid=os.getenv("EVOMACHINE_TIGER_HWID", "USB VID:PID=10C4:EA60 SER=0001 LOCATION=5-3"),
        ),
        card_address_filter_wheel=int(os.getenv("EVOMACHINE_FILTER_CARD_ADDRESS", "8")),
    )
    wheel = FilterWheelFactory.create(
        FilterWheelConfig(binding=BindingType.ASI_TIGER, available_filters=list(FilterWheelType)),
        peripheral_controllers=controller,
    )
    try:
        wheel.initialise()
        yield wheel
    finally:
        with suppress(Exception):
            wheel.finalise()
        with suppress(Exception):
            controller.shutdown()


def test_filter_wheel_initialises(physical_filter_wheel) -> None:
    assert physical_filter_wheel.is_initialised()
    assert physical_filter_wheel.is_alive()
    # ASI position readback is not implemented, so initialise must not claim a
    # physical position that it did not measure.
    assert physical_filter_wheel.get_filter_wheel() == FilterWheelType.UNKNOWN


def test_filter_wheel_exposes_configured_filters(physical_filter_wheel) -> None:
    assert set(physical_filter_wheel.get_available_filters()) == set(FilterWheelType)


def test_filter_wheel_moves_through_configured_positions(physical_filter_wheel) -> None:
    restore_filter = FilterWheelType.FILTER
    filters = [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_527nm, FilterWheelType.FILTER_592nm, FilterWheelType.NO_FILTER]
    try:
        for filter_type in filters:
            physical_filter_wheel.set_filter_wheel(filter_type, force=True)
            wait_for_filter_wheel(physical_filter_wheel)
            assert physical_filter_wheel.get_filter_wheel() == filter_type
    finally:
        physical_filter_wheel.set_filter_wheel(restore_filter, force=True)
        wait_for_filter_wheel(physical_filter_wheel)


def test_filter_wheel_repeated_target_uses_cached_position(physical_filter_wheel) -> None:
    target = FilterWheelType.FILTER
    physical_filter_wheel.set_filter_wheel(target, force=True)
    wait_for_filter_wheel(physical_filter_wheel)

    # With force=False this returns through the same cached-position path used
    # by normal acquisitions and does not issue another movement command.
    physical_filter_wheel.set_filter_wheel(target)
    assert physical_filter_wheel.get_filter_wheel() == target

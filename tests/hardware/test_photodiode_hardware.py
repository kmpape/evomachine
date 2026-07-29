"""Physical SyncBoard photodiode tests.

The default calibration range is 0 to 1 and can be overridden with
EVOMACHINE_PHOTODIODE_MIN and EVOMACHINE_PHOTODIODE_MAX.

Run all photodiode cases:
EVOMACHINE_RUN_PHOTODIODE=1 uv run pytest tests/hardware/test_photodiode_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os

import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.peripherals.photodiode import PhotodiodeConfig, PhotodiodeFactory, PhotodiodeReadingRange


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.getenv("EVOMACHINE_RUN_PHOTODIODE") != "1",
        reason="Set EVOMACHINE_RUN_PHOTODIODE=1.",
    ),
]


@pytest.fixture(scope="module")
def physical_photodiode():
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.SYNCBOARD,
            hwid=os.getenv(
                "EVOMACHINE_SYNCBOARD_HWID",
                "USB VID:PID=16C0:0483 SER=14582700 LOCATION=7-2:1.0",
            ),
        )
    )
    photodiode = PhotodiodeFactory.create(
        PhotodiodeConfig(
            binding=BindingType.SYNCBOARD,
            channel=int(os.getenv("EVOMACHINE_PHOTODIODE_CHANNEL", "8")),
            reading_range=PhotodiodeReadingRange(
                float(os.getenv("EVOMACHINE_PHOTODIODE_MIN", "0")),
                float(os.getenv("EVOMACHINE_PHOTODIODE_MAX", "1")),
            ),
        ),
        peripheral_controllers=controller,
    )
    try:
        photodiode.initialise()
        yield photodiode
    finally:
        with suppress(Exception):
            photodiode.finalise()
        with suppress(Exception):
            controller.shutdown()


def test_photodiode_initialises_and_is_alive(physical_photodiode) -> None:
    assert physical_photodiode.is_initialised()
    assert physical_photodiode.is_alive()


def test_photodiode_returns_finite_percentage(physical_photodiode) -> None:
    reading = physical_photodiode.read_photodiode()
    assert np.isfinite(reading)
    assert 0 <= reading <= 100


def test_photodiode_repeated_readings_are_valid(physical_photodiode) -> None:
    readings = [physical_photodiode.read_photodiode() for _ in range(3)]
    assert all(np.isfinite(reading) for reading in readings)
    assert all(0 <= reading <= 100 for reading in readings)

"""Physical ASI Tiger CRISP autofocus tests.

Prepare a suitable sample and objective before calibration. The basic command
only queries lifecycle/status. Calibration and lock operations require stronger
opt-in flags. Run the full command in file order so calibration precedes lock.
The CRISP card defaults to the repository hardware address (2); override it with
EVOMACHINE_CRISP_CARD_ADDRESS if the Tiger is configured differently.

Run status only:
EVOMACHINE_RUN_AUTOFOCUS=1 uv run pytest tests/hardware/test_autofocus_hardware.py -k "initialises_and_reports_status" -m hardware -v -s

Run status and calibration without locking:
EVOMACHINE_RUN_AUTOFOCUS=1 EVOMACHINE_RUN_AUTOFOCUS_CALIBRATION=1 uv run pytest tests/hardware/test_autofocus_hardware.py -m hardware -v -s

Run every case, including lock/unlock after calibration:
EVOMACHINE_RUN_AUTOFOCUS=1 EVOMACHINE_RUN_AUTOFOCUS_CALIBRATION=1 EVOMACHINE_RUN_AUTOFOCUS_LOCK=1 uv run pytest tests/hardware/test_autofocus_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os

import pytest

from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory
from evomachine.bindings.asitiger.card_addresses import CARD_ADDRESS_CRISP
from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.autofocus import AutofocusConfig, AutofocusFactory
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.types import AutoFocusStatusType


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.getenv("EVOMACHINE_RUN_AUTOFOCUS") != "1",
        reason="Set EVOMACHINE_RUN_AUTOFOCUS=1 after preparing the CRISP optical path.",
    ),
]


@pytest.fixture(scope="module")
def physical_autofocus():
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            hwid=os.getenv(
                "EVOMACHINE_TIGER_HWID",
                "USB VID:PID=10C4:EA60 SER=0001 LOCATION=5-3",
            ),
        ),
        card_address_crisp=int(
            os.getenv("EVOMACHINE_CRISP_CARD_ADDRESS", str(CARD_ADDRESS_CRISP))
        ),
    )
    config_factory = TigerAutofocusConfigFactory
    tiger_config = (
        config_factory.default_oil_config()
        if os.getenv("EVOMACHINE_AUTOFOCUS_OBJECTIVE", "air").lower() == "oil"
        else config_factory.default_config()
    )
    autofocus = AutofocusFactory.create(
        AutofocusConfig(binding=BindingType.ASI_TIGER),
        peripheral_controllers=controller,
        tiger_config=tiger_config,
    )
    try:
        autofocus.initialise()
        yield autofocus
    finally:
        with suppress(Exception):
            autofocus.disable()
        with suppress(Exception):
            autofocus.finalise()
        with suppress(Exception):
            controller.shutdown()


def test_autofocus_initialises_and_reports_status(physical_autofocus) -> None:
    assert physical_autofocus.is_initialised()
    assert physical_autofocus.is_alive()
    assert isinstance(physical_autofocus.get_status(), AutoFocusStatusType)


@pytest.mark.skipif(
    os.getenv("EVOMACHINE_RUN_AUTOFOCUS_CALIBRATION") != "1",
    reason="Set EVOMACHINE_RUN_AUTOFOCUS_CALIBRATION=1 with a suitable sample and objective.",
)
def test_autofocus_configures_and_calibrates_without_locking(physical_autofocus) -> None:
    assert physical_autofocus.initialise_autofocus(lock_after_initialise=False)
    assert not physical_autofocus.is_locked()


@pytest.mark.skipif(
    os.getenv("EVOMACHINE_RUN_AUTOFOCUS_LOCK") != "1",
    reason="Set EVOMACHINE_RUN_AUTOFOCUS_LOCK=1 only after successful CRISP calibration.",
)
def test_autofocus_locks_and_unlocks(physical_autofocus) -> None:
    physical_autofocus.lock()
    assert physical_autofocus.is_locked()
    physical_autofocus.unlock()
    assert not physical_autofocus.is_locked()

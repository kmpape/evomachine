from __future__ import annotations

import pytest

from evomachine.autofocus import AutofocusConfig, AutofocusFactory
from evomachine.bindings.asitiger.autofocus import (
    CRISPState,
    FakeTigerAutofocusController,
    TigerAutofocus,
    TigerAutofocusConfig,
    TigerAutofocusConfigFactory,
)
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.virtual.autofocus import VirtualAutofocus
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.types import AutoFocusStatusType


def _virtual_controller() -> VirtualPeripheralController:
    """
    Return an initialised virtual peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    VirtualPeripheralController
        Initialised virtual controller.
    """
    controller = VirtualPeripheralController()
    controller.initialise()
    return controller


def _tiger_controller(
        snr: float = 10,
        error: int = 200,
) -> TigerPeripheralController:
    """
    Return a TigerPeripheralController wrapping a fake CRISP controller.

    Parameters
    ----------
    snr
        Fake SNR value returned during autofocus setup.
    error
        Fake error value returned during autofocus setup.

    Returns
    -------
    TigerPeripheralController
        Controller wrapping a FakeTigerAutofocusController.
    """
    return TigerPeripheralController(
        tiger=FakeTigerAutofocusController(snr=snr, error=error),
        card_address_crisp=3,
    )


def _tiger_config() -> TigerAutofocusConfig:
    """
    Return a small valid Tiger autofocus configuration.

    Parameters
    ----------
    None

    Returns
    -------
    TigerAutofocusConfig
        Valid CRISP autofocus configuration.
    """
    return TigerAutofocusConfig(
        averaging=5,
        led_intensity=70,
        lock_range=0.1,
        loop_gain=10,
        update_rate=10,
        objective_na=0.9,
        min_snr=2,
        min_error=100,
    )


def test_autofocus_config_validation() -> None:
    """
    Check shared AutofocusConfig validation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = AutofocusConfig(binding=BindingType.VIRTUAL)
    assert config.binding == BindingType.VIRTUAL

    with pytest.raises(TypeError):
        AutofocusConfig(binding="virtual")
    with pytest.raises(TypeError):
        AutofocusConfig(binding=BindingType.VIRTUAL, name=123)
    with pytest.raises(TypeError):
        AutofocusConfig(binding=BindingType.VIRTUAL, check_initialised="yes")
    with pytest.raises(TypeError):
        AutofocusConfig(binding=BindingType.VIRTUAL, check_alive="yes")


def test_tiger_autofocus_config_validation_and_factory_defaults() -> None:
    """
    Check TigerAutofocusConfig validation and default factories.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    assert TigerAutofocusConfigFactory.default_config().objective_na == 0.9
    assert TigerAutofocusConfigFactory.default_oil_config().objective_na == 1.4

    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=-1, led_intensity=70, lock_range=0.1, loop_gain=10, update_rate=10, objective_na=0.9)
    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=5, led_intensity=101, lock_range=0.1, loop_gain=10, update_rate=10, objective_na=0.9)
    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=5, led_intensity=70, lock_range=1.0, loop_gain=10, update_rate=10, objective_na=0.9)
    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=5, led_intensity=70, lock_range=0.1, loop_gain=0, update_rate=10, objective_na=0.9)
    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=5, led_intensity=70, lock_range=0.1, loop_gain=10, update_rate=-1, objective_na=0.9)
    with pytest.raises(TypeError):
        TigerAutofocusConfig(averaging=5, led_intensity=70, lock_range=0.1, loop_gain=10, update_rate=10, objective_na=0)


def test_virtual_autofocus_lifecycle_and_state_transitions() -> None:
    """
    Check virtual autofocus lifecycle and lock/status transitions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    autofocus = AutofocusFactory.create(
        config=AutofocusConfig(binding=BindingType.VIRTUAL),
        peripheral_controllers=_virtual_controller(),
    )

    assert isinstance(autofocus, VirtualAutofocus)
    with pytest.raises(RuntimeError):
        autofocus.get_status()

    autofocus.initialise()
    assert autofocus.configure()
    assert autofocus.initialise_autofocus()
    assert autofocus.get_status() == AutoFocusStatusType.READY
    assert not autofocus.is_locked()

    autofocus.lock()
    assert autofocus.get_status() == AutoFocusStatusType.IN_FOCUS
    assert autofocus.is_locked()

    autofocus.unlock()
    assert autofocus.get_status() == AutoFocusStatusType.READY
    assert not autofocus.is_locked()

    autofocus.disable()
    assert autofocus.get_status() == AutoFocusStatusType.IDLE
    assert autofocus.command_history == [
        "configure",
        "initialise_autofocus",
        "lock",
        "unlock",
        "disable",
    ]


def test_autofocus_factory_rejects_unsupported_valid_binding() -> None:
    """
    Check shared BindingType values are scoped to autofocus factory support.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        AutofocusFactory.create(AutofocusConfig(binding=BindingType.KWR103))


def test_tiger_autofocus_configure_sends_expected_commands() -> None:
    """
    Check TigerAutofocus.configure sends CRISP settings in the expected order.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    controller = _tiger_controller()
    autofocus = TigerAutofocus(
        peripheral_ctrl=controller,
        tiger_config=_tiger_config(),
        pause_long=0,
        pause_short=0,
        sleep=lambda seconds: None,
    )

    autofocus.initialise()
    assert autofocus.configure()

    assert controller.tiger.commands == [
        ("state", CRISPState.UNLOCK),
        ("objective_na", 0.9),
        ("led_intensity", 70),
        ("loop_gain", 10),
        ("averaging", 5),
        ("update_rate", 10),
        ("lock_range", 0.1),
    ]


def test_tiger_autofocus_initialise_command_sequence_and_lock() -> None:
    """
    Check TigerAutofocus.initialise_autofocus command order and lock behavior.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    controller = _tiger_controller()
    autofocus = AutofocusFactory.create(
        config=AutofocusConfig(binding=BindingType.ASI_TIGER),
        peripheral_controllers=controller,
        tiger_config=_tiger_config(),
        pause_long=0,
        pause_short=0,
        sleep=lambda seconds: None,
    )

    autofocus.initialise()
    assert autofocus.initialise_autofocus(lock_after_initialise=True)

    assert controller.tiger.commands == [
        ("state", CRISPState.UNLOCK),
        ("objective_na", 0.9),
        ("led_intensity", 70),
        ("loop_gain", 10),
        ("averaging", 5),
        ("update_rate", 10),
        ("lock_range", 0.1),
        ("state", CRISPState.IDLE),
        ("state", CRISPState.SET_OFFSET),
        ("state", CRISPState.LOG_CAL),
        ("snr", None),
        ("state", CRISPState.DITHER),
        ("error", None),
        ("state", CRISPState.SET_GAIN),
        ("state", CRISPState.UNLOCK),
        ("state", CRISPState.LOCK),
    ]
    assert autofocus.get_status() == AutoFocusStatusType.IN_FOCUS
    assert autofocus.is_locked()


def test_tiger_autofocus_initialise_returns_false_for_low_snr_or_error() -> None:
    """
    Check TigerAutofocus.initialise_autofocus reports failed quality checks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    for controller in [_tiger_controller(snr=1, error=200), _tiger_controller(snr=10, error=50)]:
        autofocus = TigerAutofocus(
            peripheral_ctrl=controller,
            tiger_config=_tiger_config(),
            pause_long=0,
            pause_short=0,
            sleep=lambda seconds: None,
        )
        autofocus.initialise()

        assert not autofocus.initialise_autofocus(lock_after_initialise=True)
        assert ("state", CRISPState.LOCK) not in controller.tiger.commands


def test_tiger_autofocus_disable_unlock_and_status() -> None:
    """
    Check TigerAutofocus lock, unlock, disable, status, and locked-state queries.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    controller = _tiger_controller()
    autofocus = TigerAutofocus(
        peripheral_ctrl=controller,
        pause_long=0,
        pause_short=0,
        sleep=lambda seconds: None,
    )
    autofocus.initialise()

    autofocus.lock()
    assert autofocus.get_status() == AutoFocusStatusType.IN_FOCUS
    assert autofocus.is_locked()

    autofocus.unlock()
    assert autofocus.get_status() == AutoFocusStatusType.READY
    assert not autofocus.is_locked()

    autofocus.disable()
    assert autofocus.get_status() == AutoFocusStatusType.IDLE

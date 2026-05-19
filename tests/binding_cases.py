"""Configured fake and virtual binding cases for evomachine tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from evomachine.bindings.asitiger.filterwheel import FakeTigerFilterWheelController, TigerFilterWheel
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.asitiger.stage import FakeTigerStageController, TigerStage
from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.kwr103.KWR103Driver import KWR103
from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController
from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.bindings.virtual.filterwheel import VirtualFilterWheel
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.bindings.virtual.stage import VirtualStage
from evomachine.peripherals.filterwheel import FilterWheel
from evomachine.peripherals.stage import Stage
from evomachine.types import FilterWheelType
from tests.binding_test_config import BindingTestConfig


T = TypeVar("T")


@dataclass(frozen=True)
class BindingCase:
    """Named binding case selected by BindingTestConfig."""

    name: str
    binding: BindingType
    is_real_binding: bool = False


@dataclass(frozen=True)
class FactoryBindingCase(BindingCase):
    """Binding case with a concrete object factory."""

    make: Callable[[], Any] | None = None


@dataclass(frozen=True)
class StageBindingCase(BindingCase):
    """Stage binding case with a factory and readback capability flag."""

    make_stage: Callable[[], Stage] | None = None
    supports_hardware_readback: bool = True


@dataclass(frozen=True)
class FilterWheelBindingCase(BindingCase):
    """Filter wheel binding case with a concrete factory."""

    make_filter_wheel: Callable[[], FilterWheel] | None = None


class FakeSyncBoardController:
    """Small SyncBoard fake for configured peripheral-controller cases."""

    def __init__(self) -> None:
        """
        Initialise fake SyncBoard lifecycle state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._is_initialised = True

    def initialise(self, force_init: bool = False) -> None:
        """
        Mark the fake SyncBoard as initialised.

        Parameters
        ----------
        force_init
            Ignored force flag matching the real binding signature.

        Returns
        -------
        None
        """
        self._is_initialised = True

    def is_initialised(self) -> bool:
        """
        Return whether the fake SyncBoard is initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the fake controller is initialised.
        """
        return self._is_initialised

    def disable_system(self) -> None:
        """
        Accept a fake system disable command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def finalise(self) -> None:
        """
        Mark the fake SyncBoard as finalised.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._is_initialised = False


def make_virtual_stage() -> VirtualStage:
    """
    Return an initialised virtual stage case.

    Parameters
    ----------
    None

    Returns
    -------
    VirtualStage
        Initialised virtual stage with delta_fov 100.
    """
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(peripheral_ctrl=peripheral_ctrl, delta_fov=100.0)
    stage.initialise()
    return stage


def make_fake_tiger_stage() -> TigerStage:
    """
    Return an initialised fake ASI Tiger stage case.

    Parameters
    ----------
    None

    Returns
    -------
    TigerStage
        Initialised Tiger stage backed by FakeTigerStageController.
    """
    tiger = FakeTigerStageController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()
    stage = TigerStage(peripheral_ctrl=peripheral_ctrl, delta_fov=100.0)
    stage.initialise()
    return stage


AVAILABLE_FILTERS = [
    FilterWheelType.FILTER,
    FilterWheelType.FILTER_527nm,
    FilterWheelType.BLOCKING,
]
"""Filter list shared by configured filter wheel cases."""


def make_virtual_filter_wheel(
        current_filter_type: FilterWheelType = FilterWheelType.UNKNOWN,
        check_initialised: bool = True,
        check_alive: bool = True,
) -> VirtualFilterWheel:
    """
    Return a virtual filter wheel case.

    Parameters
    ----------
    current_filter_type
        Initial filter returned when the virtual wheel is initialised.
    check_initialised
        Whether public commands require initialisation.
    check_alive
        Whether public commands require alive checks.

    Returns
    -------
    VirtualFilterWheel
        Virtual filter wheel with an initialised peripheral controller.
    """
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    return VirtualFilterWheel(
        peripheral_ctrl=peripheral_ctrl,
        available_filters=AVAILABLE_FILTERS,
        current_filter_type=current_filter_type,
        check_initialised=check_initialised,
        check_alive=check_alive,
    )


def make_fake_tiger_filter_wheel() -> TigerFilterWheel:
    """
    Return a fake ASI Tiger filter wheel case.

    Parameters
    ----------
    None

    Returns
    -------
    TigerFilterWheel
        Tiger filter wheel backed by FakeTigerFilterWheelController.
    """
    tiger = FakeTigerFilterWheelController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()
    filter_wheel = TigerFilterWheel(peripheral_ctrl=peripheral_ctrl, available_filters=AVAILABLE_FILTERS)
    filter_wheel.initialise()
    return filter_wheel


def make_virtual_peripheral_controller() -> VirtualPeripheralController:
    """
    Return a virtual peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    VirtualPeripheralController
        Fresh virtual peripheral controller.
    """
    return VirtualPeripheralController()


def make_fake_tiger_peripheral_controller() -> TigerPeripheralController:
    """
    Return a fake ASI Tiger peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    TigerPeripheralController
        Tiger controller backed by FakeTigerStageController.
    """
    return TigerPeripheralController(tiger=FakeTigerStageController())


def make_fake_syncboard_peripheral_controller() -> SyncBoardPeripheralController:
    """
    Return a fake SyncBoard peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    SyncBoardPeripheralController
        SyncBoard controller backed by FakeSyncBoardController.
    """
    return SyncBoardPeripheralController(syncboard=FakeSyncBoardController())


def make_fake_kwr103_peripheral_controller() -> KWR103PeripheralController:
    """
    Return a fake KWR103 peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    KWR103PeripheralController
        KWR103 controller using a dummy serial port object.
    """
    return KWR103PeripheralController(kwr103=KWR103(port="/dev/null"))


STAGE_CASES = {
    "virtual": StageBindingCase("virtual", BindingType.VIRTUAL, make_stage=make_virtual_stage),
    "asitiger_fake": StageBindingCase("asitiger_fake", BindingType.ASI_TIGER, make_stage=make_fake_tiger_stage),
}
"""Configured stage case registry keyed by binding_test_config.json names."""

FILTERWHEEL_CASES = {
    "virtual": FilterWheelBindingCase(
        "virtual",
        BindingType.VIRTUAL,
        make_filter_wheel=lambda: make_virtual_filter_wheel(),
    ),
    "asitiger_fake": FilterWheelBindingCase(
        "asitiger_fake",
        BindingType.ASI_TIGER,
        make_filter_wheel=make_fake_tiger_filter_wheel,
    ),
}
"""Configured filter wheel case registry keyed by binding_test_config.json names."""

PERIPHERAL_CASES = {
    "virtual": FactoryBindingCase("virtual", BindingType.VIRTUAL, make=make_virtual_peripheral_controller),
    "asitiger_fake": FactoryBindingCase("asitiger_fake", BindingType.ASI_TIGER, make=make_fake_tiger_peripheral_controller),
    "syncboard_fake": FactoryBindingCase(
        "syncboard_fake",
        BindingType.SYNCBOARD,
        make=make_fake_syncboard_peripheral_controller,
    ),
    "kwr103_fake": FactoryBindingCase("kwr103_fake", BindingType.KWR103, make=make_fake_kwr103_peripheral_controller),
}
"""Configured peripheral controller case registry keyed by binding_test_config.json names."""

LED_CASES = {
    "virtual": BindingCase("virtual", BindingType.VIRTUAL),
    "asitiger_fake": BindingCase("asitiger_fake", BindingType.ASI_TIGER),
    "syncboard_fake": BindingCase("syncboard_fake", BindingType.SYNCBOARD),
    "kwr103_fake": BindingCase("kwr103_fake", BindingType.KWR103),
}
"""Configured LED case registry keyed by binding_test_config.json names."""

FILTER_ONLY_CASES = {
    "virtual": BindingCase("virtual", BindingType.VIRTUAL),
    "em_dmd_window_fake": BindingCase("em_dmd_window_fake", BindingType.EM_DMD_WINDOW),
    "pygame_fake": BindingCase("pygame_fake", BindingType.PYGAME),
    "mmc_fake": BindingCase("mmc_fake", BindingType.MMC),
    "pvcam_fake": BindingCase("pvcam_fake", BindingType.PVCAM),
    "asitiger_fake": BindingCase("asitiger_fake", BindingType.ASI_TIGER),
    "syncboard_fake": BindingCase("syncboard_fake", BindingType.SYNCBOARD),
    "kwr103_fake": BindingCase("kwr103_fake", BindingType.KWR103),
}
"""Generic case registry for groups whose factories live in focused test modules."""

AUTOFOCUS_CASES = {
    "virtual": BindingCase("virtual", BindingType.VIRTUAL),
    "asitiger_fake": BindingCase("asitiger_fake", BindingType.ASI_TIGER),
}
"""Configured autofocus case registry keyed by binding_test_config.json names."""


def selected_cases(
        config: BindingTestConfig,
        configured_names: list[str],
        registry: dict[str, T],
        group_name: str,
) -> list[T]:
    """
    Return configured cases from a named registry.

    Parameters
    ----------
    config
        Loaded binding test configuration.
    configured_names
        Case names selected for one device group.
    registry
        Available fake, virtual, or real cases keyed by config name.
    group_name
        Human-readable group name used in error messages.

    Returns
    -------
    list[T]
        Selected case objects in config order.
    """
    selected: list[T] = []
    for name in configured_names:
        if name not in registry:
            raise ValueError(f"Unknown {group_name} binding test case: {name}.")
        case = registry[name]
        if isinstance(case, BindingCase) and case.is_real_binding and not config.use_real_bindings:
            continue
        selected.append(case)
    return selected


def stage_cases(config: BindingTestConfig) -> list[StageBindingCase]:
    """
    Return stage cases selected by BindingTestConfig.

    Parameters
    ----------
    config
        Loaded binding test configuration.

    Returns
    -------
    list[StageBindingCase]
        Configured stage cases.
    """
    return selected_cases(config, config.stage_bindings, STAGE_CASES, "stage")


def filterwheel_cases(config: BindingTestConfig) -> list[FilterWheelBindingCase]:
    """
    Return filter wheel cases selected by BindingTestConfig.

    Parameters
    ----------
    config
        Loaded binding test configuration.

    Returns
    -------
    list[FilterWheelBindingCase]
        Configured filter wheel cases.
    """
    return selected_cases(config, config.filterwheel_bindings, FILTERWHEEL_CASES, "filterwheel")


def simple_cases(config: BindingTestConfig, names: list[str], group_name: str) -> list[BindingCase]:
    """
    Return generic binding cases selected by BindingTestConfig.

    Parameters
    ----------
    config
        Loaded binding test configuration.
    names
        Case names selected for one device group.
    group_name
        Human-readable group name used in error messages.

    Returns
    -------
    list[BindingCase]
        Configured generic binding cases.
    """
    return selected_cases(config, names, FILTER_ONLY_CASES, group_name)


def peripheral_cases(config: BindingTestConfig) -> list[FactoryBindingCase]:
    """
    Return peripheral controller cases selected by BindingTestConfig.

    Parameters
    ----------
    config
        Loaded binding test configuration.

    Returns
    -------
    list[FactoryBindingCase]
        Configured peripheral controller cases.
    """
    return selected_cases(config, config.peripheral_bindings, PERIPHERAL_CASES, "peripheral")

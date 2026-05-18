import pytest

from evomachine.bindings.asitiger.filterwheel import FakeTigerFilterWheelController, TigerFilterWheel
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.virtual.filterwheel import VirtualFilterWheel
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.filterwheel import FilterWheelConfig, FilterWheelFactory
from evomachine.bindings.binding_types import BindingType
from evomachine.types import FilterWheelType


# TODO(CODEX): This and all other test files should grab the settings from evomachine
AVAILABLE_FILTERS = [
    FilterWheelType.FILTER,
    FilterWheelType.FILTER_527nm,
    FilterWheelType.BLOCKING,
]

def make_virtual_filter_wheel(
        current_filter_type: FilterWheelType = FilterWheelType.UNKNOWN,
        check_initialised: bool = True,
        check_alive: bool = True,
) -> VirtualFilterWheel:
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    return VirtualFilterWheel(
        peripheral_ctrl=peripheral_ctrl,
        available_filters=AVAILABLE_FILTERS,
        current_filter_type=current_filter_type,
        check_initialised=check_initialised,
        check_alive=check_alive,
    )


def test_filter_wheel_config_requires_non_empty_filter_list():
    with pytest.raises(TypeError):
        FilterWheelConfig(
            binding="virtual",
            available_filters=[FilterWheelType.FILTER],
        )

    with pytest.raises(TypeError):
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=[FilterWheelType.FILTER],
            check_alive="yes",
        )

    with pytest.raises(TypeError):
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=FilterWheelType.FILTER,
        )

    with pytest.raises(ValueError):
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=[],
        )

    with pytest.raises(TypeError):
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=[FilterWheelType.FILTER, "bad"],
        )


def test_virtual_filter_wheel_initialise_reads_current_position():
    filter_wheel = make_virtual_filter_wheel(current_filter_type=FilterWheelType.FILTER_527nm)

    assert filter_wheel.get_filter_wheel() == FilterWheelType.UNKNOWN
    filter_wheel.initialise()

    assert filter_wheel.is_initialised()
    assert filter_wheel.is_alive()
    assert filter_wheel.get_filter_wheel() == FilterWheelType.FILTER_527nm


def test_filter_wheel_set_updates_current_position():
    filter_wheel = make_virtual_filter_wheel()
    filter_wheel.initialise()

    filter_wheel.set_filter_wheel(FilterWheelType.BLOCKING)

    assert filter_wheel.get_filter_wheel() == FilterWheelType.BLOCKING


def test_filter_wheel_rejects_invalid_filter_type():
    filter_wheel = make_virtual_filter_wheel()
    filter_wheel.initialise()

    with pytest.raises(TypeError):
        filter_wheel.set_filter_wheel("bad")


def test_filter_wheel_rejects_unavailable_filter():
    filter_wheel = make_virtual_filter_wheel()
    filter_wheel.initialise()

    with pytest.raises(ValueError):
        filter_wheel.set_filter_wheel(FilterWheelType.NO_FILTER)


def test_filter_wheel_readiness_checks_can_be_disabled():
    filter_wheel = make_virtual_filter_wheel(check_initialised=False, check_alive=False)

    filter_wheel.set_filter_wheel(FilterWheelType.FILTER)

    assert filter_wheel.get_filter_wheel() == FilterWheelType.FILTER


def test_filter_wheel_set_before_initialise_raises_by_default():
    filter_wheel = make_virtual_filter_wheel()

    with pytest.raises(RuntimeError):
        filter_wheel.set_filter_wheel(FilterWheelType.FILTER)


def test_tiger_filter_wheel_initialise_reads_without_setting():
    tiger = FakeTigerFilterWheelController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()
    filter_wheel = TigerFilterWheel(peripheral_ctrl=peripheral_ctrl, available_filters=AVAILABLE_FILTERS)

    filter_wheel.initialise()

    assert filter_wheel.get_filter_wheel() == FilterWheelType.UNKNOWN
    assert tiger.filter_wheel_calls == []


def test_tiger_filter_wheel_skips_same_filter_unless_forced():
    tiger = FakeTigerFilterWheelController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger, card_address_filter_wheel=9)
    peripheral_ctrl.initialise()
    filter_wheel = TigerFilterWheel(
        peripheral_ctrl=peripheral_ctrl,
        available_filters=AVAILABLE_FILTERS,
    )
    filter_wheel.initialise()

    filter_wheel.set_filter_wheel(FilterWheelType.FILTER_527nm)
    filter_wheel.set_filter_wheel(FilterWheelType.FILTER_527nm)
    filter_wheel.set_filter_wheel(FilterWheelType.FILTER_527nm, force=True)

    assert tiger.filter_wheel_calls == [(2, 9), (2, 9)]
    assert filter_wheel.get_filter_wheel() == FilterWheelType.FILTER_527nm


def test_tiger_filter_wheel_uses_custom_position_mapping():
    tiger = FakeTigerFilterWheelController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger, card_address_filter_wheel=10)
    peripheral_ctrl.initialise()
    filter_wheel = TigerFilterWheel(
        peripheral_ctrl=peripheral_ctrl,
        available_filters=AVAILABLE_FILTERS,
        filter_wheel_settings={FilterWheelType.FILTER: 42},
    )
    filter_wheel.initialise()

    filter_wheel.set_filter_wheel(FilterWheelType.FILTER)

    assert tiger.filter_wheel_calls == [(42, 10)]


def test_filter_wheel_factory_creates_virtual_filter_wheel():
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    filter_wheel = FilterWheelFactory.create(
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=AVAILABLE_FILTERS,
            name="Debug Filter Wheel",
        ),
        peripheral_controllers=peripheral_ctrl,
        current_filter_type=FilterWheelType.FILTER,
    )

    assert isinstance(filter_wheel, VirtualFilterWheel)
    assert filter_wheel.name == "Debug Filter Wheel"
    filter_wheel.initialise()
    assert filter_wheel.get_filter_wheel() == FilterWheelType.FILTER


def test_filter_wheel_factory_requires_tiger_peripheral_controller_for_asitiger():
    with pytest.raises(ValueError):
        FilterWheelFactory.create(
            FilterWheelConfig(
                binding=BindingType.ASI_TIGER,
                available_filters=AVAILABLE_FILTERS,
            )
        )


def test_filter_wheel_factory_creates_asitiger_filter_wheel():
    tiger = FakeTigerFilterWheelController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger, card_address_filter_wheel=11)
    peripheral_ctrl.initialise()

    filter_wheel = FilterWheelFactory.create(
        FilterWheelConfig(
            binding=BindingType.ASI_TIGER,
            available_filters=AVAILABLE_FILTERS,
            name="Fake Tiger Filter Wheel",
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert isinstance(filter_wheel, TigerFilterWheel)
    assert filter_wheel.name == "Fake Tiger Filter Wheel"
    assert filter_wheel.peripheral_ctrl.card_address_filter_wheel == 11

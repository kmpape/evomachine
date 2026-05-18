import pytest
from typing import get_args, get_origin, Annotated

from evomachine.bindings.binding_types import BindingType
from evomachine.types import (
    AutoFocusStatusType,
    AxisType,
    FilterWheelType,
    FovDirectionType,
    LEDType,
    PositiveScalingType,
)


def test_evo_type_helpers_return_expected_members():
    assert AxisType.from_string("X") == AxisType.X
    assert AxisType.from_string("missing") is None
    assert AutoFocusStatusType.from_flag("F") == AutoFocusStatusType.IN_FOCUS
    assert AxisType.get_all() == [AxisType.X, AxisType.Y, AxisType.Z]
    assert AxisType.get_all_values() == [0, 1, 2]
    assert AxisType.get_all_names() == ["X", "Y", "Z"]
    assert AxisType.get_name(1) == "Y"
    assert AxisType.get_name(99) == ""
    assert AxisType.get_dict() == {0: "X", 1: "Y", 2: "Z"}
    assert str(AxisType.Z) == "Z"
    assert FovDirectionType.get_all_names() == ["UP", "DOWN", "LEFT", "RIGHT", "HOME"]


def test_evo_type_from_flag_rejects_unknown_flag():
    with pytest.raises(ValueError):
        AutoFocusStatusType.from_flag("not-a-flag")


def test_positive_scaling_type_documents_positive_float_values():
    assert get_origin(PositiveScalingType) is Annotated
    assert get_args(PositiveScalingType) == (float, "value > 0")


def test_protocol_facing_enum_values_are_stable():
    assert AxisType.X.value == 0
    assert AxisType.Y.value == 1
    assert AxisType.Z.value == 2

    assert LEDType.NO_LED.value == -1
    assert LEDType.LED_385_NM.value == 0
    assert LEDType.LED_450_NM.value == 1
    assert LEDType.LED_515_NM.value == 2
    assert LEDType.LED_565_NM.value == 3
    assert LEDType.LED_645_NM.value == 4
    assert LEDType.LED_OVERHEAD.value == 5
    assert LEDType.LED_OVERHEAD_TIGER.value == 6

    assert FilterWheelType.UNKNOWN.value == -1
    assert FilterWheelType.FILTER.value == 0
    assert FilterWheelType.FILTER_465nm.value == 1
    assert FilterWheelType.FILTER_527nm.value == 2
    assert FilterWheelType.FILTER_592nm.value == 3
    assert FilterWheelType.NO_FILTER.value == 4
    assert FilterWheelType.BLOCKING.value == 5


def test_binding_enum_members_are_present():
    assert BindingType.get_all_names() == [
        "VIRTUAL",
        "ASI_TIGER",
        "SYNCBOARD",
        "KWR103",
        "EM_DMD_WINDOW",
        "PYGAME",
        "MMC",
        "PVCAM",
    ]

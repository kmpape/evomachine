"""Physical ASI Tiger stage tests.

Check stage and objective clearance first. The standard suite moves Z; the XY
case requires an additional explicit flag. Each movement test restores its
starting coordinate in cleanup.

Run Z and non-XY cases:
EVOMACHINE_RUN_STAGE_MOVEMENT=1 uv run pytest tests/hardware/test_stage_hardware.py -m hardware -v -s

Run every case, including XY movement:
EVOMACHINE_RUN_STAGE_MOVEMENT=1 EVOMACHINE_RUN_STAGE_XY_MOVEMENT=1 uv run pytest tests/hardware/test_stage_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os

import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.coordinates import Coordinate
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.peripherals.stage import StageConfig, StageFactory
from evomachine.types import AxisType


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_STAGE_MOVEMENT") != "1", reason="Set EVOMACHINE_RUN_STAGE_MOVEMENT=1 after checking stage clearance.")]


@pytest.fixture(scope="module")
def physical_stage():
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            hwid=os.getenv("EVOMACHINE_TIGER_HWID", "USB VID:PID=10C4:EA60 SER=0001 LOCATION=5-3"),
        )
    )
    stage = StageFactory.create(
        StageConfig(binding=BindingType.ASI_TIGER, fov_step_size=100.0),
        peripheral_controllers=controller,
    )
    try:
        stage.initialise()
        yield stage
    finally:
        with suppress(Exception):
            stage.finalise()
        with suppress(Exception):
            controller.shutdown()


def test_stage_initialises_and_reports_coordinates(physical_stage) -> None:
    assert physical_stage.is_initialised()
    assert physical_stage.is_alive()
    coordinate = physical_stage.get_coordinates(query_hardware=True)
    assert coordinate.x is not None
    assert coordinate.y is not None
    assert coordinate.z is not None


def test_stage_moves_z_and_restores_start(physical_stage) -> None:
    start = physical_stage.get_coordinates(query_hardware=True)
    delta_z = float(os.getenv("EVOMACHINE_STAGE_TEST_DELTA_Z", "20"))
    tolerance_z = float(os.getenv("EVOMACHINE_STAGE_Z_TOLERANCE", "1"))
    target = Coordinate(None, None, start.z + delta_z)
    if physical_stage.coordinate_is_out_of_bounds(target):
        pytest.skip(f"Requested stage test target is outside configured limits: {target}")
    try:
        physical_stage.move(target=target, block=True)
        reached = physical_stage.get_coordinates(query_hardware=True)
        assert reached.z == pytest.approx(target.z, abs=tolerance_z)
    finally:
        physical_stage.move(target=Coordinate(None, None, start.z), block=True)
    restored = physical_stage.get_coordinates(query_hardware=True)
    assert restored.z == pytest.approx(start.z, abs=tolerance_z)


def test_stage_reports_ordered_limits(physical_stage) -> None:
    low, high = physical_stage.get_stage_limits()
    for low_value, high_value in zip(
        (low.x, low.y, low.z),
        (high.x, high.y, high.z),
    ):
        assert low_value is not None and high_value is not None
        assert np.isfinite(low_value) and np.isfinite(high_value)
        assert low_value <= high_value


def test_stage_old_registered_position_api_is_removed(physical_stage) -> None:
    assert not hasattr(physical_stage, "get_pos")
    assert not hasattr(physical_stage, "set_pos_id_to_coordinate")
    assert not hasattr(physical_stage, "UNKNOWN_POSITION_ID")


def test_stage_fov_list_checks_autofocus_z_rules(physical_stage) -> None:
    assert physical_stage.set_fov_id_to_coordinate(
        {1: Coordinate(1, 2, 3)},
        use_autofocus=False,
    )
    assert not physical_stage.set_fov_id_to_coordinate(
        {1: Coordinate(1, 2, None)},
        use_autofocus=False,
    )
    assert not physical_stage.set_fov_id_to_coordinate(
        {1: Coordinate(1, 2, 3)},
        use_autofocus=True,
    )
    assert physical_stage.set_fov_id_to_coordinate(
        {1: Coordinate(1, 2, None)},
        use_autofocus=True,
    )


def test_stage_rejects_out_of_bounds_coordinate(physical_stage) -> None:
    with pytest.raises(ValueError):
        physical_stage.move(Coordinate(2e7, None, None))


def test_stage_partial_coordinate_preserves_unset_axes_and_restores_start(
        physical_stage,
) -> None:
    start = physical_stage.get_coordinates(query_hardware=True)
    delta_z = float(os.getenv("EVOMACHINE_STAGE_TEST_DELTA_Z", "20"))
    tolerance_z = float(os.getenv("EVOMACHINE_STAGE_Z_TOLERANCE", "1"))
    target = Coordinate(None, None, start.z + delta_z)
    if physical_stage.coordinate_is_out_of_bounds(target):
        pytest.skip(f"Requested stage test target is outside configured limits: {target}")

    try:
        physical_stage.move(target, block=True)
        reached = physical_stage.get_coordinates(query_hardware=True)
        assert reached.x == pytest.approx(start.x, abs=tolerance_z)
        assert reached.y == pytest.approx(start.y, abs=tolerance_z)
        assert reached.z == pytest.approx(target.z, abs=tolerance_z)
    finally:
        physical_stage.move(Coordinate(None, None, start.z), block=True)


def test_stage_empty_coordinate_is_noop(physical_stage) -> None:
    start = physical_stage.get_coordinates(query_hardware=True)

    physical_stage.move(Coordinate.none_coordinate(), block=True)

    assert physical_stage.get_coordinates(query_hardware=True) == start


def test_stage_can_filter_returned_axes(physical_stage) -> None:
    coordinate = physical_stage.get_coordinates(
        axes=[AxisType.X, AxisType.Z],
        query_hardware=True,
    )

    assert coordinate.x is not None
    assert coordinate.y is None
    assert coordinate.z is not None


@pytest.mark.skipif(
    os.getenv("EVOMACHINE_RUN_STAGE_XY_MOVEMENT") != "1",
    reason="Set EVOMACHINE_RUN_STAGE_XY_MOVEMENT=1 after checking XY clearance.",
)
def test_stage_moves_xy_and_restores_start(physical_stage) -> None:
    start = physical_stage.get_coordinates(query_hardware=True)
    delta_xy = float(os.getenv("EVOMACHINE_STAGE_TEST_DELTA_XY", "20"))
    tolerance_xy = float(os.getenv("EVOMACHINE_STAGE_XY_TOLERANCE", "5"))
    target = Coordinate(start.x + delta_xy, start.y + delta_xy, None)
    if physical_stage.coordinate_is_out_of_bounds(target):
        pytest.skip(f"Requested stage test target is outside configured limits: {target}")

    try:
        physical_stage.move(target, block=True)
        reached = physical_stage.get_coordinates(query_hardware=True)
        assert reached.x == pytest.approx(target.x, abs=tolerance_xy)
        assert reached.y == pytest.approx(target.y, abs=tolerance_xy)
    finally:
        physical_stage.move(Coordinate(start.x, start.y, None), block=True)

    restored = physical_stage.get_coordinates(query_hardware=True)
    assert restored.x == pytest.approx(start.x, abs=tolerance_xy)
    assert restored.y == pytest.approx(start.y, abs=tolerance_xy)

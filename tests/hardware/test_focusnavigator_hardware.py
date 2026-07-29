"""Physical FocusNavigator and XY-stage integration test.

Check stage and objective clearance. This test moves diagonally in XY and then
restores the starting coordinate.

Run the focus-navigation case:
EVOMACHINE_RUN_FOCUS_NAVIGATOR=1 EVOMACHINE_RUN_STAGE_MOVEMENT=1 EVOMACHINE_RUN_STAGE_XY_MOVEMENT=1 uv run pytest tests/hardware/test_focusnavigator_hardware.py -m hardware -v -s
"""

import os

import pytest

from evomachine.coordinates import Coordinate
from evomachine.navigation import FocusNavigator, FocusNavigatorConfig
from evomachine.types import FocusStatusType
from tests.hardware.test_stage_hardware import physical_stage as _physical_stage  # noqa: F401


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        any(
            os.getenv(variable) != "1"
            for variable in (
                "EVOMACHINE_RUN_FOCUS_NAVIGATOR",
                "EVOMACHINE_RUN_STAGE_MOVEMENT",
                "EVOMACHINE_RUN_STAGE_XY_MOVEMENT",
            )
        ),
        reason=(
            "Set EVOMACHINE_RUN_FOCUS_NAVIGATOR=1, EVOMACHINE_RUN_STAGE_MOVEMENT=1, "
            "and EVOMACHINE_RUN_STAGE_XY_MOVEMENT=1 after checking XY clearance."
        ),
    ),
]


def test_focus_navigator_moves_between_real_fovs_and_restores_start(
        _physical_stage,  # noqa: F811
) -> None:
    start = _physical_stage.get_coordinates(query_hardware=True)
    delta_xy = float(os.getenv("EVOMACHINE_STAGE_TEST_DELTA_XY", "20"))
    tolerance_xy = float(os.getenv("EVOMACHINE_STAGE_XY_TOLERANCE", "5"))
    second = Coordinate(start.x + delta_xy, start.y + delta_xy, start.z)
    if _physical_stage.coordinate_is_out_of_bounds(second):
        pytest.skip(f"Requested navigator target is outside configured limits: {second}")
    navigator = FocusNavigator(
        stage=_physical_stage,
        config=FocusNavigatorConfig(use_autofocus=False),
    )
    navigator.initialise_fovs({0: start, 1: second}, use_autofocus=False)

    try:
        record = navigator.move(1)
        reached = _physical_stage.get_coordinates(query_hardware=True)
        assert reached.x == pytest.approx(second.x, abs=tolerance_xy)
        assert reached.y == pytest.approx(second.y, abs=tolerance_xy)
        assert record.software_focus_status == FocusStatusType.UNKNOWN
        assert navigator.get_current_fov_id() == 1
    finally:
        navigator.move(0, manage_focus=False)

    restored = _physical_stage.get_coordinates(query_hardware=True)
    assert restored.x == pytest.approx(start.x, abs=tolerance_xy)
    assert restored.y == pytest.approx(start.y, abs=tolerance_xy)

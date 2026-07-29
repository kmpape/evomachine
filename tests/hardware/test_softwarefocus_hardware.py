"""Physical camera-and-stage software-focus integration test.

Close camera notebooks, provide a focusable sample, check illumination, and
verify safe Z travel before running.

Run the software-focus case:
EVOMACHINE_RUN_SOFTWARE_FOCUS=1 EVOMACHINE_RUN_ACQUISITION=1 EVOMACHINE_RUN_STAGE_MOVEMENT=1 uv run pytest tests/hardware/test_softwarefocus_hardware.py -m hardware -v -s
"""

import os

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig
from evomachine.types import FocusAlgorithmType, FocusStatusType
from tests.hardware.test_acquisition_hardware import (
    acquisition_metadata,
    physical_acquisition_manager as _physical_acquisition_manager,  # noqa: F401
)


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        any(
            os.getenv(variable) != "1"
            for variable in (
                "EVOMACHINE_RUN_SOFTWARE_FOCUS",
                "EVOMACHINE_RUN_ACQUISITION",
                "EVOMACHINE_RUN_STAGE_MOVEMENT",
            )
        ),
        reason=(
            "Set EVOMACHINE_RUN_SOFTWARE_FOCUS=1, EVOMACHINE_RUN_ACQUISITION=1, "
            "and EVOMACHINE_RUN_STAGE_MOVEMENT=1 after checking focus travel and illumination."
        ),
    ),
]


def test_software_focus_scans_real_z_stack_and_returns_result(
        _physical_acquisition_manager,  # noqa: F811
) -> None:
    rel_range = int(os.getenv("EVOMACHINE_SOFTWARE_FOCUS_RANGE", "20"))
    step_size = int(os.getenv("EVOMACHINE_SOFTWARE_FOCUS_STEP", "10"))
    metadata = acquisition_metadata(_physical_acquisition_manager)
    config = SoftwareFocusConfig(
        focus_frames=[metadata],
        acquisition_settings=FrameAcquisitionSettings(
            save=False,
            normalise=False,
            illuminate_dmd=True,
            clear_dmd_after=True,
            restore_leds_after=False,
            disable_leds_after=True,
        ),
        rel_range=rel_range,
        step_size=step_size,
        algorithm=FocusAlgorithmType.LAPLACIAN_VAR,
    )
    software_focus = SoftwareFocus(
        acquisition_manager=_physical_acquisition_manager,
        default_config=config,
    )

    result = software_focus.run(fov_id=0)

    assert result.z_coordinates.size >= 3
    assert result.focus_scores.shape == result.z_coordinates.shape
    assert np.isfinite(result.focus_scores).all()
    assert result.focus_stack.shape[-1] == result.z_coordinates.size
    assert isinstance(result.focus_status, FocusStatusType)
    assert not _physical_acquisition_manager.dmd.is_full_display()

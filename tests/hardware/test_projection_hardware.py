"""Physical ProjectionManager integration tests.

Check the camera/DMD optical path before running. The basic case only verifies
initialised devices. Calibration requires an extra flag and writes exclusively
to pytest's temporary directory, never to the existing calibration path.

Run the initialisation case:
EVOMACHINE_RUN_PROJECTION=1 EVOMACHINE_RUN_ACQUISITION=1 uv run pytest tests/hardware/test_projection_hardware.py -k "recognises_initialised_hardware" -m hardware -v -s

Run every case, including temporary calibration:
EVOMACHINE_RUN_PROJECTION=1 EVOMACHINE_RUN_ACQUISITION=1 EVOMACHINE_RUN_PROJECTION_CALIBRATION=1 uv run pytest tests/hardware/test_projection_hardware.py -m hardware -v -s
"""

import os

import pytest

from evomachine.peripherals.dmd import DmdCalibrationConfig
from evomachine.projection import ProjectionManager
from evomachine.types import LEDType
from tests.hardware.test_acquisition_hardware import (
    physical_acquisition_manager as _physical_acquisition_manager,  # noqa: F401
)


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        any(
            os.getenv(variable) != "1"
            for variable in (
                "EVOMACHINE_RUN_PROJECTION",
                "EVOMACHINE_RUN_ACQUISITION",
            )
        ),
        reason=(
            "Set EVOMACHINE_RUN_PROJECTION=1 and EVOMACHINE_RUN_ACQUISITION=1 "
            "after checking the projection optical path."
        ),
    ),
]


def test_projection_manager_recognises_initialised_hardware(
        _physical_acquisition_manager,  # noqa: F811
) -> None:
    manager = ProjectionManager(
        camera=_physical_acquisition_manager.camera,
        dmd=_physical_acquisition_manager.dmd,
        led_manager=_physical_acquisition_manager.led_manager,
    )
    assert manager.devices_are_initialised()


@pytest.mark.skipif(
    os.getenv("EVOMACHINE_RUN_PROJECTION_CALIBRATION") != "1",
    reason="Set EVOMACHINE_RUN_PROJECTION_CALIBRATION=1 to create temporary calibration data.",
)
def test_projection_manager_writes_only_temporary_calibration(
        _physical_acquisition_manager,  # noqa: F811
        tmp_path,
) -> None:
    manager = ProjectionManager(
        camera=_physical_acquisition_manager.camera,
        dmd=_physical_acquisition_manager.dmd,
        led_manager=_physical_acquisition_manager.led_manager,
        calibration_directory=tmp_path,
    )
    centre_row = _physical_acquisition_manager.dmd.width_height_DMD[0] // 2
    centre_col = _physical_acquisition_manager.dmd.width_height_DMD[1] // 2
    step = int(os.getenv("EVOMACHINE_PROJECTION_CALIBRATION_STEP", "100"))
    output = tmp_path / "hardware_projection_calibration.pkl"
    config = DmdCalibrationConfig(
        channel=LEDType.LED_515_NM,
        brightness=float(os.getenv("EVOMACHINE_LED_TEST_BRIGHTNESS", "5")),
        exposure=int(os.getenv("EVOMACHINE_CAMERA_EXPOSURE_MS", "30")),
        line_width=5,
        step=step,
        delay=float(os.getenv("EVOMACHINE_PROJECTION_DELAY_S", "1")),
        start_row=centre_row - step,
        end_row=centre_row,
        start_col=centre_col - step,
        end_col=centre_col,
        on_mothermachine=False,
    )

    manager.dmd_calibrate(cfg=config, filename=output)

    assert output.exists()
    assert _physical_acquisition_manager.dmd.is_calibrated()
    assert not _physical_acquisition_manager.dmd.is_full_display()

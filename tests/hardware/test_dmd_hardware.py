"""Physical DMD display tests.

Check the optical path before enabling the display. Patterns remain visible for
the configured dwell time and the fixture blanks the DMD during cleanup.

Run all DMD cases:
EVOMACHINE_RUN_DMD_DISPLAY=1 uv run pytest tests/hardware/test_dmd_hardware.py -m hardware -v -s
"""

from contextlib import suppress
import os
from pathlib import Path
import time

import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
from evomachine.peripherals.dmd import DmdConfig, DmdFactory


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_DMD_DISPLAY") != "1", reason="Set EVOMACHINE_RUN_DMD_DISPLAY=1.")]


@pytest.fixture(scope="module")
def physical_dmd():
    controller = EmDmdWindowPeripheralController(debug_mode=False)
    calibration_file = Path(os.getenv("EVOMACHINE_DMD_CALIBRATION", "evomachine/dmd_calibration_data.pkl")).resolve()
    dmd = DmdFactory.create(
        DmdConfig(
            binding=BindingType.EM_DMD_WINDOW,
            width_height_DMD=(2716, 1600),
            width_height_CAM=(3200, 3200),
            calibration_file=calibration_file,
        ),
        peripheral_controllers=controller,
    )
    try:
        dmd.initialise()
        dmd.display_none()
        yield dmd
    finally:
        with suppress(Exception):
            dmd.display_none()
        with suppress(Exception):
            dmd.finalise()


def test_dmd_initialises_and_generates_patterns(physical_dmd) -> None:
    assert physical_dmd.is_initialised()
    assert physical_dmd.is_alive()
    checkerboard = physical_dmd.get_checkerboard(square_size=100)
    crosshair = physical_dmd.get_calibration_image(lw=5)
    assert checkerboard.shape == physical_dmd.width_height_DMD
    assert crosshair.shape == physical_dmd.width_height_DMD
    assert checkerboard.dtype == np.uint8


def test_dmd_displays_and_blanks_pattern(physical_dmd) -> None:
    dwell_s = float(os.getenv("EVOMACHINE_DMD_DWELL_S", "2"))
    physical_dmd.display_image(physical_dmd.get_checkerboard(square_size=100))
    assert not physical_dmd.is_full_display()
    time.sleep(dwell_s)
    physical_dmd.display_full()
    assert physical_dmd.is_full_display()
    time.sleep(dwell_s)
    physical_dmd.display_none()
    assert not physical_dmd.is_full_display()


def test_dmd_displays_calibration_point(physical_dmd) -> None:
    dwell_s = float(os.getenv("EVOMACHINE_DMD_DWELL_S", "2"))
    physical_dmd.display_circle(row=1358, col=800, radius=5)
    time.sleep(dwell_s)
    physical_dmd.display_none()


def test_dmd_loaded_calibration_transforms_camera_coordinate(physical_dmd) -> None:
    assert physical_dmd.is_calibrated()
    row, col = physical_dmd.img_to_dmd_coords(1600, 1600)
    assert isinstance(row, int) and isinstance(col, int)
    camera_row, camera_col = physical_dmd.dmd_to_img_coords(row, col)
    assert isinstance(camera_row, int) and isinstance(camera_col, int)

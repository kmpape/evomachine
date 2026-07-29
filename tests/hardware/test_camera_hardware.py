"""Physical PVCAM camera tests.

Close camera notebooks and applications before running.

Run all camera cases:
EVOMACHINE_RUN_HARDWARE=1 uv run pytest tests/hardware/test_camera_hardware.py -m hardware -v -s
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from evomachine.peripherals.camera import Camera

from .conftest import HardwareCameraTestSettings


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("EVOMACHINE_RUN_HARDWARE") != "1",
        reason="Set EVOMACHINE_RUN_HARDWARE=1 to enable physical hardware tests.",
    ),
]


def test_pvcam_initialises_with_expected_configuration(
        hardware_camera: Camera,
        hardware_camera_settings: HardwareCameraTestSettings,
) -> None:
    """Mirror the notebook's initialisation and reported camera-state checks."""
    assert hardware_camera.is_initialised()
    assert hardware_camera.is_alive()
    assert hardware_camera.image.shape == (
        hardware_camera_settings.height,
        hardware_camera_settings.width,
    )
    assert hardware_camera.image.pxl_dtype == np.dtype("uint16")
    assert hardware_camera.get_exposure() == pytest.approx(hardware_camera_settings.exposure_ms)


def test_pvcam_captures_raw_frame(
        hardware_camera: Camera,
        hardware_camera_settings: HardwareCameraTestSettings,
) -> None:
    """Mirror the notebook's raw-frame capture and basic inspection."""
    frame = hardware_camera.get_frame(normalise=False)

    assert frame.shape == (
        hardware_camera_settings.height,
        hardware_camera_settings.width,
    )
    assert frame.dtype == np.dtype("uint16")
    assert np.isfinite(frame).all()
    assert frame.min() >= 0


def test_pvcam_changes_exposure_and_captures_again(
        hardware_camera: Camera,
        hardware_camera_settings: HardwareCameraTestSettings,
) -> None:
    """Mirror the notebook's changed-exposure capture and restore the default."""
    try:
        hardware_camera.set_exposure(hardware_camera_settings.changed_exposure_ms)
        assert hardware_camera.get_exposure() == pytest.approx(
            hardware_camera_settings.changed_exposure_ms
        )

        frame = hardware_camera.get_frame(normalise=False)

        assert frame.shape == (
            hardware_camera_settings.height,
            hardware_camera_settings.width,
        )
        assert frame.dtype == np.dtype("uint16")
        assert np.isfinite(frame).all()
    finally:
        hardware_camera.set_exposure(hardware_camera_settings.exposure_ms)


def test_pvcam_captures_normalised_frame(
        hardware_camera: Camera,
        hardware_camera_settings: HardwareCameraTestSettings,
) -> None:
    """Mirror the notebook's normalised capture and validate its numeric range."""
    frame = hardware_camera.get_frame(normalise=True)

    assert frame.shape == (
        hardware_camera_settings.height,
        hardware_camera_settings.width,
    )
    assert np.issubdtype(frame.dtype, np.floating)
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0
    assert frame.max() <= 1.0

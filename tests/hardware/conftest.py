from __future__ import annotations

from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
import os

import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.camera import Camera, CameraConfig, CameraFactory, ImageConfigType


RUN_HARDWARE_TESTS = any(
    os.environ.get(variable) == "1"
    for variable in ("EVOMACHINE_RUN_HARDWARE", "EVOMACHINE_RUN_ACQUISITION")
)


@dataclass(frozen=True, kw_only=True)
class HardwareCameraTestSettings:
    """Environment-configurable settings for the physical PVCAM smoke test."""

    width: int = 3200
    height: int = 3200
    exposure_ms: int = 30
    changed_exposure_ms: int = 100
    frame_timeout_ms: int = 5000

    @classmethod
    def from_environment(cls) -> HardwareCameraTestSettings:
        return cls(
            width=int(os.environ.get("EVOMACHINE_CAMERA_WIDTH", "3200")),
            height=int(os.environ.get("EVOMACHINE_CAMERA_HEIGHT", "3200")),
            exposure_ms=int(os.environ.get("EVOMACHINE_CAMERA_EXPOSURE_MS", "30")),
            changed_exposure_ms=int(os.environ.get("EVOMACHINE_CAMERA_CHANGED_EXPOSURE_MS", "100")),
            frame_timeout_ms=int(os.environ.get("EVOMACHINE_CAMERA_TIMEOUT_MS", "5000")),
        )


@pytest.fixture(scope="session")
def hardware_camera_settings() -> HardwareCameraTestSettings:
    """Return settings for the physical PVCAM test session."""
    return HardwareCameraTestSettings.from_environment()


@pytest.fixture(scope="session")
def hardware_camera(
        hardware_camera_settings: HardwareCameraTestSettings,
) -> Generator[Camera, None, None]:
    """Initialise one physical PVCAM camera and guarantee SDK cleanup."""
    if not RUN_HARDWARE_TESTS:
        pytest.skip("Set EVOMACHINE_RUN_HARDWARE=1 to enable physical hardware tests.")
    pytest.importorskip("pyvcam", reason="The optional pyvcam package is required for PVCAM tests.")

    camera = CameraFactory.create(
        CameraConfig(
            binding=BindingType.PVCAM,
            image=ImageConfigType(
                pxl_horiz=hardware_camera_settings.width,
                pxl_vert=hardware_camera_settings.height,
                pxl_dtype=np.dtype("uint16"),
            ),
            default_exposure_time=hardware_camera_settings.exposure_ms,
        ),
        frame_timeout_ms=hardware_camera_settings.frame_timeout_ms,
    )
    try:
        camera.initialise()
        yield camera
    finally:
        if camera.is_initialised():
            with suppress(Exception):
                camera.stop()
        with suppress(Exception):
            camera.finalise()

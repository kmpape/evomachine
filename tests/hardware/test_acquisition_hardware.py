"""Physical frame-acquisition integration tests.

Close camera notebooks, check illumination/DMD safety, and ensure the stage has
clear Z travel. The first command excludes Z movement; the second enables the
Z-stack case too.

Run acquisition cases without stage movement:
EVOMACHINE_RUN_ACQUISITION=1 uv run pytest tests/hardware/test_acquisition_hardware.py -m hardware -v -s

Run every acquisition case, including the Z stack:
EVOMACHINE_RUN_ACQUISITION=1 EVOMACHINE_RUN_STAGE_MOVEMENT=1 uv run pytest tests/hardware/test_acquisition_hardware.py -m hardware -v -s
"""

from contextlib import suppress
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
from evomachine.coordinates import Coordinate
from evomachine.frame import FrameMetaData
from evomachine.peripherals.camera import Camera
from evomachine.peripherals.dmd import DmdConfig, DmdFactory
from evomachine.peripherals.filterwheel import FilterWheelConfig, FilterWheelFactory
from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager
from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
from evomachine.peripherals.stage import StageConfig, StageFactory
from evomachine.types import FilterWheelType, LEDType


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_ACQUISITION") != "1", reason="Set EVOMACHINE_RUN_ACQUISITION=1 after checking the optical path.")]


def wait_for_tiger_idle(filter_wheel) -> None:
    """Wait for the shared Tiger controller before initialising the stage."""
    timeout_s = float(os.getenv("EVOMACHINE_FILTER_TIMEOUT_S", "10"))
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            is_busy = filter_wheel.tiger.is_busy()
        except ValueError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "ASI Tiger repeatedly returned malformed status replies "
                    f"for more than {timeout_s} seconds."
                ) from error
            time.sleep(0.05)
            continue
        if not is_busy:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"ASI Tiger remained busy for more than {timeout_s} seconds."
            )
        time.sleep(0.05)


@pytest.fixture(scope="module")
def physical_acquisition_manager(hardware_camera: Camera):
    syncboard = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(binding=BindingType.SYNCBOARD, hwid=os.getenv("EVOMACHINE_SYNCBOARD_HWID", "USB VID:PID=16C0:0483 SER=14582700 LOCATION=7-2:1.0"))
    )
    tiger = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(binding=BindingType.ASI_TIGER, hwid=os.getenv("EVOMACHINE_TIGER_HWID", "USB VID:PID=10C4:EA60 SER=0001 LOCATION=5-3")),
        card_address_filter_wheel=int(os.getenv("EVOMACHINE_FILTER_CARD_ADDRESS", "8")),
    )
    led = LEDType.LED_515_NM
    led_source = LedFactory.create(
        LedConfig(binding=BindingType.SYNCBOARD, available_leds=[led]),
        peripheral_controllers=syncboard,
    )
    led_manager = LedManager([led_source], name="Hardware Acquisition LED Manager")
    filter_wheel = FilterWheelFactory.create(
        FilterWheelConfig(
            binding=BindingType.ASI_TIGER,
            available_filters=[FilterWheelType.FILTER_592nm, FilterWheelType.FILTER],
        ),
        peripheral_controllers=tiger,
    )
    stage = StageFactory.create(
        StageConfig(binding=BindingType.ASI_TIGER, fov_step_size=100.0),
        peripheral_controllers=tiger,
    )
    dmd_controller = EmDmdWindowPeripheralController(debug_mode=False)
    dmd = DmdFactory.create(
        DmdConfig(
            binding=BindingType.EM_DMD_WINDOW,
            width_height_DMD=(2716, 1600),
            width_height_CAM=(3200, 3200),
            calibration_file=Path(
                os.getenv(
                    "EVOMACHINE_DMD_CALIBRATION",
                    "evomachine/dmd_calibration_data.pkl",
                )
            ).resolve(),
        ),
        peripheral_controllers=dmd_controller,
    )
    manager = FrameAcquisitionManager(
        camera=hardware_camera,
        led_manager=led_manager,
        filter_wheel=filter_wheel,
        dmd=dmd,
        stage=stage,
        default_settings=FrameAcquisitionSettings(
            save=False,
            normalise=False,
            illuminate_dmd=True,
            clear_dmd_after=True,
            restore_leds_after=False,
            disable_leds_after=True,
        ),
    )
    try:
        led_manager.initialise()
        led_manager.disable_led()
        filter_wheel.initialise()
        filter_wheel.set_filter_wheel(FilterWheelType.FILTER, force=True)
        wait_for_tiger_idle(filter_wheel)
        stage.initialise()
        dmd.initialise()
        dmd.display_none()
        yield manager
    finally:
        with suppress(Exception):
            manager.stop()
        for peripheral in (dmd, stage, filter_wheel, led_manager):
            with suppress(Exception):
                peripheral.finalise()
        with suppress(Exception):
            tiger.shutdown()
        with suppress(Exception):
            syncboard.shutdown()


def acquisition_metadata(manager: FrameAcquisitionManager) -> FrameMetaData:
    return FrameMetaData(
        frame_id=0,
        leds={LEDType.LED_515_NM: float(os.getenv("EVOMACHINE_LED_TEST_BRIGHTNESS", "5"))},
        filter_wheel=FilterWheelType.FILTER_592nm,
        exposure=int(os.getenv("EVOMACHINE_CAMERA_EXPOSURE_MS", "30")),
        fov_id=0,
        coordinate=manager.stage.get_coordinates(query_hardware=True),
    )


def test_hardware_acquisition_captures_one_frame(physical_acquisition_manager) -> None:
    frame = physical_acquisition_manager.take_frame(
        acquisition_metadata(physical_acquisition_manager)
    )
    assert frame.array.shape[0] == 1
    assert frame.array.dtype == np.uint16
    assert np.isfinite(frame.array).all()
    assert not physical_acquisition_manager.led_manager.get_led_state(LEDType.LED_515_NM).is_on
    assert not physical_acquisition_manager.dmd.is_full_display()


def test_hardware_acquisition_uses_dmd_pattern_and_clears_it(
        physical_acquisition_manager,
) -> None:
    pattern = physical_acquisition_manager.dmd.get_checkerboard(square_size=100)
    metadata = replace(
        acquisition_metadata(physical_acquisition_manager),
        frame_id=1,
        dmd_pattern=pattern,
    )

    frame = physical_acquisition_manager.take_frame(metadata)

    assert frame.array.shape[0] == 1
    assert frame.frame_metadata[0].dmd_pattern is pattern
    assert not physical_acquisition_manager.dmd.is_full_display()


def test_hardware_acquisition_captures_multiple_frames_in_metadata_order(
        physical_acquisition_manager,
) -> None:
    metadata = acquisition_metadata(physical_acquisition_manager)
    metadata_items = [
        replace(metadata, frame_id=10),
        replace(metadata, frame_id=11),
    ]

    frame = physical_acquisition_manager.take_frame(metadata_items)

    assert frame.array.shape[0] == len(metadata_items)
    assert [item.frame_id for item in frame.frame_metadata] == [10, 11]
    assert all(item.execution_time is not None for item in frame.frame_metadata)


def test_hardware_acquisition_changes_exposure_between_frames(
        physical_acquisition_manager,
) -> None:
    first_exposure = int(os.getenv("EVOMACHINE_CAMERA_EXPOSURE_MS", "30"))
    second_exposure = int(
        os.getenv("EVOMACHINE_CAMERA_CHANGED_EXPOSURE_MS", "100")
    )
    metadata = acquisition_metadata(physical_acquisition_manager)
    metadata_items = [
        replace(metadata, frame_id=20, exposure=first_exposure),
        replace(metadata, frame_id=21, exposure=second_exposure),
    ]

    frame = physical_acquisition_manager.take_frame(metadata_items)

    assert frame.array.shape[0] == 2
    assert [item.exposure for item in frame.frame_metadata] == [
        first_exposure,
        second_exposure,
    ]
    assert physical_acquisition_manager.camera.get_exposure() == second_exposure


def test_hardware_acquisition_finishes_filter_wheel_movement(
        physical_acquisition_manager,
) -> None:
    filter_wheel = physical_acquisition_manager.filter_wheel
    filter_wheel.set_filter_wheel(FilterWheelType.FILTER, force=True)
    wait_for_tiger_idle(filter_wheel)

    physical_acquisition_manager.take_frame(
        acquisition_metadata(physical_acquisition_manager)
    )

    assert not filter_wheel.tiger.is_busy()


def test_hardware_acquisition_cleans_up_after_rejected_filter(
        physical_acquisition_manager,
) -> None:
    metadata = replace(
        acquisition_metadata(physical_acquisition_manager),
        frame_id=30,
        filter_wheel=FilterWheelType.FILTER_465nm,
    )

    with pytest.raises(ValueError, match="unavailable filter"):
        physical_acquisition_manager.take_frame(metadata)

    assert not physical_acquisition_manager.dmd.is_full_display()
    assert not physical_acquisition_manager.led_manager.get_led_state(
        LEDType.LED_515_NM
    ).is_on


@pytest.mark.skipif(os.getenv("EVOMACHINE_RUN_STAGE_MOVEMENT") != "1", reason="Set EVOMACHINE_RUN_STAGE_MOVEMENT=1 after checking stage clearance.")
def test_hardware_acquisition_captures_z_stack_and_restores_z(physical_acquisition_manager) -> None:
    stage = physical_acquisition_manager.stage
    start = stage.get_coordinates(query_hardware=True)
    delta_z = float(os.getenv("EVOMACHINE_STAGE_TEST_DELTA_Z", "20"))
    tolerance_z = float(os.getenv("EVOMACHINE_STAGE_Z_TOLERANCE", "1"))
    targets = [Coordinate(None, None, start.z - delta_z), Coordinate(None, None, start.z), Coordinate(None, None, start.z + delta_z)]
    if any(stage.coordinate_is_out_of_bounds(target) for target in targets):
        pytest.skip("One or more Z-stack targets are outside the configured stage limits.")
    frame = physical_acquisition_manager.take_z_stack(
        frame_metadata=acquisition_metadata(physical_acquisition_manager),
        z_coordinates=targets,
    )
    assert frame.array.shape[0] == len(targets)
    assert [metadata.coordinate.z for metadata in frame.frame_metadata] == [target.z for target in targets]
    assert stage.get_coordinates(query_hardware=True).z == pytest.approx(
        start.z,
        abs=tolerance_z,
    )

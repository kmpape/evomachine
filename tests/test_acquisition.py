"""Tests for frame acquisition coordination."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.config_types import FrameMetaData
from evomachine.coordinates import Coordinate
from evomachine.peripherals.leds import LedState
from evomachine.types import LEDType


class FakeCamera:
    """Camera fake that records exposure, normalisation, and frame calls."""

    def __init__(self, fail_on_frame: bool = False):
        """
        Initialise fake camera state.

        Parameters
        ----------
        fail_on_frame
            If True, get_frame raises RuntimeError.

        Returns
        -------
        None
        """
        self.fail_on_frame = fail_on_frame
        self.exposures: list[float | int] = []
        self.normalise_calls: list[bool] = []
        self.stop_count = 0

    def set_exposure(self, exposure_time: float | int) -> None:
        """
        Record requested exposure time.

        Parameters
        ----------
        exposure_time
            Exposure time in milliseconds.

        Returns
        -------
        None
        """
        self.exposures.append(exposure_time)

    def get_frame(self, normalise: bool = False) -> np.ndarray:
        """
        Return a deterministic test frame.

        Parameters
        ----------
        normalise
            Whether normalised data was requested.

        Returns
        -------
        np.ndarray
            Test image.
        """
        self.normalise_calls.append(normalise)
        if self.fail_on_frame:
            raise RuntimeError("capture failed")
        return np.ones((3, 4), dtype=np.float64 if normalise else np.uint16)

    def stop(self) -> None:
        """
        Record that camera stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeLedManager:
    """LED manager fake that records set and disable commands."""

    def __init__(self):
        """
        Initialise fake LED state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.commands: list[tuple[LEDType | None, float | int]] = []
        self.disable_count = 0
        self.stop_count = 0
        self.states = {
            LEDType.LED_450_NM: LedState(led_type=LEDType.LED_450_NM, brightness=12, is_on=True),
        }

    def get_available_leds(self) -> list[LEDType]:
        """
        Return available fake LEDs.

        Parameters
        ----------
        None

        Returns
        -------
        list[LEDType]
            Available LED types.
        """
        return list(self.states)

    def get_led_state(self, led_type: LEDType) -> LedState:
        """
        Return a copy of the fake LED state.

        Parameters
        ----------
        led_type
            LED to inspect.

        Returns
        -------
        LedState
            Copied LED state.
        """
        state = self.states[led_type]
        return LedState(led_type=state.led_type, brightness=state.brightness, is_on=state.is_on)

    def set_led(self, led_type: LEDType, brightness: float | int, duration: float | None = None) -> None:
        """
        Record a set LED command.

        Parameters
        ----------
        led_type
            LED to set.
        brightness
            Brightness to set.
        duration
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        self.commands.append((led_type, brightness))
        self.states[led_type] = LedState(led_type=led_type, brightness=brightness, is_on=brightness > 0)

    def disable_led(self, led_type: LEDType | None = None) -> None:
        """
        Record a disable LED command.

        Parameters
        ----------
        led_type
            Optional LED to disable. If None, disable all.

        Returns
        -------
        None
        """
        self.disable_count += 1
        self.commands.append((led_type, 0))

    def stop(self) -> None:
        """
        Record that LED manager stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeDmd:
    """DMD fake that records full, image, and blank display commands."""

    def __init__(self):
        """
        Initialise fake DMD command logs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.full_count = 0
        self.none_count = 0
        self.stop_count = 0
        self.images: list[np.ndarray] = []

    def display_full(self) -> None:
        """
        Record full illumination.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.full_count += 1

    def display_image(self, img: np.ndarray) -> None:
        """
        Record image illumination.

        Parameters
        ----------
        img
            DMD image to display.

        Returns
        -------
        None
        """
        self.images.append(img.copy())

    def display_none(self) -> None:
        """
        Record blank illumination.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.none_count += 1

    def stop(self) -> None:
        """
        Record that DMD stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeStage:
    """Stage fake that records stop calls for acquisition manager tests."""

    def __init__(self):
        """
        Initialise fake stage state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.coordinate = Coordinate(0, 0, 0)
        self.stop_count = 0

    def get_coordinates(self, query_hardware: bool = True) -> Coordinate:
        """
        Return the fake stage coordinate.

        Parameters
        ----------
        query_hardware
            Accepted for API compatibility.

        Returns
        -------
        Coordinate
            Current fake stage coordinate.
        """
        return self.coordinate.copy()

    def move(self, target: Coordinate, block: bool = True) -> None:
        """
        Move the fake stage to a partial coordinate.

        Parameters
        ----------
        target
            Coordinate containing axes to update.
        block
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        if target.x is not None:
            self.coordinate.x = target.x
        if target.y is not None:
            self.coordinate.y = target.y
        if target.z is not None:
            self.coordinate.z = target.z

    def stop(self) -> None:
        """
        Record that stage stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeFileManager:
    """File manager fake that records saves and can raise."""

    def __init__(self, fail_on_save: bool = False):
        """
        Initialise fake save state.

        Parameters
        ----------
        fail_on_save
            If True, save_frame raises RuntimeError.

        Returns
        -------
        None
        """
        self.fail_on_save = fail_on_save
        self.saved: list[tuple[np.ndarray, FrameMetaData]] = []

    def save_frame(
            self,
            frame: np.ndarray,
            frame_metadata: FrameMetaData,
    ) -> Path:
        """
        Record a save request and return a fake path.

        Parameters
        ----------
        frame
            Image frame to save.
        frame_metadata
            Metadata to save with the frame.

        Returns
        -------
        Path
            Fake saved path.
        """
        if self.fail_on_save:
            raise RuntimeError("save failed")
        self.saved.append((frame.copy(), frame_metadata))
        return Path("/tmp/frame.tiff")


def _metadata(**updates) -> FrameMetaData:
    """
    Return simple frame metadata for acquisition tests.

    Parameters
    ----------
    **updates
        FrameMetaData fields to override.

    Returns
    -------
    FrameMetaData
        Valid frame metadata.
    """
    values = {
        "frame_id": 1,
        "leds": {LEDType.LED_450_NM: 22},
        "filter_wheel": None,
        "exposure": 50,
    }
    values.update(updates)
    return FrameMetaData(**values)


def _manager(
        camera: FakeCamera | None = None,
        led_manager: FakeLedManager | None = None,
        dmd: FakeDmd | None = None,
        file_manager: FakeFileManager | None = None,
        stage: FakeStage | None = None,
        default_settings: FrameAcquisitionSettings | None = None,
) -> FrameAcquisitionManager:
    """
    Return a FrameAcquisitionManager with fake dependencies.

    Parameters
    ----------
    camera
        Optional fake camera.
    led_manager
        Optional fake LED manager.
    dmd
        Optional fake DMD.
    file_manager
        Optional fake file manager.
    stage
        Optional fake stage.
    default_settings
        Optional default acquisition settings.

    Returns
    -------
    FrameAcquisitionManager
        Manager configured with fakes.
    """
    return FrameAcquisitionManager(
        camera=FakeCamera() if camera is None else camera,
        led_manager=FakeLedManager() if led_manager is None else led_manager,
        dmd=FakeDmd() if dmd is None else dmd,
        file_manager=file_manager,
        stage=stage,
        default_settings=default_settings,
    )


def test_take_frame_uses_constructor_default_settings() -> None:
    """
    Check default manager settings are used when no per-call settings are supplied.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    camera = FakeCamera()
    dmd = FakeDmd()
    manager = _manager(
        camera=camera,
        dmd=dmd,
        default_settings=FrameAcquisitionSettings(normalise=True, illuminate_dmd=False, restore_leds_after=False),
    )

    frame = manager.take_frame(_metadata())

    assert frame.array.shape == (1, 3, 4)
    assert camera.normalise_calls == [True]
    assert dmd.full_count == 0


def test_per_call_settings_replace_defaults_for_one_call() -> None:
    """
    Check per-call settings replace defaults without changing later calls.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    camera = FakeCamera()
    dmd = FakeDmd()
    manager = _manager(
        camera=camera,
        dmd=dmd,
        default_settings=FrameAcquisitionSettings(normalise=True, illuminate_dmd=False, restore_leds_after=False),
    )

    manager.take_frame(
        _metadata(frame_id=1),
        settings=FrameAcquisitionSettings(normalise=False, illuminate_dmd=True, restore_leds_after=False),
    )
    manager.take_frame(_metadata(frame_id=2))

    assert camera.normalise_calls == [False, True]
    assert dmd.full_count == 1


def test_update_settings_changes_later_defaults() -> None:
    """
    Check update_settings modifies manager defaults for subsequent calls.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    camera = FakeCamera()
    dmd = FakeDmd()
    manager = _manager(camera=camera, dmd=dmd)

    manager.update_settings(normalise=True, clear_dmd_after=True, restore_leds_after=False)
    manager.take_frame(_metadata())

    assert camera.normalise_calls == [True]
    assert dmd.full_count == 1
    assert dmd.none_count == 1

    manager.update_settings(FrameAcquisitionSettings(illuminate_dmd=False, restore_leds_after=False))
    manager.take_frame(_metadata(frame_id=2))

    assert camera.normalise_calls == [True, False]
    assert dmd.full_count == 1


def test_take_frame_uses_frame_metadata_dmd_pattern() -> None:
    """
    Check DMD pattern is taken from frame metadata.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pattern = np.arange(6, dtype=np.uint8).reshape(2, 3)
    dmd = FakeDmd()
    manager = _manager(dmd=dmd, default_settings=FrameAcquisitionSettings(restore_leds_after=False))

    manager.take_frame(_metadata(dmd_pattern=pattern))

    assert dmd.full_count == 0
    assert len(dmd.images) == 1
    assert np.array_equal(dmd.images[0], pattern)


def test_cleanup_runs_when_capture_raises() -> None:
    """
    Check cleanup runs when camera capture raises.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dmd = FakeDmd()
    leds = FakeLedManager()
    manager = _manager(
        camera=FakeCamera(fail_on_frame=True),
        led_manager=leds,
        dmd=dmd,
        default_settings=FrameAcquisitionSettings(clear_dmd_after=True, disable_leds_after=True),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        manager.take_frame(_metadata())

    assert dmd.full_count == 1
    assert dmd.none_count == 1
    assert leds.disable_count == 1


def test_cleanup_runs_when_save_raises() -> None:
    """
    Check cleanup runs when saving raises.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dmd = FakeDmd()
    leds = FakeLedManager()
    manager = _manager(
        led_manager=leds,
        dmd=dmd,
        file_manager=FakeFileManager(fail_on_save=True),
        default_settings=FrameAcquisitionSettings(save=True, clear_dmd_after=True, disable_leds_after=True),
    )

    with pytest.raises(RuntimeError, match="save failed"):
        manager.take_frame(_metadata())

    assert dmd.full_count == 1
    assert dmd.none_count == 1
    assert leds.disable_count == 1


def test_stop_calls_camera_and_stage_stop_directly() -> None:
    """
    Check acquisition stop uses the required peripheral stop API.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    camera = FakeCamera()
    leds = FakeLedManager()
    dmd = FakeDmd()
    stage = FakeStage()
    manager = _manager(camera=camera, led_manager=leds, dmd=dmd, stage=stage)

    manager.stop()

    assert leds.disable_count == 1
    assert dmd.none_count == 1
    assert camera.stop_count == 1
    assert stage.stop_count == 1

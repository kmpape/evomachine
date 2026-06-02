from pathlib import Path
import pickle

import numpy as np

from evomachine.peripherals.dmd import DmdCalibrationConfig
from evomachine.projection import ProjectionManager
from evomachine.types import FilterWheelType, LEDType


class FakeCamera:
    """Camera fake that turns the currently displayed DMD point into an image peak."""

    def __init__(self, dmd: "FakeDmd"):
        """
        Initialise fake camera state.

        Parameters
        ----------
        dmd
            Fake DMD whose current display state determines generated images.

        Returns
        -------
        None
        """
        self.dmd = dmd
        self.exposures: list[float | int] = []
        self.live_mode_disabled = False

    def is_initialised(self) -> bool:
        """
        Return fake initialisation state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def disable_live_mode(self) -> None:
        """
        Record that live mode was disabled.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.live_mode_disabled = True

    def set_exposure(self, exposure_time: float | int) -> None:
        """
        Record one exposure setting.

        Parameters
        ----------
        exposure_time
            Exposure time passed by ProjectionManager.

        Returns
        -------
        None
        """
        self.exposures.append(exposure_time)

    def get_frame(self, normalise: bool = False) -> np.ndarray:
        """
        Return an image containing a bright point at the current DMD coordinate.

        Parameters
        ----------
        normalise
            Accepted for camera API compatibility.

        Returns
        -------
        np.ndarray
            Generated image.
        """
        image = np.zeros(self.dmd.width_height_DMD, dtype=np.float64)
        if self.dmd.current_point is None:
            return image
        if self.dmd.current_point == "circles":
            image[5, 5] = 100.0
            return image
        row, col = self.dmd.current_point
        image[row, col] = 1.0 if (row, col) == (0, 0) else 100.0
        return image


class FakeDmd:
    """DMD fake that records display commands and loads saved calibration data."""

    def __init__(self):
        """
        Initialise fake DMD state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.width_height_DMD = (20, 20)
        self.current_point: tuple[int, int] | str | None = None
        self.displayed_points: list[tuple[int, int]] = []
        self.display_none_count = 0
        self.calibration_file: Path | None = None
        self.calibration_data: list | None = None

    def is_initialised(self) -> bool:
        """
        Return fake initialisation state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def display_circle(self, row: int, col: int, radius: int) -> None:
        """
        Record one displayed circle.

        Parameters
        ----------
        row
            DMD row coordinate.
        col
            DMD column coordinate.
        radius
            Circle radius accepted for DMD API compatibility.

        Returns
        -------
        None
        """
        self.current_point = (row, col)
        self.displayed_points.append((row, col))

    def display_circles(
            self,
            start_col: int,
            end_col: int,
            start_row: int,
            end_row: int,
            step_row: int,
            step_col: int,
            radius: int,
    ) -> None:
        """
        Record a multi-circle display request.

        Parameters
        ----------
        start_col
            First DMD column in the displayed grid.
        end_col
            Last DMD column in the displayed grid.
        start_row
            First DMD row in the displayed grid.
        end_row
            Last DMD row in the displayed grid.
        step_row
            Row spacing.
        step_col
            Column spacing.
        radius
            Circle radius.

        Returns
        -------
        None
        """
        self.current_point = "circles"

    def display_none(self) -> None:
        """
        Record that the DMD was blanked.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.current_point = None
        self.display_none_count += 1

    def calibrate(self, filepath: Path | None = None) -> None:
        """
        Load saved calibration data from a pickle file.

        Parameters
        ----------
        filepath
            Path to the calibration pickle file.

        Returns
        -------
        None
        """
        self.calibration_file = Path(filepath)
        with open(self.calibration_file, "rb") as file:
            self.calibration_data = pickle.load(file)

    def get_calibration_data(self) -> tuple[list, np.ndarray, np.ndarray, Path]:
        """
        Return loaded fake calibration data and identity homographies.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[list, np.ndarray, np.ndarray, Path]
            Calibration data, two identity matrices, and calibration path.
        """
        return self.calibration_data, np.eye(3), np.eye(3), self.calibration_file


class FakeLedManager:
    """LED manager fake that records set and disable requests."""

    def __init__(self):
        """
        Initialise fake LED command state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.set_calls: list[tuple[LEDType, float | int]] = []
        self.disable_count = 0

    def is_initialised(self) -> bool:
        """
        Return fake initialisation state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def set_led(self, led_type: LEDType, brightness: float | int) -> None:
        """
        Record one LED set command.

        Parameters
        ----------
        led_type
            LED type requested.
        brightness
            Requested brightness.

        Returns
        -------
        None
        """
        self.set_calls.append((led_type, brightness))

    def disable_led(self) -> None:
        """
        Record one request to disable LEDs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.disable_count += 1


class FakeFilterWheel:
    """Filter wheel fake that records requested positions."""

    def __init__(self, current_filter_type: FilterWheelType = FilterWheelType.FILTER):
        """
        Initialise fake filter wheel state.

        Parameters
        ----------
        current_filter_type
            Initial filter wheel type.

        Returns
        -------
        None
        """
        self.current_filter_type = current_filter_type
        self.set_calls: list[FilterWheelType] = []

    def is_initialised(self) -> bool:
        """
        Return fake initialisation state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def get_filter_wheel(self) -> FilterWheelType:
        """
        Return the current fake filter wheel type.

        Parameters
        ----------
        None

        Returns
        -------
        FilterWheelType
            Current filter wheel type.
        """
        return self.current_filter_type

    def set_filter_wheel(self, filter_type: FilterWheelType) -> None:
        """
        Record and apply one filter wheel set command.

        Parameters
        ----------
        filter_type
            Requested filter wheel type.

        Returns
        -------
        None
        """
        self.set_calls.append(filter_type)
        self.current_filter_type = filter_type


class FakePhotodiode:
    """Photodiode fake reserved for ProjectionManager construction tests."""


def make_config() -> DmdCalibrationConfig:
    """
    Return a small DMD calibration config for tests.

    Parameters
    ----------
    None

    Returns
    -------
    DmdCalibrationConfig
        Calibration config scanning four DMD points.
    """
    return DmdCalibrationConfig(
        channel=LEDType.LED_450_NM,
        brightness=25,
        exposure=12,
        line_width=1,
        step=5,
        delay=0,
        start_row=5,
        end_row=10,
        start_col=5,
        end_col=10,
        on_mothermachine=False,
    )


def test_projection_manager_calibrates_dmd_and_restores_peripherals(tmp_path: Path) -> None:
    """
    Check DMD calibration scans points, saves data, and restores peripheral state.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    dmd = FakeDmd()
    camera = FakeCamera(dmd=dmd)
    led_manager = FakeLedManager()
    filter_wheel = FakeFilterWheel()
    photodiode = FakePhotodiode()
    manager = ProjectionManager(
        camera=camera,
        dmd=dmd,
        led_manager=led_manager,
        filter_wheel=filter_wheel,
        photodiode=photodiode,
        sleep_func=lambda duration: None,
    )
    filename = tmp_path / "dmd_calibration.pkl"

    calibration_data, homography, homography_inv, calibration_file = manager.dmd_calibrate(
        cfg=make_config(),
        filename=filename,
    )

    assert camera.exposures == [12]
    assert camera.live_mode_disabled
    assert led_manager.set_calls == [(LEDType.LED_450_NM, 25)]
    assert led_manager.disable_count == 1
    assert filter_wheel.set_calls == [FilterWheelType.FILTER_527nm, FilterWheelType.FILTER]
    assert dmd.display_none_count >= 1
    assert manager.photodiode is photodiode
    assert calibration_file == filename
    assert len(calibration_data) == 4
    assert calibration_data[0] == ((5, 5), (5, 5), (100.0, 100.0))
    assert np.array_equal(homography, np.eye(3))
    assert np.array_equal(homography_inv, np.eye(3))
    assert filename.exists()


def test_projection_manager_stops_calibration_and_cleans_up(tmp_path: Path) -> None:
    """
    Check a stop request aborts calibration and cleanup still runs.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    dmd = FakeDmd()
    camera = FakeCamera(dmd=dmd)
    led_manager = FakeLedManager()
    filter_wheel = FakeFilterWheel()
    manager = ProjectionManager(
        camera=camera,
        dmd=dmd,
        led_manager=led_manager,
        filter_wheel=filter_wheel,
        sleep_func=lambda duration: None,
        stop_requested=lambda: True,
    )
    filename = tmp_path / "aborted.pkl"

    result = manager.dmd_calibrate(cfg=make_config(), filename=filename)

    assert result == (None, None, None, None)
    assert not filename.exists()
    assert led_manager.disable_count == 1
    assert filter_wheel.set_calls == [FilterWheelType.FILTER_527nm, FilterWheelType.FILTER]
    assert dmd.display_none_count == 1

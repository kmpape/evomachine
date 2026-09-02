from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import pickle
import time

import numpy as np

from evomachine.config import EVOMACHINE_DIR, format_timestamp_for_filename, now
from evomachine.peripherals.camera import Camera
from evomachine.peripherals.dmd import (
    Dmd,
    DmdCalibrationConfig,
)
from evomachine.peripherals.filterwheel import FilterWheel
from evomachine.peripherals.leds import LedManager
from evomachine.peripherals.photodiode import Photodiode
from evomachine.types import FilterWheelType, LEDType

logger = logging.getLogger(__name__)


class ProjectionManager:
    """Coordinate DMD projection calibration and related projection peripherals."""

    def __init__(
            self,
            camera: Camera,
            dmd: Dmd,
            led_manager: LedManager,
            filter_wheel: FilterWheel | None = None,
            photodiode: Photodiode | None = None,
            sleep_func: Callable[[float], None] = time.sleep,
            stop_requested: Callable[[], bool] | None = None,
            calibration_directory: Path | None = None,
    ):
        """
        Initialise a projection manager.

        Parameters
        ----------
        camera
            Camera used to capture DMD calibration images.
        dmd
            DMD used to display calibration points and load calibration data.
        led_manager
            LED manager used to illuminate calibration images.
        filter_wheel
            Optional filter wheel used to select and restore calibration filters.
        photodiode
            Optional photodiode reserved for later projection intensity workflows.
        sleep_func
            Function used to wait between displayed calibration patterns and
            camera capture.
        stop_requested
            Optional callable returning True when calibration should abort.
        calibration_directory
            Directory used for generated calibration filenames when dmd_calibrate()
            is called without an explicit filename.

        Returns
        -------
        None
        """
        if not callable(sleep_func):
            raise TypeError(f"ProjectionManager.__init__: sleep_func must be callable, received {type(sleep_func)}.")
        if stop_requested is not None and not callable(stop_requested):
            raise TypeError(
                f"ProjectionManager.__init__: stop_requested must be callable or None, received {type(stop_requested)}."
            )
        self.camera: Camera = camera
        self.dmd: Dmd = dmd
        self.led_manager: LedManager = led_manager
        self.filter_wheel: FilterWheel | None = filter_wheel
        self.photodiode: Photodiode | None = photodiode
        self.sleep_func: Callable[[float], None] = sleep_func
        self.stop_requested: Callable[[], bool] | None = stop_requested
        self.calibration_directory: Path = calibration_directory or EVOMACHINE_DIR / "calibration_data" / "dmd"

    def dmd_calibrate(
            self,
            cfg: DmdCalibrationConfig,
            filename: str | Path | None = None,
            progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """
        Calibrate DMD-to-camera projection by scanning DMD points and imaging them.

        Parameters
        ----------
        cfg
            DMD calibration configuration controlling LED, exposure, grid, delay,
            and point size.
        filename
            Optional path where calibration point data is saved. When omitted, a
            unique filename is generated in calibration_directory.

        Returns
        -------
        None
            Calibration data is saved to disk and loaded into the DMD. The DMD
            stores the computed homography matrices internally.
        """
        if not isinstance(cfg, DmdCalibrationConfig):
            raise TypeError(f"ProjectionManager.dmd_calibrate: cfg must be DmdCalibrationConfig, received {type(cfg)}.")
        if not self.devices_are_initialised():
            raise RuntimeError("ProjectionManager.dmd_calibrate: devices are not initialised.")
        filename = self._normalise_calibration_filename(filename=filename)
        logger.info(f"ProjectionManager.dmd_calibrate: starting with config {cfg} and filename {filename}.")
        rows, cols = self._build_calibration_grid(cfg=cfg)
        self._disable_camera_live_mode()
        last_filter_type = self.filter_wheel.get_filter_wheel() if self.filter_wheel is not None else None
        try:
            self._report_progress(progress_callback, 0.05, "Configuring calibration peripherals.")
            self._configure_calibration_peripherals(cfg=cfg)
            max_intensity = self._measure_on_screen_intensity(
                cfg=cfg,
                progress_callback=progress_callback,
            )
            if max_intensity is None:
                return
            self._report_progress(progress_callback, 0.22, "Measuring background intensity.")
            min_intensity = self._measure_minimum_required_intensity(
                cfg=cfg,
                max_intensity=max_intensity,
            )
            if min_intensity is None:
                raise RuntimeError(
                    "ProjectionManager.dmd_calibrate: could not determine a valid intensity threshold."
                )
            calib_data_raw = self._scan_calibration_grid(
                cfg=cfg,
                rows=rows,
                cols=cols,
                min_intensity=min_intensity,
                progress_callback=progress_callback,
            )
            if calib_data_raw is None:
                return
            self._report_progress(progress_callback, 0.97, "Saving calibration results.")
            self._save_calibration_results(filename=filename, results=calib_data_raw)
            self.dmd.calibrate_from_path(path=filename)

            logger.info(f"ProjectionManager.dmd_calibrate: saved calibration data under {filename}.")
            self._report_progress(progress_callback, 1.0, "DMD calibration complete.")

        finally:
            self._restore_calibration_peripherals(last_filter_type=last_filter_type)

    def devices_are_initialised(self) -> bool:
        """
        Return whether peripherals required for DMD calibration are initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when camera, DMD, LED manager, and optional filter wheel report
            initialised.
        """
        devices = [self.camera, self.dmd, self.led_manager]
        if self.filter_wheel is not None:
            devices.append(self.filter_wheel)
        return all(device.is_initialised() for device in devices)

    def _normalise_calibration_filename(self, filename: str | Path | None) -> Path:
        """
        Return a calibration filename path, generating a unique default when omitted.

        Parameters
        ----------
        filename
            Explicit filename or None.

        Returns
        -------
        Path
            Path where calibration results should be saved.
        """
        if filename is not None:
            return Path(filename)
        timestamp = now()
        datestr = format_timestamp_for_filename(value=timestamp)
        calib_version = 0
        candidate = self.calibration_directory / f"dmd_calibration_data_{datestr}_v{calib_version}.pkl"
        while candidate.exists():
            calib_version += 1
            candidate = self.calibration_directory / f"dmd_calibration_data_{datestr}_v{calib_version}.pkl"
        return candidate

    def _build_calibration_grid(self, cfg: DmdCalibrationConfig) -> tuple[np.ndarray, np.ndarray]:
        """
        Build DMD row and column calibration grids from a calibration config.

        Parameters
        ----------
        cfg
            Calibration configuration containing start, end, and step values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Row and column mesh grids used for calibration scans.
        """
        if cfg.step <= 0:
            raise ValueError(f"ProjectionManager.dmd_calibrate: cfg.step must be positive, received {cfg.step}.")
        if cfg.end_row >= self.dmd.width_height_DMD[0] or cfg.end_col >= self.dmd.width_height_DMD[1]:
            raise ValueError(f"ProjectionManager.dmd_calibrate: invalid row or column ranges for config: {cfg}")
        col_range = np.arange(cfg.start_col, cfg.end_col + cfg.step, cfg.step, dtype=np.dtype("int"))
        row_range = np.arange(cfg.start_row, cfg.end_row + cfg.step, cfg.step, dtype=np.dtype("int"))
        if col_range[-1] == self.dmd.width_height_DMD[1]:
            col_range[-1] -= 1
        if row_range[-1] == self.dmd.width_height_DMD[0]:
            row_range[-1] -= 1
        cols, rows = np.meshgrid(col_range, row_range)
        return rows, cols

    def _disable_camera_live_mode(self) -> None:
        """
        Disable camera live mode when the camera exposes such a method.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        disable_live_mode = getattr(self.camera, "disable_live_mode", None)
        if callable(disable_live_mode):
            disable_live_mode()

    def _configure_calibration_peripherals(self, cfg: DmdCalibrationConfig) -> None:
        """
        Apply camera, LED, and filter wheel settings used for DMD calibration.

        Parameters
        ----------
        cfg
            Calibration configuration containing exposure, LED, and brightness.

        Returns
        -------
        None
        """
        self.camera.set_exposure(exposure_time=cfg.exposure)
        self._set_calibration_leds(channel=cfg.channel, brightness=cfg.brightness)
        if self.filter_wheel is not None:
            self.filter_wheel.set_filter_wheel(FilterWheelType.FILTER_527nm)

    def _set_calibration_leds(
            self,
            channel: LEDType | list[LEDType],
            brightness: float | int,
    ) -> None:
        """
        Enable one or more calibration LEDs.

        Parameters
        ----------
        channel
            Single LEDType or list of LEDType values to enable.
        brightness
            Brightness value passed to the LED manager.

        Returns
        -------
        None
        """
        if isinstance(channel, list):
            self.led_manager.disable_led()
            for led_type in channel:
                self.led_manager.set_led(led_type=led_type, brightness=brightness)
            return
        self.led_manager.set_led(led_type=channel, brightness=brightness)

    def _measure_on_screen_intensity(
            self,
            cfg: DmdCalibrationConfig,
            progress_callback: Callable[[float, str], None] | None = None,
    ) -> float | None:
        """
        Estimate the on-screen calibration intensity.

        Parameters
        ----------
        cfg
            Calibration configuration controlling point placement and delay.

        Returns
        -------
        float | None
            Mean on-screen intensity, or None when calibration is aborted.
        """
        if cfg.on_mothermachine:
            self.dmd.display_circles(
                start_col=cfg.start_col,
                end_col=cfg.end_col,
                start_row=cfg.start_row,
                end_row=cfg.end_row,
                step_row=cfg.step,
                step_col=cfg.step,
                radius=cfg.line_width,
            )
            return float(self.camera.get_frame(normalise=False).max())
        max_intensity = 0.0
        for i_row in range(3):
            for i_col in range(3):
                if self._should_stop():
                    logger.warning("ProjectionManager.dmd_calibrate: stop requested during initial intensity scan.")
                    return None
                row = (self.dmd.width_height_DMD[0] * (i_row + 1)) // 4
                col = (self.dmd.width_height_DMD[1] * (i_col + 1)) // 4
                self.dmd.display_circle(row=row, col=col, radius=cfg.line_width)
                self.sleep_func(float(cfg.delay))
                test_img = self.camera.get_frame(normalise=False)
                max_intensity += float(test_img.max())
                point_index = i_row * 3 + i_col + 1
                self._report_progress(
                    progress_callback,
                    0.05 + 0.15 * point_index / 9,
                    f"Measuring calibration intensity ({point_index}/9).",
                )
                logger.debug(f"ProjectionManager.dmd_calibrate: init image ({row}, {col}): {test_img.max()}")
        return max_intensity / 9.0

    def _measure_minimum_required_intensity(
            self,
            cfg: DmdCalibrationConfig,
            max_intensity: float,
    ) -> float | None:
        """
        Measure off-screen intensity and derive the minimum accepted point intensity.

        Parameters
        ----------
        cfg
            Calibration configuration controlling point size and delay.
        max_intensity
            Estimated on-screen intensity.

        Returns
        -------
        float | None
            Minimum intensity required for a point to be accepted, or None when
            the off-screen intensity is too high.
        """
        self.dmd.display_circle(row=0, col=0, radius=cfg.line_width)
        self.sleep_func(float(cfg.delay))
        test_img_none = self.camera.get_frame(normalise=False)
        max_intensity_none = float(test_img_none.max())
        if max_intensity_none >= 0.9 * max_intensity:
            logger.error(
                "ProjectionManager.dmd_calibrate: max off-screen intensity is high. "
                f"off_screen={max_intensity_none} > 0.9*on_screen={0.9 * max_intensity}. "
                "Please verify. Aborting calibration."
            )
            return None
        if cfg.on_mothermachine:
            min_intensity = max_intensity_none + 0.03 * (max_intensity - max_intensity_none)
        else:
            min_intensity = max_intensity_none + 0.5 * (max_intensity - max_intensity_none)
        logger.info(
            f"ProjectionManager.dmd_calibrate: max on-screen intensity={max_intensity}, "
            f"max off-screen intensity={max_intensity_none} => min required intensity={min_intensity}."
        )
        return min_intensity

    def _scan_calibration_grid(
            self,
            cfg: DmdCalibrationConfig,
            rows: np.ndarray,
            cols: np.ndarray,
            min_intensity: float,
            progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]] | None:
        """
        Scan DMD calibration points and return accepted DMD-to-camera mappings.

        Parameters
        ----------
        cfg
            Calibration configuration controlling point size and delay.
        rows
            Row mesh grid of DMD point coordinates.
        cols
            Column mesh grid of DMD point coordinates.
        min_intensity
            Minimum image intensity needed to accept a point.

        Returns
        -------
        list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]] | None
            Accepted calibration point mappings, or None when calibration aborts.
        """
        results: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]] = []
        flat_cols = cols.flatten()
        flat_rows = rows.flatten()
        total = len(flat_cols)
        for index, (col, row) in enumerate(zip(flat_cols, flat_rows)):
            if index % 50 == 0:
                logger.info(f"ProjectionManager.dmd_calibrate: at {index + 1} of {len(cols.flatten())}.")
            if self._should_stop():
                logger.warning("ProjectionManager.dmd_calibrate: stop requested during calibration grid scan.")
                return None
            self.dmd.display_none()
            self.dmd.display_circle(row=int(row), col=int(col), radius=cfg.line_width)
            self.sleep_func(float(cfg.delay))
            img = self.camera.get_frame(normalise=False)
            img_max = float(img.max())
            img_col_max = img.max(axis=0)
            img_row_max = img.max(axis=1)
            if img_max >= min_intensity:
                results.append((
                    (int(row), int(col)),
                    (int(img_row_max.argmax()), int(img_col_max.argmax())),
                    (float(img_row_max.max()), float(img_col_max.max())),
                ))
            else:
                logger.info(
                    f"ProjectionManager.dmd_calibrate: DMD point (r{row},c{col}) off screen with intensity "
                    f"{img_max} < {min_intensity}."
                )
            self._report_progress(
                progress_callback,
                0.25 + 0.70 * (index + 1) / total,
                f"Scanning DMD calibration grid ({index + 1}/{total}).",
            )
        return results

    @staticmethod
    def _report_progress(
            callback: Callable[[float, str], None] | None,
            progress: float,
            message: str,
    ) -> None:
        if callback is not None:
            callback(progress, message)

    def _save_calibration_results(
            self,
            filename: Path,
            results: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]],
    ) -> None:
        """
        Save calibration point mappings to a pickle file.

        Parameters
        ----------
        filename
            Path where calibration results should be written.
        results
            Calibration point mappings to serialise.

        Returns
        -------
        None
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as file:
            pickle.dump(results, file)

    def _restore_calibration_peripherals(self, last_filter_type: FilterWheelType | None) -> None:
        """
        Restore filter wheel state, disable LEDs, and blank the DMD after calibration.

        Parameters
        ----------
        last_filter_type
            Filter wheel position captured before calibration, or None when no
            filter wheel is configured.

        Returns
        -------
        None
        """
        if self.filter_wheel is not None and last_filter_type is not None:
            self.filter_wheel.set_filter_wheel(last_filter_type)
        self.led_manager.disable_led()
        self.dmd.display_none()

    def _should_stop(self) -> bool:
        """
        Return whether the calibration should stop.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when stop_requested exists and returns True.
        """
        return self.stop_requested is not None and self.stop_requested()

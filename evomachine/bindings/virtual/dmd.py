from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evomachine.peripherals.dmd import Dmd
from evomachine.peripherals import PeripheralController


class VirtualDmdControl:
    """In-memory DMD control used for dry runs and tests."""

    def __init__(
            self,
            width_height_DMD: tuple[int, int] = (2716, 1600),
            width_height_CAM: tuple[int, int] = (3200, 3200),
    ):
        self.width_height_DMD: tuple[int, int] = width_height_DMD
        self.width_height_CAM: tuple[int, int] = width_height_CAM
        self._is_initialised: bool = False
        self._is_full_display: bool = False
        self._image: np.ndarray | None = None
        self._loaded_img: np.ndarray | None = None
        self._calib_file: Path | None = None

    def initialise(self, is_test: bool = False) -> None:
        """Mark the virtual DMD as initialised."""
        self._is_initialised = True

    def finalise(self) -> None:
        """Mark the virtual DMD as finalised."""
        self._is_initialised = False

    def is_initialised(self) -> bool:
        """Return whether the virtual DMD is initialised."""
        return self._is_initialised

    def is_calibrated(self) -> bool:
        """Return whether calibration data is available."""
        return False

    def is_full_display(self) -> bool:
        """Return whether the stored image is full white."""
        return self._is_full_display

    def display_none(self) -> None:
        """Store a black image."""
        self._image = self.get_zero_array()
        self._is_full_display = False

    def display_full(self, force_display: bool = False) -> None:
        """Store a white image."""
        self._image = self.get_one_array()
        self._is_full_display = True

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """Store a caller-provided image."""
        self._image = img
        self._is_full_display = _is_full_display

    def get_zero_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a black image."""
        return np.zeros(img_size or self.width_height_DMD, dtype=np.uint8)

    def get_one_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a white image."""
        return np.ones(img_size or self.width_height_DMD, dtype=np.uint8) * 255

    def get_calibration_data(self) -> tuple[None, None, None, Path | None]:
        """Return empty calibration data."""
        return None, None, None, self._calib_file

    def get_calibration_filename(self) -> Path | None:
        """Return the virtual calibration filename."""
        return self._calib_file

    def calibrate(self, filepath: Path | None = None) -> None:
        """Store the virtual calibration filename."""
        self._calib_file = filepath

    def load_image(self, filename: str, display_image: bool = True) -> None:
        """Record that an image was loaded."""
        self._loaded_img = self.get_zero_array()
        if display_image:
            self.display_loaded_image()

    def display_loaded_image(self) -> None:
        """Display the most recently loaded virtual image."""
        if self._loaded_img is None:
            raise RuntimeError("VirtualDmdControl: no loaded image is available.")
        self.display_image(self._loaded_img)

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"VirtualDmdControl: {name} is not implemented.")


class VirtualDmdPeripheralController(PeripheralController):
    """Peripheral controller for VirtualDmdControl."""

    DEFAULT_NAME: str = "Virtual DMD Peripheral Controller"

    def __init__(self, dmd_control: VirtualDmdControl | None = None, name: str = DEFAULT_NAME):
        self.dmd_control: VirtualDmdControl = dmd_control or VirtualDmdControl()
        super().__init__(name=name)

    def get_dmd_control(self) -> VirtualDmdControl:
        """Return the wrapped virtual DMD control."""
        return self.dmd_control

    def _initialise(self, force: bool = False) -> bool:
        self.dmd_control.initialise()
        return self.dmd_control.is_initialised()

    def _check_is_alive(self) -> bool:
        return self.dmd_control.is_initialised()

    def _stop(self) -> None:
        if self.dmd_control.is_initialised():
            self.dmd_control.display_none()

    def _shutdown(self, force: bool = False) -> None:
        if self.dmd_control.is_initialised():
            self.dmd_control.finalise()


class VirtualDmd(Dmd):
    """DMD wrapper for the virtual binding."""

    DEFAULT_NAME: str = "Virtual DMD"

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """Store a DMD image in memory."""
        self._check_ready()
        img = self._normalise_display_image(img=img)
        self._loaded_img = img
        self._is_full_display = _is_full_display
        self.peripheral_ctrl.dmd_control.display_image(img=img, _is_full_display=_is_full_display)

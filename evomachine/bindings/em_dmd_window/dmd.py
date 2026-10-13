from __future__ import annotations

import numpy as np

from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
from evomachine.peripherals.dmd import Dmd


class EmDmdWindowDmd(Dmd):
    """DMD wrapper for the socket-backed em_dmd_window binding."""

    DEFAULT_NAME: str = "em_dmd_window DMD"

    @property
    def peripheral_ctrl(self) -> EmDmdWindowPeripheralController:
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: EmDmdWindowPeripheralController) -> None:
        self._peripheral_ctrl = peripheral_ctrl

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """Validate and send a DMD image through the socket backend."""
        self._check_ready()
        img = self._normalise_display_image(img=img)
        self.peripheral_ctrl.send_image(img=img)
        self._is_full_display = _is_full_display
        self._record_display_state(img, is_full_display=_is_full_display)

    def initialise(self, force: bool = False) -> None:
        """Initialise and mirror the DMD window's state without changing it."""
        self.peripheral_ctrl.initialise(force=force)
        self.synchronise_display_state()

    def synchronise_display_state(self) -> None:
        """Record the image currently tracked by the DMD window controller."""
        image = self.peripheral_ctrl.get_display_image()
        if image is None:
            return
        is_full_display = bool(np.all(image == 255))
        self._is_full_display = is_full_display
        latest = self.last_display_state
        if latest is not None and np.array_equal(latest.image, image):
            return
        self._record_display_state(image, is_full_display=is_full_display)

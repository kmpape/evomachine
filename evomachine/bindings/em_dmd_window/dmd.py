from __future__ import annotations

import numpy as np

from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
from evomachine.dmd import Dmd


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

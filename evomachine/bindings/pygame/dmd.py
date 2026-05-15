from __future__ import annotations

import numpy as np

from evomachine.bindings.pygame.peripheralcontroller import PygameDmdPeripheralController
from evomachine.dmd import Dmd


class PygameDmd(Dmd):
    """DMD wrapper for the pygame binding."""

    DEFAULT_NAME: str = "pygame DMD"

    @property
    def peripheral_ctrl(self) -> PygameDmdPeripheralController:
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: PygameDmdPeripheralController) -> None:
        self._peripheral_ctrl = peripheral_ctrl

    def display_image(self, img: np.ndarray, _is_full_display: bool = False, update_display: bool = True) -> None:
        """Validate and display a DMD image through the pygame backend."""
        self._check_ready()
        img = self._normalise_display_image(img=img)
        self.peripheral_ctrl.display_array(img=img, update_display=update_display)
        self._is_full_display = _is_full_display

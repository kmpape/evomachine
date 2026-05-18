from __future__ import annotations

import numpy as np

from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.camera import Camera
from evomachine.config_types import ImageConfigType


class VirtualCamera(Camera):
    """
    In-memory camera implementation for debugging without camera hardware.

    VirtualCamera uses the same public Camera API as hardware-backed cameras,
    but all frames are generated locally.
    """

    DEFAULT_NAME = "Virtual Camera"

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            image: ImageConfigType,
            name: str = DEFAULT_NAME,
            default_exposure_time: float | int = 200,
            imaging_mode: str | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
            random_seed: int | None = 0,
    ):
        """
        Initialise an in-memory virtual camera.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated lifecycle.
        image
            Image shape and dtype returned by generated frames.
        name
            Human-readable camera name.
        default_exposure_time
            Exposure time applied during initialise(), in milliseconds.
        imaging_mode
            Optional simulated imaging mode name.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the camera to report alive.
        random_seed
            Seed used for deterministic pseudo-random frame generation. If
            None, NumPy chooses a non-deterministic seed.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualCamera.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        self.peripheral_ctrl: VirtualPeripheralController = peripheral_ctrl
        self.random_seed: int | None = random_seed
        self._rng: np.random.Generator = np.random.default_rng(random_seed)
        self._stop_was_called: bool = False
        super().__init__(
            image=image,
            name=name,
            default_exposure_time=default_exposure_time,
            imaging_mode=imaging_mode,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def stop_was_called(self) -> bool:
        """
        Return whether stop has been called.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True after stop has been called at least once.
        """
        return self._stop_was_called

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual camera as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualCamera
            behavior.

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual camera.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualCamera
            behavior.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Report whether the virtual controller is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _stop(self) -> None:
        """
        Mark that stop was called.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.peripheral_ctrl.stop()
        self._stop_was_called = True

    def _get_frame(self) -> np.ndarray:
        """
        Generate one pseudo-random image frame.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Pseudo-random frame with the configured shape and dtype.
        """
        if np.issubdtype(self.image.pxl_dtype, np.integer):
            dtype_info = np.iinfo(self.image.pxl_dtype)
            high = min(dtype_info.max, 65535) + 1
            return self._rng.integers(0, high, size=self.image.shape, dtype=self.image.pxl_dtype)
        return self._rng.random(size=self.image.shape).astype(self.image.pxl_dtype)

    def _set_exposure(self, exposure_time: float | int) -> None:
        """
        Accept a simulated exposure time.

        Parameters
        ----------
        exposure_time
            Positive exposure time in milliseconds.

        Returns
        -------
        None
        """
        return

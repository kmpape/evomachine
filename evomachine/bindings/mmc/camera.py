from __future__ import annotations

from typing import Any

import numpy as np

from evomachine.peripherals.camera import Camera, CameraReadoutMode
from evomachine.peripherals.camera import ImageConfigType


class MMCCamera(Camera):
    """Camera-only Micro-Manager binding backed by pycromanager."""

    DEFAULT_NAME = "Micro-Manager Camera"

    def __init__(
            self,
            image: ImageConfigType,
            name: str = DEFAULT_NAME,
            default_exposure_time: float | int = 200,
            readout_mode: CameraReadoutMode | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
            core: Any | None = None,
            studio: Any | None = None,
            camera_device: str = "Camera-1",
            readout_mode_property: str = "Port",
    ):
        """
        Initialise a Micro-Manager camera wrapper.

        Parameters
        ----------
        image
            Image shape and dtype expected from Micro-Manager.
        name
            Human-readable camera name.
        default_exposure_time
            Exposure time applied during initialise(), in milliseconds.
        readout_mode
            Optional Micro-Manager readout mode applied during initialise().
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the camera to report alive.
        core
            Optional injected pycromanager Core-compatible object.
        studio
            Optional injected pycromanager Studio-compatible object.
        camera_device
            Micro-Manager device name used when setting imaging mode.
        readout_mode_property
            Micro-Manager property name used when setting readout mode.

        Returns
        -------
        None
        """
        self.core: Any | None = core
        self.studio: Any | None = studio
        self.camera_device: str = camera_device
        self.readout_mode_property: str = readout_mode_property
        super().__init__(
            image=image,
            name=name,
            default_exposure_time=default_exposure_time,
            readout_mode=readout_mode,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def _initialise(self, force: bool = False) -> bool:
        """
        Initialise pycromanager Core and Studio objects.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change MMCCamera
            behavior.

        Returns
        -------
        bool
            True when a Core object is available.
        """
        if self.core is None:
            from pycromanager import Core

            self.core = Core()
        if self.studio is None:
            try:
                from pycromanager import Studio

                self.studio = Studio()
            except Exception:
                self.studio = None
        return self.core is not None

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the Micro-Manager camera wrapper.

        Parameters
        ----------
        force
            Present for API compatibility. Micro-Manager shutdown is owned by
            Micro-Manager itself in this binding.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return whether a Core object is available.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when Core is not None.
        """
        return self.core is not None

    def _stop(self) -> None:
        """
        Disable live mode when possible.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._disable_live_mode()

    def _get_frame(self) -> np.ndarray:
        """
        Capture one frame through Micro-Manager.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Captured image reshaped from the tagged image payload.
        """
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        return np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]],
        )

    def _set_exposure(self, exposure_time: float | int) -> None:
        """
        Set Micro-Manager exposure time.

        Parameters
        ----------
        exposure_time
            Positive exposure time in milliseconds.

        Returns
        -------
        None
        """
        self.core.set_exposure(exposure_time)

    def _set_readout_mode(self, readout_mode: CameraReadoutMode) -> None:
        """
        Set a Micro-Manager camera readout mode property.

        Parameters
        ----------
        readout_mode
            Readout mode value passed to Micro-Manager.

        Returns
        -------
        None
        """
        self.core.set_property(self.camera_device, self.readout_mode_property, readout_mode.value)

    def _disable_live_mode(self) -> None:
        """
        Disable Micro-Manager live mode when Studio is available.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.studio is not None:
            self.studio.live().set_live_mode(False)

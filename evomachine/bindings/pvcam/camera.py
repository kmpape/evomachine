from __future__ import annotations

from typing import Any

import numpy as np

from evomachine.peripherals.camera import Camera, CameraReadoutMode
from evomachine.peripherals.camera import ImageConfigType


class PVCAMCamera(Camera):
    """Camera-only PVCAM binding backed by pyvcam."""

    DEFAULT_NAME = "PVCAM Camera"
    DEFAULT_READOUT_MODES = [
        CameraReadoutMode.SENSITIVITY,
        CameraReadoutMode.SPEED,
        CameraReadoutMode.DYNAMIC_RANGE,
        CameraReadoutMode.SUB_ELECTRON,
    ]

    def __init__(
            self,
            image: ImageConfigType,
            name: str = DEFAULT_NAME,
            default_exposure_time: float | int = 200,
            readout_mode: CameraReadoutMode | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
            pvc_module: Any | None = None,
            camera_class: Any | None = None,
            camera: Any | None = None,
            frame_timeout_ms: int = 1000,
            exposure_mode: str = "Internal Trigger",
            readout_modes: list[CameraReadoutMode] | None = None,
    ):
        """
        Initialise a PVCAM camera wrapper.

        Parameters
        ----------
        image
            Image shape and dtype expected from PVCAM.
        name
            Human-readable camera name.
        default_exposure_time
            Exposure time applied during initialise(), in milliseconds.
        readout_mode
            Optional PVCAM readout mode applied during initialise().
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the camera to report alive.
        pvc_module
            Optional injected pyvcam.pvc-compatible module.
        camera_class
            Optional injected pyvcam Camera-compatible class.
        camera
            Optional already-created camera object.
        frame_timeout_ms
            Timeout passed to camera.get_frame().
        exposure_mode
            PVCAM exposure mode assigned during initialise().
        readout_modes
            Ordered readout modes mapped to PVCAM readout_port indices.

        Returns
        -------
        None
        """
        if not isinstance(frame_timeout_ms, int) or isinstance(frame_timeout_ms, bool):
            raise TypeError(f"PVCAMCamera.__init__: frame_timeout_ms must be int, received {type(frame_timeout_ms)}.")
        if frame_timeout_ms <= 0:
            raise ValueError(f"PVCAMCamera.__init__: frame_timeout_ms must be positive, received {frame_timeout_ms}.")
        self.pvc_module: Any | None = pvc_module
        self.camera_class: Any | None = camera_class
        self.camera: Any | None = camera
        self.frame_timeout_ms: int = frame_timeout_ms
        self.exposure_mode: str = exposure_mode
        self.readout_modes: list[CameraReadoutMode] = (
            list(readout_modes) if readout_modes else list(self.DEFAULT_READOUT_MODES)
        )
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
        Initialise pyvcam and open a camera.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change PVCAMCamera
            behavior.

        Returns
        -------
        bool
            True when a camera object is available.
        """
        if self.pvc_module is None:
            from pyvcam import pvc

            self.pvc_module = pvc
        if self.camera_class is None:
            from pyvcam.camera import Camera as PyvcamCamera

            self.camera_class = PyvcamCamera
        self.pvc_module.init_pvcam()
        if self.camera is None:
            self.camera = next(self.camera_class.detect_camera())
        self.camera.open()
        self.camera.exp_mode = self.exposure_mode
        return self.camera is not None

    def _finalise(self, force: bool = False) -> None:
        """
        Close the PVCAM camera and uninitialise pyvcam.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change PVCAMCamera
            behavior.

        Returns
        -------
        None
        """
        if self.camera is not None:
            self.camera.close()
        if self.pvc_module is not None:
            self.pvc_module.uninit_pvcam()

    def _check_is_alive(self) -> bool:
        """
        Return whether a PVCAM camera object is available.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when camera is not None.
        """
        return self.camera is not None

    def _stop(self) -> None:
        """
        Stop PVCAM activity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def _get_frame(self) -> np.ndarray:
        """
        Capture one frame through PVCAM.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Captured image frame.
        """
        return self.camera.get_frame(timeout_ms=self.frame_timeout_ms)

    def _set_exposure(self, exposure_time: float | int) -> None:
        """
        Set PVCAM exposure time.

        Parameters
        ----------
        exposure_time
            Positive exposure time in milliseconds.

        Returns
        -------
        None
        """
        self.camera.exp_time = exposure_time

    def _set_readout_mode(self, readout_mode: CameraReadoutMode) -> None:
        """
        Set PVCAM readout port by readout mode.

        Parameters
        ----------
        readout_mode
            Readout mode present in readout_modes.

        Returns
        -------
        None
        """
        if readout_mode not in self.readout_modes:
            raise ValueError(f"PVCAMCamera._set_readout_mode: {readout_mode} not in {self.readout_modes}.")
        self.camera.readout_port = self.readout_modes.index(readout_mode)

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from evomachine.bindings.binding_types import BindingType
from evomachine.config import get_logger
from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig

logger = get_logger(name=__name__, is_peripheral=True)


@dataclass
class ImageConfigType:
    pxl_horiz: int
    pxl_vert: int
    pxl_dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self.pxl_vert, self.pxl_horiz

    def __post_init__(self) -> None:
        if not isinstance(self.pxl_horiz, int) or not self.pxl_horiz > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_horiz: {self.pxl_horiz}")
        if not isinstance(self.pxl_vert, int) or not self.pxl_vert > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_vert: {self.pxl_vert}")
        if not isinstance(self.pxl_dtype, np.dtype):
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_dtype: {self.pxl_dtype}")

    def copy(self) -> "ImageConfigType":
        return ImageConfigType(**self.__dict__)

    def __str__(self) -> str:
        lines = ["ImageConfigType"]
        for index, (key, value) in enumerate(self.__dict__.items()):
            lines.append(f"{' └─ ' if index == len(self.__dict__) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)


class ImageConfigTypeFactory:
    @staticmethod
    def pv_cam() -> ImageConfigType:
        return ImageConfigType(pxl_horiz=3200, pxl_vert=3200, pxl_dtype=np.dtype("uint16"))

    @staticmethod
    def delta() -> ImageConfigType:
        return ImageConfigType(pxl_horiz=696, pxl_vert=520, pxl_dtype=np.dtype("float32"))


@dataclass
class ObjectiveConfigType:
    na: float
    mag: int
    descr: str | None = "UNKNOWN OBJECTIVE"

    def __post_init__(self) -> None:
        if not isinstance(self.na, float) or not 0 < self.na:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid numerical_aperture: {self.na}")
        if not isinstance(self.mag, int) or not self.mag > 0:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid magnification: {self.mag}")

    def copy(self) -> "ObjectiveConfigType":
        return ObjectiveConfigType(**self.__dict__)

    def __str__(self) -> str:
        lines = ["ObjectiveConfigType"]
        for index, (key, value) in enumerate(self.__dict__.items()):
            lines.append(f"{' └─ ' if index == len(self.__dict__) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)


class ObjectiveConfigTypeFactory:
    @staticmethod
    def default_oil() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=1.4, mag=60, descr="Nikon Plan Apo lambda 60x/1.4 Oil")

    @staticmethod
    def default_air() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=0.95, mag=40, descr="Nikon Plan Fluor 40x/0.95")


def calculate_fov_size(camera_config: "CameraConfig", objective_config: ObjectiveConfigType) -> float:
    """
    Return the field-of-view size for a camera/objective pair.

    Parameters
    ----------
    camera_config
        Camera configuration containing image dimensions and sensor pixel size.
    objective_config
        Objective configuration containing magnification.

    Returns
    -------
    float
        Vertical field-of-view size in micrometres.
    """
    if not isinstance(camera_config, CameraConfig):
        raise TypeError(f"calculate_fov_size: camera_config must be CameraConfig, received {type(camera_config)}.")
    if not isinstance(objective_config, ObjectiveConfigType):
        raise TypeError(
            f"calculate_fov_size: objective_config must be ObjectiveConfigType, received {type(objective_config)}."
        )
    return camera_config.sensor_pixel_size_um / objective_config.mag * camera_config.image.pxl_vert


class CameraReadoutMode(str, Enum):
    """Supported camera readout modes for Kinetix/PVCAM-style cameras."""

    DYNAMIC_RANGE = "Dynamic Range"
    SENSITIVITY = "Sensitivity"
    SPEED = "Speed"
    SUB_ELECTRON = "Sub-Electron"


@dataclass(kw_only=True)
class CameraConfig(PeripheralConfig):
    """Configuration object used by CameraFactory to create camera devices."""

    image: ImageConfigType
    default_exposure_time: float | int = 200
    sensor_pixel_size_um: float = 6.5
    readout_mode: CameraReadoutMode | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.image, ImageConfigType):
            raise TypeError(f"CameraConfig: image must be ImageConfigType, received {type(self.image)}.")
        if not isinstance(self.default_exposure_time, int | float) or isinstance(self.default_exposure_time, bool):
            raise TypeError(
                f"CameraConfig: default_exposure_time must be numeric, received {type(self.default_exposure_time)}."
            )
        if self.default_exposure_time <= 0:
            raise ValueError(
                f"CameraConfig: default_exposure_time must be positive, received {self.default_exposure_time}."
            )
        if not isinstance(self.sensor_pixel_size_um, int | float) or isinstance(self.sensor_pixel_size_um, bool):
            raise TypeError(
                f"CameraConfig: sensor_pixel_size_um must be numeric, received {type(self.sensor_pixel_size_um)}."
            )
        if self.sensor_pixel_size_um <= 0:
            raise ValueError(
                f"CameraConfig: sensor_pixel_size_um must be positive, received {self.sensor_pixel_size_um}."
            )
        self.sensor_pixel_size_um = float(self.sensor_pixel_size_um)
        if self.readout_mode is not None and not isinstance(self.readout_mode, CameraReadoutMode):
            raise TypeError(
                f"CameraConfig: readout_mode must be CameraReadoutMode or None, "
                f"received {type(self.readout_mode)}."
            )


class Camera(Peripheral):
    """
    Base class for camera-only image sensors.

    This class owns lifecycle state, readiness checks, exposure bookkeeping, and
    frame normalisation. Subclasses implement only binding-specific camera
    operations.
    """

    def __init__(
            self,
            image: ImageConfigType,
            name: str,
            default_exposure_time: float | int = 200,
            readout_mode: CameraReadoutMode | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared camera state.

        Parameters
        ----------
        image
            Image shape and dtype expected from this camera.
        name
            Human-readable camera name.
        default_exposure_time
            Exposure time applied during initialise(), in milliseconds.
        readout_mode
            Optional camera readout mode applied during initialise().
        check_initialised
            If True, public camera methods raise RuntimeError before
            initialise() succeeds.
        check_alive
            If True, public camera methods raise RuntimeError when the camera
            does not report alive.

        Returns
        -------
        None
        """
        if not isinstance(image, ImageConfigType):
            raise TypeError(f"Camera.__init__: image must be ImageConfigType, received {type(image)}.")
        self.image: ImageConfigType = image
        self.name: str = name
        self.default_exposure_time: float | int = self._validate_exposure_time(default_exposure_time)
        self.readout_mode: CameraReadoutMode | None = self._validate_readout_mode(readout_mode=readout_mode)
        self._current_exposure: float | int | None = None
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self._check_initialised: bool = check_initialised
        self._check_alive: bool = check_alive
        self.config: CameraConfig | None = None
        # TODO(Codex): Add one status flag here to track whether the camera is live streaming or not.

    @staticmethod
    def _validate_exposure_time(exposure_time: float | int) -> float | int:
        # TODO(Codex): Many classes and enum define their own argument validation. Can we add one general function to utils, that e.g. takes one variable and the types it should be, and then checks the type?
        # TODO(Codex): Similarly, can we have a more general utils function that checks ranges (depending on types), and use this everywhere in the code?
        """
        Return a validated positive exposure time.

        Parameters
        ----------
        exposure_time
            Candidate exposure time in milliseconds.

        Returns
        -------
        float | int
            Validated exposure time.
        """
        if not isinstance(exposure_time, int | float) or isinstance(exposure_time, bool):
            raise TypeError(f"Camera._validate_exposure_time: exposure_time must be numeric, received {type(exposure_time)}.")
        if exposure_time <= 0:
            raise ValueError(f"Camera._validate_exposure_time: exposure_time must be positive, received {exposure_time}.")
        return exposure_time

    @staticmethod
    def _validate_readout_mode(readout_mode: CameraReadoutMode | None) -> CameraReadoutMode | None:
        """
        Return a validated camera readout mode.

        Parameters
        ----------
        readout_mode
            Candidate camera readout mode.

        Returns
        -------
        CameraReadoutMode | None
            Validated camera readout mode.
        """
        if readout_mode is not None and not isinstance(readout_mode, CameraReadoutMode):
            raise TypeError(
                f"Camera._validate_readout_mode: readout_mode must be CameraReadoutMode or None, "
                f"received {type(readout_mode)}."
            )
        return readout_mode

    def _require_ready(self, action: str) -> None:
        """
        Raise when a camera action is not allowed by current readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self._check_initialised and not self._is_initialised:
            logger.warning("Camera.%s: %s is not initialised.", action, self.name)
            raise RuntimeError(f"Camera.{action}: camera is not initialised.")
        if self._check_alive and not self.is_alive():
            logger.warning("Camera.%s: %s is not alive.", action, self.name)
            raise RuntimeError(f"Camera.{action}: camera is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the camera and apply default exposure and readout mode.

        Parameters
        ----------
        force
            If True, run initialisation even when already initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            logger.debug("Camera.initialise: %s already initialised; skipping.", self.name)
            return
        logger.debug("Camera.initialise: initialising %s with force=%s.", self.name, force)
        self._is_initialised = self._initialise(force=force)
        if self._check_initialised and not self._is_initialised:
            logger.warning("Camera.initialise: %s failed to initialise.", self.name)
            raise RuntimeError("Camera.initialise: camera failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            logger.warning("Camera.initialise: %s is not alive after initialisation.", self.name)
            raise RuntimeError("Camera.initialise: camera is not alive after initialisation.")
        self.set_exposure(self.default_exposure_time)
        if self.readout_mode is not None:
            self.set_readout_mode(self.readout_mode)
        logger.debug("Camera.initialise: %s initialised.", self.name)

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the camera and clear lifecycle flags.

        Parameters
        ----------
        force
            If True, binding implementations may force cleanup.

        Returns
        -------
        None
        """
        logger.debug("Camera.finalise: finalising %s with force=%s.", self.name, force)
        self._finalise(force=force)
        self._is_initialised = False
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Query whether the camera hardware is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the subclass reports the camera is alive.
        """
        self._is_alive = self._check_is_alive()
        return self._is_alive

    def is_initialised(self) -> bool:
        """
        Return whether initialise has succeeded.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the camera is marked initialised.
        """
        return self._is_initialised

    def stop(self) -> None:
        """
        Stop camera activity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="stop")
        logger.debug("Camera.stop: stopping %s.", self.name)
        self._stop()

    def get_frame(self, normalise: bool = False) -> np.ndarray:
        """
        Capture one frame from the camera.

        Parameters
        ----------
        normalise
            If True, return a floating-point frame scaled to [0, 1].

        Returns
        -------
        np.ndarray
            Captured image frame.
        """
        self._require_ready(action="get_frame")
        logger.debug("Camera.get_frame: capturing frame from %s with normalise=%s.", self.name, normalise)
        frame = self._get_frame()
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Camera.get_frame: _get_frame must return np.ndarray, received {type(frame)}.")
        if frame.shape != self.image.shape:
            raise ValueError(f"Camera.get_frame: expected frame shape {self.image.shape}, received {frame.shape}.")
        return self.normalise_frame(frame=frame) if normalise else frame

    def set_exposure(self, exposure_time: float | int) -> None:
        """
        Set the camera exposure time.

        Parameters
        ----------
        exposure_time
            Positive exposure time in milliseconds.

        Returns
        -------
        None
        """
        self._require_ready(action="set_exposure")
        validated_exposure_time = self._validate_exposure_time(exposure_time)
        logger.debug("Camera.set_exposure: setting %s exposure to %s ms.", self.name, validated_exposure_time)
        self._set_exposure(exposure_time=validated_exposure_time)
        self._current_exposure = validated_exposure_time

    def get_exposure(self) -> float | int | None:
        """
        Return the last exposure time set through this Camera object.

        Parameters
        ----------
        None

        Returns
        -------
        float | int | None
            Cached exposure time in milliseconds, or None before it is set.
        """
        return self._current_exposure

    def set_readout_mode(self, readout_mode: CameraReadoutMode) -> None:
        """
        Set the camera readout mode.

        Parameters
        ----------
        readout_mode
            Camera readout mode.

        Returns
        -------
        None
        """
        self._require_ready(action="set_readout_mode")
        validated_readout_mode = self._validate_readout_mode(readout_mode=readout_mode)
        logger.debug("Camera.set_readout_mode: setting %s readout mode to %s.", self.name, validated_readout_mode)
        self._set_readout_mode(readout_mode=validated_readout_mode)
        self.readout_mode = validated_readout_mode

    def disable_live_mode(self) -> None:
        """
        Disable live image acquisition when supported by the binding.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="disable_live_mode")
        logger.debug("Camera.disable_live_mode: disabling live mode for %s.", self.name)
        self._disable_live_mode()

    @staticmethod
    def normalise_frame(frame: np.ndarray) -> np.ndarray:
        """
        Return a floating-point copy of a frame scaled to [0, 1].

        Parameters
        ----------
        frame
            Image array to normalise.

        Returns
        -------
        np.ndarray
            Floating-point normalised image.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Camera.normalise_frame: frame must be np.ndarray, received {type(frame)}.")
        frame_float = frame.astype(np.float64, copy=False)
        min_value = float(np.min(frame_float))
        max_value = float(np.max(frame_float))
        if max_value == min_value:
            return np.zeros_like(frame_float, dtype=np.float64)
        return (frame_float - min_value) / (max_value - min_value)

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """Initialise binding-specific camera resources."""
        raise NotImplementedError

    @abstractmethod
    def _finalise(self, force: bool = False) -> None:
        """Finalise binding-specific camera resources."""
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        """Return whether binding-specific camera resources are alive."""
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        """Stop binding-specific camera activity."""
        raise NotImplementedError

    @abstractmethod
    def _get_frame(self) -> np.ndarray:
        """Capture one binding-specific frame."""
        raise NotImplementedError

    @abstractmethod
    def _set_exposure(self, exposure_time: float | int) -> None:
        """Set binding-specific exposure time."""
        raise NotImplementedError

    def _set_readout_mode(self, readout_mode: CameraReadoutMode) -> None:
        """
        Set binding-specific readout mode.

        Parameters
        ----------
        readout_mode
            Camera readout mode.

        Returns
        -------
        None
        """
        return

    def _disable_live_mode(self) -> None:
        """
        Disable binding-specific live mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return


class CameraFactory:
    """Factory for creating Camera instances from typed configs."""

    @staticmethod
    def create(
            config: CameraConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> Camera:
        """
        Create a Camera from a CameraConfig.

        Parameters
        ----------
        config
            Typed camera configuration describing the desired binding and
            shared construction options.
        peripheral_controllers
            One PeripheralController or a list of available controllers. Only
            bindings that require a controller inspect this argument.
        binding_options
            Extra binding-specific constructor options.

        Returns
        -------
        Camera
            A camera instance for the requested binding.
        """
        if not isinstance(config, CameraConfig):
            raise TypeError(f"CameraFactory.create: expected CameraConfig, received {type(config)}.")

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.camera import VirtualCamera
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="CameraFactory.create",
            )
            camera = VirtualCamera(
                peripheral_ctrl=peripheral_ctrl,
                image=config.image,
                name=config.name or VirtualCamera.DEFAULT_NAME,
                default_exposure_time=config.default_exposure_time,
                readout_mode=config.readout_mode,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            camera.config = config.copy()
            return camera

        if config.binding == BindingType.MMC:
            from evomachine.bindings.mmc.camera import MMCCamera

            camera = MMCCamera(
                image=config.image,
                name=config.name or MMCCamera.DEFAULT_NAME,
                default_exposure_time=config.default_exposure_time,
                readout_mode=config.readout_mode,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            camera.config = config.copy()
            return camera

        if config.binding == BindingType.PVCAM:
            from evomachine.bindings.pvcam.camera import PVCAMCamera

            camera = PVCAMCamera(
                image=config.image,
                name=config.name or PVCAMCamera.DEFAULT_NAME,
                default_exposure_time=config.default_exposure_time,
                readout_mode=config.readout_mode,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            camera.config = config.copy()
            return camera

        raise ValueError(f"CameraFactory.create: unsupported camera binding {config.binding}.")

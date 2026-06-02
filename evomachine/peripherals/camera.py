from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evomachine.bindings.binding_types import BindingType
from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig, update_dataclass_config
from evomachine.types import FilterWheelType, LEDType


def _get_evomachine_dir() -> Path:
    from evomachine.config import EVOMACHINE_DIR

    return EVOMACHINE_DIR


def _use_sync_board() -> bool:
    from evomachine.config import USE_SYNC_BOARD

    return USE_SYNC_BOARD


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


@dataclass
class CameraSystemConfig:
    objective: ObjectiveConfigType
    image: ImageConfigType
    focus: Any
    autofocus: Any
    leds: list[LEDType]
    filters: list[FilterWheelType]
    path_to_save: Path
    default_exposure_time: float | int = 200
    default_focus_channel_id: int = 0
    cam_pxl_size: float = 6.5

    def copy(self) -> "CameraSystemConfig":
        return CameraSystemConfig(**self.__dict__)

    @property
    def pxl_size(self) -> float:
        return self.cam_pxl_size / self.objective.mag

    @property
    def fov_size(self) -> float:
        return self.cam_pxl_size / self.objective.mag * self.image.pxl_vert

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveConfigType):
            raise TypeError(f"objective must be an ObjectiveConfigType object. Provided {self.objective}.")
        if not isinstance(self.image, ImageConfigType):
            raise TypeError(f"image must be an ImageConfigType object. Provided {self.image}.")
        if not (isinstance(self.leds, list) and all(isinstance(led, LEDType) for led in self.leds)) \
                or len(self.leds) == 0 or LEDType.NO_LED not in self.leds:
            raise ConfigError("Invalid LED list.", ErrorCode.ERROR_CONFIG)
        if not (isinstance(self.filters, list) and all(isinstance(filter_type, FilterWheelType) for filter_type in self.filters)) \
                or len(self.filters) == 0:
            raise ConfigError("Invalid filter list.", ErrorCode.ERROR_CONFIG)
        if isinstance(self.path_to_save, str):
            self.path_to_save = Path(self.path_to_save)
        if not isinstance(self.path_to_save, Path):
            raise ConfigError("Invalid path_to_save.", ErrorCode.ERROR_CONFIG)
        if not self.image.pxl_vert == self.image.pxl_horiz:
            raise ConfigError("Currently limited to square images.", ErrorCode.ERROR_FOCUS_CONFIG)
        if not isinstance(self.default_exposure_time, int | float) or self.default_exposure_time <= 0:
            raise TypeError(f"Invalid default_exposure_time {self.default_exposure_time}.")
        if not (isinstance(self.default_focus_channel_id, int) and 0 <= self.default_focus_channel_id < len(self.leds)):
            raise TypeError(f"Invalid default_focus_channel_id {self.default_focus_channel_id}.")


class CameraSystemConfigFactory:
    @staticmethod
    def get_available_leds() -> list[LEDType]:
        if _use_sync_board():
            return [
                LEDType.NO_LED,
                LEDType.LED_385_NM,
                LEDType.LED_450_NM,
                LEDType.LED_515_NM,
                LEDType.LED_565_NM,
                LEDType.LED_645_NM,
                LEDType.LED_OVERHEAD,
                LEDType.LED_OVERHEAD_TIGER,
            ]
        return [LEDType.NO_LED, LEDType.LED_405_NM, LEDType.LED_450_NM, LEDType.LED_505_NM, LEDType.LED_538_NM]

    @staticmethod
    def get_available_filters() -> list[FilterWheelType]:
        return [
            FilterWheelType.FILTER,
            FilterWheelType.FILTER_465nm,
            FilterWheelType.FILTER_527nm,
            FilterWheelType.FILTER_592nm,
            FilterWheelType.BLOCKING,
            FilterWheelType.NO_FILTER,
        ]

    @staticmethod
    def default_oil_config(path_to_save: Path | None = None) -> CameraSystemConfig:
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory
        from evomachine.softwarefocus import SoftwareFocusConfigFactory

        evomachine_dir = _get_evomachine_dir()
        return CameraSystemConfig(
            objective=ObjectiveConfigTypeFactory.default_oil(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=SoftwareFocusConfigFactory.default_config(),
            autofocus=TigerAutofocusConfigFactory.default_oil_config(),
            leds=CameraSystemConfigFactory.get_available_leds(),
            filters=CameraSystemConfigFactory.get_available_filters(),
            path_to_save=evomachine_dir.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

    @staticmethod
    def default_air_config(path_to_save: Path | None = None) -> CameraSystemConfig:
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory
        from evomachine.softwarefocus import SoftwareFocusConfigFactory

        evomachine_dir = _get_evomachine_dir()
        return CameraSystemConfig(
            objective=ObjectiveConfigTypeFactory.default_air(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=SoftwareFocusConfigFactory.default_config(),
            autofocus=TigerAutofocusConfigFactory.default_config(),
            leds=CameraSystemConfigFactory.get_available_leds(),
            filters=CameraSystemConfigFactory.get_available_filters(),
            path_to_save=evomachine_dir.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )


@dataclass
class CameraConfig:
    """Configuration object used by CameraFactory to create camera devices."""

    binding: BindingType
    image: ImageConfigType
    default_exposure_time: float | int = 200
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True
    imaging_mode: str | None = None

    def __post_init__(self) -> None:
        """
        Validate camera configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.binding, BindingType):
            raise TypeError(f"CameraConfig: binding must be BindingType, received {type(self.binding)}.")
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
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"CameraConfig: name must be str or None, received {type(self.name)}.")
        if not isinstance(self.check_initialised, bool):
            raise TypeError(f"CameraConfig: check_initialised must be bool, received {type(self.check_initialised)}.")
        if not isinstance(self.check_alive, bool):
            raise TypeError(f"CameraConfig: check_alive must be bool, received {type(self.check_alive)}.")
        if self.imaging_mode is not None and not isinstance(self.imaging_mode, str):
            raise TypeError(f"CameraConfig: imaging_mode must be str or None, received {type(self.imaging_mode)}.")

    def copy(self) -> "CameraConfig":
        return CameraConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> "CameraConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"CameraConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return CameraConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> "CameraConfig":
        if not isinstance(updates, dict):
            raise TypeError("CameraConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


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
            imaging_mode: str | None = None,
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
        imaging_mode
            Optional binding-specific imaging mode applied during initialise().
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
        self.imaging_mode: str | None = imaging_mode
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
            raise RuntimeError(f"Camera.{action}: camera is not initialised.")
        if self._check_alive and not self.is_alive():
            raise RuntimeError(f"Camera.{action}: camera is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the camera and apply default exposure and imaging mode.

        Parameters
        ----------
        force
            If True, run initialisation even when already initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            return
        self._is_initialised = self._initialise(force=force)
        if self._check_initialised and not self._is_initialised:
            raise RuntimeError("Camera.initialise: camera failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            raise RuntimeError("Camera.initialise: camera is not alive after initialisation.")
        self.set_exposure(self.default_exposure_time)
        if self.imaging_mode is not None:
            self.set_imaging_mode(self.imaging_mode)

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

    def update_config(self, config: CameraConfig | None = None, **updates: Any) -> None:
        """
        Replace or update camera runtime configuration.

        Image shape/dtype changes require reinitialisation because frame
        validation depends on the image config. Exposure, imaging mode, name,
        and readiness checks are applied in place when possible.
        """
        current_config = self.config
        if current_config is None:
            if config is None:
                raise RuntimeError("Camera.update_config: this camera was not created from a CameraConfig.")
            new_config = config.copy()
        else:
            new_config = update_dataclass_config(current_config=current_config, replacement=config, **updates)
        if new_config.binding != (current_config.binding if current_config is not None else new_config.binding):
            raise RuntimeError("Camera.update_config: changing camera binding requires recreating the camera.")

        was_initialised = self.is_initialised()
        reinitialise = current_config is not None and new_config.image != current_config.image
        if reinitialise and was_initialised:
            self.stop()
            self.finalise(force=True)

        self.config = new_config.copy()
        self.image = new_config.image
        self.name = new_config.name or self.name
        self.default_exposure_time = self._validate_exposure_time(new_config.default_exposure_time)
        self.imaging_mode = new_config.imaging_mode
        self._check_initialised = new_config.check_initialised
        self._check_alive = new_config.check_alive

        if reinitialise and was_initialised:
            self.initialise(force=True)
        elif was_initialised:
            self.set_exposure(self.default_exposure_time)
            if self.imaging_mode is not None:
                self.set_imaging_mode(self.imaging_mode)

    def set_imaging_mode(self, imaging_mode: str) -> None:
        """
        Set a binding-specific camera imaging mode.

        Parameters
        ----------
        imaging_mode
            Binding-specific mode name.

        Returns
        -------
        None
        """
        self._require_ready(action="set_imaging_mode")
        if not isinstance(imaging_mode, str):
            raise TypeError(f"Camera.set_imaging_mode: imaging_mode must be str, received {type(imaging_mode)}.")
        self._set_imaging_mode(imaging_mode=imaging_mode)
        self.imaging_mode = imaging_mode

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

    def _set_imaging_mode(self, imaging_mode: str) -> None:
        """
        Set binding-specific imaging mode.

        Parameters
        ----------
        imaging_mode
            Binding-specific mode name.

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
                imaging_mode=config.imaging_mode,
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
                imaging_mode=config.imaging_mode,
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
                imaging_mode=config.imaging_mode,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            camera.config = config.copy()
            return camera

        raise ValueError(f"CameraFactory.create: unsupported camera binding {config.binding}.")

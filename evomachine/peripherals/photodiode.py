from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import Field, field_validator

from evomachine.bindings.binding_types import BindingType
from evomachine.config_models import EvoConfig
from evomachine.config import get_logger
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig

logger = get_logger(name=__name__, is_peripheral=True)


class PhotodiodeReadingRange(EvoConfig):
    """Raw reading bounds used to scale photodiode readings to percentages."""

    minimum_reading: float
    maximum_reading: float

    def __init__(self, *args: object, **data: object) -> None:
        if args:
            if len(args) != 2 or data:
                raise TypeError("PhotodiodeReadingRange expects minimum_reading and maximum_reading.")
            data = {"minimum_reading": args[0], "maximum_reading": args[1]}
        super().__init__(**data)

    @field_validator("minimum_reading", "maximum_reading", mode="before")
    @classmethod
    def _validate_reading_type(cls, value: object, info) -> object:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(
                f"PhotodiodeReadingRange: {info.field_name} must be numeric, received {type(value)}."
            )
        return value

    def model_post_init(self, __context) -> None:
        """
        Validate raw photodiode reading bounds after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        self.minimum_reading = self._validate_reading(
            reading=self.minimum_reading,
            name="minimum_reading",
        )
        self.maximum_reading = self._validate_reading(
            reading=self.maximum_reading,
            name="maximum_reading",
        )
        if self.minimum_reading >= self.maximum_reading:
            raise ValueError(
                "PhotodiodeReadingRange: minimum_reading must be less than maximum_reading."
            )

    @staticmethod
    def _validate_reading(reading: float, name: str) -> float:
        """
        Return a validated numeric photodiode reading.

        Parameters
        ----------
        reading
            Candidate raw reading value.
        name
            Field name used in exception messages.

        Returns
        -------
        float
            The reading converted to float.
        """
        if not isinstance(reading, int | float) or isinstance(reading, bool):
            raise TypeError(
                f"PhotodiodeReadingRange: {name} must be numeric, received {type(reading)}."
            )
        return float(reading)


class PhotodiodeConfig(PeripheralConfig):
    """Configuration object used by PhotodiodeFactory to create photodiodes."""

    channel: int = 8
    reading_range: PhotodiodeReadingRange = Field(
        default_factory=lambda: PhotodiodeReadingRange(minimum_reading=0.0, maximum_reading=1.0)
    )

    @field_validator("channel", mode="before")
    @classmethod
    def _validate_channel_type(cls, value: object) -> object:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"PhotodiodeConfig: channel must be int, received {type(value)}.")
        return value

    @field_validator("reading_range", mode="before")
    @classmethod
    def _validate_reading_range_type(cls, value: object) -> object:
        if not isinstance(value, PhotodiodeReadingRange | dict):
            raise TypeError(
                "PhotodiodeConfig: reading_range must be PhotodiodeReadingRange, "
                f"received {type(value)}."
            )
        return value

    def model_post_init(self, __context) -> None:
        """
        Validate photodiode configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        super().model_post_init(__context)
        self.channel = Photodiode.validate_channel(channel=self.channel)
        if not isinstance(self.reading_range, PhotodiodeReadingRange):
            raise TypeError(
                "PhotodiodeConfig: reading_range must be PhotodiodeReadingRange, "
                f"received {type(self.reading_range)}."
            )


class Photodiode(Peripheral):
    """Base class for photodiode sensors that report normalised light readings."""

    def __init__(
            self,
            peripheral_ctrl: PeripheralController,
            channel: int,
            reading_range: PhotodiodeReadingRange,
            name: str,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared photodiode state.

        Parameters
        ----------
        peripheral_ctrl
            PeripheralController that owns the binding-specific connection.
        channel
            Binding-specific photodiode channel identifier.
        reading_range
            Raw reading bounds used to scale raw readings to [0, 100].
        name
            Human-readable photodiode name used in error messages.
        check_initialised
            If True, public methods require successful initialise().
        check_alive
            If True, public methods require the controller to report alive.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, PeripheralController):
            raise TypeError(
                f"Photodiode.__init__: peripheral_ctrl must be PeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        if not isinstance(reading_range, PhotodiodeReadingRange):
            raise TypeError(
                "Photodiode.__init__: reading_range must be PhotodiodeReadingRange, "
                f"received {type(reading_range)}."
            )
        self.peripheral_ctrl: PeripheralController = peripheral_ctrl
        self.channel: int = self.validate_channel(channel=channel)
        self.reading_range: PhotodiodeReadingRange = reading_range
        self.name: str = name
        self.check_initialised: bool = check_initialised
        self.check_alive: bool = check_alive
        self._is_initialised: bool = False
        self.config: PhotodiodeConfig | None = None

    @staticmethod
    def validate_channel(channel: int) -> int:
        """
        Return a validated photodiode channel identifier.

        Parameters
        ----------
        channel
            Candidate photodiode channel. It must be a positive integer.

        Returns
        -------
        int
            Validated photodiode channel.
        """
        if not isinstance(channel, int) or isinstance(channel, bool):
            raise TypeError(f"Photodiode.validate_channel: channel must be int, received {type(channel)}.")
        if channel <= 0:
            raise ValueError(f"Photodiode.validate_channel: channel must be positive, received {channel}.")
        return channel

    def _require_ready(self, action: str) -> None:
        """
        Raise when a photodiode action is not allowed by readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self.check_initialised and not self._is_initialised:
            logger.warning("Photodiode.%s: %s is not initialised.", action, self.name)
            raise RuntimeError(f"Photodiode.{action}: photodiode is not initialised.")
        if self.check_alive and not self.is_alive():
            logger.warning("Photodiode.%s: %s is not alive.", action, self.name)
            raise RuntimeError(f"Photodiode.{action}: photodiode is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the photodiode after checking its peripheral controller.

        Parameters
        ----------
        force
            If True, run initialisation even when already initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            logger.debug("Photodiode.initialise: %s already initialised; skipping.", self.name)
            return
        if self.check_initialised and not self.peripheral_ctrl.is_initialised():
            logger.warning("Photodiode.initialise: %s controller is not initialised.", self.peripheral_ctrl.name)
            raise RuntimeError(
                f"Photodiode.initialise: {self.peripheral_ctrl.name} is not initialised."
            )
        if self.check_alive and not self.peripheral_ctrl.is_alive():
            logger.warning("Photodiode.initialise: %s controller is not alive.", self.peripheral_ctrl.name)
            raise RuntimeError(f"Photodiode.initialise: {self.peripheral_ctrl.name} is not alive.")
        logger.debug("Photodiode.initialise: initialising %s with force=%s.", self.name, force)
        self._is_initialised = self._initialise(force=force)
        if self.check_initialised and not self._is_initialised:
            logger.warning("Photodiode.initialise: %s failed to initialise.", self.name)
            raise RuntimeError("Photodiode.initialise: photodiode failed to initialise.")

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the photodiode and clear lifecycle state.

        Parameters
        ----------
        force
            If True, binding implementations may force cleanup.

        Returns
        -------
        None
        """
        logger.debug("Photodiode.finalise: finalising %s with force=%s.", self.name, force)
        self._finalise(force=force)
        self._is_initialised = False

    def is_initialised(self) -> bool:
        """
        Return whether initialise has succeeded.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the photodiode is marked initialised.
        """
        return self._is_initialised

    def is_alive(self) -> bool:
        """
        Query whether the photodiode controller is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the underlying peripheral controller reports alive.
        """
        return self.peripheral_ctrl.is_alive()

    def stop(self) -> None:
        """
        Stop photodiode activity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="stop")
        logger.debug("Photodiode.stop: stopping %s.", self.name)
        self._stop()

    def set_reading_range(
            self,
            minimum_reading: float,
            maximum_reading: float,
    ) -> None:
        """
        Update the raw reading range used to normalise photodiode values.

        Parameters
        ----------
        minimum_reading
            Raw reading that maps to 0 percent.
        maximum_reading
            Raw reading that maps to 100 percent.

        Returns
        -------
        None
        """
        logger.debug(
            "Photodiode.set_reading_range: setting %s range to (%s, %s).",
            self.name,
            minimum_reading,
            maximum_reading,
        )
        self.reading_range = PhotodiodeReadingRange(
            minimum_reading=minimum_reading,
            maximum_reading=maximum_reading,
        )

    def read_photodiode(self) -> float:
        """
        Return the current photodiode reading scaled to [0, 100].

        Parameters
        ----------
        None

        Returns
        -------
        float
            Calibrated photodiode percentage clamped between 0 and 100.
        """
        self._require_ready(action="read_photodiode")
        raw_reading = self._read_raw_photodiode()
        logger.debug("Photodiode.read_photodiode: %s raw reading=%s.", self.name, raw_reading)
        return self._normalise_reading(raw_reading=raw_reading)

    def _normalise_reading(self, raw_reading: float) -> float:
        """
        Convert one raw photodiode reading to a clamped percentage.

        Parameters
        ----------
        raw_reading
            Raw photodiode reading returned by the binding.

        Returns
        -------
        float
            Calibrated photodiode percentage clamped between 0 and 100.
        """
        raw_value = PhotodiodeReadingRange._validate_reading(
            reading=raw_reading,
            name="raw_reading",
        )
        span = self.reading_range.maximum_reading - self.reading_range.minimum_reading
        normalised = (raw_value - self.reading_range.minimum_reading) / span * 100.0
        return min(100.0, max(0.0, normalised))

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """
        Initialise binding-specific photodiode resources.

        Parameters
        ----------
        force
            If True, binding implementations may force initialisation.

        Returns
        -------
        bool
            True when binding-specific initialisation succeeds.
        """
        raise NotImplementedError

    @abstractmethod
    def _finalise(self, force: bool = False) -> None:
        """
        Finalise binding-specific photodiode resources.

        Parameters
        ----------
        force
            If True, binding implementations may force cleanup.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        """
        Stop binding-specific photodiode activity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _read_raw_photodiode(self) -> float:
        """
        Return one raw binding-specific photodiode reading.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Raw photodiode reading before calibration.
        """
        raise NotImplementedError


class PhotodiodeFactory:
    """Factory for creating Photodiode instances from typed configs."""

    @staticmethod
    def create(
            config: PhotodiodeConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> Photodiode:
        """
        Create a Photodiode from a PhotodiodeConfig.

        Parameters
        ----------
        config
            Typed photodiode configuration describing the desired binding and
            shared construction options.
        peripheral_controllers
            One PeripheralController or a list of available controllers. The
            requested photodiode binding selects the required controller type.
        binding_options
            Extra binding-specific constructor options.

        Returns
        -------
        Photodiode
            A photodiode instance for the requested binding.
        """
        if not isinstance(config, PhotodiodeConfig):
            raise TypeError(
                f"PhotodiodeFactory.create: expected PhotodiodeConfig, received {type(config)}."
            )

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
            from evomachine.bindings.virtual.photodiode import VirtualPhotodiode

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="PhotodiodeFactory.create",
            )
            photodiode = VirtualPhotodiode(
                peripheral_ctrl=peripheral_ctrl,
                channel=config.channel,
                reading_range=config.reading_range,
                name=config.name or VirtualPhotodiode.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            photodiode.config = config.copy()
            return photodiode

        if config.binding == BindingType.SYNCBOARD:
            from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
            from evomachine.bindings.syncboard.photodiode import SyncBoardPhotodiode

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=SyncBoardPeripheralController,
                action="PhotodiodeFactory.create",
            )
            photodiode = SyncBoardPhotodiode(
                peripheral_ctrl=peripheral_ctrl,
                channel=config.channel,
                reading_range=config.reading_range,
                name=config.name or SyncBoardPhotodiode.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            photodiode.config = config.copy()
            return photodiode

        raise ValueError(f"PhotodiodeFactory.create: unsupported photodiode binding {config.binding}.")

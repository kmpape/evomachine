from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig


@dataclass
class PhotodiodeReadingRange:
    """Raw reading bounds used to scale photodiode readings to percentages."""

    minimum_reading: float
    maximum_reading: float

    def __post_init__(self) -> None:
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


@dataclass
class PhotodiodeConfig(PeripheralConfig):
    """Configuration object used by PhotodiodeFactory to create photodiodes."""

    channel: int = 8
    reading_range: PhotodiodeReadingRange = field(
        default_factory=lambda: PhotodiodeReadingRange(0.0, 1.0)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.channel = Photodiode.validate_channel(channel=self.channel)
        if not isinstance(self.reading_range, PhotodiodeReadingRange):
            raise TypeError(
                "PhotodiodeConfig: reading_range must be PhotodiodeReadingRange, "
                f"received {type(self.reading_range)}."
            )

    def copy(self) -> "PhotodiodeConfig":
        return PhotodiodeConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> "PhotodiodeConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"PhotodiodeConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return PhotodiodeConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> "PhotodiodeConfig":
        if not isinstance(updates, dict):
            raise TypeError("PhotodiodeConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


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
        super().__init__(
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

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

    def _before_initialise(self, force: bool = False) -> None:
        """Check the controller before photodiode initialisation."""
        if self._check_initialised and not self.peripheral_ctrl.is_initialised():
            raise RuntimeError(
                f"Photodiode.initialise: {self.peripheral_ctrl.name} is not initialised."
            )
        if self._check_alive and not self.peripheral_ctrl.is_alive():
            raise RuntimeError(f"Photodiode.initialise: {self.peripheral_ctrl.name} is not alive.")

    def _check_is_alive(self) -> bool:
        """Return whether the photodiode controller is alive."""
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
        self.reading_range = PhotodiodeReadingRange(
            minimum_reading=minimum_reading,
            maximum_reading=maximum_reading,
        )

    def _apply_config(self, config: PhotodiodeConfig) -> None:
        """Apply photodiode-specific config fields."""
        self.channel = config.channel
        self.reading_range = config.reading_range

    def _config_requires_reinitialise(self, current_config: PhotodiodeConfig, new_config: PhotodiodeConfig) -> bool:
        """Return whether photodiode config changes require reinitialisation."""
        return new_config.channel != current_config.channel

    def update_config(self, config: PhotodiodeConfig | None = None, **updates: Any) -> None:
        """Replace or update photodiode configuration at runtime."""
        super().update_config(config=config, **updates)

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

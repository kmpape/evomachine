from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral
from evomachine.types import FilterWheelType


@dataclass
class FilterWheelConfig:
    """Configuration object used by FilterWheelFactory to create filter wheels."""

    binding: BindingType
    available_filters: list[FilterWheelType]
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BindingType):
            raise TypeError(f"FilterWheelConfig: binding must be BindingType, received {type(self.binding)}.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"FilterWheelConfig: name must be str or None, received {type(self.name)}.")
        if not isinstance(self.check_initialised, bool):
            raise TypeError(
                f"FilterWheelConfig: check_initialised must be bool, received {type(self.check_initialised)}."
            )
        if not isinstance(self.check_alive, bool):
            raise TypeError(f"FilterWheelConfig: check_alive must be bool, received {type(self.check_alive)}.")
        self.available_filters = FilterWheel._validate_available_filters(
            available_filters=self.available_filters,
        )

    def copy(self) -> "FilterWheelConfig":
        return FilterWheelConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> "FilterWheelConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"FilterWheelConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return FilterWheelConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> "FilterWheelConfig":
        if not isinstance(updates, dict):
            raise TypeError("FilterWheelConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


class FilterWheel(Peripheral):
    """
    Base class for microscope filter wheels.

    This class owns lifecycle state, readiness checks, available-filter
    validation, and the software-known current filter. Subclasses implement only
    low-level hardware operations.
    """

    def __init__(
            self,
            name: str,
            available_filters: list[FilterWheelType],
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared filter wheel state.

        Parameters
        ----------
        name
            Human-readable filter wheel name.
        available_filters
            Non-empty list of filters that can be set through this filter wheel.
        check_initialised
            If True, public hardware methods raise RuntimeError before
            initialise() succeeds.
        check_alive
            If True, public hardware methods raise RuntimeError when the device
            does not report alive.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )
        self._available_filters: list[FilterWheelType] = self._validate_available_filters(
            available_filters=available_filters,
        )
        self._current_filter_type: FilterWheelType = FilterWheelType.UNKNOWN

    @staticmethod
    def _validate_available_filters(available_filters: list[FilterWheelType]) -> list[FilterWheelType]:
        """
        Validate available filter configuration.

        Parameters
        ----------
        available_filters
            Candidate list of filters supported by this filter wheel.

        Returns
        -------
        list[FilterWheelType]
            Copy of the validated list.
        """
        if not isinstance(available_filters, list):
            raise TypeError(
                f"FilterWheel._validate_available_filters: expected list, received {type(available_filters)}."
            )
        if len(available_filters) == 0:
            raise ValueError("FilterWheel._validate_available_filters: available_filters must not be empty.")
        if not all(isinstance(filter_type, FilterWheelType) for filter_type in available_filters):
            raise TypeError("FilterWheel._validate_available_filters: all entries must be FilterWheelType.")
        return list(available_filters)

    @staticmethod
    def _validate_filter_type(filter_type: FilterWheelType, action: str) -> FilterWheelType:
        """
        Validate a filter wheel type value.

        Parameters
        ----------
        filter_type
            Candidate filter wheel type.
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        FilterWheelType
            The validated filter wheel type.
        """
        if not isinstance(filter_type, FilterWheelType):
            raise TypeError(f"FilterWheel.{action}: expected FilterWheelType, received {type(filter_type)}.")
        return filter_type

    def _post_initialise(self, force: bool = False) -> None:
        """Cache current hardware filter after initialisation."""
        self._current_filter_type = self._validate_filter_type(
            filter_type=self._get_filter_wheel(),
            action="initialise",
        )

    def stop(self) -> None:
        """
        Stop filter wheel activity.

        Filter wheels do not currently expose a generic hardware stop command in
        evomachine, so this method is intentionally a no-op.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def get_available_filters(self) -> list[FilterWheelType]:
        """
        Return the configured filters that can be set.

        Parameters
        ----------
        None

        Returns
        -------
        list[FilterWheelType]
            Copy of the configured available filters.
        """
        return list(self._available_filters)

    def _apply_config(self, config: FilterWheelConfig) -> None:
        """Apply filter-wheel-specific config fields."""
        self._available_filters = self._validate_available_filters(config.available_filters)
        if self._current_filter_type not in self._available_filters:
            self._current_filter_type = FilterWheelType.UNKNOWN

    def _config_requires_reinitialise(self, current_config: FilterWheelConfig, new_config: FilterWheelConfig) -> bool:
        """Return whether filter wheel config changes require reinitialisation."""
        return new_config.available_filters != current_config.available_filters

    def _before_config_reinitialise(self) -> None:
        """Finalise filter wheel before config reinitialisation without stop."""
        return

    def update_config(self, config: "FilterWheelConfig | None" = None, **updates: Any) -> None:
        """Replace or update filter wheel configuration at runtime."""
        super().update_config(config=config, **updates)

    def get_filter_wheel(self) -> FilterWheelType:
        """
        Return the software-known current filter wheel position.

        Parameters
        ----------
        None

        Returns
        -------
        FilterWheelType
            Last cached filter wheel type. This is FilterWheelType.UNKNOWN until
            initialise() or set_filter_wheel() records a position.
        """
        return self._current_filter_type

    def set_filter_wheel(self, filter_type: FilterWheelType, force: bool = False) -> None:
        """
        Set the filter wheel if needed.

        Parameters
        ----------
        filter_type
            Filter wheel type to set.
        force
            If True, send the hardware command even when filter_type matches the
            cached current filter.

        Returns
        -------
        None
        """
        self._require_ready(action="set_filter_wheel")
        filter_type = self._validate_filter_type(filter_type=filter_type, action="set_filter_wheel")
        if filter_type not in self._available_filters:
            raise ValueError(f"FilterWheel.set_filter_wheel: unavailable filter {filter_type}.")
        if not force and filter_type == self._current_filter_type:
            return
        self._set_filter_wheel(filter_type=filter_type)
        self._current_filter_type = filter_type

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """
        Perform binding-specific initialisation.

        Implementations should prepare the filter wheel for use and return True
        only when the device is ready enough for _get_filter_wheel().

        Parameters
        ----------
        force
            If True, re-run setup even if a binding already has an open connection.

        Returns
        -------
        bool
            True when initialisation succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    def _finalise(self, force: bool = False) -> None:
        """
        Perform binding-specific cleanup.

        Parameters
        ----------
        force
            If True, force cleanup even when normal cleanup would be skipped.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        """
        Query whether the filter wheel connection is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the filter wheel is reachable.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_filter_wheel(self) -> FilterWheelType:
        """
        Query the current hardware filter wheel position.

        Implementations should return FilterWheelType.UNKNOWN when readback is
        not available or cannot yet be parsed.

        Parameters
        ----------
        None

        Returns
        -------
        FilterWheelType
            Current hardware filter wheel position.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_filter_wheel(self, filter_type: FilterWheelType) -> None:
        """
        Set the hardware filter wheel position.

        Implementations should send exactly one hardware command for filter_type
        and should raise a standard Python exception if the command fails.

        Parameters
        ----------
        filter_type
            Filter wheel type to set.

        Returns
        -------
        None
        """
        raise NotImplementedError


class FilterWheelFactory:
    """Factory for creating FilterWheel instances from a typed FilterWheelConfig."""

    @staticmethod
    def create(
            config: FilterWheelConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> FilterWheel:
        """
        Create a FilterWheel from a FilterWheelConfig.

        Parameters
        ----------
        config
            Typed filter wheel configuration describing the desired binding and
            binding-neutral construction options.
        peripheral_controllers
            One PeripheralController or a list of available PeripheralController
            instances. The requested filter wheel binding selects the required
            controller type.
        binding_options
            Binding-specific keyword arguments such as Tiger card address or
            position mapping.

        Returns
        -------
        FilterWheel
            A filter wheel instance for the requested binding.
        """
        if not isinstance(config, FilterWheelConfig):
            raise TypeError(f"FilterWheelFactory.create: expected FilterWheelConfig, received {type(config)}.")

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
            from evomachine.bindings.virtual.filterwheel import VirtualFilterWheel

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="FilterWheelFactory.create",
            )
            filter_wheel = VirtualFilterWheel(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or "Virtual Filter Wheel",
                available_filters=config.available_filters,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            filter_wheel.config = config.copy()
            return filter_wheel

        if config.binding == BindingType.ASI_TIGER:
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
            from evomachine.bindings.asitiger.filterwheel import TigerFilterWheel

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=TigerPeripheralController,
                action="FilterWheelFactory.create",
            )
            filter_wheel = TigerFilterWheel(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or "ASI Tiger Filter Wheel",
                available_filters=config.available_filters,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            filter_wheel.config = config.copy()
            return filter_wheel

        raise ValueError(f"FilterWheelFactory.create: unsupported filter wheel binding {config.binding}.")

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import field_validator

from evomachine.bindings.binding_types import BindingType
from evomachine.config import get_logger
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig
from evomachine.types import FilterWheelType

logger = get_logger(name=__name__, is_peripheral=True)


class FilterWheelConfig(PeripheralConfig):
    """Configuration object used by FilterWheelFactory to create filter wheels."""

    available_filters: list[FilterWheelType]

    @field_validator("available_filters", mode="before")
    @classmethod
    def _validate_available_filters_type(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError(f"FilterWheelConfig: available_filters must be list, received {type(value)}.")
        return value

    def model_post_init(self, __context) -> None:
        """
        Validate filter wheel factory configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super().model_post_init(__context)
        self.available_filters = FilterWheel._validate_available_filters(
            available_filters=self.available_filters,
        )


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
        self.name: str = name
        self._available_filters: list[FilterWheelType] = self._validate_available_filters(
            available_filters=available_filters,
        )
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self._check_initialised: bool = check_initialised
        self._check_alive: bool = check_alive
        self._current_filter_type: FilterWheelType = FilterWheelType.UNKNOWN
        self.config: FilterWheelConfig | None = None

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

    def _require_ready(self, action: str) -> None:
        """
        Raise when a hardware action is not allowed by current readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self._check_initialised and not self._is_initialised:
            logger.warning("FilterWheel.%s: %s is not initialised.", action, self.name)
            raise RuntimeError(f"FilterWheel.{action}: filter wheel is not initialised.")
        if self._check_alive and not self.is_alive():
            logger.warning("FilterWheel.%s: %s is not alive.", action, self.name)
            raise RuntimeError(f"FilterWheel.{action}: filter wheel is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the filter wheel and cache its current hardware position.

        Parameters
        ----------
        force
            If True, run initialisation even if the filter wheel is already
            initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            logger.debug("FilterWheel.initialise: %s already initialised; skipping.", self.name)
            return
        logger.debug("FilterWheel.initialise: initialising %s with force=%s.", self.name, force)
        self._is_initialised = self._initialise(force=force)
        if self._check_initialised and not self._is_initialised:
            logger.warning("FilterWheel.initialise: %s failed to initialise.", self.name)
            raise RuntimeError("FilterWheel.initialise: filter wheel failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            logger.warning("FilterWheel.initialise: %s is not alive after initialisation.", self.name)
            raise RuntimeError("FilterWheel.initialise: filter wheel is not alive after initialisation.")
        self._current_filter_type = self._validate_filter_type(
            filter_type=self._get_filter_wheel(),
            action="initialise",
        )
        logger.debug("FilterWheel.initialise: %s initialised at %s.", self.name, self._current_filter_type)

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the filter wheel and clear lifecycle flags.

        Parameters
        ----------
        force
            If True, subclass implementations may force cleanup.

        Returns
        -------
        None
        """
        logger.debug("FilterWheel.finalise: finalising %s with force=%s.", self.name, force)
        self._finalise(force=force)
        self._is_initialised = False
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Query whether the filter wheel hardware is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the subclass reports the hardware is alive.
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
            True when the filter wheel is marked initialised.
        """
        return self._is_initialised

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
        logger.debug("FilterWheel.stop: %s has no generic stop command; skipping.", self.name)
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
            logger.warning("FilterWheel.set_filter_wheel: %s unavailable filter %s.", self.name, filter_type)
            raise ValueError(f"FilterWheel.set_filter_wheel: unavailable filter {filter_type}.")
        if not force and filter_type == self._current_filter_type:
            logger.debug("FilterWheel.set_filter_wheel: %s already at %s; skipping.", self.name, filter_type)
            return
        logger.debug("FilterWheel.set_filter_wheel: setting %s to %s with force=%s.", self.name, filter_type, force)
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

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, update_dataclass_config
from evomachine.types import AutoFocusStatusType


@dataclass
class AutofocusConfig:
    """Configuration object used by AutofocusFactory to create autofocus peripherals."""

    binding: BindingType
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True

    def __post_init__(self) -> None:
        """
        Validate autofocus factory configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.binding, BindingType):
            raise TypeError(f"AutofocusConfig: binding must be BindingType, received {type(self.binding)}.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"AutofocusConfig: name must be str or None, received {type(self.name)}.")
        if not isinstance(self.check_initialised, bool):
            raise TypeError(
                f"AutofocusConfig: check_initialised must be bool, received {type(self.check_initialised)}."
            )
        if not isinstance(self.check_alive, bool):
            raise TypeError(f"AutofocusConfig: check_alive must be bool, received {type(self.check_alive)}.")

    def copy(self) -> "AutofocusConfig":
        return AutofocusConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> "AutofocusConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"AutofocusConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return AutofocusConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> "AutofocusConfig":
        if not isinstance(updates, dict):
            raise TypeError("AutofocusConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


class Autofocus(Peripheral):
    """
    Base class for autofocus peripherals.

    The class owns lifecycle state and readiness checks. Binding implementations
    provide device-specific configuration, status, and lock commands.
    """

    def __init__(
            self,
            name: str,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared autofocus state.

        Parameters
        ----------
        name
            Human-readable autofocus peripheral name.
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
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self._check_initialised: bool = check_initialised
        self._check_alive: bool = check_alive
        self.config: AutofocusConfig | None = None

    def _require_ready(self, action: str) -> None:
        """
        Raise when an autofocus action is not allowed by readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self._check_initialised and not self._is_initialised:
            raise RuntimeError(f"Autofocus.{action}: autofocus is not initialised.")
        if self._check_alive and not self.is_alive():
            raise RuntimeError(f"Autofocus.{action}: autofocus is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the autofocus peripheral.

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
            raise RuntimeError("Autofocus.initialise: autofocus failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            raise RuntimeError("Autofocus.initialise: autofocus is not alive after initialisation.")

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the autofocus peripheral and clear lifecycle flags.

        Parameters
        ----------
        force
            If True, subclass implementations may force cleanup.

        Returns
        -------
        None
        """
        self._finalise(force=force)
        self._is_initialised = False
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Query whether the autofocus peripheral is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the subclass reports the peripheral is alive.
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
            True when the autofocus peripheral is marked initialised.
        """
        return self._is_initialised

    def update_config(self, config: AutofocusConfig | None = None, **updates: Any) -> None:
        """Replace or update generic autofocus configuration."""
        current_config = self.config
        if current_config is None:
            if config is None:
                raise RuntimeError("Autofocus.update_config: this autofocus was not created from an AutofocusConfig.")
            new_config = config.copy()
        else:
            new_config = update_dataclass_config(current_config=current_config, replacement=config, **updates)
        if current_config is not None and new_config.binding != current_config.binding:
            raise RuntimeError("Autofocus.update_config: changing binding requires recreating the autofocus.")
        self.config = new_config.copy()
        self.name = new_config.name or self.name
        self._check_initialised = new_config.check_initialised
        self._check_alive = new_config.check_alive

    def stop(self) -> None:
        """
        Stop autofocus activity by disabling the autofocus peripheral.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.disable()

    def configure(self, config: Any | None = None) -> bool:
        """
        Configure binding-specific autofocus parameters.

        Parameters
        ----------
        config
            Optional binding-specific configuration object. If None, the
            binding uses its current or default configuration.

        Returns
        -------
        bool
            True when configuration was accepted by the binding.
        """
        self._require_ready(action="configure")
        return self._configure(config=config)

    def initialise_autofocus(
            self,
            config: Any | None = None,
            lock_after_initialise: bool = False,
    ) -> bool:
        """
        Run the binding-specific autofocus setup/calibration sequence.

        Parameters
        ----------
        config
            Optional binding-specific configuration object. If None, the
            binding uses its current or default configuration.
        lock_after_initialise
            If True, lock autofocus after a successful setup sequence.

        Returns
        -------
        bool
            True when setup succeeded according to the binding's acceptance
            checks.
        """
        self._require_ready(action="initialise_autofocus")
        return self._initialise_autofocus(config=config, lock_after_initialise=lock_after_initialise)

    def lock(self) -> None:
        """
        Lock the autofocus peripheral.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="lock")
        self._lock()

    def unlock(self) -> None:
        """
        Unlock the autofocus peripheral.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="unlock")
        self._unlock()

    def disable(self) -> None:
        """
        Disable autofocus activity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="disable")
        self._disable()

    def get_status(self) -> AutoFocusStatusType:
        """
        Return the current autofocus status.

        Parameters
        ----------
        None

        Returns
        -------
        AutoFocusStatusType
            Current autofocus status reported by the binding.
        """
        self._require_ready(action="get_status")
        return self._get_status()

    def is_locked(self) -> bool:
        """
        Return whether autofocus is currently locked or actively locking.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the binding reports a locked autofocus state.
        """
        self._require_ready(action="is_locked")
        return self._is_locked()

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """Initialise binding-specific autofocus resources."""
        raise NotImplementedError

    @abstractmethod
    def _finalise(self, force: bool = False) -> None:
        """Finalise binding-specific autofocus resources."""
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        """Return whether binding-specific autofocus resources are alive."""
        raise NotImplementedError

    @abstractmethod
    def _configure(self, config: Any | None = None) -> bool:
        """Configure binding-specific autofocus settings."""
        raise NotImplementedError

    @abstractmethod
    def _initialise_autofocus(
            self,
            config: Any | None = None,
            lock_after_initialise: bool = False,
    ) -> bool:
        """Run binding-specific autofocus setup."""
        raise NotImplementedError

    @abstractmethod
    def _lock(self) -> None:
        """Lock binding-specific autofocus."""
        raise NotImplementedError

    @abstractmethod
    def _unlock(self) -> None:
        """Unlock binding-specific autofocus."""
        raise NotImplementedError

    @abstractmethod
    def _disable(self) -> None:
        """Disable binding-specific autofocus."""
        raise NotImplementedError

    @abstractmethod
    def _get_status(self) -> AutoFocusStatusType:
        """Return binding-specific autofocus status."""
        raise NotImplementedError

    @abstractmethod
    def _is_locked(self) -> bool:
        """Return whether binding-specific autofocus is locked."""
        raise NotImplementedError


class AutofocusFactory:
    """Factory for creating Autofocus instances from typed configs."""

    @staticmethod
    def create(
            config: AutofocusConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> Autofocus:
        """
        Create an Autofocus peripheral from an AutofocusConfig.

        Parameters
        ----------
        config
            Typed autofocus configuration describing the desired binding and
            shared construction options.
        peripheral_controllers
            One PeripheralController or a list of available controllers. The
            selected binding determines which controller type is required.
        binding_options
            Extra binding-specific constructor options.

        Returns
        -------
        Autofocus
            Autofocus peripheral for the requested binding.
        """
        if not isinstance(config, AutofocusConfig):
            raise TypeError(f"AutofocusFactory.create: expected AutofocusConfig, received {type(config)}.")

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.autofocus import VirtualAutofocus
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="AutofocusFactory.create",
            )
            autofocus = VirtualAutofocus(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or VirtualAutofocus.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            autofocus.config = config.copy()
            return autofocus

        if config.binding == BindingType.ASI_TIGER:
            from evomachine.bindings.asitiger.autofocus import TigerAutofocus
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=TigerPeripheralController,
                action="AutofocusFactory.create",
            )
            autofocus = TigerAutofocus(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or TigerAutofocus.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            autofocus.config = config.copy()
            return autofocus

        raise ValueError(f"AutofocusFactory.create: unsupported autofocus binding {config.binding}.")

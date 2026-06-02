from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from evomachine.bindings.binding_types import BindingType

# TODO(CODEX): overall, the config system is a bit confusing. do the following:
# - Make a peripheralconfig base class that implements the basics
# - Each peripheral constructor should take a min amount of arugments (mainly runtime)
# - Bindings should subclass the base peripheral config class
# - Minimise the overall number of configs, but you may allow for one config for peripheral
# - Keep things simple
# - Don't add binding-specific things to the base config or classes
# - Make sure that all configs can be updated during runtime, and care for re-initialisation when this happens
# - Pull a clean, not not complex stop system across all peripherals and make sure it used correctly; explain behavior in strings


class ReconfigurationPolicy(Enum):
    """How a runtime config update should be applied."""

    NO_REINIT = auto()
    REINITIALISE = auto()
    RECREATE_REQUIRED = auto()


@dataclass
class PeripheralConfig:
    """Common binding-neutral fields shared by peripheral factory configs."""

    binding: BindingType
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BindingType):
            raise TypeError(f"{type(self).__name__}: binding must be BindingType, received {type(self.binding)}.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"{type(self).__name__}: name must be str or None, received {type(self.name)}.")
        if not isinstance(self.check_initialised, bool):
            raise TypeError(
                f"{type(self).__name__}: check_initialised must be bool, received {type(self.check_initialised)}."
            )
        if not isinstance(self.check_alive, bool):
            raise TypeError(f"{type(self).__name__}: check_alive must be bool, received {type(self.check_alive)}.")

    def copy(self):
        """Return a validated shallow copy of this config."""
        return type(self)(**self.__dict__)

    def updated(self, **kwargs: Any):
        """Return a validated copy with selected fields replaced."""
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"{type(self).__name__}.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return type(self)(**values)

    def update_from_mapping(self, updates: dict[str, Any]):
        """Return a validated copy updated from a mapping."""
        if not isinstance(updates, dict):
            raise TypeError(f"{type(self).__name__}.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


def update_dataclass_config(current_config: Any, replacement: Any | None = None, **updates: Any) -> Any:
    """Validate and return a replacement dataclass-style config."""
    if replacement is not None and updates:
        raise ValueError("update_dataclass_config: provide config or updates, not both.")
    if replacement is not None:
        if not isinstance(replacement, type(current_config)):
            raise TypeError(
                f"update_dataclass_config: expected {type(current_config).__name__}, "
                f"received {type(replacement).__name__}."
            )
        try:
            copy = replacement.copy
        except AttributeError:
            return type(current_config)(**replacement.__dict__)
        return copy()
    try:
        updated = current_config.updated
    except AttributeError:
        updated = None
    if updated is not None:
        return updated(**updates)
    values = dict(current_config.__dict__)
    unknown_keys = [key for key in updates if key not in values]
    if unknown_keys:
        raise ValueError(f"update_dataclass_config: unknown fields {unknown_keys}.")
    values.update(updates)
    return type(current_config)(**values)


class Peripheral:
    """Base class shared by all high-level peripheral devices."""

    def __init__(
            self,
            name: str,
            check_initialised: bool = True,
            check_alive: bool = True,
            config: Any | None = None,
    ):
        """
        Initialise shared peripheral lifecycle and configuration state.

        Parameters
        ----------
        name
            Human-readable peripheral name.
        check_initialised
            If True, public hardware methods require successful initialise().
        check_alive
            If True, public hardware methods require a live backing device.
        config
            Optional config object used to create this peripheral.

        Returns
        -------
        None
        """
        self.name: str = name
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self._check_initialised: bool = check_initialised
        self._check_alive: bool = check_alive
        self.config: Any | None = config.copy() if hasattr(config, "copy") else config

    def _require_ready(self, action: str) -> None:
        """
        Raise when a hardware action is blocked by readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self._check_initialised and not self.is_initialised():
            raise RuntimeError(f"{type(self).__name__}.{action}: {self.name} is not initialised.")
        if self._check_alive and not self.is_alive():
            raise RuntimeError(f"{type(self).__name__}.{action}: {self.name} is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the peripheral if needed.

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
        self._before_initialise(force=force)
        initialise_result = self._initialise(force=force)
        self._is_initialised = True if initialise_result is None else bool(initialise_result)
        if self._check_initialised and not self._is_initialised:
            raise RuntimeError(f"{type(self).__name__}.initialise: {self.name} failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            raise RuntimeError(f"{type(self).__name__}.initialise: {self.name} is not alive after initialisation.")
        self._post_initialise(force=force)

    def _before_initialise(self, force: bool = False) -> None:
        """
        Run subclass-specific checks before binding initialisation.

        Parameters
        ----------
        force
            If True, initialise was forced.

        Returns
        -------
        None
        """
        return

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the peripheral and clear lifecycle flags.

        Parameters
        ----------
        force
            If True, force binding-specific cleanup where supported.

        Returns
        -------
        None
        """
        self._finalise(force=force)
        self._is_initialised = False
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Return whether the peripheral reports an active backing device.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the peripheral is alive.
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
            True when the peripheral is marked initialised.
        """
        return self._is_initialised

    def stop(self) -> None:
        """
        Stop peripheral activity without finalising the peripheral.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError

    def update_config(self, config: Any | None = None, **updates: Any) -> None:
        """
        Replace or update the stored peripheral configuration.

        Parameters
        ----------
        config
            Optional replacement config. If None, keyword updates are applied to
            the current config.
        updates
            Field updates applied to the current config.

        Returns
        -------
        None
        """
        current_config = self.config
        if current_config is None:
            if config is None:
                raise RuntimeError(f"{type(self).__name__}.update_config: {self.name} has no stored config.")
            new_config = config.copy() if hasattr(config, "copy") else config
        else:
            new_config = update_dataclass_config(current_config=current_config, replacement=config, **updates)
        if (
                current_config is not None
                and hasattr(current_config, "binding")
                and hasattr(new_config, "binding")
                and new_config.binding != current_config.binding
        ):
            raise RuntimeError(f"{type(self).__name__}.update_config: changing binding requires recreating the peripheral.")
        was_initialised = self.is_initialised()
        reinitialise = current_config is not None and self._config_requires_reinitialise(current_config, new_config)
        if reinitialise and was_initialised:
            self._before_config_reinitialise()
            self.finalise(force=True)
        self.config = new_config.copy() if hasattr(new_config, "copy") else new_config
        self._apply_base_config(config=new_config)
        self._apply_config(config=new_config)
        if reinitialise and was_initialised:
            self.initialise(force=True)
            self._after_config_reinitialise()

    def _apply_base_config(self, config: Any) -> None:
        """
        Apply config fields common to most peripherals.

        Parameters
        ----------
        config
            Config object with optional name/check fields.

        Returns
        -------
        None
        """
        name = getattr(config, "name", None)
        if name:
            self.name = name
        if hasattr(config, "check_initialised"):
            self._check_initialised = config.check_initialised
        if hasattr(config, "check_alive"):
            self._check_alive = config.check_alive

    def _post_initialise(self, force: bool = False) -> None:
        """
        Run subclass-specific work after successful initialisation.

        Parameters
        ----------
        force
            If True, initialise was forced.

        Returns
        -------
        None
        """
        return

    def _config_requires_reinitialise(self, current_config: Any, new_config: Any) -> bool:
        """
        Return whether applying a config requires reinitialisation.

        Parameters
        ----------
        current_config
            Current config object.
        new_config
            Replacement config object.

        Returns
        -------
        bool
            True when update_config should finalise and initialise.
        """
        return False

    def _before_config_reinitialise(self) -> None:
        """
        Run subclass-specific work before config-driven finalisation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop()

    def _after_config_reinitialise(self) -> None:
        """
        Run subclass-specific work after config-driven reinitialisation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def _apply_config(self, config: Any) -> None:
        """
        Apply subclass-specific config fields.

        Parameters
        ----------
        config
            Config object being applied.

        Returns
        -------
        None
        """
        return

    def _initialise(self, force: bool = False) -> bool:
        """
        Run binding-specific initialisation.

        Parameters
        ----------
        force
            If True, force binding-specific initialisation.

        Returns
        -------
        bool
            True when initialisation succeeded.
        """
        raise NotImplementedError

    def _finalise(self, force: bool = False) -> None:
        """
        Run binding-specific finalisation.

        Parameters
        ----------
        force
            If True, force binding-specific finalisation.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _check_is_alive(self) -> bool:
        """
        Return whether the binding-specific backing device is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the backing device is alive.
        """
        raise NotImplementedError


from evomachine.peripherals.peripheralcontrollers import (  # noqa: E402
    PeripheralController,
    PeripheralControllerConfig,
    PeripheralControllerFactory,
    SerialPeripheralController,
    SerialPeripheralControllerConfig,
    SocketPeripheralController,
    SocketPeripheralControllerConfig,
    get_peripheral_controller,
    normalise_peripheral_controllers,
)

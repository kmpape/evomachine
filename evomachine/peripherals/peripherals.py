from __future__ import annotations

from abc import ABC, abstractmethod
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


class Peripheral(ABC):
    """Base interface shared by all high-level peripheral devices."""

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def initialise(self, force: bool = False) -> None:
        """
        Initialise the peripheral.

        Parameters
        ----------
        force
            If True, run initialisation even when already initialised.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def finalise(self, force: bool = False) -> None:
        """
        Finalise the peripheral and release its active lifecycle state.

        Parameters
        ----------
        force
            If True, force binding-specific finalisation where supported.

        Returns
        -------
        None
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

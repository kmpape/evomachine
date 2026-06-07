from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from evomachine.bindings.binding_types import BindingType


@dataclass(kw_only=True)
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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evomachine.types import PeripheralControllerBindingType


class PeripheralController(ABC):

    def __init__(self, name: str):
        self.name: str = name
        self._is_initialised: bool = False
        self._is_alive: bool = False

    def initialise(self, force: bool = False) -> None:
        if self._is_initialised and not force:
            return
        self._is_initialised = self._initialise(force=force)
        self._is_alive = self._check_is_alive()
        if not self._is_initialised:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} failed to initialise.")
        if not self._is_alive:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} is not alive after initialisation.")

    def reinitialise(self, force: bool = False) -> None:
        self.shutdown(force=force)
        self.initialise(force=True)

    def is_alive(self) -> bool:
        self._is_alive = self._check_is_alive()
        return self._is_alive

    def is_initialised(self) -> bool:
        return self._is_initialised

    def stop(self) -> None:
        self._stop()

    def shutdown(self, force: bool = False) -> None:
        self._shutdown(force=force)
        self._is_initialised = False
        self._is_alive = False

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _shutdown(self, force: bool = False) -> None:
        raise NotImplementedError


@dataclass
class PeripheralControllerConfig:
    """Configuration object used by PeripheralControllerFactory."""

    binding: PeripheralControllerBindingType
    name: str | None = None
    port: str | None = None
    use_thread: bool = False
    initialise: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, PeripheralControllerBindingType):
            raise TypeError(
                f"PeripheralControllerConfig: binding must be PeripheralControllerBindingType, "
                f"received {type(self.binding)}."
            )


class PeripheralControllerFactory:
    """Factory for creating PeripheralController instances from typed configs."""

    @staticmethod
    def create(config: PeripheralControllerConfig, **binding_options: Any) -> PeripheralController:
        if not isinstance(config, PeripheralControllerConfig):
            raise TypeError(
                f"PeripheralControllerFactory.create: expected PeripheralControllerConfig, received {type(config)}."
            )

        if config.binding == PeripheralControllerBindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController

            controller = VirtualPeripheralController(
                name=config.name or "Virtual Peripheral Controller",
                **binding_options,
            )
        elif config.binding == PeripheralControllerBindingType.ASI_TIGER:
            if config.port is None:
                raise ValueError("PeripheralControllerFactory.create: port is required for ASI_TIGER controllers.")
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController

            controller = TigerPeripheralController.from_serial_port(
                port=config.port,
                name=config.name or "ASI Tiger Peripheral Controller",
                use_thread=config.use_thread,
                **binding_options,
            )
        elif config.binding == PeripheralControllerBindingType.SYNCBOARD:
            if config.port is None:
                raise ValueError("PeripheralControllerFactory.create: port is required for SYNCBOARD controllers.")
            from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController

            controller = SyncBoardPeripheralController.from_serial_port(
                port=config.port,
                name=config.name or "SyncBoard Peripheral Controller",
                **binding_options,
            )
        else:
            raise ValueError(f"PeripheralControllerFactory.create: unsupported binding {config.binding}.")

        if config.initialise:
            controller.initialise()
        return controller


def normalise_peripheral_controllers(
        peripheral_controllers: PeripheralController | list[PeripheralController] | None,
) -> list[PeripheralController]:
    if peripheral_controllers is None:
        return []
    if isinstance(peripheral_controllers, PeripheralController):
        return [peripheral_controllers]
    if not isinstance(peripheral_controllers, list):
        raise TypeError(
            "normalise_peripheral_controllers: expected PeripheralController, list[PeripheralController], or None."
        )
    if not all(isinstance(controller, PeripheralController) for controller in peripheral_controllers):
        raise TypeError("normalise_peripheral_controllers: all entries must be PeripheralController.")
    return list(peripheral_controllers)


def get_peripheral_controller(
        peripheral_controllers: PeripheralController | list[PeripheralController] | None,
        controller_type: type[PeripheralController],
        action: str,
) -> PeripheralController:
    controllers = normalise_peripheral_controllers(peripheral_controllers=peripheral_controllers)
    for controller in controllers:
        if isinstance(controller, controller_type):
            return controller
    raise ValueError(f"{action}: {controller_type.__name__} is required.")


class Peripheral(ABC):
    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def initialise(self, force: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def finalise(self, force: bool = False) -> None:
        raise NotImplementedError

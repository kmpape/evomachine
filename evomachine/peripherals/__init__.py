"""Compatibility exports for peripheral base classes and helpers."""

from evomachine.peripherals.peripheralcontrollers import (
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
from evomachine.peripherals.peripherals import Peripheral

__all__ = [
    "Peripheral",
    "PeripheralController",
    "PeripheralControllerConfig",
    "PeripheralControllerFactory",
    "SerialPeripheralController",
    "SerialPeripheralControllerConfig",
    "SocketPeripheralController",
    "SocketPeripheralControllerConfig",
    "get_peripheral_controller",
    "normalise_peripheral_controllers",
]

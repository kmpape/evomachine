"""Napari plugin-facing widget exports."""

from __future__ import annotations

from evomachine.gui.docks.controls import EvoMachineControlsDock
from evomachine.gui.docks.controller_status import PeripheralControllerStatusDock
from evomachine.gui.docks.logs import ApplicationLogsDock

__all__ = ["ApplicationLogsDock", "EvoMachineControlsDock", "PeripheralControllerStatusDock"]

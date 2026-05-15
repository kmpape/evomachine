from __future__ import annotations

from typing import Any

from syncboard.syncboardcontroller import SyncBoardController

from evomachine.peripherals import PeripheralController


class SyncBoardPeripheralController(PeripheralController):
    """
    Peripheral controller for a SyncBoard serial controller.

    Device bindings should receive this controller and access the shared
    SyncBoard connection through the syncboard attribute.
    """

    def __init__(
            self,
            syncboard: SyncBoardController,
            name: str = "SyncBoard Peripheral Controller",
            close_on_shutdown: bool = True,
    ):
        self.syncboard: SyncBoardController = syncboard
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @classmethod
    def from_serial_port(
            cls,
            port: str,
            name: str = "SyncBoard Peripheral Controller",
            close_on_shutdown: bool = True,
            **syncboard_options: Any,
    ) -> "SyncBoardPeripheralController":
        syncboard = SyncBoardController.from_serial_port(port=port, **syncboard_options)
        return cls(syncboard=syncboard, name=name, close_on_shutdown=close_on_shutdown)

    def _initialise(self, force: bool = False) -> bool:
        self.syncboard.initialise(force_init=force)
        return self.syncboard.is_initialised()

    def _check_is_alive(self) -> bool:
        connection = getattr(self.syncboard, "connection", None)
        serial_connection = getattr(connection, "connection", None)
        is_open = getattr(serial_connection, "is_open", None)
        if isinstance(is_open, bool):
            return is_open
        return connection is not None

    def _stop(self) -> None:
        self.syncboard.disable_system()

    def _shutdown(self, force: bool = False) -> None:
        if self.syncboard.is_initialised():
            self.syncboard.finalise()
        if not (force or self.close_on_shutdown):
            return
        connection = getattr(self.syncboard, "connection", None)
        disconnect = getattr(connection, "disconnect", None)
        if callable(disconnect):
            disconnect()

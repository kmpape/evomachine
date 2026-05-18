from __future__ import annotations

from typing import Any

from syncboard.syncboardcontroller import SyncBoardController

from evomachine.peripherals import SerialPeripheralController, SerialPeripheralControllerConfig
from evomachine.bindings.binding_types import BindingType


class SyncBoardPeripheralController(SerialPeripheralController):
    """
    Peripheral controller for a SyncBoard serial controller.

    Device bindings should receive this controller and access the shared
    SyncBoard connection through the syncboard attribute.
    """

    DEFAULT_NAME = "SyncBoard Peripheral Controller"
    DEFAULT_HWID = "16C0:0483"

    def __init__(
            self,
            syncboard: SyncBoardController,
            name: str = "",
            close_on_shutdown: bool = True,
    ):
        self.syncboard: SyncBoardController = syncboard
        super().__init__(name=name or self.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    @classmethod
    def default_config(cls) -> SerialPeripheralControllerConfig:
        return SerialPeripheralControllerConfig(
            binding=BindingType.SYNCBOARD,
            name=cls.DEFAULT_NAME,
            hwid=cls.DEFAULT_HWID,
        )

    @classmethod
    def from_serial_port(
            cls,
            port: str,
            name: str = "",
            close_on_shutdown: bool = True,
            **syncboard_options: Any,
    ) -> "SyncBoardPeripheralController":
        syncboard = SyncBoardController.from_serial_port(port=port, **syncboard_options)
        return cls(syncboard=syncboard, name=name or cls.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    def _get_serial_controller(self) -> SyncBoardController:
        return self.syncboard

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

    def _before_disconnect(self, force: bool = False) -> None:
        if self.syncboard.is_initialised():
            self.syncboard.finalise()

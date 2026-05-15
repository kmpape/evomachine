from __future__ import annotations

from typing import Any

from asitiger.tigercontroller import TigerController

from evomachine.peripherals import PeripheralController


class TigerPeripheralController(PeripheralController):
    """
    Peripheral controller for an ASI Tiger serial controller.

    Device bindings should receive this controller and access the shared Tiger
    connection through the tiger attribute.
    """

    def __init__(
            self,
            tiger: TigerController,
            name: str = "ASI Tiger Peripheral Controller",
            close_on_shutdown: bool = True,
    ):
        self.tiger: TigerController = tiger
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @classmethod
    def from_serial_port(
            cls,
            port: str,
            name: str = "ASI Tiger Peripheral Controller",
            use_thread: bool = False,
            close_on_shutdown: bool = True,
            **tiger_options: Any,
    ) -> "TigerPeripheralController":
        if use_thread:
            from asitiger.tigerthread import TigerThread

            tiger = TigerThread(port=port)
        else:
            tiger = TigerController.from_serial_port(port=port, **tiger_options)
        return cls(tiger=tiger, name=name, close_on_shutdown=close_on_shutdown)

    def _initialise(self, force: bool = False) -> bool:
        return self._check_is_alive()

    def _check_is_alive(self) -> bool:
        try:
            self.tiger.status()
            return True
        except Exception:
            return False

    def _stop(self) -> None:
        self.tiger.halt()

    def _shutdown(self, force: bool = False) -> None:
        if not (force or self.close_on_shutdown):
            return
        connection = getattr(self.tiger, "connection", None)
        disconnect = getattr(connection, "disconnect", None)
        if callable(disconnect):
            disconnect()

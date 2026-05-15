from __future__ import annotations

from typing import Any

from asitiger.tigercontroller import TigerController

from evomachine.peripherals import SerialPeripheralController, SerialPeripheralControllerConfig
from evomachine.types import PeripheralControllerBindingType


class TigerPeripheralController(SerialPeripheralController):
    """
    Peripheral controller for an ASI Tiger serial controller.

    Device bindings should receive this controller and access the shared Tiger
    connection through the tiger attribute.
    """

    DEFAULT_NAME = "ASI Tiger Peripheral Controller"
    DEFAULT_HWID = "10C4:EA60"

    def __init__(
            self,
            tiger: TigerController,
            name: str = "",
            close_on_shutdown: bool = True,
    ):
        self.tiger: TigerController = tiger
        super().__init__(name=name or self.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    @classmethod
    def default_config(cls) -> SerialPeripheralControllerConfig:
        return SerialPeripheralControllerConfig(
            binding=PeripheralControllerBindingType.ASI_TIGER,
            name=cls.DEFAULT_NAME,
            hwid=cls.DEFAULT_HWID,
        )

    @classmethod
    def from_serial_port(
            cls,
            port: str,
            name: str = "",
            use_thread: bool = False,
            close_on_shutdown: bool = True,
            **tiger_options: Any,
    ) -> "TigerPeripheralController":
        if use_thread:
            from asitiger.tigerthread import TigerThread

            tiger = TigerThread(port=port)
        else:
            tiger = TigerController.from_serial_port(port=port, **tiger_options)
        return cls(tiger=tiger, name=name or cls.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    def _get_serial_controller(self) -> TigerController:
        return self.tiger

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

from __future__ import annotations

from typing import Any

from evomachine.bindings.kwr103.KWR103Driver import KWR103
from evomachine.peripherals import SerialPeripheralController, SerialPeripheralControllerConfig
from evomachine.bindings.binding_types import BindingType


class KWR103PeripheralController(SerialPeripheralController):
    """
    Peripheral controller for a KWR103 serial power supply.
    """

    DEFAULT_NAME = "KWR103 Peripheral Controller"
    DEFAULT_HWID = "16C0:0483"

    def __init__(
            self,
            kwr103: KWR103,
            name: str = "",
            close_on_shutdown: bool = True,
    ):
        self.kwr103: KWR103 = kwr103
        super().__init__(name=name or self.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    @classmethod
    def default_config(cls) -> SerialPeripheralControllerConfig:
        return SerialPeripheralControllerConfig(
            binding=BindingType.KWR103,
            name=cls.DEFAULT_NAME,
            hwid=cls.DEFAULT_HWID,
        )

    @classmethod
    def from_serial_port(
            cls,
            port: str,
            name: str = "",
            close_on_shutdown: bool = True,
            **kwr103_options: Any,
    ) -> "KWR103PeripheralController":
        kwr103 = KWR103(port=port, **kwr103_options)
        return cls(kwr103=kwr103, name=name or cls.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    def _get_serial_controller(self) -> KWR103:
        return self.kwr103

    def _get_connection(self) -> KWR103:
        return self.kwr103

    def _initialise(self, force: bool = False) -> bool:
        if not self.kwr103.is_connected():
            self.kwr103.connect()
        return self.kwr103.is_connected()

    def _check_is_alive(self) -> bool:
        return self.kwr103.is_connected()

    def _stop(self) -> None:
        self.kwr103.set_output(False)

    def _before_disconnect(self, force: bool = False) -> None:
        if self.kwr103.is_connected():
            self.kwr103.set_output(False)

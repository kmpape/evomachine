from __future__ import annotations

from typing import Any

from asitiger.tigercontroller import TigerController

from evomachine.peripherals.peripherals import SerialPeripheralController, SerialPeripheralControllerConfig
from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.asitiger.card_addresses import (
    CARD_ADDRESS_CRISP,
    CARD_ADDRESS_FILTER_WHEEL,
    CARD_ADDRESS_LED,
)


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
            card_address_crisp: int = CARD_ADDRESS_CRISP,
            card_address_led: int = CARD_ADDRESS_LED,
            card_address_filter_wheel: int = CARD_ADDRESS_FILTER_WHEEL,
    ):
        self.tiger: TigerController = tiger
        self.card_address_crisp: int = self._validate_card_address(
            card_address=card_address_crisp,
            name="card_address_crisp",
        )
        self.card_address_led: int = self._validate_card_address(
            card_address=card_address_led,
            name="card_address_led",
        )
        self.card_address_filter_wheel: int = self._validate_card_address(
            card_address=card_address_filter_wheel,
            name="card_address_filter_wheel",
        )
        super().__init__(name=name or self.DEFAULT_NAME, close_on_shutdown=close_on_shutdown)

    @staticmethod
    def _validate_card_address(card_address: int, name: str) -> int:
        """
        Validate one ASI Tiger card address.

        Parameters
        ----------
        card_address
            Candidate card address.
        name
            Field name used in exception messages.

        Returns
        -------
        int
            Validated card address.
        """
        if not isinstance(card_address, int) or isinstance(card_address, bool):
            raise TypeError(f"TigerPeripheralController: {name} must be int, received {type(card_address)}.")
        if card_address < 0:
            raise ValueError(f"TigerPeripheralController: {name} must be non-negative, received {card_address}.")
        return card_address

    @classmethod
    def default_config(cls) -> SerialPeripheralControllerConfig:
        return SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
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
            card_address_crisp: int = CARD_ADDRESS_CRISP,
            card_address_led: int = CARD_ADDRESS_LED,
            card_address_filter_wheel: int = CARD_ADDRESS_FILTER_WHEEL,
            **tiger_options: Any,
    ) -> "TigerPeripheralController":
        if use_thread:
            from asitiger.tigerthread import TigerThread

            tiger = TigerThread(port=port)
        else:
            tiger = TigerController.from_serial_port(port=port, **tiger_options)
        return cls(
            tiger=tiger,
            name=name or cls.DEFAULT_NAME,
            close_on_shutdown=close_on_shutdown,
            card_address_crisp=card_address_crisp,
            card_address_led=card_address_led,
            card_address_filter_wheel=card_address_filter_wheel,
        )

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

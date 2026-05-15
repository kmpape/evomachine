from __future__ import annotations

from typing import Any

from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.leds import LedSource
from evomachine.types import BrightnessType, LEDType


class TigerLedSource(LedSource):
    """LED source controlled through ASI Tiger LED channels."""

    DEFAULT_LED_TO_INTERNAL = {
        LEDType.LED_OVERHEAD_TIGER: "Y",
        LEDType.TIGER_LED_1: "X",
        LEDType.TIGER_LED_2: "Y",
        LEDType.TIGER_LED_3: "Z",
        LEDType.TIGER_LED_4: "F",
    }

    def __init__(
            self,
            peripheral_ctrl: TigerPeripheralController,
            available_leds: list[LEDType],
            led_to_internal: dict[LEDType, Any] | None = None,
            name: str = "ASI Tiger LED Source",
            check_initialised: bool = True,
            check_alive: bool = True,
            card_address: int = 7,
    ):
        self.card_address: int = card_address
        led_to_internal = led_to_internal or {
            led_type: self.DEFAULT_LED_TO_INTERNAL[led_type] for led_type in available_leds
        }
        super().__init__(
            peripheral_ctrl=peripheral_ctrl,
            available_leds=available_leds,
            led_to_internal=led_to_internal,
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @property
    def peripheral_ctrl(self) -> TigerPeripheralController:
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: TigerPeripheralController) -> None:
        self._peripheral_ctrl = peripheral_ctrl

    def _set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None = None,
    ) -> None:
        brightness_int = int(brightness)
        led_brightnesses = {
            internal: brightness_int if current_led_type == led_type else 0
            for current_led_type, internal in self.led_to_internal.items()
        }
        self.peripheral_ctrl.tiger.led(led_brightnesses=led_brightnesses, card_address=self.card_address)

    def _disable_led(self, led_type: LEDType) -> None:
        led_brightnesses = {internal: 0 for internal in self.led_to_internal.values()}
        self.peripheral_ctrl.tiger.led(led_brightnesses=led_brightnesses, card_address=self.card_address)

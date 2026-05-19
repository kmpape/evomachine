from __future__ import annotations

from typing import Any

from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.peripherals.leds import LedSource
from evomachine.types import BrightnessType, LEDType


class VirtualLedSource(LedSource):
    """In-memory LED source for tests and dry runs."""

    DEFAULT_LED_TO_INTERNAL = {
        LEDType.LED_385_NM: LEDType.LED_385_NM,
        LEDType.LED_450_NM: LEDType.LED_450_NM,
        LEDType.LED_515_NM: LEDType.LED_515_NM,
        LEDType.LED_565_NM: LEDType.LED_565_NM,
        LEDType.LED_645_NM: LEDType.LED_645_NM,
        LEDType.LED_OVERHEAD: LEDType.LED_OVERHEAD,
        LEDType.LED_OVERHEAD_TIGER: LEDType.LED_OVERHEAD_TIGER,
        LEDType.TIGER_LED_1: LEDType.TIGER_LED_1,
        LEDType.TIGER_LED_2: LEDType.TIGER_LED_2,
        LEDType.TIGER_LED_3: LEDType.TIGER_LED_3,
        LEDType.TIGER_LED_4: LEDType.TIGER_LED_4,
    }

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            available_leds: list[LEDType],
            led_to_internal: dict[LEDType, Any] | None = None,
            name: str = "Virtual LED Source",
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        self.commands: list[tuple[LEDType, BrightnessType]] = []
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

    def _set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None = None,
    ) -> None:
        self.commands.append((led_type, brightness))

    def _disable_led(self, led_type: LEDType) -> None:
        self.commands.append((led_type, 0.0))

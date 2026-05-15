from __future__ import annotations

from typing import Any

from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.leds import LedSource
from evomachine.types import BrightnessType, LEDType


class SyncBoardLedSource(LedSource):
    """LED source controlled through a SyncBoard."""

    DEFAULT_LED_TO_INTERNAL = {
        LEDType.LED_385_NM: 7,
        LEDType.LED_450_NM: 1,
        LEDType.LED_515_NM: 2,
        LEDType.LED_565_NM: 3,
        LEDType.LED_645_NM: 4,
    }

    def __init__(
            self,
            peripheral_ctrl: SyncBoardPeripheralController,
            available_leds: list[LEDType],
            led_to_internal: dict[LEDType, Any] | None = None,
            name: str = "SyncBoard LED Source",
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
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
    def peripheral_ctrl(self) -> SyncBoardPeripheralController:
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: SyncBoardPeripheralController) -> None:
        self._peripheral_ctrl = peripheral_ctrl

    def _set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None = None,
    ) -> None:
        led_id = self.led_to_internal[led_type]
        if brightness == 0:
            self.peripheral_ctrl.syncboard.disable_led(led_id=led_id)
            return
        self.peripheral_ctrl.syncboard.enable_led(
            led_id=led_id,
            intensity=float(brightness) / 100.0,
            duration=duration,
        )

    def _disable_led(self, led_type: LEDType) -> None:
        self.peripheral_ctrl.syncboard.disable_led(led_id=self.led_to_internal[led_type])

    def _start_timer(self, led_type: LEDType, duration: float | None) -> None:
        return

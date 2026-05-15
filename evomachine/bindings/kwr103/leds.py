from __future__ import annotations

from typing import Any

from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController
from evomachine.leds import LedSource
from evomachine.types import BrightnessType, LEDType


class KWR103LedSource(LedSource):
    """LED source controlled through a KWR103 power supply."""

    DEFAULT_LED_TO_INTERNAL = {LEDType.LED_OVERHEAD: "output"}

    def __init__(
            self,
            peripheral_ctrl: KWR103PeripheralController,
            available_leds: list[LEDType],
            led_to_internal: dict[LEDType, Any] | None = None,
            name: str = "KWR103 LED Source",
            check_initialised: bool = True,
            check_alive: bool = True,
            min_voltage: float = 7.0,
            max_voltage: float = 9.0,
            current: float = 0.1,
    ):
        self.min_voltage: float = min_voltage
        self.max_voltage: float = max_voltage
        self.current: float = current
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
    def peripheral_ctrl(self) -> KWR103PeripheralController:
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: KWR103PeripheralController) -> None:
        self._peripheral_ctrl = peripheral_ctrl

    def _initialise(self, force: bool = False) -> None:
        self.peripheral_ctrl.kwr103.set_output(False)
        self.peripheral_ctrl.kwr103.set_current(self.current)

    def _set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None = None,
    ) -> None:
        if brightness == 0:
            self.peripheral_ctrl.kwr103.set_output(False)
            return
        voltage = self.min_voltage + (self.max_voltage - self.min_voltage) * float(brightness) / 100.0
        self.peripheral_ctrl.kwr103.set_voltage(min(self.max_voltage, voltage))
        self.peripheral_ctrl.kwr103.set_output(True)

    def _disable_led(self, led_type: LEDType) -> None:
        self.peripheral_ctrl.kwr103.set_output(False)

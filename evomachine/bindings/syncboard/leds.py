from __future__ import annotations

from typing import Any

from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.peripherals.leds import LedSource
from evomachine.types import BrightnessType, LEDType
from syncboard.syncboardcontroller import LED_ID

SYNCBOARD_TIMED_BRIGHTNESS_THRESHOLD = 29.0
SYNCBOARD_DEFAULT_TIMED_DURATION_MS = 3000.0


class FakeSyncBoardController:
    """Deterministic SyncBoard-like controller for LED and peripheral tests."""

    def __init__(self):
        """
        Initialise fake SyncBoard state and command recording.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.enabled_leds: list[tuple[int, float, float | None]] = []
        self.disabled_leds: list[int | None] = []
        self.finalise_was_called = False
        self._is_initialised = False
        self.connection = type(
            "FakeConnection",
            (),
            {
                "connection": type("FakeSerialConnection", (), {"is_open": True})(),
                "disconnect": lambda connection: setattr(connection.connection, "is_open", False),
            },
        )()

    def initialise(self, force_init: bool = False) -> None:
        """Mark the fake SyncBoard as initialised."""
        self._is_initialised = True

    def is_initialised(self) -> bool:
        """Return whether the fake SyncBoard is initialised."""
        return self._is_initialised

    def disable_system(self) -> None:
        """Accept a fake disable-system command."""
        return

    def finalise(self) -> None:
        """Record fake finalisation and clear initialisation state."""
        self.finalise_was_called = True
        self._is_initialised = False

    def enable_led(self, led_id: int, intensity: float = 0.1, duration: float | None = None) -> None:
        """Record a fake native SyncBoard enable command."""
        self.enabled_leds.append((led_id, intensity, duration))

    def disable_led(self, led_id: int | None = None) -> None:
        """Record a fake SyncBoard disable command."""
        self.disabled_leds.append(led_id)


class SyncBoardLedSource(LedSource):
    """LED source controlled through a SyncBoard."""

    DEFAULT_LED_TO_INTERNAL = {
        LEDType.LED_385_NM: LED_ID.LED_385_NM,
        LEDType.LED_450_NM: LED_ID.LED_450_NM,
        LEDType.LED_515_NM: LED_ID.LED_515_NM,
        LEDType.LED_565_NM: LED_ID.LED_565_NM,
        LEDType.LED_645_NM: LED_ID.LED_645_NM,
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

    def _normalise_duration(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None,
    ) -> float | None:
        if duration is None and float(brightness) > SYNCBOARD_TIMED_BRIGHTNESS_THRESHOLD:
            return SYNCBOARD_DEFAULT_TIMED_DURATION_MS
        return duration

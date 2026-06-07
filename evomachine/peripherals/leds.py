from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import threading
import time
from typing import Any

from evomachine.bindings.binding_types import BindingType
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig
from evomachine.types import BrightnessType, LEDType


@dataclass
class LedState:
    """Cached state for one logical LED."""

    led_type: LEDType
    brightness: BrightnessType = 0.0
    is_on: bool = False
    stop_time: float | None = None


@dataclass(kw_only=True)
class LedConfig(PeripheralConfig):
    """Configuration object used by LedFactory to create LED sources."""

    available_leds: list[LEDType]
    led_to_internal: dict[LEDType, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate LED source configuration after dataclass construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place. available_leds is
            normalised to a copied list.
        """
        super().__post_init__()
        self.available_leds = LedSource.validate_available_leds(self.available_leds)
        if self.led_to_internal is not None:
            if not isinstance(self.led_to_internal, dict):
                raise TypeError("LedConfig: led_to_internal must be dict[LEDType, Any].")
            if not all(isinstance(led_type, LEDType) for led_type in self.led_to_internal):
                raise TypeError("LedConfig: led_to_internal keys must be LEDType.")
            LedSource._validate_led_to_internal(
                led_to_internal=self.led_to_internal,
                available_leds=self.available_leds,
            )


class LedSource(Peripheral):
    """Base class for LEDs controlled by one peripheral controller."""

    def __init__(
            self,
            peripheral_ctrl: PeripheralController,
            available_leds: list[LEDType],
            led_to_internal: dict[LEDType, Any],
            name: str,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared state for one LED source and its controller.

        Parameters
        ----------
        peripheral_ctrl
            PeripheralController that owns the hardware connection used by this
            LED source.
        available_leds
            Non-empty list of logical LEDType values this source can actuate.
        led_to_internal
            Mapping from each available LEDType to the binding-specific channel
            identifier used by the hardware API.
        name
            Human-readable source name used in error messages.
        check_initialised
            If True, hardware actions require this source to be initialised.
        check_alive
            If True, hardware actions require the controller to report alive.

        Returns
        -------
        None
        """
        self.peripheral_ctrl: PeripheralController = peripheral_ctrl
        self.name: str = name
        self.available_leds: list[LEDType] = self.validate_available_leds(available_leds)
        self.led_to_internal: dict[LEDType, Any] = self._validate_led_to_internal(
            led_to_internal=led_to_internal,
            available_leds=self.available_leds,
        )
        self.check_initialised: bool = check_initialised
        self.check_alive: bool = check_alive
        self._is_initialised: bool = False
        self._states: dict[LEDType, LedState] = {
            led_type: LedState(led_type=led_type) for led_type in self.available_leds
        }
        self._timers: dict[LEDType, threading.Timer] = {}
        self.config: LedConfig | None = None

    @staticmethod
    def validate_available_leds(available_leds: list[LEDType]) -> list[LEDType]:
        """
        Return a validated copy of the configured logical LEDs.

        Parameters
        ----------
        available_leds
            Candidate list of logical LEDType values. It must be non-empty and
            cannot contain LEDType.NO_LED because NO_LED is a manager command.

        Returns
        -------
        list[LEDType]
            Copy of the validated LED list.
        """
        if not isinstance(available_leds, list):
            raise TypeError("LedSource: available_leds must be list[LEDType].")
        if not available_leds:
            raise ValueError("LedSource: available_leds must not be empty.")
        if not all(isinstance(led_type, LEDType) for led_type in available_leds):
            raise TypeError("LedSource: all available_leds entries must be LEDType.")
        if LEDType.NO_LED in available_leds:
            raise ValueError("LedSource: LEDType.NO_LED is a manager command, not a source LED.")
        return list(available_leds)

    @staticmethod
    def _validate_led_to_internal(
            led_to_internal: dict[LEDType, Any],
            available_leds: list[LEDType],
    ) -> dict[LEDType, Any]:
        """
        Validate and trim the source-specific mapping for configured LEDs.

        Parameters
        ----------
        led_to_internal
            Mapping from logical LEDType values to binding-specific hardware
            channel identifiers.
        available_leds
            LEDType values that must be present in led_to_internal.

        Returns
        -------
        dict[LEDType, Any]
            Mapping containing only entries for available_leds, preserving the
            original binding-specific channel values.
        """
        if not isinstance(led_to_internal, dict):
            raise TypeError("LedSource: led_to_internal must be dict[LEDType, Any].")
        missing_leds = [led_type for led_type in available_leds if led_type not in led_to_internal]
        if missing_leds:
            raise ValueError(f"LedSource: led_to_internal missing mappings for {missing_leds}.")
        return {led_type: led_to_internal[led_type] for led_type in available_leds}

    def initialise(self, force: bool = False) -> None:
        """
        Initialise this LED source after checking its peripheral controller.

        Parameters
        ----------
        force
            If True, run initialisation even when this source is already marked
            initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            return
        if self.check_initialised and not self.peripheral_ctrl.is_initialised():
            raise RuntimeError(f"LedSource.initialise: {self.peripheral_ctrl.name} is not initialised.")
        if self.check_alive and not self.peripheral_ctrl.is_alive():
            raise RuntimeError(f"LedSource.initialise: {self.peripheral_ctrl.name} is not alive.")
        self._initialise(force=force)
        self._is_initialised = True

    def is_initialised(self) -> bool:
        """
        Return whether this LED source has been initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True after initialise() succeeds, False after construction or
            finalise().
        """
        return self._is_initialised

    def is_alive(self) -> bool:
        """
        Query and return the liveness reported by the peripheral controller.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the underlying controller reports alive.
        """
        return self.peripheral_ctrl.is_alive()

    def stop(self) -> None:
        """
        Disable all LEDs controlled by this source.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.disable_led()

    def finalise(self, force: bool = False) -> None:
        """
        Cancel timers, disable LEDs when allowed, and clear initialisation state.

        Parameters
        ----------
        force
            Reserved for API compatibility with Peripheral. It does not change
            LED finalisation behavior in the base implementation.

        Returns
        -------
        None
        """
        self._cancel_all_timers()
        if self._is_initialised or not self.check_initialised:
            self.disable_led()
        self._is_initialised = False

    def get_available_leds(self) -> list[LEDType]:
        """
        Return the logical LEDs configured for this source.

        Parameters
        ----------
        None

        Returns
        -------
        list[LEDType]
            Copy of the logical LED list controlled by this source.
        """
        return list(self.available_leds)

    def set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType = 100.0,
            duration: float | None = None,
    ) -> None:
        """
        Set one logical LED to a brightness, optionally for a timed duration.

        Parameters
        ----------
        led_type
            Logical LED to actuate. It must be available on this source.
        brightness
            Numeric BrightnessType value in the inclusive range [0, 100].
        duration
            Optional duration in milliseconds. The base source uses a local
            timer that calls disable_led() when the duration expires; bindings
            may override timer behavior when hardware provides native timing.

        Returns
        -------
        None
        """
        self._check_ready()
        if not isinstance(led_type, LEDType):
            raise TypeError(f"LedSource.set_led: led_type must be LEDType, received {type(led_type)}.")
        if led_type not in self.available_leds:
            raise ValueError(f"LedSource.set_led: {led_type} is not available for {self.name}.")
        brightness = self._validate_brightness(brightness=brightness)
        if duration is not None and duration < 0:
            raise ValueError(f"LedSource: duration must be non-negative, received {duration}.")

        self._cancel_timer(led_type=led_type)
        self._set_led(led_type=led_type, brightness=brightness, duration=duration)
        self._update_state(led_type=led_type, brightness=brightness, duration=duration)
        self._start_timer(led_type=led_type, duration=duration)

    def disable_led(self, led_type: LEDType | None = None) -> None:
        """
        Disable one configured LED, or every LED on this source when omitted.

        Parameters
        ----------
        led_type
            LEDType to disable. If None, all LEDs configured on this source are
            disabled.

        Returns
        -------
        None
        """
        self._check_ready()
        led_types = self.available_leds if led_type is None else [led_type]
        for selected_led_type in led_types:
            if selected_led_type not in self.available_leds:
                raise ValueError(f"LedSource.disable_led: {selected_led_type} is not available for {self.name}.")
            self._cancel_timer(led_type=selected_led_type)
            self._disable_led(led_type=selected_led_type)
            self._states[selected_led_type] = LedState(led_type=selected_led_type)

    def get_led_state(self, led_type: LEDType) -> LedState:
        """
        Return a copy of the cached state for one configured LED.

        Parameters
        ----------
        led_type
            LEDType whose cached state should be returned.

        Returns
        -------
        LedState
            Copy of the cached brightness, on/off state, and optional stop time.
        """
        if led_type not in self.available_leds:
            raise ValueError(f"LedSource.get_led_state: {led_type} is not available for {self.name}.")
        state = self._states[led_type]
        return LedState(
            led_type=state.led_type,
            brightness=state.brightness,
            is_on=state.is_on,
            stop_time=state.stop_time,
        )

    def led_is_on(self, led_type: LEDType) -> bool:
        """
        Return whether the cached state marks one LED as on.

        Parameters
        ----------
        led_type
            LEDType to inspect in the cached state map.

        Returns
        -------
        bool
            True when the cached state has is_on=True.
        """
        return self.get_led_state(led_type=led_type).is_on

    def _check_ready(self) -> None:
        """
        Raise if readiness checks block a LED hardware operation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.check_initialised and not self.is_initialised():
            raise RuntimeError(f"LedSource: {self.name} is not initialised.")
        if self.check_alive and not self.is_alive():
            raise RuntimeError(f"LedSource: {self.name} is not alive.")

    @staticmethod
    def _validate_brightness(brightness: BrightnessType) -> BrightnessType:
        """
        Validate and normalise brightness to a float in [0, 100].

        Parameters
        ----------
        brightness
            Candidate BrightnessType value. Numeric int and float values are
            accepted.

        Returns
        -------
        BrightnessType
            Brightness converted to float after validation.
        """
        if not isinstance(brightness, int | float):
            raise TypeError(f"LedSource: brightness must be numeric, received {type(brightness)}.")
        brightness = float(brightness)
        if not 0 <= brightness <= 100:
            raise ValueError(f"LedSource: brightness must be in [0, 100], received {brightness}.")
        return brightness

    def _update_state(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None,
    ) -> None:
        """
        Update the cached LED state after a successful set command.

        Parameters
        ----------
        led_type
            LEDType whose cached state should be updated.
        brightness
            Validated brightness value in [0, 100].
        duration
            Optional duration in milliseconds used to compute stop_time.

        Returns
        -------
        None
        """
        stop_time = None if duration is None else time.time() + duration / 1000.0
        self._states[led_type] = LedState(
            led_type=led_type,
            brightness=brightness,
            is_on=brightness > 0,
            stop_time=stop_time,
        )

    def _start_timer(self, led_type: LEDType, duration: float | None) -> None:
        """
        Start a local duration timer for one LED when a duration is provided.

        Parameters
        ----------
        led_type
            LEDType to disable when the timer expires.
        duration
            Duration in milliseconds. None skips timer creation.

        Returns
        -------
        None
        """
        if duration is None:
            return
        if duration < 0:
            raise ValueError(f"LedSource: duration must be non-negative, received {duration}.")
        timer = threading.Timer(duration / 1000.0, self.disable_led, kwargs={"led_type": led_type})
        timer.daemon = True
        self._timers[led_type] = timer
        timer.start()

    def _cancel_timer(self, led_type: LEDType) -> None:
        """
        Cancel and forget the local timer for one LED if it exists.

        Parameters
        ----------
        led_type
            LEDType whose timer should be cancelled.

        Returns
        -------
        None
        """
        timer = self._timers.pop(led_type, None)
        if timer is not None:
            timer.cancel()

    def _cancel_all_timers(self) -> None:
        """
        Cancel every local LED duration timer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for led_type in list(self._timers):
            self._cancel_timer(led_type=led_type)

    def _initialise(self, force: bool = False) -> None:
        """
        Run binding-specific LED source initialisation.

        Parameters
        ----------
        force
            If True, subclasses may force their hardware initialisation.

        Returns
        -------
        None
        """
        return

    @abstractmethod
    def _set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType,
            duration: float | None = None,
    ) -> None:
        """
        Set one LED through the binding-specific hardware API.

        Parameters
        ----------
        led_type
            LEDType to actuate.
        brightness
            Validated brightness value in [0, 100].
        duration
            Optional duration in milliseconds. Bindings may use this directly or
            ignore it when timing is handled by the base class.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _disable_led(self, led_type: LEDType) -> None:
        """
        Disable one LED through the binding-specific hardware API.

        Parameters
        ----------
        led_type
            LEDType to disable.

        Returns
        -------
        None
        """
        raise NotImplementedError


class LedManager(Peripheral):
    """Route logical LED commands to multiple LedSource instances."""

    def __init__(self, led_sources: list[LedSource], name: str = "LED Manager"):
        """
        Build a manager and validate that each logical LED has one source.

        Parameters
        ----------
        led_sources
            Non-empty list of LedSource instances to route through the manager.
            A logical LED may appear in only one source.
        name
            Human-readable manager name.

        Returns
        -------
        None
        """
        self.name: str = name
        self.led_sources: list[LedSource] = list(led_sources)
        if not self.led_sources:
            raise ValueError("LedManager: led_sources must not be empty.")
        self._led_to_source: dict[LEDType, LedSource] = {}
        for source in self.led_sources:
            for led_type in source.get_available_leds():
                if led_type in self._led_to_source:
                    raise ValueError(f"LedManager: {led_type} is handled by more than one LedSource.")
                self._led_to_source[led_type] = source

    def initialise(self, force: bool = False) -> None:
        """
        Initialise every managed LED source.

        Parameters
        ----------
        force
            If True, pass force=True to each source initialisation call.

        Returns
        -------
        None
        """
        for source in self.led_sources:
            source.initialise(force=force)

    def is_initialised(self) -> bool:
        """
        Return True when every managed source is initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if all managed sources report initialised.
        """
        return all(source.is_initialised() for source in self.led_sources)

    def is_alive(self) -> bool:
        """
        Query every managed source and return True when all are alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if every managed source reports alive.
        """
        source_states = [source.is_alive() for source in self.led_sources]
        return all(source_states)

    def stop(self) -> None:
        """
        Disable all LEDs on all managed sources.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.disable_led()

    def finalise(self, force: bool = False) -> None:
        """
        Finalise every managed LED source.

        Parameters
        ----------
        force
            Passed through to each managed source finalise() call.

        Returns
        -------
        None
        """
        for source in self.led_sources:
            source.finalise(force=force)

    def get_available_leds(self) -> list[LEDType]:
        """
        Return every logical LED routed by this manager.

        Parameters
        ----------
        None

        Returns
        -------
        list[LEDType]
            Logical LEDs routed by this manager.
        """
        return list(self._led_to_source)

    def set_led(
            self,
            led_type: LEDType,
            brightness: BrightnessType = 100.0,
            duration: float | None = None,
    ) -> None:
        """
        Route a LED set command to the source that owns the logical LED.

        Parameters
        ----------
        led_type
            LEDType to set. LEDType.NO_LED is treated as a command to disable
            all LEDs.
        brightness
            BrightnessType value in [0, 100] passed to the selected source.
        duration
            Optional duration in milliseconds passed to the selected source.

        Returns
        -------
        None
        """
        if led_type == LEDType.NO_LED:
            self.disable_led()
            return
        source = self._get_source(led_type=led_type)
        source.set_led(led_type=led_type, brightness=brightness, duration=duration)

    def disable_led(self, led_type: LEDType | None = None) -> None:
        """
        Disable one routed LED, or all routed LEDs when omitted or NO_LED.

        Parameters
        ----------
        led_type
            LEDType to disable. None or LEDType.NO_LED disables all managed
            sources.

        Returns
        -------
        None
        """
        if led_type is None or led_type == LEDType.NO_LED:
            for source in self.led_sources:
                source.disable_led()
            return
        self._get_source(led_type=led_type).disable_led(led_type=led_type)

    def get_led_state(self, led_type: LEDType) -> LedState:
        """
        Return the cached state for one routed LED.

        Parameters
        ----------
        led_type
            LEDType whose state should be returned.

        Returns
        -------
        LedState
            Copy of the selected source's cached LED state.
        """
        return self._get_source(led_type=led_type).get_led_state(led_type=led_type)

    def led_is_on(self, led_type: LEDType) -> bool:
        """
        Return whether one routed LED is currently cached as on.

        Parameters
        ----------
        led_type
            LEDType to inspect.

        Returns
        -------
        bool
            True when the routed source caches the LED as on.
        """
        return self.get_led_state(led_type=led_type).is_on

    def _get_source(self, led_type: LEDType) -> LedSource:
        """
        Return the source that owns one logical LED.

        Parameters
        ----------
        led_type
            LEDType to resolve to a LedSource.

        Returns
        -------
        LedSource
            Managed source responsible for led_type.
        """
        if led_type not in self._led_to_source:
            raise ValueError(f"LedManager: {led_type} is not available.")
        return self._led_to_source[led_type]


class LedFactory:
    """Factory for creating LedSource instances from typed configs."""

    @staticmethod
    def create(
            config: LedConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> LedSource:
        """
        Create the binding-specific LED source described by config.

        Parameters
        ----------
        config
            LedConfig selecting the binding, source name, available LEDs,
            readiness checks, and optional logical-to-internal mapping.
        peripheral_controllers
            One PeripheralController, a list of controllers, or None. The
            factory selects the controller required by config.binding.
        **binding_options
            Extra keyword arguments forwarded to the binding-specific LedSource
            constructor.

        Returns
        -------
        LedSource
            Binding-specific LED source instance.
        """
        if not isinstance(config, LedConfig):
            raise TypeError(f"LedFactory.create: expected LedConfig, received {type(config)}.")

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.leds import VirtualLedSource
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="LedFactory.create",
            )
            source = VirtualLedSource(
                peripheral_ctrl=peripheral_ctrl,
                available_leds=config.available_leds,
                led_to_internal=config.led_to_internal,
                name=config.name or "Virtual LED Source",
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            source.config = config.copy()
            return source

        if config.binding == BindingType.ASI_TIGER:
            from evomachine.bindings.asitiger.leds import TigerLedSource
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=TigerPeripheralController,
                action="LedFactory.create",
            )
            source = TigerLedSource(
                peripheral_ctrl=peripheral_ctrl,
                available_leds=config.available_leds,
                led_to_internal=config.led_to_internal,
                name=config.name or "ASI Tiger LED Source",
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            source.config = config.copy()
            return source

        if config.binding == BindingType.SYNCBOARD:
            from evomachine.bindings.syncboard.leds import SyncBoardLedSource
            from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=SyncBoardPeripheralController,
                action="LedFactory.create",
            )
            source = SyncBoardLedSource(
                peripheral_ctrl=peripheral_ctrl,
                available_leds=config.available_leds,
                led_to_internal=config.led_to_internal,
                name=config.name or "SyncBoard LED Source",
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            source.config = config.copy()
            return source

        if config.binding == BindingType.KWR103:
            from evomachine.bindings.kwr103.leds import KWR103LedSource
            from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=KWR103PeripheralController,
                action="LedFactory.create",
            )
            source = KWR103LedSource(
                peripheral_ctrl=peripheral_ctrl,
                available_leds=config.available_leds,
                led_to_internal=config.led_to_internal,
                name=config.name or "KWR103 LED Source",
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **binding_options,
            )
            source.config = config.copy()
            return source

        raise ValueError(f"LedFactory.create: unsupported LED binding {config.binding}.")

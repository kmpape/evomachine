from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from asitiger.command import CRISPState

from evomachine.peripherals.autofocus import Autofocus
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.types import AutoFocusStatusType


@dataclass
class TigerAutofocusConfig:
    """ASI Tiger CRISP autofocus configuration."""

    averaging: int
    led_intensity: int
    lock_range: float
    loop_gain: int
    update_rate: int
    objective_na: float
    min_snr: int | float = 2
    min_error: int | float = 100

    def __post_init__(self) -> None:
        """
        Validate ASI Tiger CRISP configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.averaging, int) or isinstance(self.averaging, bool) or not 0 <= self.averaging < 100:
            raise TypeError(f"TigerAutofocusConfig: averaging must be int in [0, 100), received {self.averaging}.")
        if (
                not isinstance(self.led_intensity, int)
                or isinstance(self.led_intensity, bool)
                or not 1 < self.led_intensity <= 100
        ):
            raise TypeError(
                f"TigerAutofocusConfig: led_intensity must be int in (1, 100], received {self.led_intensity}."
            )
        if not isinstance(self.loop_gain, int) or isinstance(self.loop_gain, bool) or not 1 <= self.loop_gain <= 100:
            raise TypeError(f"TigerAutofocusConfig: loop_gain must be int in [1, 100], received {self.loop_gain}.")
        if not isinstance(self.update_rate, int) or isinstance(self.update_rate, bool) or self.update_rate < 0:
            raise TypeError(f"TigerAutofocusConfig: update_rate must be non-negative int, received {self.update_rate}.")
        if not isinstance(self.lock_range, int | float) or isinstance(self.lock_range, bool) or not 0 < self.lock_range < 1:
            raise TypeError(f"TigerAutofocusConfig: lock_range must be numeric in (0, 1), received {self.lock_range}.")
        if not isinstance(self.objective_na, int | float) or isinstance(self.objective_na, bool) or not 0 < self.objective_na < 10:
            raise TypeError(
                f"TigerAutofocusConfig: objective_na must be numeric in (0, 10), received {self.objective_na}."
            )
        if not isinstance(self.min_snr, int | float) or isinstance(self.min_snr, bool) or self.min_snr < 0:
            raise TypeError(f"TigerAutofocusConfig: min_snr must be non-negative numeric, received {self.min_snr}.")
        if not isinstance(self.min_error, int | float) or isinstance(self.min_error, bool) or self.min_error < 0:
            raise TypeError(
                f"TigerAutofocusConfig: min_error must be non-negative numeric, received {self.min_error}."
            )
        self.lock_range = float(self.lock_range)
        self.objective_na = float(self.objective_na)

    def copy(self) -> "TigerAutofocusConfig":
        """
        Return a shallow copy of the CRISP configuration.

        Parameters
        ----------
        None

        Returns
        -------
        TigerAutofocusConfig
            Copy of this configuration.
        """
        return TigerAutofocusConfig(**self.__dict__)


class TigerAutofocusConfigFactory:
    """Factory for common ASI Tiger CRISP autofocus configurations."""

    @staticmethod
    def default_config() -> TigerAutofocusConfig:
        """
        Return the default air-objective CRISP configuration.

        Parameters
        ----------
        None

        Returns
        -------
        TigerAutofocusConfig
            Default CRISP configuration.
        """
        return TigerAutofocusConfig(
            led_intensity=70,
            loop_gain=10,
            averaging=5,
            update_rate=10,
            objective_na=0.9,
            lock_range=0.1,
        )

    @staticmethod
    def default_oil_config() -> TigerAutofocusConfig:
        """
        Return the default oil-objective CRISP configuration.

        Parameters
        ----------
        None

        Returns
        -------
        TigerAutofocusConfig
            Default oil-objective CRISP configuration.
        """
        return TigerAutofocusConfig(
            led_intensity=70,
            loop_gain=10,
            averaging=5,
            update_rate=10,
            objective_na=1.4,
            lock_range=0.1,
        )


class FakeTigerAutofocusController:
    """Deterministic Tiger-like controller for CRISP autofocus tests and dry runs."""

    def __init__(
            self,
            snr: float = 10,
            error: int = 200,
            status: str = AutoFocusStatusType.READY.value,
    ):
        """
        Initialise fake Tiger CRISP state.

        Parameters
        ----------
        snr
            Signal-to-noise value returned during fake calibration.
        error
            Error value returned during fake calibration.
        status
            Initial CRISP status flag returned by state queries.

        Returns
        -------
        None
        """
        self.snr: float = snr
        self.error: int = error
        self.status_flag: str = status
        self.connection = None
        self.commands: list[tuple[str, Any]] = []
        self.config_values: dict[str, int | float] = {}
        self.halt_was_called: bool = False

    def status(self) -> bool:
        """
        Return True to indicate that the fake controller is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def halt(self) -> None:
        """
        Record a fake halt command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.halt_was_called = True

    def crisp_get_set_state(self, card_address: int, value: Any | None) -> str:
        """
        Record and apply a fake CRISP state command.

        Parameters
        ----------
        card_address
            CRISP card address used by the caller.
        value
            CRISPState value to set, or None to query.

        Returns
        -------
        str
            Current fake CRISP status flag.
        """
        self.commands.append(("state", value))
        if value == CRISPState.IDLE:
            self.status_flag = AutoFocusStatusType.IDLE.value
        elif value == CRISPState.LOG_CAL:
            self.status_flag = AutoFocusStatusType.LOG_CAL.value
        elif value == CRISPState.LOCK:
            self.status_flag = AutoFocusStatusType.IN_FOCUS.value
        elif value == CRISPState.UNLOCK:
            self.status_flag = AutoFocusStatusType.READY.value
        elif value == CRISPState.DITHER:
            self.status_flag = AutoFocusStatusType.OUT_OF_FOCUS.value
        elif value == CRISPState.SET_GAIN:
            self.status_flag = AutoFocusStatusType.READY.value
        return self.status_flag

    def _get_set_config_value(self, name: str, value: int | float | None) -> int | float | str:
        """
        Record and apply a fake CRISP configuration command.

        Parameters
        ----------
        name
            Configuration field name.
        value
            Value to set, or None to query.

        Returns
        -------
        int | float | str
            Stored value for queries, otherwise an acknowledgement string.
        """
        self.commands.append((name, value))
        if value is not None:
            self.config_values[name] = value
            return ":A"
        return self.config_values[name]

    def crisp_get_set_objective_na(self, card_address: int, value: float | None) -> float | str:
        """Record or query fake objective NA."""
        return self._get_set_config_value("objective_na", value)

    def crisp_get_set_led_intensity(self, card_address: int, value: int | None) -> int | str:
        """Record or query fake CRISP LED intensity."""
        return self._get_set_config_value("led_intensity", value)

    def crisp_get_set_loop_gain(self, card_address: int, value: int | None) -> int | str:
        """Record or query fake CRISP loop gain."""
        return self._get_set_config_value("loop_gain", value)

    def crisp_get_set_num_avg(self, card_address: int, value: int | None) -> int | str:
        """Record or query fake CRISP averaging."""
        return self._get_set_config_value("averaging", value)

    def crisp_get_set_update_rate(self, card_address: int, value: int | None) -> int | str:
        """Record or query fake CRISP update rate."""
        return self._get_set_config_value("update_rate", value)

    def crisp_get_set_lock_range(self, card_address: int, value: float | None) -> float | str:
        """Record or query fake CRISP lock range."""
        return self._get_set_config_value("lock_range", value)

    def crisp_get_snr(self, card_address: int) -> float:
        """Return the configured fake SNR value."""
        self.commands.append(("snr", None))
        return self.snr

    def crisp_get_err(self, card_address: int) -> int:
        """Return the configured fake error value."""
        self.commands.append(("error", None))
        return self.error


class TigerAutofocus(Autofocus):
    """Autofocus implementation backed by an ASI Tiger CRISP module."""

    DEFAULT_NAME = "ASI Tiger CRISP Autofocus"

    def __init__(
            self,
            peripheral_ctrl: TigerPeripheralController,
            name: str = DEFAULT_NAME,
            tiger_config: TigerAutofocusConfig | None = None,
            pause_long: float = 5,
            pause_short: float = 1,
            sleep: Callable[[float], None] = time.sleep,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise a Tiger-backed CRISP autofocus peripheral.

        Parameters
        ----------
        peripheral_ctrl
            TigerPeripheralController that owns the shared Tiger connection and
            CRISP card address.
        name
            Human-readable autofocus name.
        tiger_config
            Default ASI Tiger CRISP configuration. If None, the factory default
            is used.
        pause_long
            Long pause in seconds used during CRISP setup.
        pause_short
            Short pause in seconds used during CRISP setup.
        sleep
            Sleep function used between CRISP commands.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require a live controller.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, TigerPeripheralController):
            raise TypeError(
                f"TigerAutofocus.__init__: peripheral_ctrl must be TigerPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        self.peripheral_ctrl: TigerPeripheralController = peripheral_ctrl
        self.tiger = peripheral_ctrl.tiger
        self.tiger_config: TigerAutofocusConfig = (
            tiger_config.copy() if tiger_config else TigerAutofocusConfigFactory.default_config()
        )
        self.pause_long: float = self._validate_pause(pause=pause_long, name="pause_long")
        self.pause_short: float = self._validate_pause(pause=pause_short, name="pause_short")
        self.sleep: Callable[[float], None] = sleep
        super().__init__(
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @staticmethod
    def _validate_pause(pause: float, name: str) -> float:
        """
        Return a validated non-negative pause duration.

        Parameters
        ----------
        pause
            Pause duration in seconds.
        name
            Field name used in exception messages.

        Returns
        -------
        float
            Validated pause duration.
        """
        if not isinstance(pause, int | float) or isinstance(pause, bool):
            raise TypeError(f"TigerAutofocus: {name} must be numeric, received {type(pause)}.")
        if pause < 0:
            raise ValueError(f"TigerAutofocus: {name} must be non-negative, received {pause}.")
        return float(pause)

    def _normalise_config(self, config: Any | None = None) -> TigerAutofocusConfig:
        """
        Return a validated TigerAutofocusConfig.

        Parameters
        ----------
        config
            Optional TigerAutofocusConfig. If None, the current default config is
            used.

        Returns
        -------
        TigerAutofocusConfig
            Copy of the validated config.
        """
        if config is None:
            return self.tiger_config.copy()
        if not isinstance(config, TigerAutofocusConfig):
            raise TypeError(
                f"TigerAutofocus._normalise_config: config must be TigerAutofocusConfig or None, "
                f"received {type(config)}."
            )
        return config.copy()

    @property
    def card_address(self) -> int:
        """
        Return the CRISP card address from the shared Tiger controller.

        Parameters
        ----------
        None

        Returns
        -------
        int
            CRISP card address.
        """
        return self.peripheral_ctrl.card_address_crisp

    def _initialise(self, force: bool = False) -> bool:
        """
        Check that the supplied Tiger controller is ready.

        Parameters
        ----------
        force
            Present for the Autofocus interface. Since this class receives an
            existing controller, force does not recreate the connection.

        Returns
        -------
        bool
            True when the Tiger controller responds to status().
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the Tiger-backed autofocus peripheral.

        Parameters
        ----------
        force
            Present for API compatibility. Tiger connection ownership belongs to
            the peripheral controller.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return whether the Tiger controller responds to status().

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when status() succeeds, otherwise False.
        """
        return self.peripheral_ctrl.is_alive()

    def _configure(self, config: Any | None = None) -> bool:
        """
        Send CRISP configuration values to the Tiger controller.

        Parameters
        ----------
        config
            Optional TigerAutofocusConfig. If None, the current default config is
            used.

        Returns
        -------
        bool
            Always True after commands are sent without raising.
        """
        tiger_config = self._normalise_config(config=config)
        self._unlock()
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_objective_na(card_address=self.card_address, value=tiger_config.objective_na)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_led_intensity(card_address=self.card_address, value=tiger_config.led_intensity)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_loop_gain(card_address=self.card_address, value=tiger_config.loop_gain)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_num_avg(card_address=self.card_address, value=tiger_config.averaging)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_update_rate(card_address=self.card_address, value=tiger_config.update_rate)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_lock_range(card_address=self.card_address, value=tiger_config.lock_range)
        self.tiger_config = tiger_config
        return True

    def _initialise_autofocus(
            self,
            config: Any | None = None,
            lock_after_initialise: bool = False,
    ) -> bool:
        """
        Run the ASI Tiger CRISP setup/calibration sequence.

        Parameters
        ----------
        config
            Optional TigerAutofocusConfig. If None, the current default config is
            used.
        lock_after_initialise
            If True, lock CRISP after a successful setup sequence.

        Returns
        -------
        bool
            True when SNR and error checks pass.
        """
        tiger_config = self._normalise_config(config=config)
        if not self._configure(config=tiger_config):
            return False

        is_success = True
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.IDLE)
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.SET_OFFSET)
        self.sleep(self.pause_short)
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.LOG_CAL)
        self.sleep(self.pause_long)
        if self.tiger.crisp_get_snr(card_address=self.card_address) < tiger_config.min_snr:
            is_success = False
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.DITHER)
        self.sleep(self.pause_long)
        if np.abs(self.tiger.crisp_get_err(card_address=self.card_address)) < tiger_config.min_error:
            is_success = False
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.SET_GAIN)
        self.sleep(self.pause_short)
        self._unlock()
        self.sleep(self.pause_short)
        if lock_after_initialise and is_success:
            self._lock()
        return is_success

    def _lock(self) -> None:
        """
        Send the CRISP lock command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.LOCK)

    def _unlock(self) -> None:
        """
        Send the CRISP unlock command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.UNLOCK)

    def _disable(self) -> None:
        """
        Set CRISP to idle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiger.crisp_get_set_state(card_address=self.card_address, value=CRISPState.IDLE)

    def _get_status(self) -> AutoFocusStatusType:
        """
        Query the current CRISP status.

        Parameters
        ----------
        None

        Returns
        -------
        AutoFocusStatusType
            Current CRISP status.
        """
        return AutoFocusStatusType.from_flag(
            status_flag=self.tiger.crisp_get_set_state(card_address=self.card_address, value=None)
        )

    def _is_locked(self) -> bool:
        """
        Return whether CRISP reports a locked state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when CRISP reports IN_FOCUS or OUT_OF_FOCUS.
        """
        status_flag = self.tiger.crisp_get_set_state(card_address=self.card_address, value=None)
        return status_flag in {
            AutoFocusStatusType.IN_FOCUS.value,
            AutoFocusStatusType.OUT_OF_FOCUS.value,
        }

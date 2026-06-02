from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any

from evomachine.coordinates import Coordinate
from evomachine.peripherals.autofocus import Autofocus
from evomachine.peripherals.stage import Stage
from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig
from evomachine.types import AutoFocusStatusType, FocusStatusType
from evomachine.utils import validate_dataclass_fields


@dataclass
class FovConfig:
    """Focus policy and optional focus configs for one field of view."""

    lock_autofocus_during_arrival_move: bool = True
    "If True, keep autofocus locked while moving to this fov."
    lock_autofocus_on_fov: bool = True
    "If True, lock autofocus after arriving and any requested focus setup."
    lock_autofocus_during_departure_move: bool = True
    "If True, keep autofocus locked while moving away from this fov."
    run_software_focus_on_arrival: bool = False
    "If True, run software focus after arriving and before autofocus locking."
    software_focus_config: SoftwareFocusConfig | None = None
    "Optional fov-specific SoftwareFocusConfig."
    autofocus_initialise_config: Any | None = None
    "Optional binding-specific autofocus config for this fov."

    def __post_init__(self) -> None:
        validate_dataclass_fields(
            dataclass_name="FovConfig",
            checks=[
                (self.lock_autofocus_during_arrival_move, bool),
                (self.lock_autofocus_on_fov, bool),
                (self.lock_autofocus_during_departure_move, bool),
                (self.run_software_focus_on_arrival, bool),
                (self.software_focus_config, [SoftwareFocusConfig, None]),
            ],
        )


@dataclass
class FocusNavigatorConfig:
    """Runtime configuration for FocusNavigator movement and recovery behavior."""

    use_autofocus: bool = False
    "If True, manage autofocus lock state during movement and focus checks."
    refocus: bool = True
    "If True, attempt focus recovery after autofocus lock loss."
    refocus_using_software_focus: bool = True
    "If True, use SoftwareFocus before reinitialising autofocus during recovery."
    refocus_on_all_fovs: bool = False
    "If True, try software focus recovery across all registered fovs."
    max_refocus_trials: int = 10
    "Maximum number of lock-loss recovery attempts before raising."
    out_of_focus_wait_s: float = 10
    "Seconds to wait after an OUT_OF_FOCUS autofocus status before checking lock state."
    post_autofocus_wait_s: float = 3
    "Seconds to wait after autofocus initialisation/lock commands."
    post_move_wait_s: float = 1
    "Seconds to wait after move-time software focus before reinitialising autofocus."
    toggle_autofocus_on_channel_change: bool = True
    "If True, unlock/refocus when moving between fovs with different channel IDs."
    autofocus_initialise_config: Any | None = None
    "Optional binding-specific autofocus config passed to Autofocus.initialise_autofocus."
    default_fov_config: FovConfig = field(default_factory=FovConfig)
    "Default per-fov focus policy used when no fov-specific override is provided."

    def __post_init__(self) -> None:
        validate_dataclass_fields(
            dataclass_name="FocusNavigatorConfig",
            checks=[
                (self.use_autofocus, bool),
                (self.refocus, bool),
                (self.refocus_using_software_focus, bool),
                (self.refocus_on_all_fovs, bool),
                (self.toggle_autofocus_on_channel_change, bool),
                (self.default_fov_config, FovConfig),
            ],
        )
        if not isinstance(self.max_refocus_trials, int) or isinstance(self.max_refocus_trials, bool):
            raise TypeError(
                f"FocusNavigatorConfig: max_refocus_trials must be int, received {type(self.max_refocus_trials)}."
            )
        if self.max_refocus_trials < 1:
            raise ValueError(
                f"FocusNavigatorConfig: max_refocus_trials must be positive, received {self.max_refocus_trials}."
            )
        for field_name in ("out_of_focus_wait_s", "post_autofocus_wait_s", "post_move_wait_s"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"FocusNavigatorConfig: {field_name} must be numeric, received {type(value)}.")
            if value < 0:
                raise ValueError(f"FocusNavigatorConfig: {field_name} must be non-negative, received {value}.")
            setattr(self, field_name, float(value))


@dataclass
class FocusNavigatorFovRecord:
    """Tracked focus/navigation state and latest outcome for one fov ID."""

    fov_id: int
    "FoV ID for the navigation record."
    coordinate: Coordinate
    "Last full coordinate known for this fov."
    fov_config: FovConfig
    "Per-fov focus policy and optional configs."
    is_locked: bool = False
    "Whether autofocus is locked after the latest operation."
    refocusing: bool = False
    "Whether focus recovery was attempted during the latest operation."
    software_focus_status: FocusStatusType = FocusStatusType.UNKNOWN
    "Software focus status from the latest operation, or UNKNOWN when not run."
    skipped: bool = False
    "True when this fov should be skipped by experiment execution."
    skip_reason: str | None = None
    "Human-readable reason for skipping this fov."
    max_refocus_trials_reached: bool = False
    "True when recovery failed because the configured trial limit was reached."
    z_time_series: list[tuple[float | int | None, float, bool]] = field(default_factory=list)
    "History of recorded Z coordinate, timestamp, and autofocus locked state."

    def __post_init__(self) -> None:
        validate_dataclass_fields(
            dataclass_name="FocusNavigatorFovRecord",
            checks=[
                (self.coordinate, Coordinate),
                (self.fov_config, FovConfig),
                (self.is_locked, bool),
                (self.refocusing, bool),
                (self.software_focus_status, FocusStatusType),
                (self.skipped, bool),
                (self.skip_reason, [str, None]),
                (self.max_refocus_trials_reached, bool),
                (self.z_time_series, list),
            ],
        )
        if not isinstance(self.fov_id, int) or isinstance(self.fov_id, bool):
            raise TypeError("FocusNavigatorFovRecord: fov_id must be int.")


class FocusNavigator:
    """Coordinate fov movement, autofocus state, and software focus recovery."""

    def __init__(
            self,
            stage: Stage,
            autofocus: Autofocus | None = None,
            software_focus: SoftwareFocus | None = None,
            config: FocusNavigatorConfig | None = None,
            sleep_func: Callable[[float], None] = time.sleep,
            time_func: Callable[[], float] = time.time,
    ):
        """
        Initialise a focus navigator.

        Parameters
        ----------
        stage
            Stage used for fov movement and coordinate readback.
        autofocus
            Optional autofocus peripheral used when config.use_autofocus is True.
        software_focus
            Optional SoftwareFocus object used for focus recovery scans.
        config
            Optional FocusNavigatorConfig. Defaults are used when omitted.
        sleep_func
            Function used for wait periods, injected for tests.
        time_func
            Function used to timestamp Z records, injected for tests.

        Returns
        -------
        None
        """
        if not isinstance(stage, Stage):
            raise TypeError(f"FocusNavigator.__init__: stage must be Stage, received {type(stage)}.")
        if autofocus is not None and not isinstance(autofocus, Autofocus):
            raise TypeError(
                f"FocusNavigator.__init__: autofocus must be Autofocus or None, received {type(autofocus)}."
            )
        self.stage: Stage = stage
        self.autofocus: Autofocus | None = autofocus
        self.software_focus: SoftwareFocus | None = software_focus
        self.config: FocusNavigatorConfig = config if config is not None else FocusNavigatorConfig()
        if not isinstance(self.config, FocusNavigatorConfig):
            raise TypeError(
                f"FocusNavigator.__init__: config must be FocusNavigatorConfig or None, received {type(config)}."
            )
        if not callable(sleep_func):
            raise TypeError(f"FocusNavigator.__init__: sleep_func must be callable, received {type(sleep_func)}.")
        if not callable(time_func):
            raise TypeError(f"FocusNavigator.__init__: time_func must be callable, received {type(time_func)}.")
        self._sleep_func = sleep_func
        self._time_func = time_func
        self._fov_states: dict[int, FocusNavigatorFovRecord] = {}
        self._fov_order: list[int] = []
        self._current_fov_id: int | None = None
        self._num_refocus: int = 0
        self._active_autofocus_initialise_config: Any | None = None

    def initialise_fovs(
            self,
            fov_id_to_coordinate: dict[int, Coordinate],
            use_autofocus: bool | None = None,
            fov_configs: dict[int, FovConfig] | None = None,
    ) -> None:
        """
        Store full fov coordinates and register stage movement coordinates.

        Parameters
        ----------
        fov_id_to_coordinate
            Mapping from fov ID to full or XY-only Coordinate.
        use_autofocus
            Optional override for whether stage registrations omit Z values.
        fov_configs
            Optional per-fov focus policy overrides. Omitted fovs use the
            navigator default fov config.

        Returns
        -------
        None
        """
        if not isinstance(fov_id_to_coordinate, dict) or not fov_id_to_coordinate:
            raise TypeError("FocusNavigator.initialise_fovs: fov_id_to_coordinate must be a non-empty dict.")
        autofocus_enabled = self.config.use_autofocus if use_autofocus is None else self._validate_bool(
            value=use_autofocus,
            name="use_autofocus",
        )
        if fov_configs is not None and not isinstance(fov_configs, dict):
            raise TypeError("FocusNavigator.initialise_fovs: fov_configs must be dict[int, FovConfig] or None.")
        states: dict[int, FocusNavigatorFovRecord] = {}
        stage_fovs: dict[int, Coordinate] = {}
        software_focus_configs: dict[int, SoftwareFocusConfig] = {}
        for fov_id, coordinate in fov_id_to_coordinate.items():
            if not isinstance(fov_id, int) or isinstance(fov_id, bool):
                raise TypeError("FocusNavigator.initialise_fovs: every fov ID must be int.")
            if not isinstance(coordinate, Coordinate):
                raise TypeError("FocusNavigator.initialise_fovs: every coordinate must be Coordinate.")
            fov_config = self._fov_config_for_id(fov_id=fov_id, fov_configs=fov_configs)
            states[fov_id] = FocusNavigatorFovRecord(
                fov_id=fov_id,
                coordinate=coordinate.copy(),
                fov_config=fov_config,
            )
            if fov_config.software_focus_config is not None:
                software_focus_configs[fov_id] = fov_config.software_focus_config
            stage_fovs[fov_id] = self._stage_registration_coordinate(
                coordinate=coordinate,
                use_autofocus=autofocus_enabled,
            )
        if fov_configs is not None:
            unknown_fov_ids = [fov_id for fov_id in fov_configs if fov_id not in fov_id_to_coordinate]
            if unknown_fov_ids:
                raise KeyError(f"FocusNavigator.initialise_fovs: unknown config fov IDs {unknown_fov_ids}.")
        if not self.stage.set_fov_id_to_coordinate(
                fov_id_to_coordinate=stage_fovs,
                use_autofocus=autofocus_enabled,
        ):
            raise RuntimeError("FocusNavigator.initialise_fovs: stage rejected fov coordinates.")
        self._fov_states = states
        self._fov_order = list(fov_id_to_coordinate)
        self._current_fov_id = None
        self._num_refocus = 0
        self._active_autofocus_initialise_config = None
        if self.software_focus is not None:
            initialise_fovs = getattr(self.software_focus, "initialise_fovs", None)
            if callable(initialise_fovs):
                initialise_fovs(
                    fov_ids=list(self._fov_order),
                    fov_configs=software_focus_configs or None,
                )

    def move(self, fov_id: int, manage_focus: bool = True) -> FocusNavigatorFovRecord:
        """
        Move to a registered fov and optionally manage focus afterwards.

        Parameters
        ----------
        fov_id
            Registered fov ID to move to.
        manage_focus
            If True, run manage_focus after movement.

        Returns
        -------
        FocusNavigatorFovRecord
            Navigation result after movement and optional focus management.
        """
        self._require_fov(fov_id=fov_id)
        if not isinstance(manage_focus, bool):
            raise TypeError(f"FocusNavigator.move: manage_focus must be bool, received {type(manage_focus)}.")
        self._prepare_for_departure(target_fov_id=fov_id)
        if self._should_toggle_autofocus(target_fov_id=fov_id):
            self._move_with_autofocus_toggle(fov_id=fov_id)
        else:
            self.stage.move(target=fov_id, block=True)
            self._handle_arrival(fov_id=fov_id)
        self._current_fov_id = fov_id
        if manage_focus:
            return self.manage_focus(fov_id=fov_id)
        return self._result(
            fov_id=fov_id,
            is_locked=self._autofocus_is_locked(default=False),
            refocusing=False,
            software_focus_status=FocusStatusType.UNKNOWN,
        )

    def manage_focus(
            self,
            fov_id: int,
            refocus_on_all_fovs: bool | None = None,
    ) -> FocusNavigatorFovRecord:
        """
        Check autofocus state and recover focus when configured to do so.

        Parameters
        ----------
        fov_id
            Current registered fov ID.
        refocus_on_all_fovs
            Optional override for config.refocus_on_all_fovs.

        Returns
        -------
        FocusNavigatorFovRecord
            Focus management result.
        """
        self._require_fov(fov_id=fov_id)
        if refocus_on_all_fovs is not None:
            refocus_on_all_fovs = self._validate_bool(
                value=refocus_on_all_fovs,
                name="refocus_on_all_fovs",
            )
        self._current_fov_id = fov_id
        if not self.config.use_autofocus:
            self._record_current_z(fov_id=fov_id, is_locked=False)
            return self._result(
                fov_id=fov_id,
                is_locked=False,
                refocusing=False,
                software_focus_status=FocusStatusType.UNKNOWN,
            )
        self._require_autofocus()
        if self.autofocus.get_status() == AutoFocusStatusType.OUT_OF_FOCUS:
            self._sleep_func(self.config.out_of_focus_wait_s)
        if self.autofocus.is_locked():
            self._record_current_z(fov_id=fov_id, is_locked=True)
            return self._result(
                fov_id=fov_id,
                is_locked=True,
                refocusing=False,
                software_focus_status=FocusStatusType.UNKNOWN,
            )
        if not self.config.refocus:
            raise RuntimeError("FocusNavigator.manage_focus: autofocus lock lost and refocus is disabled.")
        if self._num_refocus >= self.config.max_refocus_trials:
            self._record_current_z(fov_id=fov_id, is_locked=False)
            return self._result(
                fov_id=fov_id,
                is_locked=False,
                refocusing=True,
                software_focus_status=FocusStatusType.UNKNOWN,
                skipped=True,
                skip_reason=(
                    "FocusNavigator.manage_focus: maximum refocus trials reached "
                    f"({self.config.max_refocus_trials})."
                ),
                max_refocus_trials_reached=True,
            )
        self._num_refocus += 1
        self.autofocus.unlock()
        try:
            if self.config.refocus_using_software_focus:
                software_focus_status = self._recover_with_software_focus(
                    original_fov_id=fov_id,
                    refocus_on_all_fovs=(
                        self.config.refocus_on_all_fovs
                        if refocus_on_all_fovs is None
                        else refocus_on_all_fovs
                    ),
                )
            else:
                software_focus_status = FocusStatusType.UNKNOWN
                self._recover_with_previous_fov(original_fov_id=fov_id)
        except RuntimeError as exc:
            self._record_current_z(fov_id=fov_id, is_locked=False)
            if self._num_refocus >= self.config.max_refocus_trials:
                return self._result(
                    fov_id=fov_id,
                    is_locked=False,
                    refocusing=True,
                    software_focus_status=FocusStatusType.UNKNOWN,
                    skipped=True,
                    skip_reason=str(exc),
                    max_refocus_trials_reached=True,
                )
            raise
        self._record_current_z(fov_id=fov_id, is_locked=self._autofocus_is_locked(default=False))
        return self._result(
            fov_id=fov_id,
            is_locked=self._autofocus_is_locked(default=False),
            refocusing=True,
            software_focus_status=software_focus_status,
        )

    def reset_refocus_count(self) -> None:
        """
        Reset the refocus trial counter.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._num_refocus = 0

    def get_fov_state(self, fov_id: int) -> FocusNavigatorFovRecord:
        """
        Return tracked state for one fov.

        Parameters
        ----------
        fov_id
            Registered fov ID to retrieve.

        Returns
        -------
        FocusNavigatorFovRecord
            Tracked fov state.
        """
        self._require_fov(fov_id=fov_id)
        return self._fov_states[fov_id]

    def get_next_fov_id(self, fov_id: int) -> int:
        """
        Return the next registered fov ID, wrapping at the end.

        Parameters
        ----------
        fov_id
            Current registered fov ID.

        Returns
        -------
        int
            Next registered fov ID.
        """
        self._require_fov(fov_id=fov_id)
        index = self._fov_order.index(fov_id)
        return self._fov_order[(index + 1) % len(self._fov_order)]

    @staticmethod
    def _validate_bool(value: bool, name: str) -> bool:
        """
        Return a validated boolean value.

        Parameters
        ----------
        value
            Candidate boolean value.
        name
            Field name used in error messages.

        Returns
        -------
        bool
            Validated value.
        """
        if not isinstance(value, bool):
            raise TypeError(f"FocusNavigator: {name} must be bool, received {type(value)}.")
        return value

    def _fov_config_for_id(
            self,
            fov_id: int,
            fov_configs: dict[int, FovConfig] | None,
    ) -> FovConfig:
        """
        Return the focus policy for one fov.

        Parameters
        ----------
        fov_id
            Registered fov ID.
        fov_configs
            Optional per-fov focus policies.

        Returns
        -------
        FovConfig
            Per-fov override or navigator default.
        """
        if fov_configs is None or fov_id not in fov_configs:
            return self.config.default_fov_config
        fov_config = fov_configs[fov_id]
        if not isinstance(fov_config, FovConfig):
            raise TypeError("FocusNavigator.initialise_fovs: every fov config must be FovConfig.")
        return fov_config

    @staticmethod
    def _stage_registration_coordinate(coordinate: Coordinate, use_autofocus: bool) -> Coordinate:
        """
        Return the coordinate registered with Stage for one navigator fov.

        Parameters
        ----------
        coordinate
            Full coordinate stored by the navigator.
        use_autofocus
            If True, omit Z because autofocus owns focus height.

        Returns
        -------
        Coordinate
            Coordinate passed to Stage.set_fov_id_to_coordinate.
        """
        if use_autofocus:
            return Coordinate(
                x=coordinate.x,
                y=coordinate.y,
                z=None,
                channel_id=coordinate.get_channel_id(),
            )
        return coordinate.copy()

    def _require_fov(self, fov_id: int) -> None:
        """
        Raise if a fov ID is not registered.

        Parameters
        ----------
        fov_id
            FoV ID to validate.

        Returns
        -------
        None
        """
        if not isinstance(fov_id, int) or isinstance(fov_id, bool):
            raise TypeError(f"FocusNavigator: fov_id must be int, received {type(fov_id)}.")
        if fov_id not in self._fov_states:
            raise KeyError(f"FocusNavigator: unknown fov ID {fov_id}.")

    def _require_autofocus(self) -> Autofocus:
        """
        Return the configured autofocus peripheral or raise.

        Parameters
        ----------
        None

        Returns
        -------
        Autofocus
            Configured autofocus peripheral.
        """
        if self.autofocus is None:
            raise RuntimeError("FocusNavigator: autofocus is required when use_autofocus is True.")
        return self.autofocus

    def _require_software_focus(self) -> SoftwareFocus:
        """
        Return the configured software focus object or raise.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocus
            SoftwareFocus object exposing run(fov_id=...).
        """
        if self.software_focus is None:
            raise RuntimeError("FocusNavigator: software_focus is required for software focus recovery.")
        run = getattr(self.software_focus, "run", None)
        if not callable(run):
            raise RuntimeError("FocusNavigator: software_focus must expose a callable run method.")
        return self.software_focus

    def _prepare_for_departure(self, target_fov_id: int) -> None:
        """
        Apply the current fov departure focus policy before movement.

        Parameters
        ----------
        target_fov_id
            Registered fov ID that will be moved to.

        Returns
        -------
        None
        """
        if self._current_fov_id is None or self._current_fov_id == target_fov_id:
            return
        if not self.config.use_autofocus or self.autofocus is None:
            return
        current_config = self._fov_states[self._current_fov_id].fov_config
        if not current_config.lock_autofocus_during_departure_move and self.autofocus.is_locked():
            self.autofocus.unlock()

    def _handle_arrival(self, fov_id: int) -> FocusStatusType:
        """
        Apply target fov arrival focus policy after movement.

        Parameters
        ----------
        fov_id
            Registered fov ID that has just been reached.

        Returns
        -------
        FocusStatusType
            Software focus status from arrival focus, or UNKNOWN when not run.
        """
        fov_config = self._fov_states[fov_id].fov_config
        if (
                self.config.use_autofocus
                and self.autofocus is not None
                and not fov_config.lock_autofocus_during_arrival_move
                and self.autofocus.is_locked()
        ):
            self.autofocus.unlock()
        software_focus_status = FocusStatusType.UNKNOWN
        if fov_config.run_software_focus_on_arrival:
            software_focus_status = self._run_software_focus(fov_id=fov_id)
            if software_focus_status != FocusStatusType.IN_FOCUS:
                raise RuntimeError(
                    f"FocusNavigator.move: software focus failed with status {software_focus_status}."
                )
            self._update_recorded_z(fov_id=fov_id)
            self._sleep_func(self.config.post_move_wait_s)
        reinitialised = False
        if self.config.use_autofocus:
            reinitialised = self._ensure_autofocus_config(fov_id=fov_id)
            if fov_config.lock_autofocus_on_fov and (
                    reinitialised
                    or software_focus_status == FocusStatusType.IN_FOCUS
            ):
                self._require_autofocus().lock()
        return software_focus_status

    def _ensure_autofocus_config(self, fov_id: int) -> bool:
        """
        Reinitialise autofocus when the target fov config differs from active config.

        Parameters
        ----------
        fov_id
            Registered fov ID whose autofocus config should be active.

        Returns
        -------
        bool
            True when autofocus was reinitialised.
        """
        autofocus = self._require_autofocus()
        target_config = self._autofocus_initialise_config_for_fov(fov_id=fov_id)
        if target_config == self._active_autofocus_initialise_config:
            return False
        if autofocus.is_locked():
            autofocus.unlock()
        self._initialise_and_lock_autofocus(config=target_config, lock_after_initialise=False)
        self._sleep_func(self.config.post_autofocus_wait_s)
        self._active_autofocus_initialise_config = target_config
        return True

    def _autofocus_initialise_config_for_fov(self, fov_id: int) -> Any | None:
        """
        Return the autofocus initialise config for one fov.

        Parameters
        ----------
        fov_id
            Registered fov ID.

        Returns
        -------
        Any | None
            Per-fov autofocus config, global fallback config, or None.
        """
        fov_config = self._fov_states[fov_id].fov_config.autofocus_initialise_config
        return self.config.autofocus_initialise_config if fov_config is None else fov_config

    def _should_toggle_autofocus(self, target_fov_id: int) -> bool:
        """
        Return whether a move should use autofocus toggle/refocus behavior.

        Parameters
        ----------
        target_fov_id
            Registered target fov ID.

        Returns
        -------
        bool
            True when channel IDs differ and toggle behavior is enabled.
        """
        if not (
                self.config.use_autofocus
                and self.config.toggle_autofocus_on_channel_change
                and self._current_fov_id is not None
        ):
            return False
        current_coordinate = self._fov_states[self._current_fov_id].coordinate
        target_coordinate = self._fov_states[target_fov_id].coordinate
        current_channel_id = current_coordinate.get_channel_id()
        target_channel_id = target_coordinate.get_channel_id()
        return (
            current_channel_id is not None
            and target_channel_id is not None
            and current_channel_id != target_channel_id
        )

    def _move_with_autofocus_toggle(self, fov_id: int) -> None:
        """
        Move using software focus and autofocus reinitialisation for channel changes.

        Parameters
        ----------
        fov_id
            Registered target fov ID.

        Returns
        -------
        None
        """
        autofocus = self._require_autofocus()
        autofocus.unlock()
        self.stage.move(target=self._fov_states[fov_id].coordinate.copy(), block=True)
        fov_config = self._fov_states[fov_id].fov_config
        software_focus_status = (
            self._run_software_focus(fov_id=fov_id)
            if fov_config.run_software_focus_on_arrival or self.config.refocus_using_software_focus
            else FocusStatusType.UNKNOWN
        )
        if software_focus_status != FocusStatusType.IN_FOCUS:
            raise RuntimeError(
                f"FocusNavigator.move: software focus failed with status {software_focus_status}."
            )
        self._sleep_func(self.config.post_move_wait_s)
        target_config = self._autofocus_initialise_config_for_fov(fov_id=fov_id)
        self._initialise_and_lock_autofocus(
            config=target_config,
            lock_after_initialise=fov_config.lock_autofocus_on_fov,
        )
        self._active_autofocus_initialise_config = target_config
        self._sleep_func(self.config.post_autofocus_wait_s)
        self._record_current_z(fov_id=fov_id, is_locked=autofocus.is_locked())

    def _recover_with_previous_fov(self, original_fov_id: int) -> None:
        """
        Recover autofocus by reinitialising at the previous registered fov.

        Parameters
        ----------
        original_fov_id
            FoV ID where lock loss was detected.

        Returns
        -------
        None
        """
        previous_fov_id = self._previous_fov_id(fov_id=original_fov_id)
        self.stage.move(target=self._fov_states[previous_fov_id].coordinate.copy(), block=True)
        self._initialise_and_lock_autofocus(
            config=self._autofocus_initialise_config_for_fov(fov_id=previous_fov_id),
            lock_after_initialise=True,
        )
        self._active_autofocus_initialise_config = self._autofocus_initialise_config_for_fov(fov_id=previous_fov_id)
        self._sleep_func(self.config.post_autofocus_wait_s)
        self.stage.move(target=original_fov_id, block=True)
        self._current_fov_id = original_fov_id
        self._update_recorded_z(fov_id=original_fov_id)

    def _recover_with_software_focus(
            self,
            original_fov_id: int,
            refocus_on_all_fovs: bool,
    ) -> FocusStatusType:
        """
        Recover focus by running software focus on one or more fovs.

        Parameters
        ----------
        original_fov_id
            FoV ID where lock loss was detected.
        refocus_on_all_fovs
            If True, try each registered fov until software focus succeeds.

        Returns
        -------
        FocusStatusType
            Software focus status from the successful recovery run.
        """
        candidate_fov_id = original_fov_id
        attempts = len(self._fov_order) if refocus_on_all_fovs else 1
        last_status = FocusStatusType.UNKNOWN
        for _ in range(attempts):
            self.stage.move(target=self._fov_states[candidate_fov_id].coordinate.copy(), block=True)
            last_status = self._run_software_focus(fov_id=candidate_fov_id)
            if last_status == FocusStatusType.IN_FOCUS:
                target_config = self._autofocus_initialise_config_for_fov(fov_id=candidate_fov_id)
                self._initialise_and_lock_autofocus(config=target_config, lock_after_initialise=True)
                self._active_autofocus_initialise_config = target_config
                self._update_recorded_z(fov_id=candidate_fov_id)
                if candidate_fov_id != original_fov_id:
                    self.stage.move(target=original_fov_id, block=True)
                    self._current_fov_id = original_fov_id
                return last_status
            candidate_fov_id = self.get_next_fov_id(fov_id=candidate_fov_id)
        raise RuntimeError(
            f"FocusNavigator.manage_focus: software focus failed with status {last_status}."
        )

    def _run_software_focus(self, fov_id: int) -> FocusStatusType:
        """
        Run software focus for one fov and return its status.

        Parameters
        ----------
        fov_id
            Registered fov ID passed to SoftwareFocus.run.

        Returns
        -------
        FocusStatusType
            Focus status from the software focus result.
        """
        software_focus = self._require_software_focus()
        result = software_focus.run(fov_id=fov_id)
        focus_status = getattr(result, "focus_status", None)
        if not isinstance(focus_status, FocusStatusType):
            raise RuntimeError("FocusNavigator: software_focus.run result must expose focus_status.")
        return focus_status

    def _initialise_and_lock_autofocus(
            self,
            config: Any | None = None,
            lock_after_initialise: bool = True,
    ) -> None:
        """
        Initialise autofocus and optionally lock it, raising if initialisation fails.

        Parameters
        ----------
        config
            Optional binding-specific autofocus initialise config.
        lock_after_initialise
            If True, lock autofocus after successful initialisation.

        Returns
        -------
        None
        """
        autofocus = self._require_autofocus()
        is_success = autofocus.initialise_autofocus(
            config=self.config.autofocus_initialise_config if config is None else config,
            lock_after_initialise=False,
        )
        if not is_success:
            raise RuntimeError("FocusNavigator: autofocus initialisation failed.")
        if lock_after_initialise:
            autofocus.lock()

    def _previous_fov_id(self, fov_id: int) -> int:
        """
        Return the previous registered fov ID, wrapping at the start.

        Parameters
        ----------
        fov_id
            Current registered fov ID.

        Returns
        -------
        int
            Previous registered fov ID.
        """
        self._require_fov(fov_id=fov_id)
        index = self._fov_order.index(fov_id)
        return self._fov_order[index - 1]

    def _autofocus_is_locked(self, default: bool) -> bool:
        """
        Return autofocus lock state or a default when no autofocus is configured.

        Parameters
        ----------
        default
            Value returned when autofocus is None.

        Returns
        -------
        bool
            Current lock state or default.
        """
        if self.autofocus is None:
            return default
        return self.autofocus.is_locked()

    def _record_current_z(self, fov_id: int, is_locked: bool) -> None:
        """
        Read current Z, update stored coordinate, and append time-series state.

        Parameters
        ----------
        fov_id
            Registered fov ID to update.
        is_locked
            Autofocus lock state to store with the Z record.

        Returns
        -------
        None
        """
        coordinate = self._update_recorded_z(fov_id=fov_id)
        self._fov_states[fov_id].z_time_series.append(
            (coordinate.z, self._time_func(), is_locked)
        )
        self._fov_states[fov_id].is_locked = is_locked

    def _update_recorded_z(self, fov_id: int) -> Coordinate:
        """
        Update one stored fov coordinate from the stage's current Z.

        Parameters
        ----------
        fov_id
            Registered fov ID to update.

        Returns
        -------
        Coordinate
            Updated stored coordinate.
        """
        current_coordinate = self.stage.get_coordinates(query_hardware=True)
        state_coordinate = self._fov_states[fov_id].coordinate.copy()
        state_coordinate.z = current_coordinate.z
        self._fov_states[fov_id].coordinate = state_coordinate
        return state_coordinate

    def _result(
            self,
            fov_id: int,
            is_locked: bool,
            refocusing: bool,
            software_focus_status: FocusStatusType,
            skipped: bool = False,
            skip_reason: str | None = None,
            max_refocus_trials_reached: bool = False,
    ) -> FocusNavigatorFovRecord:
        """
        Update and return a FocusNavigatorFovRecord for the current stored fov state.

        Parameters
        ----------
        fov_id
            Registered fov ID for the result.
        is_locked
            Current autofocus lock state.
        refocusing
            Whether recovery was attempted.
        software_focus_status
            Software focus status associated with the result.
        skipped
            Whether this fov should be skipped by experiment execution.
        skip_reason
            Optional human-readable reason for skipping this fov.
        max_refocus_trials_reached
            Whether the refocus trial limit was reached.

        Returns
        -------
        FocusNavigatorFovRecord
            Updated fov record.
        """
        record = self._fov_states[fov_id]
        record.is_locked = is_locked
        record.refocusing = refocusing
        record.software_focus_status = software_focus_status
        record.skipped = skipped
        record.skip_reason = skip_reason
        record.max_refocus_trials_reached = max_refocus_trials_reached
        return record

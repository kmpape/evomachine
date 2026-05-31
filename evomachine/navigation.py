from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any

from evomachine.coordinates import Coordinate
from evomachine.peripherals.autofocus import Autofocus
from evomachine.peripherals.stage import Stage
from evomachine.softwarefocus import SoftwareFocus
from evomachine.types import AutoFocusStatusType, FocusStatusType


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

    def __post_init__(self) -> None:
        """
        Validate focus navigator configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        for field_name in (
                "use_autofocus",
                "refocus",
                "refocus_using_software_focus",
                "refocus_on_all_fovs",
                "toggle_autofocus_on_channel_change",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"FocusNavigatorConfig: {field_name} must be bool, received {type(value)}.")
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
class FocusNavigatorFovState:
    """Tracked focus/navigation state for one fov ID."""

    coordinate: Coordinate
    "Last full coordinate known for this fov."
    z_time_series: list[tuple[float | int | None, float, bool]] = field(default_factory=list)
    "History of recorded Z coordinate, timestamp, and autofocus locked state."

    def __post_init__(self) -> None:
        """
        Validate focus navigator fov state after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.coordinate, Coordinate):
            raise TypeError(
                f"FocusNavigatorFovState: coordinate must be Coordinate, received {type(self.coordinate)}."
            )
        if not isinstance(self.z_time_series, list):
            raise TypeError(
                f"FocusNavigatorFovState: z_time_series must be list, received {type(self.z_time_series)}."
            )


@dataclass
class FocusNavigatorResult:
    """Outcome returned by FocusNavigator movement and focus management calls."""

    fov_id: int
    "FoV ID after the navigation operation."
    coordinate: Coordinate
    "Full coordinate recorded for the fov."
    is_locked: bool
    "Whether autofocus is locked after the operation."
    refocusing: bool
    "Whether focus recovery was attempted."
    software_focus_status: FocusStatusType
    "Software focus status from the recovery attempt, or UNKNOWN when not run."
    max_refocus_trials_reached: bool = False
    "True when recovery failed because the configured trial limit was reached."

    def __post_init__(self) -> None:
        """
        Validate focus navigator result after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.fov_id, int) or isinstance(self.fov_id, bool):
            raise TypeError(f"FocusNavigatorResult: fov_id must be int, received {type(self.fov_id)}.")
        if not isinstance(self.coordinate, Coordinate):
            raise TypeError(f"FocusNavigatorResult: coordinate must be Coordinate, received {type(self.coordinate)}.")
        if not isinstance(self.is_locked, bool):
            raise TypeError(f"FocusNavigatorResult: is_locked must be bool, received {type(self.is_locked)}.")
        if not isinstance(self.refocusing, bool):
            raise TypeError(f"FocusNavigatorResult: refocusing must be bool, received {type(self.refocusing)}.")
        if not isinstance(self.software_focus_status, FocusStatusType):
            raise TypeError(
                f"FocusNavigatorResult: software_focus_status must be FocusStatusType, "
                f"received {type(self.software_focus_status)}."
            )
        if not isinstance(self.max_refocus_trials_reached, bool):
            raise TypeError(
                f"FocusNavigatorResult: max_refocus_trials_reached must be bool, "
                f"received {type(self.max_refocus_trials_reached)}."
            )


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
        self._fov_states: dict[int, FocusNavigatorFovState] = {}
        self._fov_order: list[int] = []
        self._current_fov_id: int | None = None
        self._num_refocus: int = 0

    def initialise_fovs(
            self,
            fov_id_to_coordinate: dict[int, Coordinate],
            use_autofocus: bool | None = None,
    ) -> None:
        """
        Store full fov coordinates and register stage movement coordinates.

        Parameters
        ----------
        fov_id_to_coordinate
            Mapping from fov ID to full or XY-only Coordinate.
        use_autofocus
            Optional override for whether stage registrations omit Z values.

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
        states: dict[int, FocusNavigatorFovState] = {}
        stage_fovs: dict[int, Coordinate] = {}
        for fov_id, coordinate in fov_id_to_coordinate.items():
            if not isinstance(fov_id, int) or isinstance(fov_id, bool):
                raise TypeError("FocusNavigator.initialise_fovs: every fov ID must be int.")
            if not isinstance(coordinate, Coordinate):
                raise TypeError("FocusNavigator.initialise_fovs: every coordinate must be Coordinate.")
            states[fov_id] = FocusNavigatorFovState(coordinate=coordinate.copy())
            stage_fovs[fov_id] = self._stage_registration_coordinate(
                coordinate=coordinate,
                use_autofocus=autofocus_enabled,
            )
        if not self.stage.set_fov_id_to_coordinate(
                fov_id_to_coordinate=stage_fovs,
                use_autofocus=autofocus_enabled,
        ):
            raise RuntimeError("FocusNavigator.initialise_fovs: stage rejected fov coordinates.")
        self._fov_states = states
        self._fov_order = list(fov_id_to_coordinate)
        self._current_fov_id = None
        if self.software_focus is not None:
            initialise_fovs = getattr(self.software_focus, "initialise_fovs", None)
            if callable(initialise_fovs):
                initialise_fovs(fov_ids=list(self._fov_order))

    def move(self, fov_id: int, manage_focus: bool = True) -> FocusNavigatorResult:
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
        FocusNavigatorResult
            Navigation result after movement and optional focus management.
        """
        self._require_fov(fov_id=fov_id)
        if not isinstance(manage_focus, bool):
            raise TypeError(f"FocusNavigator.move: manage_focus must be bool, received {type(manage_focus)}.")
        if self._should_toggle_autofocus(target_fov_id=fov_id):
            self._move_with_autofocus_toggle(fov_id=fov_id)
        else:
            self.stage.move(target=fov_id, block=True)
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
    ) -> FocusNavigatorResult:
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
        FocusNavigatorResult
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
            raise RuntimeError(
                "FocusNavigator.manage_focus: maximum refocus trials reached "
                f"({self.config.max_refocus_trials})."
            )
        self._num_refocus += 1
        self.autofocus.unlock()
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

    def get_fov_state(self, fov_id: int) -> FocusNavigatorFovState:
        """
        Return tracked state for one fov.

        Parameters
        ----------
        fov_id
            Registered fov ID to retrieve.

        Returns
        -------
        FocusNavigatorFovState
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
        software_focus_status = self._run_software_focus(fov_id=fov_id)
        if software_focus_status != FocusStatusType.IN_FOCUS:
            raise RuntimeError(
                f"FocusNavigator.move: software focus failed with status {software_focus_status}."
            )
        self._sleep_func(self.config.post_move_wait_s)
        self._initialise_and_lock_autofocus()
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
        self._initialise_and_lock_autofocus()
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
                self._initialise_and_lock_autofocus()
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

    def _initialise_and_lock_autofocus(self) -> None:
        """
        Initialise autofocus and lock it, raising if initialisation fails.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        autofocus = self._require_autofocus()
        is_success = autofocus.initialise_autofocus(
            config=self.config.autofocus_initialise_config,
            lock_after_initialise=False,
        )
        if not is_success:
            raise RuntimeError("FocusNavigator: autofocus initialisation failed.")
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
            max_refocus_trials_reached: bool = False,
    ) -> FocusNavigatorResult:
        """
        Build a FocusNavigatorResult for the current stored fov state.

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
        max_refocus_trials_reached
            Whether the refocus trial limit was reached.

        Returns
        -------
        FocusNavigatorResult
            Result object.
        """
        return FocusNavigatorResult(
            fov_id=fov_id,
            coordinate=self._fov_states[fov_id].coordinate.copy(),
            is_locked=is_locked,
            refocusing=refocusing,
            software_focus_status=software_focus_status,
            max_refocus_trials_reached=max_refocus_trials_reached,
        )

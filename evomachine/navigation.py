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
    refocus_on_all_positions: bool = False
    "If True, try software focus recovery across all registered positions."
    max_refocus_trials: int = 10
    "Maximum number of lock-loss recovery attempts before raising."
    out_of_focus_wait_s: float = 10
    "Seconds to wait after an OUT_OF_FOCUS autofocus status before checking lock state."
    post_autofocus_wait_s: float = 3
    "Seconds to wait after autofocus initialisation/lock commands."
    post_move_wait_s: float = 1
    "Seconds to wait after move-time software focus before reinitialising autofocus."
    toggle_autofocus_on_channel_change: bool = True
    "If True, unlock/refocus when moving between positions with different channel IDs."
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
                "refocus_on_all_positions",
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
class FocusNavigatorPositionState:
    """Tracked focus/navigation state for one position ID."""

    coordinate: Coordinate
    "Last full coordinate known for this position."
    z_time_series: list[tuple[float | int | None, float, bool]] = field(default_factory=list)
    "History of recorded Z coordinate, timestamp, and autofocus locked state."

    def __post_init__(self) -> None:
        """
        Validate focus navigator position state after construction.

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
                f"FocusNavigatorPositionState: coordinate must be Coordinate, received {type(self.coordinate)}."
            )
        if not isinstance(self.z_time_series, list):
            raise TypeError(
                f"FocusNavigatorPositionState: z_time_series must be list, received {type(self.z_time_series)}."
            )


@dataclass
class FocusNavigatorResult:
    """Outcome returned by FocusNavigator movement and focus management calls."""

    position_id: int
    "Position ID after the navigation operation."
    coordinate: Coordinate
    "Full coordinate recorded for the position."
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
        if not isinstance(self.position_id, int) or isinstance(self.position_id, bool):
            raise TypeError(f"FocusNavigatorResult: position_id must be int, received {type(self.position_id)}.")
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
    """Coordinate position movement, autofocus state, and software focus recovery."""

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
            Stage used for position movement and coordinate readback.
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
        self._position_states: dict[int, FocusNavigatorPositionState] = {}
        self._position_order: list[int] = []
        self._current_position_id: int | None = None
        self._num_refocus: int = 0

    def initialise_positions(
            self,
            position_id_to_coordinate: dict[int, Coordinate],
            use_autofocus: bool | None = None,
    ) -> None:
        """
        Store full position coordinates and register stage movement coordinates.

        Parameters
        ----------
        position_id_to_coordinate
            Mapping from position ID to full or XY-only Coordinate.
        use_autofocus
            Optional override for whether stage registrations omit Z values.

        Returns
        -------
        None
        """
        if not isinstance(position_id_to_coordinate, dict) or not position_id_to_coordinate:
            raise TypeError("FocusNavigator.initialise_positions: position_id_to_coordinate must be a non-empty dict.")
        autofocus_enabled = self.config.use_autofocus if use_autofocus is None else self._validate_bool(
            value=use_autofocus,
            name="use_autofocus",
        )
        states: dict[int, FocusNavigatorPositionState] = {}
        stage_positions: dict[int, Coordinate] = {}
        for position_id, coordinate in position_id_to_coordinate.items():
            if not isinstance(position_id, int) or isinstance(position_id, bool):
                raise TypeError("FocusNavigator.initialise_positions: every position ID must be int.")
            if not isinstance(coordinate, Coordinate):
                raise TypeError("FocusNavigator.initialise_positions: every coordinate must be Coordinate.")
            states[position_id] = FocusNavigatorPositionState(coordinate=coordinate.copy())
            stage_positions[position_id] = self._stage_registration_coordinate(
                coordinate=coordinate,
                use_autofocus=autofocus_enabled,
            )
        if not self.stage.set_pos_id_to_coordinate(
                pos_id_to_coordinate=stage_positions,
                use_autofocus=autofocus_enabled,
        ):
            raise RuntimeError("FocusNavigator.initialise_positions: stage rejected position coordinates.")
        self._position_states = states
        self._position_order = list(position_id_to_coordinate)
        self._current_position_id = None
        if self.software_focus is not None:
            initialise_positions = getattr(self.software_focus, "initialise_positions", None)
            if callable(initialise_positions):
                initialise_positions(position_ids=list(self._position_order))

    def move(self, position_id: int, manage_focus: bool = True) -> FocusNavigatorResult:
        """
        Move to a registered position and optionally manage focus afterwards.

        Parameters
        ----------
        position_id
            Registered position ID to move to.
        manage_focus
            If True, run manage_focus after movement.

        Returns
        -------
        FocusNavigatorResult
            Navigation result after movement and optional focus management.
        """
        self._require_position(position_id=position_id)
        if not isinstance(manage_focus, bool):
            raise TypeError(f"FocusNavigator.move: manage_focus must be bool, received {type(manage_focus)}.")
        if self._should_toggle_autofocus(target_position_id=position_id):
            self._move_with_autofocus_toggle(position_id=position_id)
        else:
            self.stage.move(target=position_id, block=True)
        self._current_position_id = position_id
        if manage_focus:
            return self.manage_focus(position_id=position_id)
        return self._result(
            position_id=position_id,
            is_locked=self._autofocus_is_locked(default=False),
            refocusing=False,
            software_focus_status=FocusStatusType.UNKNOWN,
        )

    def manage_focus(
            self,
            position_id: int,
            refocus_on_all_positions: bool | None = None,
    ) -> FocusNavigatorResult:
        """
        Check autofocus state and recover focus when configured to do so.

        Parameters
        ----------
        position_id
            Current registered position ID.
        refocus_on_all_positions
            Optional override for config.refocus_on_all_positions.

        Returns
        -------
        FocusNavigatorResult
            Focus management result.
        """
        self._require_position(position_id=position_id)
        if refocus_on_all_positions is not None:
            refocus_on_all_positions = self._validate_bool(
                value=refocus_on_all_positions,
                name="refocus_on_all_positions",
            )
        self._current_position_id = position_id
        if not self.config.use_autofocus:
            self._record_current_z(position_id=position_id, is_locked=False)
            return self._result(
                position_id=position_id,
                is_locked=False,
                refocusing=False,
                software_focus_status=FocusStatusType.UNKNOWN,
            )
        self._require_autofocus()
        if self.autofocus.get_status() == AutoFocusStatusType.OUT_OF_FOCUS:
            self._sleep_func(self.config.out_of_focus_wait_s)
        if self.autofocus.is_locked():
            self._record_current_z(position_id=position_id, is_locked=True)
            return self._result(
                position_id=position_id,
                is_locked=True,
                refocusing=False,
                software_focus_status=FocusStatusType.UNKNOWN,
            )
        if not self.config.refocus:
            raise RuntimeError("FocusNavigator.manage_focus: autofocus lock lost and refocus is disabled.")
        if self._num_refocus >= self.config.max_refocus_trials:
            self._record_current_z(position_id=position_id, is_locked=False)
            raise RuntimeError(
                "FocusNavigator.manage_focus: maximum refocus trials reached "
                f"({self.config.max_refocus_trials})."
            )
        self._num_refocus += 1
        self.autofocus.unlock()
        if self.config.refocus_using_software_focus:
            software_focus_status = self._recover_with_software_focus(
                original_position_id=position_id,
                refocus_on_all_positions=(
                    self.config.refocus_on_all_positions
                    if refocus_on_all_positions is None
                    else refocus_on_all_positions
                ),
            )
        else:
            software_focus_status = FocusStatusType.UNKNOWN
            self._recover_with_previous_position(original_position_id=position_id)
        self._record_current_z(position_id=position_id, is_locked=self._autofocus_is_locked(default=False))
        return self._result(
            position_id=position_id,
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

    def get_position_state(self, position_id: int) -> FocusNavigatorPositionState:
        """
        Return tracked state for one position.

        Parameters
        ----------
        position_id
            Registered position ID to retrieve.

        Returns
        -------
        FocusNavigatorPositionState
            Tracked position state.
        """
        self._require_position(position_id=position_id)
        return self._position_states[position_id]

    def get_next_position_id(self, position_id: int) -> int:
        """
        Return the next registered position ID, wrapping at the end.

        Parameters
        ----------
        position_id
            Current registered position ID.

        Returns
        -------
        int
            Next registered position ID.
        """
        self._require_position(position_id=position_id)
        index = self._position_order.index(position_id)
        return self._position_order[(index + 1) % len(self._position_order)]

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
        Return the coordinate registered with Stage for one navigator position.

        Parameters
        ----------
        coordinate
            Full coordinate stored by the navigator.
        use_autofocus
            If True, omit Z because autofocus owns focus height.

        Returns
        -------
        Coordinate
            Coordinate passed to Stage.set_pos_id_to_coordinate.
        """
        if use_autofocus:
            return Coordinate(
                x=coordinate.x,
                y=coordinate.y,
                z=None,
                channel_id=coordinate.get_channel_id(),
            )
        return coordinate.copy()

    def _require_position(self, position_id: int) -> None:
        """
        Raise if a position ID is not registered.

        Parameters
        ----------
        position_id
            Position ID to validate.

        Returns
        -------
        None
        """
        if not isinstance(position_id, int) or isinstance(position_id, bool):
            raise TypeError(f"FocusNavigator: position_id must be int, received {type(position_id)}.")
        if position_id not in self._position_states:
            raise KeyError(f"FocusNavigator: unknown position ID {position_id}.")

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
            SoftwareFocus object exposing run(position_id=...).
        """
        if self.software_focus is None:
            raise RuntimeError("FocusNavigator: software_focus is required for software focus recovery.")
        run = getattr(self.software_focus, "run", None)
        if not callable(run):
            raise RuntimeError("FocusNavigator: software_focus must expose a callable run method.")
        return self.software_focus

    def _should_toggle_autofocus(self, target_position_id: int) -> bool:
        """
        Return whether a move should use autofocus toggle/refocus behavior.

        Parameters
        ----------
        target_position_id
            Registered target position ID.

        Returns
        -------
        bool
            True when channel IDs differ and toggle behavior is enabled.
        """
        if not (
                self.config.use_autofocus
                and self.config.toggle_autofocus_on_channel_change
                and self._current_position_id is not None
        ):
            return False
        current_coordinate = self._position_states[self._current_position_id].coordinate
        target_coordinate = self._position_states[target_position_id].coordinate
        current_channel_id = current_coordinate.get_channel_id()
        target_channel_id = target_coordinate.get_channel_id()
        return (
            current_channel_id is not None
            and target_channel_id is not None
            and current_channel_id != target_channel_id
        )

    def _move_with_autofocus_toggle(self, position_id: int) -> None:
        """
        Move using software focus and autofocus reinitialisation for channel changes.

        Parameters
        ----------
        position_id
            Registered target position ID.

        Returns
        -------
        None
        """
        autofocus = self._require_autofocus()
        autofocus.unlock()
        self.stage.move(target=self._position_states[position_id].coordinate.copy(), block=True)
        software_focus_status = self._run_software_focus(position_id=position_id)
        if software_focus_status != FocusStatusType.IN_FOCUS:
            raise RuntimeError(
                f"FocusNavigator.move: software focus failed with status {software_focus_status}."
            )
        self._sleep_func(self.config.post_move_wait_s)
        self._initialise_and_lock_autofocus()
        self._sleep_func(self.config.post_autofocus_wait_s)
        self._record_current_z(position_id=position_id, is_locked=autofocus.is_locked())

    def _recover_with_previous_position(self, original_position_id: int) -> None:
        """
        Recover autofocus by reinitialising at the previous registered position.

        Parameters
        ----------
        original_position_id
            Position ID where lock loss was detected.

        Returns
        -------
        None
        """
        previous_position_id = self._previous_position_id(position_id=original_position_id)
        self.stage.move(target=self._position_states[previous_position_id].coordinate.copy(), block=True)
        self._initialise_and_lock_autofocus()
        self._sleep_func(self.config.post_autofocus_wait_s)
        self.stage.move(target=original_position_id, block=True)
        self._current_position_id = original_position_id
        self._update_recorded_z(position_id=original_position_id)

    def _recover_with_software_focus(
            self,
            original_position_id: int,
            refocus_on_all_positions: bool,
    ) -> FocusStatusType:
        """
        Recover focus by running software focus on one or more positions.

        Parameters
        ----------
        original_position_id
            Position ID where lock loss was detected.
        refocus_on_all_positions
            If True, try each registered position until software focus succeeds.

        Returns
        -------
        FocusStatusType
            Software focus status from the successful recovery run.
        """
        candidate_position_id = original_position_id
        attempts = len(self._position_order) if refocus_on_all_positions else 1
        last_status = FocusStatusType.UNKNOWN
        for _ in range(attempts):
            self.stage.move(target=self._position_states[candidate_position_id].coordinate.copy(), block=True)
            last_status = self._run_software_focus(position_id=candidate_position_id)
            if last_status == FocusStatusType.IN_FOCUS:
                self._initialise_and_lock_autofocus()
                self._update_recorded_z(position_id=candidate_position_id)
                if candidate_position_id != original_position_id:
                    self.stage.move(target=original_position_id, block=True)
                    self._current_position_id = original_position_id
                return last_status
            candidate_position_id = self.get_next_position_id(position_id=candidate_position_id)
        raise RuntimeError(
            f"FocusNavigator.manage_focus: software focus failed with status {last_status}."
        )

    def _run_software_focus(self, position_id: int) -> FocusStatusType:
        """
        Run software focus for one position and return its status.

        Parameters
        ----------
        position_id
            Registered position ID passed to SoftwareFocus.run.

        Returns
        -------
        FocusStatusType
            Focus status from the software focus result.
        """
        software_focus = self._require_software_focus()
        result = software_focus.run(position_id=position_id)
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

    def _previous_position_id(self, position_id: int) -> int:
        """
        Return the previous registered position ID, wrapping at the start.

        Parameters
        ----------
        position_id
            Current registered position ID.

        Returns
        -------
        int
            Previous registered position ID.
        """
        self._require_position(position_id=position_id)
        index = self._position_order.index(position_id)
        return self._position_order[index - 1]

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

    def _record_current_z(self, position_id: int, is_locked: bool) -> None:
        """
        Read current Z, update stored coordinate, and append time-series state.

        Parameters
        ----------
        position_id
            Registered position ID to update.
        is_locked
            Autofocus lock state to store with the Z record.

        Returns
        -------
        None
        """
        coordinate = self._update_recorded_z(position_id=position_id)
        self._position_states[position_id].z_time_series.append(
            (coordinate.z, self._time_func(), is_locked)
        )

    def _update_recorded_z(self, position_id: int) -> Coordinate:
        """
        Update one stored position coordinate from the stage's current Z.

        Parameters
        ----------
        position_id
            Registered position ID to update.

        Returns
        -------
        Coordinate
            Updated stored coordinate.
        """
        current_coordinate = self.stage.get_coordinates(query_hardware=True)
        state_coordinate = self._position_states[position_id].coordinate.copy()
        state_coordinate.z = current_coordinate.z
        self._position_states[position_id].coordinate = state_coordinate
        return state_coordinate

    def _result(
            self,
            position_id: int,
            is_locked: bool,
            refocusing: bool,
            software_focus_status: FocusStatusType,
            max_refocus_trials_reached: bool = False,
    ) -> FocusNavigatorResult:
        """
        Build a FocusNavigatorResult for the current stored position state.

        Parameters
        ----------
        position_id
            Registered position ID for the result.
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
            position_id=position_id,
            coordinate=self._position_states[position_id].coordinate.copy(),
            is_locked=is_locked,
            refocusing=refocusing,
            software_focus_status=software_focus_status,
            max_refocus_trials_reached=max_refocus_trials_reached,
        )

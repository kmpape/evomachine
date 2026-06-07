"""Tests for focus navigation across stage, autofocus, and software focus."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evomachine.navigation import (
    FovConfig,
    FocusNavigator,
    FocusNavigatorConfig,
    FocusNavigatorFovRecord,
)
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.peripherals.autofocus import Autofocus
from evomachine.peripherals.stage import Stage
from evomachine.softwarefocus import SoftwareFocusConfigFactory
from evomachine.types import AutoFocusStatusType, FocusStatusType
from evomachine.utils import validate_dataclass_fields


class FakeStage(Stage):
    """In-memory Stage used by FocusNavigator tests."""

    def __init__(self, coordinate: Coordinate | None = None):
        """
        Initialise a fake stage.

        Parameters
        ----------
        coordinate
            Optional initial coordinate. Coordinate(0, 0, 0) is used when omitted.

        Returns
        -------
        None
        """
        self.coordinate = coordinate.copy() if coordinate else Coordinate(0, 0, 0)
        self.moves: list[Coordinate | int] = []
        super().__init__(
            name="Fake Stage",
            fov_step_size=100,
            coordinate_bounds=CoordinateBounds(
                low=Coordinate(-1e6, -1e6, -1e6),
                high=Coordinate(1e6, 1e6, 1e6),
            ),
            check_initialised=False,
            check_alive=False,
        )

    def _initialise(self, force: bool = False) -> bool:
        """
        Report fake initialisation success.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the fake stage.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return fake liveness.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _get_coordinates(self) -> Coordinate:
        """
        Return the current fake coordinate.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Current coordinate copy.
        """
        return self.coordinate.copy()

    def _get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Return broad fake stage limits.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Lower and upper fake coordinate limits.
        """
        return Coordinate(-1e6, -1e6, -1e6), Coordinate(1e6, 1e6, 1e6)

    def _move(self, coordinate: Coordinate, block: bool = True) -> Coordinate:
        """
        Merge a target coordinate into the fake coordinate.

        Parameters
        ----------
        coordinate
            Full or partial target coordinate.
        block
            Accepted for API compatibility.

        Returns
        -------
        Coordinate
            Updated fake coordinate.
        """
        self.moves.append(coordinate.copy())
        self.coordinate = self.coordinate.merge(update=coordinate)
        return self.coordinate.copy()

    def _home(self, block: bool = False) -> Coordinate:
        """
        Move the fake stage home.

        Parameters
        ----------
        block
            Accepted for API compatibility.

        Returns
        -------
        Coordinate
            Coordinate(0, 0, 0).
        """
        self.coordinate = Coordinate(0, 0, 0)
        return self.coordinate.copy()

    def halt(self) -> None:
        """
        Stop fake stage motion.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def _zero_coordinates(self) -> Coordinate:
        """
        Zero the fake stage coordinate.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Coordinate(0, 0, 0).
        """
        self.coordinate = Coordinate(0, 0, 0)
        return self.coordinate.copy()


class FakeAutofocus(Autofocus):
    """Autofocus fake that records lock and initialise commands."""

    def __init__(
            self,
            locked: bool = True,
            status: AutoFocusStatusType = AutoFocusStatusType.IN_FOCUS,
            initialise_success: bool = True,
    ):
        """
        Initialise fake autofocus state.

        Parameters
        ----------
        locked
            Initial lock state returned by is_locked().
        status
            Status returned by get_status().
        initialise_success
            Result returned by initialise_autofocus().

        Returns
        -------
        None
        """
        self.locked = locked
        self.status = status
        self.initialise_success = initialise_success
        self.history: list[str] = []
        self.configs: list[object] = []
        super().__init__(name="Fake Autofocus", check_initialised=False, check_alive=False)

    def _initialise(self, force: bool = False) -> bool:
        """
        Report fake initialisation success.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise fake autofocus.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return fake liveness.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _configure(self, config=None) -> bool:
        """
        Record a fake configure command.

        Parameters
        ----------
        config
            Accepted for API compatibility.

        Returns
        -------
        bool
            Always True.
        """
        self.history.append("configure")
        return True

    def _initialise_autofocus(self, config=None, lock_after_initialise: bool = False) -> bool:
        """
        Record a fake autofocus initialisation command.

        Parameters
        ----------
        config
            Optional config passed by the navigator.
        lock_after_initialise
            If True and initialisation succeeds, set locked to True.

        Returns
        -------
        bool
            Configured initialisation result.
        """
        self.history.append(f"initialise:{config}")
        self.configs.append(config)
        if self.initialise_success and lock_after_initialise:
            self.locked = True
        return self.initialise_success

    def _lock(self) -> None:
        """
        Lock fake autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history.append("lock")
        self.locked = True
        self.status = AutoFocusStatusType.IN_FOCUS

    def _unlock(self) -> None:
        """
        Unlock fake autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history.append("unlock")
        self.locked = False

    def _disable(self) -> None:
        """
        Disable fake autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history.append("disable")
        self.locked = False
        self.status = AutoFocusStatusType.IDLE

    def _get_status(self) -> AutoFocusStatusType:
        """
        Return fake autofocus status.

        Parameters
        ----------
        None

        Returns
        -------
        AutoFocusStatusType
            Current fake status.
        """
        return self.status

    def _is_locked(self) -> bool:
        """
        Return fake lock state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Current fake lock state.
        """
        return self.locked


class FakeSoftwareFocus:
    """Software focus fake that records runs and optionally moves stage Z."""

    def __init__(
            self,
            stage: FakeStage,
            statuses: dict[int, FocusStatusType] | None = None,
            z_by_fov: dict[int, float | int] | None = None,
    ):
        """
        Initialise fake software focus state.

        Parameters
        ----------
        stage
            FakeStage whose Z coordinate is updated on successful runs.
        statuses
            Optional focus status by fov ID.
        z_by_fov
            Optional best Z value by fov ID.

        Returns
        -------
        None
        """
        self.stage = stage
        self.statuses = statuses or {}
        self.z_by_fov = z_by_fov or {}
        self.runs: list[int] = []
        self.initialised_fovs: list[int] = []
        self.initialised_fov_configs = None
        self.updated_configs: list[tuple[int, object]] = []

    def initialise_fovs(self, fov_ids: list[int], fov_configs=None) -> None:
        """
        Record initialised fov IDs.

        Parameters
        ----------
        fov_ids
            FoV IDs supplied by FocusNavigator.
        fov_configs
            Optional software focus configs supplied by FocusNavigator.

        Returns
        -------
        None
        """
        self.initialised_fovs = list(fov_ids)
        self.initialised_fov_configs = fov_configs

    def run(self, fov_id: int):
        """
        Record a software focus run and return a focus status.

        Parameters
        ----------
        fov_id
            FoV ID being focused.

        Returns
        -------
        SimpleNamespace
            Object exposing focus_status.
        """
        self.runs.append(fov_id)
        status = self.statuses.get(fov_id, FocusStatusType.IN_FOCUS)
        if status == FocusStatusType.IN_FOCUS and fov_id in self.z_by_fov:
            self.stage.move(target=Coordinate(None, None, self.z_by_fov[fov_id]), block=True)
        return SimpleNamespace(focus_status=status)

    def update_config(self, config, fov_id: int | None = None) -> None:
        """
        Record one config update.

        Parameters
        ----------
        config
            Replacement software focus config.
        fov_id
            Optional FoV ID for the replacement config.

        Returns
        -------
        None
        """
        self.updated_configs.append((fov_id, config))


def _navigator(
        config: FocusNavigatorConfig | None = None,
        autofocus: FakeAutofocus | None = None,
        software_focus: FakeSoftwareFocus | None = None,
        stage: FakeStage | None = None,
) -> tuple[FocusNavigator, FakeStage, FakeAutofocus, FakeSoftwareFocus, list[float]]:
    """
    Return a FocusNavigator with fake peripherals.

    Parameters
    ----------
    config
        Optional navigator config.
    autofocus
        Optional fake autofocus.
    software_focus
        Optional fake software focus.
    stage
        Optional fake stage.

    Returns
    -------
    tuple[FocusNavigator, FakeStage, FakeAutofocus, FakeSoftwareFocus, list[float]]
        Navigator, fakes, and recorded sleep durations.
    """
    stage = stage or FakeStage()
    autofocus = autofocus or FakeAutofocus()
    software_focus = software_focus or FakeSoftwareFocus(stage=stage)
    sleeps: list[float] = []
    navigator = FocusNavigator(
        stage=stage,
        autofocus=autofocus,
        software_focus=software_focus,
        config=config or FocusNavigatorConfig(),
        sleep_func=sleeps.append,
        time_func=lambda: 123.0,
    )
    return navigator, stage, autofocus, software_focus, sleeps


def _fovs() -> dict[int, Coordinate]:
    """
    Return deterministic focus navigator fovs.

    Parameters
    ----------
    None

    Returns
    -------
    dict[int, Coordinate]
        Position coordinates keyed by fov ID.
    """
    return {
        0: Coordinate(0, 0, 10, channel_id=0),
        1: Coordinate(100, 0, 20, channel_id=0),
        2: Coordinate(200, 0, 30, channel_id=1),
    }


def test_focus_navigator_config_validation() -> None:
    """
    Check FocusNavigatorConfig validates field types and ranges.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    assert FocusNavigatorConfig().max_refocus_trials == 10
    assert isinstance(FocusNavigatorConfig().default_fov_config, FovConfig)
    assert isinstance(
        FocusNavigatorFovRecord(
            fov_id=0,
            coordinate=Coordinate(0, 0, 0),
            fov_config=FovConfig(),
        ),
        FocusNavigatorFovRecord,
    )
    with pytest.raises(TypeError):
        FocusNavigatorConfig(use_autofocus="yes")
    with pytest.raises(ValueError):
        FocusNavigatorConfig(max_refocus_trials=0)
    with pytest.raises(ValueError):
        FocusNavigatorConfig(out_of_focus_wait_s=-1)


def test_validate_dataclass_fields_accepts_lists_and_none() -> None:
    """
    Check shared dataclass validation accepts list type options and explicit None.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    validate_dataclass_fields(
        dataclass_name="ExampleConfig",
        checks=[
            ("hello", [str, int]),
            (None, [str, None]),
            (1.0, (int, float)),
        ],
    )
    with pytest.raises(TypeError, match="argument 0"):
        validate_dataclass_fields(
            dataclass_name="ExampleConfig",
            checks=[(None, [str])],
        )
    with pytest.raises(TypeError, match="str \\| None") as exc_info:
        validate_dataclass_fields(
            dataclass_name="ExampleConfig",
            checks=[(1, [str, None])],
        )
    assert "received int" in str(exc_info.value)


def test_initialise_fovs_registers_xy_only_when_using_autofocus() -> None:
    """
    Check autofocus mode stores full coordinates but registers XY-only fovs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, stage, _, software_focus, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))

    navigator.initialise_fovs(_fovs())

    assert stage._fov_id_to_coordinate[0] == Coordinate(0, 0, None, channel_id=0)
    assert navigator.get_fov_state(0).coordinate == Coordinate(0, 0, 10, channel_id=0)
    assert software_focus.initialised_fovs == [0, 1, 2]


def test_initialise_fovs_registers_full_coordinates_without_autofocus() -> None:
    """
    Check non-autofocus mode registers full fov coordinates.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, stage, _, _, _ = _navigator(FocusNavigatorConfig(use_autofocus=False))

    navigator.initialise_fovs(_fovs())

    assert stage._fov_id_to_coordinate[0] == Coordinate(0, 0, 10, channel_id=0)


def test_move_direct_without_autofocus_toggle() -> None:
    """
    Check movement without channel change uses registered stage fov directly.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, stage, autofocus, software_focus, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))
    navigator.initialise_fovs(_fovs())

    result = navigator.move(0, manage_focus=False)

    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert result.fov_id == 0
    assert result.is_locked
    assert autofocus.history == []
    assert software_focus.runs == []


def test_channel_change_move_unlocks_focuses_locks_and_records_z() -> None:
    """
    Check channel-change movement toggles autofocus and records focused Z.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stage = FakeStage()
    software_focus = FakeSoftwareFocus(stage=stage, z_by_fov={2: 42})
    navigator, _, autofocus, _, sleeps = _navigator(
        FocusNavigatorConfig(use_autofocus=True, post_move_wait_s=0.5, post_autofocus_wait_s=0.75),
        software_focus=software_focus,
        stage=stage,
    )
    navigator.initialise_fovs(_fovs())
    navigator.move(0, manage_focus=False)

    result = navigator.move(2, manage_focus=False)

    assert software_focus.runs == [2]
    assert autofocus.history == ["unlock", "initialise:None", "lock"]
    assert sleeps == [0.5, 0.75]
    assert navigator.get_fov_state(2).coordinate.z == 42
    assert result.is_locked


def test_initialise_fovs_passes_per_fov_software_focus_configs() -> None:
    """
    Check fov-specific software focus configs are passed to SoftwareFocus initialisation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    software_focus_config = SoftwareFocusConfigFactory.default_config()
    navigator, _, _, software_focus, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))
    fov_config = FovConfig(software_focus_config=software_focus_config)

    navigator.initialise_fovs(_fovs(), fov_configs={1: fov_config})

    assert software_focus.initialised_fov_configs == {1: software_focus_config}


def test_update_fov_config_stores_policy_and_updates_software_focus_config() -> None:
    """
    Check runtime FoV config updates are stored and propagated to SoftwareFocus.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    software_focus_config = SoftwareFocusConfigFactory.default_config()
    navigator, _, _, software_focus, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))
    navigator.initialise_fovs(_fovs())
    fov_config = FovConfig(
        lock_autofocus_on_fov=False,
        software_focus_config=software_focus_config,
    )

    record = navigator.update_fov_config(fov_id=1, fov_config=fov_config)

    assert record.fov_config is fov_config
    assert navigator.get_fov_state(1).fov_config is fov_config
    assert software_focus.updated_configs == [(1, software_focus_config)]


def test_update_fov_config_rejects_unknown_fov_and_bad_config() -> None:
    """
    Check runtime FoV config updates validate FoV IDs and config types.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, _, _, _, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))
    navigator.initialise_fovs(_fovs())

    with pytest.raises(KeyError, match="unknown fov ID 99"):
        navigator.update_fov_config(fov_id=99, fov_config=FovConfig())

    with pytest.raises(TypeError, match="fov_config must be FovConfig"):
        navigator.update_fov_config(fov_id=1, fov_config="bad")


def test_per_fov_autofocus_config_reinitialises_on_config_change() -> None:
    """
    Check moving between fovs with different autofocus configs reinitialises autofocus.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, _, autofocus, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, post_autofocus_wait_s=0),
    )
    navigator.initialise_fovs(
        _fovs(),
        fov_configs={
            0: FovConfig(autofocus_initialise_config="cfg-a"),
            1: FovConfig(autofocus_initialise_config="cfg-a"),
            2: FovConfig(autofocus_initialise_config="cfg-b"),
        },
    )

    navigator.move(0, manage_focus=False)
    navigator.move(1, manage_focus=False)
    navigator.move(2, manage_focus=False)

    assert autofocus.configs == ["cfg-a", "cfg-b"]


def test_arrival_software_focus_runs_before_autofocus_lock() -> None:
    """
    Check arrival software focus happens before per-fov autofocus reinitialisation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stage = FakeStage()
    software_focus = FakeSoftwareFocus(stage=stage, z_by_fov={1: 66})
    navigator, _, autofocus, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, post_autofocus_wait_s=0, post_move_wait_s=0),
        autofocus=FakeAutofocus(locked=False),
        software_focus=software_focus,
        stage=stage,
    )
    navigator.initialise_fovs(
        _fovs(),
        fov_configs={
            1: FovConfig(
                run_software_focus_on_arrival=True,
                autofocus_initialise_config="arrival",
            ),
        },
    )

    result = navigator.move(1, manage_focus=False)

    assert software_focus.runs == [1]
    assert autofocus.history == ["initialise:arrival", "lock"]
    assert result.is_locked
    assert navigator.get_fov_state(1).coordinate.z == 66


def test_locked_autofocus_records_z_without_refocusing() -> None:
    """
    Check locked autofocus records current Z and does not run recovery.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, stage, autofocus, software_focus, _ = _navigator(FocusNavigatorConfig(use_autofocus=True))
    navigator.initialise_fovs(_fovs())
    stage.move(Coordinate(None, None, 55), block=True)

    result = navigator.manage_focus(0)

    assert result.is_locked
    assert not result.refocusing
    assert navigator.get_fov_state(0).z_time_series == [(55, 123.0, True)]
    assert autofocus.history == []
    assert software_focus.runs == []


def test_out_of_focus_status_waits_before_lock_check() -> None:
    """
    Check OUT_OF_FOCUS status triggers the configured wait.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    autofocus = FakeAutofocus(locked=True, status=AutoFocusStatusType.OUT_OF_FOCUS)
    navigator, _, _, _, sleeps = _navigator(
        FocusNavigatorConfig(use_autofocus=True, out_of_focus_wait_s=2.5),
        autofocus=autofocus,
    )
    navigator.initialise_fovs(_fovs())

    navigator.manage_focus(0)

    assert sleeps == [2.5]


def test_lost_lock_raises_when_refocus_disabled() -> None:
    """
    Check lost autofocus lock raises when refocus is disabled.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, _, _, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, refocus=False),
        autofocus=FakeAutofocus(locked=False),
    )
    navigator.initialise_fovs(_fovs())

    with pytest.raises(RuntimeError, match="refocus is disabled"):
        navigator.manage_focus(0)


def test_max_refocus_trials_returns_skipped_record_at_configured_limit() -> None:
    """
    Check max_refocus_trials returns a skipped fov record.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, _, _, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, max_refocus_trials=1),
        autofocus=FakeAutofocus(locked=False),
        software_focus=FakeSoftwareFocus(stage=FakeStage(), statuses={0: FocusStatusType.BAD_FOCUS_CURVE}),
    )
    navigator.initialise_fovs(_fovs())
    result = navigator.manage_focus(0)

    assert result.skipped
    assert result.max_refocus_trials_reached
    assert result.refocusing


def test_recovery_via_previous_fov_without_software_focus() -> None:
    """
    Check non-software recovery reinitialises at previous fov and moves back.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    navigator, stage, autofocus, software_focus, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, refocus_using_software_focus=False),
        autofocus=FakeAutofocus(locked=False),
    )
    navigator.initialise_fovs(_fovs())

    result = navigator.manage_focus(1)

    assert autofocus.history == ["unlock", "initialise:None", "lock"]
    assert software_focus.runs == []
    assert stage.get_coordinates(query_hardware=False).x == 100
    assert result.refocusing
    assert result.is_locked


def test_recovery_via_software_focus_on_current_fov() -> None:
    """
    Check software focus recovery succeeds on the current fov.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stage = FakeStage()
    software_focus = FakeSoftwareFocus(stage=stage, z_by_fov={0: 77})
    navigator, _, autofocus, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True),
        autofocus=FakeAutofocus(locked=False),
        software_focus=software_focus,
        stage=stage,
    )
    navigator.initialise_fovs(_fovs())

    result = navigator.manage_focus(0)

    assert software_focus.runs == [0]
    assert autofocus.history == ["unlock", "initialise:None", "lock"]
    assert navigator.get_fov_state(0).coordinate.z == 77
    assert result.software_focus_status == FocusStatusType.IN_FOCUS


def test_recovery_via_software_focus_across_all_fovs_moves_back() -> None:
    """
    Check all-fov recovery tries fovs in order and returns to original.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stage = FakeStage()
    software_focus = FakeSoftwareFocus(
        stage=stage,
        statuses={0: FocusStatusType.BAD_FOCUS_CURVE, 1: FocusStatusType.IN_FOCUS},
        z_by_fov={1: 88},
    )
    navigator, _, _, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True, refocus_on_all_fovs=True),
        autofocus=FakeAutofocus(locked=False),
        software_focus=software_focus,
        stage=stage,
    )
    navigator.initialise_fovs(_fovs())

    result = navigator.manage_focus(0)

    assert software_focus.runs == [0, 1]
    assert navigator.get_fov_state(1).coordinate.z == 88
    assert stage.get_coordinates(query_hardware=False).x == 0
    assert result.refocusing


def test_failed_software_focus_raises() -> None:
    """
    Check failed software focus recovery raises RuntimeError.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stage = FakeStage()
    software_focus = FakeSoftwareFocus(stage=stage, statuses={0: FocusStatusType.BAD_FOCUS_CURVE})
    navigator, _, _, _, _ = _navigator(
        FocusNavigatorConfig(use_autofocus=True),
        autofocus=FakeAutofocus(locked=False),
        software_focus=software_focus,
        stage=stage,
    )
    navigator.initialise_fovs(_fovs())

    with pytest.raises(RuntimeError, match="software focus failed"):
        navigator.manage_focus(0)

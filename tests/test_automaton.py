from collections import deque
from multiprocessing import Event
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionManager
from evomachine.automaton import Automaton
from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.frame import Frame, FrameMetaData
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.coordinates import Coordinate
from evomachine.navigation import FocusNavigator, FovConfig
from evomachine.peripherals.dmd import (
    Dmd,
    DmdCalibrationConfig,
)
from evomachine.projection import ProjectionManager
from evomachine.strategy import AbstractStrategy, BasicStrategy
from evomachine.types import AutomatonCommandType, LEDType, UNKNOWN_FOV_ID
from evomachine.utils import normalise_frame


class FakePeripheral:
    """Peripheral fake that records lifecycle calls."""

    def __init__(self):
        """
        Initialise fake peripheral lifecycle state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.initialised = False
        self.initialise_count = 0
        self.stop_count = 0
        self.finalise_count = 0
        self.unlock_count = 0
        self.live_mode_history: list[bool] = []

    def initialise(self, force: bool = False) -> None:
        """
        Mark the fake peripheral initialised.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        self.initialise_count += 1
        self.initialised = True

    def is_initialised(self) -> bool:
        """
        Return fake initialisation state.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True after initialise().
        """
        return self.initialised

    def stop(self) -> None:
        """
        Record a stop call.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1

    def finalise(self, force: bool = False) -> None:
        """
        Record a finalise call and clear initialisation state.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        self.finalise_count += 1
        self.initialised = False

    def enable_live_mode(self) -> None:
        """Record live-mode enable."""
        self.live_mode_history.append(True)

    def disable_live_mode(self) -> None:
        """Record live-mode disable."""
        self.live_mode_history.append(False)

    def unlock(self) -> None:
        """Record an unlock command."""
        self.unlock_count += 1


class FakeLedManager(FakePeripheral):
    """LED manager fake that records illumination commands."""

    def __init__(self):
        """Initialise LED command recording."""
        super().__init__()
        self.set_calls: list[tuple[LEDType, float | int, float | None]] = []
        self.disable_count = 0

    def set_led(self, led_type: LEDType, brightness: float | int, duration: float | None = None) -> None:
        """
        Record one LED set command.

        Parameters
        ----------
        led_type
            LED type to set.
        brightness
            Requested brightness.
        duration
            Optional duration.

        Returns
        -------
        None
        """
        self.set_calls.append((led_type, brightness, duration))

    def disable_led(self, led_type: LEDType | None = None) -> None:
        """
        Record one LED disable command.

        Parameters
        ----------
        led_type
            Optional LED type to disable.

        Returns
        -------
        None
        """
        self.disable_count += 1


class FakeDmd(Dmd):
    """DMD fake for automaton tests."""

    def __init__(self):
        """Initialise fake DMD state."""
        self.initialised = False
        self.initialise_count = 0
        self.finalise_count = 0
        self.images: list[np.ndarray] = []
        self.none_count = 0

    def initialise(self, force: bool = False) -> None:
        """Mark fake DMD initialised."""
        self.initialise_count += 1
        self.initialised = True

    def is_initialised(self) -> bool:
        """Return fake initialisation state."""
        return self.initialised

    def is_alive(self) -> bool:
        """Return fake liveness."""
        return True

    def stop(self) -> None:
        """Record a DMD stop call."""
        self.display_none()

    def finalise(self, force: bool = False) -> None:
        """Mark fake DMD finalised."""
        self.finalise_count += 1
        self.initialised = False

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """Record one displayed image."""
        self.images.append(img.copy())

    def display_none(self) -> None:
        """Record one blank display command."""
        self.none_count += 1

    def pattern_from_roi_boxes(self, **kwargs) -> np.ndarray:
        """Return a deterministic ROI projection pattern."""
        return np.ones((2, 2), dtype=np.uint8) * 255


class FakeAcquisitionManager(FrameAcquisitionManager):
    """Frame acquisition manager fake that returns deterministic frames."""

    def __init__(
            self,
            camera: FakePeripheral | None = None,
            led_manager: FakeLedManager | None = None,
            stage: FakePeripheral | None = None,
            dmd: FakeDmd | None = None,
            filter_wheel: FakePeripheral | None = None,
    ):
        """Initialise fake acquisition state."""
        self.camera = camera if camera is not None else FakePeripheral()
        self.led_manager = led_manager if led_manager is not None else FakeLedManager()
        self.stage = stage
        self.dmd = dmd
        self.filter_wheel = filter_wheel
        self.calls: list[tuple[FrameMetaData | list[FrameMetaData], object]] = []
        self.stop_count = 0

    def take_frame(self, frame_metadata, settings=None) -> Frame:
        """
        Record metadata and return a deterministic frame stack.

        Parameters
        ----------
        frame_metadata
            Frame metadata supplied by Automaton.
        settings
            Acquisition settings supplied by Automaton.

        Returns
        -------
        Frame
            Deterministic acquired frame stack.
        """
        self.calls.append((frame_metadata, settings))
        metadata_items = frame_metadata if isinstance(frame_metadata, list) else [frame_metadata]
        base = np.arange(12, dtype=np.uint16).reshape(3, 4)
        array = np.stack([base + index for index, _ in enumerate(metadata_items)])
        return Frame(frame_metadata=metadata_items, array=array, saved_paths=[None for _ in metadata_items])

    def stop(self) -> None:
        """Record one acquisition stop call."""
        self.stop_count += 1


class FakeFocusNavigator(FocusNavigator):
    """Focus navigator fake that records moves and fovs."""

    def __init__(
            self,
            stage: FakePeripheral | None = None,
            autofocus: FakePeripheral | None = None,
            software_focus: object | None = None,
    ):
        """Initialise fake focus navigator state."""
        self.stage = stage if stage is not None else FakePeripheral()
        self.autofocus = autofocus
        self.software_focus = software_focus
        self.moves: list[int] = []
        self.fovs: dict[int, Coordinate] = {}
        self.fov_order: list[int] = []
        self.current_fov_id: int = UNKNOWN_FOV_ID
        self.fov_configs = None
        self.updated_fov_configs: list[tuple[int, FovConfig]] = []
        self.skipped_fov_ids: set[int] = set()

    def initialise_fovs(self, fov_id_to_coordinate, use_autofocus=None, fov_configs=None) -> None:
        """Record fov initialisation."""
        self.fovs = fov_id_to_coordinate
        self.fov_order = list(fov_id_to_coordinate)
        self.current_fov_id = UNKNOWN_FOV_ID
        self.fov_configs = fov_configs

    def move(self, fov_id: int, manage_focus: bool = True):
        """Record one move request."""
        self.moves.append(fov_id)
        self.current_fov_id = fov_id
        return SimpleNamespace(
            fov_id=fov_id,
            manage_focus=manage_focus,
            skipped=fov_id in self.skipped_fov_ids,
            skip_reason="focus failed" if fov_id in self.skipped_fov_ids else None,
        )

    def update_fov_config(self, fov_id: int, fov_config: FovConfig):
        """Record one FoV config update request."""
        self.updated_fov_configs.append((fov_id, fov_config))
        return SimpleNamespace(fov_id=fov_id, fov_config=fov_config)

    def get_current_fov_id(self) -> int:
        """Return the fake current FOV ID."""
        return self.current_fov_id

    def get_next_fov_id(self, fov_id: int) -> int:
        """Return the next fake FOV ID."""
        if fov_id not in self.fov_order:
            raise KeyError(f"unknown fov ID {fov_id}")
        index = self.fov_order.index(fov_id)
        return self.fov_order[(index + 1) % len(self.fov_order)]


class FakeProjectionManager(ProjectionManager):
    """Projection manager fake that records calibration calls."""

    def __init__(
            self,
            camera: FakePeripheral | None = None,
            dmd: FakeDmd | None = None,
            led_manager: FakeLedManager | None = None,
            filter_wheel: FakePeripheral | None = None,
            photodiode: FakePeripheral | None = None,
    ):
        """Initialise fake projection state."""
        self.camera = camera
        self.dmd = dmd
        self.led_manager = led_manager
        self.filter_wheel = filter_wheel
        self.photodiode = photodiode
        self.calls: list[tuple[DmdCalibrationConfig, str | Path | None]] = []

    def dmd_calibrate(
            self,
            cfg: DmdCalibrationConfig,
            filename: str | Path | None = None,
            progress_callback=None,
    ):
        """Record one calibration request."""
        self.calls.append((cfg, filename))
        if progress_callback is not None:
            progress_callback(1.0, "complete")
        return None


class FakeSoftwareFocus:
    """Software-focus fake that records stop calls."""

    def __init__(self):
        """Initialise fake software-focus state."""
        self.stop_count = 0

    def stop(self) -> None:
        """Record one stop request."""
        self.stop_count += 1


class FakeFovProcessor:
    """FOV processor fake exposing ROI boxes."""

    def __init__(self):
        """Initialise deterministic ROI boxes."""
        self.roi_boxes = {0: object()}


class FakeStrategy(AbstractStrategy):
    """Strategy fake used by automaton tests."""

    def __init__(self, cfg):
        """Initialise fake strategy state."""
        super().__init__(cfg=cfg)
        self.initialise_count = 0
        self.callbacks = 0

    def _initialise(self) -> list[AutomatonCommand]:
        """Return no initial commands."""
        self.initialise_count += 1
        return []

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return all command types manually exercised by automaton tests."""
        return {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.UPDATE_FOV_CONFIG,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.PROJECT,
            AutomatonCommandType.PROJECT_ROI,
            AutomatonCommandType.WAIT,
            AutomatonCommandType.STOP,
            AutomatonCommandType.LIVE_MODE,
            AutomatonCommandType.SAVE_STATE,
        }

    def _callback(self, fov_id: int, data: list[AutomatonCommand], errors: list[Exception]) -> list[AutomatonCommand]:
        """Record one callback and return no commands."""
        self.callbacks += 1
        return []

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []


def make_cfg():
    """
    Return a small image processor config for automaton tests.

    Parameters
    ----------
    None

    Returns
    -------
    ImageProcessorConfig
        Test config.
    """
    return ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )


def make_automaton(**automaton_kwargs):
    """
    Return an automaton and its fake dependencies.

    Parameters
    ----------
    **automaton_kwargs
        Extra Automaton constructor arguments.

    Returns
    -------
    tuple
        Automaton and fake dependencies.
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    dmd = FakeDmd()
    photodiode = FakePeripheral()
    acquisition_manager = FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=dmd)
    focus_navigator = FakeFocusNavigator(stage=stage)
    projection_manager = FakeProjectionManager(
        camera=camera,
        dmd=dmd,
        led_manager=led_manager,
        photodiode=photodiode,
    )
    strategy = FakeStrategy(cfg=cfg)
    automaton = Automaton(
        acq_mngr=acquisition_manager,
        focus_nav=focus_navigator,
        strategy=strategy,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        proj_mngr=projection_manager,
        **automaton_kwargs,
    )
    automaton.initialise(fovs={0: Coordinate(0, 0, 0), 1: Coordinate(1, 0, 0)})
    return automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd


def test_automaton_constructor_validates_dependencies() -> None:
    """
    Check constructor rejects invalid manager dependencies.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()

    with pytest.raises(TypeError, match="acq_mngr"):
        Automaton(
            acq_mngr=object(),
            focus_nav=FakeFocusNavigator(),
            strategy=FakeStrategy(cfg=cfg),
            cfg_processor=cfg,
            start_strategy_event=Event(),
            stop_strategy_event=Event(),
            stop_event=Event(),
            shutdown_event=Event(),
        )


def test_automaton_direct_peripheral_attributes_are_removed() -> None:
    """
    Check Automaton does not expose duplicate direct peripheral ownership.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()

    for attribute_name in (
        "camera",
        "stage",
        "led_manager",
        "filter_wheel",
        "dmd",
        "autofocus",
        "photodiode",
        "acquisition_manager",
        "focus_navigator",
        "projection_manager",
    ):
        assert not hasattr(automaton, attribute_name)


def test_automaton_exposes_shorthand_managers_and_cached_refs() -> None:
    """
    Check manager shorthand attributes and cached private dependencies.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd = make_automaton()

    assert automaton.acq_mngr is acquisition_manager
    assert automaton.focus_nav is focus_navigator
    assert automaton.proj_mngr is projection_manager
    assert automaton._camera is acquisition_manager.camera
    assert automaton._stage is focus_navigator.stage
    assert automaton._led_mngr is led_manager
    assert automaton._filt_wheel is acquisition_manager.filter_wheel
    assert automaton._dmd is dmd
    assert automaton._photodiode is projection_manager.photodiode
    assert not hasattr(automaton, "_filter_wheel")
    assert not hasattr(automaton, "_software_focus")


def test_automaton_old_position_api_is_removed() -> None:
    """
    Check old automaton position API names are absent.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()

    assert not hasattr(automaton, "get_pos_id")
    assert not hasattr(automaton, "get_next_pos_id")
    assert not hasattr(automaton, "get_period")


def test_automaton_fov_id_is_unknown_before_first_move() -> None:
    """
    Check current FOV is not guessed during initialisation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()

    assert automaton.get_fov_id() == UNKNOWN_FOV_ID


def test_automaton_move_delegates_to_focus_navigator() -> None:
    """
    Check MOVE command uses FocusNavigator.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, focus_navigator, _, _, _ = make_automaton()

    command = CommandFactory(cfg=make_cfg()).command_move(fov_id=1)
    automaton.next_commands = [command]
    automaton._process()

    assert focus_navigator.moves == [1]
    assert automaton.get_fov_id() == 1
    assert command.command_data.fov_id == 1


def test_automaton_initialise_passes_strategy_fov_configs_to_focus_navigator() -> None:
    """
    Check strategy initial FoV configs are passed into focus navigator setup.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    dmd = FakeDmd()
    acquisition_manager = FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=dmd)
    focus_navigator = FakeFocusNavigator(stage=stage)
    projection_manager = FakeProjectionManager(camera=camera, dmd=dmd, led_manager=led_manager)
    strategy = FakeStrategy(cfg=cfg)
    fov_config = FovConfig(run_software_focus_on_arrival=True)
    strategy.fov_configs = {1: fov_config}
    automaton = Automaton(
        acq_mngr=acquisition_manager,
        focus_nav=focus_navigator,
        strategy=strategy,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        proj_mngr=projection_manager,
    )

    automaton.initialise(fovs={1: Coordinate(0, 0, 0)})

    assert focus_navigator.fov_configs == {1: fov_config}


def test_automaton_strategy_uses_acq_mngr_dmd() -> None:
    """
    Check strategy initialisation receives the acquisition-manager DMD.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()

    assert automaton._strategy.dmd is automaton.acq_mngr.dmd


def test_automaton_constructor_allows_missing_acq_dmd() -> None:
    """
    Check Automaton allows FrameAcquisitionManager.dmd to be None.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()

    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=None),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=None,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    assert automaton._dmd is None


def test_automaton_dmd_free_basic_strategy_initialises_and_starts() -> None:
    """
    Check DMD-free imaging strategies can initialise and start.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    cfg.preproc_enabled = True
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=None),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=BasicStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    automaton.initialise(fovs={0: Coordinate(0, 0, 0)})
    automaton.start_strategy()

    assert automaton.strategy_has_started()
    assert automaton._strategy.dmd is None


class MissingDmdStrategy(FakeStrategy):
    """Strategy fake that declares DMD projection support."""

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Declare DMD projection."""
        return {AutomatonCommandType.PROJECT}


class InvalidCommandDeclarationStrategy(FakeStrategy):
    """Strategy fake with invalid command declarations."""

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return invalid command declaration entries."""
        return {"bad"}


def test_automaton_dmd_strategy_fails_when_dmd_missing() -> None:
    """
    Check strategies declaring DMD commands fail before strategy execution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=None),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=MissingDmdStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    with pytest.raises(RuntimeError, match="PROJECT: DMD"):
        automaton.initialise(fovs={0: Coordinate(0, 0, 0)})


def test_automaton_invalid_strategy_command_declaration_raises() -> None:
    """
    Check strategy command declarations must use AutomatonCommandType entries.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=None),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=InvalidCommandDeclarationStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    with pytest.raises(TypeError, match="non-AutomatonCommandType"):
        automaton.initialise(fovs={0: Coordinate(0, 0, 0)})


def test_automaton_rejects_split_dmd_managers() -> None:
    """
    Check acquisition and projection managers cannot own different DMDs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()

    with pytest.raises(ValueError, match="proj_mngr.dmd"):
        Automaton(
            acq_mngr=FakeAcquisitionManager(
                camera=camera,
                led_manager=led_manager,
                stage=stage,
                dmd=FakeDmd(),
            ),
            focus_nav=FakeFocusNavigator(stage=stage),
            strategy=FakeStrategy(cfg=cfg),
            cfg_processor=cfg,
            start_strategy_event=Event(),
            stop_strategy_event=Event(),
            stop_event=Event(),
            shutdown_event=Event(),
            proj_mngr=FakeProjectionManager(camera=camera, dmd=FakeDmd(), led_manager=led_manager),
        )


def test_automaton_initialise_without_strategy_sets_up_fovs_only() -> None:
    """
    Check strategy-less construction still supports device and FoV setup.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    dmd = FakeDmd()
    focus_navigator = FakeFocusNavigator(stage=stage)
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=dmd),
        focus_nav=focus_navigator,
        strategy=None,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    automaton.initialise(fovs={1: Coordinate(0, 0, 0)})

    assert focus_navigator.fovs == {1: Coordinate(0, 0, 0)}
    assert automaton._strategy is None
    assert not automaton._strategy_is_initialised
    assert not automaton.is_initialised()
    assert automaton.get_strategy_name() is None


def test_automaton_start_strategy_requires_strategy() -> None:
    """
    Check strategy execution cannot start before a strategy is set.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    stage = FakePeripheral()
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(stage=stage, dmd=FakeDmd()),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=None,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    with pytest.raises(RuntimeError, match="strategy is required"):
        automaton.start_strategy()


def test_automaton_set_strategy_initialises_after_fovs() -> None:
    """
    Check set_strategy initialises immediately when FoVs already exist.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    stage = FakePeripheral()
    focus_navigator = FakeFocusNavigator(stage=stage)
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(stage=stage, dmd=FakeDmd()),
        focus_nav=focus_navigator,
        strategy=None,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )
    automaton.initialise(fovs={1: Coordinate(0, 0, 0)})
    strategy = FakeStrategy(cfg=cfg)
    fov_config = FovConfig(lock_autofocus_on_fov=False)
    strategy.fov_configs = {1: fov_config}

    automaton.set_strategy(strategy=strategy)

    assert automaton._strategy is strategy
    assert automaton._strategy_is_initialised
    assert strategy.initialise_count == 1
    assert focus_navigator.updated_fov_configs == [(1, fov_config)]
    assert automaton.get_strategy_name() == "FakeStrategy"


def test_automaton_set_strategy_rejects_after_start_event() -> None:
    """
    Check strategy replacement is blocked once strategy execution has started.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()
    original_strategy = automaton._strategy
    automaton.start_strategy()
    replacement = FakeStrategy(cfg=make_cfg())

    with pytest.raises(RuntimeError, match="start_strategy_event"):
        automaton.set_strategy(strategy=replacement)

    assert automaton._strategy is original_strategy


def test_automaton_update_fov_config_delegates_to_focus_navigator() -> None:
    """
    Check UPDATE_FOV_CONFIG commands update the focus navigator.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, focus_navigator, _, _, _ = make_automaton()
    fov_config = FovConfig(lock_autofocus_on_fov=False)

    command = CommandFactory(cfg=make_cfg()).command_update_fov_config(fov_id=1, fov_config=fov_config)
    automaton.next_commands = [command]
    automaton._process()

    assert focus_navigator.updated_fov_configs == [(1, fov_config)]
    assert command.command_data.fov_config is fov_config


def test_automaton_move_next_wraps_registered_fov_order() -> None:
    """
    Check MOVE -1 wraps according to registered FOV insertion order.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, focus_navigator, _, _, _ = make_automaton()
    automaton.initialise(fovs={2: Coordinate(2, 0, 0), 7: Coordinate(7, 0, 0)})

    automaton.next_commands = [
        CommandFactory(cfg=make_cfg()).command_move(fov_id=2),
        CommandFactory(cfg=make_cfg()).command_move(fov_id=-1),
    ]
    automaton._process()

    assert focus_navigator.moves[-2:] == [2, 7]
    assert automaton.get_fov_id() == 7


def test_automaton_image_uses_frame_metadata_and_updates_buffers() -> None:
    """
    Check IMAGE command delegates to FrameAcquisitionManager and stores frames.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, acquisition_manager, _, _, _, _ = make_automaton()
    metadata = FrameMetaData(
        frame_id=0,
        leds={LEDType.LED_450_NM: 20},
        filter_wheel=None,
        exposure=50,
        fov_id=0,
    )
    command = CommandFactory(cfg=make_cfg()).command_image(frame_metadata=metadata, segment=False, save=True)

    automaton.next_commands = [command]
    automaton._process()

    assert acquisition_manager.calls[0][0] is metadata
    assert acquisition_manager.calls[0][1].save is True
    assert command.command_data["frame_metadata"] == [metadata]
    assert automaton.get_frame(0, LEDType.LED_450_NM, time_id=0).shape == (3, 4)
    assert isinstance(automaton._all_frames[0], deque)
    assert isinstance(automaton._all_frames[0][0], Frame)
    assert automaton._all_frames[0][0].array[0, 0, 1] == 1
    assert not hasattr(automaton, "_all_frames_raw")
    assert not hasattr(automaton, "_ref_frames")


def test_automaton_frame_buffers_use_fov_ids_and_time_zero_latest() -> None:
    """
    Check FOV frame buffers are keyed by fov ID and latest frames use time_id 0.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()
    automaton.initialise(fovs={2: Coordinate(2, 0, 0), 7: Coordinate(7, 0, 0)})
    first_metadata = FrameMetaData(
        frame_id=0,
        leds={LEDType.LED_450_NM: 20},
        filter_wheel=None,
        exposure=50,
        fov_id=7,
    )
    second_metadata = FrameMetaData(
        frame_id=1,
        leds={LEDType.LED_450_NM: 20},
        filter_wheel=None,
        exposure=50,
        fov_id=7,
    )
    older_frame = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.uint16,
    )
    newer_frame = np.array(
        [
            [0, 10, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 11],
        ],
        dtype=np.uint16,
    )
    frame = Frame(
        frame_metadata=[first_metadata, second_metadata],
        array=np.stack([older_frame, newer_frame]),
        saved_paths=[None, None],
    )

    automaton._store_frame(frame=frame)

    assert set(automaton._all_frames) == {7}
    assert isinstance(automaton._all_frames[7], deque)
    assert list(automaton._all_frames[7]) == [frame]
    np.testing.assert_allclose(automaton.get_frame(7, LEDType.LED_450_NM, time_id=0), normalise_frame(newer_frame))
    np.testing.assert_allclose(automaton.get_frame(7, LEDType.LED_450_NM, time_id=1), normalise_frame(older_frame))
    with pytest.raises(IndexError, match="time_id 2"):
        automaton.get_frame(7, LEDType.LED_450_NM, time_id=2)
    assert 2 not in automaton._all_frames


def test_automaton_frame_history_default_keeps_two_frames_per_fov() -> None:
    """
    Check default frame history keeps only the newest two Frame objects.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()
    stored_frames = []
    for frame_id in range(3):
        frame = Frame(
            frame_metadata=[
                FrameMetaData(
                    frame_id=frame_id,
                    leds={LEDType.LED_450_NM: 20},
                    filter_wheel=None,
                    exposure=50,
                    fov_id=0,
                )
            ],
            array=np.stack([np.arange(12, dtype=np.uint16).reshape(3, 4) + frame_id]),
        )
        stored_frames.append(frame)
        automaton._store_frame(frame=frame)

    assert list(automaton._all_frames[0]) == [stored_frames[2], stored_frames[1]]


def test_automaton_frame_history_none_keeps_all_frames_per_fov() -> None:
    """
    Check frame_history_limit=None keeps every stored Frame object.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton(frame_history_limit=None)
    stored_frames = []
    for frame_id in range(3):
        frame = Frame(
            frame_metadata=[
                FrameMetaData(
                    frame_id=frame_id,
                    leds={LEDType.LED_450_NM: 20},
                    filter_wheel=None,
                    exposure=50,
                    fov_id=0,
                )
            ],
            array=np.stack([np.arange(12, dtype=np.uint16).reshape(3, 4) + frame_id]),
        )
        stored_frames.append(frame)
        automaton._store_frame(frame=frame)

    assert list(automaton._all_frames[0]) == [stored_frames[2], stored_frames[1], stored_frames[0]]


def test_automaton_frame_history_limit_validates_constructor_value() -> None:
    """
    Check frame history limits are positive integers or None.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="frame_history_limit"):
        make_automaton(frame_history_limit=0)
    with pytest.raises(TypeError, match="frame_history_limit"):
        make_automaton(frame_history_limit=1.5)


def test_automaton_skipped_move_suppresses_image_acquisition() -> None:
    """
    Check a skipped focus move prevents imaging for that fov.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, acquisition_manager, focus_navigator, _, _, _ = make_automaton()
    focus_navigator.skipped_fov_ids.add(1)
    move_command = CommandFactory(cfg=make_cfg()).command_move(fov_id=1)
    image_command = CommandFactory(cfg=make_cfg()).command_image(
        frame_metadata=FrameMetaData(
            frame_id=0,
            leds={LEDType.LED_450_NM: 20},
            filter_wheel=None,
            exposure=50,
            fov_id=-1,
        ),
        segment=False,
        save=True,
    )

    automaton.next_commands = [move_command, image_command]
    automaton._process()

    assert acquisition_manager.calls == []
    assert image_command.command_data["skipped"]
    assert image_command.command_data["skip_reason"] == "focus failed"


def test_automaton_project_uses_dmd_and_led_manager() -> None:
    """
    Check PROJECT command displays an image and actuates the LED manager.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, _, _, led_manager, dmd = make_automaton()
    image = np.ones(DMD_WIDTH_HEIGHT, dtype=np.uint8)
    command = CommandFactory(cfg=make_cfg()).command_project(
        channel=LEDType.LED_450_NM,
        image=image,
        duration=0.001,
        brightness=10,
    )

    automaton.next_commands = [command]
    automaton._process()

    assert np.array_equal(dmd.images[0], image)
    assert led_manager.set_calls[0] == (LEDType.LED_450_NM, 10, 1.0)
    assert led_manager.disable_count == 1


def test_automaton_project_raises_when_dmd_missing() -> None:
    """
    Check runtime projection paths fail clearly when no DMD is configured.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=None),
        focus_nav=FakeFocusNavigator(stage=stage),
        strategy=None,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )
    image = np.ones(DMD_WIDTH_HEIGHT, dtype=np.uint8)
    command = CommandFactory(cfg=make_cfg()).command_project(
        channel=LEDType.LED_450_NM,
        image=image,
        duration=0.001,
        brightness=10,
    )

    with pytest.raises(RuntimeError, match="DMD"):
        automaton._execute_project(command=command)


def test_automaton_project_roi_uses_dict_backed_fov_processors() -> None:
    """
    Check PROJECT_ROI looks up FOV processors by FOV ID.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, _, _, led_manager, dmd = make_automaton()
    automaton._fov_processors = {7: FakeFovProcessor()}
    automaton._fov_to_roi = {7: [0]}
    factory = CommandFactory(cfg=make_cfg())
    factory.update_region_of_interests({7: [0]})
    command = factory.command_project_roi(
        channel=LEDType.LED_450_NM,
        fov_id=7,
        roi_ids=[0],
        duration=0.001,
        brightness=10,
    )

    automaton.next_commands = [command]
    automaton._process()

    assert dmd.images
    assert led_manager.disable_count == 1


def test_automaton_project_roi_raises_for_missing_fov_processor() -> None:
    """
    Check PROJECT_ROI raises when no processor is registered for the FOV ID.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()
    automaton._fov_to_roi = {7: [0]}
    factory = CommandFactory(cfg=make_cfg())
    factory.update_region_of_interests({7: [0]})
    command = factory.command_project_roi(
        channel=LEDType.LED_450_NM,
        fov_id=7,
        roi_ids=[0],
        duration=0.001,
        brightness=10,
    )

    with pytest.raises(KeyError, match="unknown fov ID 7"):
        automaton._execute_project_roi(command=command)


def test_automaton_dmd_calibrate_delegates_to_projection_manager() -> None:
    """
    Check DMD calibration is delegated to ProjectionManager.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, _, _, projection_manager, _, _ = make_automaton()
    cfg = DmdCalibrationConfig(
        channel=LEDType.LED_450_NM,
        brightness=10,
        exposure=50,
        line_width=1,
        step=1,
        delay=0,
        start_row=0,
        end_row=1,
        start_col=0,
        end_col=1,
        on_mothermachine=False,
    )

    result = automaton.dmd_calibrate(cfg=cfg, filename="calibration.pkl")

    assert projection_manager.calls == [(cfg, "calibration.pkl")]
    assert result is None



def test_automaton_initialises_unique_manager_devices_once() -> None:
    """
    Check shared manager-owned devices are initialised once by identity.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd = make_automaton()

    assert acquisition_manager.camera.initialise_count == 1
    assert focus_navigator.stage.initialise_count == 1
    assert led_manager.initialise_count == 1
    assert dmd.initialise_count == 1
    assert projection_manager.photodiode.initialise_count == 1


def test_automaton_act_on_halt_uses_manager_paths() -> None:
    """
    Check halt cleanup uses acquisition and focus managers.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    camera = FakePeripheral()
    stage = FakePeripheral()
    led_manager = FakeLedManager()
    dmd = FakeDmd()
    autofocus = FakePeripheral()
    software_focus = FakeSoftwareFocus()
    acquisition_manager = FakeAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=dmd)
    focus_navigator = FakeFocusNavigator(stage=stage, autofocus=autofocus, software_focus=software_focus)
    automaton = Automaton(
        acq_mngr=acquisition_manager,
        focus_nav=focus_navigator,
        strategy=FakeStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
    )

    automaton.act_on_halt()

    assert automaton._swfocus is software_focus
    assert acquisition_manager.stop_count == 1
    assert software_focus.stop_count == 1
    assert autofocus.unlock_count == 1


def test_automaton_stop_sets_event_and_halts_initialised_devices() -> None:
    automaton, acquisition_manager, *_deps = make_automaton()

    automaton.stop()

    assert automaton.stopped()
    assert acquisition_manager.stop_count == 1


def test_automaton_shutdown_stops_and_finalises_peripherals() -> None:
    """
    Check shutdown uses current peripheral stop/finalise APIs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd = make_automaton()

    automaton.shutdown()

    assert acquisition_manager.stop_count == 1
    assert acquisition_manager.camera.finalise_count == 1
    assert focus_navigator.stage.finalise_count == 1
    assert led_manager.finalise_count == 1
    assert dmd.finalise_count == 1
    assert projection_manager.photodiode.finalise_count == 1
    assert automaton.has_shutdown()


def test_automaton_shutdown_attempts_all_cleanup_after_partial_failures(monkeypatch) -> None:
    automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd = make_automaton()

    def fail_stop():
        raise RuntimeError("stop failed")

    def fail_dmd_finalise(force=False):
        dmd.finalise_count += 1
        raise RuntimeError("DMD finalise failed")

    monkeypatch.setattr(acquisition_manager, "stop", fail_stop)
    monkeypatch.setattr(dmd, "finalise", fail_dmd_finalise)

    with pytest.raises(RuntimeError, match="stop failed.*DMD finalise failed"):
        automaton.shutdown()

    assert dmd.finalise_count == 1
    assert acquisition_manager.camera.finalise_count == 1
    assert focus_navigator.stage.finalise_count == 1
    assert led_manager.finalise_count == 1
    assert projection_manager.photodiode.finalise_count == 1
    assert automaton.has_shutdown()


def test_automaton_run_services_bounded_gui_request_processor() -> None:
    """
    Check the typed GUI hook is called from the automaton loop with its budget.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()
    shutdown_event = Event()
    budgets: list[int] = []

    def process_gui_requests(max_jobs: int) -> None:
        budgets.append(max_jobs)
        shutdown_event.set()

    automaton = Automaton(
        acq_mngr=FakeAcquisitionManager(dmd=FakeDmd()),
        focus_nav=FakeFocusNavigator(),
        strategy=FakeStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=shutdown_event,
        gui_request_processor=process_gui_requests,
        gui_request_budget=3,
    )

    automaton.run()

    assert budgets == [3]


def test_automaton_gui_set_request_processor_updates_hook_and_budget() -> None:
    """
    Check the typed GUI hook can be installed after automaton construction.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    automaton, *_deps = make_automaton()
    budgets: list[int] = []

    def process_gui_requests(max_jobs: int) -> None:
        budgets.append(max_jobs)

    automaton.gui_set_request_processor(process_gui_requests, budget=5)
    automaton.gui_process_requests()

    assert budgets == [5]
    assert automaton.gui_request_processor is process_gui_requests
    assert automaton.gui_request_budget == 5

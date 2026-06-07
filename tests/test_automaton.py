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
from evomachine.peripherals.dmd import DmdCalibrationConfig
from evomachine.coordinates import Coordinate
from evomachine.navigation import FocusNavigator, FovConfig
from evomachine.peripherals.dmd import Dmd
from evomachine.projection import ProjectionManager
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType


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
        self.stop_count = 0
        self.finalise_count = 0

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
        self.images: list[np.ndarray] = []
        self.none_count = 0

    def initialise(self, force: bool = False) -> None:
        """Mark fake DMD initialised."""
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

    def __init__(self):
        """Initialise fake acquisition state."""
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

    def __init__(self):
        """Initialise fake focus navigator state."""
        self.moves: list[int] = []
        self.fovs: dict[int, Coordinate] = {}
        self.fov_configs = None
        self.updated_fov_configs: list[tuple[int, FovConfig]] = []
        self.skipped_fov_ids: set[int] = set()

    def initialise_fovs(self, fov_id_to_coordinate, use_autofocus=None, fov_configs=None) -> None:
        """Record fov initialisation."""
        self.fovs = fov_id_to_coordinate
        self.fov_configs = fov_configs

    def move(self, fov_id: int, manage_focus: bool = True):
        """Record one move request."""
        self.moves.append(fov_id)
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


class FakeProjectionManager(ProjectionManager):
    """Projection manager fake that records calibration calls."""

    def __init__(self):
        """Initialise fake projection state."""
        self.calls: list[tuple[DmdCalibrationConfig, str | None]] = []

    def dmd_calibrate(self, cfg: DmdCalibrationConfig, filename: str | Path | None = None):
        """Record one calibration request."""
        self.calls.append((cfg, filename))
        return [], np.eye(3), np.eye(3), Path("calibration.pkl")


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


def make_automaton():
    """
    Return an automaton and its fake dependencies.

    Parameters
    ----------
    None

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
    acquisition_manager = FakeAcquisitionManager()
    focus_navigator = FakeFocusNavigator()
    projection_manager = FakeProjectionManager()
    strategy = FakeStrategy(cfg=cfg)
    automaton = Automaton(
        camera=camera,
        stage=stage,
        led_manager=led_manager,
        acquisition_manager=acquisition_manager,
        focus_navigator=focus_navigator,
        strategy=strategy,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        dmd=dmd,
        projection_manager=projection_manager,
    )
    automaton.initialise(fovs={0: Coordinate(0, 0, 0), 1: Coordinate(1, 0, 0)})
    return automaton, acquisition_manager, focus_navigator, projection_manager, led_manager, dmd


def test_automaton_constructor_validates_dependencies() -> None:
    """
    Check constructor rejects dependencies missing required methods.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = make_cfg()

    with pytest.raises(TypeError):
        Automaton(
            camera=object(),
            stage=FakePeripheral(),
            led_manager=FakeLedManager(),
            acquisition_manager=FakeAcquisitionManager(),
            focus_navigator=FakeFocusNavigator(),
            strategy=FakeStrategy(cfg=cfg),
            cfg_processor=cfg,
            start_strategy_event=Event(),
            stop_strategy_event=Event(),
            stop_event=Event(),
            shutdown_event=Event(),
            dmd=FakeDmd(),
        )


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
    focus_navigator = FakeFocusNavigator()
    strategy = FakeStrategy(cfg=cfg)
    fov_config = FovConfig(run_software_focus_on_arrival=True)
    strategy.fov_configs = {1: fov_config}
    automaton = Automaton(
        camera=FakePeripheral(),
        stage=FakePeripheral(),
        led_manager=FakeLedManager(),
        acquisition_manager=FakeAcquisitionManager(),
        focus_navigator=focus_navigator,
        strategy=strategy,
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        dmd=FakeDmd(),
        projection_manager=FakeProjectionManager(),
    )

    automaton.initialise(fovs={1: Coordinate(0, 0, 0)})

    assert focus_navigator.fov_configs == {1: fov_config}


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

    automaton.next_commands = [CommandFactory(cfg=make_cfg()).command_move(fov_id=-1)]
    automaton._process()

    assert focus_navigator.moves[-1] == 7
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
    assert automaton._all_frames_raw[0][0, 0, 0, 1] == 1


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
    automaton, acquisition_manager, *_deps = make_automaton()
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

    automaton.next_commands = [
        CommandFactory(cfg=make_cfg()).command_image(
            frame_metadata=[first_metadata, second_metadata],
            segment=False,
            save=False,
        ),
    ]
    automaton._process()

    assert set(automaton._all_frames_raw) == {2, 7}
    assert len(acquisition_manager.calls) == 1
    assert automaton._all_frames_raw[7][0, 0, 0, 1] == 2
    assert automaton._all_frames_raw[7][1, 0, 0, 1] == 1
    automaton._all_frames[7][0, 0, 0, 1] = 10
    automaton._all_frames[7][1, 0, 0, 1] = 5
    assert automaton.get_frame(7, LEDType.LED_450_NM, time_id=0)[0, 1] == 10
    assert automaton.get_frame(7, LEDType.LED_450_NM, time_id=1)[0, 1] == 5
    assert 0 not in automaton._all_frames_raw


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
    assert result[3] == Path("calibration.pkl")


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
    automaton, acquisition_manager, _, _, led_manager, _ = make_automaton()

    automaton.shutdown()

    assert acquisition_manager.stop_count == 1
    assert led_manager.finalise_count == 1
    assert automaton.has_shutdown()

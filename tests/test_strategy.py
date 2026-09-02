import pickle

import numpy as np
import pytest

from evomachine.commands import AutomatonCommand
from evomachine.frame import FrameMetaData
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.coordinates import Coordinate
from evomachine.navigation import FovConfig
from evomachine.peripherals.dmd import Dmd
from evomachine.strategy import (
    AbstractStrategy,
    BasicStrategy,
    NoStrategy,
    create_strategy_from_definition,
    list_strategy_definitions,
)
from evomachine.types import AutomatonCommandType, LEDType


class FakeDmd(Dmd):
    """Minimal Dmd subclass used for strategy tests."""

    def __init__(self):
        """
        Initialise fake DMD state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.images: list[np.ndarray] = []

    def initialise(self, force: bool = False) -> None:
        """
        Accept initialisation calls.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        return

    def is_initialised(self) -> bool:
        """
        Return fake initialisation status.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return True

    def is_alive(self) -> bool:
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

    def finalise(self, force: bool = False) -> None:
        """
        Accept finalise calls.

        Parameters
        ----------
        force
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        return

    def stop(self) -> None:
        """
        Accept stop calls.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """
        Record displayed images.

        Parameters
        ----------
        img
            Image to record.
        _is_full_display
            Accepted for Dmd API compatibility.

        Returns
        -------
        None
        """
        self.images.append(img)


class InvalidStrategy(AbstractStrategy):
    """Strategy returning an invalid command list."""

    def _initialise(self) -> list[AutomatonCommand]:
        """
        Return an invalid command list.

        Parameters
        ----------
        None

        Returns
        -------
        list[AutomatonCommand]
            Invalid command list.
        """
        return "bad"

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """
        Return an empty command list.

        Parameters
        ----------
        fov_id
            Current FoV ID.
        data
            Previous command data.
        errors
            Previous errors.

        Returns
        -------
        list[AutomatonCommand]
            Empty command list.
        """
        return []

    def finalise(self) -> list[AutomatonCommand]:
        """
        Return final commands.

        Parameters
        ----------
        None

        Returns
        -------
        list[AutomatonCommand]
            Empty command list.
        """
        return []

    @staticmethod
    def is_valid_command_list(command_list: list[AutomatonCommand]) -> bool:
        """
        Reject every command list.

        Parameters
        ----------
        command_list
            Command list to validate.

        Returns
        -------
        bool
            Always False.
        """
        return False


def test_strategy_initialise_injects_dmd() -> None:
    """
    Check strategy initialisation stores the injected DMD.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    cfg.preproc_enabled = True
    strategy = BasicStrategy(
        cfg=cfg,
    )
    dmd = FakeDmd()

    commands = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=dmd,
    )

    assert strategy.dmd is dmd
    image_command = next(command for command in commands if command.command_type == AutomatonCommandType.IMAGE)
    assert isinstance(image_command.command_args["frame_metadata"], list)
    assert all(isinstance(metadata, FrameMetaData) for metadata in image_command.command_args["frame_metadata"])


def test_strategy_initialise_accepts_missing_dmd() -> None:
    """
    Check non-DMD strategies can initialise without a DMD object.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    cfg.preproc_enabled = True
    strategy = BasicStrategy(
        cfg=cfg,
    )

    commands = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )

    assert strategy.dmd is None
    assert any(command.command_type == AutomatonCommandType.IMAGE for command in commands)


def test_builtin_and_example_strategies_register_automaton_commands() -> None:
    """
    Check bundled strategies declare the command types they may emit.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    assert NoStrategy(cfg=cfg).register_automaton_commands() == set()
    assert BasicStrategy(cfg=cfg).register_automaton_commands() == {
        AutomatonCommandType.MOVE,
        AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
        AutomatonCommandType.LIVE_MODE,
    }
    expected_by_strategy = {
        "SimpleImagingStrategy": {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.WAIT,
        },
        "DmdProjectFullFovStrategy": {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.PROJECT,
            AutomatonCommandType.WAIT,
        },
        "DmdProjectByRoiStrategy": {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.PROJECT_ROI,
            AutomatonCommandType.WAIT,
        },
    }
    definitions = {definition.name: definition for definition in list_strategy_definitions()}
    for strategy_name, expected_commands in expected_by_strategy.items():
        strategy = create_strategy_from_definition(
            name=strategy_name,
            file_path=definitions[strategy_name].file_path,
            cfg=cfg,
        )
        assert strategy.register_automaton_commands() == expected_commands


def test_strategy_serialization_excludes_dmd() -> None:
    """
    Check pickled strategy state does not contain the DMD object.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    strategy = BasicStrategy(
        cfg=cfg,
    )
    strategy.dmd = FakeDmd()

    restored = pickle.loads(pickle.dumps(strategy))

    assert restored.dmd is None
    assert restored.fov_processors == {}


def test_invalid_strategy_command_list_raises_runtime_error() -> None:
    """
    Check invalid strategy command lists raise standard RuntimeError.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    strategy = InvalidStrategy(
        cfg=ImageProcessorConfigFactory.default_config(
            channels=[LEDType.LED_450_NM],
            channels_seg=[LEDType.LED_450_NM],
        )
    )

    with pytest.raises(RuntimeError):
        strategy.initialise(
            fovs={0: Coordinate(0, 0, 0)},
            region_of_interests={0: []},
            fov_processors={},
            dmd=FakeDmd(),
        )


def test_strategy_initial_fov_configs_validates_and_copies() -> None:
    """
    Check strategies expose validated per-FoV focus policies.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    cfg.preproc_enabled = True
    strategy = BasicStrategy(
        cfg=cfg,
    )
    fov_config = FovConfig(run_software_focus_on_arrival=True)
    strategy.initialise(
        fovs={1: Coordinate(0, 0, 0)},
        region_of_interests={1: []},
        fov_processors={},
        dmd=FakeDmd(),
    )
    strategy.fov_configs = {1: fov_config}

    initial_configs = strategy.initial_fov_configs()

    assert initial_configs == {1: fov_config}
    assert initial_configs is not strategy.fov_configs


def test_strategy_initial_fov_configs_rejects_unknown_fov() -> None:
    """
    Check strategy FoV focus policies must reference known FoV IDs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    cfg.preproc_enabled = True
    strategy = BasicStrategy(
        cfg=cfg,
    )
    strategy.initialise(
        fovs={1: Coordinate(0, 0, 0)},
        region_of_interests={1: []},
        fov_processors={},
        dmd=FakeDmd(),
    )
    strategy.fov_configs = {2: FovConfig()}

    with pytest.raises(KeyError, match="unknown fov ID 2"):
        strategy.initial_fov_configs()


def test_strategy_discovery_and_creation_use_name_file_pair() -> None:
    """
    Check strategy discovery returns definitions that can instantiate strategies.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    definitions = list_strategy_definitions()
    assert definitions
    definition = next(item for item in definitions if item.name == "SimpleImagingStrategy")
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM],
        channels_seg=[LEDType.LED_450_NM],
    )

    strategy = create_strategy_from_definition(
        name=definition.name,
        file_path=definition.file_path,
        cfg=cfg,
    )

    assert strategy.name() == "SimpleImagingStrategy"

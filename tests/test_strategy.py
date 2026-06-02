import pickle

import numpy as np
import pytest

from evomachine.commands import AutomatonCommand
from evomachine.frame import FrameMetaData
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.peripherals.camera import CameraSystemConfigFactory
from evomachine.coordinates import Coordinate
from evomachine.peripherals.dmd import Dmd
from evomachine.strategy import (
    AbstractStrategy,
    BasicStrategy,
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
        save_path=".",
    )
    dmd = FakeDmd()

    commands = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        config_camera=CameraSystemConfigFactory.default_air_config(),
        fov_processors=[],
        dmd=dmd,
    )

    assert strategy.dmd is dmd
    image_command = next(command for command in commands if command.command_type == AutomatonCommandType.IMAGE)
    assert isinstance(image_command.command_args["frame_metadata"], list)
    assert all(isinstance(metadata, FrameMetaData) for metadata in image_command.command_args["frame_metadata"])


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
    strategy = BasicStrategy(
        cfg=ImageProcessorConfigFactory.default_config(
            channels=[LEDType.LED_450_NM],
            channels_seg=[LEDType.LED_450_NM],
        ),
        save_path=".",
    )
    strategy.dmd = FakeDmd()

    restored = pickle.loads(pickle.dumps(strategy))

    assert restored.dmd is None
    assert restored.fov_processors == []


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
            config_camera=CameraSystemConfigFactory.default_air_config(),
            fov_processors=[],
            dmd=FakeDmd(),
        )


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

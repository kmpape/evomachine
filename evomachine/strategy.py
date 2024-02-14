from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

from evomachine.commands import AutomatonCommand, AutomatonCommandType, CommandFactory
from evomachine.config import ConfigLED, EVOMACHINE_DIR, get_logger
from evomachine.coordinates import Coordinate
from evomachine.exceptions import ErrorCode, EvoMachineError, StrategyError


logger = get_logger(name=__name__)


class AbstractStrategy(ABC):
    """
    TODO
    - Probably also needs a camera object? Unclear whether new command classes provide all what is needed.
    - callback should also take a list of EvoMachineError defined in evomachine.exceptions
    """

    def __init__(self):
        self.field_of_views: Dict[int, Coordinate] = {}
        "Dictionary indexed by fov_id with FoV coordinates."
        self.region_of_interests: Dict[int, List[int]] = {}
        "Dictionary indexed by fov_id with RoI IDs."
        self.path_to_save: Union[Path, None] = None
        "Path to save images. If None, Automaton will use path in ConfigDevice. If Path, extracted after initialise()."
        self.command_factory: CommandFactory = CommandFactory()
        "Factory object to create AutomatonCommands."


    def callback(
            self,
            fov_id: int,  # TODO need to consider several move commands passed or allow for one only
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        """Callback function for the strategy. This function is called by the\\
        automaton when new data is available.


        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `t` : int
            The time of the data.
        `data` : dict
            Processed image data such as cell positions.
        """
        new_command_list = self._callback(fov_id=fov_id, data=data, errors=errors)
        if not self.is_valid_command_list(new_command_list):
            raise StrategyError(message=f"AbstractStrategy.callback: invalid command list ({new_command_list}).",
                                error_code=ErrorCode.ERROR_STRATEGY)
        return new_command_list


    @abstractmethod
    def _callback(
            self,
            fov_id: int,  # TODO need to consider several move commands passed or allow for one only
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        """Callback function for the strategy. This function is called by the\\
        automaton when new data is available. 
        

        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `t` : int
            The time of the data.
        `data` : dict
            Processed image data such as cell positions.
        """
        pass

    @staticmethod
    def is_valid_command_list(command_list: List[AutomatonCommand]):
        # Max one MOVE command
        if len([cmd for cmd in command_list if cmd.command_type == AutomatonCommandType.MOVE]) > 1:
            return False

    def initialise(
            self,
            field_of_views: Dict[int, Coordinate],
            region_of_interests: Dict[int, List[int]],
    ) -> List[AutomatonCommand]:
        """Initialise the strategy.
        """
        new_command_list = self._initialise(field_of_views=field_of_views, region_of_interests=region_of_interests)
        if not self.is_valid_command_list(new_command_list):
            raise StrategyError(message=f"AbstractStrategy.callback: invalid command list ({new_command_list}).",
                                error_code=ErrorCode.ERROR_STRATEGY)
        return new_command_list

    @abstractmethod
    def _initialise(
            self,
            field_of_views: Dict[int, Coordinate],
            region_of_interests: Dict[int, List[int]],
    ) -> List[AutomatonCommand]:
        """Initialise the strategy.
        """
        pass


class NoStrategy(AbstractStrategy):
    """Strategy that does nothing.
    """

    def _callback(
            self,
            fov_id: int,
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        return []

    def _initialise(
            self,
            field_of_views: Dict[int, Coordinate],
            region_of_interests: Dict[int, List[int]],
    ) -> List[AutomatonCommand]:
        return []


class DummyStrategy(AbstractStrategy):
    """Dummy strategy for testing purposes.
    """
    def __init__(self):
        super().__init__()

        self.path_to_save = EVOMACHINE_DIR.parent / "images/dummy_strategy"

        # Define default commands
        self.default_move_command: AutomatonCommand = self.command_factory.command_move(fov_id=-1)
        self.default_image_command: AutomatonCommand = self.command_factory.command_image(
            channels=[ConfigLED.LED_450_NM, ConfigLED.LED_505_NM],
            exposure_time=1000,
            segment=False,
            save=True,
        )
        self.default_wait_command: AutomatonCommand = self.command_factory.command_wait(
            duration=1,
        )

        # Reset command IDs
        self.command_factory.reset()

        # Variable to keep track of commands
        self.last_commands: List[AutomatonCommand] = []

    def _initialise(
            self,
            field_of_views: Dict[int, Coordinate],
            region_of_interests: Dict[int, List[int]],
    ) -> List[AutomatonCommand]:
        self.field_of_views = field_of_views
        self.region_of_interests = region_of_interests

        # Create commands for first iteration
        self.last_commands = [
            self.command_factory.command_move(fov_id=0),
            self.command_factory.command_from_template(self.default_wait_command),
            self.command_factory.command_from_template(self.default_image_command),
        ]
        return self.last_commands

    def _callback(
            self,
            fov_id: int,
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:

        logger.info(f"FOV {fov_id} with data {data} and errors {errors}.")

        # Check received data
        for data_sent, data_recv in zip(self.last_commands, data):
            if not (data_sent.command_id == data_recv.command_id and data_sent.command_type == data_recv.command_type):
                logger.warning(f"Received {str(data_recv)} != {str(data_sent)}")

        # Define next commands
        self.last_commands = [
            self.command_factory.command_from_template(self.default_move_command),
            self.command_factory.command_from_template(self.default_wait_command),
            self.command_factory.command_from_template(self.default_image_command),
        ]
        return self.last_commands


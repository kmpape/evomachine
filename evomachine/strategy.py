from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import EVOMACHINE_DIR, get_logger, ConfigCamera
from evomachine.coordinates import Coordinate
from evomachine.exceptions import ErrorCode, EvoMachineError, StrategyError
from evomachine.evotypes import AutomatonCommandType, LEDType


logger = get_logger(name=__name__)


class AbstractStrategy(ABC):
    """
    Strategy class to be used by the automaton. This class is an abstract class that should be inherited by a specific
    strategy. Upon starting a strategy, the automaton calls

    - `initialise` to initialise the strategy,
    - `callback` repeatedly to get the next commands to be executed,
    - `finalise` to finalise the strategy, e.g., saving data.

    The children classes must implement _initialise, _callback and finalise methods.

    TODO strategy needs a test function to check if it is working properly before starting
    """

    def __init__(self):
        self.field_of_views: Dict[int, Coordinate] = {}
        "Dictionary indexed by fov_id with FoV coordinates."
        self.positions: Dict[int, List[int]] = {}
        "Dictionary indexed by fov_id with pos IDs."
        self.region_of_interests: Dict[int, List[int]] = {}
        "Dictionary indexed by pos_id with RoI IDs."
        self.path_to_save: Union[Path, None] = None
        "Path to save images. If None, Automaton will use path in ConfigDevice. If Path, extracted after initialise()."
        self.command_factory: CommandFactory = CommandFactory()
        "Factory object to create AutomatonCommands."
        self.config_camera: Union[None, ConfigCamera] = None
        "Camera configuration object."
        # TODO add configuration objects here as None. Can be overwritten by base strategies, then read by automaton

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
            fov_id: int,
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
        # TODO channel NO_LED should give false
        # TODO should only have one command_type per list (?)
        if len([cmd for cmd in command_list if cmd.command_type == AutomatonCommandType.MOVE]) > 1:
            return False
        else:
            return True

    def initialise(
            self,
            field_of_views: Dict[int, Coordinate],
            positions: Dict[int, List[int]],
            region_of_interests: Dict[int, List[int]],
            config_camera: ConfigCamera,
    ) -> List[AutomatonCommand]:
        """
        Initialise the strategy.

        Parameters
        ----------
        field_of_views : Dict[int, Coordinate]
            Dictionary with fov_id as key and Coordinate as value.
        positions: Dict[int, List[int]]
            Dictionary with fov_id as key and list of pos_id as value.
        region_of_interests: Dict[int, List[int]]
            Dictionary with pos_id as key and list of roi_id as value.
        config_camera: ConfigCamera
            Object defining camera configuration.
        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        self.field_of_views = field_of_views
        self.positions = positions
        self.region_of_interests = region_of_interests
        self.config_camera = config_camera
        new_command_list = self._initialise()
        if not self.is_valid_command_list(new_command_list):
            raise StrategyError(message=f"AbstractStrategy.callback: invalid command list ({new_command_list}).",
                                error_code=ErrorCode.ERROR_STRATEGY)
        return new_command_list

    @abstractmethod
    def finalise(self) -> List[AutomatonCommand]:
        pass

    @abstractmethod
    def _initialise(self) -> List[AutomatonCommand]:
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

    def _initialise(self) -> List[AutomatonCommand]:
        return []

    def finalise(self) -> List[AutomatonCommand]:
        return []


class BasicStrategy(AbstractStrategy):
    """Basic strategy for testing purposes.
    """
    def __init__(self):
        super().__init__()

        self.path_to_save = Path("/mnt/ImageData/Idris/2024-02-16")

        # Define default commands
        self.default_move_command: AutomatonCommand = self.command_factory.command_move(fov_id=-1)
        self.default_image_command: AutomatonCommand = self.command_factory.command_image(
            channels=[LEDType.LED_450_NM, LEDType.LED_505_NM],
            exposure_time=1000,
            segment=False,
            save=True,
        )
        self.default_wait_command: AutomatonCommand = self.command_factory.command_wait(
            duration=300,
        )

        # Reset command IDs
        self.command_factory.reset()

        # Variable to keep track of commands
        self.last_commands: List[AutomatonCommand] = []

    def _initialise(self) -> List[AutomatonCommand]:
        # Create commands for first iteration
        self.last_commands = [
            self.command_factory.command_move(fov_id=-1),
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

        logger.info("FOV {} with data =\n{} \nand errors = {}.".format(
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))

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

    def finalise(self) -> List[AutomatonCommand]:
        return []


def get_all_strategies() -> List[Tuple[Type[AbstractStrategy], str]]:
    subclasses = []
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and issubclass(obj, AbstractStrategy) and obj != AbstractStrategy and obj != NoStrategy:
            subclasses.append((obj, name))
    return subclasses

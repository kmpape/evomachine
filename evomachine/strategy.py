from abc import ABC, abstractmethod
import copy
import inspect
import importlib.util
import numpy as np
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import get_logger, ConfigCamera, ConfigCameraFactory, USE_DMD_SOCKET, ConfigImageProcessor
from evomachine.coordinates import Coordinate
from evomachine.exceptions import ConfigError, ErrorCode, EvoMachineError, StrategyError
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.utils import normalise_frame
from evomachine.coordinates import Coordinate
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl

from delta.rt import PositionRT

logger = get_logger(name=__name__)


class AbstractStrategy(ABC):
    """
    Strategy class to be used by the automaton. This class is an abstract class that should be inherited by a specific
    strategy. Upon starting a strategy, the automaton calls:

    - `initialise` to initialise the strategy,
    - `callback` repeatedly to get the next commands to be executed,
    - `finalise` to finalise the strategy, e.g., saving data.

    The children classes must implement _initialise, _callback and finalise methods. Use self.dmd to warp perspectives
    for pattern projections.

    TODO add functions to reconfigure camera object through GUI
    """

    def __init__(self, cfg: ConfigImageProcessor):
        self.callback_counter: int = 0
        "Incremented after each callback. First callback is = 0."
        self.cfg: ConfigImageProcessor = cfg
        "Image processor configuration object."
        self.field_of_views: dict[int, Coordinate] = {}
        "Dictionary indexed by fov_id with FoV coordinates."
        self.positions: dict[int, list[int]] = {}
        "Dictionary indexed by fov_id with pos IDs."
        self.region_of_interests: dict[int, list[int]] = {}
        "Dictionary indexed by pos_id with RoI IDs."
        self.pos_processors: list[PositionRT] = []
        "List of processors to access FoV data, RoI, cell lineages etc. Do not modify and treat as read-only."
        self.path_to_save: Path | None = None
        "Path to save images. If None, Automaton will use path in ConfigDevice. If Path, extracted after initialise()."
        self.command_factory: CommandFactory = CommandFactory(cfg=cfg)
        "Factory object to create AutomatonCommands."
        self.config_camera: ConfigCamera | None = None
        "Camera configuration object."
        self.dmd: DMDControl = DMDControl(debug_mode=True)
        "DMD object (without display control) to warp camera -> DMD image format."

    def __getstate__(self) -> dict:
        """
        This function and __setstate__ are used to save & load the strategy. Child classes may modify those.

        Returns
        -------
        state: dict
            Dictionary to serialise.
        """
        state = self.__dict__.copy()
        del state['dmd']
        del state['pos_processors']
        return state

    def __setstate__(self, state: dict) -> None:
        """
        This function and __getstate__ are used to save & load the strategy. Child classes may modify those.
        """
        state['dmd'] = DMDControl(debug_mode=True)
        state['pos_processors'] = []
        self.__dict__.update(state)

    @abstractmethod
    def _initialise(self) -> list[AutomatonCommand]:
        """
        See initialise().

        Returns
        -------
        list[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        pass

    def initialise(
            self,
            field_of_views: dict[int, Coordinate],
            positions: dict[int, list[int]],
            region_of_interests: dict[int, list[int]],
            config_camera: ConfigCamera,
            pos_processors: list[PositionRT],
    ) -> list[AutomatonCommand]:
        """
        Initialise the strategy. Note that initialise will be called several times, and it therefore MUST reset
        the strategy to an initial state. Changes to hard-coded constructor attributes that are changed after the
        strategy started MUST be reset here. Following attributes are available:

        Available properties:
        ---------------------
        callback_counter : int
            Counter is zero during initialise.
        field_of_views : dict[int, Coordinate]
            Dictionary with fov_id as key and Coordinate as value.
        positions: dict[int, list[int]]
            Dictionary with fov_id as key and list of pos_id as value. If no additional cropping boxes are specified in
            the GUI, this is a unique mapping, i.e. len(positions)==len(field_of_views) and pos_id==fov_id.
            NOTE: pos_id and fov_id are unique, e.g. positions = {0: [0, 1], 1: [2, 3], ...}
        region_of_interests: dict[int, list[int]]
            Dictionary with pos_id as key and list of roi_id as value.
        config_camera: ConfigCamera
            Object defining camera configuration.
        pos_processors: list[PositionRT]
            List of position processors to access lineages etc. Access via pos_processors[pos_id].
        Returns
        -------
        list[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        self.callback_counter = 0
        self.field_of_views = field_of_views
        self.positions = positions
        self.region_of_interests = region_of_interests
        self.command_factory.update_region_of_interests(region_of_interests=region_of_interests)
        self.config_camera = config_camera
        self.pos_processors = pos_processors
        self.dmd.initialise()  # Load calibration data
        new_command_list = self._initialise()
        if not self.is_valid_command_list(new_command_list):
            raise StrategyError(message=f"AbstractStrategy.callback: invalid command list ({new_command_list}).",
                                error_code=ErrorCode.ERROR_STRATEGY)
        return new_command_list

    @abstractmethod
    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[EvoMachineError],
    ) -> list[AutomatonCommand]:
        """
        See callback().

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the first iteration.
        """
        pass

    def callback(
            self,
            fov_id: int,  # TODO need to consider several move commands passed or allow for one only
            data: list[AutomatonCommand],
            errors: list[EvoMachineError],
    ) -> list[AutomatonCommand]:
        """
        Callback function for the strategy. This function is called by the
        automaton when new data is available. The callback_counter is 0 for the first call of this function,
        then increment by 1 each time.

        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `data` : list[AutomatonCommand]
            List of AutomatonCommand executed at the last time step.
        `errors` : list[EvoMachineError]
            List of errors that occurred during execution.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the first iteration.
        """
        new_command_list = self._callback(fov_id=fov_id, data=data, errors=errors)
        if not self.is_valid_command_list(new_command_list):
            raise StrategyError(message=f"AbstractStrategy.callback: invalid command list ({new_command_list}).",
                                error_code=ErrorCode.ERROR_STRATEGY)
        self.callback_counter += 1
        return new_command_list

    @abstractmethod
    def finalise(self) -> list[AutomatonCommand]:
        """
        Finalise strategy and potentially save data. Provide a last list of commands to be executed. Note that
        callback will not be called after finalise.

        Returns
        -------
        list[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        pass

    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def is_valid_command_list(command_list: list[AutomatonCommand]):   # noqa
        # Max one MOVE command
        # TODO channel NO_LED should give false
        # TODO should only have one command_type per list (?)
        # if len([cmd for cmd in command_list if cmd.command_type == AutomatonCommandType.MOVE]) > 1:
        #     return False
        # else:
        #     return True
        # TODO figure out why I resctricted this to one move command per list? I do not see why
        return True

    def test_strategy(self):
        """
        Basic test calling initialise(), callback(), and finalise() to check for bugs.

        TODO this should be more elaborate and directly use Automaton._process with a TestCamera
        TODO image processing not included yet
        TODO This should be moved to Automaton
        """
        def check_cmd_list(_cmd_list: list[AutomatonCommand], _curr_pos_id: int) -> list[AutomatonCommand]:
            for cmd in _cmd_list:
                cmd.command_data = None
                if cmd.command_type == AutomatonCommandType.MOVE:
                    fov_id = cmd.command_args
                    is_valid_move = (fov_id in self.field_of_views) or (fov_id == -1) or (fov_id is None)
                    if not is_valid_move:
                        raise KeyError(f"fov_id {fov_id} is invalid for {self.field_of_views}")
                    if not (fov_id is None or fov_id == -1):
                        _curr_pos_id = fov_id
                elif cmd.command_type == AutomatonCommandType.WAIT:
                    if not cmd.command_args['duration'] >= 0:
                        raise KeyError(f"invalid wait command arguments {cmd.command_args}")
                elif cmd.command_type == AutomatonCommandType.IMAGE:
                    im_shape = (len(cmd.command_args['channels']), *cfg_camera.image.shape)
                    rand_img = (np.random.rand(*im_shape) * 65535).astype(np.uint16)
                    rand_img_norm = normalise_frame(rand_img)
                    cmd.command_data = rand_img_norm
                cmd.command_execution_time = time.time()
                cmd.fov_id = _curr_pos_id
            return _cmd_list

        logger.info("AbstractStrategy: Testing strategy. Ignore following messages. -------------->")
        field_of_views: dict[int, Coordinate] = {0: Coordinate(0, 0, 0), 1: Coordinate(1, 0, 0), 2: Coordinate(0, 1, 0)}
        positions: dict[int, list[int]] = {0: [0], 1: [1], 2: [2]}
        region_of_interests: dict[int, list[int]] = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}
        cfg_camera: ConfigCamera = ConfigCameraFactory.default_air_config()
        curr_pos_id: int = 0
        try:
            cmd_list = self.initialise(
                field_of_views=field_of_views,
                positions=positions,
                region_of_interests=region_of_interests,
                config_camera=cfg_camera,
                pos_processors=[],
            )
            if self.path_to_save is not None:
                if not self.path_to_save.exists():
                    raise ConfigError(f"AbstractStrategy.test_strategy: path_to_save provided by strategy is invalid "
                                      f"({self.path_to_save}).", ErrorCode.ERROR_DEVICE_CONFIG)
            cmd_list = check_cmd_list(_cmd_list=cmd_list, _curr_pos_id=curr_pos_id)
            cmd_list = self.callback(fov_id=curr_pos_id, data=cmd_list, errors=[])
            _ = check_cmd_list(_cmd_list=cmd_list, _curr_pos_id=curr_pos_id)
            _ = check_cmd_list(_cmd_list=self.finalise(), _curr_pos_id=curr_pos_id)
            logger.info("AbstractStrategy: Strategy successfully tested. Stop ignoring messages. <--------------")
            return True
        except Exception as e:
            logger.error(f"AbstractStrategy.test_strategy: failed with error {str(e)}.")
            traceback.print_exc()
            return False


class NoStrategy(AbstractStrategy):
    """Strategy that does nothing.
    """

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[EvoMachineError],
    ) -> list[AutomatonCommand]:
        return []

    def _initialise(self) -> list[AutomatonCommand]:
        return []

    def finalise(self) -> list[AutomatonCommand]:
        return []


class BasicStrategy(AbstractStrategy):
    """Basic strategy for testing purposes.
    """
    def __init__(self, cfg: ConfigImageProcessor, save_path: str):
        super().__init__(cfg=cfg)
        self.path_to_save = Path(save_path)
        self.imaging_interval: int = 60*3  # seconds
        self.imaging_channels: list[LEDType] = [LEDType.LED_565_NM]  # GFP
        self.exposure_time: int = 100  # milliseconds
        self.num_fovs: int | None = None

    def _initialise(self) -> list[AutomatonCommand]:
        """
        Initialise the strategy.

        Available properties:
        ----------
        field_of_views : dict[int, Coordinate]
            Dictionary with fov_id as key and Coordinate as value.
        positions: dict[int, list[int]]
            Dictionary with fov_id as key and list of pos_id as value.
        region_of_interests: dict[int, list[int]]
            Dictionary with pos_id as key and list of roi_id as value.
        config_camera: ConfigCamera
            Object defining camera configuration.
        Returns
        -------
        list[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        # Create commands for first iteration
        self.num_fovs = len(self.field_of_views)
        cmd_list = []
        for i in range(len(self.field_of_views)):
            cmd_move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(cmd_move)
            cmd_image = self.command_factory.command_image(
                channels=self.imaging_channels,
                exposure_time=self.exposure_time,
                segment=True,
                save=True,
            )
            cmd_list.append(cmd_image)
            cmd_wait = self.command_factory.command_wait(
                duration=self.imaging_interval/self.num_fovs,
                set_live_mode=False,
                channel=LEDType.LED_450_NM,
                brightness=10,
            )
            cmd_list.append(cmd_wait)

        logger.info(f"Initialised strategy with {self.imaging_interval}s imaging interval and {self.num_fovs} FoVs.")
        return cmd_list

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[EvoMachineError],
    ) -> list[AutomatonCommand]:
        """
        Callback function for the strategy. This function is called by the
        automaton when new data is available.


        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `data` : list[AutomatonCommand]
            List of AutomatonCommand.
        `errors` : list[EvoMachineError]
            List of errors that occurred during execution.
        """

        logger.info("FOV {} with data =\n{} \nand errors = {}.".format(
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))
        cmd_list = []
        for i in range(len(self.field_of_views)):
            cmd_move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(cmd_move)
            cmd_image = self.command_factory.command_image(
                channels=self.imaging_channels,
                exposure_time=self.exposure_time,
                segment=False,
                save=True,
            )
            cmd_list.append(cmd_image)
            cmd_wait = self.command_factory.command_wait(
                duration=self.imaging_interval/self.num_fovs,
                set_live_mode=False,
                channel=LEDType.LED_450_NM,
                brightness=10,
            )
            cmd_list.append(cmd_wait)
        return cmd_list

    def finalise(self) -> list[AutomatonCommand]:
        """
        Finalise strategy and potentially save data. Provide a last list of commands to be executed. Note:
        - Callback will not be called again after finalise
        - The WAIT command is disabled in finalise and will have no effect

        Returns
        -------
        list[AutomatonCommand]
            List of commands to be executed by the automaton.
        """
        logger.info("Finalising strategy.")
        cmd_live_mode_on = self.command_factory.command_live_mode(status=True)
        return [cmd_live_mode_on]


def get_all_strategies() -> list[tuple[Type[AbstractStrategy], str]]:
    subclasses = []













    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and issubclass(obj, AbstractStrategy) and obj != AbstractStrategy and obj != NoStrategy:
            subclasses.append((obj, name))

    strategies_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'strategies'))
    for file_name in os.listdir(strategies_folder):
        if file_name.endswith('.py') and file_name != '__init__.py':
            module_name = os.path.splitext(file_name)[0]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(strategies_folder, file_name))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, AbstractStrategy):
                    subclasses.append((obj, name))
    return subclasses

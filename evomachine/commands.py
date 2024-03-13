from dataclasses import dataclass
from enum import Enum, auto
import logging
from time import time, gmtime, strftime
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from delta.utils import CroppingBox as DeltaCroppingBox

from evomachine.config import EVO_FORMATTER, get_logger, USE_DMD_SOCKET
from evomachine.coordinates import Coordinate
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.utils import EvoCroppingBox


logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(EVO_FORMATTER)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class AutomatonCommand:
    command_id: int
    "Unique command ID."
    command_type: AutomatonCommandType
    "A command type defined in AutomatonCommandType."
    command_args: Any
    "Necessary command arguments. Use AutomatonCommandFactory to provide the correct arguments."
    command_creation_time: float
    "Time at which the command was created. Produced via time.time()."
    command_data: Any = None
    "Data collected after executing the command."
    command_execution_time: Union[float, None] = None
    "Time at which the command was created. Produced via time.time()."
    fov_id: Union[int, None] = None
    "Field of view ID. Used for commands sent to GUI."

    @staticmethod
    def _get_time(t: Union[float, None]) -> str:
        return "None" if t is None else strftime('%Y-%m-%d %H:%M:%S', gmtime(t))

    def get_exec_time(self):
        return self._get_time(self.command_execution_time)

    def __str__(self):
        creation_time = self._get_time(self.command_creation_time)
        exec_time = self._get_time(self.command_execution_time)
        has_data = "True" if self.command_data is not None else "False"
        return f"Command(Type={self.command_type}, ID={self.command_id}, has_data={has_data}, " \
               f"Creation Time={creation_time}, Exec Time={exec_time})"


class CommandFactory:
    """
    Factory class for creating AutomatonCommand. Each command type will produce certain output data that will be
    stored in Automaton.command_output.

    The output data will be provided as another command, where command.command_type and command.command_id
    match the type and ID of the initial command. The data will be stored in command_id and the field
    command_execution_time will contain the time just after executing the command.
    """
    def __init__(self):
        self._command_id_counter: int = -1

    def command_from_template(self, template: AutomatonCommand) -> AutomatonCommand:
        """
        Assigns a valid command ID and a new command_creation_time to the template.

        Parameters
        ----------
        template: Template AutomatonCommand.

        Returns
        -------

        """
        template.command_id = self.get_next_id()
        template.command_creation_time = time()
        template.command_execution_time = None
        return template

    def get_next_id(self):
        self._command_id_counter += 1
        return self._command_id_counter

    def command_image(
            self,
            channels: List[LEDType],
            exposure_time: Union[int, None],
            segment: bool,
            brightness: Union[int, List[int]] = 100,
            save: bool = False,
    ) -> AutomatonCommand:
        """
        Create a command for taking an image.

        Parameters
        ----------
        channels        : List of LED channels.
        exposure_time   : If None, uses default exposure, otherwise, in MILLISECONDS.
        segment         : Segments image and tracks cells if True.
        brightness      : Brightness as value in [0,100].
        save            : Save image(s). Uses ConfigDevice.path_to_save passed to Automaton.

        Returns in AbstractStrategy.callback
        ------------------------------------
        If segment was False:
            command_data (np.array) : 3D int16 numpy array with 1st dimension = len(channels)
        If segment was True:
            command_data (Dict[int, Lineage]) : A dictionary with ROI IDs as keys and a delta.Lineage object as values.
                                                In case of a mothermachine experiment, the ROIs will be the trenches in
                                                the corresponding FoV. Otherwise, the single key will be 0 and the
                                                Lineage object will correspond to all cells in the current FoV.
        """
        if not (isinstance(channels, list) and all(isinstance(channel, LEDType) for channel in channels)):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument channel ({type(channels)}).")
        if not isinstance(exposure_time, int):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument exposure_time ({type(exposure_time)}).")
        if not isinstance(segment, bool):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument segment ({type(segment)}).")
        if not ((isinstance(brightness, int) and 0 <= brightness <= 100) or
                (isinstance(brightness, list) and
                 all(0 <= b <= 100 for b in brightness) and len(brightness) == len(channels))):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type or range or format for argument brightness.")
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        command_args = {
            'channels': channels, 'exposure_time': exposure_time, 'segment': segment, 'brightness': brightness,
            'save': save
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.IMAGE,
            command_args=command_args,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_move(self, fov_id: Union[int, None]) -> AutomatonCommand:
        """
        Create a command for moving the stage.

        Parameters
        ----------
        fov_id  : Either a valid position index, -1 for next position in the list, or None for no movement.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns True.
        """
        if not (isinstance(fov_id, int) or fov_id is None):
            raise TypeError(f"AutomatonCommandFactory.move: Wrong type provided ({type(fov_id)}).")
        return AutomatonCommand(
            command_type=AutomatonCommandType.MOVE,
            command_args=fov_id,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_project(
            self,
            channel: LEDType,
            image: np.ndarray[(int, int), np.uint8],
            duration: Union[float, int],
            brightness: int = 100
    ) -> AutomatonCommand:
        """

        Parameters
        ----------
        channel         : LED channel.
        image           : 2D numpy array to be projected via DMD.
        duration        : Duration of the projection in SECONDS.
        brightness      : Brightness as value in [0,100].

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns True.
        -------

        """
        if not isinstance(channel, LEDType):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument channel ({type(channel)}).")
        if not (isinstance(image, np.ndarray) and image.shape == DMD_WIDTH_HEIGHT and image.dtype == np.uint8):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument image ({type(image)}).")
        if not (isinstance(duration, float) or isinstance(duration, int)):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument duration ({type(duration)}).")
        if (not isinstance(brightness, int)) or not (0 <= brightness <= 100):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type or range for argument brightness.")
        return AutomatonCommand(
            command_type=AutomatonCommandType.PROJECT,
            command_args={'channel': channel, 'image': image, 'duration': duration, 'brightness': brightness},
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_stop(self) -> AutomatonCommand:
        """
        Stop automaton. Note that AbstractStrategy.callback is not called anymore after this.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data (bool)  : Always returns True.
        """
        return AutomatonCommand(
            command_type=AutomatonCommandType.STOP,
            command_args=None,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_wait(self, duration: float) -> AutomatonCommand:
        """

        Parameters
        ----------
            duration (float) : Time to wait in SECONDS.


        Returns in AbstractStrategy.callback
        ------------------------------------
            command_data : Always returns None.
        """
        return AutomatonCommand(
            command_type=AutomatonCommandType.WAIT,
            command_args=duration,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def reset(self):
        self._command_id_counter = -1

    @staticmethod
    def command_focus_data(
            focus_curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
            focus_stack: np.ndarray,
            focus_prev_stack: np.ndarray,
            focus_prev_z_coords: np.ndarray,
            fovs: Dict[int, Coordinate],
    ):
        command_args = {
            'focus_curves': focus_curves,
            'focus_stack': focus_stack,
            'focus_prev_stack': focus_prev_stack,
            'focus_prev_z_coords': focus_prev_z_coords,
            'fovs': fovs,
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.FOCUS_DATA,
            command_args=command_args,
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_fov_data(
            fovs: Dict[int, Coordinate],
            cropping_boxes: Dict[int, List[EvoCroppingBox]],
            fov_to_pos: Dict[int, List[int]],
            pos_to_fov_index: Dict[int, int]
    ):
        command_args = {
            'fovs': fovs,
            'cropping_boxes': cropping_boxes,
            'fov_to_pos': fov_to_pos,
            'pos_to_fov_index': pos_to_fov_index,
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.FOV_DATA,
            command_args=command_args,
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_info_text(
            text: str,
    ):
        return AutomatonCommand(
            command_type=AutomatonCommandType.INFO_TEXT,
            command_args={'text': text},
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_ref_data(ref_frames: List[np.ndarray]) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.REF_DATA,
            command_args=ref_frames,
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_roi_data(fov_id: int, rotation: float, roi_boxes: list[DeltaCroppingBox]) -> AutomatonCommand:
        command_args = {
            'fov_id': fov_id,
            'rotation': rotation,
            'roi_boxes': roi_boxes,
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.ROI_DATA,
            command_args=command_args,
            command_id=-1,
            command_creation_time=time(),
            fov_id=fov_id,
        )



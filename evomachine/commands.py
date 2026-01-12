from dataclasses import dataclass
from enum import Enum, auto
import logging
from time import time, gmtime, strftime
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from delta.utils import CroppingBox as DeltaCroppingBox

from evomachine.config import EVO_FORMATTER, get_logger, USE_DMD_SOCKET, ConfigImageProcessor
from evomachine.coordinates import Coordinate
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.evotypes import AutomatonCommandType, FilterWheelType, FocusStatusType, LEDType, MagnetModeType
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
    # command_start_time: Union[float, None] = None
    # "Time pre execution. Produced via time.time()."
    command_execution_time: Union[float, None] = None
    "Time post execution. Produced via time.time()."
    fov_id: int | None = None
    "Field of view ID. Used for commands sent to GUI."

    @staticmethod
    def _get_time(t: Union[float, None]) -> str:
        return "None" if t is None else strftime('%Y-%m-%d %H:%M:%S', gmtime(t))

    def get_exec_time(self):
        return self._get_time(self.command_execution_time)

    @staticmethod
    def _format_args(command_args):
        if isinstance(command_args, dict):
            def _format_val(val):
                if isinstance(val, np.ndarray):
                    return f"shape {val.shape}, min={val.min()}, max={val.max()}, mean={np.mean(val)}"
                else:
                    return val

            "\t\n".join([f"{key}: {_format_val(val)}" for key, val in command_args.items()])
        else:
            return command_args

    def __str__(self):
        creation_time = self._get_time(self.command_creation_time)
        exec_time = self._get_time(self.command_execution_time)
        has_data = "True" if self.command_data is not None else "False"
        return f"Command(Type={self.command_type}, ID={self.command_id}, has_data={has_data}, " \
               f"Creation Time={creation_time}, Exec Time={exec_time}) " \
               f"with args {self._format_args(self.command_args)}."


class CommandFactory:
    """
    Factory class for creating AutomatonCommand. Each command type will produce certain output data that will be
    stored in Automaton.command_output.

    The output data will be provided as another command, where command.command_type and command.command_id
    match the type and ID of the initial command. The data will be stored in command_id and the field
    command_execution_time will contain the time just after executing the command.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        self._cfg: ConfigImageProcessor = cfg
        "Image processor config used for checking channels of imaging commands."
        self._command_id_counter: int = -1
        "Command ID generator. First ID is 0. Does not care for overflow."
        self._pos_to_roi: dict[int, list[int]] = {}
        "Dictionary mapping position_id to roi_ids set in update_region_of_interests. Used to check commands."

    def get_next_id(self):
        self._command_id_counter += 1
        return self._command_id_counter

    def update_region_of_interests(self, region_of_interests: dict[int, list[int]]):
        self._pos_to_roi = region_of_interests

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

    def command_magnet(self, 
                       enable = None,
                       value: float = 0.0,
                       mode: MagnetModeType = MagnetModeType.CURRENT_SET) -> AutomatonCommand:
        """Sets the magnet either using current control or field, or switches it on or off entirely.
        If enable is None, the magnet state is not changed.
        If enable is True, the magnet is switched on.
        If enable is False, the magnet is switched off.

        Args:
            enable (_type_, optional): _description_. Defaults to None.
            value (float, optional): _description_. Defaults to 0.0.
            mode (MagnetModeType, optional): _description_. Defaults to MagnetModeType.CURRENT_SET.

        Returns:
            AutomatonCommand: _description_
        """
        command_args = {'enable': enable, 'value': value, 'mode': mode}
        return AutomatonCommand(
            command_type=AutomatonCommandType.MAGNET,
            command_args=command_args,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )
        
    def command_calibrate_magnet(self) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.CALIBRATE_MAGNET,
            command_args=None,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_calibrate_hall(self, hall_id: int) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.CALIBRATE_HALL,
            command_args=hall_id,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )
        
    def command_read_hall(self, hall_id: int) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.READ_HALL,
            command_args=hall_id,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_image(
            self,
            channels: list[LEDType],
            exposure_time: int | None,
            segment: bool,
            brightness: int | float | list[int | float] = 10,
            save: bool = False,
            filename_suffix: str | None = None,
            pattern: np.ndarray | None = None,
            filter_wheel: FilterWheelType | list[FilterWheelType] | None = None,
            force_led: bool = False,
            reset_led: bool = False,
    ) -> AutomatonCommand:
        """
        Create a command for taking an image.

        Parameters
        ----------
        channels        : List of LED channels. If any image processing is enabled, channels must
                          contain ConfigImageProcessor.channels_seg.
        exposure_time   : If None, uses default exposure, otherwise, in MILLISECONDS.
        segment         : Segments image and tracks cells if True. See channels for channel requirements. If segment is
                          True, and ConfigImageProcessor.preproc_enabled is False, this function throws an exception.
        brightness      : Brightness as value in [0,29].
        save            : Save image(s). Uses ConfigDevice.path_to_save passed to Automaton.
        filename_suffix : Suffix to append to filenames.
        pattern         : An optional pattern of size width_height_DMD (see DMDControl) that will be displayed using
                          dmd.display_image(). If None, the DMD is set via dmd.display_full().
        filter_wheel    : Optional filter wheel to set. Otherwise, current setting is used.
        force_led       : TODO
        reset_led       : TODO

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data: Dictionary
        # TODO
        command_data['img']: 3D int16 numpy array (normalised & rotated images) with 1st dimension = len(channels)
        command_data['seg']: Provided if segment is True. A dictionary with ROI IDs as keys and a delta.Lineage object
                             as values. In case of a mothermachine experiment, the ROIs will be the trenches in
                             the corresponding FoV. Otherwise, the single key will be 0 and the Lineage object will
                             correspond to all cells in the current FoV.

        Returns
        -------
        command: AutomatonCommand
        """
        if not (isinstance(channels, list) and all(isinstance(channel, LEDType) for channel in channels)):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument channel ({type(channels)}).")
        if not isinstance(exposure_time, int):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument exposure_time ({type(exposure_time)}).")
        if not isinstance(segment, bool):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument segment ({type(segment)}).")
        if not isinstance(force_led, bool):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument force_led ({type(force_led)}).")
        if not isinstance(reset_led, bool):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument reset_led ({type(reset_led)}).")
        if segment:  # self._cfg.preproc_enabled or
            if not all([ch_seg in channels for ch_seg in self._cfg.channels_seg]):
                raise TypeError(f"AutomatonCommandFactory.image: channels_seg={self._cfg.channels_seg} not in channels={channels} for segment=True.")
        if segment and not self._cfg.preproc_enabled:
            raise TypeError(f"AutomatonCommandFactory.image: segment=True but preproc_enabled=False.")
        if not ((isinstance(brightness, int) and 0 <= brightness <= 100) or
                (isinstance(brightness, list) and
                 all([0 <= b <= 100 for b in brightness]) and len(brightness) == len(channels))):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type or range or format for argument brightness.")
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        if filter_wheel is not None:
            if isinstance(filter_wheel, list) and (len(channels) != len(filter_wheel) or
                                                   any([not isinstance(f, FilterWheelType) for f in filter_wheel])):
                raise TypeError("AutomatonCommandFactory.image: If list, len(filter_wheel) must be equal len(channels).")
            elif not isinstance(filter_wheel, list) and not isinstance(filter_wheel, FilterWheelType):
                raise TypeError("AutomatonCommandFactory.image: filter_wheel must be of type FilterWheelType.")
        if not (isinstance(filename_suffix, str) or filename_suffix is None):
            raise TypeError(f"AutomatonCommandFactory.image: Wrong type for argument filename_suffix ({type(filename_suffix)}).")
        command_args = {
            'channels': channels, 'exposure_time': exposure_time, 'segment': segment, 'brightness': brightness,
            'save': save, 'pattern': pattern, 'filter_wheel': filter_wheel,
            'force_led': force_led, 'reset_led': reset_led, 'filename_suffix': filename_suffix,
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.IMAGE,
            command_args=command_args,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_live_mode(self, status: bool) -> AutomatonCommand:
        """
        Sets MMC live mode for EvoCamera.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data (bool)  : Always returns True.

        Parameters
        ----------
        status: bool    Set to true to enable live mode.

        Returns
        -------
        command: AutomatonCommand
        """
        return AutomatonCommand(
            command_type=AutomatonCommandType.LIVE_MODE,
            command_args=status,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_move(self, fov_id: int | None) -> AutomatonCommand:
        """
        Create a command for moving the stage.

        Parameters
        ----------
        fov_id  : Either a valid position index, -1 for next position in the list, or None for no movement.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns True.

        Returns
        -------
        command: AutomatonCommand
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
            duration: int | float,
            brightness: int | float = 29,
    ) -> AutomatonCommand:
        """
        Projects a pattern onto the current FoV.

        Parameters
        ----------
        channel         : LED channel.
        image           : 2D numpy array to be projected via DMD.
        duration        : Duration of the projection in SECONDS.
        brightness      : Brightness as value in [0,100].

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns True.

        Returns
        -------
        command: AutomatonCommand
        -------

        """
        if not isinstance(channel, LEDType):
            raise TypeError(f"AutomatonCommandFactory.project: Wrong type for argument channel ({type(channel)}).")
        if not (isinstance(image, np.ndarray) and image.shape == DMD_WIDTH_HEIGHT and image.dtype == np.uint8):
            raise TypeError(f"AutomatonCommandFactory.project: Wrong type for argument image ({type(image)}).")
        if not (isinstance(brightness, int) or not isinstance(brightness, float)) or not (0 <= brightness <= 100):
            msg = f"AutomatonCommandFactory.project: Brightness must satisfy {0} < {duration} (actual) < 100."
            raise TypeError(msg)
        max_duration = 60*60 if brightness > 29 else 3600
        if not (isinstance(duration, float) or isinstance(duration, int)) or not (0 < duration < max_duration):
            msg = f"AutomatonCommandFactory.project: Duration must satisfy {0} < {duration} (actual) < {max_duration}."
            raise TypeError(msg)
        return AutomatonCommand(
            command_type=AutomatonCommandType.PROJECT,
            command_args={'channel': channel, 'image': image, 'duration': duration, 'brightness': brightness},
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_project_roi(
            self,
            channel: LEDType,
            pos_id: int,
            roi_ids: list[int],
            duration: int | float,
            brightness: int | float = 29,
            fill_x: float = 1.0,
            fill_y: float = 1.0,
            invert: bool = False,
            set_live_mode: bool = False,
    ) -> AutomatonCommand:
        """
        Projects a pattern built from the specified RoI boxes onto the current FoV. NOTE: The automaton will NOT move to
        the provided pos_id. A MOVE command to the corresponding pos_id / fov_id must be provided first.

        Special behavior with invert=True:
        If invert=True, the automaton will add black patches to the left and right of the trench columns. These black
        patches extend to the trench box boundary for fill_x>=1. IF fill_x<1, the black patches will extend into the
        box.

        Parameters
        ----------
        channel         : LED channel.
        pos_id          : Position ID to obtain the RoI boxes from.
        roi_ids         : List of roi_ids to build the pattern. Must correspond to the roi_ids of pos_id.
        duration        : Duration of the projection in SECONDS.
        brightness      : Brightness as value in [0,100].
        fill_x          : Determines the percentage of the RoI boxes filled (in X/horizontal/column direction).
        fill_y          : Determines the percentage of the RoI boxes filled (in Y/vertical/row direction).
        invert          : Inverts black/white after creating the projection pattern. See Automaton._process.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns True.

        Returns
        -------
        command: AutomatonCommand
        -------

        """
        if not isinstance(channel, LEDType):
            raise TypeError(f"AutomatonCommandFactory.command_project_roi: Wrong type for argument channel "
                            f"({type(channel)}).")
        if not (isinstance(pos_id, int) and (pos_id in self._pos_to_roi.keys())):
            raise TypeError(f"AutomatonCommandFactory.command_project_roi: pos_id={pos_id} does not exist.")
        if not (all(isinstance(r, int) and (r in self._pos_to_roi[pos_id]) for r in roi_ids)):
            raise TypeError(f"AutomatonCommandFactory.command_project_roi: roi_ids do not exist for pos_id={pos_id}.")
        if not (isinstance(brightness, int) or not isinstance(brightness, float)) or not (0 <= brightness <= 100):
            raise TypeError(f"AutomatonCommandFactory.project: Wrong type or range for argument brightness.")
        max_duration = 60*60 if brightness > 29 else 3600
        if not (isinstance(duration, float) or isinstance(duration, int)) or not (0 < duration < max_duration):
            msg = f"AutomatonCommandFactory.project: Duration must satisfy {0} < {duration} (actual) < {max_duration}"
            raise TypeError(msg)
        if not isinstance(fill_x, float) or not (0 <= fill_x):
            raise TypeError(f"AutomatonCommandFactory.command_project_roi: Wrong type or range for argument fill_x.")
        if not isinstance(fill_y, float) or not (0 <= fill_y):
            raise TypeError(f"AutomatonCommandFactory.command_project_roi: Wrong type or range for argument fill_y.")
        return AutomatonCommand(
            command_type=AutomatonCommandType.PROJECT_ROI,
            command_args={'channel': channel, 'pos_id': pos_id, 'roi_ids': roi_ids, 'duration': duration,
                          'brightness': brightness, 'fill_x': fill_x, 'fill_y': fill_y, 'invert': invert,
                          'set_live_mode': set_live_mode},
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_save_state(self, suffix: str = "") -> AutomatonCommand:
        """
        Save Automaton state.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data (bool)  : Always returns True.

        Parameters
        ----------
        suffix : str    Appended to pickle filename.

        Returns
        -------
        command: AutomatonCommand
        """
        if not isinstance(suffix, str):
            raise TypeError(f"AutomatonCommandFactory.command_save_state: Wrong type or range for argument suffix.")
        return AutomatonCommand(
            command_type=AutomatonCommandType.SAVE_STATE,
            command_args=suffix,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_stop(self) -> AutomatonCommand:
        """
        Stop automaton. Note that AbstractStrategy.callback is not called anymore after this.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data (bool)  : Always returns True.

        Returns
        -------
        command: AutomatonCommand
        """
        return AutomatonCommand(
            command_type=AutomatonCommandType.STOP,
            command_args=None,
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def command_wait(
            self,
            duration: float,
            set_live_mode: bool = False,
            channel: LEDType = LEDType.LED_450_NM,
            brightness: int | float = 10,   # TODO this must be divided by 100 in automaton
    ) -> AutomatonCommand:
        """

        Parameters
        ----------
        duration: float
            Time to wait in SECONDS.
        set_live_mode: bool
            MM live mode will be enabled before start of waiting time and disabled afterwards.
        channel: LEDType
            Channel to set if set_live_mode=True.
        brightness: int | float
            Brightness level in [0,100] to be applied if set_live_mode=True.

        Returns in AbstractStrategy.callback
        ------------------------------------
        command_data : Always returns None.

        Returns
        -------
        command: AutomatonCommand
        """
        return AutomatonCommand(
            command_type=AutomatonCommandType.WAIT,
            command_args={
                'duration': duration, 'set_live_mode': set_live_mode, 'channel': channel, 'brightness': brightness
            },
            command_id=self.get_next_id(),
            command_creation_time=time(),
        )

    def reset(self) -> None:
        self._command_id_counter = -1

    # Methods below used by Automaton for GUI communication

    @staticmethod
    def command_focus_data(
            focus_curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
            focus_stack: np.ndarray,
            focus_prev_stack: np.ndarray,
            focus_prev_z_coords: np.ndarray,
            fovs: Dict[int, Coordinate],
    ) -> AutomatonCommand:
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
    ) -> AutomatonCommand:
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
    def command_autofocus(
            is_locked: bool,
            refocusing: bool,
            max_num_trials_reached: bool,
            software_focus_status: FocusStatusType,
    ) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.AUTOFOCUS_DATA,
            command_args={
                'is_locked': is_locked,
                'refocusing': refocusing,
                'max_num_trials_reached': max_num_trials_reached,
                'software_focus_status': software_focus_status,
            },
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_info_text(
            text: str,
    ) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.INFO_TEXT,
            command_args={'text': text},
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_ref_data(ref_frames: dict[int, np.ndarray]) -> AutomatonCommand:
        return AutomatonCommand(
            command_type=AutomatonCommandType.REF_DATA,
            command_args=ref_frames,
            command_id=-1,
            command_creation_time=time(),
        )

    @staticmethod
    def command_roi_data(
            fov_id: int,
            rotation: float,
            roi_boxes: list[DeltaCroppingBox],
    ) -> AutomatonCommand:
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

    @staticmethod
    def command_seg_data(
            fov_id: int,
            seg_masks: dict[int, np.ndarray],
    ) -> AutomatonCommand:
        command_args = {
            'fov_id': fov_id,
            'seg_masks': seg_masks,
        }
        return AutomatonCommand(
            command_type=AutomatonCommandType.SEG_DATA,
            command_args=command_args,
            command_id=-1,
            command_creation_time=time(),
            fov_id=fov_id,
        )

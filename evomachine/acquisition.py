from datetime import datetime
import itertools
import logging
import numpy as np
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple

import cv2
import matplotlib.pyplot as plt
from pycromanager import Core, Studio
from pynput import keyboard
import skimage

from asitiger.command import CRISPState
from asitiger.status import Status
import asitiger.tigercontroller
import delta

from evomachine.config import ConfigCRISP, ConfigDevice, ConfigFocus, ConfigFocusAlgorithm, ConfigImage, ConfigLED, \
    ConfigObjective, CRISP_CONFIG_DEFAULT, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, OBJECTIVE_CONFIG_DEFAULT, \
    get_logger
from evomachine.coordinates import Coordinate
from evomachine.dmd import DMDColor
from evomachine.exceptions import CameraError, ConfigError, DMDError, ErrorCode, ErrorContainer, \
    EvoMachineError, StageError, TigerError
from evomachine.software_focus import get_focus_score


logger = get_logger(name=__name__)


class AbstractCamera:
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: Optional[ConfigImage] = IMAGE_CONFIG_DEFAULT,
            cfg_objective: Optional[ConfigObjective] = OBJECTIVE_CONFIG_DEFAULT,
            cfg_focus: Optional[ConfigFocus] = FOCUS_CONFIG_DEFAULT,
    ):
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self.cfg_device: ConfigDevice = cfg_device
        "Device configuration object."
        self.cfg_image: ConfigImage = cfg_image
        "Image configuration object."
        self.cfg_objective: ConfigObjective = cfg_objective
        "Objective configuration object."
        self.cfg_focus: ConfigFocus = cfg_focus
        "Focus configuration object."
        self._step: int = -1
        "Increments each time an image is taken."
        self._curr_pos: int = 0  # TODO need to initialise with current position ID
        "Current position equalling 0 or i_pos passed to move_to_pos."
        self._pos_id_to_coordinate: Dict[int, Any] = {}
        "Map defining coordinate of each position ID. Format of values specified by children classes."
        self._autofocus_is_locked: bool = False
        "Switches to True after enabling autofocus. Use self.autofocus_is_locked() to query status."
        self._curr_exposure: Union[float, None] = None
        "Currently set exposure time. Note: changes from micromanager are NOT registered."

        self.focus_scores: Union[None, np.array] = None
        "Initialised in software_focus. Contains the focus score of each image. Larger score = sharper image."
        self.focus_stack: Union[None, np.array] = None
        "Initialised in software_focus. Contains images of focus stack."
        self.focus_prev_image: Union[None, np.array] = None
        "Initialised in software_focus. Contains the image from before starting focus."
        self.focus_Z_coords: Union[None, np.array] = None
        "Initialised in software_focus. Contains Z coordinates of focus stack. Use focus_curr_pos for X/Y coordinates."
        self._focus_is_initialised: bool = False
        "Changes to True after initialisation and to False after finalisation."

        self.cfg_device.check_config()

    def autofocus_enable(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        raise NotImplementedError()

    def autofocus_disable(self):
        raise NotImplementedError()

    def autofocus_is_locked(self):
        raise NotImplementedError()

    def autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        raise NotImplementedError()

    def autofocus_unlock(self):
        raise NotImplementedError()

    def coordinate_is_out_of_bounds(self, coordinate: Union[Dict[str, float], Coordinate]) -> bool:
        raise NotImplementedError()

    def initialise(self):
        self.reset_counter()
        self._initialise()

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logger.warning(msg=msg)
        else:
            logger.warning("No errors for acquisition found.")

    def disable_led(self):
        raise NotImplementedError()

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        raise NotImplementedError()

    def get_software_focus_z_coord(self) -> Union[int, None]:
        if self.focus_scores is None:
            return None
        focus_best_position = np.argmax(self.focus_scores)
        return self.focus_Z_coords[focus_best_position]

    def get_software_focus_z_frame(self) -> Union[np.ndarray, None]:
        if self.focus_scores is None:
            return None
        focus_best_position = np.argmax(self.focus_scores)
        return self.focus_stack[:, :, focus_best_position]

    def get_filename(
            self,
            i_pos: Optional[int] = None,
            i_channel: Optional[Union[int, ConfigLED]] = None,
    ) -> str:
        return "evom_pos{:02d}_{}".format(self._curr_pos, datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))

    def get_frame(
            self,
            i_chan: Union[int, None],
            normalise: bool = False,
            brightness: int = 100,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        self._step += 1
        frame = self._take_frame(i_chan=i_chan, brightness=brightness, block=block, reset_led=reset_led)
        return self.normalise_frame(frame=frame) if (normalise and (frame is not None)) else frame

    def get_pos(self) -> int:
        return self._curr_pos

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        raise NotImplementedError()

    def halt_stage(self):
        raise NotImplementedError()

    def display_save_frame(
            self,
            i_chan: Union[int, None],
            path_to_save: Optional[Union[Path, str, None, bool]] = None,
            filename: Optional[Union[str, None]] = None,
            display_frame: Optional[bool] = True,
            normalise: bool = False,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), np.uint16]]:
        """
        Takes a frame (image) and returns it. Additionally, saves the frame if path_to_save is not None. If filename is
        None, it will use a default filename format defined in get_filename(). If display_frame is True, it also shows
        the acquired frame. If normalise is False, it shows the raw frame, if normalise is True, it shows the normalised
        frame. Frames are always saved in raw format.

        CAREFUL: EvoCamera returns image in uint16.

        Parameters
        ----------
        i_chan          : LED channel.
        path_to_save    : If not None or (bool and False), saves image. If (bool and true), uses path in cfg_device.
        filename        : If None, uses filename from get_filename() to save (overwritten by EvoCam).
        display_frame   : If True, displays image.
        normalise       : If True, normalises displayed (not saved) image.
        block           : Waits until device is not busy anymore.
        reset_led       : Reset LED to value before calling this function.

        Returns
        -------

        """
        frame = self.get_frame(i_chan=i_chan, block=block, reset_led=reset_led)
        if frame is None:
            logger.warning(f"AbstractCamera.display_save_frame: self.get_frame returned None. Aborting...")
            return None
        if display_frame:
            self.plot_frame(frame=frame, normalise=normalise)
        if path_to_save is not None:
            self.save_frame(frame=frame, path_to_save=path_to_save, filename=filename)
        return frame

    def get_delta_fov(self):
        raise NotImplementedError()

    def get_exposure(self) -> Union[int, float]:
        return self._curr_exposure

    def keyboard_control(self):
        raise NotImplementedError()

    def move_home(self, block: Optional[bool] = False):
        raise NotImplementedError()

    def move_fov_up(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        raise NotImplementedError()

    def move_fov_down(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        raise NotImplementedError()

    def move_fov_left(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        raise NotImplementedError()

    def move_fov_right(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        raise NotImplementedError()

    def move_to(self, coordinate: Union[Dict[str, int], Coordinate], block: Optional[bool] = False):
        raise NotImplementedError()

    def move_to_pos(self, i_pos: int) -> None:
        if i_pos not in self._pos_id_to_coordinate:
            raise StageError("Position index {} out of range".format(i_pos),
                             ErrorCode.ERROR_STAGE_COORDINATES)
        success = self._move_stage_to_pos(i_pos=i_pos)
        if not success:
            raise StageError("Fault moving to position={}.".format(i_pos), ErrorCode.ERROR_STAGE_MOVEMENT)
        self._curr_pos = i_pos

    @staticmethod
    def normalise_frame(
            frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype'],
            colormap: Union['plt.cm', bool, None] = True,
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        """
        Convenience function to normalise frames.

        Parameters
        ----------
        frame       Numpy 2D array.
        colormap    If bool and True or plt.cm, returns a coloured 3D array. If False or None, only normalises.

        Returns
        -------
        Either a 2D or a 3D numpy array depending on colormap argument.
        """
        norm = plt.Normalize(vmin=frame.min(), vmax=frame.max())
        if (colormap is None) or (isinstance(colormap, bool) and not colormap):
            return norm(frame)
        elif isinstance(colormap, bool) and colormap:
            return plt.cm.jet(norm(frame))
        else:
            return colormap(norm(frame))

    def plot_frame(self, frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype'], normalise: bool = True):
        image = self.normalise_frame(frame=frame) if normalise else frame
        plt.imshow(image)
        plt.show()

    def reset_counter(self):
        self._step = -1

    def save_frame(
            self,
            frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype'],
            path_to_save: Optional[Union[Path, str, bool]] = True,
            filename: Optional[Union[str, None]] = None,
            i_pos: Optional[int] = None,
            i_channel: Optional[Union[int, ConfigLED]] = None,
    ) -> None:
        """
        Image is saved under path_to_save / filename. See arguments for different options. If provided, checks whether
        path_to_save exists.

        Parameters
        ----------
        frame           Image to save (numpy array)
        path_to_save    Can be Path/str or bool. If Path/str, path is used. If True, path taken from cfg_device.
        filename        Can be str or None. If None, default filename used from get_filename().

        Returns
        -------

        """
        if not filename:
            filename = self.get_filename(i_pos=i_pos, i_channel=i_channel)

        if isinstance(path_to_save, str):
            path_to_save = Path(path_to_save)
        elif isinstance(path_to_save, bool):
            if not path_to_save:
                return
            path_to_save = self.cfg_device.path_to_save
        if not path_to_save.exists():
            logger.warning(f"AbstractCamera.save_frame: Path {path_to_save} does not exist. "
                           f"Returning image without saving...")
            return
        logger.info(f"Saving image {path_to_save / filename}.")
        skimage.io.imsave(path_to_save / filename, frame, plugin="tifffile", check_contrast=False)

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None, # ((xmin,xmax), (ymin,ymax))
            algorithm_override: Optional[ConfigFocusAlgorithm] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        self.software_focus_initialise(
            cfg_focus=cfg_focus,
            focus_channel_override=focus_channel_override,
            rel_range_override=rel_range_override,
            cropping_indices=cropping_indices,
            algorithm_override=algorithm_override,
            user_input_override=user_input_override,
            countdown_override=countdown_override,
        )
        if not self._focus_is_initialised:
            logger.error(
                f"software_focus: Focus not initialised or initialisation failed. Check log. Aborting focus."
            )
            return
        for ipos in range(len(self.focus_Z_coords)):
            self.software_focus_step(ipos=ipos)
        self.software_focus_finalise()

    def software_focus_finalise(self):
        raise NotImplementedError()

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None, # ((xmin,xmax), (ymin,ymax))
            algorithm_override: Optional[ConfigFocusAlgorithm] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        """
        Initialises software focus routine and validates parameter and limits.

        Parameters
        ----------
        cfg_focus               : ConfigFocus object. If None, uses self.cfg_focus provided in constructor.
        focus_channel_override  : Overrides cfg_focus.focus_channel.
        rel_range_override      : Overrides cfg_focus.rel_range.
        cropping_indices        : Apply focus to img[]
        algorithm_override
        user_input_override
        countdown_override

        Returns
        -------

        """
        raise NotImplementedError()

    def software_focus_is_initialised(self) -> bool:
        return self._focus_is_initialised

    def software_focus_step(self, ipos: int):
        raise NotImplementedError()

    def set_exposure(self, exposure_time: Union[int, None] = None):
        if exposure_time is None:
            exposure_time = self.cfg_focus.exposure_time
        self._set_exposure(exposure_time=exposure_time)
        self._curr_exposure = exposure_time

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        raise NotImplementedError()

    def set_led(self, i_chan: Union[int, ConfigLED], brightness: int = 100, block: bool = False):
        raise NotImplementedError()

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Any]) -> bool:
        raise NotImplementedError()

    def zero_coordinates(self):
        raise NotImplementedError()

    def _initialise(self) -> None:
        raise NotImplementedError()

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        raise NotImplementedError()

    def _set_filter_wheel(self, i_pos: int):
        raise NotImplementedError()

    def _take_frame(
            self,
            i_chan: Union[int, None],
            brightness: int = 100,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        raise NotImplementedError()


class DeltaCamera(AbstractCamera):
    """
    A class to mock the acquisition of frames.
    """
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: Optional[ConfigImage] = IMAGE_CONFIG_DEFAULT,
            cfg_objective: Optional[ConfigObjective] = OBJECTIVE_CONFIG_DEFAULT,
            cfg_focus: Optional[ConfigFocus] = FOCUS_CONFIG_DEFAULT,
    ):
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image, cfg_objective=cfg_objective, cfg_focus=cfg_focus)
        self.all_frames: List[np.ndarray[(int, int, int, int), np.float32]] = [
            np.empty((1, 1, 1, 1)) for _ in range(cfg_device.num_periods)
        ]
        "List of all frames by position."
        self._curr_period: int = -1
        "Incremented after completing one round of imaging the whole device."

        delta_reader: delta.utils.XPReader = \
            delta.utils.XPReader(self.cfg_device.path_to_images / "Position{p}Channel{c}Frames{t}.tif")
        for i_pos, i_delta_pos in enumerate(delta_reader.positions, start=0):
            self.all_frames[i_pos] = delta_reader.getframes(position=i_delta_pos)
        self.set_pos_id_to_coordinate({i_pos: "" for i_pos in range(len(delta_reader.positions))})

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        return True

    def _initialise(self) -> None:
        self._curr_period = -1

    def _set_filter_wheel(self, i_pos: int):
        return

    def _take_frame(
            self,
            i_chan: int,
            brightness: int = 100,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        return self.all_frames[self._curr_pos][self._curr_period, i_chan, :, :]

    def autofocus_enable(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        self._autofocus_is_locked = True

    def autofocus_disable(self):
        self._autofocus_is_locked = False

    def autofocus_is_locked(self):
        return self._autofocus_is_locked

    def autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        return True

    def autofocus_unlock(self):
        self._autofocus_is_locked = False

    def coordinate_is_out_of_bounds(self, coordinate: Union[Dict[str, float], Coordinate]) -> bool:
        return False

    def disable_led(self):
        return

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10

    def get_filename(
            self,
            i_pos: Optional[int] = None,
            i_channel: Optional[Union[int, ConfigLED]] = None,
    ) -> str:
        return ""

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        return Coordinate(-np.Inf, -np.Inf, -np.Inf), Coordinate(np.Inf, np.Inf, np.Inf)

    def increment_period(self, delta_period: int = 1):
        self._curr_period += delta_period

    def set_period(self, i_period: int):
        self._curr_period = i_period

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Any]) -> bool:
        return True

    def software_focus_finalise(self):
        return

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None, # ((xmin,xmax), (ymin,ymax))
            algorithm_override: Optional[ConfigFocusAlgorithm] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        return

    def software_focus_step(self, ipos: int):
        return

    def move_home(self, block: Optional[bool] = False):
        return

    def move_fov_up(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        return

    def move_fov_down(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        return

    def move_fov_left(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        return

    def move_fov_right(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        return

    def move_to(self, coordinate: Union[Dict[str, int], Coordinate], block: Optional[bool] = False):
        return

    def keyboard_control(self):
        return

    def set_led(self, i_chan: Union[int, ConfigLED], brightness: int = 100, block: bool = False):
        return

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        return


class TestCamera(AbstractCamera):
    """
    A class to mock the acquisition of frames.
    """
    def __init__(
            self,
            cfg_device: ConfigDevice,
            filenames: List[Union[str, Path]],
            cfg_crisp: Optional[ConfigCRISP] = CRISP_CONFIG_DEFAULT,
            cfg_image: Optional[ConfigImage] = IMAGE_CONFIG_DEFAULT,
            cfg_objective: Optional[ConfigObjective] = OBJECTIVE_CONFIG_DEFAULT,
            pos_to_filename: Optional[Union[Dict[int, Union[Path, str]], None]] = None,
            cfg_focus: Optional[ConfigFocus] = FOCUS_CONFIG_DEFAULT,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ):
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image, cfg_objective=cfg_objective, cfg_focus=cfg_focus)
        if len(np.unique(filenames)) != len(filenames):
            raise ConfigError("TestCamera.__init__: must provide list with unique filenames.",
                              ErrorCode.ERROR_TEST_CAMERA_CONFIG)
        self.filenames: List[Union[str, Path]] = filenames
        "List of filenames for mock images."
        self.indices: Iterator[int] = itertools.cycle(range(len(filenames)))
        "Cyclic indices."
        self.cfg_crisp: ConfigCRISP = cfg_crisp
        "Settings for CRISP autofocus. Required for GUI interaction."
        self.pos_to_filename: Union[Dict[int, Union[Path, str]], None] = pos_to_filename
        "Optional dictionary mapping from unique position numbers (0,1,2,...) to filename."
        self._led_channel_keys: Dict[int, str] = {
            ConfigLED.LED_405_NM.value: "X",
            ConfigLED.LED_450_NM.value: "Y",
            ConfigLED.LED_505_NM.value: "Z",
            ConfigLED.LED_538_NM.value: "F",
        }
        "LED keys i_chan=0,...,3 for communication with Tiger."
        self._crop_inds: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = \
            cropping_indices if cropping_indices else None
        "Optional cropping indices applied to all images. If provided, must be of the form ((xmin, xmax), (ymin, ymax))"

        self._next_filename_index: int = next(self.indices)

        if self.pos_to_filename is not None:
            if len(pos_to_filename) != len(filenames):
                self.pos_to_filename = None
                raise ConfigError("TestCamera.__init__: if providing pos_to_filename, "
                                  "then it must have the same length as filenames.",
                                  ErrorCode.ERROR_TEST_CAMERA_CONFIG)
            self.pos_to_filename = dict(sorted(self.pos_to_filename.items()))  # warning in pycharm is a bug
            self.filenames = list(self.pos_to_filename.values())
        self.set_pos_id_to_coordinate({i_pos: "" for i_pos in range(len(self.pos_to_filename))})

    def increment_filename_index(self):
        self._next_filename_index = next(self.indices)

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        if self.pos_to_filename is not None:
            if i_pos not in self.pos_to_filename:
                raise EvoMachineError("TestCamera._move_stage_to_pos: i_pos not in pos_to_filename.")
            self._next_filename_index = i_pos
        else:
            self.increment_filename_index()
        return True

    def _initialise(self) -> None:
        return

    def _set_filter_wheel(self, i_pos: int):
        return

    def _take_frame(
            self,
            i_chan: int,
            brightness: int = 100,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        image = skimage.io.imread(self.filenames[self._next_filename_index])
        if self._crop_inds:
            return image[self._crop_inds[0][0]:self._crop_inds[0][1], self._crop_inds[1][0]:self._crop_inds[1][1]]
        else:
            return image

    def get_filename(
            self,
            i_pos: Optional[int] = None,
            i_channel: Optional[Union[int, ConfigLED]] = None,
    ) -> str:
        return str(self.filenames[self._next_filename_index]).split("/")[-1]

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        return {ax.upper(): 0.0 for ax in axes}

    def get_led_channels(self) -> Tuple[int]:
        return tuple(self._led_channel_keys.keys())

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        return Coordinate(-np.Inf, -np.Inf, -np.Inf), Coordinate(np.Inf, np.Inf, np.Inf)

    def halt_stage(self):
        return

    def coordinate_is_out_of_bounds(self, coordinate: Dict[str, float]) -> bool:
        return False

    def disable_led(self):
        return

    def autofocus_enable(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        self._autofocus_is_locked = True

    def autofocus_disable(self):
        self._autofocus_is_locked = False

    def autofocus_is_locked(self):
        return self._autofocus_is_locked

    def autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        return True

    def autofocus_unlock(self):
        self._autofocus_is_locked = False

    def software_focus_finalise(self):
        return

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None, # ((xmin,xmax), (ymin,ymax))
            algorithm_override: Optional[ConfigFocusAlgorithm] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        return

    def software_focus_step(self, ipos: int):
        return

    def move_home(self, block: Optional[bool] = False):
        self.increment_filename_index()

    def move_fov_up(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self.increment_filename_index()

    def move_fov_down(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self.increment_filename_index()

    def move_fov_left(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self.increment_filename_index()

    def move_fov_right(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self.increment_filename_index()

    def move_to(self, coordinate: Dict[str, int], block: Optional[bool] = False):
        self.increment_filename_index()

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10

    def keyboard_control(self):
        return

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        return

    def set_led(self, i_chan: Union[int, ConfigLED], brightness: int = 100):
        return

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Any]) -> bool:
        return True

    def zero_coordinates(self):
        return


class EvoCamera(AbstractCamera):
    """
    EvoMachine acquisition class.
    """
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: Optional[ConfigImage] = IMAGE_CONFIG_DEFAULT,
            cfg_crisp: Optional[ConfigCRISP] = CRISP_CONFIG_DEFAULT,
            cfg_focus: Optional[ConfigFocus] = FOCUS_CONFIG_DEFAULT,
            cfg_objective: Optional[ConfigObjective] = OBJECTIVE_CONFIG_DEFAULT,
    ):
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image, cfg_objective=cfg_objective, cfg_focus=cfg_focus)

        self.tiger: Optional[asitiger.tigercontroller.TigerController, asitiger.tigerthread.TigerThread] = None
        "Object for serial communication with ASI tiger."
        self._tiger_is_alive: bool = False
        "Flag set in _initialise."
        self._is_multi_threaded: bool = False
        "If true, will use Threading wrappers for objects like self.tiger."
        self.current_channel: int = -1
        "Current LED channel set."
        self._last_frame_channel: int = -1
        "Channel used to take last frame."
        self._led_channel_keys: Dict[int, Union[str, None]] = {
            ConfigLED.LED_405_NM.value: "X",
            ConfigLED.LED_450_NM.value: "Y",
            ConfigLED.LED_505_NM.value: "Z",
            ConfigLED.LED_538_NM.value: "F",
            ConfigLED.LED_NO_LED.value: None,
        }
        "LED keys i_chan=0,...,3 for communication with Tiger."
        self._current_led_brightness: int = 0
        "Last set LED brightness"
        self.card_address_led: int = 7
        "LED card address on ASI tiger."
        self.card_address_fw: int = 8
        "Filter wheel card address on ASI tiger."
        self.filter_wheel_settings: Dict[int, str] = {0: "Filter", 1: "Blocking", 2: "Nothing"}
        "Available filter wheels."
        self.card_address_crisp: int = 2
        "CRISP card address on ASI tiger."
        self.cfg_crisp: ConfigCRISP = cfg_crisp
        "Settings for CRISP autofocus."
        self._cfg_focus: ConfigFocus = cfg_focus.copy()
        "Internal focus configuration used for overrides."

        self.focus_cols: Optional[Tuple[int]] = None
        "Initialised in software_focus. Contains min/max indices of image to apply focus routine to."
        self.focus_curr_pos: Optional[Dict[str, int]] = None
        "Initialised in software_focus. X/Y/Z coordinates of stage before last start of focus routine."
        self.focus_old_channel: Optional[int] = None
        "Initialised in software_focus. LED channel before last start of focus routine."
        self.focus_rows: Optional[Tuple[int]] = None
        "Initialised in software_focus. Contains min/max indices of image to apply focus routine to."
        self._focus_is_initialised: bool = False
        "Changes to True after initialisation and to False after finalisation."

        self._pos_id_to_coordinate: Dict[int, Coordinate] = {}
        "Dictionary position ID -> Coordinate. Initialise through set_pos_id_to_coordinate."

        self.mmc: Union[Core, None] = None
        "Micromanager Core object for taking images."
        self.studio: Union[Studio, None] = None
        "Micromanager Studio object for additional functions."
        self._mmc_is_alive: bool = False
        "Flag set in _initialise."

        self.initialise()  # Must be called before using EvoCamera

    def _initialise(self) -> None:
        """ Initialises EvoCamera objects with peripherals. Tests connections and sets is_alive flags. """
        # Tiger box communication
        try:
            if self._is_multi_threaded:
                self.tiger: asitiger.tigerthread.TigerThread = \
                    asitiger.tigerthread.TigerThread.from_serial_port(port=self.cfg_device.tiger_port)
            else:
                self.tiger: asitiger.tigercontroller.TigerController = \
                    asitiger.tigercontroller.TigerController.from_serial_port(port=self.cfg_device.tiger_port)
        except Exception as e:
            self._tiger_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to Tiger: {e}.")
            self.error_container.add_error(
                new_error=TigerError(message=str(e), error_code=ErrorCode.ERROR_TIGER_SERIAL_CONNECTION)
            )
        if not self._get_tiger_is_alive():
            self._tiger_is_alive = False
            logger.warning("EvoCamera._initialise: Tiger is not alive.")
            self.error_container.add_error(
                new_error=TigerError(message="Tiger is not alive.", error_code=ErrorCode.ERROR_TIGER_NOT_ALIVE)
            )
        else:
            self._tiger_is_alive = True
        # Camera communication
        try:
            self.mmc = Core()
            self.studio = Studio()
            self._mmc_is_alive = True
        except Exception as e:
            self._mmc_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to MMC: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_MMC_NOT_ALIVE)
            )
        self.disable_led()
        self.set_exposure()

    def _get_tiger_is_alive(self) -> bool:
        if not self.tiger:
            return False
        try:
            _ = self.tiger.status()
            return True
        except ValueError:
            return False

    def _set_filter_wheel(self, i_pos: int):
        if i_pos not in self.filter_wheel_settings.keys():
            logger.error(msg=f"EvoCamera._set_filter_wheel: i_pos={i_pos} not in wheels={self.filter_wheel_settings}.")
            return
        if self._tiger_is_alive:
            self.tiger.filter_wheel(position=i_pos, card_address=self.card_address_fw)
        else:
            logger.error(msg=f"EvoCamera._set_filter_wheel: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")

    def disable_led(self):
        self.set_led(i_chan=-1)

    def _move_fov(self, x_or_y: str, sign: int, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        pos = self.tiger.where([x_or_y.upper()])
        if x_or_y.upper() not in pos:
            raise TigerError(f"EvoCamera._move_fov: queried coordinate {x_or_y.upper()} not in response: {pos}.",
                             ErrorCode.ERROR_TIGER_NO_DATA)
        pos[x_or_y.upper()] += int(sign * self.cfg_objective.fov_size * 10 * multiplier)
        self.move_to(coordinate=pos, block=block)

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        return self._move_stage_to_coord(self._pos_id_to_coordinate[i_pos].to_dict())

    def _move_stage_to_coord(
            self,
            coordinates: Dict[str, int],
    ) -> bool:
        answer = None
        if self._tiger_is_alive:
            answer = self.tiger.move(coordinates=coordinates)
        else:
            logger.error(msg=f"EvoCamera._move_stage_to_coord: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")
        return True if isinstance(answer, str) else False

    def _take_frame(
            self,
            i_chan: Union[int, None],
            brightness: int = 100,
            block: bool = False,
            reset_led: bool = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        if not self._mmc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        if i_chan is not None:
            self._last_frame_channel = i_chan
            curr_channel = self.current_channel
            self.set_led(i_chan=i_chan, brightness=brightness, block=block)
        try:
            self.mmc.snap_image()
        except Exception as e:
            logger.warning(f"EvoCamera._take_frame: Received exception:\n{e}\nHave you disabled MM live mode?")
            return None
        tagged_image = self.mmc.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        if i_chan is not None and reset_led:
            self.set_led(i_chan=curr_channel, block=False)
        return pixels

    def coordinate_is_out_of_bounds(self, coordinate: Union[Dict[str, float], Coordinate]) -> bool:
        return self.tiger.coordinate_is_out_of_bounds(
            coordinate.to_dict() if isinstance(coordinate, Coordinate) else coordinate
        )

    def autofocus_enable(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_enable: Device not alive.")
            return

        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg_crisp
        ask_user = cfg_crisp.user_input and user_input

        if ask_user:
            user_input_str = input("Starting CRISP autofocus. Do you want to proceed? (yes/no): ")
            if user_input_str.lower() == "yes":
                logger.info("CRISP: Proceeding with configuring and setting up CRISP autofocus.")
            else:
                logger.info("CRISP: Aborting CRISP configuration.")
                return
        is_configured = self.autofocus_configure(this_cfg_crisp=cfg_crisp)
        if not is_configured:
            return

        logger.info("CRISP: Setting IDLE status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.IDLE)
        time.sleep(cfg_crisp.pause_short)
        logger.info("CRISP: Setting LOG_CAL status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.LOG_CAL)
        time.sleep(cfg_crisp.pause_long)
        val = self.tiger.crisp_get_snr(card_address=self.card_address_crisp)
        if val < cfg_crisp.min_snr:
            logger.warning(f"EvoCamera.autofocus: Low SNR = {val:.2d}. Increase CRISP LED intensity and repeat.")
        logger.info("CRISP: Setting DITHER status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.DITHER)
        time.sleep(cfg_crisp.pause_long)
        val = self.tiger.crisp_get_err(card_address=self.card_address_crisp)
        if np.abs(val) < cfg_crisp.min_error:
            logger.warning(f"EvoCamera.autofocus: Low error = {val}. Check ASI guide.")
        logger.info("CRISP: Setting SET_GAIN status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.SET_GAIN)
        time.sleep(cfg_crisp.pause_short)

        do_lock = True
        if ask_user:
            user_input_str = input("Do you want to lock CRISP autofocus? (yes/no): ")
            do_lock = True if user_input_str.lower() == "yes" else False
        if do_lock:
            logger.info("CRISP: Setting LOCK status.")
            self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.LOCK)
        else:
            logger.info("CRISP: Setting UNLOCK status.")
            self.autofocus_unlock()
        time.sleep(cfg_crisp.pause_short)
        curr_state = self.tiger.autofocus_get_set_state(card_address=self.card_address_crisp, value=None)
        logger.info(f"CRISP: Finalising autofocus. Current state is {curr_state}.")

    def autofocus_disable(self):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_disable: Device not alive. Trying to disable anyway.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.IDLE)

    def autofocus_is_locked(self):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_is_locked: Device not alive.")
            return
        return self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None) == 'F'

    def autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_configure: Device not alive.")
            return False
        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg_crisp
        try:
            cfg_crisp.check_config()
            logger.info(f"CRISP: Configuring CRISP with following parameters:\n{cfg_crisp}")
        except ConfigError as e:
            logger.error(f"CRISP: Bad configuration:\n{e}\nCannot use CRISP.")
            return False
        self.autofocus_unlock()
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_objective_na(card_address=self.card_address_crisp, value=cfg_crisp.objective_na)
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_led_intensity(card_address=self.card_address_crisp, value=cfg_crisp.led_intensity)
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_loop_gain(card_address=self.card_address_crisp, value=cfg_crisp.loop_gain)
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_num_avg(card_address=self.card_address_crisp, value=cfg_crisp.averaging)
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_update_rate(card_address=self.card_address_crisp, value=cfg_crisp.update_rate)
        time.sleep(cfg_crisp.pause_short)
        self.tiger.crisp_get_set_lock_range(card_address=self.card_address_crisp, value=cfg_crisp.lock_range)
        time.sleep(cfg_crisp.pause_short)
        new_cfg = ConfigCRISP(
            objective_na=self.tiger.crisp_get_set_objective_na(card_address=self.card_address_crisp, value=None),
            led_intensity=self.tiger.crisp_get_set_led_intensity(card_address=self.card_address_crisp, value=None),
            loop_gain=self.tiger.crisp_get_set_loop_gain(card_address=self.card_address_crisp, value=None),
            averaging=self.tiger.crisp_get_set_num_avg(card_address=self.card_address_crisp, value=None),
            update_rate=self.tiger.crisp_get_set_update_rate(card_address=self.card_address_crisp, value=None),
            lock_range=self.tiger.crisp_get_set_lock_range(card_address=self.card_address_crisp, value=None),
            min_snr=cfg_crisp.min_snr,
            min_error=cfg_crisp.min_error,
        )
        logger.info(f"CRISP: Parameters set to:\n{new_cfg}")
        return True

    def finalise(self):
        if self._is_multi_threaded:
            self.tiger.stop()
            self.tiger.join()

    def autofocus_unlock(self):
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.UNLOCK)

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        return self.tiger.where(axes)

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10

    def get_filename(
            self,
            i_pos: Optional[int] = None,
            i_channel: Optional[Union[int, ConfigLED]] = None,
    ) -> str:
        if i_pos is not None:
            pos = self._pos_id_to_coordinate[i_pos].to_dict()
        else:
            pos = self.tiger.where(['X', 'Y', 'Z'])
        if i_channel is not None:
            if isinstance(i_channel, ConfigLED):
                i_channel = i_channel.value
        else:
            i_channel = self._last_frame_channel
        return "{}_P{}_X{}_Y{}_Z{}_{}.tiff".format(
            ConfigLED.get_name(value_to_find=i_channel).replace("_", ""),
            i_pos if i_pos is not None else "",
            pos['X'],
            pos['Y'],
            pos['Z'],
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        )

    def get_led_channels(self) -> Tuple[int]:
        return tuple(self._led_channel_keys.keys())

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        lim = self.tiger.get_stage_limits()
        return Coordinate(lim['X'][0], lim['Y'][0], lim['Z'][0]), Coordinate(lim['X'][1], lim['Y'][1], lim['Z'][1])

    def halt_stage(self):
        self.tiger.halt()

    def keyboard_control(self):
        def on_key_release(key):
            try:
                delta_pos = 100
                do_move = True
                pos = self.tiger.where(['X', 'Y'])
                print(f"X = {pos['X'] / 10:06.1f} μm, Y = {pos['Y'] / 10:06.1f} μm", end='\r')
                if key == keyboard.KeyCode.from_char('w'):
                    pos['Y'] -= delta_pos
                elif key == keyboard.KeyCode.from_char('s'):
                    pos['Y'] += delta_pos
                elif key == keyboard.KeyCode.from_char('a'):
                    pos['X'] -= delta_pos
                elif key == keyboard.KeyCode.from_char('d'):
                    pos['X'] += delta_pos
                elif (key == keyboard.Key.esc) or (key == keyboard.KeyCode.from_char('q')):
                    return False
                else:
                    do_move = False
                if do_move:
                    self.tiger.move(coordinates=pos)
                    time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Exception: {e}\n")
                return False

        with keyboard.Listener(on_release=on_key_release, suppress=True) as listener:
            listener.join()

    def move_home(self, block: Optional[bool] = False):
        _ = self.tiger.home()
        if block:
            self.tiger.wait_until_idle()

    def move_fov_up(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='Y', sign=-1, multiplier=multiplier, block=block)

    def move_fov_down(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='Y', sign=+1, multiplier=multiplier, block=block)

    def move_fov_left(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='X', sign=-1, multiplier=multiplier, block=block)

    def move_fov_right(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='X', sign=+1, multiplier=multiplier, block=block)

    def move_to(self, coordinate: Union[Dict[str, int], Coordinate], block: Optional[bool] = False):
        if isinstance(coordinate, Coordinate):
            coordinate = coordinate.to_dict()
        if not isinstance(coordinate, Dict) or not all(k in ['X', 'Y', 'Z'] for k in coordinate.keys()):
            raise TigerError(f"EvoCamera.move_to: Badly formatted coordinates: {coordinate}.",
                             ErrorCode.ERROR_WRONG_FORMAT)
        self.tiger.move(coordinates=coordinate)
        if block:
            self.tiger.wait_until_idle()

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        if self._mmc_is_alive:
            self.mmc.set_exposure(exposure_time)
        else:
            logger.warning("EvoCamera._set_exposure: cannot set exposure as MMC is not alive.")
            # TODO raise error?

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Coordinate]) -> bool:
        for i_pos, coord in pos_id_to_coordinate.items():
            if not coord.has_z():
                logger.warning(f"EvoCamera.set_pos_id_to_coordinate: Position {i_pos} is missing Z "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
            if self.coordinate_is_out_of_bounds(coord):
                logger.warning(f"EvoCamera.set_pos_id_to_coordinate: Position {i_pos} is out of bounds. "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
        self._pos_id_to_coordinate = {key: val for key, val in pos_id_to_coordinate.items()}
        return True

    def set_led(self, i_chan: Union[int, ConfigLED], brightness: int = 100, block: bool = False):
        if isinstance(i_chan, ConfigLED):
            i_chan = i_chan.value
        if i_chan not in self._led_channel_keys.keys():
            logger.error(msg=f"EvoCamera._set_channel: i_chan={i_chan} not in channels={self._led_channel_keys.keys()}.")
            return
        if self._tiger_is_alive:
            led_settings = {val: (brightness if ((key == i_chan) and (i_chan != ConfigLED.LED_NO_LED.value)) else 0)
                            for key, val in self._led_channel_keys.items()}
            if (0 <= brightness <= 100) or (i_chan != ConfigLED.LED_NO_LED.value):
                is_good_brightness_value = True
            else:
                is_good_brightness_value = False
            if is_good_brightness_value:
                self.tiger.led(led_brightnesses=led_settings, card_address=self.card_address_led)
                self.current_channel = i_chan
                self._current_led_brightness = 0 if i_chan == ConfigLED.LED_NO_LED.value else brightness
            else:
                logger.error(msg=f"Cannot set brightness: {brightness} is out of range [0, 100]. LED not set.")
        else:
            logger.error(msg=f"EvoCamera._set_channel: Tiger is not alive. Check ASI Tiger box and serial connection.")
        if block:
            self.tiger.wait_until_idle()

    def software_focus_is_valid_range(self) -> bool:
        if self.focus_Z_coords is None:
            return False
        else:
            for z_coord in [self.focus_Z_coords[0], self.focus_Z_coords[-1]]:
                is_out_of_bounds = self.tiger.coordinate_is_out_of_bounds({'Z': z_coord})
                if is_out_of_bounds:
                    logger.warning(
                        f"EvoCamera.software_focus_check_range: Coordinates are out of bounds. "
                        f"Received min, max = {[self.focus_Z_coords[0], self.focus_Z_coords[-1]]}. "
                        f"Stage limits = {self.tiger.get_stage_limits()}). Reset stage limits."
                    )
                    return False
        return True

    def software_focus_finalise(self):
        self._focus_is_initialised = False
        focus_best_coordinate = self.get_software_focus_z_coord()
        if focus_best_coordinate is None or self.coordinate_is_out_of_bounds({'Z': focus_best_coordinate}):
            logger.error(f"EvoCamera.software_focus_finalise: invalid z coordinate {focus_best_coordinate}.")
            return
        logger.info(f"EvoCamera.software_focus: Finished scanning. Coordinate before focus="
                    f"{self.focus_curr_pos['Z'] / 10} μm,"
                    f"coordinate after focus={focus_best_coordinate / 10} μm. Finalising software_focus.")
        self._move_stage_to_coord({'Z': focus_best_coordinate})
        self.set_led(i_chan=self.focus_old_channel)

        # FIXME remove below
        import pickle
        with open(str(self.cfg_device.path_to_save / "FOCUS_STACK_GUI/pickled_data.pkl"), 'wb') as f:
            data = (
                self.focus_Z_coords,
                self.focus_stack,
                self.focus_scores,
            )
            pickle.dump(data, f)

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None, # ((xmin,xmax), (ymin,ymax))
            algorithm_override: Optional[ConfigFocusAlgorithm] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        """
        Initialises software focus routine and validates parameter and limits.

        Parameters
        ----------
        cfg_focus               : ConfigFocus object. If None, uses self.cfg_focus provided in constructor.
        focus_channel_override  : Overrides cfg_focus.focus_channel.
        rel_range_override      : Overrides cfg_focus.rel_range.
        cropping_indices        : Apply focus to img[]
        algorithm_override
        user_input_override
        countdown_override

        Returns
        -------

        """
        if not all([self._mmc_is_alive, self._tiger_is_alive]):
            logger.error(f"EvoCamera.software_focus: Device(s) not alive. "
                         f"Tiger={self._tiger_is_alive}, MMC={self._mmc_is_alive}.")
            return
        # Assign optional arguments
        if cropping_indices is None:
            self.focus_rows = 0, self.cfg_image.pxl_vert
            self.focus_cols = 0, self.cfg_image.pxl_horiz
        else:
            logger.info("EvoCam.software_focus_initialise: using cropping boxes for focus.")
            self.focus_rows = (cropping_indices[1][0], cropping_indices[1][1])
            self.focus_cols = (cropping_indices[0][0], cropping_indices[0][1])
        self._cfg_focus = cfg_focus.copy() if cfg_focus else self.cfg_focus.copy()
        if focus_channel_override is not None:
            self._cfg_focus.focus_channel = focus_channel_override
        if algorithm_override is not None:
            self._cfg_focus.algorithm = algorithm_override
        if rel_range_override is not None:
            self._cfg_focus.rel_range = rel_range_override
        try:
            self._cfg_focus.check_config()
        except ConfigError as e:
            logger.warning(f"EvoCamera.software_focus: Invalid focus configuration:\n{e.message}\nAborting...")
            return
        # Initialise Z stack
        self.focus_curr_pos = self.tiger.where()
        self.focus_Z_coords = range(
            self.focus_curr_pos['Z']-self._cfg_focus.rel_range,
            self.focus_curr_pos['Z']+self._cfg_focus.rel_range,
            self._cfg_focus.steps_size,
        )
        if not self.software_focus_is_valid_range():
            self._focus_is_initialised = False
            return
        if user_input_override:
            if not countdown_override:
                sleep_time = 5
                logger.warning(f"ThreadSWFocus.run: Starting software autofocus configured as\n"
                               f"{self._cfg_focus.__str__()}\nThis will move the stage up and down in the range "
                               f"[{(self.focus_curr_pos['Z'] - self._cfg_focus.rel_range) / 10},"
                               f"{(self.focus_curr_pos['Z'] + self._cfg_focus.rel_range) / 10}] μm"
                               f" (current position = {self.focus_curr_pos['Z'] / 10} μm). "
                               f"If there are objects blocking the stage movement, this will crash the "
                               f"objective and break it. You have {sleep_time} seconds to press STOP. ")
                for i in range(sleep_time, 0, -1):
                    logger.warning(f"ThreadSWFocus.run: Starting software focus in {i} s.")
                    time.sleep(1)
        else:
            user_input = input(f"EvoCamera.software_focus: Starting software autofocus configured as\n"
                               f"{self._cfg_focus.__str__()}\nThis will move the stage up and down in the range "
                               f"[{(self.focus_curr_pos['Z']-self._cfg_focus.rel_range)/10},"
                               f"{(self.focus_curr_pos['Z']+self._cfg_focus.rel_range)/10}] μm"
                               f" (current position = {self.focus_curr_pos['Z']/10} μm). "
                               f"If there are objects blocking the stage movement, this will crash the "
                               f"objective and break it. Do you want to proceed? (yes/no): ")
            if user_input.lower() == "yes":
                logger.info("EvoCamera.software_focus: Proceeding with software focus. Disabling MM live mode.")
            else:
                logger.info("EvoCamera.software_focus: Aborting software focus.")
                return
        self.focus_old_channel = self.current_channel
        self.set_exposure(exposure_time=int(self._cfg_focus.exposure_time))
        self.studio.live().set_live_mode(False)
        self.focus_scores = np.zeros(len(self.focus_Z_coords))
        self.focus_stack = np.zeros(shape=(self.cfg_image.pxl_vert, self.cfg_image.pxl_horiz, len(self.focus_Z_coords)),
                                    dtype=np.float64)
        self.set_led(i_chan=ConfigLED.LED_NO_LED.value)
        self.focus_prev_image = self.display_save_frame(
            i_chan=self._cfg_focus.focus_channel,
            path_to_save=False,
            filename=None,
            display_frame=False,
        ).astype(np.float64)
        self._focus_is_initialised = True

    def software_focus_step(self, ipos: int):
        z_coord = self.focus_Z_coords[ipos]
        self.move_to(coordinate={'Z': z_coord}, block=True)
        image_raw = self.display_save_frame(
            i_chan=self._cfg_focus.focus_channel,
            path_to_save=False,
            filename=None,
            display_frame=False,
        )
        if image_raw is None:
            logger.warning("EvoCamera.software_focus: self._take_frame returned None. Aborting...")
            return
        else:
            self.focus_stack[:, :, ipos] = image_raw
        self.focus_scores[ipos] = get_focus_score(
            img=self.focus_stack[self.focus_rows[0]:self.focus_rows[1], self.focus_cols[0]:self.focus_cols[1], ipos],
            algorithm=self._cfg_focus.algorithm,
        )

    def zero_coordinates(self):
        self.tiger.zero()







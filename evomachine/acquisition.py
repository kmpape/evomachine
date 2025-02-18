from datetime import datetime
import itertools
import numpy as np
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
from pycromanager import Core, Studio
import skimage
from serial import SerialException

from asitiger.command import CRISPState
from asitiger.status import CRISPStatus
import asitiger.tigercontroller

from syncboard.syncboardcontroller import SyncBoardController, LED_ID
from KWR103Driver import KWR103

from evomachine.config import ConfigCamera, ConfigCRISP, ConfigFocus, get_logger
from evomachine.coordinates import Coordinate
from evomachine.exceptions import CameraError, ConfigError, ErrorCode, ErrorContainer, \
    EvoMachineError, StageError, TigerError, SyncBoardError
from evomachine.software_focus import get_focus_score, get_focus_score_is_good, get_focus_curve_type
from evomachine.utils import EvoCroppingBox, list_serial_ports, get_psu_port
from evomachine.evotypes import AutoFocusStatusType, FilterWheelType, FocusAlgorithmType, ImageConfigType, LEDType, \
    FocusStatusType, FocusCurveType

from pyvcam import pvc
from pyvcam.camera import Camera
from pyvcam import constants

logger = get_logger(name=__name__)


class AbstractCamera:
    def __init__(
            self,
            cfg_camera: ConfigCamera,
    ):
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self.cfg: ConfigCamera = cfg_camera
        "Configuration object for the camera."
        self._is_initialised: bool = False
        "Set by initialise(). Query status through is_initialised()."

        self._crisp_initialised: bool = False
        "Flag to indicate whether CRISP is initialised."

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
        self._current_filter_type: FilterWheelType = FilterWheelType.FILTER
        "Currently set filter type. Note: changes from ASI Tiger are NOT registered."

        self.focus_scores: Union[None, np.ndarray] = None
        "Initialised in software_focus. Contains the focus score of each image. Larger score = sharper image."
        self.focus_stack: Union[None, np.ndarray] = None
        "Initialised in software_focus. Contains images of focus stack."
        self.focus_prev_image: Union[None, np.ndarray] = None
        "Initialised in software_focus. Contains the image from before starting focus."
        self.focus_Z_coords: Union[None, np.ndarray] = None
        "Initialised in software_focus. Contains Z coordinates of focus stack. Use focus_curr_pos for X/Y coordinates."
        self.focus_curr_pos: Optional[Dict[str, int]] = None
        "Initialised in software_focus. X/Y/Z coordinates of stage before last start of focus routine."
        self.focus_cropping_box: Union[None, EvoCroppingBox] = None
        "Initialised in software_focus. Cropping box applied to focus images."
        self._focus_old_channel: Union[LEDType, None] = None
        "Initialised in software_focus. LED channel before last start of focus routine."
        self._focus_is_initialised: bool = False
        "Changes to True after initialisation and to False after finalisation."
        self._focus_status: FocusStatusType = FocusStatusType.UNKNOWN
        "Flag can be queries through get_software_focus_status() and is set during software_focus()."
        self._focus_curve_status: FocusCurveType = FocusCurveType.UNKNOWN
        "Flag can be queries through get_focus_curve_status() and is set during software_focus()."

        self.coord_pre_autofocus_lock: Dict[str, float] | None = None
        "Coordinate before autofocus lock."
        self.coord_post_autofocus_lock: Dict[str, float] | None = None
        "Coordinate after autofocus lock."
        self.coord_pre_autofocus_config: Dict[str, float] | None = None
        "Coordinate before autofocus configuration."
        self.coord_post_autofocus_config: Dict[str, float] | None = None
        "Coordinate after autofocus configuration."

    def autofocus_initialise(
            self,
            this_cfg_crisp: Optional[ConfigCRISP] = None,
            user_input: Optional[bool] = True
    ) -> bool:
        """

        Parameters
        ----------
        this_cfg_crisp : ConfigCRISP
            Provide ConfigCRISP to override self.cfg.autofocus
        user_input : bool
            Ask for user input to lock autofocus. Otherwise, autofocus is NOT locked.
        Returns
        -------
        success : bool
            True if initialisation was successful.
        """
        raise NotImplementedError()

    def autofocus_disable(self):
        raise NotImplementedError()

    def autofocus_get_status(self) -> AutoFocusStatusType:
        raise NotImplementedError()

    def autofocus_is_locked(self):
        raise NotImplementedError()

    def autofocus_configure(self, this_cfg_crisp: ConfigCRISP | None = None) -> bool:
        self.coord_pre_autofocus_config = self.get_coordinates(['X', 'Y', 'Z'])
        res = self._autofocus_configure(this_cfg_crisp=this_cfg_crisp)
        self.coord_post_autofocus_config = self.get_coordinates(['X', 'Y', 'Z'])
        msg = f"AbstractCamera.autofocus_configure: Coordinate " \
              f"before = {self.coord_pre_autofocus_config}, after = {self.coord_post_autofocus_config}"
        logger.info(msg)
        return res

    def _autofocus_configure(self, this_cfg_crisp: ConfigCRISP | None = None) -> bool:
        raise NotImplementedError()

    def autofocus_lock(self):
        self.coord_pre_autofocus_lock = self.get_coordinates(['X', 'Y', 'Z'])
        self._autofocus_lock()
        self.coord_post_autofocus_lock = self.get_coordinates(['X', 'Y', 'Z'])
        msg = f"AbstractCamera.autofocus_configure: Coordinate " \
              f"before = {self.coord_pre_autofocus_lock}, after = {self.coord_post_autofocus_lock}"
        logger.info(msg)

    def _autofocus_lock(self):
        raise NotImplementedError()

    def autofocus_unlock(self):
        raise NotImplementedError()

    def coordinate_is_out_of_bounds(self, coordinate: Union[Dict[str, float], Coordinate]) -> bool:
        raise NotImplementedError()

    def is_initialised(self):
        return self._is_initialised

    def initialise(self):
        """
        Initialise devices. Sets is_initialised() flag to true if successful. Also, automatically sets the filter wheel.
        """
        self.reset_counter()
        self._is_initialised = self._initialise()
        if self._is_initialised:
            self.set_filter_wheel(FilterWheelType.FILTER)

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logger.warning(msg=msg)
        else:
            logger.warning("No errors for acquisition found.")

    def disable_led(self):
        raise NotImplementedError()

    def disable_live_mode(self):
        raise NotImplementedError()

    def finalise(self):
        self._finalise()
        self._is_initialised = False

    def _finalise(self):
        raise NotImplementedError()

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        """Returns the current coordinates of the stage."""
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

    def get_software_focus_status(self) -> FocusStatusType:
        return self._focus_status

    def get_software_focus_curve_status(self) -> FocusCurveType:
        return self._focus_curve_status

    def get_default_filename(self) -> str:
        return self.get_filename(i_pos=None)

    def get_filename(
            self,
            i_pos: int | None = None,
            i_channel: LEDType | None = None,
            suffix: str | None = None,
            filter_wheel: FilterWheelType | None = None
    ) -> str:
        return "evom_pos{:02d}_{}_{}{}{}.tiff".format(
            self._curr_pos,
            i_channel if i_channel is not None else "nopos",
            "" if filter_wheel is None else f"F{filter_wheel.value}",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
            suffix if suffix is not None else "",
        )

    @staticmethod
    def add_filename_suffix(filename: str, filename_suffix: str) -> str:
        if '.' not in filename:
            filename = filename + filename_suffix
        else:
            last_period_index = filename.rfind(".")
            stem = filename[:last_period_index]
            extension = filename[last_period_index + 1:]
            if extension.isalpha():
                filename = f"{stem}{filename_suffix}.{extension}"
            else:
                filename = f"{stem}.{extension}{filename_suffix}"
        return filename

    def get_frame(
            self,
            i_chan: Union[LEDType, None],
            normalise: bool = False,
            brightness: float = 29,
            block: bool = False,
            reset_led: bool = True,
            disable_led: bool = False,
    ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
        """

        Parameters
        ----------
        i_chan:         LED channel.
        normalise:      Return normalised image.
        brightness:     Brightness value between 0 and 100.
        block:          Wait for devices before returning.
        reset_led:      Reset LED to previously set channel
        disable_led:    Disable LED before returning. Overrides reset_led.

        Returns
        -------
        frame:          Image if successful, otherwise None.
        """
        self._step += 1
        frame = self._take_frame(
            i_chan=i_chan,
            brightness=brightness,
            block=block,
            reset_led=reset_led,
            disable_led=disable_led,
        )
        return self.normalise_frame(frame=frame) if (normalise and (frame is not None)) else frame

    def get_pos(self) -> int:
        return self._curr_pos

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        raise NotImplementedError()

    def halt_stage(self):
        raise NotImplementedError()

    def display_save_frame(
            self,
            i_chan: Union[LEDType, None],
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

    def move_to_pos(self, i_pos: int, block: bool = True) -> None:
        if i_pos not in self._pos_id_to_coordinate:
            raise StageError("Position index {} out of range".format(i_pos),
                             ErrorCode.ERROR_STAGE_COORDINATES)
        success = self._move_stage_to_pos(i_pos=i_pos, block=block)
        if not success:
            raise StageError("Fault moving to position={}.".format(i_pos), ErrorCode.ERROR_STAGE_MOVEMENT)
        self._curr_pos = i_pos

    @staticmethod
    def normalise_frame(
            frame: np.ndarray[(int, int), 'ImageConfigType.pxl_dtype'],
            colormap: Union['plt.cm', bool, None] = True,
    ) -> np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']:
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
        norm = plt.Normalize(vmin=frame.min(), vmax=frame.max())  # noqa
        if (colormap is None) or (isinstance(colormap, bool) and not colormap):
            return norm(frame)
        elif isinstance(colormap, bool) and colormap:
            return plt.cm.jet(norm(frame))  # noqa
        else:
            return colormap(norm(frame))

    def plot_frame(self, frame: np.ndarray[(int, int), 'ImageConfigType.pxl_dtype'], normalise: bool = True):
        image = self.normalise_frame(frame=frame) if normalise else frame
        plt.imshow(image)
        plt.show()

    def reset_counter(self):
        self._step = -1

    def save_frame(
            self,
            frame: np.ndarray,
            path_to_save: Path | str | bool = True,
            filename: str | None = None,
            filename_suffix: str | None = None,
            i_pos: int | None = None,
            i_channel: LEDType | None = None,
            filter_wheel: FilterWheelType | None = None,
    ) -> None:
        """
        Image is saved under path_to_save / filename. See arguments for different options. If provided, checks whether
        path_to_save exists.

        Parameters
        ----------
        frame : np.ndarray
            Image to save.
        path_to_save : Path | str | bool
            If bool and true, uses cfg.path_to_save. Returns on bool and false.
        filename : str | None
            Uses a get_filename() if None. Note: this method can be overwritten by child classes.
        filename_suffix : str | None
            Adds a suffic to the filename stem if not None.
        i_pos : int | None,
            Use i_pos for get_filename().
        i_channel : LEDType | None
            Use i_channel for get_filename().
        i_channel : LEDType | None
            Use filter_wheel for get_filename().
        Returns
        -------

        """
        if not filename:
            filename = self.get_filename(i_pos=i_pos, i_channel=i_channel)

        if filename_suffix is not None:
            filename = self.add_filename_suffix(filename=filename, filename_suffix=filename_suffix)

        if isinstance(path_to_save, str):
            path_to_save = Path(path_to_save)
        elif isinstance(path_to_save, bool):
            if not path_to_save:
                return
            path_to_save = self.cfg.path_to_save
        if not path_to_save.exists():
            logger.warning(f"AbstractCamera.save_frame: Path {path_to_save} does not exist. "
                           f"Returning image without saving...")
            return
        logger.debug(f"Saving image {path_to_save / filename}.")
        if '.tif' in filename:
            skimage.io.imsave(path_to_save / filename, frame, plugin="tifffile", check_contrast=False)
        else:
            skimage.io.imsave(path_to_save / filename, frame, check_contrast=False)

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[LEDType] = None,
            rel_range_override: Optional[int] = None,
            cropping_box: Optional[EvoCroppingBox] = None,
            algorithm_override: Optional[FocusAlgorithmType] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        self.software_focus_initialise(
            cfg_focus=cfg_focus,
            focus_channel_override=focus_channel_override,
            rel_range_override=rel_range_override,
            cropping_box=cropping_box,
            algorithm_override=algorithm_override,
            user_input_override=user_input_override,
            countdown_override=countdown_override,
        )
        if not self._focus_is_initialised:
            logger.error(
                f"software_focus: Focus not initialised or initialisation failed. Check log. Aborting focus."
            )
            return

        # Image and compute focus scores
        self.set_led(i_chan=cfg_focus.focus_channel, brightness=cfg_focus.brightness)
        for ipos in range(len(self.focus_Z_coords)):
            success = self.software_focus_step(ipos=ipos)
            if not success:
                logger.error(
                    f"Error during software focus step. Check log. Aborting focus."
                )
                return

        # Verify focus curve

        self.software_focus_finalise()

    def software_focus_finalise(self, move_on_any_focus_status: bool = True, debug_save: bool = True):
        raise NotImplementedError()

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[LEDType] = None,
            rel_range_override: Optional[int] = None,
            cropping_box: Optional[EvoCroppingBox] = None,
            algorithm_override: Optional[FocusAlgorithmType] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        """
        Initialises software focus routine and validates parameter and limits. May be
        overwritten by child classes.

        Parameters
        ----------
        cfg_focus               : ConfigFocus object. If None, uses self.cfg_focus provided in constructor.
        focus_channel_override  : Overrides cfg_focus.focus_channel.
        rel_range_override      : Overrides cfg_focus.rel_range.
        cropping_box            : Apply focus to cropped image. If None, uses full image.
        algorithm_override      : Overrides cfg_focus.algorithm.
        user_input_override     : Overrides cfg_focus.user_input.
        countdown_override      : Overrides cfg_focus.countdown (applicable only if user_input_override is True).

        Returns
        -------

        Initialises
        -----------
        self.focus_curr_pos: Dict[str, float]
        self.focus_Z_coords: range of Z coordinates (int)
        self.focus_stack: 3D np.ndarray of images with dimensions X, Y, len(focus_Z_coords)
        self.focus_scores: 1D np.ndarray of focus scores
        self.focus_prev_image: 2D np.ndarray of image before starting focus
        self.focus_cropping_box: EvoCroppingBox
        """
        raise NotImplementedError()

    def software_focus_is_initialised(self) -> bool:
        return self._focus_is_initialised

    def software_focus_step(self, ipos: int) -> bool:
        """
        Moves the stage to self.focus_Z_coords[ipos], takes an image stored in self.focus_stack[:, :, ipos], computes the
        focus score, and stores the score in self.focus_scores[ipos].

        Parameters
        ----------
        ipos: int       Z position ID.

        Returns
        -------
        success: bool   True if successful. Otherwise, check return value of self.get_focus_status().

        Assigns
        -------
        self.focus_stack[:, :, ipos]: np.array  Raw image
        self.focus_scores[ipos]: float          Focus score

        """
        raise NotImplementedError()

    def set_exposure(self, exposure_time: Union[int, None] = None):
        if exposure_time is None:
            exposure_time = self.cfg.focus.exposure_time
        self._set_exposure(exposure_time=exposure_time)
        self._curr_exposure = exposure_time

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        raise NotImplementedError()

    def set_led(self, i_chan: LEDType, brightness: float = 29, block: bool = False, duration: float | None = None):
        raise NotImplementedError()

    def calibrate_magnet(self):
        raise NotImplementedError()

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Any], use_autofocus: bool) -> bool:
        raise NotImplementedError()

    def zero_coordinates(self):
        raise NotImplementedError()

    def _initialise(self) -> bool:
        raise NotImplementedError()

    def _move_stage_to_pos(
            self,
            i_pos: int,
            block: bool = True,
    ) -> bool:
        raise NotImplementedError()

    def get_filter_wheel(self) -> FilterWheelType:
        """
        Get current filter wheel set. Variable only set after set_filter_wheel was called. Assumes FILTER otherwise.

        Returns
        -------
        filter_wheel : FilterWheelType
            Last recorded FilterWheelType.
        """
        return self._current_filter_type

    def set_filter_wheel(self, filter_type: FilterWheelType):
        """
        Abstract method for setting filter wheel. Implemented by _set_filter_wheel.

        Parameters
        ----------
        filter_type : FilterWheelType
            Filter to set in camera.

        Returns
        -------

        """
        self._set_filter_wheel(filter_type=filter_type)
        self._current_filter_type = filter_type

    def _set_filter_wheel(self, filter_type: FilterWheelType):
        raise NotImplementedError()

    def _take_frame(
            self,
            i_chan: Optional[LEDType] = None,
            brightness: float = 29,
            block: bool = False,
            reset_led: bool = True,
            disable_led: bool = False,
    ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
        raise NotImplementedError()

    def autofocus_is_initialised(self) -> bool:
        return self._crisp_initialised


class TestCamera(AbstractCamera):
    """
    A class to mock the acquisition of frames.
    """
    def __init__(
            self,
            cfg_camera: ConfigCamera,
            filenames: List[Union[str, Path]],
            pos_to_filename: Optional[Union[Dict[int, int], None]] = None,
            cropping_indices: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ):
        super().__init__(cfg_camera=cfg_camera)
        if len(np.unique(filenames)) != len(filenames):
            raise ConfigError("TestCamera.__init__: must provide list with unique filenames.",
                              ErrorCode.ERROR_TEST_CAMERA_CONFIG)
        self.filenames: List[Union[str, Path]] = filenames
        "List of filenames for mock images."
        self.indices: Iterator[int] = itertools.cycle(range(len(filenames)))
        "Cyclic indices."
        self._cfg_focus: ConfigFocus = self.cfg.focus.copy()
        "Settings for CRISP autofocus. Required for GUI interaction."
        self.pos_to_filename: Union[Dict[int, int], None] = pos_to_filename
        "Optional dictionary mapping from unique position numbers (0,1,2,...) to filename."
        if self.pos_to_filename is not None:
            self.set_pos_id_to_coordinate(
                pos_id_to_coordinate={i: {'X': 0, 'Y': 0, 'Z': 0} for i in pos_to_filename.keys()},
                use_autofocus=True,
            )
        self._led_channel_keys: Dict[LEDType, Union[str, None]] = {
            LEDType.LED_405_NM: "X",
            LEDType.LED_450_NM: "Y",
            LEDType.LED_505_NM: "Z",
            LEDType.LED_538_NM: "F",
            LEDType.NO_LED: None,
        }
        "LED keys i_chan=0,...,3 for communication with Tiger."
        self._crop_inds: Optional[Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]] = \
            cropping_indices if cropping_indices else None
        "Optional cropping indices applied to all images. If provided, must be of the form ((xmin, xmax), (ymin, ymax))"
        self._current_led_channel: LEDType = LEDType.NO_LED
        self._next_filename_index: int = next(self.indices)
        self._current_pos: Coordinate = Coordinate(0, 0, 0)
        self.focus_curr_pos: Dict[str, float] = {}

        try:
            pvc.init_pvcam()
            self.cam = next(Camera.detect_camera())
            self.cam.open()
            self.cam.exp_mode = "Internal Trigger"
            self._pvc_is_alive = True
            logger.info(f"_initialise: pvcam initialised.")
        except Exception as e:
            self._pvc_is_alive = False
            logger.warning(f"EvoCamerav2._initialise: Error connecting to pvcam: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_PVC_NOT_ALIVE)
            )

        self.autofocus_lock()

    def increment_filename_index(self):
        self._next_filename_index = next(self.indices)

    def _move_stage_to_pos(
            self,
            i_pos: int,
            block: bool = True,
    ) -> bool:
        logger.info("TestCamera._move_stage_to_pos: moving to pos={} (block={})".format(i_pos, block))
        if self.pos_to_filename is not None:
            if i_pos not in self.pos_to_filename:
                raise EvoMachineError(f"TestCamera._move_stage_to_pos: i_pos={i_pos} not in pos_to_filename.",
                                      ErrorCode.ERROR_TEST_CAMERA_CONFIG)
            self._next_filename_index = self.pos_to_filename[i_pos]
        else:
            self.increment_filename_index()
        return True

    def _initialise(self) -> bool:
        logger.info("TestCamera._initialise: initialising TestCamera.")
        return True

    def _set_filter_wheel(self, filter_type: FilterWheelType):
        logger.info(f"TestCamera._set_filter_wheel={self._current_filter_type}.")
        return

    # def _take_frame(
    #         self,
    #         i_chan: Optional[LEDType] = None,
    #         brightness: float = 29,
    #         block: bool = False,
    #         reset_led: bool = True,
    #         disable_led: bool = False,
    # ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
    #     random_matrix = np.random.randint(0, 2 ** 6, size=self.cfg.image.shape, dtype=np.uint16)
    #     image = skimage.io.imread(self.filenames[self._next_filename_index]) + random_matrix
    #     if self._crop_inds:
    #         return image[self._crop_inds[0][0]:self._crop_inds[0][1], self._crop_inds[1][0]:self._crop_inds[1][1]]
    #     else:
    #         return image

    def _take_frame(
            self,
            i_chan: Optional[LEDType] = None,
            brightness: float = 29,
            block: bool = False,
            reset_led: bool = True,
            disable_led: bool = False,
    ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
        if not self._pvc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        curr_channel = self.current_channel
        if i_chan is not None:
            self._last_frame_channel = i_chan
            self.set_led(i_chan=i_chan, brightness=brightness, block=block)
        try:
            # self.mmc.snap_image()  # noqa
            pixels = self.cam.get_frame(timeout_ms=1000)
        except Exception as e:
            logger.warning(f"EvoCamera._take_frame: Received exception:\n{e}\nHave you disabled MM live mode?")
            return None
        # tagged_image = self.mmc.get_tagged_image()  # noqa
        # pixels = np.reshape(
        #     tagged_image.pix,
        #     newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        # )
        if i_chan is not None and reset_led and (not disable_led):
            self.set_led(i_chan=curr_channel, block=False)
        if disable_led:
            self.disable_led()
        return pixels

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        if self._pvc_is_alive:
            self.cam.exp_time = exposure_time
        else:
            logger.warning("EvoCamera._set_exposure: cannot set exposure as pvc is not alive.")

    def _set_imaging_mode(self, imaging_mode: str = "Dynamic Range"):
        available_modes = ["Sensitivity", "Speed", "Dynamic Range", "Sub-Electron"]
        if imaging_mode not in available_modes:
            msg = f"EvoCamera._set_imaging_mode: {imaging_mode} not in {available_modes}."
            logger.warning(msg)
            return
        if self._pvc_is_alive:
            self.cam.readout_port = available_modes.index(imaging_mode)
        else:
            logger.warning("EvoCamera._set_imaging_mode: cannot set mode as PVC is not alive.")






    def _finalise(self):
        logger.info("TestCamera.finalise: finalising TestCamera.")
        self.cam.close()
        pvc.uninit_pvcam()
        return

    def get_filename(
            self,
            i_pos: int | None = None,
            i_channel: LEDType | None = None,
            suffix: str | None = None,
            filter_wheel: FilterWheelType | None = None,
    ) -> str:
        if i_pos is not None:
            pos = self._pos_id_to_coordinate[i_pos].to_dict()
        else:
            pos = self._current_pos.to_dict()
        if i_channel is None:
            i_channel = self._current_led_channel
        if filter_wheel is None:
            i_channel = self._current_filter_type
        return "{}_P{}_X{}_Y{}_Z{}_F{}_{}{}.tiff".format(
            LEDType.get_name(value_to_find=i_channel.value).replace("_", ""),
            i_pos if i_pos is not None else "",
            pos['X'],
            pos['Y'],
            pos['Z'] if 'Z' in pos else "auto",
            filter_wheel.value if filter_wheel is not None else "None",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
            suffix if suffix is not None else "",
        )

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        return {key: val for key, val in self._current_pos.to_dict().items() if key in [tmp.upper() for tmp in axes]}

    def get_led_channels(self) -> List[LEDType]:
        return self.cfg.leds

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        return Coordinate(-1e7, -1e7, -1e7), Coordinate(1e7, 1e7, 1e7)

    def halt_stage(self):
        logger.info("TestCamera.halt_stage.")
        return

    def coordinate_is_out_of_bounds(self, coordinate: Dict[str, float]) -> bool:
        return False

    def disable_led(self):
        self._current_led_channel = LEDType.NO_LED
        logger.info("TestCamera.disable_led.")
        return

    def disable_live_mode(self):
        logger.info("TestCamera.disable_live_mode.")
        return

    def autofocus_initialise(
            self,
            this_cfg_crisp: Optional[ConfigCRISP] = None,
            user_input: Optional[bool] = True,
    ) -> bool:
        logger.info("TestCamera.autofocus_initialise.")
        self._autofocus_is_locked = False
        self._crisp_initialised = True
        time.sleep(1)
        return True

    def autofocus_disable(self):
        logger.info("TestCamera.autofocus_disable.")
        self._autofocus_is_locked = False

    def autofocus_get_status(self) -> AutoFocusStatusType:
        return AutoFocusStatusType.IN_FOCUS if not self._autofocus_is_locked else AutoFocusStatusType.READY

    def autofocus_is_locked(self):
        return self._autofocus_is_locked

    def _autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg.autofocus
        logger.info(f"TestCamera.autofocus_configure with cfg={cfg_crisp} (this_cfg_crisp={this_cfg_crisp}).")
        return True

    def _autofocus_lock(self):
        logger.info("TestCamera.autofocus_lock.")
        self._autofocus_is_locked = True

    def autofocus_unlock(self):
        logger.info("TestCamera.autofocus_unlock.")
        self._autofocus_is_locked = False

    def software_focus_finalise(self, move_on_any_focus_status: bool = True, debug_save: bool = True):
        logger.info("TestCamera.software_focus_finalise.")
        self._focus_status = FocusStatusType.IN_FOCUS
        return

    def software_focus_is_valid_range(self) -> bool:
        if self.focus_Z_coords is None:
            return False
        else:
            for z_coord in [self.focus_Z_coords[0], self.focus_Z_coords[-1]]:
                is_out_of_bounds = self.coordinate_is_out_of_bounds({'Z': z_coord})
                if is_out_of_bounds:
                    logger.warning(
                        f"EvoCamera.software_focus_check_range: Coordinates are out of bounds. "
                        f"Received min, max = {[self.focus_Z_coords[0], self.focus_Z_coords[-1]]}. "
                        f"Stage limits = {self.get_stage_limits()}). Reset stage limits."
                    )
                    return False
        return True

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[LEDType] = None,
            rel_range_override: Optional[int] = None,
            cropping_box: Optional[EvoCroppingBox] = None,
            algorithm_override: Optional[FocusAlgorithmType] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        args_str = f"cfg_focus={cfg_focus}\nfocus_channel_override={focus_channel_override}, " \
                   f"rel_range_override={rel_range_override}, cropping_box={cropping_box}, " \
                   f"algorithm_override={algorithm_override}, user_input_override={user_input_override}, " \
                   f"countdown_override={countdown_override}."
        logger.info(f"TestCamera.software_focus_initialise with args={args_str}.")
        # Assign optional arguments
        if cropping_box is None:
            self.focus_cropping_box = EvoCroppingBox.full(np.zeros(self.cfg.image.shape))
        else:
            self.focus_cropping_box = cropping_box
        self._cfg_focus = cfg_focus.copy() if cfg_focus else self.cfg.focus.copy()
        if focus_channel_override is not None:
            self._cfg_focus.focus_channel = focus_channel_override
        if algorithm_override is not None:
            self._cfg_focus.algorithm = algorithm_override
        if rel_range_override is not None:
            self._cfg_focus.rel_range = rel_range_override
        # Initialise Z stack
        self.focus_curr_pos = self.get_coordinates(axes=["X", "Y", "Z"])
        self.focus_Z_coords = range(
            int(self.focus_curr_pos['Z']-self._cfg_focus.rel_range),
            int(self.focus_curr_pos['Z']+self._cfg_focus.rel_range),
            self._cfg_focus.step_size,
        )
        if not self.software_focus_is_valid_range():
            self._focus_is_initialised = False
            return
        self.focus_scores = np.zeros(len(self.focus_Z_coords))
        self.focus_stack = np.zeros(shape=(*self.cfg.image.shape, len(self.focus_Z_coords)), dtype=np.float64)
        self.disable_led()
        self.focus_prev_image = self.get_frame(i_chan=self._cfg_focus.focus_channel).astype(np.float64)
        self._focus_is_initialised = True

    def software_focus_step(self, ipos: int) -> bool:
        logger.info(f"TestCamera.software_focus_step at pos_z={ipos}.")
        z_coord = self.focus_Z_coords[ipos]
        self.move_to(coordinate={'Z': z_coord}, block=True)
        image_raw = self.display_save_frame(
            i_chan=None,  # self._cfg_focus.focus_channel,
            path_to_save=False,
            filename=None,
            display_frame=False,
        )
        if image_raw is None:
            logger.warning("EvoCamera.software_focus: self._take_frame returned None. Aborting...")
            return False
        else:
            self.focus_stack[:, :, ipos] = image_raw
        self.focus_scores[ipos] = get_focus_score(
            img=self.focus_cropping_box.crop(self.focus_stack[:, :, ipos]),
            algorithm=self._cfg_focus.algorithm,
        )
        return True

    def move_home(self, block: Optional[bool] = False):
        self._current_pos = Coordinate(0, 0, 0)
        logger.info("TestCamera.move_home.")
        self.increment_filename_index()

    def move_fov_up(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='Y', sign=-1, multiplier=multiplier, block=block)
        logger.info("TestCamera.move_fov_up.")
        self.increment_filename_index()

    def move_fov_down(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='Y', sign=+1, multiplier=multiplier, block=block)
        logger.info("TestCamera.move_fov_down.")
        self.increment_filename_index()

    def move_fov_left(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='X', sign=-1, multiplier=multiplier, block=block)
        logger.info("TestCamera.move_fov_left.")
        self.increment_filename_index()

    def move_fov_right(self, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        self._move_fov(x_or_y='X', sign=+1, multiplier=multiplier, block=block)
        logger.info("TestCamera.move_fov_right.")
        self.increment_filename_index()

    def _move_fov(self, x_or_y: str, sign: int, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        pos = self._current_pos.to_dict()
        if x_or_y.upper() not in pos:
            raise TigerError(f"EvoCamera._move_fov: queried coordinate {x_or_y.upper()} not in response: {pos}.",
                             ErrorCode.ERROR_TIGER_NO_DATA)
        pos[x_or_y.upper()] += int(sign * self.cfg.fov_size * 10 * multiplier)
        self._current_pos = Coordinate.from_dict(pos)

    def move_to(self, coordinate: Union[Dict[str, int], Coordinate], block: Optional[bool] = False):
        if isinstance(coordinate, dict):
            if not all([key in coordinate for key in ['X', 'Y', 'Z']]):
                for key in ['X', 'Y', 'Z']:
                    if key not in coordinate:
                        coordinate[key] = self._current_pos.to_dict()[key]
        else:
            if coordinate.x is None:
                coordinate.x = self._current_pos.x
            if coordinate.y is None:
                coordinate.y = self._current_pos.y
            if coordinate.z is None:
                coordinate.z = self._current_pos.z
        self._current_pos = Coordinate.from_dict(coordinate) if isinstance(coordinate, dict) else coordinate
        logger.info(f"TestCamera.move_to={str(coordinate)}, block={block}.")
        self.increment_filename_index()

    def get_delta_fov(self):
        return self.cfg.fov_size * 10

    def keyboard_control(self):
        return

    # def _set_exposure(self, exposure_time: Union[int, None] = None):
    #     logger.info(f"TestCamera._set_exposure={exposure_time}.")
    #     return

    def set_led(self, i_chan: LEDType, brightness: float = 29, block: bool = False, duration: float | None = None):
        self._current_led_channel = i_chan
        logger.info(f"TestCamera.set_led={i_chan}, brightness={brightness}, block={block}.")
        return

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Any], use_autofocus: bool) -> bool:
        for i_pos, coord in pos_id_to_coordinate.items():
            if (not use_autofocus) and (not coord.has_z()):
                logger.warning(f"TestCamera.set_pos_id_to_coordinate: Position {i_pos} is missing Z "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
            if self.coordinate_is_out_of_bounds(coord):
                logger.warning(f"TestCamera.set_pos_id_to_coordinate: Position {i_pos} is out of bounds. "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
        self._pos_id_to_coordinate = {key: val for key, val in pos_id_to_coordinate.items()}
        logger.info(f"TestCamera.set_pos_id_to_coordinate={pos_id_to_coordinate}.")
        return True

    def zero_coordinates(self):
        self._current_pos = Coordinate(0, 0, 0)
        logger.info("TestCamera.zero_coordinates.")
        return


class EvoCamera(AbstractCamera):
    """
    EvoMachine acquisition class.

    Orientation is "Left on camera == left on stage":

    -> Camera view:
        __________________________________
        | 111111211111111111111111111111 |
        |       2                        |
        |       2                        |
        |       2                        |
        |       2                        |
        |_______2________________________|

    -> Stage view (incubator door at bottom):

        _____________BACK_________________
        | 111111211111111111111111111111 |
        |       2                        |
        |       2                        |
        |       2                        |
        |       2                        |
        |_______2____DOOR________________|

    """
    def __init__(
            self,
            cfg_camera: ConfigCamera,
            tiger_port: str = "/dev/ttyUSB0",
    ):
        super().__init__(cfg_camera=cfg_camera)

        self._tiger_port = tiger_port
        "Tiger port for serial communication."
        self.tiger: Optional[asitiger.tigercontroller.TigerController, asitiger.tigerthread.TigerThread] = None
        "Object for serial communication with ASI tiger."
        self._tiger_is_alive: bool = False
        "Flag set in _initialise."
        self._is_multi_threaded: bool = False
        "If true, will use Threading wrappers for objects like self.tiger."
        self.current_channel: LEDType = LEDType.NO_LED
        "Current LED channel set."
        self._last_frame_channel: LEDType = LEDType.NO_LED
        "Channel used to take last frame."
        self._led_channel_keys: Dict[LEDType, Union[str, None]] = {
            LEDType.LED_405_NM: "X",
            LEDType.LED_450_NM: "Y",
            LEDType.LED_505_NM: "Z",
            LEDType.LED_538_NM: "F",
            LEDType.NO_LED: None,
        }
        "LED keys i_chan=0,...,3 for communication with Tiger."
        self._current_led_brightness: int = 0
        "Last set LED brightness"
        self.card_address_led: int = 7
        "LED card address on ASI tiger."
        self.card_address_fw: int = 8
        "Filter wheel card address on ASI tiger."
        self.filter_wheel_settings: Dict[FilterWheelType, int] = {
            # FilterWheelType.FILTER: 1, FilterWheelType.BLOCKING: 0, FilterWheelType.NO_FILTER: 2
            FilterWheelType.FILTER: 0,
            FilterWheelType.FILTER_465nm: 1,
            FilterWheelType.FILTER_527nm: 2,
            FilterWheelType.FILTER_592nm: 3,
            FilterWheelType.NO_FILTER: 4,
            FilterWheelType.BLOCKING: 5,
        }
        "Available filter wheels."
        self.card_address_crisp: int = 2
        "CRISP card address on ASI tiger."
        self._cfg_focus: ConfigFocus = self.cfg.focus.copy()
        "Internal focus configuration used for overrides."

        self._pos_id_to_coordinate: Dict[int, Coordinate] = {}
        "Dictionary position ID -> Coordinate. Initialise through set_pos_id_to_coordinate."

        self.mmc: Union[Core, None] = None
        "Micromanager Core object for taking images."
        self.studio: Union[Studio, None] = None
        "Micromanager Studio object for additional functions."
        self._mmc_is_alive: bool = False
        "Flag set in _initialise."

        # self.initialise()  # Must be called before using EvoCamera

    def _initialise(self) -> bool:
        """ Initialises EvoCamera objects with peripherals. Tests connections and sets is_alive flags. """
        # Tiger box communication
        try:
            if self._is_multi_threaded:
                self.tiger: asitiger.tigerthread.TigerThread = \
                    asitiger.tigerthread.TigerThread.from_serial_port(port=self._tiger_port)
            else:
                self.tiger: asitiger.tigercontroller.TigerController = \
                    asitiger.tigercontroller.TigerController.from_serial_port(port=self._tiger_port)
            logger.info(f"_initialise: tiger initialised on {self._tiger_port}.")
        except Exception as e:
            self._tiger_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to Tiger on port {self._tiger_port}: {e}.")
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
            logger.info(f"_initialise: MMC initialised.")
        except Exception as e:
            self._mmc_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to MMC: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_MMC_NOT_ALIVE)
            )
        self.disable_led()
        self.set_exposure()
        self._set_imaging_mode()

        return self._mmc_is_alive and self._tiger_is_alive

    def _get_tiger_is_alive(self) -> bool:
        if not self.tiger:
            return False
        try:
            _ = self.tiger.status()
            return True
        except ValueError:
            return False

    def _set_filter_wheel(self, filter_type: FilterWheelType):
        if filter_type not in self.filter_wheel_settings.keys():
            logger.error(msg=f"EvoCamera._set_filter_wheel: filter_type={filter_type} not in wheels={self.filter_wheel_settings}.")
            return
        if self._tiger_is_alive:
            self.tiger.filter_wheel(position=self.filter_wheel_settings[filter_type], card_address=self.card_address_fw)
        else:
            logger.error(msg=f"EvoCamera._set_filter_wheel: Tiger is not alive.")

    def disable_led(self):
        self.set_led(i_chan=LEDType.NO_LED)

    def disable_live_mode(self):
        self.studio.live().set_live_mode(False)  # noqa

    def _move_fov(self, x_or_y: str, sign: int, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        pos = self.tiger.where([x_or_y.upper()])
        if x_or_y.upper() not in pos:
            raise TigerError(f"EvoCamera._move_fov: queried coordinate {x_or_y.upper()} not in response: {pos}.",
                             ErrorCode.ERROR_TIGER_NO_DATA)
        pos[x_or_y.upper()] += int(sign * self.cfg.fov_size * 10 * multiplier)
        self.move_to(coordinate=pos, block=block)

    def _move_stage_to_pos(
            self,
            i_pos: int,
            block: bool = True,
    ) -> bool:
        return self._move_stage_to_coord(self._pos_id_to_coordinate[i_pos].to_dict(), block=block)

    def _move_stage_to_coord(
            self,
            coordinates: Dict[str, int],
            block: bool = True,
    ) -> bool:
        answer = None
        if self._tiger_is_alive:
            answer = self.tiger.move(coordinates=coordinates)
            if block:
                self.tiger.wait_until_idle(card_address_crisp=self.card_address_crisp)
        else:
            logger.error(msg=f"EvoCamera._move_stage_to_coord: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")
        return True if isinstance(answer, str) else False

    def _take_frame(
            self,
            i_chan: Optional[LEDType] = None,
            brightness: float = 29,
            block: bool = False,
            reset_led: bool = True,
            disable_led: bool = False,
    ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
        if not self._mmc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        curr_channel = self.current_channel
        if i_chan is not None:
            self._last_frame_channel = i_chan
            self.set_led(i_chan=i_chan, brightness=brightness, block=block)
        try:
            self.mmc.snap_image()  # noqa
        except Exception as e:
            logger.warning(f"EvoCamera._take_frame: Received exception:\n{e}\nHave you disabled MM live mode?")
            return None
        tagged_image = self.mmc.get_tagged_image()  # noqa
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        if i_chan is not None and reset_led and (not disable_led):
            self.set_led(i_chan=curr_channel, block=False)
        if disable_led:
            self.disable_led()
        return pixels

    def coordinate_is_out_of_bounds(self, coordinate: Union[Dict[str, float], Coordinate]) -> bool:
        return self.tiger.coordinate_is_out_of_bounds(
            coordinate.to_dict() if isinstance(coordinate, Coordinate) else coordinate
        )

    def autofocus_initialise(
            self,
            this_cfg_crisp: Optional[ConfigCRISP] = None,
            user_input: Optional[bool] = True,
    ) -> bool:
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_initialise: Device not alive.")
            return False

        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg.autofocus
        ask_user = cfg_crisp.user_input and user_input

        if ask_user:
            user_input_str = input("Starting CRISP autofocus. Do you want to proceed? (yes/no): ")
            if user_input_str.lower() == "yes":
                logger.info("CRISP: Proceeding with configuring and setting up CRISP autofocus.")
            else:
                logger.info("CRISP: Aborting CRISP configuration.")
                return False
        is_configured = self.autofocus_configure(this_cfg_crisp=cfg_crisp)
        if not is_configured:
            return False

        is_success = True
        logger.info("CRISP: Setting IDLE status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.IDLE)
        logger.info("CRISP: Resetting offsets.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.SET_OFFSET)
        time.sleep(cfg_crisp.pause_short)
        logger.info("CRISP: Setting LOG_CAL status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.LOG_CAL)
        time.sleep(cfg_crisp.pause_long)
        val = self.tiger.crisp_get_snr(card_address=self.card_address_crisp)
        if val < cfg_crisp.min_snr:
            is_success = False
            logger.warning(f"EvoCamera.autofocus: Low SNR = {val:.2d}. Increase CRISP LED intensity and repeat.")
        logger.info("CRISP: Setting DITHER status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.DITHER)
        time.sleep(cfg_crisp.pause_long)
        val = self.tiger.crisp_get_err(card_address=self.card_address_crisp)
        if np.abs(val) < cfg_crisp.min_error:
            is_success = False
            logger.warning(f"EvoCamera.autofocus: Low error = {val}. Check ASI guide.")
        logger.info("CRISP: Setting SET_GAIN status.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.SET_GAIN)
        time.sleep(cfg_crisp.pause_short)
        logger.info("CRISP: Setting UNLOCK status.")
        self.autofocus_unlock()
        time.sleep(cfg_crisp.pause_short)

        do_lock = False
        if ask_user:
            user_input_str = input("Do you want to lock CRISP autofocus? (yes/no): ")
            do_lock = True if user_input_str.lower() == "yes" else False
        if do_lock:
            logger.info("CRISP: Setting LOCK status.")
            self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.LOCK)
        time.sleep(cfg_crisp.pause_short)
        curr_state = self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None)
        logger.info(f"CRISP: Finalising autofocus. Current state is {curr_state}.")
        if not is_success:
            logger.warning(f"autofocus_initialise: initialisation was not successful.")
        return is_success

    def autofocus_disable(self):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_disable: Device not alive. Trying to disable anyway.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.IDLE)

    def autofocus_get_status(self) -> AutoFocusStatusType:
        retval: str = self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None)
        return AutoFocusStatusType.from_flag(status_flag=retval)

    def autofocus_is_locked(self) -> bool | None:
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_is_locked: Device not alive.")
            return None
        retval: str = self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None)
        logger.info(f"autofocus_is_locked: crisp_get_set_state returned {retval} ({CRISPStatus.from_flag(retval)}).")
        return retval in CRISPStatus.get_locked_state_flags()

    def _autofocus_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.autofocus_configure: Device not alive.")
            return False
        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg.autofocus
        try:
            logger.info(f"CRISP: Configuring CRISP with following parameters:\n{cfg_crisp}")
        except ConfigError as e:
            logger.error(f"CRISP: Bad configuration:\n{e}\nCannot use CRISP.")
            return False
        self.autofocus_unlock()
        time.sleep(cfg_crisp.pause_short)
        # self.tiger.crisp_reset_offset(card_address=self.card_address_crisp)
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
            led_intensity=self.tiger.crisp_get_set_led_intensity(card_address=self.card_address_crisp, value=None),
            loop_gain=self.tiger.crisp_get_set_loop_gain(card_address=self.card_address_crisp, value=None),
            averaging=self.tiger.crisp_get_set_num_avg(card_address=self.card_address_crisp, value=None),
            update_rate=self.tiger.crisp_get_set_update_rate(card_address=self.card_address_crisp, value=None),
            lock_range=self.tiger.crisp_get_set_lock_range(card_address=self.card_address_crisp, value=None),
            objective_na=self.tiger.crisp_get_set_objective_na(card_address=self.card_address_crisp, value=None),
            min_snr=cfg_crisp.min_snr,
            min_error=cfg_crisp.min_error,
        )
        self._crisp_initialised = True
        logger.info(f"CRISP: Parameters set to:\n{new_cfg}")
        return True

    def _autofocus_lock(self):
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.LOCK)

    def autofocus_unlock(self):
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.UNLOCK)

    def _finalise(self):
        logger.info(f"EvoCamera.finalise: Finalising EvoCamera.")

        # Note: apparently this is all that is stopping the GUI from shutting completely,
        #       so if the Tiger is not turned on, then the GUI quits faster and never shuts
        #       down the syncboard.
        if self._is_multi_threaded:
            self.tiger.stop()
            self.tiger.join()

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        """ Returns current coordinates of the stage. """
        return self.tiger.where(axes)

    def get_delta_fov(self):
        return self.cfg.fov_size * 10

    def get_filename(
            self,
            i_pos: int | None = None,
            i_channel: LEDType | None = None,
            suffix: str | None = None,
            filter_wheel: FilterWheelType | None = None,
    ) -> str:
        if i_pos is not None:
            if i_pos in self._pos_id_to_coordinate.keys():
                pos = self._pos_id_to_coordinate[i_pos].to_dict()
            else:
                msg = f"EvoCamera.get_filename: {i_pos} not in {self._pos_id_to_coordinate}. Using current coords."
                logger.warning(msg)
                pos = self.tiger.where(['X', 'Y', 'Z'])
        else:
            pos = self.tiger.where(['X', 'Y', 'Z'])
        if i_channel is None:
            i_channel = self._last_frame_channel
        if filter_wheel is None:
            filter_wheel = self._current_filter_type
        return "{}_P{}_X{}_Y{}_Z{}_F{}_{}{}.tiff".format(
            LEDType.get_name(value_to_find=i_channel.value).replace("_", ""),
            i_pos if i_pos is not None else "",
            np.round(pos['X']),
            np.round(pos['Y']),
            np.round(pos['Z']) if 'Z' in pos else "auto",
            filter_wheel.value,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
            suffix if suffix is not None else ""
        )

    def get_led_channels(self) -> List[LEDType]:
        return list(self._led_channel_keys.keys())

    def get_stage_limits(self) -> Tuple[Coordinate, Coordinate]:
        lim = self.tiger.get_stage_limits()
        return Coordinate(lim['X'][0], lim['Y'][0], lim['Z'][0]), Coordinate(lim['X'][1], lim['Y'][1], lim['Z'][1])

    def halt_stage(self):
        self.tiger.halt()

    def keyboard_control(self):
        return

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
            self.mmc.set_exposure(exposure_time)  # noqa
        else:
            logger.warning("EvoCamera._set_exposure: cannot set exposure as MMC is not alive.")

    def _set_imaging_mode(self, imaging_mode: str = "Dynamic Range"):
        available_modes = ["Dynamic Range", "Sensitivity", "Speed"]
        if imaging_mode not in available_modes:
            msg = f"EvoCamera._set_imaging_mode: {imaging_mode} not in {available_modes}."
            logger.warning(msg)
            return
        if self._mmc_is_alive:
            self.mmc.set_property("Camera-1", "Port", imaging_mode) # noqa
        else:
            logger.warning("EvoCamera._set_imaging_mode: cannot set mode as MMC is not alive.")

    def set_pos_id_to_coordinate(self, pos_id_to_coordinate: Dict[int, Coordinate], use_autofocus: bool) -> bool:
        for i_pos, coord in pos_id_to_coordinate.items():
            if (not use_autofocus) and (not coord.has_z()):
                logger.warning(f"EvoCamera.set_pos_id_to_coordinate: Position {i_pos} is missing Z "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
            if use_autofocus and coord.has_z():
                logger.warning(f"EvoCamera.set_pos_id_to_coordinate: Position {i_pos} has Z "
                               f"coordinate ({coord}) even though autofocus enabled.")
                return False
            if self.coordinate_is_out_of_bounds(coord):
                logger.warning(f"EvoCamera.set_pos_id_to_coordinate: Position {i_pos} is out of bounds. "
                               f"coordinate ({coord}). Position list not initialised.")
                return False
        self._pos_id_to_coordinate = {key: val for key, val in pos_id_to_coordinate.items()}
        return True

    def set_led(self, i_chan: LEDType, brightness: float = 29, block: bool = False, duration: float | None = None):
        if i_chan not in self._led_channel_keys.keys():
            logger.error(msg=f"EvoCamera._set_channel: i_chan={i_chan} not in channels={self._led_channel_keys.keys()}.")
            return
        if self._tiger_is_alive:
            brightness = int(brightness)
            led_settings = {val: (brightness if ((key == i_chan) and (i_chan != LEDType.NO_LED)) else 0)
                            for key, val in self._led_channel_keys.items()}
            if (0 <= brightness <= 100) or (i_chan != LEDType.NO_LED):
                is_good_brightness_value = True
            else:
                is_good_brightness_value = False
            if is_good_brightness_value:
                self.tiger.led(led_brightnesses=led_settings, card_address=self.card_address_led)
                self.current_channel = i_chan
                self._current_led_brightness = 0 if i_chan == LEDType.NO_LED else brightness
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

    def software_focus_finalise(self, move_on_any_focus_status: bool = True, debug_save: bool = True):
        self._focus_is_initialised = False
        self._focus_curve_status = get_focus_curve_type(self.focus_scores)
        self._focus_status = FocusStatusType.IN_FOCUS if get_focus_score_is_good(self.focus_scores) \
            else FocusStatusType.BAD_FOCUS_CURVE

        if (not move_on_any_focus_status) and self._focus_status != FocusStatusType.IN_FOCUS:
            msg = f"EvoCamera.software_focus_finalise: received focus status {self.get_software_focus_status()} with " \
                  f"focus curve status {self.get_software_focus_curve_status()}. Returning without moving Z."
            logger.error(msg)
            return

        focus_best_coordinate = self.get_software_focus_z_coord()
        if focus_best_coordinate is None or self.coordinate_is_out_of_bounds({'Z': focus_best_coordinate}):
            msg = f"EvoCamera.software_focus_finalise: invalid z coordinate {focus_best_coordinate}."
            logger.error(msg)
            return

        logger.info(f"EvoCamera.software_focus: Finished scanning. Coordinate before focus="
                    f"{self.focus_curr_pos['Z'] / 10} μm,"
                    f"coordinate after focus={focus_best_coordinate / 10} μm. "
                    f"Focus status is {self.get_software_focus_status()} and curve status is "
                    f"{self.get_software_focus_curve_status()}. Finalising software_focus.")
        self._move_stage_to_coord({'Z': focus_best_coordinate})
        self.set_led(i_chan=self._focus_old_channel)

        if debug_save:
            filename = self.get_filename(
                i_pos=self.get_pos(), i_channel=self._cfg_focus.focus_channel, suffix="_fstack",
            ).replace(".tiff", ".npy")
            folder = Path("/mnt/nvme1/data/DebugData/")
            if folder.exists() and folder.is_dir():
                filename = str(folder) + filename
            logger.info(f"Saving focus stack under {filename}.")
            np.save(filename, self.focus_stack)

    def software_focus_initialise(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[LEDType] = None,
            rel_range_override: Optional[int] = None,
            cropping_box: Optional[EvoCroppingBox] = None,
            algorithm_override: Optional[FocusAlgorithmType] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ):
        """
        Initialises software focus routine and validates parameter and limits.

        """
        if not all([self._mmc_is_alive, self._tiger_is_alive]):
            logger.error(f"EvoCamera.software_focus: Device(s) not alive. "
                         f"Tiger={self._tiger_is_alive}, MMC={self._mmc_is_alive}.")
            self._focus_status = FocusStatusType.DEVICE_ERROR
            return
        # Assign optional arguments
        if cropping_box is None:
            self.focus_cropping_box = EvoCroppingBox.full(np.zeros(self.cfg.image.shape))
        else:
            self.focus_cropping_box = cropping_box
        self._cfg_focus = cfg_focus.copy() if cfg_focus else self.cfg.focus.copy()
        if focus_channel_override is not None:
            self._cfg_focus.focus_channel = focus_channel_override
        if algorithm_override is not None:
            self._cfg_focus.algorithm = algorithm_override
        if rel_range_override is not None:
            self._cfg_focus.rel_range = rel_range_override
        # Initialise Z stack
        self.focus_curr_pos = self.tiger.where()
        self.focus_Z_coords = range(
            int(self.focus_curr_pos['Z'])-self._cfg_focus.rel_range,
            int(self.focus_curr_pos['Z'])+self._cfg_focus.rel_range,
            self._cfg_focus.step_size,
        )
        if not self.software_focus_is_valid_range():
            self._focus_is_initialised = False
            self._focus_status = FocusStatusType.OUT_OF_RANGE
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
                logger.warning(f"ThreadSWFocus.run: Starting software autofocus configured as\n"
                               f"{self._cfg_focus.__str__()}\nThis will move the stage up and down in the range "
                               f"[{(self.focus_curr_pos['Z'] - self._cfg_focus.rel_range) / 10},"
                               f"{(self.focus_curr_pos['Z'] + self._cfg_focus.rel_range) / 10}] μm"
                               f" (current position = {self.focus_curr_pos['Z'] / 10} μm). ")
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
        self._focus_old_channel = self.current_channel
        self.set_exposure(exposure_time=int(self._cfg_focus.exposure_time))
        self.disable_live_mode()
        self.focus_scores = np.zeros(len(self.focus_Z_coords))
        self.focus_stack = np.zeros(shape=(*self.cfg.image.shape, len(self.focus_Z_coords)), dtype=np.float64)
        self.disable_led()
        self.focus_prev_image = self.display_save_frame(
            i_chan=self._cfg_focus.focus_channel,
            path_to_save=False,
            filename=None,
            display_frame=False,
        ).astype(np.float64)
        self._focus_is_initialised = True

    def software_focus_step(self, ipos: int) -> bool:
        z_coord = self.focus_Z_coords[ipos]
        self.move_to(coordinate={'Z': z_coord}, block=True)
        image_raw = self.display_save_frame(
            i_chan=None,    # self._cfg_focus.focus_channel
            path_to_save=False,
            filename=None,
            display_frame=False,
        )
        if image_raw is None:
            logger.warning("EvoCamera.software_focus: self._take_frame returned None. Aborting...")
            self._focus_status = FocusStatusType.NO_IMAGE
            return False
        else:
            self.focus_stack[:, :, ipos] = image_raw
        self.focus_scores[ipos] = get_focus_score(
            img=self.focus_cropping_box.crop(self.focus_stack[:, :, ipos]),
            algorithm=self._cfg_focus.algorithm,
        )
        return True

    def zero_coordinates(self):
        self.tiger.zero()


class EvoCamerav2(EvoCamera):
    """
    EvoMachine acquisition class. This class overrides some methods of EvoCamera as it uses the syncboard for LED
    control instead of the ASI Tiger Box.

    Orientation is "Left on camera == left on stage":

    -> Camera view:
        __________________________________
        | 111111211111111111111111111111 |
        |       2                        |
        |       2                        |
        |       2                        |
        |       2                        |
        |_______2________________________|

    -> Stage view (incubator door at bottom):

        _____________BACK_________________
        | 111111211111111111111111111111 |
        |       2                        |
        |       2                        |
        |       2                        |
        |       2                        |
        |_______2____DOOR________________|

    """
    def __init__(
            self,
            cfg_camera: ConfigCamera,
            tiger_port: str = "/dev/ttyUSB0",  # TODO move this to config
            syncboard_port: str = "/dev/syncboard",  # TODO move this to config
    ):
        super().__init__(cfg_camera=cfg_camera, tiger_port=tiger_port)

        self.syncboard: Optional[SyncBoardController] = None
        "Object to control sync board functions such as LEDs"
        self._syncboard_port: str = syncboard_port
        "Expected sync board serial port. If not found, the code will look for patterns like /dev/ttyACMX"
        self._syncboard_is_alive: bool = False
        "Set to true once successfully connected. Flag is not automatically updated on connection loss."
        # self._led_channel_keys: Dict[LEDType, Union[int, None]] = {
        #     LEDType.LED_385_NM: 7,
        #     LEDType.LED_450_NM: 1,
        #     LEDType.LED_515_NM: 2,
        #     LEDType.LED_565_NM: 3,
        #     LEDType.LED_645_NM: 4,
        #     LEDType.NO_LED: None,
        # }
        self._led_channel_keys: Dict[LEDType, LED_ID] = {
            LEDType.LED_385_NM: LED_ID.LED_385_NM,
            LEDType.LED_450_NM: LED_ID.LED_450_NM,
            LEDType.LED_515_NM: LED_ID.LED_515_NM,
            LEDType.LED_565_NM: LED_ID.LED_565_NM,
            LEDType.LED_645_NM: LED_ID.LED_645_NM,
            LEDType.NO_LED: LED_ID.NO_LED,
            LEDType.LED_OVERHEAD: -99,
        }
        "Map from LEDType to channel ID on sync board (hard-coded)."
        self.brightfield_psu: KWR103 | None = None
        "Serial object for brightfield control."

    def _initialise(self) -> bool:
        """ Initialises EvoCamera objects with peripherals. Tests connections and sets is_alive flags. """
        # Tiger box communication
        try:
            if self._is_multi_threaded:
                self.tiger: asitiger.tigerthread.TigerThread = \
                    asitiger.tigerthread.TigerThread.from_serial_port(port=self._tiger_port)
            else:
                self.tiger: asitiger.tigercontroller.TigerController = \
                    asitiger.tigercontroller.TigerController.from_serial_port(port=self._tiger_port)
            logger.info(f"_initialise: tiger initialised on {self._tiger_port}.")
        except Exception as e:
            self._tiger_is_alive = False
            logger.warning(f"EvoCamerav2._initialise: Error connecting to Tiger on port {self._tiger_port}: {e}.")
            self.error_container.add_error(
                new_error=TigerError(message=str(e), error_code=ErrorCode.ERROR_TIGER_SERIAL_CONNECTION)
            )
        if not self._get_tiger_is_alive():
            self._tiger_is_alive = False
            logger.warning("EvoCamerav2._initialise: Tiger is not alive.")
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
            logger.info(f"_initialise: MMC initialised.")
        except Exception as e:
            self._mmc_is_alive = False
            logger.warning(f"EvoCamerav2._initialise: Error connecting to MMC: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_MMC_NOT_ALIVE)
            )
        # SyncBoard communication
        try:
            self.syncboard: SyncBoardController = SyncBoardController.from_serial_port(port=self._syncboard_port)
            self.syncboard.initialise()
            if not self.syncboard.is_initialised():
                raise ConfigError("EvoCamerav2._initialise: Unable to initialise SyncBoard.",
                                  error_code=ErrorCode.ERROR_SYNC_BOARD)
            self._syncboard_is_alive = True
            logger.info(f"_initialise: syncboard initialised on {self._syncboard_port}.")
        except Exception as e:
            self._syncboard_is_alive = False
            logger.debug(f"EvoCamerav2._initialise: Error connecting to SyncBoard on port {self._syncboard_port}: {e}.")
            self.error_container.add_error(
                new_error=SyncBoardError(message=str(e), error_code=ErrorCode.ERROR_SYNC_BOARD)
            )
            # Retry on different ports
            possible_ports = list_serial_ports(starts_with="/dev/ttyACM")
            if not possible_ports:
                logger.warning(f"EvoCamerav2._initialise: Unable to initialise SyncBoard. No ports starting with "
                               f"/dev/ttyACM found. Please check the connection, or change the port in "
                               f"acquisition.EvoCamerav2.__init__ and restart.")
            elif len(possible_ports) > 1:
                logger.warning(f"EvoCamerav2._initialise: Found multiple ports matching the pattern /dev/ttyACMX: "
                               f"{possible_ports}. Please let me know which port I should connect to by specifying "
                               f"the correct one in acquisition.EvoCamerav2.__init__ and restart.")
            else:
                self._syncboard_port = possible_ports[0]
                logger.debug(f"EvoCamerav2._initialise: Re-trying on port {self._syncboard_port}.")
                try:
                    self.syncboard: SyncBoardController = SyncBoardController.from_serial_port(port=self._syncboard_port)
                    self.syncboard.initialise()
                    if not self.syncboard.is_initialised():
                        raise ConfigError("EvoCamerav2._initialise: Unable to initialise SyncBoard.",
                                          error_code=ErrorCode.ERROR_SYNC_BOARD)
                    self._syncboard_is_alive = True
                    logger.info(f"_initialise: syncboard initialised on {self._syncboard_port}.")
                    logger.info(f"EvoCamerav2._initialise: Connected to SyncBoard on port {self._syncboard_port}.")
                except Exception as e:
                    self._syncboard_is_alive = False
                    logger.warning(
                        f"EvoCamerav2._initialise: Error connecting to SyncBoard on port {self._syncboard_port}: {e}.")
                    self.error_container.add_error(
                        new_error=SyncBoardError(message=str(e), error_code=ErrorCode.ERROR_SYNC_BOARD)
                    )
        if not self._get_syncboard_is_alive():
            self._syncboard_is_alive = False
            logger.warning("EvoCamerav2._initialise: SyncBoard is not alive.")
            self.error_container.add_error(
                new_error=SyncBoardError(message="SyncBoard is not alive.", error_code=ErrorCode.ERROR_SYNC_BOARD)
            )
        else:
            self._syncboard_is_alive = True

        self.disable_led()
        self.set_exposure()
        self._set_imaging_mode()

        try:
            self.brightfield_psu = KWR103(get_psu_port())
            self.brightfield_psu.connect()
            self.brightfield_psu.set_output(False)
            self.brightfield_psu.set_current(0.1)
            self.brightfield_psu.set_voltage(8)
            logger.warning(f"EvoCamerav2._initialise: Connecting to PSU on {get_psu_port()}.")
        except SerialException:
            logger.warning("EvoCamerav2._initialise: Brightfield not connected.")
            self.brightfield_psu = None

        return self._mmc_is_alive and self._tiger_is_alive and \
            self._syncboard_is_alive  # and (self.brightfield_psu is not None) IDRIS

    def set_brightfield(self, brightness: float = 50):
        if brightness == 0:
            self.brightfield_psu.set_output(False)
        else:
            # map 0 -> 100 to 7V -> 9V
            self.brightfield_psu.set_voltage(min(9.0, 7 + 2 * brightness / 100))
            self.brightfield_psu.set_output(True)

    def calibrate_magnet(self):
        self.syncboard.calibrate_magnet()

    def calibrate_hall(self, hall_id: int):
        self.syncboard.calibrate_hall(hall_id)

    def set_led(self, i_chan: LEDType, brightness: float = 29, block: bool = False, duration: float | None = None):
        """

        Parameters
        ----------
        i_chan: LEDType
            LED used for projection.
        brightness: float
            If brightness > 29, a duration must be provided. If None, duration is set to 3 seconds.
        block: bool
            NOT IMPLEMENTED. Block until response from syncboard is received.
        duration: float | None
            In milliseconds. If provided, must be smaller than 1 hour. If none, brightness must be <= 29.

        Returns
        -------

        """
        if i_chan not in self._led_channel_keys.keys():
            logger.error(msg=f"EvoCamerav2.set_led: i_chan={i_chan} not in channels={self._led_channel_keys.keys()}.")
            return
        if i_chan == LEDType.LED_OVERHEAD:
            if self.brightfield_psu is None:
                msg = f"EvoCamerav2.set_led: Brightfield not connected. Cannot set {i_chan}."
                logger.error(msg)
                return
            self.set_brightfield(brightness=brightness)
            return
        if i_chan == LEDType.NO_LED and (self.brightfield_psu is not None):
            self.brightfield_psu.set_output(False)
        if self._syncboard_is_alive:
            if i_chan == LEDType.NO_LED:
                self.syncboard.disable_led()
                return
            if (0 < brightness <= 100) or (i_chan != LEDType.NO_LED):
                is_good_brightness_value = True
            else:
                is_good_brightness_value = False
            if is_good_brightness_value:
                if self.current_channel != LEDType.NO_LED:
                    self.syncboard.disable_led(led_id=self._led_channel_keys[self.current_channel])
                if brightness > 29 and duration is None:
                    duration = 120*1000
                self.syncboard.enable_led(
                    led_id=self._led_channel_keys[i_chan],
                    intensity=float(brightness) / 100.0,
                    duration=duration,
                )
                self.current_channel = i_chan
                self._current_led_brightness = 0 if i_chan == LEDType.NO_LED else brightness
            else:
                logger.error(msg=f"Cannot set brightness: {brightness} is out of range [0, 29]. LED not set.")
        else:
            logger.error(msg=f"EvoCamera._set_channel: SyncBoard is not alive.")

    def _finalise(self):
        logger.warning("Shutting down camera, ASI tiger, and sync board.")
        self.autofocus_unlock()
        if self.brightfield_psu is not None:
            self.brightfield_psu.set_output(False)
        if self.syncboard is not None:
            self.syncboard.finalise()
        else:
            logger.warning("Syncboard not detected, cannot disable")
        if self._is_multi_threaded:
            self.tiger.stop()
            self.tiger.join()

        self._tiger_is_alive = False
        self._syncboard_is_alive = False

    def _get_syncboard_is_alive(self):
        if not self.syncboard:
            return False
        return self._syncboard_is_alive

class EvoCamerav3(EvoCamerav2): # noqa

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialise(self) -> bool:
        """ Initialises EvoCamera objects with peripherals. Tests connections and sets is_alive flags. """
        # Tiger box communication
        try:
            if self._is_multi_threaded:
                self.tiger: asitiger.tigerthread.TigerThread = \
                    asitiger.tigerthread.TigerThread.from_serial_port(port=self._tiger_port)
            else:
                self.tiger: asitiger.tigercontroller.TigerController = \
                    asitiger.tigercontroller.TigerController.from_serial_port(port=self._tiger_port)
            logger.info(f"_initialise: tiger initialised on {self._tiger_port}.")
        except Exception as e:
            self._tiger_is_alive = False
            logger.warning(f"EvoCamerav2._initialise: Error connecting to Tiger on port {self._tiger_port}: {e}.")
            self.error_container.add_error(
                new_error=TigerError(message=str(e), error_code=ErrorCode.ERROR_TIGER_SERIAL_CONNECTION)
            )
        if not self._get_tiger_is_alive():
            self._tiger_is_alive = False
            logger.warning("EvoCamerav2._initialise: Tiger is not alive.")
            self.error_container.add_error(
                new_error=TigerError(message="Tiger is not alive.", error_code=ErrorCode.ERROR_TIGER_NOT_ALIVE)
            )
        else:
            self._tiger_is_alive = True

        # Camera communication
        try:
            pvc.init_pvcam()
            self.cam = next(Camera.detect_camera())
            self.cam.open()
            self.cam.exp_mode = "Internal Trigger"
            self._pvc_is_alive = True
            logger.info(f"_initialise: pvcam initialised.")
        except Exception as e:
            self._pvc_is_alive = False
            logger.warning(f"EvoCamerav2._initialise: Error connecting to pvcam: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_PVC_NOT_ALIVE)
            )

        # SyncBoard communication
        try:
            self.syncboard: SyncBoardController = SyncBoardController.from_serial_port(port=self._syncboard_port)
            self.syncboard.initialise()
            if not self.syncboard.is_initialised():
                raise ConfigError("EvoCamerav2._initialise: Unable to initialise SyncBoard.",
                                  error_code=ErrorCode.ERROR_SYNC_BOARD)
            self._syncboard_is_alive = True
            logger.info(f"_initialise: syncboard initialised on {self._syncboard_port}.")
        except Exception as e:
            self._syncboard_is_alive = False
            logger.debug(f"EvoCamerav2._initialise: Error connecting to SyncBoard on port {self._syncboard_port}: {e}.")
            self.error_container.add_error(
                new_error=SyncBoardError(message=str(e), error_code=ErrorCode.ERROR_SYNC_BOARD)
            )
            # Retry on different ports
            possible_ports = list_serial_ports(starts_with="/dev/ttyACM")
            if not possible_ports:
                logger.warning(f"EvoCamerav2._initialise: Unable to initialise SyncBoard. No ports starting with "
                               f"/dev/ttyACM found. Please check the connection, or change the port in "
                               f"acquisition.EvoCamerav2.__init__ and restart.")
            elif len(possible_ports) > 1:
                logger.warning(f"EvoCamerav2._initialise: Found multiple ports matching the pattern /dev/ttyACMX: "
                               f"{possible_ports}. Please let me know which port I should connect to by specifying "
                               f"the correct one in acquisition.EvoCamerav2.__init__ and restart.")
            else:
                self._syncboard_port = possible_ports[0]
                logger.debug(f"EvoCamerav2._initialise: Re-trying on port {self._syncboard_port}.")
                try:
                    self.syncboard: SyncBoardController = SyncBoardController.from_serial_port(port=self._syncboard_port)
                    self.syncboard.initialise()
                    if not self.syncboard.is_initialised():
                        raise ConfigError("EvoCamerav2._initialise: Unable to initialise SyncBoard.",
                                          error_code=ErrorCode.ERROR_SYNC_BOARD)
                    self._syncboard_is_alive = True
                    logger.info(f"_initialise: syncboard initialised on {self._syncboard_port}.")
                    logger.info(f"EvoCamerav2._initialise: Connected to SyncBoard on port {self._syncboard_port}.")
                except Exception as e:
                    self._syncboard_is_alive = False
                    logger.warning(
                        f"EvoCamerav2._initialise: Error connecting to SyncBoard on port {self._syncboard_port}: {e}.")
                    self.error_container.add_error(
                        new_error=SyncBoardError(message=str(e), error_code=ErrorCode.ERROR_SYNC_BOARD)
                    )
        if not self._get_syncboard_is_alive():
            self._syncboard_is_alive = False
            logger.warning("EvoCamerav2._initialise: SyncBoard is not alive.")
            self.error_container.add_error(
                new_error=SyncBoardError(message="SyncBoard is not alive.", error_code=ErrorCode.ERROR_SYNC_BOARD)
            )
        else:
            self._syncboard_is_alive = True

        self.disable_led()
        self.set_exposure()
        self._set_imaging_mode()

        try:
            self.brightfield_psu = KWR103(get_psu_port())
            self.brightfield_psu.connect()
            self.brightfield_psu.set_output(False)
            self.brightfield_psu.set_current(0.1)
            self.brightfield_psu.set_voltage(8)
            logger.warning(f"EvoCamerav2._initialise: Connecting to PSU on {get_psu_port()}.")
        except SerialException:
            logger.warning("EvoCamerav2._initialise: Brightfield not connected.")
            self.brightfield_psu = None

        return self._pvc_is_alive and self._tiger_is_alive and \
            self._syncboard_is_alive  # and (self.brightfield_psu is not None) IDRIS

    def _finalise(self):
        self.cam.close()
        pvc.uninit_pvcam()
        super()._finalise()

    def _take_frame(
            self,
            i_chan: Optional[LEDType] = None,
            brightness: float = 29,
            block: bool = False,
            reset_led: bool = True,
            disable_led: bool = False,
    ) -> Union[None, np.ndarray[(int, int), 'ImageConfigType.pxl_dtype']]:
        if not self._pvc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        curr_channel = self.current_channel
        if i_chan is not None:
            self._last_frame_channel = i_chan
            self.set_led(i_chan=i_chan, brightness=brightness, block=block)
        try:
            # self.mmc.snap_image()  # noqa
            pixels = self.cam.get_frame(timeout_ms=1000)
        except Exception as e:
            logger.warning(f"EvoCamera._take_frame: Received exception:\n{e}\nHave you disabled MM live mode?")
            return None
        # tagged_image = self.mmc.get_tagged_image()  # noqa
        # pixels = np.reshape(
        #     tagged_image.pix,
        #     newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        # )
        if i_chan is not None and reset_led and (not disable_led):
            self.set_led(i_chan=curr_channel, block=False)
        if disable_led:
            self.disable_led()
        return pixels

    def _set_exposure(self, exposure_time: Union[int, None] = None):
        if self._pvc_is_alive:
            self.cam.exp_time = exposure_time
        else:
            logger.warning("EvoCamera._set_exposure: cannot set exposure as pvc is not alive.")

    def _set_imaging_mode(self, imaging_mode: str = "Dynamic Range"):
        available_modes = ["Sensitivity", "Speed", "Dynamic Range", "Sub-Electron"]
        if imaging_mode not in available_modes:
            msg = f"EvoCamera._set_imaging_mode: {imaging_mode} not in {available_modes}."
            logger.warning(msg)
            return
        if self._pvc_is_alive:
            self.cam.readout_port = available_modes.index(imaging_mode)
        else:
            logger.warning("EvoCamera._set_imaging_mode: cannot set mode as PVC is not alive.")


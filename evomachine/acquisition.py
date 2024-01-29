from datetime import datetime
import itertools
import logging
import numpy as np
from pathlib import Path
import time
from typing import Dict, Iterator, List, Optional, Union, Tuple

import cv2
import matplotlib.pyplot as plt
from pycromanager import Core, Studio
from pynput import keyboard
import skimage

from asitiger.command import CRISPState
from asitiger.status import Status
import asitiger.tigercontroller
import delta

from evomachine.config import ConfigCRISP, CRISP_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigImage, ConfigLED, \
    ConfigObjective, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, OBJECTIVE_CONFIG_DEFAULT, EVO_FORMATTER
from evomachine.dmd import DMDColor
from evomachine.exceptions import CameraError, ConfigError, DMDError, ErrorCode, ErrorContainer, \
    EvoMachineError, StageError, TigerError


logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(EVO_FORMATTER)
logger.addHandler(handler)
logger.propagate = False


class AbstractCamera:
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: Optional[ConfigImage] = IMAGE_CONFIG_DEFAULT,
    ):
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self.cfg_device: ConfigDevice = cfg_device
        "Device configuration object."
        self.cfg_image: ConfigImage = cfg_image
        "Image configuration object."
        self._step: int = -1
        "Increments each time an image is taken."
        self._curr_pos: int = 0
        "Current position equalling 0 or i_pos passed to move_to_pos."

        self.cfg_device.check_config()

    def initialise(self):
        self._step = -1
        self._initialise()

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logger.warning(msg=msg)
        else:
            logger.warning("No errors for acquisition found.")

    def move_to_pos(self, i_pos: int) -> None:
        if i_pos not in range(self.cfg_device.num_pos):
            raise StageError("Position index {} out of range".format(i_pos),
                             ErrorCode.ERROR_STAGE_COORDINATES)
        self._curr_pos = i_pos
        success = self._move_stage_to_pos(i_pos=i_pos)
        if not success:
            raise StageError("Fault moving to position={}.".format(i_pos), ErrorCode.ERROR_STAGE_MOVEMENT)

    def get_frame(
            self,
            i_chan: Union[int, None],
            i_period: Union[int, None] = None,
            normalise: Optional[bool] = False,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        self._step += 1
        frame = self._take_frame(i_chan=i_chan, i_period=i_period)
        return self.normalise_frame(frame=frame) if (normalise and (frame is not None)) else frame

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        raise NotImplementedError()

    def halt_stage(self):
        raise NotImplementedError()

    def coordinate_is_out_of_bounds(self, coordinate: Dict[str, float]) -> bool:
        raise NotImplementedError()

    def display_save_frame(
            self,
            i_chan: Union[int, None],
            i_period: Optional[Union[int, None]] = None,
            path_to_save: Optional[Union[Path, str, None, bool]] = None,
            filename: Optional[Union[str, None]] = None,
            display_frame: Optional[bool] = True,
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        frame = self.get_frame(i_chan=i_chan, i_period=i_period)
        if frame is None:
            logger.warning(f"AbstractCamera.display_save_frame: self.get_frame returned None. Aborting...")
            return None
        if display_frame:
            self.plot_normalised_frame(frame=frame)
        if path_to_save is not None:
            self.save_frame(frame=frame, path_to_save=path_to_save, filename=filename)
        return frame

    @staticmethod
    def normalise_frame(frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype']):
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=frame.min(), vmax=frame.max())
        return cmap(norm(frame))

    def plot_normalised_frame(self, frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype']):
        image = self.normalise_frame(frame=frame)
        plt.imshow(image)
        plt.show()

    def save_frame(
            self,
            frame: np.ndarray[(int, int), 'ConfigImage.pxl_dtype'],
            path_to_save: Optional[Union[Path, str, bool]] = True,
            filename: Optional[Union[str, None]] = None,
    ):
        if not filename:
            filename = self.get_filename()

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

    def get_pos(self) -> int:
        return self._curr_pos

    def get_filename(self) -> str:
        return "evom_pos{:02d}_{}".format(self._curr_pos, datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))

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
            i_period: Union[int, None],
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        raise NotImplementedError()

    def crisp_autofocus(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        raise NotImplementedError()

    def crisp_disable(self):
        raise NotImplementedError()

    def crisp_is_locked(self):
        raise NotImplementedError()

    def crisp_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        raise NotImplementedError()

    def crisp_unlock(self):
        raise NotImplementedError()

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,

    ):
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

    def move_to(self, coordinates: Dict[str, int], block: Optional[bool] = False):
        raise NotImplementedError()

    def get_delta_fov(self):
        raise NotImplementedError()

    def keyboard_control(self):
        raise NotImplementedError()

    def set_exposure(self, exposure_time: Union[int, None] = None):
        raise NotImplementedError()

    def zero_coordinates(self):
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
    ):
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image)
        self.cfg_objective: ConfigObjective = cfg_objective
        "Objective configuration for mock images."
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
            i_period: Union[int, None],
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        return self.all_frames[self._curr_pos][i_period, i_chan, :, :]

    def get_filename(self) -> str:
        return ""

    def crisp_autofocus(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        self._crisp_is_locked = True

    def crisp_disable(self):
        self._crisp_is_locked = False

    def crisp_is_locked(self):
        return self._crisp_is_locked

    def crisp_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        return True

    def crisp_unlock(self):
        self._crisp_is_locked = False

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,

    ):
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

    def move_to(self, coordinates: Dict[str, int], block: Optional[bool] = False):
        return

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10
    def keyboard_control(self):
        return

    def set_exposure(self, exposure_time: Union[int, None] = None):
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
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image)
        if len(np.unique(filenames)) != len(filenames):
            raise ConfigError("TestCamera.__init__: must provide list with unique filenames.",
                              ErrorCode.ERROR_TEST_CAMERA_CONFIG)
        self.filenames: List[Union[str, Path]] = filenames
        "List of filenames for mock images."
        self.indices: Iterator[int] = itertools.cycle(range(len(filenames)))
        "Cyclic indices."
        self.cfg_crisp: ConfigCRISP = cfg_crisp
        "Settings for CRISP autofocus. Required for GUI interaction."
        self.cfg_focus: ConfigFocus = cfg_focus
        "Settings for initial software focus. Required for GUI interaction."
        self.cfg_objective: ConfigObjective = cfg_objective
        "Parameters of objective. Required for GUI interaction."
        self.pos_to_filename: Union[Dict[int, Union[Path, str]], None] = pos_to_filename
        "Optional dictionary mapping from unique position numbers (0,1,2,...) to filename."
        self.channel_settings: Dict[int, Dict] = {
            ConfigLED.LED_405_NM.value: {"X": 100, "Y": 0, "Z": 0, "F": 0},
            ConfigLED.LED_450_NM.value: {"X": 0, "Y": 100, "Z": 0, "F": 0},
            ConfigLED.LED_505_NM.value: {"X": 0, "Y": 0, "Z": 100, "F": 0},
            ConfigLED.LED_538_NM.value: {"X": 0, "Y": 0, "Z": 0, "F": 100},
            ConfigLED.LED_NO_LED.value: {"X": 0, "Y": 0, "Z": 0, "F": 0},
        }
        "Although this has no functionality for TestCamera, it is needed for tests together with the GUI."
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
            i_period: Union[int, None],
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        image = skimage.io.imread(self.filenames[self._next_filename_index])
        if self._crop_inds:
            return image[self._crop_inds[0][0]:self._crop_inds[0][1], self._crop_inds[1][0]:self._crop_inds[1][1]]
        else:
            return image

    def get_filename(self) -> str:
        return str(self.filenames[self._next_filename_index]).split("/")[-1]

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        return {ax.upper(): 0.0 for ax in axes}

    def halt_stage(self):
        return

    def coordinate_is_out_of_bounds(self, coordinate: Dict[str, float]) -> bool:
        return False

    def crisp_autofocus(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        self._crisp_is_locked = True

    def crisp_disable(self):
        self._crisp_is_locked = False

    def crisp_is_locked(self):
        return self._crisp_is_locked

    def crisp_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        return True

    def crisp_unlock(self):
        self._crisp_is_locked = False

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,

    ):
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

    def move_to(self, coordinates: Dict[str, int], block: Optional[bool] = False):
        self.increment_filename_index()

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10

    def keyboard_control(self):
        return

    def set_exposure(self, exposure_time: Union[int, None] = None):
        return

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
        super().__init__(cfg_device=cfg_device, cfg_image=cfg_image)

        self.tiger: Union[asitiger.tigercontroller.TigerController, None] = None
        "Object for serial communication with ASI tiger."
        self._tiger_is_alive: bool = False
        "Flag set in _initialise."
        self.current_channel: int = -1
        "Current LED channel set."
        self._last_frame_channel: int = -1
        "Channel used to take last frame."
        self.channel_settings: Dict[int, Dict] = {
            ConfigLED.LED_405_NM.value: {"X": 100, "Y": 0, "Z": 0, "F": 0},
            ConfigLED.LED_450_NM.value: {"X": 0, "Y": 100, "Z": 0, "F": 0},
            ConfigLED.LED_505_NM.value: {"X": 0, "Y": 0, "Z": 100, "F": 0},
            ConfigLED.LED_538_NM.value: {"X": 0, "Y": 0, "Z": 0, "F": 100},
            ConfigLED.LED_NO_LED.value: {"X": 0, "Y": 0, "Z": 0, "F": 0},
        }
        "LED intensity for i_chan=0,...,3."
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
        self.cfg_focus: ConfigFocus = cfg_focus
        "Settings for initial software focus."
        self.cfg_objective: ConfigObjective = cfg_objective
        "Parameters of objective."
        self.current_exposure_time: int = 0
        "Exposure time. Set in _initialise and provided in cfg_focus."

        self.mmc: Union[Core, None] = None
        "Micromanager Core object for taking images."
        self.studio: Union[Studio, None] = None
        "Micromanager Studio object for additional functions."
        self._mmc_is_alive: bool = False
        "Flag set in _initialise."

        self.initialise()  # Must be called before using EvoCamera

    def _initialise(self) -> None:
        try:
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
        self._disable_channels()
        self.set_exposure()

    def _get_tiger_is_alive(self) -> bool:
        if not self.tiger:
            return False
        try:
            _ = self.tiger.status()
            return True
        except ValueError:
            return False

    def _set_channel(self, i_chan: int):
        if i_chan not in self.channel_settings.keys():
            logger.error(msg=f"EvoCamera._set_channel: i_chan={i_chan} not in channels={self.channel_settings.keys()}.")
            return
        if self._tiger_is_alive:
            self.tiger.led(led_brightnesses=self.channel_settings[i_chan], card_address=self.card_address_led)
            self.current_channel = i_chan
        else:
            logger.error(msg=f"EvoCamera._set_channel: Tiger is not alive. Check ASI Tiger box and serial connection.")

    def _set_filter_wheel(self, i_pos: int):
        if i_pos not in self.filter_wheel_settings.keys():
            logger.error(msg=f"EvoCamera._set_filter_wheel: i_pos={i_pos} not in wheels={self.filter_wheel_settings}.")
            return
        if self._tiger_is_alive:
            self.tiger.filter_wheel(position=i_pos, card_address=self.card_address_fw)
        else:
            logger.error(msg=f"EvoCamera._set_filter_wheel: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")

    def _disable_channels(self):
        self._set_channel(i_chan=-1)

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        pos = {
            'X': self.cfg_device.coord_pos[i_pos][0],
            'Y': self.cfg_device.coord_pos[i_pos][1],
            'Z': self.cfg_device.coord_pos[i_pos][2],
        }
        return self._move_stage_to_coord(pos)

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
            i_period: Union[int, None],
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        if not self._mmc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        if i_chan is not None:
            self._last_frame_channel = i_chan
            curr_channel = self.current_channel
            self._set_channel(i_chan=i_chan)
        try:
            self.mmc.snap_image()
        except Exception as e:
            logger.warning(f"EvoCamera._take_frame: Received exception:\n{e}\nHave you disabled MM live mode?")
            return None
        self._disable_channels()
        tagged_image = self.mmc.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        if i_chan is not None:
            self._set_channel(i_chan=curr_channel)
        return pixels

    def get_filename(self) -> str:
        pos = self.tiger.where(['X', 'Y'])
        return "{}_X{}_Y{}_{}.tiff".format(
            ConfigLED.get_name(value_to_find=self._last_frame_channel).replace("_", ""),
            pos['X'],
            pos['Y'],
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        )

    def get_coordinates(self, axes: List[str]) -> Dict[str, float]:
        return self.tiger.where(axes)

    def halt_stage(self):
        self.tiger.halt()

    def coordinate_is_out_of_bounds(self, coordinate: Dict[str, float]) -> bool:
        return self.tiger.coordinate_is_out_of_bounds(coordinate)

    def crisp_autofocus(self, this_cfg_crisp: Optional[ConfigCRISP] = None, user_input: Optional[bool] = True):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_autofocus: Device not alive.")
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
        is_configured = self.crisp_configure(this_cfg_crisp=cfg_crisp)
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
            self.crisp_unlock()
        time.sleep(cfg_crisp.pause_short)
        curr_state = self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None)
        logger.info(f"CRISP: Finalising autofocus. Current state is {curr_state}.")

    def crisp_disable(self):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_disable: Device not alive. Trying to disable anyway.")
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.IDLE)

    def crisp_is_locked(self):
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_is_locked: Device not alive.")
            return
        return self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=None) == 'F'

    def crisp_configure(self, this_cfg_crisp: Optional[ConfigCRISP] = None) -> bool:
        if not self._tiger_is_alive:
            logger.error(f"EvoCamera.crisp_set_parameters: Device not alive.")
            return False
        cfg_crisp = this_cfg_crisp if this_cfg_crisp else self.cfg_crisp
        try:
            cfg_crisp.check_config()
            logger.info(f"CRISP: Configuring CRISP with following parameters:\n{cfg_crisp}")
        except ConfigError as e:
            logger.error(f"CRISP: Bad configuration:\n{e}\nCannot use CRISP.")
            return False
        self.crisp_unlock()
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

    def crisp_unlock(self):
        self.tiger.crisp_get_set_state(card_address=self.card_address_crisp, value=CRISPState.UNLOCK)

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[int] = None,
            rel_range_override: Optional[int] = None,

    ):
        if not all([self._mmc_is_alive, self._tiger_is_alive]):
            logger.error(f"EvoCamera.software_focus: Device(s) not alive. "
                         f"Tiger={self._tiger_is_alive}, MMC={self._mmc_is_alive}.")
            return
        cfg_focus = cfg_focus if cfg_focus else self.cfg_focus
        if focus_channel_override is not None:
            cfg_focus.focus_channel = focus_channel_override
        if rel_range_override is not None:
            cfg_focus.rel_range = rel_range_override
        try:
            cfg_focus.check_config()
        except ConfigError as e:
            logger.warning(f"EvoCamera.software_focus: Invalid focus configuration:\n{e.message}\nAborting...")
            return
        curr_pos = self.tiger.where(['Z'])['Z']
        coords = range(curr_pos-cfg_focus.rel_range, curr_pos+cfg_focus.rel_range, cfg_focus.steps_size)
        user_input = input(f"EvoCamera.software_focus: Starting software autofocus configured as\n"
                           f"{cfg_focus.__str__()}\nThis will move the stage up and down in the range "
                           f"[{(curr_pos-cfg_focus.rel_range)/10},{(curr_pos+cfg_focus.rel_range)/10}] μm"
                           f" (current position = {curr_pos/10} μm). "
                           f"If there are objects blocking the stage movement, this will crash the "
                           f"objective and break it. Do you want to proceed? (yes/no): ")
        if user_input.lower() == "yes":
            logger.info("EvoCamera.software_focus: Proceeding with software focus. Disabling MM live mode.")
        else:
            logger.info("EvoCamera.software_focus: Aborting software focus.")
            return
        old_channel = self.current_channel
        self.set_exposure(exposure_time=int(cfg_focus.exposure_time))
        self.studio.live().set_live_mode(False)
        best_focus_score = 0
        best_focus_position = 0
        for ipos, z_coord in enumerate(coords):
            self._move_stage_to_coord({'Z': z_coord})
            image_raw = self.display_save_frame(
                i_chan=cfg_focus.focus_channel,
                i_period=None,
                path_to_save=False,
                filename=None,
                display_frame=False,
            )
            if image_raw is None:
                logger.warning("EvoCamera.software_focus: self._take_frame returned None. Aborting...")
                return
            laplacian = cv2.Laplacian(image_raw, cv2.CV_64F)
            focus_score = laplacian.var()
            if focus_score > best_focus_score:
                best_focus_position = ipos
                best_focus_score = focus_score
        best_coordinate = coords[best_focus_position]
        logger.info(f"EvoCamera.software_focus: Finished scanning. Coordinate before focus={curr_pos / 10} μm,"
                    f"coordinate after focus={best_coordinate / 10} μm. Finalising software_focus.")
        self._move_stage_to_coord({'Z': best_coordinate})
        self._set_channel(i_chan=old_channel)

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

    def move_to(self, coordinates: Dict[str, int], block: Optional[bool] = False):
        self.tiger.move(coordinates=coordinates)
        if block:
            self.tiger.wait_until_idle()

    def get_delta_fov(self):
        return self.cfg_objective.fov_size * 10

    def _move_fov(self, x_or_y: str, sign: int, multiplier: Optional[float] = 1.0, block: Optional[bool] = False):
        pos = self.tiger.where([x_or_y.upper()])
        if x_or_y.upper() not in pos:
            raise TigerError(f"EvoCamera._move_fov: queried coordinate {x_or_y.upper()} not in response: {pos}.",
                             ErrorCode.ERROR_TIGER_NO_DATA)
        pos[x_or_y.upper()] += int(sign * self.cfg_objective.fov_size * 10 * multiplier)
        self.move_to(coordinates=pos, block=block)

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

    def set_exposure(self, exposure_time: Union[int, None] = None):
        if self._mmc_is_alive:
            new_exposure_time = self.cfg_focus.exposure_time if exposure_time is None else exposure_time
            self.mmc.set_exposure(new_exposure_time)
            self.current_exposure_time = new_exposure_time

    def zero_coordinates(self):
        self.tiger.zero()







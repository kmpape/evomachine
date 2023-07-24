import logging
import numpy as np
from typing import Dict, List, Optional, Union

from pycromanager import Core

import asitiger.tigercontroller
import delta

from evomachine.config import ConfigDevice, ConfigImage
from evomachine.exceptions import CameraError, ErrorCode, ErrorContainer, EvoMachineError, StageError, TigerError


logger = logging.getLogger(__name__)


class AbstractCamera:
    def __init__(self, cfg_device: ConfigDevice):
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self.cfg_device: ConfigDevice = cfg_device
        "Device configuration object."
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
            logging.warning(msg=msg)
        else:
            logging.warning("No errors for acquisition found.")

    def move_to_pos(self, i_pos: int) -> None:
        if i_pos not in range(self.cfg_device.num_pos):
            raise StageError("Position index {} out of range".format(i_pos),
                             ErrorCode.ERROR_STAGE_COORDINATES)
        self._curr_pos = i_pos
        success = self._move_stage(i_pos=i_pos)
        if not success:
            raise StageError("Fault moving to position={}.".format(i_pos), ErrorCode.ERROR_STAGE_MOVEMENT)

    def get_frame(
            self,
            i_chan: int,
            i_period: Union[int, None] = None,
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:  # TODO: check frame data type
        self._step += 1
        return self._take_frame(i_chan=i_chan, i_period=i_period)

    def get_pos(self) -> int:
        return self._curr_pos

    def _initialise(self) -> None:
        raise NotImplementedError()

    def _move_stage(
            self,
            i_pos: int,
    ) -> bool:
        raise NotImplementedError()

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        raise NotImplementedError()


class DeltaCamera(AbstractCamera):
    """
    A class to mock the acquisition of frames.
    """
    def __init__(self, cfg_device: ConfigDevice):
        super().__init__(cfg_device=cfg_device)

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

    def _move_stage(
            self,
            i_pos: int,
    ) -> bool:
        return True

    def _initialise(self) -> None:
        self._curr_period = -1

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        return self.all_frames[self._curr_pos][i_period, i_chan, :, :]


class EvoCamera(AbstractCamera):
    """
    EvoMachine acquisition class.
    """
    def __init__(self, cfg_device: ConfigDevice):
        super().__init__(cfg_device=cfg_device)

        self.tiger: Union[asitiger.tigercontroller.TigerController, None] = None
        "Object for serial communication with ASI tiger."
        try:
            self.tiger: asitiger.tigercontroller.TigerController = \
                asitiger.tigercontroller.TigerController.from_serial_port(port=cfg_device.tiger_port)
        except Exception as e:
            logging.warning(f"Error connecting to ASITiger:\n{e}\n---\nContinuing with execution.")
            self.error_container.add_error(
                new_error=TigerError(message=str(e), error_code=ErrorCode.ERROR_TIGER_SERIAL_CONNECTION)
            )

        self.channel_settings: Dict[int, Dict] = {
            0: {"X": 100, "Y": 0, "Z": 0, "F": 0},
            1: {"X": 0, "Y": 100, "Z": 0, "F": 0},
            2: {"X": 0, "Y": 0, "Z": 100, "F": 0},
            3: {"X": 0, "Y": 0, "Z": 0, "F": 100},
            -1: {"X": 0, "Y": 0, "Z": 0, "F": 0},
        }
        "LED intensity for i_chan=0,...,3."
        self.card_address: int = 7
        "LED card address on ASI tiger."
        self.mmc: Union[Core, None] = None
        "Micromanager object for taking images."
        try:
            self.mmc = Core()
        except Exception as e:
            logging.warning(f"Error connecting to Micro Manager:\n{e}\n---\nContinuing with execution.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_MMC_NOT_ALIVE)
            )

        if not self._tiger_is_alive():
            logging.warning("ASITiger is not alive.\n---\nContinuing with execution.")
            self.error_container.add_error(
                new_error=TigerError(message="ASITiger is not alive.", error_code=ErrorCode.ERROR_TIGER_NOT_ALIVE)
            )

    def _tiger_is_alive(self) -> bool:
        if not self.tiger:
            return False
        try:
            answer = self.tiger.status()
            return True
        except ValueError:
            return False

    def _set_channel(self, i_chan: int):
        self.tiger.led(led_brightnesses=self.channel_settings[i_chan], card_address=self.card_address)

    def _disable_channels(self):
        self._set_channel(i_chan=-1)

    def _move_stage(
            self,
            i_pos: int,
    ) -> bool:
        pos = {
            'X': self.cfg_device.coord_pos[i_pos][0],
            'Y': self.cfg_device.coord_pos[i_pos][1],
            'Z': self.cfg_device.coord_pos[i_pos][2],
        }
        self.tiger.move(coordinates=pos)
        return True

    def _initialise(self) -> None:
        self._disable_channels()

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        self._set_channel(i_chan=i_chan)
        self.mmc.snap_image()
        self._disable_channels()
        tagged_image = self.mmc.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        return pixels


import numpy as np
from typing import List, Union

import delta

from evomachine.config import ConfigDevice, ConfigImage
from evomachine.exceptions import StageError, ErrorCode


class AbstractCamera:
    def __init__(self, cfg_device: ConfigDevice):
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
            i_period: Union[int, None],
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

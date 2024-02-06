import logging
import numpy as np
from typing import List

import delta
from delta.config import Config

from evomachine.acquisition import AbstractCamera
from evomachine.config import ConfigDevice, ConfigImage
from evomachine.exceptions import ErrorContainer
from evomachine.positionrt import PositionRT

from evomachine.strategy import AbstractStrategy

logger = logging.getLogger(__name__)


class Automaton:
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: ConfigImage,
            cfg_delta: delta.config.Config,
            camera: AbstractCamera,
            strategy: AbstractStrategy,
    ):
        self._cfg_device: ConfigDevice = cfg_device
        "Device configuration object defining geometry."
        self._cfg_image: ConfigImage = cfg_image
        "Image configuration object defining size and data type."
        self._curr_pos: int = 0
        "Current position."
        self._curr_period: int = 0
        "Incremented after completing one round of imaging the whole device."
        self._curr_step: int = 0
        "Incremented every time a picture is taken."
        self._camera: AbstractCamera = camera
        "Camera object which can be a real camera or a class that reads from the disk."
        self._pos_processor: List[PositionRT] = [
            PositionRT(
                position_nb=i,
                config=cfg_delta,
                cfg_image=cfg_image,
                verbose=cfg_device.image_processing_verbosity
            )
            for i in range(self._cfg_device.num_pos)
        ]
        "List of Delta objects to process the images."
        self._all_frames: List[np.ndarray[(int, int, int, int), ConfigImage.pxl_dtype]] = [
            np.empty((2, self._cfg_device.num_chan, self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz),
                     dtype=self._cfg_image.pxl_dtype)
            for _ in range(self._cfg_device.num_pos)
        ]
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._ref_frames: List[np.ndarray[(int, int, int), ConfigImage.pxl_dtype]] = [
            np.empty((self._cfg_device.num_chan, self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz),
                     dtype=self._cfg_image.pxl_dtype)
            for _ in range(self._cfg_device.num_pos)
        ]
        "List indexed by i_pos w. reference image array: channels x pxl_vert x pxl_horiz."

        self.error_container: ErrorContainer = ErrorContainer()
        "Container for errors."

        self._strategy: AbstractStrategy = strategy

    def initialise(self):
        
        # FIXME: should this called twice? Also called below
        self._camera.initialise()
        
        for i_pos in range(self._cfg_device.num_pos):
            self._camera.move_to_pos(i_pos=i_pos)
            for i_chan in range(self._cfg_device.num_chan):
                self._ref_frames[i_pos][i_chan, :, :] = self._camera.get_frame(i_chan=i_chan, i_period=0)
            self._pos_processor[i_pos].initialise(self._ref_frames[i_pos])
            self.increment_pos()
        
        # FIXME: should this called twice? Also called above
        self._camera.initialise()

        assert self._curr_pos == 0
        assert self._curr_period == 1 # Note that each ROI keeps track of _curr_period as well

        self._strategy.initialise()

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logging.warning(msg=msg)
        else:
            logging.warning("No errors for automaton found.")
        self._camera.check_status()

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_pos + 1 == self._cfg_device.num_pos)
                             else self._curr_period)
        self._curr_pos = (self._curr_pos + 1) % self._cfg_device.num_pos

    def process(self):
        self._take_image()
        self._process_position()
        self._strategy.callback(
            fov_id=self._curr_pos, 
            t=self._curr_period, 
            data=self._pos_processor[self._curr_pos].get_data()
        )
        self.increment_pos()

    def _take_image(self):
        self._camera.move_to_pos(i_pos=self._curr_pos)
        for i_chan in range(self._cfg_device.num_chan):  # TODO: do we need this? Images also held in PipelineRT
            self._all_frames[self._curr_pos][0, i_chan, :, :] = self._all_frames[self._curr_pos][1, i_chan, :, :]
            self._all_frames[self._curr_pos][1, i_chan, :, :] = self._camera.get_frame(i_chan=i_chan,
                                                                                       i_period=self._curr_period)

    def _process_position(self):
        self._pos_processor[self._curr_pos].process_new_frame(
            new_frame=self._all_frames[self._curr_pos][1, :, :, :]  # NOTE: frame passed by reference
        )

    def get_period(self) -> int:
        return self._curr_period

    def get_pos(self) -> int:
        return self._curr_pos

    def get_frame(self, i_pos: int, i_chan: int) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        return self._all_frames[i_pos][1, i_chan, :, :]



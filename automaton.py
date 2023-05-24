import numpy as np
from numpy import ndarray
from typing import List, Type

import delta
from delta.config import Config

from evomachine.acquisition import AbstractCamera
from evomachine.config import ConfigDevice, ConfigImage
from evomachine.positionrt import PositionRT


class Automaton:
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: ConfigImage,
            cfg_delta: delta.config.Config,
            camera: AbstractCamera,
    ):
        self._cfg_device: ConfigDevice = cfg_device
        "Device configuration object defining geometry."
        self._cfg_image: ConfigImage = cfg_image
        "Image configuration object defining size and data type."
        self._curr_pos: int = 0
        "Current position."
        self._curr_period: int = 0
        "Incremented after completing one round of imaging the whole device."
        self._curr_step: int = 0  # TODO: should be called differently
        "Incremented every time a picture is taken."
        self._camera: AbstractCamera = camera
        "Camera object which can be a real camera or a class that reads from the disk."
        self._pos_processor: List[PositionRT] = []
        "List of Delta objects to process the images."
        self._all_frames: List[np.ndarray[(int, int, int, int), ConfigImage.pxl_dtype]] = []
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._ref_frames: List[np.ndarray[(int, int, int), ConfigImage.pxl_dtype]] = []
        "List indexed by i_pos w. reference image array: channels x pxl_vert x pxl_horiz."

        for i in range(self._cfg_device.num_pos):
            self._pos_processor[i] = PositionRT(position_nb=i, config=cfg_delta, cfg_image=cfg_image,
                                                verbose=cfg_device.image_processing_verbosity)
            self._all_frames[i] = np.empty((2,
                                            self._cfg_device.num_chan,
                                            self._cfg_image.pxl_vert,
                                            self._cfg_image.pxl_horiz),
                                           dtype=self._cfg_image.pxl_dtype)
            self._ref_frames[i] = np.empty((self._cfg_device.num_chan,
                                            self._cfg_image.pxl_vert,
                                            self._cfg_image.pxl_horiz),
                                           dtype=self._cfg_image.pxl_dtype)

    def initialise(self):
        self._camera.initialise()
        for i_pos in range(self._cfg_device.num_pos):
            self._camera.move_to_pos(i_pos=i_pos)
            for i_chan in range(self._cfg_device.num_chan):
                self._ref_frames[i_pos][i_chan, :, :] = self._camera.get_frame(i_chan=i_chan, i_period=0)
            self._pos_processor[i_pos].initialise(self._ref_frames[i_pos])
            self.increment_pos()
        self._camera.initialise()
        self._curr_pos = 0  # TODO: should not need to set this
        self._curr_period = 0

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_pos + 1 == self._cfg_device.num_pos)
                             else self._curr_period)
        self._curr_pos = (self._curr_pos + 1) % self._cfg_device.num_pos

    def process(self):
        for i_chan in range(self._cfg_device.num_chan):  # TODO: do we need this? Images also held in PipelineRT
            self._all_frames[self._curr_pos][0, i_chan, :, :] = self._all_frames[self._curr_pos][1, i_chan, :, :]
            self._all_frames[self._curr_pos][1, i_chan, :, :] = self._camera.get_frame(i_chan=i_chan,
                                                                                       i_period=self._curr_period)

        self._process_position()
        # TODO: controller, DMD actuation etc.
        self.increment_pos()

    def _process_position(self):
        self._pos_processor[self._curr_pos].process_new_frame(new_frame=self._all_frames[self._curr_pos][1, :, :, :])

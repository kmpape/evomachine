import logging
import numpy as np
from time import perf_counter
from typing import List

from cellpose_omni import models

import config
from cells import MotherMachine

logger = logging.getLogger(__name__)


class ImageProcessor(object):
    def __init__(self, frame_rows: int, frame_cols: int, num_frames: int, dtype: 'np.dtype'):
        self.new_frames: List[np.array] = [np.zeros((frame_rows, frame_cols), dtype=dtype)
                                           for _ in range(0, num_frames)]
        self.old_frames: List[np.array] = [np.zeros((frame_rows, frame_cols), dtype=dtype)
                                           for _ in range(0, num_frames)]

    def process_image(self, new_frame: np.ndarray, frame_id: int, mother_machine: 'MotherMachine'):
        self._update_frames(new_frame, frame_id)
        if config.DEBUG_MODE:
            time_start = perf_counter()
        self._process_image(mother_machine)
        if config.DEBUG_MODE:
            elapsed: float = perf_counter() - time_start
            logger.debug(f"Processed image in {elapsed:.9f}s")

    def _update_frames(self, new_frame: np.ndarray, frame_id: int):
        self.old_frames[frame_id] = self.new_frames[frame_id]
        self.new_frames[frame_id] = new_frame

    def _process_image(self, mother_machine: 'MotherMachine'):
        raise NotImplementedError()


# class OmniposeProcessor(ImageProcessor):
#     def __init__(self, model_path: str = config.OMNIPOSE_MODEL_PATH):
#         super().__init__(debug_mode=debug_mode)
#         self.model_path: str = model_path
#         self.model: 'models.CellposeModel' = models.CellposeModel(gpu=True, pretrained_model=self.model_path, omni=True,
#                                                                   concatenation=True)
#         logger.debug("Using CellposeModel={}".format(self.model_path))
#
#     def process_image(self, new_image: np.ndarray):
#         if self.debug_mode:
#             time_start = perf_counter()
#         # TODO
#         masks, flows, styles = self.model.eval(x=new_image, batch_size=config.OMNIPOSE_BATCH_SIZE, channels=[0, 0],
#                                                rescale=None, mask_threshold=-1, transparency=True, flow_threshold=0., omni=omni,
#                                                resample=True, verbose=0)
#         if self.debug_mode:
#             elapsed: float = perf_counter() - time_start

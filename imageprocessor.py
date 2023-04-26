import logging
import numpy as np
from time import perf_counter
from typing import List, Tuple

# from cellpose_omni import models
import delta

import config
from mothermachine import MotherMachine

logger = logging.getLogger(__name__)


class ImageProcessor(object):
    def __init__(self, num_channels: int, frame_rows: int, frame_cols: int, num_frames: int, dtype: 'np.dtype'):
        self.frame_size: Tuple[int, int, int] = (num_channels, frame_rows, frame_cols)
        self.new_frames: List[np.array] = [np.zeros(self.frame_size, dtype=dtype) for _ in range(0, num_frames)]
        self.old_frames: List[np.array] = [np.zeros(self.frame_size, dtype=dtype) for _ in range(0, num_frames)]

    def process_image(self, new_frame: np.ndarray, frame_id: int, mother_machine: 'MotherMachine'):
        self._update_frames(new_frame, frame_id)
        time_start = 0.0
        if config.DEBUG_MODE:
            time_start = perf_counter()
        self._process_image(mother_machine=mother_machine, frame_id=frame_id)
        if config.DEBUG_MODE:
            elapsed: float = perf_counter() - time_start
            logger.debug(f"Processed image in {elapsed:.9f}s")

    def _update_frames(self, new_frame: np.ndarray, frame_id: int):
        self.old_frames[frame_id] = self.new_frames[frame_id]
        self.new_frames[frame_id] = new_frame

    def _process_image(self, mother_machine: 'MotherMachine', frame_id: int):
        raise NotImplementedError()


# class OmniposeProcessor(ImageProcessor):
#     def __init__(self, frame_rows: int, frame_cols: int, num_frames: int, dtype: 'np.dtype',
#                  model_path: str = config.OMNIPOSE_MODEL_PATH):
#         super().__init__(frame_rows=frame_rows, frame_cols=frame_cols, num_frames=num_frames, dtype=dtype)
#         self.model_path: str = model_path
#         self.model: 'models.CellposeModel' = models.CellposeModel(gpu=config.USE_GPU, pretrained_model=self.model_path,
#                                                                   omni=True, concatenation=True)
#
#     def _process_image(self, mother_machine: 'MotherMachine', frame_id: int):
#         masks, flows, styles = self.model.eval(x=self.new_frames[frame_id], batch_size=config.OMNIPOSE_BATCH_SIZE,
#                                                channels=[0, 0], rescale=None, mask_threshold=-1, transparency=True,
#                                                flow_threshold=0., omni=True, resample=True, verbose=0)


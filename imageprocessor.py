import logging
import numpy as np
from time import perf_counter

from cellpose_omni import models

import config

logger = logging.getLogger(__name__)


class ImageProcessor(object):
    def __init__(self, debug_mode: bool = config.DEBUG_MODE):
        self.debug_mode: bool = debug_mode

    def process_image(self, new_image: np.ndarray, old_image: np.ndarray) -> np.ndarray:
        if self.debug_mode:
            time_start = perf_counter()
        masks, flows, styles = self.model.eval(x=new_image, batch_size=batch_size, channels=[0, 0], rescale=None,
                                               mask_threshold=-1, transparency=True, flow_threshold=0., omni=omni,
                                               resample=True, verbose=0)
        if self.debug_mode:
            elapsed: float = perf_counter() - time_start


class OmniposeProcessor(ImageProcessor):
    def __init__(self, debug_mode: bool = config.DEBUG_MODE, model_path: str = config.OMNIPOSE_MODEL_PATH):
        super().__init__(debug_mode=debug_mode)
        self.model_path: str = model_path
        self.model: 'models.CellposeModel' = models.CellposeModel(gpu=True, pretrained_model=self.model_path, omni=True,
                                                                  concatenation=True)
        logger.debug("Using CellposeModel={}".format(self.model_path))

    def process_image(self, new_image: np.ndarray):
        if self.debug_mode:
            time_start = perf_counter()
        masks, flows, styles = self.model.eval(x=new_image, batch_size=config.OMNIPOSE_BATCH_SIZE, channels=[0, 0],
                                               rescale=None, mask_threshold=-1, transparency=True, flow_threshold=0., omni=omni,
                                               resample=True, verbose=0)
        if self.debug_mode:
            elapsed: float = perf_counter() - time_start

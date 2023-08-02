import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import unittest

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.acquisition import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, IMAGE_CONFIG_DELTA_BENCH, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.positionrt import TIMER_POSITION

TEST_VERBOSITY = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(TEST_VERBOSITY)
handler = logging.StreamHandler()
handler.setLevel(TEST_VERBOSITY)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

EPS_REL = 10**(-6)
EPS_REL_APPROX = 10**(-2)
APPROX_EQUAL = False  # allow for assertions with approximate equality
NUM_PXL_DIFF = 10  # number of admissible pixel difference in an image/mask
ABS_PXL_DIFF = 2  # number of admissible pixel difference for features

THIS_DIR = Path(__file__).parent

if __name__ == '__main__':
    this_cfg_device = DEVICE_CONFIG_DELTA_SIM
    this_cfg_device.image_processing_verbosity = 0
    this_cfg_image = IMAGE_CONFIG_DELTA_BENCH
    this_cfg_image.use_track_RT = True
    automaton: Automaton = Automaton(
        cfg_device=this_cfg_device,
        cfg_image=this_cfg_image,
        cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
        camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
    )
    automaton.initialise()

    for i_period in range(1, DEVICE_CONFIG_DELTA_SIM.num_periods):
        # print("PROCESS AT POS {} and PERIOD {}".format(automaton.get_pos(), automaton.get_period()))
        # Process an additional step at position 0
        assert automaton.get_pos() == 0
        assert automaton.get_period() == i_period
        automaton.process()
        # Process an additional step at position 1
        assert automaton.get_pos() == 1
        assert automaton.get_period() == i_period
        automaton.process()

    print(f"Total number of positions {len(automaton._pos_processor)}")
    print(f"Number of ROI per position {[len(pos.rois) for pos in automaton._pos_processor]}")
    TIMER_POSITION.display_timings(scaling=1.0 / np.mean([len(pos.rois) for pos in automaton._pos_processor]))
    timings_pos = TIMER_POSITION.get_timings_per_call()
    for key, val in timings_pos.items():
        print(f"{key}: {val}")
    if TIMER_ROI.enabled:
        TIMER_ROI.display_timings()

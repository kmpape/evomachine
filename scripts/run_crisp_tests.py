import datetime
import glob
from multiprocessing import Event, Lock, Process, Queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from pathlib import Path
import random
import sys
import time
import tensorrt  # noqa

sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/de-lta-rt")

from evomachine.acquisition import TestCamera, EvoCamera, EvoCamerav2
from evomachine.automaton import Automaton
from evomachine.config_delta import ConfigImageProcessor, ConfigImageProcessorFactory
from evomachine.config import ConfigCamera, ConfigCameraFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
    import pygame

from evomachine.evotypes import DMDCalibConfigTypeFactory, FilterWheelType, LEDType
from evomachine.coordinates import Coordinate
from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import BasicStrategy
from evomachine.utils import rotation_correction
from delta.roirt import *
from delta.rt import *
from asitiger.command import CRISPState


camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()
processor_config.cfg_delta.whole_frame_drift = True
cfg = DMDCalibConfigTypeFactory.default()
cfg.brightness = 1.0

cam = EvoCamerav2(cfg_camera=camera_config)
cam.initialise()
cam.set_led(i_chan=LEDType.LED_450_NM, brightness=15.0)
cam.set_filter_wheel(FilterWheelType.FILTER)
cam.set_exposure(exposure_time=100)

dmd = DMDControl()
dmd.initialise()
dmd.display_full()

# cam.get_coordinates(['X', 'Y', 'Z'])
pos0 = {'X': 1319, 'Y': -73351}
pos1 = {'X': 188, 'Y': 0}

cam.autofocus_initialise(user_input=False)
cam.tiger.crisp_get_set_state(card_address=cam.card_address_crisp, value=CRISPState.LOCK)

duration = 5
now = time.perf_counter()
end = now + duration

cam.move_to(coordinate=pos0, block=False)

cam.move_to(coordinate=pos1, block=False)
while now < end:
    print("STAGE=" + str(cam.tiger.status()) + ", CRISP=" +
          cam.tiger.crisp_get_set_state(card_address=cam.card_address_crisp, value=None))
    print(cam.tiger.is_busy(cam.card_address_crisp))
    time.sleep(0.001)

print("End")

cam.tiger.crisp_get_set_state(card_address=cam.card_address_crisp, value=CRISPState.UNLOCK)

# New function
cam._move_stage_to_coord(pos0, block=True)

import sys
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
from evomachine.acquisition import EvoCamera, DMDControl
from evomachine.config import DEVICE_CONFIG_EVO_TEST
from evomachine.utils import Timer
import pygame
import sys
import os
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    i_chan = 2
    cam = EvoCamera(DEVICE_CONFIG_EVO_TEST)
    dmd = DMDControl()
    tig = cam.tiger
    cam.initialise()
    cam._set_channel(-1)
    i_trials = 100
    delay = 0.09
    path_to_save = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/images/2023-09-28/"
    mytimer = Timer(timer_level=0, name="bench_dmd", enabled=True)
    for i_trial in range(i_trials):
        dmd.display_none()
        time.sleep(0.5)
        mytimer.start("delay", 0)
        dmd.display_full()
        time.sleep(delay)
        ref_full = cam.get_frame(i_chan=i_chan)
        mytimer.stop("delay", 0)
        cam.save_frame(
            path_to_save=path_to_save,
            frame=ref_full,
            filename=f"bench_ops_delay{delay:.2f}_trial{i_trial}.png",
            normalise=True,
        )
    mytimer.display_timings()

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest
import sys

sys.path.append('/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append('/home/hslab/workspace_python/conda_evomachine3.9/de-lta-rt')
sys.path.append('/home/hslab/workspace_python/conda_evomachine3.9/asitiger')

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.acquisition import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.positionrt import TIMER_POSITION

this_cfg_device = DEVICE_CONFIG_DELTA_SIM
this_cfg_device.image_processing_verbosity = 0
automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=IMAGE_CONFIG_DELTA_SIM,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
automaton.initialise()
automaton.process()

pos = automaton._pos_processor[0]
roi = pos.rois[10]
img_0 = roi.img_stack[0]
img_1 = roi.img_stack[1]

x_0 = cv2.resize(img_0, dsize=DEFAULT_CONFIG_MOTHERMACHINE.target_size_seg[::-1])
x_0 = x_0[np.newaxis, :, :, np.newaxis]
x_0 = x_0[0, :, :, 0]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=1, extent=(0, img_0.shape[1], 0, img_0.shape[0]), aspect='equal')
axs[0].set_title('Input')
axs[1].imshow(x_0, cmap='gray', vmin=0, vmax=1, extent=(0, x_0.shape[1], 0, x_0.shape[0]), aspect='equal')
axs[1].set_title('Input Resized')
plt.show()



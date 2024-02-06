import copy

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.strategy import NoStrategy
from evomachine.acquisition import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_BENCH, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.positionrt import TIMER_POSITION

this_cfg_device = DEVICE_CONFIG_DELTA_SIM
this_cfg_device.image_processing_verbosity = 0
automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=IMAGE_CONFIG_DELTA_BENCH,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
            strategy=NoStrategy(),
        )
automaton.initialise()
ipos = 0
iroi = 0
roi = automaton._pos_processor[ipos].rois[iroi]
fig, axs = plt.subplots(2, 10)
axs[0, 0].imshow(roi.img_stack[1], cmap='gray',
              vmin=0, vmax=max(roi.img_stack[1].max(), roi.img_stack[1].max()))
for i in range(9):
    for j in range(len(automaton._pos_processor)):
        automaton.process()
    labels = delta.utils.label_seg(roi.get_seg(1))
    axs[0, i+1].imshow(roi.img_stack[1], cmap='gray',
                       vmin=0, vmax=max(roi.img_stack[1].max(), roi.img_stack[1].max()))
    cmap = plt.cm.get_cmap('tab10', labels.max())
    axs[1, i+1].imshow(labels, cmap=cmap, vmin=0, vmax=labels.max())

plt.show()

ipos = 0
pos = automaton._pos_processor[ipos]
img_0 = pos.reference[0, :, :]
img_1 = copy.deepcopy(pos.reference[0, :, :])
img_1_patched = img_1
for iroi in range(len(pos.rois)):
    box = pos.rois[iroi].box
    patch = np.zeros((box.ybr-box.ytl, box.xbr - box.xtl))
    img_1_patched = box.patch(img_1_patched, patch)

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[0].set_title('Reference')
axs[1].imshow(img_1_patched, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[1].set_title('Reference with boxes')
plt.show()

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
automaton.process()

pos = automaton._pos_processor[1]
roi = pos.rois[6]
img_0 = roi.img_stack[0]
img_1 = roi.img_stack[1]

fig, axs = plt.subplots(1, 1)
axs.imshow(pos.new_frame[1, :, :], cmap='gray', vmin=pos.new_frame[1, :, :].min(), vmax=pos.new_frame[1, :, :].max())
axs.set_title('Concatenated frame')
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('ROI 6 at t-1')
axs[1].imshow(img_1, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('ROI 6 at t')
plt.show()

box = roi.box

img_0 = np.tile(automaton._all_frames[0][0, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
img_1 = np.tile(automaton._all_frames[0][1, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[0].set_title('t-1')
axs[1].imshow(img_1, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[1].set_title('t')
plt.show()

ipos = 0
pos = automaton._pos_processor[ipos]
img_0 = np.tile(automaton._all_frames[ipos][0, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
img_1 = np.tile(automaton._all_frames[ipos][1, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
img_1_patched = img_1
for iroi in range(20):
    box = pos.rois[iroi].box
    patch = np.zeros((box.ybr-box.ytl, box.xbr - box.xtl))
    img_1_patched = box.patch(img_1_patched, patch)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[0].set_title('t-1')
axs[1].imshow(img_1_patched, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[1].set_title('t')
plt.show()


roi = pos.rois[7]
img_0 = roi.img_stack[0]
img_1 = roi.img_stack[1]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('ROI 7 at t-1')
axs[1].imshow(img_1, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('ROI 7 at t')
plt.show()


roi = pos.rois[5]
img_0 = roi.img_stack[0]
img_1 = roi.img_stack[1]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('ROI 5 at t-1')
axs[1].imshow(img_1, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('ROI 5 at t')
plt.show()


ipos = 0
pos = automaton._pos_processor[ipos]
img_0 = np.tile(automaton._all_frames[ipos][0, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
img_1 = np.tile(automaton._all_frames[ipos][1, 0, :, :], IMAGE_CONFIG_DELTA_BENCH.tile_image)
img_1_patched = img_1
for iroi in range(20):
    box = pos.rois[iroi].box
    patch = np.zeros((box.ybr-box.ytl, box.xbr - box.xtl))
    img_1_patched = box.patch(img_1_patched, patch)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[0].set_title('t-1')
axs[1].imshow(img_1_patched, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
axs[1].set_title('t')
plt.show()
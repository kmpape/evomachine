import copy

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest
import sys

sys.path.append('/home/lady5906/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append('/home/lady5906/workspace_python/conda_evomachine3.9/de-lta-rt')
sys.path.append('/home/lady5906/workspace_python/conda_evomachine3.9/asitiger')

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

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
        )
automaton.initialise()
automaton.process()

ipos = 0
iroi = 0
pos = automaton._pos_processor[ipos]
roi = pos.rois[iroi]
prev_seg = roi.get_seg(0)
curr_seg = roi.get_seg(1)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(prev_seg, cmap='gray', vmin=0, vmax=max(prev_seg.max(), curr_seg.max()))
axs[0].set_title('Seg Mask t-1')
axs[1].imshow(curr_seg, cmap='gray', vmin=0, vmax=max(prev_seg.max(), curr_seg.max()))
axs[1].set_title('Seg Mask t')
plt.show()
fig.savefig('segmentation_masks.png')

prev_cell_contours = utils.find_contours(prev_seg)

prev_drawn_0 = cv2.drawContours(
    np.zeros(DEFAULT_CONFIG_MOTHERMACHINE.target_size_track, dtype=np.float32),
    [prev_cell_contours[0]],
    0,
    offset=None,
    color=1.0,
    thickness=cv2.FILLED,
)

prev_drawn_1 = cv2.drawContours(
    np.zeros(DEFAULT_CONFIG_MOTHERMACHINE.target_size_track, dtype=np.float32),
    [prev_cell_contours[1]],
    0,
    offset=None,
    color=1.0,
    thickness=cv2.FILLED,
)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(prev_drawn_0, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[0].set_title('Contour Mask t-1 Cell 0')
axs[1].imshow(prev_drawn_1, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[1].set_title('Contour Mask t-1 Cell 1')
plt.show()
fig.savefig('contour.png')

area_0 = cv2.contourArea(prev_cell_contours[0])
area_1 = cv2.contourArea(prev_cell_contours[1])

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
#plt.show()
fig.savefig('ROIs_with_cropping.png')

enable = False
if enable:
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
    for iroi in range(len(pos.rois)):
        box = pos.rois[iroi].box
        patch = np.zeros((box.ybr-box.ytl, box.xbr - box.xtl))
        img_1_patched = box.patch(img_1_patched, patch)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_0, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
    axs[0].set_title('t-1')
    axs[1].imshow(img_1_patched, cmap='gray', vmin=0, vmax=max(img_0.max(), img_1.max()))
    axs[1].set_title('t')
    plt.show()
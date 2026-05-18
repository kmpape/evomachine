import copy

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest
from skimage.transform import hough_line, hough_line_peaks
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
from evomachine.acquisition_bkp import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import EVOMACHINE_DIR, ConfigDevice, ConfigImage
from evomachine.positionrt import TIMER_POSITION

this_cfg_device = ConfigDevice(
    num_pos=1,
    coord_pos=[(0, 0) for _ in range(1)],
    num_chan=1,
    num_periods=1,
    read_from_disk=True,
    path_to_images=EVOMACHINE_DIR.parent / "tests/data/device_ROI_test",
    image_processing_verbosity=0,
    tiger_port=None,
)
this_cam = DeltaCamera(cfg_device=this_cfg_device)
this_cam.initialise()
this_cfg_image = ConfigImage(
    pxl_horiz=3200,
    pxl_vert=2750,
    pxl_dtype=np.dtype("float32"),
    tile_image=(1, 1),
    crop_out_ROI=True,
)
automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=this_cfg_image,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=this_cam,
            strategy=NoStrategy(),
        )
#automaton.initialise()

ipos = 0
pos = automaton._pos_processor[ipos]
ref = this_cam.get_frame(0, 0)
ref = np.ones(ref.shape) - ref
ref = ref[:, :]

rotate_delta = delta.utils.deskew(ref[:, :], None)
rotate_evo = delta.utils.deskew(ref[:, :], (148, 255))
# DESKEW
reference = copy.deepcopy(ref)
reference_delta = delta.utils.imrotate(reference, rotate_delta)
reference_evo = delta.utils.imrotate(reference, rotate_evo)


fig, axs = plt.subplots(1, 3)
axs[0].imshow(ref, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('Reference')
axs[1].imshow(reference_delta, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('Reference rotated\nDelta')
axs[2].imshow(reference_evo, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('Reference rotated\nEvo')
plt.show()

# For debugging
self.reference = copy.deepcopy(reference)

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
#fig.savefig('ROIs_with_cropping.png')

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
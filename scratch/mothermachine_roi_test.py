import copy
import cv2
import glob
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
from evomachine.acquisition_bkp import TestCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCRISP, ConfigFocus, ConfigFocusAlgorithm, DEVICE_CONFIG_EVO_TEST, \
    CRISP_CONFIG_DEFAULT, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, \
    OBJECTIVE_CONFIG_AIR, ConfigLED, EVO_FORMATTER, OBJECTIVE_CONFIG_OIL, CRISP_CONFIG_OIL
from evomachine.positionrt import TIMER_POSITION

filenames = sorted(glob.glob("/mnt/ImageData/Scott/2023-12-08/*.tiff"))
filenames = [filenames[0]]
DEVICE_CONFIG_EVO_TEST.num_pos = 1
DEVICE_CONFIG_EVO_TEST.coord_pos = [(0, 0, 0)]
cropping_indices = ((800, 1000), (0, 3200))
IMAGE_CONFIG_DEFAULT.pxl_vert = 200
this_cam = TestCamera(
    cfg_device=DEVICE_CONFIG_EVO_TEST,
    cfg_objective=OBJECTIVE_CONFIG_AIR,
    cfg_image=IMAGE_CONFIG_DEFAULT,
    cfg_crisp=CRISP_CONFIG_DEFAULT,
    cfg_focus=FOCUS_CONFIG_DEFAULT,
    filenames=filenames,
    pos_to_filename=None,
    cropping_indices=cropping_indices,
)
this_cam.initialise()
DEFAULT_CONFIG_MOTHERMACHINE.rotation_correction = False
DEFAULT_CONFIG_MOTHERMACHINE.drift_correction = False
DEFAULT_CONFIG_MOTHERMACHINE.min_roi_area = 10
automaton = Automaton(
    cfg_device=DEVICE_CONFIG_EVO_TEST,
    cfg_image=IMAGE_CONFIG_DEFAULT,
    cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
    camera=this_cam,
    strategy=NoStrategy(),
)
automaton.initialise()

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

# This does not work currently due to fluorescence images having different color properties
if False:
    ipos = 0
    pos = automaton._pos_processor[ipos]
    ref = this_cam.get_frame(0, 0)
    ref = (ref - ref.min()) / ref.ptp()

    rotate_delta = delta.utils.deskew(ref[:, :], None)
    rotate_evo = delta.utils.deskew(ref[:, :], (148, 255))

    image8 = delta.utils.to_integer_values(ref, np.uint8)
    low_threshold = 148
    high_threshold = 255
    edges = cv2.Canny(image8, low_threshold, high_threshold, L2gradient=True)
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
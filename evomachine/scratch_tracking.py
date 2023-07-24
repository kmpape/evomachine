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
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
import evomachine.trackingrt as trackingrt
from evomachine.positionrt import TIMER_POSITION

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

this_cfg_device = DEVICE_CONFIG_DELTA_SIM
this_cfg_device.image_processing_verbosity = 0
this_cfg_image = IMAGE_CONFIG_DELTA_SIM
this_cfg_image.use_track_RT = True
this_cfg_image.crop_out_ROI = True
automaton_RT: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=this_cfg_image,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
automaton_RT.initialise()

this_cfg_image.use_track_RT = False
automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=this_cfg_image,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
automaton.initialise()

for i in range(1, 10):
    curr_pos = automaton.get_pos()
    print(f"Processing position {curr_pos} at time {i}")
    automaton_RT.process()
    automaton.process()
    is_eq = automaton_RT._pos_processor[curr_pos] == automaton._pos_processor[curr_pos]
    if not is_eq:
        print(f"Error for position {curr_pos} at time {i}")
    curr_pos = automaton.get_pos()
    print(f"Processing position {curr_pos} at time {i}")
    automaton_RT.process()
    automaton.process()
    is_eq = automaton_RT._pos_processor[curr_pos] == automaton._pos_processor[curr_pos]
    if not is_eq:
        print(f"Error for position {curr_pos} at time {i}")
    x = 0


ipos = 0
pos_RT_init = copy.deepcopy(automaton._pos_processor[ipos])
automaton.process()
automaton.process()
pos_RT_1 = copy.deepcopy(automaton._pos_processor[ipos])

this_cfg_image.use_track_RT = False
automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=this_cfg_image,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
automaton.initialise()
pos_init = copy.deepcopy(automaton._pos_processor[ipos])
automaton.process()
automaton.process()
pos_1 = copy.deepcopy(automaton._pos_processor[ipos])

pos_1.rois[0].lineage


ipos = 0
iroi = 0
pos = automaton._pos_processor[ipos]
roi = pos.rois[iroi]

pos1 = automaton._pos_processor[1]
roi1 = pos1.rois[0]
fig, axs = plt.subplots(1, 1)
axs.imshow(roi1.img_stack[0], cmap='gray', vmin=0, vmax=roi1.img_stack[0].max())

print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions.shape)
print(automaton._pos_processor[ipos].rois[iroi].tmp_previous_cell_nbs)
print(np.unique(automaton._pos_processor[ipos].rois[iroi].tmp_labels))
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions)

automaton.process()
automaton.process()
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions.shape)
print(automaton._pos_processor[ipos].rois[iroi].tmp_previous_cell_nbs)
print(np.unique(automaton._pos_processor[ipos].rois[iroi].tmp_labels))
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(roi.img_stack[0], cmap='gray', vmin=0, vmax=roi.img_stack[0].max())
axs[0].set_title('ROI 0 at t=0')
axs[1].imshow(roi.img_stack[1], cmap='gray', vmin=0, vmax=roi.img_stack[1].max())
axs[1].set_title('ROI 0 at t=1')
plt.show()

automaton.process()
automaton.process()
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions.shape)
print(automaton._pos_processor[ipos].rois[iroi].tmp_previous_cell_nbs)
print(np.unique(automaton._pos_processor[ipos].rois[iroi].tmp_labels))
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions)

automaton.process()
automaton.process()
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions.shape)
print(automaton._pos_processor[ipos].rois[iroi].tmp_previous_cell_nbs)
print(np.unique(automaton._pos_processor[ipos].rois[iroi].tmp_labels))
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions)
automaton.process()
automaton.process()
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions.shape)
print(automaton._pos_processor[ipos].rois[iroi].tmp_previous_cell_nbs)
print(np.unique(automaton._pos_processor[ipos].rois[iroi].tmp_labels))
print(automaton._pos_processor[ipos].rois[iroi].tmp_attributions)

prev_seg = roi.get_seg(0)
curr_seg = roi.get_seg(1)
prev_cell_contours = utils.find_contours(prev_seg)


x_old = [{"y": 1.0, "div": False, "area": 10.0, "id": 1},
         {"y": 2.0, "div": False, "area": 10.0, "id": 2}]
u_new = [{"y": 2.0, "area": 11.0}, {"y": 3.0, "area": 12.0}]
max_id = 2

# Plot segmentation output
fig, axs = plt.subplots(1, 2)
axs[0].imshow(prev_seg, cmap='gray', vmin=0, vmax=max(prev_seg.max(), curr_seg.max()))
axs[0].set_title('Seg Mask t-1')
axs[1].imshow(curr_seg, cmap='gray', vmin=0, vmax=max(prev_seg.max(), curr_seg.max()))
axs[1].set_title('Seg Mask t')
plt.show()
fig.savefig('segmentation_masks.png')

prev_drawn_0 = cv2.drawContours(np.zeros(DEFAULT_CONFIG_MOTHERMACHINE.target_size_track, dtype=np.float32),
    [prev_cell_contours[0]], 0, offset=None, color=1.0, thickness=cv2.FILLED,
)

prev_drawn_1 = cv2.drawContours(np.zeros(DEFAULT_CONFIG_MOTHERMACHINE.target_size_track, dtype=np.float32),
    [prev_cell_contours[1]], 0, offset=None, color=1.0, thickness=cv2.FILLED,
)

# Plot contours and compute areas
area_0 = cv2.contourArea(prev_cell_contours[0])
area_1 = cv2.contourArea(prev_cell_contours[1])
fig, axs = plt.subplots(1, 2)
axs[0].imshow(prev_drawn_0, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[0].set_title(f"Contour Mask t-1 Cell 0\n Area={area_0}")
axs[1].imshow(prev_drawn_1, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[1].set_title(f"Contour Mask t-1 Cell 1\n Area={area_1}")
#plt.show()
fig.savefig('contour.png')

fig, axs = plt.subplots(1, 2)
axs[0].imshow(prev_drawn_0, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[0].set_title(f"Contour Mask t-1 Cell 0\n Area={area_0}")
axs[1].imshow(prev_drawn_1, cmap='gray', vmin=0, vmax=max(prev_drawn_0.max(), prev_drawn_1.max()))
axs[1].set_title(f"Contour Mask t-1 Cell 1\n Area={area_1}")




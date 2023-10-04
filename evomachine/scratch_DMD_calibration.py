import sys
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
from asitiger.command import CRISPState, Command
from evomachine.acquisition import EvoCamera, DMDControl
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT
from evomachine.utils import Timer
import pygame
import sys
import os
from pygame.locals import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

i_chan = 2
cam = EvoCamera(DEVICE_CONFIG_EVO_TEST)
dmd = DMDControl()
tig = cam.tiger
cam.initialise()
cam._set_channel(2)
dmd.display_full()


def get_full(this_delay: float):
    dmd.display_none(update_display=False)
    dmd.display_full()
    time.sleep(this_delay)
    return cam.get_frame(i_chan=i_chan)


def get_none(this_delay: float):
    dmd.display_none(update_display=False)
    dmd.display_none()
    time.sleep(this_delay)
    return cam.get_frame(i_chan=i_chan)


def get_horiz(this_at_pos: int, this_delay: float, this_line_width: int):
    dmd.display_none(update_display=False)
    dmd.display_line_horiz(at_pos=this_at_pos, line_width=this_line_width)
    time.sleep(this_delay)
    return cam.get_frame(i_chan=i_chan)


def get_vert(this_at_pos: int, this_delay: float, this_line_width: int):
    dmd.display_none(update_display=False)
    dmd.display_line_vert(at_pos=this_at_pos, line_width=this_line_width)
    time.sleep(this_delay)
    return cam.get_frame(i_chan=i_chan)


line_width = 11
step = 1
delay = 1
ref_full = get_full(this_delay=delay)
ref_none = get_none(this_delay=delay)

horiz_1 = get_horiz(this_at_pos=500, this_delay=delay, this_line_width=line_width)
horiz_2 = get_horiz(this_at_pos=1000, this_delay=delay, this_line_width=line_width)
diff_none = np.abs(ref_none.astype(float)-horiz_1.astype(float))
diff_full = np.abs(ref_full.astype(float)-horiz_1.astype(float))

vmin = ref_full.min()
vmax = 0.6*ref_full.max()
fig, axs = plt.subplots(2, 4)
axs[0, 0].imshow(ref_full, vmin=vmin, vmax=vmax)
axs[0, 0].set_title("FULL")
axs[0, 1].imshow(ref_none, vmin=vmin, vmax=vmax)
axs[0, 1].set_title("NONE")
axs[0, 2].imshow(horiz_1, vmin=vmin, vmax=vmax)
axs[0, 2].set_title(f"LINE_HORIZ at pxl 500")
axs[0, 3].imshow(horiz_2, vmin=vmin, vmax=vmax)
axs[0, 3].set_title("LINE_HORIZ at pxl 1000")
axs[1, 2].plot(horiz_1.max(axis=0))
axs[1, 2].set_title(f"Column-wise max")
axs[1, 3].plot(horiz_2.max(axis=0))
axs[1, 3].set_title(f"Column-wise max")
plt.show()

ref_full = get_full(this_delay=delay)
ref_none = get_none(this_delay=delay)
horiz_range = range(0, dmd.width_height_DMD[1], step)
# columns: pos_dmd, column-wise-max(horiz_img), argmax, same w. diff_none
data = np.empty((len(horiz_range), 3200))
results = np.empty((len(horiz_range), 7))
for i, at_pos in enumerate(horiz_range):
    horiz_img = get_horiz(this_at_pos=at_pos, this_delay=delay, this_line_width=line_width)
    horiz_max = horiz_img.max(axis=0)
    data[i, :] = horiz_max
    diff_none = np.abs(ref_none.astype(float) - horiz_img.astype(float))
    diff_max = diff_none.max(axis=0)
    results[i, 0] = at_pos
    results[i, 1] = horiz_max.max()
    results[i, 2] = horiz_max.argmax()
    results[i, 3] = diff_max.max()
    results[i, 4] = diff_max.argmax()
    results[i, 5] = horiz_img.min()
    results[i, 6] = horiz_img.max()


if len(horiz_range) <= 15:
    fig, axs = plt.subplots(len(horiz_range), 1)
    for i, at_pos in enumerate(horiz_range):
        axs[i].plot(data[i, :])
        axs[i].set_title(f"dmd pos = {at_pos}")

fig, axs = plt.subplots(1, 1)
axs.plot(results[:, 0], results[:, 2])
axs.set_title('Horizontal map DMD-CAM')
axs.set_xlabel('DMD loc')
axs.set_ylabel('CAM loc')
plt.show()


with open('results_calib_horiz.pkl', 'wb') as file:
    pickle.dump(results, file)


vert_1 = get_vert(this_at_pos=1000, this_delay=delay, this_line_width=line_width)
vert_2 = get_vert(this_at_pos=1500, this_delay=delay, this_line_width=line_width)
ref_full = get_full(this_delay=delay)
ref_none = get_none(this_delay=delay)
diff_none = np.abs(ref_none.astype(float)-vert_1.astype(float))
diff_full = np.abs(ref_full.astype(float)-vert_1.astype(float))


vmin = ref_full.min()
vmax = 0.6*ref_full.max()
fig, axs = plt.subplots(2, 4)
axs[0, 0].imshow(ref_full, vmin=vmin, vmax=vmax)
axs[0, 0].set_title("FULL")
axs[0, 1].imshow(ref_none, vmin=vmin, vmax=vmax)
axs[0, 1].set_title("NONE")
axs[0, 2].imshow(vert_1, vmin=vmin, vmax=vmax)
axs[0, 2].set_title(f"LINE_VERT at pxl 1000")
axs[0, 3].imshow(vert_2, vmin=vmin, vmax=vmax)
axs[0, 3].set_title("LINE_VERT at pxl 1500")
axs[1, 2].plot(vert_1.max(axis=1))
axs[1, 2].set_title(f"Column-wise max")
axs[1, 3].plot(vert_2.max(axis=1))
axs[1, 3].set_title(f"Column-wise max")
plt.show()

step = 1
ref_full = get_full(this_delay=delay)
ref_none = get_none(this_delay=delay)
vert_range = range(0, dmd.width_height_DMD[0], step)
# columns: pos_dmd, column-wise-max(horiz_img), argmax, same w. diff_none
data_vert = np.empty((len(vert_range), 3200))
results_vert = np.empty((len(vert_range), 7))
for i, at_pos in enumerate(vert_range):
    vert_img = get_vert(this_at_pos=at_pos, this_delay=delay, this_line_width=line_width)
    vert_max = vert_img.max(axis=1)
    data_vert[i, :] = vert_max
    diff_none = np.abs(ref_none.astype(float) - vert_img.astype(float))
    diff_max = diff_none.max(axis=1)
    results_vert[i, 0] = at_pos
    results_vert[i, 1] = vert_max.max()
    results_vert[i, 2] = vert_max.argmax()
    results_vert[i, 3] = diff_max.max()
    results_vert[i, 4] = diff_max.argmax()
    results_vert[i, 5] = vert_img.min()
    results_vert[i, 6] = vert_img.max()


with open('results_calib_vert.pkl', 'wb') as file:
    pickle.dump(results_vert, file)

fig, axs = plt.subplots(1, 1)
axs.plot(results_vert[:, 0], results_vert[:, 2])
axs.set_title('Vertical map DMD-CAM')
axs.set_xlabel('DMD loc')
axs.set_ylabel('CAM loc')
plt.show()


# Benchmark functions
# i_trials = 100
# mytimer = Timer(timer_level=0, name="bench_dmd", enabled=True)
# for i in range(i_trials):
#     mytimer.start("surface", 0)
#     dmd.display_full()
#     dmd.display_none()
#     mytimer.stop("surface", 0)
#     mytimer.start("array", 0)
#     dmd.display_none_depreciated()
#     dmd.display_full_depreciated()
#     mytimer.stop("array", 0)
#     mytimer.start("rect", 0)
#     dmd.display_full_depreciated2()
#     dmd.display_none_depreciated2()
#     mytimer.stop("rect", 0)
#
# mytimer.display_timings()

# Timings bench_dmd (timer_level 0):
#
#       name  n_calls       avg    median       min       max
# 0  surface      100  0.010367  0.009648  0.009565  0.028761
# 1    array      100  0.054365  0.054289  0.053789  0.057373
# 2     rect      100  0.010159  0.009699  0.009607  0.022191

# 1. Take reference images
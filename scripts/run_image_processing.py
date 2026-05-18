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

from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE_ROOT / "asitiger"))
sys.path.append(str(WORKSPACE_ROOT / "evomachine_repo"))
sys.path.append(str(WORKSPACE_ROOT / "de-lta-rt"))

from evomachine.acquisition_bkp import TestCamera, EvoCamera, EvoCamerav2
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
    import pygame

from evomachine.coordinates import Coordinate
from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import BasicStrategy
from evomachine.utils import rotation_correction
from delta.roirt import *
from delta.rt import *


def get_position(filename):
    parts = filename.split('_')
    position_str = parts[1][1:]
    return int(position_str)


def get_time(filename):
    parts = filename.split('_')
    time_str = parts[-2] + '_' + parts[-1].rstrip('.tiff')
    return datetime.datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S.%f')

def my_set_xticks(ax, scol, ecol, srow, erow, step=100):
    ax.set_xticks(range(0, ecol - scol, step))
    ax.set_xticklabels(range(scol, ecol, step))
    ax.set_yticks(range(0, erow - srow, step))
    ax.set_yticklabels(range(srow, erow, step))
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)


process_queue: Queue = Queue()
gui_to_automaton_queue: Queue = Queue()
automaton_to_gui_queue: Queue = Queue()
shutdown_event: Event = Event()
start_strategy_event: Event = Event()
stop_strategy_event: Event = Event()
stop_event: Event = Event()
request_lock: Lock = Lock()

camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()
processor_config.cfg_delta.whole_frame_drift = True

folder_path = str(EVOMACHINE_DIR.parent / "data")
filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith('.tiff')]
filenames = sorted(filenames, key=lambda x: (get_position(x), get_time(x)))
pos_to_filename = {get_position(filename): len(filenames)-1-index for index, filename in enumerate(filenames[::-1])}

cam = TestCamera(
    cfg_camera=camera_config,
    filenames=[folder_path + '/' + f for f in filenames],
    pos_to_filename=pos_to_filename
)

cam.move(0)
frame_0 = cam.get_frame(i_chan=None)
cam.move(1)
frame_1 = cam.get_frame(i_chan=None)
frames = [frame_0, frame_1]

if True:
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    ax[0].imshow(frame_0, cmap='gray')
    ax[0].set_title("0")
    ax[1].imshow(frame_1, cmap='gray')
    ax[1].set_title("1")
    # plt.show()

dmd = DMDControl(debug_mode=True)

strategy = BasicStrategy(save_path=str(camera_config.path_to_save))

use_seg = True
automaton = Automaton(
    camera=cam,
    cfg_processor=processor_config,
    dmd=dmd,
    strategy=strategy,
    start_strategy_event=start_strategy_event,
    stop_strategy_event=stop_strategy_event,
    stop_event=stop_event,
    shutdown_event=shutdown_event,
    process_q=process_queue,
    gui_to_automaton_q=gui_to_automaton_queue,
    automaton_to_gui_q=automaton_to_gui_queue,
    use_seg=use_seg,
    run_timeout=0,
)
automaton.initialise_devices()

use_autofocus = True
field_of_views = {0: Coordinate(0, 0, 0), 1: Coordinate(0, 0, 0)}
automaton.initialise_field_of_view_list(
    field_of_views=field_of_views,
    cropping_boxes=None,
    use_autofocus=use_autofocus,
)
automaton.initialise_fov_focus(use_autofocus=use_autofocus)
automaton.initialise_reference_frames()
rotations = [rotation_correction(img) for img in frames]
automaton.initialise_position_processor()

proc = automaton._pos_processor[0]
roi_boxes, cols_s_e = find_roi_boxes_rt(
    img=proc.reference[proc.channel_roi],
    col_s_e_override=(None, None, None, None),
)
img = prep_img(proc.reference[proc.channel_roi])
fig, axs = plt.subplots(1, 1, figsize=(30, 30))
axs.imshow(img, cmap='gray')
axs.grid(visible=True, color='blue')
my_set_xticks(axs, 0, 3200, 0, 3200, step=100)
for i, box in enumerate(roi_boxes):
    width = box.xbr - box.xtl
    height = box.ybr - box.ytl
    rect = Rectangle((box.xtl, box.ytl), width, height, edgecolor='red', facecolor='red', alpha=0.1)
    _ = axs.add_patch(rect)
    _ = axs.text((box.xbr+box.xtl)*0.5, (box.ybr+box.ytl)*0.5, str(i), color='blue', fontsize=6)

plt.tight_layout()


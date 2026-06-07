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

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
from evomachine.acquisition_bkp import TestCamera, EvoCamera, EvoCamerav2
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd_pygame import DMDControl
    import pygame

from evomachine.types import FilterWheelType, LEDType
from evomachine.config_types import DMDCalibConfigTypeFactory
from evomachine.coordinates import Coordinate
from evomachine.strategy import BasicStrategy
from evomachine.utils import rotation_correction
from delta.roirt import *
from delta.rt import *


camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()
processor_config.cfg_delta.whole_frame_drift = True

cfg = DMDCalibConfigTypeFactory.default()
cfg.brightness = 0.4
cfg.exposure = 50
cfg.line_width = 1
cfg.channel = LEDType.LED_450_NM

cfg.start_col = 0
cfg.end_col = 1599
cfg.start_row = 500
cfg.end_row = 2200
cfg.step = 100

cam = EvoCamerav2(cfg_camera=camera_config)
cam.initialise()
cam.set_led(i_chan=cfg.channel, brightness=cfg.brightness)
cam.set_filter_wheel(FilterWheelType.NO_FILTER)
cam.set_exposure(exposure_time=cfg.exposure)

dmd = DMDControl()
dmd.initialise()
dmd.display_calibration_image()

save_path: str = str(IMAGE_DIR / "DEFAULT")
cam.studio.live().set_live_mode(False)
assert cfg.end_col < dmd.width_height_DMD[0]
assert cfg.end_col < dmd.width_height_DMD[1]
col_range = np.arange(cfg.start_col, cfg.end_col+cfg.step, cfg.step)
if col_range[-1] == dmd.width_height_DMD[1]:
    col_range[-1] -= 1

row_range = np.arange(cfg.start_row, cfg.end_row+cfg.step, cfg.step)
if row_range[-1] == dmd.width_height_DMD[0]:
    row_range[-1] -= 1

cols, rows = np.meshgrid(col_range, row_range)


# Get an intensity level for a point that is in the FoV
max_intensity = 0
for i_row in range(3):
    for i_col in range(3):
        row = (dmd.width_height_DMD[0]*(i_row+1)) // 4
        col = (dmd.width_height_DMD[1]*(i_col+1)) // 4
        dmd.display_circle(row=row, col=col, radius=cfg.line_width)
        time.sleep(cfg.delay)
        test_img = cam.get_frame(i_chan=None, normalise=False)
        max_intensity += test_img.max()
        print(f"Init image ({row}, {col}): {test_img.max()}")

max_intensity = float(max_intensity) / 9
print(f"Average max intensity on-screen: {max_intensity}")

dmd.display_circle(
    row=0,
    col=0,
    radius=cfg.line_width,
)
time.sleep(cfg.delay)
test_img_none = cam.get_frame(
    i_chan=None,
    normalise=False
)
max_intensity_none = test_img_none.max()
print(f"Max intensity off-screen: {max_intensity_none}")

min_intensity = max_intensity_none + 0.5 * (max_intensity - max_intensity_none)

results = []
cam.set_led(i_chan=cfg.channel, brightness=cfg.brightness)
for i, (col, row) in enumerate(zip(cols.flatten(), rows.flatten())):
    if not USE_DMD_SOCKET:
        dmd.display_none(update_display=False)
    dmd.display_circle(row=row, col=col, radius=cfg.line_width)
    time.sleep(cfg.delay)
    img = cam.get_frame(
        i_chan=None,
        normalise=False
    )
    if img.max() >= min_intensity:
        print(f"At {i + 1} of {len(cols.flatten())}")
        img_col_max = img.max(axis=0)
        img_row_max = img.max(axis=1)
        results.append(((row, col), (img_row_max.argmax(), img_col_max.argmax()), (img_row_max.max(), img_col_max.max())))
    else:
        print(f"At {i + 1} of {len(cols.flatten())}: DMD point row={row} and col={col} outside of image with {img.max()} > {min_intensity}.")


cam.disable_led()
dmd.display_none()

dmd_calibration_data = results

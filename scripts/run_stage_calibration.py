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

cfg_cam = ConfigCameraFactory.default_air_config()

L = 101.6   # mm
screw_pitch = 0.5  # mm

n_meas = 6
# x = np.array(range(n_meas)) * cfg_cam.fov_size  # um
# y = np.array([438.1, 433.7, 429.7, 425.5, 421.3, 417.0])  # um
x = np.array([6302, -14498, -35298, -71698, -102898]) / 10
y = np.array([-1829.5, -1638.9, -1438.9, -1138.8, -936]) / 10
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
angle = np.arctan(m)
angle_deg = np.rad2deg(angle)
print(f"With {n_meas} measurements: m={m} and angle={angle_deg} degrees.")
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()

delta_y = - m * L  # mm
rot_deg = delta_y / screw_pitch * 360
print(f"Requires moving stage by {delta_y} mm or rotating screws by {rot_deg} (= {rot_deg/90} x 90 deg).")

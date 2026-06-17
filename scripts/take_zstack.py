import time

import sys, os
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE_ROOT / "asitiger"))
sys.path.append(str(WORKSPACE_ROOT / "evomachine_repo"))
sys.path.append(str(WORKSPACE_ROOT / "de-lta-rt"))

from asitiger.command import CRISPSetState, Command
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
from evomachine.acquisition_bkp import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT, OBJECTIVE_CONFIG_OIL, \
    OBJECTIVE_CONFIG_AIR, IMAGE_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigLED, ConfigCRISP, EVOMACHINE_DIR
from evomachine.coordinates import Coordinate
from evomachine.dmd_pygame import DMDControl, DMDColor
from evomachine.software_focus_bkp import get_focus_score_steel

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import skimage

test_pos_list = [(-10000, 0, 0), (0, 0, 0), (0, 10000, 0)]
DEVICE_CONFIG_MOTHERMACHINE = ConfigDevice(
    num_pos=len(test_pos_list),
    coord_pos=test_pos_list,
    num_chan=4,
    num_periods=None,
    read_from_disk=False,
    path_to_images=None,
    path_to_save=IMAGE_DIR / "2023-11-17-MM",
    image_processing_verbosity=1,
    tiger_port="/dev/ttyUSB0",
)

FOCUS_CONFIG_MOTHERMACHINE = ConfigFocus(
    exposure_time=1000,
    focus_channel=2,
    rel_range=100,
    steps_size=1,
)

CRISP_CONFIG_MOTHERMACHINE = ConfigCRISP(
    led_intensity=80,
    objective_na=0.95,
    loop_gain=5,
    averaging=0,
    update_rate=100,
    lock_range=0.05,
)

cam = EvoCamera(cfg_device=DEVICE_CONFIG_MOTHERMACHINE, cfg_objective=OBJECTIVE_CONFIG_AIR,
                cfg_focus=FOCUS_CONFIG_MOTHERMACHINE, cfg_crisp=CRISP_CONFIG_MOTHERMACHINE)
dmd = DMDControl()
cam.initialise()
cam.disable_led()

dmd.display_full()

# Settings
load_images = False
shape = cam.cfg_image.pxl_vert
path_to_save = Path("/mnt/ImageData/Idris/2024-02-17-zstack/pos1")
rel_range = 100  # in 1/10 microns
step_size = 10
exposure_time = 1000  # in ms
focus_channel = ConfigLED.LED_450_NM
row_min, row_max, col_min, col_max = 0, cam.cfg_image.pxl_vert, 0, cam.cfg_image.pxl_horiz
cam.set_led(i_chan=focus_channel.value)

# Camera settings
cam.set_exposure(exposure_time=int(exposure_time))
cam.studio.live().set_live_mode(False)

time.sleep(1)

# Get current position and reset stage limits
if load_images:
    filenames = [file.name for file in path_to_save.glob("z_*.tif") if "prev" not in file.name]
    prev_filename = [file.name for file in path_to_save.glob("z_*.tif") if "prev" in file.name]
    if prev_filename:
        prev_filename = prev_filename[0]
    else:
        prev_filename = None
    coords = [int(file.stem.split("_")[1]) for file in path_to_save.glob("z_*.tif")]
    tmp = list(zip(coords, filenames))
    tmp.sort(key=lambda x: x[0])
    coords = [item[0] for item in tmp]
    filenames = [item[1] for item in tmp]
    num_coords = len(filenames)
else:
    filenames = []
    curr_pos = cam.tiger.where(['Z'])['Z']
    stage_limits = cam.tiger.get_stage_limits()
    stage_limits['Z'] = (curr_pos-2*rel_range, curr_pos+2*rel_range)
    cam.tiger.set_stage_limits(stage_limits=stage_limits)
    coords = range(curr_pos - rel_range, curr_pos + rel_range, step_size)
    num_coords = len(coords)


def get_laplacian_var_focus_score(img) -> float:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return (lap[1:-1, 1:-1]**2).var()

def laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.int64(lap)
    return (lap[1:-1, 1:-1]**2).mean()


def sq_grad(img, thres=0.1):
    p = np.int64(img)
    tmp = abs(p[:, 1:] - p[:, :-1])
    tmp[tmp < thres] = 0
    return (tmp**2).mean()


def sq_grad_float(img, thres=0.1):
    tmp = abs(img[:, 1:] - img[:, :-1])
    tmp[tmp < thres] = 0
    return (tmp**2).mean()


focus_algs = [get_laplacian_var_focus_score, laplacian, sq_grad, sq_grad_float, get_focus_score_steel]
focus_algs_str = ["Lvar", "Lavg", "SQavg", "SQavgfloat", "Steel"]

num_algs = len(focus_algs)
curr_z = cam.tiger.where(['Z'])['Z']
if load_images:
    if prev_filename:
        prev_image = skimage.io.imread(path_to_save / prev_filename)
    else:
        prev_image = None
else:
    prev_image = cam.display_save_frame(
        i_chan=focus_channel.value,
        path_to_save=path_to_save,
        filename=f"z_{curr_z:04d}_{str(focus_channel)}_prev_.tif",
        display_frame=False,
        block=True,
    )
images = np.zeros((cam.cfg_image.pxl_vert, cam.cfg_image.pxl_horiz, num_coords))
images_norm = np.zeros((cam.cfg_image.pxl_vert, cam.cfg_image.pxl_horiz, num_coords))
focus_scores = np.zeros((num_coords, num_algs))
focus_scores_norm = np.zeros((num_coords, num_algs))
for ipos, z_coord in enumerate(coords):
    if load_images:
        images[:, :, ipos] = skimage.io.imread(path_to_save / filenames[ipos])
    else:
        cam.move(Coordinate(None, None, z_coord), block=True)
        time.sleep(0.1)
        # Save images without displaying
        images[:, :, ipos] = cam.display_save_frame(
            i_chan=focus_channel.value,
            path_to_save=path_to_save,
            filename=f"z_{z_coord:04d}_{str(focus_channel)}_.tif",
            display_frame=False,
            block=True,
        )
    images_norm[:, :, ipos] = cam.normalise_frame(frame=images[:, :, ipos], colormap=None)
    for ialg, alg in enumerate(focus_algs):
        focus_scores[ipos, ialg] = alg(images[row_min:row_max, col_min:col_max, ipos])

cam.move(Coordinate(None, None, curr_z), block=True)
cam.disable_led()

best_focus_positions = np.argmax(focus_scores, axis=0)

fig, axs = plt.subplots(num_algs, 1)
for ialg in range(num_algs):
    axs[ialg].plot(focus_scores[:, ialg])
    axs[ialg].set_title(focus_algs_str[ialg])

fig, axs = plt.subplots(1, num_algs+1)
if prev_image is not None:
    _ = axs[0].imshow(prev_image, vmin=prev_image.min(), vmax=prev_image.max())
_ = axs[0].set_title("Previous")
for ialg in range(num_algs):
    b = best_focus_positions[ialg]
    _ = axs[ialg+1].imshow(images[:, :, b], vmin=images[:, :, b].min(), vmax=images[:, :, b].max())
    _ = axs[ialg+1].set_title(focus_algs_str[ialg])


if False:
    irows = int(np.sqrt(num_coords))
    icols = int(np.ceil(num_coords/irows))
    # vmin = images[:, :, 0].min()
    # vmax = 0.8*images[:, :, 0].max()
    fig, axs = plt.subplots(irows, icols)
    for ipos, z_coord in enumerate(coords):
        _ = axs[int(ipos/icols), np.mod(ipos, icols)].imshow(images[:, :, ipos], vmin=images[:, :, ipos].min(), vmax=images[:, :, ipos].max())
        if ipos in best_focus_positions:
            msg = ",".join([focus_algs_str[i] for i in range(num_algs) if ipos == best_focus_positions[i]])
            _ = axs[int(ipos/icols), np.mod(ipos, icols)].set_title(f"{z_coord} [{ipos}] ({msg})")
        else:
            _ = axs[int(ipos/icols), np.mod(ipos, icols)].set_title(f"{z_coord} [{ipos}]")

    plt.tight_layout()

    indmin, indmax = 2000, 3000
    fig, axs = plt.subplots(irows, icols)
    for ipos, z_coord in enumerate(coords):
        _ = axs[int(ipos/icols), np.mod(ipos, icols)].imshow(images[indmin:indmax, indmin:indmax, ipos], vmin=images[:, :, ipos].min(), vmax=images[:, :, ipos].max())
        if ipos in best_focus_positions:
            msg = ",".join([focus_algs_str[i] for i in range(num_algs) if ipos == best_focus_positions[i]])
            _ = axs[int(ipos/icols), np.mod(ipos, icols)].set_title(f"{z_coord} [{ipos}] ({msg})")
        else:
            _ = axs[int(ipos/icols), np.mod(ipos, icols)].set_title(f"{z_coord} [{ipos}]")

plt.show()

import time

import sys, os
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE_ROOT / "asitiger"))
sys.path.append(str(WORKSPACE_ROOT / "evomachine_repo"))
sys.path.append(str(WORKSPACE_ROOT / "de-lta-rt"))

from asitiger.command import CRISPState, Command
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
from evomachine.acquisition_bkp import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT, OBJECTIVE_CONFIG_OIL, \
    OBJECTIVE_CONFIG_AIR, IMAGE_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigLED, ConfigCRISP
from evomachine.dmd_pygame import DMDControl
from evomachine.types import FovDirectionType
from pathlib import Path

test_pos_list = [(-10000, 0, 0), (0, 0, 0), (0, 10000, 0)]
DEVICE_CONFIG_RASTONIA = ConfigDevice(
    num_pos=len(test_pos_list),
    coord_pos=test_pos_list,
    num_chan=4,
    num_periods=None,
    read_from_disk=False,
    path_to_images=None,
    path_to_save=IMAGE_DIR / "2023-10-12-rastonia-bacillus-405nm",
    image_processing_verbosity=1,
    tiger_port="/dev/ttyUSB0",
)

FOCUS_CONFIG_RASTONIA = ConfigFocus(
    exposure_time=0.01,
    focus_channel=2,
    rel_range=100,
    steps_size=1,
)

CRISP_CONFIG_RASTONIA = ConfigCRISP(
    led_intensity=80,
    objective_na=1.4,
    loop_gain=5,
    averaging=0,
    update_rate=100,
    lock_range=0.05,
)


i_chan = 2
cam = EvoCamera(cfg_device=DEVICE_CONFIG_RASTONIA, cfg_objective=OBJECTIVE_CONFIG_OIL,
                cfg_focus=FOCUS_CONFIG_RASTONIA, cfg_crisp=CRISP_CONFIG_RASTONIA)
dmd = DMDControl()
tig = cam.tiger
cam.initialise()
cam.set_led(-1)
dmd.display_none()

# _ = cam.display_save_frame(i_chan=2, path_to_save=False, display_frame=True)
move_vert = range(5)
move_horiz = range(5)
img_channel = 2

for i_vert in move_vert:
    for i_horiz in move_horiz:
        print(f"At i_vert={i_vert}/{len(move_vert)}, i_vert={i_horiz}/{len(move_horiz)}", end='\r')
        _ = cam.display_save_frame(i_chan=img_channel, path_to_save=True, display_frame=False)
        if i_horiz % 2 == 0:
            cam.move([(FovDirectionType.RIGHT, 1.0)])
        else:
            cam.move([(FovDirectionType.LEFT, 1.0)])
        time.sleep(1)
    cam.move([(FovDirectionType.UP, 1.0)])
    time.sleep(1)

import time

import sys, os
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')

from asitiger.command import CRISPState, Command
from evomachine.acquisition import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT, OBJECTIVE_CONFIG_OIL, \
    OBJECTIVE_CONFIG_AIR, IMAGE_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigLED, ConfigCRISP
from evomachine.dmd import DMDControl, DMDColor
from pathlib import Path

test_pos_list = [(-10000, 0, 0), (0, 0, 0), (0, 10000, 0)]
DEVICE_CONFIG_MOTHERMACHINE = ConfigDevice(
    num_pos=len(test_pos_list),
    coord_pos=test_pos_list,
    num_chan=4,
    num_periods=None,
    read_from_disk=False,
    path_to_images=None,
    path_to_save=Path("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/images/"
                      "2023-11-17-MM"),
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
tig = cam.tiger
cam.initialise()
cam.set_led(-1)
dmd.display_none()

tig.move()

# _ = cam.display_save_frame(i_chan=2, path_to_save=False, display_frame=True)
move_vert = range(5)
move_horiz = range(5)
img_channel = 2

# for i_vert in move_vert:
#     for i_horiz in move_horiz:
#         print(f"At i_vert={i_vert}/{len(move_vert)}, i_vert={i_horiz}/{len(move_horiz)}", end='\r')
#         _ = cam.display_save_frame(i_chan=img_channel, path_to_save=True, display_frame=False)
#         if i_horiz % 2 == 0:
#             cam.move_fov_right(block=True)
#         else:
#             cam.move_fov_left(block=True)
#     cam.move_fov_up(block=True)


img_channel = 1
move_horiz = range(20)
for i_horiz in move_horiz:
    print(f"At i_horiz={i_horiz}/{len(move_horiz)}", end='\r')
    _ = cam.display_save_frame(i_chan=img_channel, path_to_save=True, display_frame=False)
    cam.move_fov_right(block=True)

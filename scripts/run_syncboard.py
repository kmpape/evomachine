import numpy as np
import matplotlib.pyplot as plt
import sys
import time

sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/de-lta-rt")

from evomachine.types import LEDType  # noqa
from syncboard.syncboardcontroller import SyncBoardController  # noqa
from syncboard.command import Command
from evomachine.dmd_socket import DMDControl  # noqa

led_channel_keys: dict[LEDType, int | None] = {
    LEDType.LED_385_NM: 7,
    LEDType.LED_450_NM: 1,
    LEDType.LED_515_NM: 2,
    LEDType.LED_565_NM: 3,
    LEDType.LED_645_NM: 4,
    LEDType.NO_LED: None,
}

ctr = SyncBoardController.from_serial_port(port='/dev/ttyACM1')
ctr.initialise()

dmd = DMDControl()
dmd.initialise()
dmd.display_full()

ctr.finalise()
dmd.finalise()

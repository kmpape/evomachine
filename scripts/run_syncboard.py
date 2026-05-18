import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE_ROOT / "asitiger"))
sys.path.append(str(WORKSPACE_ROOT / "evomachine_repo"))
sys.path.append(str(WORKSPACE_ROOT / "de-lta-rt"))

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

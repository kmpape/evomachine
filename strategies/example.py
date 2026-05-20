from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import threading
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger, ConfigCameraFactory, ConfigImageProcessorFactory, USE_DMD_SOCKET, \
    ConfigImageProcessor
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import EvoMachineError
from evomachine.evotypes import LEDType
from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class ExampleStrategy(AbstractStrategy):
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)

    def _initialise(self) -> List[AutomatonCommand]:
        """
        Initialise the strategy.
        """

    def _callback(
            self,
            fov_id: int,
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        """
        Callback function when new data is available.
        """
        return []

    def finalise(self) -> List[AutomatonCommand]:
        """
        Save everything else than images here.
        """
        return []







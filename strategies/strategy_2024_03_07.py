from abc import ABC, abstractmethod
from datetime import datetime
import inspect
import numpy as np
from pathlib import Path
import pickle
import sys
import threading
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import EVOMACHINE_DIR, get_logger, ConfigCameraFactory, ConfigImageProcessorFactory
from evomachine.coordinates import Coordinate
from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import ErrorCode, EvoMachineError, StrategyError
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class JessStrategy(AbstractStrategy):
    """
    Strategy for experiment on 2024-03-07 with Jess and Laura.
    Strategy:

    - Shine light at 535 nm for 3h and image every 5 min at 480 nm.
    - Shine light at 670 nm for 3h and image every 5 min at 480 nm.
    - Repeat.

    NOTE: There is no LED at 670 nm currently. Using NO_LED and the red spot instead.
    """
    def __init__(self):
        super().__init__()
        self.path_to_save = Path("/media/hslab/Data/ImageData/Idris/2024-03-07")

        self.exposure_time: int = 1000  # in ms
        self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_interval: float = 5*60  # in seconds

        self.current_color: LEDType = LEDType.LED_405_NM
        self.next_color: LEDType = LEDType.NO_LED
        self.timer_interval: float = 3*60*60  # in seconds
        self.timer: threading.Timer = threading.Timer(self.timer_interval, self.change_color)
        self.time_changes: List[Tuple[datetime, LEDType]] = [(datetime.now(), self.current_color)]
        self.timer_started: bool = False

    def change_color(self):
        tmp = self.current_color
        self.current_color = self.next_color
        self.next_color = tmp
        self.time_changes.append((datetime.now(), self.current_color))
        self.timer = threading.Timer(self.timer_interval, self.change_color)
        self.timer.start()
        logger.info(f"Changed current_color to {self.current_color} from {tmp} at {self.time_changes[-1]}")

    def _initialise(self) -> List[AutomatonCommand]:
        """
        Initialise the strategy. Following attributes are available:

        self.field_of_views : Dict[int, Coordinate]
            Dictionary with fov_id as key and Coordinate as value.
        self.positions: Dict[int, List[int]]
            Dictionary with fov_id as key and list of pos_id as value.
        self.region_of_interests: Dict[int, List[int]]
            Dictionary with pos_id as key and list of roi_id as value.
        self.config_camera: ConfigCamera
            Object defining camera configuration.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the first iteration.
        """
        logger.info("Initialising JessStrategy.")
        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            save=True,
        )
        project = self.command_factory.command_project(
            channel=self.current_color,
            image=np.ones(DMD_WIDTH_HEIGHT),
            duration=self.imaging_interval,
            brightness=100,
        )
        return [image, project]

    def _callback(
            self,
            fov_id: int,
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        """
        Callback function for the strategy. This function is called by the
        automaton when new data is available.

        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `t` : int
            The time of the data.
        `data` : dict
            Processed image data such as cell positions.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed at the next iteration.
        """
        logger.info("FOV {} with data =\n{} \nand errors = {}.".format(
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))

        # Restart timer in case it was stopped
        if not self.timer_started:
            self.timer = threading.Timer(self.timer_interval, self.change_color)
            self.timer.start()
            logger.info(f"Starting timer for changing color at {self.timer_interval} seconds.")

        # Define next commands
        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            save=True,
        )
        project = self.command_factory.command_project(
            channel=self.current_color,
            image=np.ones(DMD_WIDTH_HEIGHT),
            duration=self.imaging_interval,
            brightness=100,
        )
        return [image, project]

    def finalise(self) -> List[AutomatonCommand]:
        """
        Save everything else than images here.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the last iteration.
        """
        logger.info("Finalising strategy and saving data.")
        if self.timer.is_alive():
            self.timer.cancel()
        self.timer_started = False
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = self.path_to_save / f"strategy_2024_03_07_savetime_{current_time}.pkl"
        with open(str(filename), "wb") as file:
            pickle.dump({'time_changes': self.time_changes}, file)

        # Define new command list in case this is a temporary stop only
        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            save=True,
        )
        project = self.command_factory.command_project(
            channel=self.current_color,
            image=np.ones(DMD_WIDTH_HEIGHT),
            duration=self.imaging_interval,
            brightness=100,
        )
        return [image, project]


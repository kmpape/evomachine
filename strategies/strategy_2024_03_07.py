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


class JessStrategy(AbstractStrategy):
    """
    Example strategy that projects light on half of the FoV for <timer_interval> seconds and then switches to the other
    half. The strategy images every <imaging_interval> seconds and saves the images. Additionally, the strategy saves
    the time at which the changes happened and saves them in a pickle file.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        self.path_to_save = Path("/media/hslab/Data/ImageData/Idris/2024-03-14")

        self.exposure_time: int = 2000  # in ms
        self.imaging_channel: LEDType = LEDType.LED_505_NM
        self.imaging_interval: float = 30*60  # in seconds

        self.dmd_led_channel: LEDType = LEDType.LED_538_NM
        self.dmd_image_1 = np.ones(shape=DMD_WIDTH_HEIGHT, dtype=np.uint8)*255
        self.dmd_image_1[0:int(DMD_WIDTH_HEIGHT[0]/2), :] = 0
        self.dmd_image_2 = np.ones(shape=DMD_WIDTH_HEIGHT, dtype=np.uint8)*255
        self.dmd_image_2[int(DMD_WIDTH_HEIGHT[0]/2):-1, :] = 0
        # HACK for avoiding overfill
        self.dmd_image_1[2250:-1, :] = 0
        self.dmd_image_2[2250:-1, :] = 0

        self.current_dmd_image = self.dmd_image_1
        self.next_dmd_image = self.dmd_image_2

        self.timer_interval: float = 60*60*5  # in seconds
        # Change DMD projection using a timer
        self.timer: threading.Timer = threading.Timer(self.timer_interval, self.change_image)
        self.time_changes: List[datetime] = []
        self.timer_started: bool = False
        self.timer_must_stop: bool = False

    def change_image(self):
        tmp = self.current_dmd_image
        self.current_dmd_image = self.next_dmd_image
        self.next_dmd_image = tmp
        self.time_changes.append(datetime.now())
        if not self.timer_must_stop:
            self.timer = threading.Timer(self.timer_interval, self.change_image)
            self.timer.start()
            logger.info(f"Changed image. Switching to next image in {self.timer_interval/60/60} min.")
        else:
            logger.info("Cancelling timer.")

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
            channel=self.dmd_led_channel,
            image=self.current_dmd_image,
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
            self.timer = threading.Timer(self.timer_interval, self.change_image)
            self.timer.start()
            self.timer_started = True
            logger.info(f"Starting timer for changing image at {self.timer_interval} seconds.")

        # Define next commands
        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            save=True,
        )
        project = self.command_factory.command_project(
            channel=self.dmd_led_channel,
            image=self.current_dmd_image,
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
        self.timer_must_stop = True
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
            channel=self.dmd_led_channel,
            image=self.current_dmd_image,
            duration=self.imaging_interval,
            brightness=100,
        )
        return [image, project]

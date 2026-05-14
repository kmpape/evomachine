from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import threading

from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger, ConfigCameraFactory, ConfigImageProcessorFactory, USE_DMD_SOCKET, \
    ConfigImageProcessor
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import EvoMachineError
from evomachine.types import LEDType
from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class UVTestingStrategyv2(AbstractStrategy):
    """
    ...
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        self.path_to_save = Path("/media/hslab/Data/ImageData/Idris/2024-04-30")

        self.callback_counter: int = 0  # TODO add this to base class and reset after testing routine

        # Imaging properties
        self.exposure_time: int = 100  # in ms
        self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_interval: float = 10*60  # in seconds
        self.imaging_brightness: float = 15

        # UV Projection properties
        self.proj_channel: LEDType = LEDType.LED_385_NM
        self.proj_img = np.zeros(shape=DMD_WIDTH_HEIGHT, dtype=np.uint8)
        self.proj_img[self.proj_img.shape[0] // 4:self.proj_img.shape[0] // 2, :] = 255
        self.proj_time: float = 60  # seconds
        self.proj_brightness: float = 15

    def _initialise(self) -> list[AutomatonCommand]:
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
        logger.info("Initialising UVTestingStrategy.")
        self.callback_counter = 0

        cmd_list = []

        logger.info(f"Initialising strategy.")
        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            brightness=[self.imaging_brightness],
            save=True,
        )
        cmd_list.append(image)

        return cmd_list

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[EvoMachineError],
    ) -> list[AutomatonCommand]:
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
        logger.info("At callback {}: FOV {} with data={} and errors = {}.".format(
            self.callback_counter,
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))

        cmd_list = []

        project = self.command_factory.command_project(
            channel=self.proj_channel,
            image=self.proj_img,
            duration=self.proj_time,
            brightness=self.proj_brightness,
        )
        cmd_list.append(project)

        wait = self.command_factory.command_wait(
            duration=self.imaging_interval,
            set_live_mode=False,
        )
        cmd_list.append(wait)

        image = self.command_factory.command_image(
            channels=[self.imaging_channel],
            exposure_time=self.exposure_time,
            segment=False,
            brightness=[self.imaging_brightness],
            save=True,
        )
        cmd_list.append(image)

        return cmd_list

    def finalise(self) -> list[AutomatonCommand]:
        """
        Save everything else than images here.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the last iteration.
        """
        logger.info("Finalising strategy and saving data.")

        return []

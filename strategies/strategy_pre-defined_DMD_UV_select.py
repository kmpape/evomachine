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
from skimage.io import imread_collection


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class DMD_UV_select(AbstractStrategy):
    """
    ...
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        self.path_to_save = Path("/media/hslab/Data/ImageData/Idris/2024-XX-XX-mCherry_GFP_select")

        self.callback_counter: int = 0  # TODO add this to base class and reset after testing routine

        # Imaging properties
        self.exposure_time: int = 100  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]  # mCherry GFP, correct format for multiple??
        self.imaging_interval: float = 10*60  # in seconds
        self.imaging_brightness: float = 15

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.proj_channel: LEDType = LEDType.LED_385_NM
        patterns = imread_collection('pathtoDMDimages*') # Import pre-prepared DMD_images 3200 x 3200
        DMD_images = [self.dmd.img_to_dmd_array(pattern) for pattern in patterns]
        for n,i in enumerate(DMD_images): # Check DMD dimensions.
            if i.shape != DMD_WIDTH_HEIGHT:
                print(f"Incorrect DMD dimensions. Expected shape {DMD_WIDTH_HEIGHT} for image {n}, got shape {i.shape}.")
        self.proj_img = {i: val for i, val in enumerate(DMD_images)} 
        self.proj_times = 10
        self.proj_brightness = 29
        self.proj_repeat = True

        # Remove the repeated projections from the waiting time
        self.imaging_interval = self.imaging_interval - self.proj_times
        assert self.imaging_interval > 0

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
        logger.info("Initialising DMD_UV_select.")
        self.callback_counter = 0

        cmd_list = []

        logger.info(f"Initialise DMD_UV_select.\n"
                    f"proj_times={self.proj_times}\n"
                    f"proj_brightness={self.proj_brightness}\n"
                    f"proj_repeat={self.proj_repeat}\n"
                    f"imaging_interval={self.imaging_interval}")

        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            image = self.command_factory.command_image(
                channels=self.imaging_channels, ## Need to check this due to double LED
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
        logger.info("At callback {}: FOV {} with data={} and errors = {}.".format(
            self.callback_counter,
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))

        cmd_list = []
        if self.callback_counter == 0 and self.do_project:
            logger.info("Starting UV projections on all FoVs.")
            for i in range(len(self.field_of_views)):
                move = self.command_factory.command_move(fov_id=i)
                cmd_list.append(move)
                project = self.command_factory.command_project(
                    channel=self.proj_channel,
                    image=self.proj_img[i],
                    duration=self.proj_times,
                    brightness=self.proj_brightness,
                )
                cmd_list.append(project)
        else:
            for i in range(len(self.field_of_views)):
                move = self.command_factory.command_move(fov_id=i)
                cmd_list.append(move)
                image = self.command_factory.command_image(
                    channels=self.imaging_channels,
                    exposure_time=self.exposure_time,
                    brightness=[self.imaging_brightness],
                    segment=False,
                    save=True,
                )
                cmd_list.append(image)
                if self.proj_repeat and self.proj_times > 0 and self.do_project:
                    project = self.command_factory.command_project(
                        channel=self.proj_channel,
                        image=self.proj_img[i],
                        duration=self.proj_times,
                        brightness=self.proj_brightness,
                    )
                    cmd_list.append(project)

            wait = self.command_factory.command_wait(
                duration=self.imaging_interval,
                set_live_mode=False,
            )
            cmd_list.append(wait)

        self.callback_counter += 1

        return cmd_list

    def finalise(self) -> List[AutomatonCommand]:
        """
        Save everything else than images here.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the last iteration.
        """
        logger.info("Finalising strategy and saving data.")

        return []

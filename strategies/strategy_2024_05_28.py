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


class UVTestingStrategyv5(AbstractStrategy):
    """
    ...
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        self.path_to_save = Path("/media/hslab/Data/ImageData/Vicente/2024-06-25")

        # Imaging properties
        self.exposure_time: int = 100  # in ms
        self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_interval: float = 10*60  # in seconds
        self.imaging_brightness: float = 15

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.proj_channel: LEDType = LEDType.LED_385_NM
        self.proj_img = np.zeros(shape=DMD_WIDTH_HEIGHT, dtype=np.uint8)
        self.proj_img[self.proj_img.shape[0] // 4:self.proj_img.shape[0] // 2, :] = 255  # For FoV 1
        self.proj_times = {i: val for i, val in enumerate([0, 60, 60, 60])}  # in seconds
        self.proj_brightness = {i: val for i, val in enumerate([0, 15, 15, 15])}  # brightness in (0,29]
        self.proj_roi = 3
        self.proj_roi_fill_x = 1.0
        self.proj_roi_fill_y = 0.3
        self.proj_repeat = {i: False for i in self.proj_brightness.keys()}
        self.proj_repeat[len(self.proj_brightness) - 3] = True
        self.proj_repeat[len(self.proj_brightness) - 2] = True
        self.proj_repeat[len(self.proj_brightness) - 1] = True
        assert(len(self.proj_times) == len(self.proj_brightness))
        assert(len(self.proj_times) == self.proj_roi+1)

        # Remove the repeated projections from the waiting time
        self.imaging_interval = self.imaging_interval - sum([t for t, rep in
                                                             zip(self.proj_times.values(), self.proj_repeat.values())])
        assert self.imaging_interval > 0

    def _initialise(self) -> list[AutomatonCommand]:
        """
        Initialise the strategy. Note that initialise will be called several times, and it therefore MUST reset
        the strategy to an initial state. Changes to hard-coded constructor attributes that are changed after the
        strategy started MUST be reset here. Following attributes are available:

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

        cmd_list = []

        if len(self.field_of_views) < len(self.proj_times):
            logger.warning(f"Received {len(self.field_of_views)} FoVs, "
                           f"but expected at least {len(self.proj_times)} FoVs.")
        else:
            logger.info(f"Initialise UVTestingStrategyv5.\n"
                        f"proj_times={self.proj_times.values()}\n"
                        f"proj_brightness={self.proj_brightness.values()}\n"
                        f"proj_repeat={self.proj_repeat.values()}\n"
                        f"imaging_interval={self.imaging_interval}")

        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
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
        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            image = self.command_factory.command_image(
                channels=[self.imaging_channel],
                exposure_time=self.exposure_time,
                brightness=[self.imaging_brightness],
                segment=False,
                save=True,
            )
            cmd_list.append(image)
            if self.do_project:
                if i < self.proj_roi:
                    if (self.proj_repeat[i] or self.callback_counter == 0) and self.proj_times[i] > 0:
                        project = self.command_factory.command_project(
                            channel=self.proj_channel,
                            image=self.proj_img,
                            duration=self.proj_times[i],
                            brightness=self.proj_brightness[i],
                        )
                        cmd_list.append(project)
                else:
                    roi_ids = [j for j in self.region_of_interests[i] if j % 2 == 0]
                    project_roi = self.command_factory.command_project_roi(
                        channel=self.proj_channel,
                        pos_id=i,
                        roi_ids=roi_ids,
                        duration=self.proj_times[self.proj_roi],
                        fill_x=self.proj_roi_fill_x,
                        fill_y=self.proj_roi_fill_y,
                    )
                    cmd_list.append(project_roi)

        wait = self.command_factory.command_wait(
            duration=self.imaging_interval,
            set_live_mode=False,
        )
        cmd_list.append(wait)

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

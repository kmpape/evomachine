from datetime import datetime
import numpy as np
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger, ConfigCameraFactory, ConfigImageProcessorFactory, USE_DMD_SOCKET, \
    ConfigImageProcessor, DATA_DIR
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT, CAM_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import EvoMachineError
from evomachine.evotypes import LEDType
from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class UVStrategy(AbstractStrategy):
    """
    Strategy for testing UV.
    All combinations of 60s, 30s, 5s treatment with 29%, 10%, 5% UV.
    All FoVs are split in half to assess the effect of each treatment on neighbouring cells.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = Path("/mnt/nvme1/data/ImageData/UV_Testing_" + datestr)
        # self.path_to_save = Path(DATA_DIR).joinpath("UV_Testing_" + datestr)
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)
        self.path_to_save = self.path_to_save.joinpath("UVStrategy")
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        # Imaging properties
        self.exposure_time: int = 500  # in ms
        # self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_channel: LEDType = LEDType.LED_565_NM
        self.imaging_interval: float = 5*60  # in seconds
        self.imaging_brightness: float = 29

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.proj_channel: LEDType = LEDType.LED_385_NM
        proj_img_cam = np.zeros(shape=CAM_WIDTH_HEIGHT, dtype=np.uint8)
        proj_img_cam[500:CAM_WIDTH_HEIGHT[0] // 2, 200:3000] = 255
        self.proj_img = self.dmd.img_to_dmd_array(proj_img_cam)
        # self.proj_img = np.zeros(shape=DMD_WIDTH_HEIGHT, dtype=np.uint8)
        # self.proj_img[self.proj_img.shape[0] // 4:self.proj_img.shape[0] // 2, :] = 255

        self.start_time_UV: float | None = None
        self.proj_delay: float = 13.5 * 60 * 60  # in seconds
        self.proj_times: dict[int, int] = {i: val for i, val in enumerate([300, 30,  45, 60, 60, 60, 60, 0, 0])}  # in seconds
        self.proj_brightness: dict[int, int] = {i: val for i, val in enumerate([60,  60,  60, 60, 15, 30, 90, 0, 0])}  # brightness in (0,29]
        self.has_projected: dict[int, bool] = {i: False for i in range(len(self.proj_times))}

    @staticmethod
    def format_time(t: float | int | None) -> str:
        return "None" if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))

    def _initialise(self) -> List[AutomatonCommand]:
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
                        f"imaging_interval={self.imaging_interval}")

        current_time = time.time()
        self.start_time_UV = current_time + self.proj_delay
        logger.info(f"Current time is {self.format_time(current_time)}. "
                    f"Starting UV projections at {self.format_time(self.start_time_UV)}.")

        for i in range(len(self.field_of_views)):
            self.has_projected[i] = False
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
            image = self.command_factory.command_image(
                channels=[self.imaging_channel],
                exposure_time=self.exposure_time,
                brightness=[self.imaging_brightness],
                segment=False,
                pattern=self.proj_img,
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
            if i < len(self.proj_times):
                if (time.time() >= self.start_time_UV) and (not self.has_projected[i]) and self.proj_times[i] > 0:
                    logger.info(f"Imaging UV pattern now.")
                    image = self.command_factory.command_image(
                        channels=[self.imaging_channel],
                        exposure_time=self.exposure_time,
                        brightness=[self.imaging_brightness],
                        segment=False,
                        pattern=self.proj_img,
                        save=True,
                    )
                    cmd_list.append(image)

                    logger.info(f"Projecting UV on FoV {i} now.")
                    self.has_projected[i] = True
                    project = self.command_factory.command_project(
                        channel=self.proj_channel,
                        image=self.proj_img,
                        duration=self.proj_times[i],
                        brightness=self.proj_brightness[i],
                    )
                    cmd_list.append(project)

        wait = self.command_factory.command_wait(
            duration=self.imaging_interval,
            set_live_mode=False,
        )
        cmd_list.append(wait)

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

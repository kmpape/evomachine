from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import threading
import time
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

from delta.rt import ROIRT


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class ROITestingStrategy(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = Path("/mnt/nvme1/data/ImageData/UV_by_ROI_" + datestr + "-b")
        self.path_to_save.mkdir(parents=False, exist_ok=True)

        # Imaging properties
        self.exposure_time: int = 200  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]
        self.imaging_interval: float = 5*60  # in seconds
        self.imaging_brightness: float = 29

        # IP properties
        self.cfg.channels_seg = [LEDType.LED_450_NM, LEDType.LED_565_NM]  # channels will be averaged for segmentation
        self.red_threshold = 50
        self.is_red: dict[int, list[bool]] = {}
        self.is_red_id: dict[int, list[int]] = {}

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.proj_channel: LEDType = LEDType.LED_385_NM
        self.start_time_UV: float | None = None
        self.proj_delay: float = 120 * 60  # in seconds
        self.proj_time = 300
        self.proj_brightness = 99
        self.has_projected = [False, False, False, False]
        self.proj_imgs = []

    @staticmethod
    def format_time(t: float | int | None) -> str:
        return "None" if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))

    def roi_is_red(self, roi: ROIRT) -> bool:
        fluo = roi.get_fluo(frame=1)  # get the newest fluo frames
        fluo_1 = fluo[self.cfg.channel_to_index[LEDType.LED_450_NM]]
        fluo_2 = fluo[self.cfg.channel_to_index[LEDType.LED_565_NM]]
        return fluo_1.mean() > fluo_2.mean()

    def _initialise(self) -> List[AutomatonCommand]:
        current_time = time.time()
        self.start_time_UV = current_time + self.proj_delay
        logger.info(f"Current time is {self.format_time(current_time)}. "
                    f"Starting UV projections at {self.format_time(self.start_time_UV)}.")

        # Reset
        self.has_projected = [False for _ in self.has_projected]

        # Categorise trenches
        self.is_red = {
            pos: [self.roi_is_red(roi) for roi in proc.rois] for pos, proc in enumerate(self.pos_processors)
        }
        self.is_red_id = {
            pos: [iroi for iroi, roi_is_red in enumerate(self.is_red[pos]) if roi_is_red] for pos in self.is_red.keys()
        }
        for i in range(len(self.field_of_views)):
            logger.info(f"Field of view {i}: #ROI={len(self.pos_processors[i].rois)} "
                        f"#is_red={sum(self.is_red[i])}, IDs={self.is_red_id[i]}")

        cmd_list = []
        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            image = self.command_factory.command_image(
                channels=self.imaging_channels,
                exposure_time=self.exposure_time,
                segment=False,
                brightness=[self.imaging_brightness],
                save=True,
            )
            cmd_list.append(image)

            if self.pos_processors:
                boxes_to_project = self.is_red_id[i]
                self.proj_imgs.append(self.dmd.pattern_from_roi_boxes(
                    boxes=boxes_to_project,
                    fill_x=1,
                    fill_y=0.5,
                ))
                image = self.command_factory.command_image(
                    channels=self.imaging_channels,
                    exposure_time=self.exposure_time,
                    brightness=[self.imaging_brightness],
                    segment=False,
                    pattern=self.proj_imgs[i],
                    save=True,
                )
                cmd_list.append(image)
            else:
                logger.warning(f"strategy_UV_by_ROI: no position processors. Cannot project onto ROI.")

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
                channels=self.imaging_channels,
                exposure_time=self.exposure_time,
                brightness=[self.imaging_brightness],
                segment=False,
                save=True,
            )
            cmd_list.append(image)
            if i < len(self.has_projected):
                if (time.time() >= self.start_time_UV) and (not self.has_projected[i]) and i > 1:
                    logger.info(f"Imaging UV pattern now.")
                    image = self.command_factory.command_image(
                        channels=self.imaging_channels,
                        exposure_time=self.exposure_time,
                        brightness=[self.imaging_brightness],
                        segment=False,
                        pattern=self.proj_imgs[i],
                        save=True,
                    )
                    cmd_list.append(image)

                    logger.info(f"Projecting UV on FoV {i} now.")
                    self.has_projected[i] = True
                    project = self.command_factory.command_project(
                        channel=self.proj_channel,
                        image=self.proj_imgs[i],
                        duration=self.proj_time,
                        brightness=self.proj_brightness,
                    )
                    cmd_list.append(project)

        wait = self.command_factory.command_wait(
            duration=self.imaging_interval,
            set_live_mode=False,
        )
        cmd_list.append(wait)

        return cmd_list

    def finalise(self) -> List[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")

        return []

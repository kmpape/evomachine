from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import skimage
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
from evomachine.evotypes import LEDType, FilterWheelType
from evomachine.strategy import AbstractStrategy

from delta.rt import ROIRT


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class ROIbyColorStrategy(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = Path("/mnt/nvme1/data/ImageData/UV_by_color_" + datestr)
        self.path_to_save.mkdir(parents=False, exist_ok=True)

        # Imaging properties
        self.exposure_time: int = 500  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]
        self.imaging_filters: list[FilterWheelType] = [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_592nm]
        self.imaging_interval: float = 5*60  # in seconds
        self.imaging_brightness: list[float] = [29, 29]

        # IP properties
        self.cfg.channels_seg = [LEDType.LED_450_NM, LEDType.LED_565_NM]  # channels will be averaged for segmentation
        self.is_red: dict[int, list[bool]] = {}
        self.is_not_red_id: dict[int, list[int]] = {}

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.project_from_pos: int = 2
        self.fill_x: float = 1
        self.fill_y: float = 1
        self.invert_pattern: bool = True
        self.proj_channel: LEDType = LEDType.LED_385_NM
        self.start_time_UV: float | None = None
        self.proj_delay: float = 11 * 60 * 60  # in seconds
        self.proj_time = 300
        self.proj_brightness = 60
        self.has_projected = []
        self.proj_imgs = []
        self.made_proj_imgs = False

    @staticmethod
    def format_time(t: float | int | None) -> str:
        return "None" if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))

    def roi_is_red(self, roi: ROIRT) -> bool:
        fluo = roi.get_fluo(frame=1)  # get the newest fluo frames
        fluo_1 = fluo[self.cfg.channel_to_index[LEDType.LED_450_NM]]
        fluo_2 = fluo[self.cfg.channel_to_index[LEDType.LED_565_NM]]
        is_red = fluo_1.max() < fluo_2.max()

        return is_red

    def make_projection_images(self):
        if self.pos_processors:
            self.is_red = {
                pos: [self.roi_is_red(roi) for roi in proc.rois]
                for pos, proc in enumerate(self.pos_processors)
            }
            self.is_not_red_id = {
                pos: [iroi for iroi, roi_is_red in enumerate(self.is_red[pos]) if (not roi_is_red)]
                for pos in self.is_red.keys()
            }
            for i in range(len(self.field_of_views)):
                if not self.pos_processors[i].rois:
                    msg = f"No ROI boxes found for position {i}. Projecting UV on full image."
                    logger.warning(msg)
                    cam_img = self.dmd.get_one_array(img_size=self.dmd.width_height_CAM) * 255
                    self.proj_imgs.append(self.dmd.img_to_dmd_array(cam_img))
                else:
                    logger.info(f"Field of view {i}: #ROI={len(self.pos_processors[i].rois)} "
                                f"#is_red={sum(self.is_red[i])}, IDs={self.is_not_red_id[i]}")

                    boxes_to_project = [self.pos_processors[i].roi_boxes[iroi] for iroi in self.is_not_red_id[i]]
                    self.proj_imgs.append(self.dmd.pattern_from_roi_boxes(
                        boxes=boxes_to_project,
                        fill_x=self.fill_x,
                        fill_y=self.fill_y,
                        invert=self.invert_pattern,
                    ))
        else:
            logger.warning(f"strategy_UV_by_ROI: no position processors. Cannot project onto ROI.")

    def _initialise(self) -> List[AutomatonCommand]:
        current_time = time.time()
        self.start_time_UV = current_time + self.proj_delay
        logger.info(f"Current time is {self.format_time(current_time)}. "
                    f"Starting UV projections at {self.format_time(self.start_time_UV)}.")

        # Reset
        self.has_projected = [False for _ in range(len(self.field_of_views))]
        self.proj_imgs = []

        # logger.info("Making boxes for projection.")
        # self.make_projection_images()
        self.made_proj_imgs = False

        cmd_list = []
        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            for i_img in range(2):
                image = self.command_factory.command_image(
                    channels=[self.imaging_channels[i_img]],
                    filter_wheel=self.imaging_filters[i_img],
                    exposure_time=self.exposure_time,
                    segment=False,
                    brightness=[self.imaging_brightness[i_img]],
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
            for i_img in range(2):
                image = self.command_factory.command_image(
                    channels=[self.imaging_channels[i_img]],
                    filter_wheel=self.imaging_filters[i_img],
                    exposure_time=self.exposure_time,
                    segment=False,
                    brightness=[self.imaging_brightness[i_img]],
                    save=True,
                )
                cmd_list.append(image)
            if i < len(self.has_projected):
                if (time.time() >= self.start_time_UV) and (not self.has_projected[i]):
                    if not self.made_proj_imgs:
                        logger.info("Making boxes for projection.")
                        self.make_projection_images()
                        self.made_proj_imgs = True
                    logger.info(f"Imaging UV pattern now.")
                    for i_img in range(2):
                        image = self.command_factory.command_image(
                            channels=[self.imaging_channels[i_img]],
                            filter_wheel=self.imaging_filters[i_img],
                            exposure_time=self.exposure_time,
                            segment=False,
                            brightness=[self.imaging_brightness[i_img]],
                            save=True,
                            pattern=self.proj_imgs[i],
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

        for cmd in cmd_list:
            logger.info(f"Sending command {cmd} with args {cmd.command_args}.")

        return cmd_list

    def finalise(self) -> List[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")

        return []


class ROIbyColorStrategy_v20250301(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = Path("/mnt/nvme1/data/ImageData/UV_by_color_" + datestr)
        self.path_to_save.mkdir(parents=False, exist_ok=True)

        # Imaging properties
        self.exposure_time: int = 200  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]
        self.imaging_filters: list[FilterWheelType] = [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_592nm]
        self.imaging_interval: float = 5*60  # in seconds
        self.imaging_brightness: list[float] = [29, 29]

        # IP properties
        self.cfg.channels_seg = [LEDType.LED_450_NM, LEDType.LED_565_NM]  # channels will be averaged for segmentation
        self.red_threshold = 50
        self.is_red: dict[int, list[bool]] = {}
        self.is_red_id: dict[int, list[int]] = {}

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.project_from_pos: int = 2
        self.fill_x: float = 1.0
        self.fill_y: float = 0.5
        self.proj_channel: LEDType = LEDType.LED_385_NM
        self.start_time_UV: float | None = None
        self.proj_delay: float = 10 * 60  # in seconds
        self.proj_time = 300
        self.proj_brightness = 60
        self.has_projected = []
        self.proj_imgs = []

    @staticmethod
    def format_time(t: float | int | None) -> str:
        return "None" if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))

    def roi_is_red(self, roi: ROIRT) -> bool:
        fluo = roi.get_fluo(frame=1)  # get the newest fluo frames
        fluo_1 = fluo[self.cfg.channel_to_index[LEDType.LED_450_NM]]
        fluo_2 = fluo[self.cfg.channel_to_index[LEDType.LED_565_NM]]
        is_red = fluo_1.max() < fluo_2.max()

        return is_red

    def make_projection_images(self):
        if self.pos_processors:
            self.is_red = {
                pos: [self.roi_is_red(roi) for roi in proc.rois]
                for pos, proc in enumerate(self.pos_processors)
            }
            self.is_red_id = {
                pos: [iroi for iroi, roi_is_red in enumerate(self.is_red[pos]) if roi_is_red]
                for pos in self.is_red.keys()
            }
            for i in range(len(self.field_of_views)):
                logger.info(f"Field of view {i}: #ROI={len(self.pos_processors[i].rois)} "
                            f"#is_red={sum(self.is_red[i])}, IDs={self.is_red_id[i]}")

                boxes_to_project = [self.pos_processors[i].roi_boxes[iroi] for iroi in self.is_red_id[i]]
                self.proj_imgs.append(self.dmd.pattern_from_roi_boxes(
                    boxes=boxes_to_project,
                    fill_x=self.fill_x,
                    fill_y=self.fill_y,
                ))
        else:
            logger.warning(f"strategy_UV_by_ROI: no position processors. Cannot project onto ROI.")

    def _initialise(self) -> List[AutomatonCommand]:
        current_time = time.time()
        self.start_time_UV = current_time + self.proj_delay
        logger.info(f"Current time is {self.format_time(current_time)}. "
                    f"Starting UV projections at {self.format_time(self.start_time_UV)}.")

        # Reset
        self.has_projected = [False for _ in range(len(self.field_of_views))]
        self.proj_imgs = []

        logger.info("Making boxes for projection.")
        self.make_projection_images()

        cmd_list = []
        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            for i_img in range(2):
                image = self.command_factory.command_image(
                    channels=[self.imaging_channels[i_img]],
                    filter_wheel=self.imaging_filters[i_img],
                    exposure_time=self.exposure_time,
                    segment=False,
                    brightness=[self.imaging_brightness[i_img]],
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
            for i_img in range(2):
                image = self.command_factory.command_image(
                    channels=[self.imaging_channels[i_img]],
                    filter_wheel=self.imaging_filters[i_img],
                    exposure_time=self.exposure_time,
                    segment=False,
                    brightness=[self.imaging_brightness[i_img]],
                    save=True,
                )
                cmd_list.append(image)
            if i < len(self.has_projected):
                if (time.time() >= self.start_time_UV) and (not self.has_projected[i]):
                    logger.info(f"Imaging UV pattern now.")
                    for i_img in range(2):
                        image = self.command_factory.command_image(
                            channels=[self.imaging_channels[i_img]],
                            filter_wheel=self.imaging_filters[i_img],
                            exposure_time=self.exposure_time,
                            segment=False,
                            brightness=[self.imaging_brightness[i_img]],
                            save=True,
                            pattern=self.proj_imgs[i],
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

        for cmd in cmd_list:
            logger.info(f"Sending command {cmd} with args {cmd.command_args}.")

        return cmd_list

    def finalise(self) -> List[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")

        return []

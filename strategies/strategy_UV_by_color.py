from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import skimage
import threading
import time

import delta.imgops
from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger, ConfigCameraFactory, ConfigImageProcessorFactory, USE_DMD_SOCKET, \
    ConfigImageProcessor
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import EvoMachineError
from evomachine.types import LEDType, FilterWheelType
from evomachine.strategy import AbstractStrategy
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"

from delta.rt import ROIRT


logger = get_logger(name=__name__)


# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = ConfigImageProcessorFactory.default_config()


class ROIbyColorStrategyv2(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = IMAGE_DIR / ("UV_by_color_" + datestr)
        self.path_to_save.mkdir(parents=False, exist_ok=True)

        self.debug = False

        # Imaging properties
        self.exposure_time: int = 500  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]
        self.imaging_filters: list[FilterWheelType] = [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_592nm]
        if not self.debug:
            self.imaging_interval: float = 3*60  # in seconds TODO
        else:
            self.imaging_interval: float = 30  # in seconds
        self.imaging_brightness: list[float] = [29, 29]

        # IP properties
        self.cfg.channels_seg = [LEDType.LED_450_NM, LEDType.LED_565_NM]  # channels will be averaged for segmentation
        self.is_red: dict[int, list[bool]] = {}
        self.is_red_id: dict[int, list[int]] = {}
        self.is_not_red_id: dict[int, list[int]] = {}
        self.project_onto_red: bool = True

        # UV Projection properties
        self.do_project: bool = True  # Enable/disable UV projection
        self.invert_pattern: bool = True
        if self.invert_pattern:
            self.box_fill_x = 1.1
            self.box_fill_y = 1.1
        else:
            self.box_fill_x = 1.1
            self.box_fill_y = 0.5
        if not self.debug:
            self.proj_channel: LEDType = LEDType.LED_385_NM
        else:
            self.proj_channel: LEDType = LEDType.LED_645_NM
        self.start_time_UV: float | None = None
        if not self.debug:
            self.proj_delay: float = 3 * 60 * 60  # in seconds TODO
        else:
            self.proj_delay: float = 60  # in seconds
        self.proj_time = 60 * 2  # in seconds
        self.proj_brightness = 90
        self.has_projected = []
        self.all_boxes = {}
        self.proj_imgs = []
        self.made_proj_indices = False
        self.fluo_images = {}

    @staticmethod
    def format_time(t: float | int | None) -> str:
        return "None" if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))

    def roi_is_red(self, roi: ROIRT) -> bool:
        fluo = roi.get_fluo(frame=1)  # get the newest fluo frames
        fluo_1 = fluo[self.cfg.channel_to_index[LEDType.LED_450_NM]]
        fluo_2 = fluo[self.cfg.channel_to_index[LEDType.LED_565_NM]]
        is_red = fluo_1.mean() < fluo_2.mean()

        return is_red

    def make_projection_indices(self):
        if self.pos_processors:
            self.fluo_images = {
                pos: [(roi.get_fluo(frame=1)[self.cfg.channel_to_index[LEDType.LED_450_NM]],
                       roi.get_fluo(frame=1)[self.cfg.channel_to_index[LEDType.LED_565_NM]]) for roi in proc.rois]
                for pos, proc in enumerate(self.pos_processors)
            }
            self.is_red = {
                pos: [self.roi_is_red(roi) for roi in proc.rois]
                for pos, proc in enumerate(self.pos_processors)
            }
            self.is_red_id = {
                pos: [iroi for iroi, is_red in enumerate(self.is_red[pos]) if is_red]
                for pos, proc in enumerate(self.pos_processors)
            }
            self.is_not_red_id = {
                pos: [iroi for iroi, roi_is_red in enumerate(self.is_red[pos]) if (not roi_is_red)]
                for pos in self.is_red.keys()
            }
            logger.info(f"Pattern onto red: {self.project_onto_red}")
            logger.info(f"Invert pattern: {self.invert_pattern}")
            for pos in self.is_red_id.keys():
                logger.info(f"Pos {pos:02d}: num red={len(self.is_red_id[pos]):03d}, "
                            f"num green={len(self.is_not_red_id[pos]):03d}, "
                            f" total={len(self.pos_processors[pos].rois):03d}.")
        else:
            logger.warning(f"strategy_UV_by_ROI: no position processors. Cannot project onto ROI.")

    def _initialise(self) -> list[AutomatonCommand]:
        current_time = time.time()
        self.start_time_UV = current_time + self.proj_delay
        logger.info(f"Current time is {self.format_time(current_time)}. "
                    f"Starting UV projections at {self.format_time(self.start_time_UV)}.")

        # Reset
        self.has_projected = [False for _ in range(len(self.field_of_views))]
        self.proj_imgs = []
        self.made_proj_indices = False

        logger.info("Making boxes for projection.")
        if self.pos_processors:
            self.make_projection_indices()

        cmd_list = []
        for i in range(len(self.field_of_views)):
            if self.pos_processors:
                self.all_boxes[i] = self.pos_processors[i].roi_boxes
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            image = self.command_factory.command_image(
                channels=self.imaging_channels,
                filter_wheel=self.imaging_filters,
                exposure_time=self.exposure_time,
                segment=False,
                brightness=self.imaging_brightness,
                save=True,
            )
            cmd_list.append(image)
            if self.pos_processors:
                # Just as test: project with imaging
                project = self.command_factory.command_project_roi(
                    channel=self.imaging_channels[0],
                    pos_id=i,
                    roi_ids=self.is_red_id[i] if self.project_onto_red else self.is_not_red_id[i],
                    duration=10,
                    brightness=self.proj_brightness,
                    fill_x=self.box_fill_x,
                    fill_y=self.box_fill_y,
                    invert=self.invert_pattern,
                    set_live_mode=True,
                )
                cmd_list.append(project)


        if True:
            fname = self.path_to_save / Path("proj_imgs.pkl")
            with open(fname, "wb") as file:

                to_save = {
                    "all_boxes": self.all_boxes,
                    "is_red_id": self.is_red_id,
                    "is_not_red_id": self.is_not_red_id,
                    "start_time_UV": self.format_time(self.start_time_UV),
                    "invert_pattern": self.invert_pattern,
                    "project_onto_red": self.project_onto_red,
                    "fluo_images": self.fluo_images,
                }
                pickle.dump(to_save, file)

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
        # logger.info("At callback {}: FOV {} with data={} and errors = {}.".format(
        #     self.callback_counter,
        #     fov_id,
        #     '    \n'.join(str(d) for d in data),
        #     errors,
        # ))
        logger.info("At callback {}: FOV {} with errors = {}.".format(
            self.callback_counter,
            fov_id,
            errors,
        ))

        # if self.debug:
        self.make_projection_indices()

        cmd_list = []
        for i in range(len(self.field_of_views)):
            move = self.command_factory.command_move(fov_id=i)
            cmd_list.append(move)
            image = self.command_factory.command_image(
                channels=self.imaging_channels,
                filter_wheel=self.imaging_filters,
                exposure_time=self.exposure_time,
                segment=False,
                brightness=self.imaging_brightness,
                save=True,
            )
            cmd_list.append(image)

        for i in range(len(self.field_of_views)):
            if i < len(self.has_projected):
                if (time.time() >= self.start_time_UV) and (not self.has_projected[i]):
                    move = self.command_factory.command_move(fov_id=i)
                    cmd_list.append(move)
                    if not self.made_proj_indices:
                        self.make_projection_indices()
                        self.made_proj_indices = True
                    logger.info(f"Imaging UV pattern now.")
                    pos_id = i
                    proj_ids = self.is_red_id[i] if self.project_onto_red else self.is_not_red_id[i]
                    roi_boxes = [self.pos_processors[pos_id].roi_boxes[r] for r in proj_ids]
                    drift = self.pos_processors[pos_id].drift_values[-1] if self.cfg.cfg_delta.drift_correction else None
                    if drift is not None:
                        drift = (-drift[0], -drift[1])
                    if self.invert_pattern:
                        black_patches = self.dmd.patches_from_roi_groups(
                            roi_boxes_group_ids=self.pos_processors[pos_id].roi_boxes_group_ids,
                            roi_boxes=self.pos_processors[pos_id].roi_boxes,
                            xshift=0,
                        )
                    else:
                        black_patches = None
                    pattern = self.dmd.pattern_from_roi_boxes(
                        boxes=roi_boxes,
                        fill_x=self.box_fill_x,
                        fill_y=self.box_fill_y,
                        warp=True,
                        invert=self.invert_pattern,
                        drift=drift,
                        black_patches=black_patches,
                    )
                    image = self.command_factory.command_image(
                        channels=self.imaging_channels,
                        filter_wheel=self.imaging_filters,
                        exposure_time=self.exposure_time,
                        segment=False,
                        brightness=self.imaging_brightness,
                        save=True,
                        pattern=pattern,
                        filename_suffix="_pattern",
                    )
                    cmd_list.append(image)

                    logger.info(f"Projecting UV on FoV {i} now.")
                    self.has_projected[i] = True
                    project = self.command_factory.command_project_roi(
                        channel=self.proj_channel,
                        pos_id=i,
                        roi_ids=self.is_red_id[i] if self.project_onto_red else self.is_not_red_id[i],
                        duration=self.proj_time,
                        brightness=self.proj_brightness,
                        fill_x=self.box_fill_x,
                        fill_y=self.box_fill_y,
                        invert=self.invert_pattern,
                        set_live_mode=False,
                    )
                    cmd_list.append(project)

                    pattern_img = self.dmd.pattern_from_roi_boxes(
                        boxes=roi_boxes,
                        fill_x=self.box_fill_x,
                        fill_y=self.box_fill_y,
                        warp=False,
                        invert=self.invert_pattern,
                        drift=drift,
                        black_patches=black_patches,
                    )
                    self.proj_imgs.append(pattern_img)

        wait = self.command_factory.command_wait(
            duration=self.imaging_interval,
            set_live_mode=False,
        )
        cmd_list.append(wait)

        # for cmd in cmd_list:
        #     logger.info(f"Sending command {cmd} with args {cmd.command_args}.")

        return cmd_list

    def finalise(self) -> list[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")
        if True:
            fname = self.path_to_save / Path("proj_imgs_final.pkl")
            drifts = [proc.drift_values for proc in self.pos_processors]
            with open(fname, "wb") as file:
                to_save = {
                    "all_boxes": self.all_boxes,
                    "is_red_id": self.is_red_id,
                    "is_not_red_id": self.is_not_red_id,
                    "start_time_UV": self.format_time(self.start_time_UV),
                    "drifts": drifts,
                    "proj_imgs": self.proj_imgs,
                    "fluo_images": self.fluo_images,
                }
                pickle.dump(to_save, file)

        return []


class ROIbyColorStrategy(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = IMAGE_DIR / ("UV_by_color_" + datestr)
        self.path_to_save.mkdir(parents=False, exist_ok=True)

        # Imaging properties
        self.exposure_time: int = 500  # in ms
        self.imaging_channels: list[LEDType] = [LEDType.LED_450_NM, LEDType.LED_565_NM]
        self.imaging_filters: list[FilterWheelType] = [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_592nm]
        self.imaging_interval: float = 3*60  # in seconds
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
        self.proj_delay: float = 10 * 60  # in seconds
        self.proj_time = 60*10
        self.proj_brightness = 90
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

    def _initialise(self) -> list[AutomatonCommand]:
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

    def finalise(self) -> list[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")

        return []


class ROIbyColorStrategy_v20250301(AbstractStrategy):
    """
    See AbstractStrategy for function documentations and available attributes.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)
        datestr = datetime.today().strftime('%Y-%m-%d')
        self.path_to_save = IMAGE_DIR / ("UV_by_color_" + datestr)
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

    def _initialise(self) -> list[AutomatonCommand]:
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

    def finalise(self) -> list[AutomatonCommand]:
        logger.info("Finalising strategy and saving data.")

        return []

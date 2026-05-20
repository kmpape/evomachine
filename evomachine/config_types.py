from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import json
import numpy as np
from pathlib import Path
from typing import Any, TYPE_CHECKING

import delta
from delta.utils import CroppingBox
from delta.rttypes import TrackingSetting

from evomachine.coordinates import Coordinate
from evomachine.types import BrightnessType, EvoType, ExposureType, FilterWheelType, FocusAlgorithmType, LEDType, \
    ChamberOrientationType, UNKNOWN_POSITION_ID
from evomachine.exceptions import ConfigError, ErrorCode

if TYPE_CHECKING:
    from evomachine.acquisition import FrameAcquisitionSettings


def _get_evomachine_dir() -> Path:
    from evomachine.config import EVOMACHINE_DIR

    return EVOMACHINE_DIR


def _use_sync_board() -> bool:
    from evomachine.config import USE_SYNC_BOARD

    return USE_SYNC_BOARD


@dataclass
class DMDCalibConfigType:
    channel: LEDType | list[LEDType]
    "LED type for calibration."
    brightness: float | int
    "Brightness of LED."
    exposure: float | int
    "Exposure time for calibration in milliseconds."
    line_width: int
    "Thickness of calibration lines."
    step: int
    "Step size for calibration in pixels."
    delay: float | int
    "Delay between calibration steps in seconds."
    start_row: int
    "Start index for rows (DMD coordinates). Should be off-camera-screen."
    end_row: int
    "End index for rows (DMD coordinates). Should be off-camera-screen."
    start_col: int
    "Start index for columns (DMD coordinates). Should be off-camera-screen."
    end_col: int
    "End index for columns (DMD coordinates). Should be off-camera-screen."
    on_mothermachine: bool
    "Executed on mothermachine or normal slide."

    def __post_init__(self):
        if not ((0 <= self.start_row) and (self.start_row < self.end_row) and (self.end_row < 2716)):
            raise ValueError("Indices must be within DMD boundaries.")
        if not ((0 <= self.start_col) and (self.start_col < self.end_col) and (self.end_col < 1600)):
            raise ValueError("Indices must be within DMD boundaries.")

    def __str__(self):
        s = ["DMDCalibConfigType"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)#


class DMDCalibConfigTypeFactory:
    @staticmethod
    def default(channel: LEDType | list[LEDType] = LEDType.LED_450_NM) -> DMDCalibConfigType:
        """
        This configuration should be used together with a mother machine. Modify channels if needed.

        Parameters
        ----------
        channel : LEDType
            Use this to override default channel.
        Returns
        -------
        cfg : DMDCalibConfigType
        """
        return DMDCalibConfigType(
            channel=channel,
            brightness=29,
            exposure=100,  # 100
            line_width=5,
            step=150,  # 400
            delay=0.75,
            start_row=200,  # should be off-screen
            end_row=2500,  # 2500,  # 2250
            start_col=0,
            end_col=1599,
            on_mothermachine=True,
        )

    @staticmethod
    def thin_fluo_slide(channel: LEDType = LEDType.LED_565_NM) -> DMDCalibConfigType:
        """
        This configuration should be used together with a thin fluorescent slide (e.g., coated with rhodamine or with
        dense cells).

        Parameters
        ----------
        channel : LEDType
            Use this to override default channel.
        Returns
        -------
        cfg : DMDCalibConfigType
        """
        return DMDCalibConfigType(
            channel=channel,
            brightness=29,
            exposure=100,  # 100
            line_width=5,
            step=150,  # 400
            delay=0.5,
            start_row=200,  # should be off-screen
            end_row=2200,  # 2500,  # 2250
            start_col=0,
            end_col=1599,
            on_mothermachine=False,
        )

    @staticmethod
    def fluo_slide(channel: LEDType = LEDType.LED_450_NM) -> DMDCalibConfigType:
        """
        This configuration should be used together with a thick fluorescent slide.

        Parameters
        ----------
        channel : LEDType
            Use this to override default channel.
        Returns
        -------
        cfg : DMDCalibConfigType
        """
        return DMDCalibConfigType(
            channel=channel,
            brightness=0.4,  # 100
            exposure=50,  # 100
            line_width=2,
            step=50,  # 400
            delay=0.5,
            start_row=200,  # should be off-screen
            end_row=2200,  # 2500,  # 2250
            start_col=0,
            end_col=1599,
            on_mothermachine=False,
        )


@dataclass
class ImageConfigType:
    pxl_horiz: int
    "Number of pixels in horizontal direction (=number of columns of matrix)."
    pxl_vert: int
    "Number of pixels in vertical direction (=number of rows of matrix)."
    pxl_dtype: np.dtype
    "Datatype of image."

    @property
    def shape(self) -> tuple[int, int]:
        return self.pxl_vert, self.pxl_horiz

    def __post_init__(self):
        if not isinstance(self.pxl_horiz, int) or not self.pxl_horiz > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_horiz: {self.pxl_horiz}")
        if not isinstance(self.pxl_vert, int) or not self.pxl_vert > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_vert: {self.pxl_vert}")
        if not isinstance(self.pxl_dtype, np.dtype):
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_dtype: {self.pxl_dtype}")

    def __str__(self):
        s = ["ImageConfigType"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


class ImageConfigTypeFactory:
    @staticmethod
    def pv_cam() -> ImageConfigType:
        return ImageConfigType(pxl_horiz=3200, pxl_vert=3200, pxl_dtype=np.dtype("uint16"))

    @staticmethod
    def delta() -> ImageConfigType:
        return ImageConfigType(pxl_horiz=696, pxl_vert=520, pxl_dtype=np.dtype("float32"))


@dataclass
class ObjectiveConfigType:
    na: float
    "Numerical aperture NA=n*sin(theta)."
    mag: int
    "Magnification of objective."
    descr: str | None = "UNKNOWN OBJECTIVE"
    "Optional description of objective."

    def __post_init__(self):
        if not isinstance(self.na, float) or not 0 < self.na:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid numerical_aperture: {self.na}")
        if not isinstance(self.mag, int) or not self.mag > 0:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid magnification: {self.mag}")

    def __str__(self):
        s = ["ObjectiveConfigType"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


class ObjectiveConfigTypeFactory:
    @staticmethod
    def default_oil() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=1.4, mag=60, descr="Nikon Plan Apo λ 60x/1.4 Oil")

    @staticmethod
    def default_air() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=0.95, mag=40, descr="Nikon Plan Fluor 40x/0.95")



@dataclass
class ConfigImageProcessor:
    cfg_delta: delta.config.Config
    "Delta configuration object. Contains paths to RoI ID, segmentation, and tracking models."
    channels: list[LEDType]
    "List of channels to be imaged. Used for taking reference frames. Do not include any UV here. The list must " \
    "include channels_seg."  # noqa
    channels_seg: list[LEDType]
    "Channel(s) used for segmentation. If a list is provided, channels are averaged."
    preproc_enabled: bool = True
    "Enable image preprocessing."
    roi_enabled: bool = True
    "Enable ROI identification (preproc_enabled must be true)."
    roi_min_area: int | None = 1500
    "Min. area of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    roi_max_area: int | None = 7000
    "Max. area of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    roi_max_height: int | None = 40
    "Max. height of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    seg_enabled: bool = False
    "Enable image segmentation (preproc_enabled must be true)."
    track_enabled: bool = False
    "Enable tracking (preproc_enabled and seg_enabled must be true)."
    lineage_enabled: bool = False
    "Enable lineage computations(preproc_enabled, seg_enabled, and track_enabled must be true)."
    tracking_setting: TrackingSetting = TrackingSetting.MOTHERONLY  # 2025-11-04
    "Pick tracking algorithm."
    delta_roi_preprocess_target_size: tuple[int, int] = (3200, 3200)
    "Size of microscope images just before being input to DeLTA. This can be different from cfg_delta.target_size_rois."
    image_processing_verbosity: int = 0
    "Lowest verbosity is 0."
    refocus: bool = True
    "Refocus after autofocus loss (otherwise shuts down). Set refocus_using_software_focus=False to avoid using autofocus."
    refocus_using_software_focus: bool = True
    "Use software focus to refocus after autofocus loss. refocus must be True."
    refocus_on_all_positions: bool = False
    "Try refocusing on all recorded positions. refocus and refocus_using_software_focus must be True."
    max_refocus_trials: int = 10
    "Maximum number of refocusing trials before stopping execution."
    chamber_orientation: ChamberOrientationType = ChamberOrientationType.HORIZONTAL
    "Orientation of chambers."

    def copy(self):
        return ConfigImageProcessor(**self.__dict__)

    @property
    def channel_to_index(self) -> dict[LEDType, int]:
        return {c: i for i, c in enumerate(self.channels)}

    def __post_init__(self):
        if not isinstance(self.cfg_delta, delta.config.Config):
            raise TypeError("cfg_delta must be a delta.config.Config object.")
        if not (isinstance(self.channels, list) and all(isinstance(channel, LEDType) for channel in self.channels))\
                or len(self.channels) == 0 or LEDType.NO_LED in self.channels:
            raise ConfigError("Invalid channel list.", ErrorCode.ERROR_CONFIG)
        if not (isinstance(self.channels_seg, list) and all(isinstance(ch, LEDType) for ch in self.channels_seg))\
                or len(self.channels_seg) == 0 or LEDType.NO_LED in self.channels_seg:
            raise ConfigError("Invalid channels_seg list.", ErrorCode.ERROR_CONFIG)
        if not all([ch_seg in self.channels for ch_seg in self.channels_seg]):
            raise ConfigError("All channels_seg must be contained in channels.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.tracking_setting, TrackingSetting):
            raise TypeError("Tracking setting must have type TrackingSetting.")
        if not isinstance(self.image_processing_verbosity, int) or self.image_processing_verbosity < 0:
            raise TypeError("image_processing_verbosity must be an integer >= 0.")
        if self.refocus and self.max_refocus_trials < 1:
            raise TypeError(f"Refocus is True but max_refocus_trials={self.max_refocus_trials}")
        if not isinstance(self.preproc_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.roi_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.seg_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.track_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.lineage_enabled, bool):
            raise TypeError("lineage_enabled must be a boolean.")
        if self.seg_enabled and not self.preproc_enabled:
            raise TypeError("preproc_enabled must be true if seg_enabled is true.")
        if self.track_enabled and ((not self.preproc_enabled) or (not self.seg_enabled)):
            print(f"self.track_enabled = {self.track_enabled} self.preproc_enabled = {self.preproc_enabled} self.seg_enabled = {self.seg_enabled}")
            raise TypeError("preproc_enabled and seg_enabled must be true if track_enabled is true.")
        if self.lineage_enabled and ((not self.preproc_enabled) or (not self.seg_enabled) or (not self.track_enabled)):
            raise TypeError("preproc_enabled and seg_enabled must be true if track_enabled is true.")

    def __str__(self):
        s = ["ConfigImageProcessor"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if k == 'cfg_delta':
                tmp = str(v)
                v = tmp.replace("\n", "\n\t")
            if i < len(self.__dict__)-1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


class ConfigImageProcessorFactory:
    @staticmethod
    def default_config(
            channels: list[LEDType] | None = None,
            channels_seg: list[LEDType] | None = None
    ) -> ConfigImageProcessor:
        evomachine_dir = _get_evomachine_dir()
        default_channels = [LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM, LEDType.LED_645_NM]
        cfg_delta = delta.config.Config.default("mothermachine")
        cfg_delta.whole_frame_drift = False
        # cfg_delta.target_size_rois = (1024, 1024)
        # cfg_delta.target_size_rois = (1600, 1600)
        cfg_delta.target_size_rois = (200, 800)
        cfg_delta.tolerable_resizing_rois = 0
        # cfg_delta.model_file_rois = evomachine_dir.parent / "delta_models/evo_roi_2024-05-08.keras"
        # cfg_delta.model_file_rois = evomachine_dir.parent / "delta_models/evo_roi_M9_2024-12-10.keras"
        cfg_delta.model_file_rois = evomachine_dir.parent / "delta_models/evo_roi_mixed_200x800_CKS5_2025-02-11.keras"  # 2025-11-04
        # cfg_delta.target_size_seg = (64, 256)
        # cfg_delta.model_file_seg = evomachine_dir.parent / "delta_models/evo_seg_2024-06-27.keras"
        cfg_delta.target_size_seg = (64, 512)  # 2025-11-04
        cfg_delta.model_file_seg = evomachine_dir.parent / "delta_models/evo_seg_64x512_kernel5_levels5_2025-05-29_realdata.keras"  # 2025-11-04
        # cfg_delta.model_file_track = evomachine_dir.parent / "delta_models/unet_moma_track.hdf5"
        cfg_delta.target_size_seg = (64, 256)  # 2025-11-04
        cfg_delta.model_file_track = evomachine_dir.parent / "delta_models/evo_track_64x256_2024-08-30.keras"  # 2025-11-04
        return ConfigImageProcessor(
            cfg_delta=cfg_delta,
            channels=default_channels if channels is None else channels,
            channels_seg=default_channels if channels_seg is None else channels_seg,
            roi_enabled=False,
            seg_enabled=False,
            preproc_enabled=False,
        )


@dataclass
class ConfigCRISP:
    averaging: int
    "Number of samples to average."
    led_intensity: int
    "LED intensity of the CRISP device."
    lock_range: float
    "Prevent the axis from moving too far out of focus lock. Value in mm."
    loop_gain: int
    "Adjust to change the responsiveness of CRISP."
    update_rate: int
    "The time in ms to wait between updates to the CRISP trajectory."
    objective_na: float
    "NA of the objective used to calculate dither steps. Can be different from the actual objective NA."

    user_input: bool | None = True
    "Ask for user input before configuring and locking CRISP autofocus."
    min_snr: int | None = 2
    "Minimum acceptable signal to noise ratio measured during calibration."
    min_error: int | None = 100
    "Minimum acceptable absolute error measured during calibration."
    pause_long: int | None = 5
    "Value of long pause in s between CRISP configuration steps."
    pause_short: int | None = 1
    "Value of short pause in s between CRISP configuration steps."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) -> int | float | bool | None:
        if attr_name == 'lock_range' or attr_name == 'objective_na':
            return float(attr_value_str)
        else:
            return int(attr_value_str)

    @staticmethod
    def attr_is_valid(attr_name: str, attr_value) -> bool:
        if attr_name == 'averaging':
            return isinstance(attr_value, int) and (attr_value >= 0) and (attr_value < 100)
        elif attr_name == 'led_intensity':
            return isinstance(attr_value, int) and (attr_value > 1) and (attr_value <= 100)
        elif attr_name == 'loop_gain':
            return isinstance(attr_value, int) and (attr_value >= 1) and (attr_value <= 100)
        elif attr_name == 'lock_range':
            return isinstance(attr_value, float) and (attr_value > 0) and (attr_value < 1)
        elif attr_name == 'objective_na':
            return isinstance(attr_value, float) and (attr_value > 0) and (attr_value < 10.0)
        elif attr_name == 'update_rate':
            return isinstance(attr_value, int)
        else:
            return False

    def __post_init__(self):
        if not self.attr_is_valid('led_intensity', self.led_intensity):
            raise TypeError(f"led_intensity must be an integer in the range (0,100]. Provided {self.led_intensity}.")
        if not self.attr_is_valid('loop_gain', self.loop_gain):
            raise TypeError(f"loop_gain must be an integer in the range [1,10]. Provided {self.loop_gain}.")
        if not self.attr_is_valid('averaging', self.averaging):
            raise TypeError(f"averaging must be an integer in the range [0,Inf). Provided {self.averaging}.")
        if not self.attr_is_valid('update_rate', self.update_rate):
            raise TypeError(f"update_rate must be an integer in the range [0,Inf). Provided {self.update_rate}.")
        if not self.attr_is_valid('lock_range', self.lock_range):
            raise TypeError(f"lock_range may lead to objective crashing into the sample. Provided {self.lock_range}.")

    def __str__(self):
        s = ["ConfigCRISP"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)

    def copy(self):
        return ConfigCRISP(**self.__dict__)


class ConfigCRISPFactory:
    """
    Compatibility factory for legacy CRISP configuration callers.

    New ASI Tiger autofocus code should use
    evomachine.bindings.asitiger.autofocus.TigerAutofocusConfigFactory.
    """

    @staticmethod
    def default_config() -> ConfigCRISP:
        """
        Return the legacy default CRISP configuration.

        Parameters
        ----------
        None

        Returns
        -------
        ConfigCRISP
            Legacy CRISP configuration with compatibility pause fields.
        """
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory

        tiger_config = TigerAutofocusConfigFactory.default_config()
        return ConfigCRISP(**tiger_config.__dict__)

    @staticmethod
    def default_oil_config() -> ConfigCRISP:
        """
        Return the legacy oil-objective CRISP configuration.

        Parameters
        ----------
        None

        Returns
        -------
        ConfigCRISP
            Legacy oil-objective CRISP configuration with compatibility pause
            fields.
        """
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory

        tiger_config = TigerAutofocusConfigFactory.default_oil_config()
        return ConfigCRISP(**tiger_config.__dict__)


@dataclass
class SoftwareFocusConfigNew:
    """Configuration object for the new software focus acquisition path."""

    focus_frames: list["FrameMetaData"]
    "Frame acquisition metadata captured at each Z coordinate."
    acquisition_settings: "FrameAcquisitionSettings | None"
    "Optional frame acquisition settings used for software focus captures."
    rel_range: int
    "Relative Z scan range around the current stage Z coordinate."
    step_size: int
    "Z scan step size."
    algorithm: FocusAlgorithmType
    "Focus scoring algorithm."
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    "Keyword arguments forwarded to the focus algorithm factory."
    cropping_box: CroppingBox | list[CroppingBox] | None = None
    "Optional crop or crops used for scoring."

    def __post_init__(self) -> None:
        """
        Validate new software focus configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.focus_frames, list) or not self.focus_frames:
            raise TypeError("SoftwareFocusConfigNew: focus_frames must be a non-empty list[FrameMetaData].")
        if not all(isinstance(frame, FrameMetaData) for frame in self.focus_frames):
            raise TypeError("SoftwareFocusConfigNew: every focus frame must be FrameMetaData.")
        if self.acquisition_settings is not None:
            from evomachine.acquisition import FrameAcquisitionSettings

            if not isinstance(self.acquisition_settings, FrameAcquisitionSettings):
                raise TypeError(
                    "SoftwareFocusConfigNew: acquisition_settings must be FrameAcquisitionSettings or None."
                )
        if not isinstance(self.rel_range, int) or isinstance(self.rel_range, bool) or not 0 < self.rel_range < 2000:
            raise TypeError("SoftwareFocusConfigNew: rel_range must be int in (0, 2000).")
        if not isinstance(self.step_size, int) or isinstance(self.step_size, bool) or not 0 < self.step_size <= self.rel_range:
            raise TypeError("SoftwareFocusConfigNew: step_size must be int in [1, rel_range].")
        if not isinstance(self.algorithm, FocusAlgorithmType):
            raise TypeError("SoftwareFocusConfigNew: algorithm must be FocusAlgorithmType.")
        if not isinstance(self.algorithm_kwargs, dict):
            raise TypeError("SoftwareFocusConfigNew: algorithm_kwargs must be dict[str, Any].")
        if not all(isinstance(key, str) for key in self.algorithm_kwargs):
            raise TypeError("SoftwareFocusConfigNew: algorithm_kwargs keys must be str.")
        self.cropping_box = self._validate_cropping_box(self.cropping_box)

    @staticmethod
    def _validate_cropping_box(
            cropping_box: CroppingBox | list[CroppingBox] | None,
    ) -> CroppingBox | list[CroppingBox] | None:
        """
        Validate a scoring crop selection.

        Parameters
        ----------
        cropping_box
            None, one CroppingBox, or a non-empty list of CroppingBox objects.

        Returns
        -------
        CroppingBox | list[CroppingBox] | None
            Validated crop selection.
        """
        if cropping_box is None or isinstance(cropping_box, CroppingBox):
            return cropping_box
        if isinstance(cropping_box, list):
            if not cropping_box:
                raise ValueError("SoftwareFocusConfigNew: cropping_box list must not be empty.")
            if not all(isinstance(box, CroppingBox) for box in cropping_box):
                raise TypeError("SoftwareFocusConfigNew: every cropping_box entry must be CroppingBox.")
            return list(cropping_box)
        raise TypeError("SoftwareFocusConfigNew: cropping_box must be CroppingBox, list[CroppingBox], or None.")

    def copy(self) -> "SoftwareFocusConfigNew":
        """
        Return a shallow copy of this configuration.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocusConfigNew
            New config object with the same field values.
        """
        return SoftwareFocusConfigNew(**self.__dict__)


class SoftwareFocusConfigNewFactory:
    """Factory for new software focus configuration defaults."""

    @staticmethod
    def default_config() -> SoftwareFocusConfigNew:
        """
        Return the default new software focus configuration.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocusConfigNew
            Default config for new acquisition-backed software focus.
        """
        return SoftwareFocusConfigNew(
            focus_frames=[
                FrameMetaData(
                    frame_id=-1,
                    leds={LEDType.LED_450_NM: 29},
                    filter_wheel=None,
                    exposure=200,
                )
            ],
            acquisition_settings=None,
            rel_range=50,
            step_size=5,
            algorithm=FocusAlgorithmType.STEEL,
            algorithm_kwargs={"rowshift": 25, "colshift": 0},
            cropping_box=CroppingBox(xtl=200, xbr=3000, ytl=300, ybr=2900),
        )


@dataclass
class SoftwareFocusConfig:
    """Configuration object for software focus scoring and scan parameters."""

    exposure_time: float | int
    "Exposure time for focusing in ms."
    focus_channel: LEDType
    "LED channel to use while scanning. See LEDType for available channels."
    rel_range: int
    "Relative range for Z-movement of stage in 1/10 μm, e.g., stage will move current_position+-rel_range."
    step_size: int
    "Step size for Z-movement of stage in 1/10 μm, e.g., step_size=1 -> stage moves in 0.1 μm."
    brightness: float
    "Brightness value in (1,29) for LED brightness during focus."

    algorithm: FocusAlgorithmType = FocusAlgorithmType.STEEL
    "Algorithm used to focus. See FocusAlgorithmType for available algorithms."
    rowshift_px: int = 25
    "Focus algorithm parameter. See software_focus.py."
    colshift_px: int = 0
    "Focus algorithm parameter. See software_focus.py. Note: Can also use 50px (trench length) here."
    cropping_box: CroppingBox | None = None
    "Box to crop out image area to focus on."
    user_input: bool | None = True
    "Ask for user input before configuring and starting software focus."
    focus_frames: FrameMetaData | list[FrameMetaData] | None = None
    "Frame acquisition settings for future scoring frame capture support."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) \
            -> int | float | bool | FocusAlgorithmType | LEDType | None:
        """
        Convert a GUI string value into the type expected by one config field.

        Parameters
        ----------
        attr_name
            Name of the SoftwareFocusConfig attribute being updated.
        attr_value_str
            String value supplied by a GUI or text configuration source.

        Returns
        -------
        int | float | bool | FocusAlgorithmType | LEDType | None
            Parsed value suitable for validation with attr_is_valid().
        """
        if attr_name == 'exposure_time':
            return float(attr_value_str)
        elif attr_name == 'user_input':
            return bool(attr_value_str)
        elif attr_name == 'algorithm':
            return FocusAlgorithmType.from_string(attr_value_str)
        elif attr_name == 'focus_channel':
            return LEDType(int(attr_value_str))
        elif attr_name == 'brightness':
            return float(attr_value_str)
        else:
            return int(attr_value_str)

    def attr_is_valid(self, attr_name: str, attr_value: Any) -> bool:
        """
        Return whether a candidate value is valid for one config field.

        Parameters
        ----------
        attr_name
            Name of the SoftwareFocusConfig attribute being checked.
        attr_value
            Candidate value for the attribute.

        Returns
        -------
        bool
            True when attr_value satisfies the field validation rules.
        """
        if attr_name == 'exposure_time':
            return (isinstance(attr_value, int) or isinstance(attr_value, float)) and attr_value >= 0.01
        elif attr_name == 'focus_channel':
            return isinstance(attr_value, LEDType)
        elif attr_name == 'rel_range':
            return isinstance(attr_value, int) and (attr_value > 0) and (attr_value < 2000)
        elif attr_name == 'step_size':
            return isinstance(attr_value, int) and (attr_value > 0) and (attr_value <= self.rel_range)
        elif attr_name == 'algorithm':
            return isinstance(attr_value, FocusAlgorithmType)
        elif attr_name == 'brightness':
            return (isinstance(attr_value, float) or isinstance(attr_value, int))\
                and (attr_value >= 0) and (attr_value <= 29)
        elif attr_name == 'user_input':
            return isinstance(attr_value, bool)
        elif attr_name == 'focus_frames':
            return self._focus_frames_are_valid(attr_value)
        else:
            return False

    @staticmethod
    def _focus_frames_are_valid(focus_frames: FrameMetaData | list[FrameMetaData] | None) -> bool:
        """
        Return whether focus frame settings have a supported shape.

        Parameters
        ----------
        focus_frames
            None, one FrameMetaData, or a list of FrameMetaData objects.

        Returns
        -------
        bool
            True when focus_frames is None, a FrameMetaData, or a list of
            FrameMetaData objects.
        """
        if focus_frames is None:
            return True
        if isinstance(focus_frames, FrameMetaData):
            return True
        if isinstance(focus_frames, list):
            return all(isinstance(frame, FrameMetaData) for frame in focus_frames)
        return False

    def __post_init__(self) -> None:
        """
        Validate software focus configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not self.attr_is_valid('step_size', self.step_size):
            raise TypeError(f"step_size must be an int in [1, rel_range={self.rel_range}]. Provided {self.step_size}.")
        if not self.attr_is_valid('rel_range', self.rel_range):
            raise TypeError(f"rel_range must be an integer in the range [1, Inf]. Provided {self.rel_range}.")
        if not self.attr_is_valid('focus_channel', self.focus_channel):
            raise TypeError(f"focus_channel must be a led type.")
        if not self.attr_is_valid('exposure_time', self.exposure_time):
            raise TypeError(f"exposure_time must be an int in [0.01, Inf]. Provided {self.exposure_time}.")
        if not self.attr_is_valid('brightness', self.brightness):
            raise TypeError(f"brightness must be an int or float in [0, 29]. Provided {self.brightness}.")
        if not self.attr_is_valid('algorithm', self.algorithm):
            raise TypeError(f"algorithm must be an instance of FocusAlgorithmType. Provided {self.algorithm}.")
        if not self.attr_is_valid('focus_frames', self.focus_frames):
            raise TypeError("focus_frames must be None, FrameMetaData, or list[FrameMetaData].")

    def copy(self) -> SoftwareFocusConfig:
        """
        Return a shallow copy of this software focus configuration.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocusConfig
            New config object with the same field values.
        """
        return SoftwareFocusConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> SoftwareFocusConfig:
        """
        Return a copy with selected fields updated and validated.

        Parameters
        ----------
        **kwargs
            Attribute names and replacement values for the returned config.

        Returns
        -------
        SoftwareFocusConfig
            New validated config with the requested updates applied.
        """
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"SoftwareFocusConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return SoftwareFocusConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> SoftwareFocusConfig:
        """
        Return a copy updated from a mapping of field values.

        Parameters
        ----------
        updates
            Mapping from SoftwareFocusConfig field names to replacement values.

        Returns
        -------
        SoftwareFocusConfig
            New validated config with the requested updates applied.
        """
        if not isinstance(updates, dict):
            raise TypeError(f"SoftwareFocusConfig.update_from_mapping: updates must be dict, received {type(updates)}.")
        return self.updated(**updates)

    def __str__(self) -> str:
        """
        Return a multi-line display string for this config.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Human-readable configuration summary.
        """
        s = ["SoftwareFocusConfig"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


ConfigFocus = SoftwareFocusConfig


class SoftwareFocusConfigFactory:
    """Factory for default SoftwareFocusConfig instances."""

    @staticmethod
    def default_config() -> SoftwareFocusConfig:
        """
        Return the default software focus configuration.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocusConfig
            Default focus configuration for the current application defaults.
        """
        return SoftwareFocusConfig(
            exposure_time=200,
            focus_channel=LEDType.LED_450_NM,
            brightness=29,
            rel_range=50,
            step_size=5,
            cropping_box=CroppingBox(xtl=200, xbr=3000, ytl=300, ybr=2900),  # Note: must be changed for other chips
        )


class ConfigFocusFactory(SoftwareFocusConfigFactory):
    """Compatibility alias factory for legacy ConfigFocus callers."""


@dataclass
class ConfigCamera:
    objective: ObjectiveConfigType
    "Objective type. See ObjectiveType."
    image: ImageConfigType
    "Image configuration. See ImageConfig."
    focus: SoftwareFocusConfig
    "Focus configuration. See SoftwareFocusConfig."
    autofocus: ConfigCRISP
    "Autofocus configuration. See ConfigCRISP."
    leds: list[LEDType]
    "Available LED channels. See LEDType."
    filters: list[FilterWheelType]
    "Available filter wheels. See FilterWheelType."
    path_to_save: Path
    "Path to save images."
    default_exposure_time: float | int = 200
    "Default exposure time in ms."
    default_focus_channel_id: int = 0
    "Default LED channel index in self.leds."
    cam_pxl_size: float = 6.5
    "Pixel size of camera in μm."

    def copy(self):
        return ConfigCamera(**self.__dict__)

    @property
    def pxl_size(self) -> float:
        """
        Returns the size of one pixel in micrometers.

        Returns
        -------
        pixel_size: float
        """
        return self.cam_pxl_size / self.objective.mag  # in μm

    @property
    def fov_size(self) -> float:
        """
        Returns the size of the field of view in micrometers.

        Returns
        -------
        fov_size: float
        """
        return self.cam_pxl_size / self.objective.mag * self.image.pxl_vert  # in μm

    def __post_init__(self):
        if not isinstance(self.objective, ObjectiveConfigType):
            raise TypeError(f"objective must be a ObjectiveType object. Provided {self.objective}.")
        if not isinstance(self.image, ImageConfigType):
            raise TypeError(f"image must be a ImageConfigType object. Provided {self.image}.")
        if not isinstance(self.focus, SoftwareFocusConfig):
            raise TypeError(f"focus must be a SoftwareFocusConfig object. Provided {self.focus}.")
        if not isinstance(self.autofocus, ConfigCRISP):
            raise TypeError(f"autofocus must be a ConfigCRISP object. Provided {self.autofocus}.")
        if not (isinstance(self.leds, list) and all(isinstance(led, LEDType) for led in self.leds))\
                or len(self.leds) == 0 or LEDType.NO_LED not in self.leds:
            raise ConfigError("Invalid LED list.", ErrorCode.ERROR_CONFIG)
        if not (isinstance(self.filters, list) and all(isinstance(f, FilterWheelType) for f in self.filters))\
                or len(self.filters) == 0:
            raise ConfigError("Invalid filter list.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.path_to_save, Path) and self.path_to_save.exists():
            raise ConfigError("Invalid path_to_save.", ErrorCode.ERROR_CONFIG)
        if not self.image.pxl_vert == self.image.pxl_horiz:
            raise ConfigError(f"Currently limited to square images.", ErrorCode.ERROR_FOCUS_CONFIG)
        if ((not (isinstance(self.default_exposure_time, int) or isinstance(self.default_exposure_time, float))) or
                self.default_exposure_time <= 0):
            raise TypeError(f"Invalid default_exposure_time {self.default_exposure_time}.")
        if not (isinstance(self.default_focus_channel_id, int) and 0 <= self.default_focus_channel_id < len(self.leds)):
            raise TypeError(f"Invalid default_focus_channel_id {self.default_focus_channel_id}.")


class ConfigCameraFactory:
    @staticmethod
    def get_available_leds() -> list[LEDType]:
        if _use_sync_board():
            return [LEDType.NO_LED, LEDType.LED_385_NM, LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM,
                    LEDType.LED_645_NM, LEDType.LED_OVERHEAD, LEDType.LED_OVERHEAD_TIGER]
        else:
            return [LEDType.NO_LED, LEDType.LED_405_NM, LEDType.LED_450_NM, LEDType.LED_505_NM, LEDType.LED_538_NM]

    @staticmethod
    def get_available_filters() -> list[FilterWheelType]:
        return [FilterWheelType.FILTER, FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_527nm,
                FilterWheelType.FILTER_592nm, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER]

    @staticmethod
    def default_oil_config(path_to_save: Path | None = None) -> ConfigCamera:
        evomachine_dir = _get_evomachine_dir()
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_oil(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_oil_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=ConfigCameraFactory.get_available_filters(), # [FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=evomachine_dir.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

    @staticmethod
    def default_air_config(path_to_save: Path | None = None) -> ConfigCamera:
        evomachine_dir = _get_evomachine_dir()
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_air(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=ConfigCameraFactory.get_available_filters(), # [FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=evomachine_dir.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )


@dataclass
class FileNameConfig:
    """Configuration object for FileManager output paths and filename patterns."""

    directory: Path
    filename_pattern: str = "{channel}_P{position_id}_X{x}_Y{y}_Z{z}_F{filter_wheel}_{timestamp}"
    create_directory: bool = True

    def __post_init__(self) -> None:
        """
        Validate file naming configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)
        if not isinstance(self.directory, Path):
            raise TypeError(f"FileNameConfig: directory must be Path or str, received {type(self.directory)}.")
        if not isinstance(self.filename_pattern, str):
            raise TypeError(
                f"FileNameConfig: filename_pattern must be str, received {type(self.filename_pattern)}."
            )
        if self.filename_pattern == "":
            raise ValueError("FileNameConfig: filename_pattern must not be empty.")
        if not isinstance(self.create_directory, bool):
            raise TypeError(
                f"FileNameConfig: create_directory must be bool, received {type(self.create_directory)}."
            )

    def updated(self, **kwargs: Any) -> FileNameConfig:
        """
        Return a copy with selected fields updated and validated.

        Parameters
        ----------
        **kwargs
            Attribute names and replacement values for the returned config.

        Returns
        -------
        FileNameConfig
            New validated file name configuration.
        """
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"FileNameConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return FileNameConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> FileNameConfig:
        """
        Return a copy updated from a mapping of field values.

        Parameters
        ----------
        updates
            Mapping from FileNameConfig field names to replacement values.

        Returns
        -------
        FileNameConfig
            New validated file name configuration.
        """
        if not isinstance(updates, dict):
            raise TypeError(f"FileNameConfig.update_from_mapping: updates must be dict, received {type(updates)}.")
        return self.updated(**updates)


@dataclass
class FrameMetaData:
    """
    Store frame acquisition settings and metadata associated with one image frame.
    """
    frame_id: int
    "Frame identifier used to associate saved files and callbacks."
    # Frame settings
    leds: dict[LEDType, BrightnessType] | None
    "Dict with LED/brightness to actuate simultaneously for the frame."
    filter_wheel: FilterWheelType | None
    "Filter wheel type."
    exposure: ExposureType | None
    "Camera exposure."
    dmd_pattern: np.ndarray | None = None
    "Optional per-frame DMD pattern to display before acquisition."
    position_id: int = UNKNOWN_POSITION_ID
    "Registered position ID for the frame, or UNKNOWN_POSITION_ID when unknown."
    coordinate: Coordinate | None = None
    "Stage coordinate recorded for the frame, if known."
    creation_time: datetime = field(default_factory=datetime.now)
    "Time when this frame configuration was created."
    execution_time: datetime | None = None
    "Time when this frame was acquired or saved, if known."
    callback_id: int | None = None
    "Callback identifier filled by the automaton when available."
    additional_metadata: dict[str, Any] = field(default_factory=dict)
    "Additional JSON-serializable metadata for filename generation and saved metadata."

    def __post_init__(self) -> None:
        """
        Validate frame metadata after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool):
            raise TypeError(f"FrameMetaData: frame_id must be int, received {type(self.frame_id)}.")
        if self.leds is not None:
            if not isinstance(self.leds, dict):
                raise TypeError(
                    f"FrameMetaData: leds must be dict[LEDType, BrightnessType] or None, received {type(self.leds)}."
                )
            for led_type, brightness in self.leds.items():
                if not isinstance(led_type, LEDType):
                    raise TypeError(f"FrameMetaData: LED keys must be LEDType, received {type(led_type)}.")
                if not isinstance(brightness, int | float):
                    raise TypeError(f"FrameMetaData: LED brightness must be numeric, received {type(brightness)}.")
                if not 0 <= float(brightness) <= 100:
                    raise ValueError(f"FrameMetaData: LED brightness must be in [0, 100], received {brightness}.")
        if self.filter_wheel is not None and not isinstance(self.filter_wheel, FilterWheelType):
            raise TypeError(
                f"FrameMetaData: filter_wheel must be FilterWheelType or None, received {type(self.filter_wheel)}."
            )
        if self.exposure is not None:
            if not isinstance(self.exposure, int | float):
                raise TypeError(f"FrameMetaData: exposure must be numeric or None, received {type(self.exposure)}.")
            if not 1 <= float(self.exposure) <= 1000:
                raise ValueError(f"FrameMetaData: exposure must be in [1, 1000], received {self.exposure}.")
        if self.dmd_pattern is not None and not isinstance(self.dmd_pattern, np.ndarray):
            raise TypeError(
                f"FrameMetaData: dmd_pattern must be np.ndarray or None, received {type(self.dmd_pattern)}."
            )
        if not isinstance(self.position_id, int) or isinstance(self.position_id, bool):
            raise TypeError(f"FrameMetaData: position_id must be int, received {type(self.position_id)}.")
        if self.coordinate is not None and not isinstance(self.coordinate, Coordinate):
            raise TypeError(f"FrameMetaData: coordinate must be Coordinate or None, received {type(self.coordinate)}.")
        if not isinstance(self.creation_time, datetime):
            raise TypeError(f"FrameMetaData: creation_time must be datetime, received {type(self.creation_time)}.")
        if self.execution_time is not None and not isinstance(self.execution_time, datetime):
            raise TypeError(
                f"FrameMetaData: execution_time must be datetime or None, received {type(self.execution_time)}."
            )
        if self.callback_id is not None and (not isinstance(self.callback_id, int) or isinstance(self.callback_id, bool)):
            raise TypeError(f"FrameMetaData: callback_id must be int or None, received {type(self.callback_id)}.")
        if not isinstance(self.additional_metadata, dict):
            raise TypeError(
                f"FrameMetaData: additional_metadata must be dict[str, Any], "
                f"received {type(self.additional_metadata)}."
            )
        if not all(isinstance(key, str) for key in self.additional_metadata):
            raise TypeError("FrameMetaData: additional_metadata keys must be str.")

    def to_metadata_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable metadata dictionary for this frame.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, Any]
            JSON-serializable representation of the frame metadata.
        """
        metadata = {
            "frame_id": self.frame_id,
            "callback_id": self.callback_id,
            "leds": None if self.leds is None else {
                led_type.name: {
                    "value": led_type.value,
                    "brightness": float(brightness),
                }
                for led_type, brightness in self.leds.items()
            },
            "filter_wheel": self._enum_to_metadata(self.filter_wheel),
            "exposure": self.exposure,
            "dmd_pattern": self._dmd_pattern_to_metadata(),
            "position_id": self.position_id,
            "coordinate": None if self.coordinate is None else self.coordinate.to_dict(),
            "creation_time": self.creation_time.isoformat(),
            "execution_time": None if self.execution_time is None else self.execution_time.isoformat(),
            "additional_metadata": self.additional_metadata,
        }
        try:
            json.dumps(metadata)
        except TypeError as error:
            raise TypeError("FrameMetaData.to_metadata_dict: metadata must be JSON serializable.") from error
        return metadata

    def __str__(self) -> str:
        """
        Return a readable summary of this frame metadata.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Human-readable metadata summary. The DMD pattern is represented by
            presence, shape, and dtype rather than by array contents.
        """
        dmd_pattern = self._dmd_pattern_to_metadata()
        values = {
            "frame_id": self.frame_id,
            "leds": self.leds,
            "filter_wheel": self.filter_wheel,
            "exposure": self.exposure,
            "dmd_pattern": dmd_pattern,
            "position_id": self.position_id,
            "coordinate": self.coordinate,
            "creation_time": self.creation_time,
            "execution_time": self.execution_time,
            "callback_id": self.callback_id,
            "additional_metadata": self.additional_metadata,
        }
        summary = ["FrameMetaData"]
        for index, (key, value) in enumerate(values.items()):
            prefix = " └─ " if index == len(values) - 1 else " ├─ "
            summary.append(f"{prefix}{key}: {value}")
        return "\n".join(summary)

    @staticmethod
    def _enum_to_metadata(value: EvoType | None) -> dict[str, Any] | None:
        """
        Return a JSON-serializable enum representation.

        Parameters
        ----------
        value
            EvoType enum value or None.

        Returns
        -------
        dict[str, Any] | None
            Enum name and value, or None.
        """
        if value is None:
            return None
        return {"name": value.name, "value": value.value}

    def _dmd_pattern_to_metadata(self) -> dict[str, Any] | None:
        """
        Return a compact metadata summary for the DMD pattern.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, Any] | None
            None when no DMD pattern is configured, otherwise a JSON-safe
            summary containing presence, shape, and dtype.
        """
        if self.dmd_pattern is None:
            return None
        return {
            "present": True,
            "shape": list(self.dmd_pattern.shape),
            "dtype": str(self.dmd_pattern.dtype),
        }


class FrameMetaDataFactory:
    """Factory for common FrameMetaData instances with generated frame IDs."""

    _next_frame_id = 0

    @classmethod
    def reset_counter(cls, start: int = 0) -> None:
        """
        Reset the generated frame ID counter.

        Parameters
        ----------
        start
            Integer frame ID to use for the next generated frame.

        Returns
        -------
        None
        """
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(f"FrameMetaDataFactory.reset_counter: start must be int, received {type(start)}.")
        cls._next_frame_id = start

    @classmethod
    def default(
            cls,
            leds: dict[LEDType, BrightnessType] | None = None,
            filter_wheel: FilterWheelType | None = None,
            exposure: ExposureType | None = None,
            dmd_pattern: np.ndarray | None = None,
            position_id: int = UNKNOWN_POSITION_ID,
            coordinate: Coordinate | None = None,
            frame_id: int | None = None,
            callback_id: int | None = None,
            additional_metadata: dict[str, Any] | None = None,
    ) -> FrameMetaData:
        """
        Return a FrameMetaData with common defaults.

        Parameters
        ----------
        leds
            Optional LED brightness settings.
        filter_wheel
            Optional filter wheel setting.
        exposure
            Optional exposure setting.
        dmd_pattern
            Optional per-frame DMD pattern displayed before acquisition.
        position_id
            Registered position ID or UNKNOWN_POSITION_ID.
        coordinate
            Optional stage coordinate associated with the frame.
        frame_id
            Optional explicit frame ID. When None, the factory counter is used.
        callback_id
            Optional callback ID associated with the frame.
        additional_metadata
            Optional extra metadata mapping for filenames and saved metadata.

        Returns
        -------
        FrameMetaData
            Validated frame metadata.
        """
        if frame_id is None:
            frame_id = cls._next_frame_id
            cls._next_frame_id += 1
        return FrameMetaData(
            frame_id=frame_id,
            leds=leds,
            filter_wheel=filter_wheel,
            exposure=exposure,
            dmd_pattern=dmd_pattern,
            position_id=position_id,
            coordinate=coordinate,
            callback_id=callback_id,
            additional_metadata={} if additional_metadata is None else additional_metadata,
        )
    

@dataclass
class Frame:
    """Store one or more acquired image arrays with their frame metadata."""

    frame_metadata: list[FrameMetaData]
    "Frame metadata entries corresponding to the leading axis of array."
    array: np.ndarray
    "Image data with shape (len(frame_metadata), height, width)."
    saved_paths: list[Path | None] = field(default_factory=list)
    "Optional saved output paths corresponding to the leading axis of array."

    def __post_init__(self) -> None:
        """
        Validate acquired frame data after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.frame_metadata, list):
            raise TypeError(f"Frame: frame_metadata must be list[FrameMetaData], received {type(self.frame_metadata)}.")
        if not self.frame_metadata:
            raise ValueError("Frame: frame_metadata must not be empty.")
        if not all(isinstance(metadata, FrameMetaData) for metadata in self.frame_metadata):
            raise TypeError("Frame: every frame_metadata entry must be FrameMetaData.")
        if not isinstance(self.array, np.ndarray):
            raise TypeError(f"Frame: array must be np.ndarray, received {type(self.array)}.")
        if self.array.ndim != 3:
            raise ValueError(f"Frame: array must be 3D with shape (n_frames, height, width), received {self.array.shape}.")
        if self.array.shape[0] != len(self.frame_metadata):
            raise ValueError(
                f"Frame: leading array dimension {self.array.shape[0]} must match "
                f"{len(self.frame_metadata)} metadata entries."
            )
        if self.saved_paths is None:
            self.saved_paths = [None] * len(self.frame_metadata)
        if not isinstance(self.saved_paths, list):
            raise TypeError(f"Frame: saved_paths must be list[Path | None], received {type(self.saved_paths)}.")
        if len(self.saved_paths) != len(self.frame_metadata):
            raise ValueError("Frame: saved_paths length must match frame_metadata length.")
        if not all(path is None or isinstance(path, Path) for path in self.saved_paths):
            raise TypeError("Frame: saved_paths entries must be Path or None.")

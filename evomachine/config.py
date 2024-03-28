import copy
from dataclasses import dataclass
from enum import Enum, auto
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# from delta import utils
import delta

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.evotypes import FilterWheelType, FocusAlgorithmType, LEDType, ImageConfigType, ObjectiveConfigType, \
    ImageConfigTypeFactory, ObjectiveConfigTypeFactory

# DeLTA lib install directory
EVOMACHINE_DIR: Path = Path(__file__).parent

# Switch between pygame dmd.py and dmd_socket.py
USE_DMD_SOCKET: bool = True

# Switch between ASI Tiger LEDs and SyncBoard
USE_SYNC_BOARD: bool = True

# EVO_FORMATTER = logging.Formatter('--->\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n<---')
EVO_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
EVO_LOGGING_LEVEL = logging.INFO
EVO_GUI_LOGGING_LEVEL = logging.DEBUG


def get_logger(name: str, is_gui: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(EVO_LOGGING_LEVEL if not is_gui else EVO_GUI_LOGGING_LEVEL)
    handler = logging.StreamHandler()
    handler.setFormatter(EVO_FORMATTER)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class ConfigImageProcessor:
    cfg_delta: delta.config.Config
    "Delta configuration object."
    channels: List[LEDType]
    "List of channels to be imaged. Used for taking reference frames."
    channel_seg: LEDType
    "Channel used for segmentation."
    channel_rot: LEDType
    "Channel used for rotation identification."
    channel_roi: LEDType
    "Channel used for region-of-interest identification."
    crop_out_ROI: bool = True
    "Apply ROI segmentation to overlapping image portions with size of ROI segmentation model."
    use_track_RT: bool = False
    "Use special tracking function for tracking in trenches."
    image_processing_verbosity: int = 0
    "Lowest verbosity is 0."

    @property
    def channel_to_index(self) -> Dict[LEDType, int]:
        return {c: i for i, c in enumerate(self.channels)}

    def __post_init__(self):
        if not isinstance(self.cfg_delta, delta.config.Config):
            raise TypeError("cfg_delta must be a delta.config.Config object.")
        if not (isinstance(self.channels, list) and all(isinstance(channel, LEDType) for channel in self.channels))\
                or len(self.channels) == 0 or LEDType.NO_LED in self.channels:
            raise ConfigError("Invalid channel list.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.channel_seg, LEDType) or self.channel_seg not in self.channels:
            raise TypeError("channel_seg must be a LEDType object in channels.")
        if not isinstance(self.channel_rot, LEDType) or self.channel_rot not in self.channels:
            raise TypeError("channel_rot must be a LEDType object in channels.")
        if not isinstance(self.channel_roi, LEDType) or self.channel_roi not in self.channels:
            raise TypeError("channel_roi must be a LEDType object in channels.")
        if not isinstance(self.crop_out_ROI, bool):
            raise TypeError("crop_out_ROI must be a boolean.")
        if not isinstance(self.use_track_RT, bool):
            raise TypeError("use_track_RT must be a boolean.")
        if not isinstance(self.image_processing_verbosity, int) or self.image_processing_verbosity < 0:
            raise TypeError("image_processing_verbosity must be an integer >= 0.")


class ConfigImageProcessorFactory:
    @staticmethod
    def default_config(channels: Optional[List[LEDType]] = None) -> ConfigImageProcessor:
        default_channels = [LEDType.LED_405_NM, LEDType.LED_450_NM, LEDType.LED_505_NM, LEDType.LED_538_NM]
        return ConfigImageProcessor(
            cfg_delta=delta.config.DEFAULT_CONFIG_MOTHERMACHINE,
            channels=default_channels if channels is None else channels,
            channel_seg=LEDType.LED_450_NM,
            channel_rot=LEDType.LED_450_NM,
            channel_roi=LEDType.LED_450_NM,
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

    user_input: Optional[bool] = True
    "Ask for user input before configuring and locking CRISP autofocus."
    min_snr: Optional[int] = 2
    "Minimum acceptable signal to noise ratio measured during calibration."
    min_error: Optional[int] = 100
    "Minimum acceptable absolute error measured during calibration."
    pause_long: Optional[int] = 5
    "Value of long pause in s between CRISP configuration steps."
    pause_short: Optional[int] = 1
    "Value of short pause in s between CRISP configuration steps."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) -> Union[int, float, bool, None]:
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
            return isinstance(attr_value, float) and (attr_value > 0) and (attr_value < 0.5)
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
        attributes = [
            f"- led_intensity={self.led_intensity}",
            f"- loop_gain={self.loop_gain}",
            f"- averaging={self.averaging}",
            f"- update_rate={self.update_rate}",
            f"- lock_range={self.lock_range:.3f}",
            f"- min_snr={self.min_snr}",
            f"- min_error={self.min_error}",
        ]
        return "\n".join(attributes)

    def copy(self):
        return ConfigCRISP(
            led_intensity=self.led_intensity,
            loop_gain=self.loop_gain,
            averaging=self.averaging,
            update_rate=self.update_rate,
            lock_range=self.lock_range,
            user_input=self.user_input,
            min_snr=self.min_snr,
            min_error=self.min_error,
            pause_long=self.pause_long,
            pause_short=self.pause_short,
        )


class ConfigCRISPFactory:
    @staticmethod
    def default_config() -> ConfigCRISP:
        return ConfigCRISP(
            led_intensity=95,
            loop_gain=5,
            averaging=0,
            update_rate=100,
            lock_range=0.025,
        )


@dataclass
class ConfigFocus:
    exposure_time: Union[float, int]
    "Exposure time for focusing in ms."
    focus_channel: LEDType
    "LED channel to use while scanning. See LEDType for available channels."
    rel_range: int
    "Relative range for Z-movement of stage in 1/10 μm, e.g., stage will move current_position+-rel_range."
    step_size: int
    "Step size for Z-movement of stage in 1/10 μm, e.g., step_size=1 -> stage moves in 0.1 μm."

    algorithm: FocusAlgorithmType = FocusAlgorithmType.STEEL
    "Algorithm used to focus. See FocusAlgorithmType for available algorithms."
    user_input: Optional[bool] = True
    "Ask for user input before configuring and starting software focus."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) \
            -> Union[int, float, bool, FocusAlgorithmType, None, LEDType]:
        if attr_name == 'exposure_time':
            return float(attr_value_str)
        elif attr_name == 'user_input':
            return bool(attr_value_str)
        elif attr_name == 'algorithm':
            return FocusAlgorithmType.from_string(attr_value_str)
        elif attr_name == 'focus_channel':
            return LEDType(int(attr_value_str))
        else:
            return int(attr_value_str)

    def attr_is_valid(self, attr_name: str, attr_value) -> bool:
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
        elif attr_name == 'user_input':
            return isinstance(attr_value, bool)
        else:
            return False

    def __post_init__(self):
        if not self.attr_is_valid('step_size', self.step_size):
            raise TypeError(f"step_size must be an int in [1, rel_range={self.rel_range}]. Provided {self.step_size}.")
        if not self.attr_is_valid('rel_range', self.rel_range):
            raise TypeError(f"rel_range must be an integer in the range [1, Inf]. Provided {self.rel_range}.")
        if not self.attr_is_valid('focus_channel', self.focus_channel):
            raise TypeError(f"focus_channel must be a led type.")
        if not self.attr_is_valid('exposure_time', self.exposure_time):
            raise TypeError(f"exposure_time must be an int in [0.01, Inf]. Provided {self.exposure_time}.")
        if not self.attr_is_valid('algorithm', self.algorithm):
            raise TypeError(f"algorithm must be an instance of FocusAlgorithmType. Provided {self.algorithm}.")

    def copy(self):
        return ConfigFocus(
            exposure_time=self.exposure_time,
            focus_channel=self.focus_channel,
            rel_range=self.rel_range,
            step_size=self.step_size,
            algorithm=self.algorithm,
            user_input=self.user_input,
        )

    def __str__(self):
        attributes = [
            f"- exposure_time={self.exposure_time} ms",
            f"- focus_channel={self.focus_channel} ({LEDType.get_name(self.focus_channel)})",
            f"- rel_range={self.rel_range / 10} μm",
            f"- step_size={self.step_size / 10} μm",
            f"- algorithm={self.algorithm.value} ({FocusAlgorithmType.get_name(self.algorithm.value)})",
        ]
        return "\n".join(attributes)


class ConfigFocusFactory:
    @staticmethod
    def default_config() -> ConfigFocus:
        return ConfigFocus(
            exposure_time=1000,
            focus_channel=LEDType.LED_450_NM,
            rel_range=50,
            step_size=10,
        )


@dataclass
class ConfigCamera:
    objective: ObjectiveConfigType
    "Objective type. See ObjectiveType."
    image: ImageConfigType
    "Image configuration. See ImageConfig."
    focus: ConfigFocus
    "Focus configuration. See ConfigFocus."
    autofocus: ConfigCRISP
    "Autofocus configuration. See ConfigCRISP."
    leds: List[LEDType]
    "Available LED channels. See LEDType."
    filters: List[FilterWheelType]
    "Available filter wheels. See FilterWheelType."
    path_to_save: Path
    "Path to save images."
    default_exposure_time: Union[float, int] = 1000
    "Default exposure time in ms."
    default_focus_channel_id: int = 0
    "Default LED channel index in self.leds."
    cam_pxl_size: float = 6.5
    "Pixel size of camera in μm."

    @property
    def pxl_size(self) -> float:
        return self.cam_pxl_size / self.objective.mag  # in μm

    @property
    def fov_size(self) -> float:
        return self.cam_pxl_size / self.objective.mag * self.image.pxl_vert  # in μm

    def __post_init__(self):
        if not isinstance(self.objective, ObjectiveConfigType):
            raise TypeError(f"objective must be a ObjectiveType object. Provided {self.objective}.")
        if not isinstance(self.image, ImageConfigType):
            raise TypeError(f"image must be a ImageConfigType object. Provided {self.image}.")
        if not isinstance(self.focus, ConfigFocus):
            raise TypeError(f"focus must be a ConfigFocus object. Provided {self.focus}.")
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
    def get_available_leds() -> List[LEDType]:
        if USE_SYNC_BOARD:
            return [LEDType.NO_LED, LEDType.LED_385_NM, LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_560_NM,
                    LEDType.LED_625_NM]
        else:
            return [LEDType.NO_LED, LEDType.LED_405_NM, LEDType.LED_450_NM, LEDType.LED_505_NM, LEDType.LED_538_NM]

    @staticmethod
    def default_oil_config(path_to_save: Optional[Path] = None) -> ConfigCamera:
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_oil(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=[FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=EVOMACHINE_DIR.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

    @staticmethod
    def default_air_config(path_to_save: Optional[Path] = None) -> ConfigCamera:
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_air(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=[FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=EVOMACHINE_DIR.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

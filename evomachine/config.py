from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from delta import utils

from evomachine.exceptions import ConfigError, ErrorCode

# DeLTA lib install directory
EVOMACHINE_DIR = Path(__file__).parent


class ConfigLED(Enum):
    LED_NO_LED = -1
    LED_405_NM = 0
    LED_450_NM = 1
    LED_505_NM = 2
    LED_538_NM = 3

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]

    @classmethod
    def get_name(cls, value_to_find) -> str:
        for member in cls:
            if member.value == value_to_find:
                return str(member.name)
        return ""


class DMDColor(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]

    @classmethod
    def get_name(cls, value_to_find) -> str:
        for member in cls:
            if member.value == value_to_find:
                return str(member.name)
        return ""


@dataclass
class ConfigDevice:
    """Configuration class for the device geometry and scanning paths"""

    num_pos: int
    "Number of camera/stage positions to scan the entire device."
    coord_pos: List[Tuple[int, int]]
    "X and Y coordinates for each position. Must have length num_position."

    num_chan: int
    "Number of channels (wavelengths) used in imaging."

    num_periods: Union[int, None]  # TODO: this should go somewhere else
    "Number of times the device is scanned. Use None for unlimited."

    read_from_disk: bool
    "Specify if images must be read from disk. If yes, provide a path."
    path_to_images: Union[Path, None]
    "Path to folder containing images. File names follow Delta's naming convention."

    path_to_save: Union[Path, None]
    "Path to save images. If triggered manually, path can be specified when calling the function."

    image_processing_verbosity: int
    "Path to folder containing images. File names follow Delta's naming convention."

    tiger_port: Union[str, None]
    "Serial port for asitiger connection."
    def check_config(self):
        if self.num_pos != len(self.coord_pos):
            raise ConfigError("num_pos must match number of X/Y coordinates.",
                              ErrorCode.ERROR_DEVICE_CONFIG.value)
        if self.read_from_disk:
            if self.path_to_images is None or not self.path_to_images.exists():
                raise ConfigError("Must provide a valid path if read_from_disk is True."
                                  "Received {}".format(self.path_to_images),
                                  ErrorCode.ERROR_DEVICE_CONFIG.value)
            delta_reader: utils.XPReader = \
                utils.XPReader(self.path_to_images / "Position{p}Channel{c}Frames{t}.tif")

            if self.num_pos > len(delta_reader.positions):
                raise ConfigError("Found {} positions, but require {}.".format(
                    len(delta_reader.positions), self.num_pos),
                    ErrorCode.ERROR_DEVICE_CONFIG.value)

            if self.num_periods > len(delta_reader.frames):
                raise ConfigError("Found {} periods, but require {}.".format(
                    len(delta_reader.frames), self.num_periods),
                    ErrorCode.ERROR_DEVICE_CONFIG.value)

            if self.num_chan > len(delta_reader.channels):
                raise ConfigError("Found {} channels, but require {}.".format(
                    len(delta_reader.channels), self.num_chan),
                    ErrorCode.ERROR_DEVICE_CONFIG.value)

        if self.path_to_save is None or not self.path_to_save.exists():
            raise ConfigError("Must provide a valid path_to_save or None."
                              "Received {}".format(self.path_to_save),
                              ErrorCode.ERROR_DEVICE_CONFIG.value)


DEVICE_CONFIG_DELTA_SIM = ConfigDevice(
    num_pos=2,
    coord_pos=[(0, 0) for _ in range(2)],
    num_chan=2,
    num_periods=10,
    read_from_disk=True,
    path_to_images=EVOMACHINE_DIR.parent / "tests/data/movie_mothermachine_tif",
    path_to_save=None,
    image_processing_verbosity=1,
    tiger_port=None,
)

test_pos_list = [(-10000, 0, 0), (0, 0, 0), (0, 10000, 0)]
DEVICE_CONFIG_EVO_TEST = ConfigDevice(
    num_pos=len(test_pos_list),
    coord_pos=test_pos_list,
    num_chan=4,
    num_periods=None,
    read_from_disk=False,
    path_to_images=None,
    path_to_save=EVOMACHINE_DIR.parent / "images/DEFAULT",
    image_processing_verbosity=1,
    tiger_port="/dev/ttyUSB0",
)


@dataclass
class ConfigImage:
    pxl_horiz: int
    "Number of pixels in horizontal direction."
    pxl_vert: int
    "Number of pixels in vertical direction."
    pxl_dtype: np.dtype
    "Datatype of images"
    tile_image: Optional[Tuple[int, int]] = (1, 1)
    "Tile images returned from camera."
    crop_out_ROI: Optional[bool] = True
    "Apply ROI segmentation to overlapping image portions with size of ROI segmentation model."
    use_track_RT: Optional[bool] = False

    def check_config(self):
        if self.pxl_horiz <= 0 or self.pxl_vert <= 0:
            raise ConfigError("Number of pixels must be strictly positive.",
                              ErrorCode.ERROR_IMAGE_CONFIG.value)


IMAGE_CONFIG_DELTA_SIM = ConfigImage(
    pxl_horiz=696,
    pxl_vert=520,
    pxl_dtype=np.dtype("float32"),
)

IMAGE_CONFIG_DELTA_BENCH = ConfigImage(
    pxl_horiz=696,
    pxl_vert=520,
    pxl_dtype=np.dtype("float32"),
    tile_image=(1, 5),
    crop_out_ROI=True,
)

IMAGE_CONFIG_EVO_CAM = ConfigImage(
    pxl_horiz=3200,
    pxl_vert=3200,
    pxl_dtype=np.dtype("uint16"),
)

IMAGE_CONFIG_DEFAULT = IMAGE_CONFIG_EVO_CAM


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
    objective_na: float
    "Objective numerical aperture."
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

    def check_config(self):
        if (not isinstance(self.led_intensity, int)) or self.led_intensity <= 0 or self.led_intensity > 100:
            raise ConfigError(f"led_intensity must be an integer in the range (0,100]. Provided {self.led_intensity}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)
        if not isinstance(self.objective_na, float):
            raise ConfigError(f"objective_na must be a floating point number. Provided {self.objective_na}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)
        if (not isinstance(self.loop_gain, int)) or self.loop_gain > 10 or self.loop_gain < 1:
            raise ConfigError(f"loop_gain must be an integer in the range [1,10]. Provided {self.loop_gain}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)
        if (not isinstance(self.averaging, int)) or self.averaging < 0:
            raise ConfigError(f"averaging must be an integer in the range [0,Inf). Provided {self.averaging}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)
        if (not isinstance(self.update_rate, int)) or self.update_rate < 0:
            raise ConfigError(f"update_rate must be an integer in the range [0,Inf). Provided {self.update_rate}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)
        if (not isinstance(self.lock_range, float)) or self.lock_range > 0.1:
            raise ConfigError(f"lock_range may lead to objective crashing into the sample. Provided {self.lock_range}.",
                              ErrorCode.ERROR_CRISP_CONFIG.value)

    def __str__(self):
        attributes = [
            f"- led_intensity={self.led_intensity}",
            f"- objective_na={self.objective_na:.3f}",
            f"- loop_gain={self.loop_gain}",
            f"- averaging={self.averaging}",
            f"- update_rate={self.update_rate}",
            f"- lock_range={self.lock_range:.3f}",
            f"- min_snr={self.min_snr}",
            f"- min_error={self.min_error}",
        ]
        return "\n".join(attributes)


CRISP_CONFIG_DEFAULT = ConfigCRISP(
    led_intensity=80,
    objective_na=0.95,
    loop_gain=5,
    averaging=0,
    update_rate=100,
    lock_range=0.05,
)


@dataclass
class ConfigFocus:
    exposure_time: float
    "Exposure time for focusing in ms."
    focus_channel: int
    "LED channel to use while scanning. See ConfigLED for available channels."
    rel_range: int
    "Relative range for Z-movement of stage in 1/10 μm, e.g., stage will move current_position+-rel_range."
    steps_size: int
    "Step size for Z-movement of stage in 1/10 μm, e.g., steps_size=1 -> stage moves in 0.1 μm."

    user_input: Optional[bool] = True
    "Ask for user input before configuring and starting software focus."

    def check_config(self):
        if (not isinstance(self.steps_size, int)) or self.steps_size <= 0:
            raise ConfigError(f"steps_size must be an integer in the range [1, Inf]. Provided {self.steps_size}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)
        if (not isinstance(self.rel_range, int)) or self.rel_range <= 0:
            raise ConfigError(f"rel_range must be an integer in the range [1, Inf]. Provided {self.rel_range}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)
        if (not isinstance(self.focus_channel, int)) or (self.focus_channel not in ConfigLED.get_all_values()):
            raise ConfigError(f"focus_channel must be an integer in the range "
                              f"[{min(ConfigLED.get_all_values())}, {max(ConfigLED.get_all_values())}]. "
                              f"Provided {self.focus_channel}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)
        if (not isinstance(self.exposure_time, float)) or self.exposure_time < 0.01:
            raise ConfigError(f"exposure_time must be an integer in the range [0.01, Inf]. Provided {self.exposure_time}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)

    def __str__(self):
        attributes = [
            f"- exposure_time={self.exposure_time} ms",
            f"- focus_channel={self.focus_channel} ({ConfigLED.get_name(self.focus_channel)})",
            f"- rel_range={self.rel_range/10} μm",
            f"- steps_size={self.steps_size/10} μm",
        ]
        return "\n".join(attributes)


FOCUS_CONFIG_DEFAULT = ConfigFocus(
    exposure_time=100,
    focus_channel=ConfigLED.LED_538_NM.value,
    rel_range=100,
    steps_size=10,
)


@dataclass
class ConfigObjective:
    numerical_aperture: float
    "Numerical aperture NA=n*sin(theta)."
    is_oil: bool
    "Yes if oil objective otherwise air."
    magnification: int
    "Magnification of objective."

    cam_pxl_size: Optional[float] = 6.5
    "Pixel size of camera in μm."
    cam_img_size: Optional[float] = 3200
    "Number of pixels that camera returns (assuming a square FoV)."

    @property
    def pxl_size(self):
        return self.cam_pxl_size / self.magnification  # in μm

    @property
    def fov_size(self):
        return self.cam_pxl_size / self.magnification * self.cam_img_size  # in μm

    def check_config(self):
        if (not isinstance(self.numerical_aperture, float)) or self.numerical_aperture <= 0:
            raise ConfigError(f"numerical_aperture must be an integer in the range [0, Inf]. "
                              f"Provided {self.numerical_aperture}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)
        if not isinstance(self.is_oil, bool):
            raise ConfigError(f"is_oil must be a boolean. Provided {self.is_oil}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)
        if (not isinstance(self.magnification, int)) or self.magnification <= 1:
            raise ConfigError(f"magnification must be an integer in the range (1, Inf]. "
                              f"Provided {self.magnification}.",
                              ErrorCode.ERROR_FOCUS_CONFIG.value)

    def __str__(self):
        attributes = [
            f"- numerical_aperture={self.numerical_aperture}",
            f"- is_oil={self.is_oil}",
            f"- magnification={self.magnification}x",
        ]
        return "\n".join(attributes)


OBJECTIVE_CONFIG_OIL = ConfigObjective(
    numerical_aperture=1.4,
    is_oil=True,
    magnification=60,
)

OBJECTIVE_CONFIG_AIR = ConfigObjective(
    numerical_aperture=0.95,
    is_oil=False,
    magnification=40,
)

OBJECTIVE_CONFIG_DEFAULT = OBJECTIVE_CONFIG_AIR

# Evomachine configuration
DEBUG_MODE: bool = True
USE_GPU: bool = True

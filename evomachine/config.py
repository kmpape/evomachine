from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from delta import utils

from evomachine.exceptions import ConfigError, ErrorCode

# DeLTA lib install directory
EVOMACHINE_DIR = Path(__file__).parent

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


DEVICE_CONFIG_DELTA_SIM = ConfigDevice(
    num_pos=2,
    coord_pos=[(0, 0) for _ in range(2)],
    num_chan=2,
    num_periods=10,
    read_from_disk=True,
    path_to_images=EVOMACHINE_DIR.parent / "tests/data/movie_mothermachine_tif",
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
        if self.pxl_x <= 0 or self.pxl_y <= 0:
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
    "The time in milliseconds to wait between updates to the CRISP trajectory."

    user_input: Optional[bool] = True
    "Ask for user input before configuring and locking CRISP autofocus."
    min_snr: Optional[int] = 2
    "Minimum acceptable signal to noise ratio measured during calibration."
    min_error: Optional[int] = 100
    "Minimum acceptable absolute error measured during calibration."
    pause_long: Optional[int] = 5
    "Value of long pause in seconds between CRISP configuration steps."
    pause_short: Optional[int] = 1
    "Value of short pause in seconds between CRISP configuration steps."

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


# Evomachine configuration
DEBUG_MODE: bool = True
USE_GPU: bool = True

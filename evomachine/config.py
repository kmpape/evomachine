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


@dataclass
class ConfigImage:
    pxl_horiz: int
    "Number of pixels in horizontal direction."
    pxl_vert: int
    "Number of pixels in vertical direction."
    pxl_dtype: np.dtype
    "Datatype of images"

    def check_config(self):
        if self.pxl_x <= 0 or self.pxl_y <= 0:
            raise ConfigError("Number of pixels must be strictly positive.",
                              ErrorCode.ERROR_IMAGE_CONFIG.value)


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


IMAGE_CONFIG_DELTA_SIM = ConfigImage(
    pxl_horiz=696,
    pxl_vert=520,
    pxl_dtype=np.dtype("float32"),
)


# Evomachine configuration
DEBUG_MODE: bool = True
USE_GPU: bool = True

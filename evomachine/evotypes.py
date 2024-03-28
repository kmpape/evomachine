from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from evomachine.exceptions import ConfigError, ErrorCode


class EvoType(Enum):
    """ Convenience class for Enumerate classes."""
    @classmethod
    def from_string(cls, s: str) -> Union['EvoType', None]:
        for member in cls:
            if str(member.name) == s:
                return member
        return None

    @classmethod
    def get_all(cls) -> List['EvoType']:
        return [member for member in cls]

    @classmethod
    def get_all_values(cls) -> List[int]:
        """ Returns all member values defined in Enum class."""
        return [member.value for member in cls]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """ Returns all member names for members defined in Enum class."""
        return [cls.get_name(member.value) for member in cls]

    @classmethod
    def get_name(cls, value_to_find) -> str:
        """ Returns the variable name for a particular enum value. Returns an empty string if not defined."""
        for member in cls:
            if member.value == value_to_find:
                return str(member.name)
        return ""

    @classmethod
    def get_dict(cls) -> Dict[int, str]:
        """ Returns a value-variable_name dictionary. """
        return {v: cls.get_name(value_to_find=v) for v in cls.get_all_values()}

    def __str__(self):
        return str(self.name)


class AutomatonCommandType(EvoType):
    IMAGE = auto()
    MOVE = auto()
    PROJECT = auto()
    WAIT = auto()
    STOP = auto()
    LIVE_MODE = auto()

    # The types below are used by the GUI
    FOCUS_DATA = auto()
    FOV_DATA = auto()
    INFO_TEXT = auto()
    PROCESS_DATA = auto()
    REF_DATA = auto()
    ROI_DATA = auto()


# class AutomatonQueueDataType(EvoType):
#     FOCUS_DATA = auto()
#     FOCUS_FRAMES = auto()
#     INFO_TEXT = auto()
#     PROCESS_DATA = auto()
#     PROCESSOR_INIT_DATA = auto()


class AxisType(EvoType):
    X = 0
    Y = 1
    Z = 2


class FocusAlgorithmType(EvoType):
    LAPLACIAN_VAR = auto()
    SQUARED_GRAD_AVG = auto()
    STEEL = auto()


class LEDType(EvoType):
    NO_LED = -1
    # LED_405_NM = 0
    # LED_450_NM = 1
    # LED_505_NM = 2
    # LED_538_NM = 3
    LED_385_NM = 0
    LED_450_NM = 1
    LED_515_NM = 2
    LED_560_NM = 3
    LED_625_NM = 4

    # OLD LEDs
    LED_405_NM = 5
    LED_505_NM = 6
    LED_538_NM = 7


class FilterWheelType(EvoType):
    FILTER = 0
    BLOCKING = 1
    NO_FILTER = 2


class DMDDirType(EvoType):
    HORIZ = 0
    VERT = 1


@dataclass
class DMDCalibConfigType:
    channel: LEDType
    "LED type for calibration."
    brightness: int
    "Brightness of LED."
    exposure: Union[float, int]
    "Exposure time for calibration in milliseconds."
    line_width: int
    "Thickness of calibration lines."
    step: int
    "Step size for calibration in pixels."
    delay: Union[int, float]
    "Delay between calbriation steps in seconds."
    start_row: int
    "Start index for rows (DMD coordinates)."
    end_row: int
    "End index for rows (DMD coordinates)."
    start_col: int
    "Start index for columns (DMD coordinates)."
    end_col: int
    "End index for columns (DMD coordinates)."

    def __post_init__(self):
        if not ((0 <= self.start_row) and (self.start_row < self.end_row) and (self.end_row < 2716)):
            raise ValueError("Indices must be within DMD boundaries.")
        if not ((0 <= self.start_col) and (self.start_col < self.end_col) and (self.end_col < 1600)):
            raise ValueError("Indices must be within DMD boundaries.")


class DMDCalibConfigTypeFactory:
    @staticmethod
    def default(channel: LEDType = LEDType.LED_505_NM) -> DMDCalibConfigType:
        """
        This configuration should be used together with a fluorescent slide.

        Parameters
        ----------
        channel: LEDType
            Use this to override default channel.
        Returns
        -------
        cfg: DMDCalibConfigType
        """
        return DMDCalibConfigType(
            channel=channel,
            brightness=100,
            exposure=100,
            line_width=10,
            step=400,
            delay=0.5,
            start_row=750,
            end_row=2250,
            start_col=80,
            end_col=1550,
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
    def shape(self) -> Tuple[int, int]:
        return self.pxl_vert, self.pxl_horiz

    def __post_init__(self):
        if not isinstance(self.pxl_horiz, int) or not self.pxl_horiz > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_horiz: {self.pxl_horiz}")
        if not isinstance(self.pxl_vert, int) or not self.pxl_vert > 0:
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_vert: {self.pxl_vert}")
        if not isinstance(self.pxl_dtype, np.dtype):
            raise ConfigError(error_code=ErrorCode.ERROR_IMAGE_CONFIG, message=f"Invalid pxl_dtype: {self.pxl_dtype}")


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
    descr: Optional[str] = "UNKNOWN OBJECTIVE"
    "Optional description of objective."

    def __post_init__(self):
        if not isinstance(self.na, float) or not 0 < self.na:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid numerical_aperture: {self.na}")
        if not isinstance(self.mag, int) or not self.mag > 0:
            raise ConfigError(error_code=ErrorCode.ERROR, message=f"Invalid magnification: {self.mag}")


class ObjectiveConfigTypeFactory:
    @staticmethod
    def default_oil() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=1.4, mag=60, descr="Nikon Plan Apo λ 60x/1.4 Oil")

    @staticmethod
    def default_air() -> ObjectiveConfigType:
        return ObjectiveConfigType(na=0.95, mag=40, descr="Nikon Plan Fluor 40x/0.95")

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from typing import Annotated


from evomachine.exceptions import ConfigError, ErrorCode


BrightnessType = Annotated[float, "0 <= value <= 100"]
"""LED or projection brightness as a percentage-like float in the range 0 to 100."""

ExposureType = Annotated[float, "1 <= value in ms <= 1000"]
"""Camera exposure duration in milliseconds for imaging and calibration settings."""

PositiveScalingType = Annotated[float, "value > 0"]
"""Positive scale factor used for relative movement and other multiplicative settings."""

UNKNOWN_FOV_ID = -1
"""FoV ID used when a frame or stage location is not associated with a registered FoV."""


class EvoType(Enum):
    """Base enum with helper methods used by evomachine UI, config, and command types."""
    @classmethod
    def from_string(cls, s: str) -> 'EvoType' | None:
        for member in cls:
            if str(member.name) == s:
                return member
        return None

    @classmethod
    def from_flag(cls, status_flag: str) -> 'EvoType':
        return cls(status_flag)

    @classmethod
    def get_all(cls) -> list['EvoType']:
        return [member for member in cls]

    @classmethod
    def get_all_values(cls) -> list[int]:
        """ Returns all member values defined in Enum class."""
        return [member.value for member in cls]

    @classmethod
    def get_all_names(cls) -> list[str]:
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
    def get_dict(cls) -> dict[int, str]:
        """ Returns a value-variable_name dictionary. """
        return {v: cls.get_name(value_to_find=v) for v in cls.get_all_values()}

    def __str__(self):
        return str(self.name)


class MagnetModeType(EvoType):
    """Magnet control mode used when commanding current-driven or field-driven magnet settings."""

    CURRENT_SET = auto()
    FIELD_SET = auto()


class ChamberOrientationType(EvoType):
    """Physical chamber orientation used by image processing and mother-machine layout configuration."""

    HORIZONTAL = auto()
    VERTICAL = auto()


# TODO(Codex): This is CRISP specific. This should be IN_FOCUS, OUT_OF_FOCUS, and UNKNOWN. This below should be moved to CRISP and then have a mapping to this.
class AutoFocusStatusType(EvoType):
    """CRISP autofocus status flags reported by the device and consumed by camera/autofocus logic."""

    # These stati match the stati from CRISPStatus in asitiger.status
    IDLE = "I"              # LED is tuned off going from Ready to Idle
    READY = "R"             # LED on
    DIM = "D"               # Low returned light signal (prevents Ready state)
    OUT_OF_FOCUS = "K"      # Active but not within focus tolerance
    IN_FOCUS = "F"          # Active and within focus tolerance
    INHIBIT = "N"           # Low returned signal (unlocks system)
    ERROR = "E"             # Usually Out-of-Range Error
    LOG_CAL = "G"           # Initiate basic Log-Amp Calibration


class AutomatonCommandType(EvoType):
    """Command and message categories passed between strategies, the automaton, and the GUI."""

    MAGNET = auto()
    CALIBRATE_MAGNET = auto()
    CALIBRATE_HALL = auto()
    READ_HALL = auto()
    
    IMAGE = auto()
    MOVE = auto()
    UPDATE_FOV_CONFIG = auto()
    PROJECT = auto()
    PROJECT_ROI = auto()
    WAIT = auto()
    STOP = auto()
    LIVE_MODE = auto()
    SAVE_STATE = auto()

    # The types below are used by the GUI
    FOCUS_DATA = auto()
    FOV_DATA = auto()
    INFO_TEXT = auto()
    PROCESS_DATA = auto()
    REF_DATA = auto()
    ROI_DATA = auto()
    SEG_DATA = auto()
    AUTOFOCUS_DATA = auto()

    # Strategy lifecycle controls. These remain distinct from STOP, which is a
    # generic execution halt and does not itself define finalisation semantics.
    TERMINATE_STRATEGY = auto()
    ABORT_STRATEGY = auto()


# TODO(Codex): rename as SoftwareFocusStatusType
class FocusStatusType(EvoType):
    """Software focus result status used to decide whether focusing succeeded or why it failed."""

    IN_FOCUS = auto()
    BAD_FOCUS_CURVE = auto()
    OUT_OF_RANGE = auto()
    DEVICE_ERROR = auto()
    NO_IMAGE = auto()
    UNKNOWN = auto()


class FocusCurveType(EvoType):
    """Classification of a focus-score curve produced by software focus analysis."""

    HAS_GLOBAL_MAXIMUM = auto()     # Has one global maximum (good)
    HAS_MAXIMA = auto()             # Has more than one maximum (bad)
    HAS_BOUNDARY_MAXIMUM = auto()   # Has maximum at a boundary (bad)
    UNKNOWN = auto()


class AxisType(EvoType):
    """Stage axis identifiers for coordinate and movement code."""

    X = 0
    Y = 1
    Z = 2

class FovDirectionType(EvoType):
    """FOV movement directions."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    HOME = auto()

class FocusAlgorithmType(EvoType):
    """Software focus scoring algorithm selection used by focus configuration and routines."""

    LAPLACIAN_VAR = auto()
    SQUARED_GRAD_AVG = auto()
    STEEL = auto()
    BANDPASS_FFT = auto()


class LEDType(EvoType):
    """Logical LED channel identifiers used for illumination, imaging, projection, and GUI controls."""

    NO_LED = -1
    LED_385_NM = 0
    LED_450_NM = 1
    LED_515_NM = 2
    LED_565_NM = 3
    LED_645_NM = 4
    LED_OVERHEAD = 5
    LED_OVERHEAD_TIGER = 6

    # TODO(CODEX): are the TIGER_LED_1-4 used anywhere? If not remove them.
    # TIGER
    TIGER_LED_1 = auto()
    TIGER_LED_2 = auto()
    TIGER_LED_3 = auto()
    TIGER_LED_4 = auto()


class FilterWheelType(EvoType):
    """Filter wheel positions used when acquiring fluorescence and brightfield images."""

    UNKNOWN = -1
    FILTER = 0          # non-specific
    FILTER_465nm = 1
    FILTER_527nm = 2
    FILTER_592nm = 3
    NO_FILTER = 4
    BLOCKING = 5


class DMDDirType(EvoType):
    """DMD calibration/projection direction identifiers for horizontal or vertical patterns."""

    HORIZ = 0
    VERT = 1

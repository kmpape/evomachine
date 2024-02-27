from enum import Enum
from typing import List

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

SMALL = QFont("Arial", 10)
NORMAL = QFont("Arial", 12)
LEFT = Qt.AlignLeft
CENTER = Qt.AlignCenter
RIGHT = Qt.AlignRight

AXES = ['X', 'Y', 'Z']

ARROW_LEFT = "\u2190"
ARROW_RIGHT = "\u2192"
ARROW_UP = "\u2191"
ARROW_DOWN = "\u2193"

class Direction(Enum):
    LEFT = 0   # DECR_X
    RIGHT = 1  # INCR_X
    UP = 2     # DECR_Y
    DOWN = 3   # INCR_Y
    DOWN_Z = 4
    UP_Z = 5
    HOME = 6
    MOVETO = 7

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]


class DMDModes(Enum):
    DISPLAY_NONE = 0
    DISPLAY_FULL = 1


class DisplayMode(Enum):
    NO_CROP = 0
    SHOW_FRAME = 1
    CROP = 2
    UNKNOWN = 3

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]

    def get_string(self) -> str:
        if self == DisplayMode.NO_CROP:
            return "No Crop"
        elif self == DisplayMode.SHOW_FRAME:
            return "Show Frame(s)"
        elif self == DisplayMode.CROP:
            return "Crop"
        else:
            return "Unknown"

    @classmethod
    def from_string(cls, s: str) -> 'DisplayMode':
        for member in cls:
            if member.get_string() == s:
                return member
        return cls.UNKNOWN


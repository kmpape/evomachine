from collections import deque
from enum import Enum, auto
import time
from typing import Dict, List


MAX_ERROR_LEN = 1000


class ErrorCode(Enum):
    NO_ERROR = 0
    ERROR = 1
    ERROR_NOT_INITIALISED = auto()
    ERROR_WRONG_FORMAT = auto()

    # Configuration errors
    ERROR_CRISP_CONFIG = auto()
    ERROR_DEVICE_CONFIG = auto()
    ERROR_FOCUS_CONFIG = auto()
    ERROR_IMAGE_CONFIG = auto()
    ERROR_TEST_CAMERA_CONFIG = auto()

    # Camera errors
    ERROR_MMC_NOT_ALIVE = auto()

    # DMD errors
    ERROR_MONITORS = auto()

    # Tiger errors
    ERROR_TIGER_SERIAL_CONNECTION = auto()
    ERROR_TIGER_NOT_ALIVE = auto()
    ERROR_TIGER_NO_DATA = auto()
    ERROR_STAGE_MOVEMENT = auto()
    ERROR_STAGE_COORDINATES = auto()

    # Tracking errors
    ERROR_TRACK_DIV_NOT_DETECTED = auto()
    ERROR_TRACK_NO_INPUTS = auto()
    ERROR_TRACK_NO_PREV_STATE = auto()

    # Strategy errors
    ERROR_STRATEGY = auto()

    @classmethod
    def get_all_values(cls) -> List[int]:
        """ Returns all member values defined in Enum class."""
        return [member.value for member in cls]

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


class EvoMachineError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.message = message
        self.time = time.time()
        self.error_code = error_code

    def __str__(self):
        t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.time))
        return f"{t_str} (code {self.error_code}): {self.message}"

    def __reduce__(self):
        return self.__class__, (self.message, self.error_code)


class ConfigError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class ImageProcessingError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class StageError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class TigerError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class DMDError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class CameraError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class StrategyError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)

    def __reduce__(self):
        return super().__reduce__()


class ErrorContainer:
    def __init__(self):
        self.error_list = deque(maxlen=MAX_ERROR_LEN)
        self.has_overflow = False

    def add_error(self, new_error: EvoMachineError):
        if len(self.error_list) == MAX_ERROR_LEN:
            self.has_overflow = True
        self.error_list.append(new_error)

    def clear_errors(self):
        self.error_list.clear()

    def __len__(self):
        return len(self.error_list)

    def __str__(self):
        all_errors = [err.__str__() for err in self.error_list]
        return "\n".join(all_errors)

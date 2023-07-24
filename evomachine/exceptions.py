from collections import deque
from enum import Enum, auto
import time

MAX_ERROR_LEN = 1000


class ErrorCode(Enum):
    NO_ERROR = 0
    ERROR = 1
    ERROR_NOT_INITIALISED = auto()

    # Configuration errors
    ERROR_DEVICE_CONFIG = auto()
    ERROR_IMAGE_CONFIG = auto()

    # Camera errors
    ERROR_MMC_NOT_ALIVE = auto()

    # Tiger errors
    ERROR_TIGER_SERIAL_CONNECTION = auto()
    ERROR_TIGER_NOT_ALIVE = auto()
    ERROR_STAGE_MOVEMENT = auto()
    ERROR_STAGE_COORDINATES = auto()

    # Tracking errors
    ERROR_TRACK_DIV_NOT_DETECTED = auto()
    ERROR_TRACK_NO_INPUTS = auto()


class EvoMachineError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.message = message
        self.time = time.time()
        self.error_code = error_code

    def __str__(self):
        t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.time))
        return f"{t_str} (code {self.error_code.value}): {self.message}"


class ConfigError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)


class ImageProcessingError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)


class StageError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)


class TigerError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)


class CameraError(EvoMachineError):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message=message, error_code=error_code)


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

from enum import Enum, auto


class ErrorCode(Enum):
    NO_ERROR = auto()

    ERROR_DEVICE_CONFIG = auto()
    ERROR_IMAGE_CONFIG = auto()

    ERROR_STAGE_MOVEMENT = auto()
    ERROR_STAGE_COORDINATES = auto()

    ERROR_NOT_INITIALISED = auto()

    ERROR_TIGER_NOT_ALIVE = auto()

    # Tracking errors
    ERROR_TRACK_DIV_NOT_DETECTED = auto()


class ConfigError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.error_code = error_code


class ImageProcessingError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.error_code = error_code


class StageError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.error_code = error_code


class TigerError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.error_code = error_code

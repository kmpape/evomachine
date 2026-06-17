from dataclasses import dataclass, field
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Path to large data storage to store logs and files
EVOMACHINE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR = EVOMACHINE_DIR / "images"
DMD_WIDTH_HEIGHT = (2716, 1600)
CAM_WIDTH_HEIGHT = (3200, 3200)

# DeLTA/evomachine/asitiger/syncboard/dmdwindow lib install directory

# Switch between pygame dmd.py and dmd_socket.py
USE_DMD_SOCKET: bool = True

# Switch between ASI Tiger LEDs and SyncBoard
USE_SYNC_BOARD: bool = True

EVO_LOGGING_LEVEL = logging.INFO
EVO_GUI_LOGGING_LEVEL = logging.INFO
EVO_PERIPHERAL_LOGGING_LEVEL = logging.DEBUG
EVO_BINDING_LOGGING_LEVEL = logging.DEBUG
# EVO_LOGGING_LEVEL = logging.DEBUG
# EVO_GUI_LOGGING_LEVEL = logging.DEBUG


@dataclass(frozen=True)
class TimeConfig:
    """
    Configure timezone handling for timestamps.

    Parameters
    ----------
    timezone_name
        IANA timezone name used for generated and displayed timestamps.

    Returns
    -------
    TimeConfig
        Immutable timezone configuration.
    """

    timezone_name: str = "Europe/London"

    def __post_init__(self) -> None:
        """
        Validate the configured timezone name.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not isinstance(self.timezone_name, str):
            raise TypeError(f"TimeConfig: timezone_name must be str, received {type(self.timezone_name)}.")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"TimeConfig: unknown timezone {self.timezone_name}.") from error


@dataclass(frozen=True)
class EvoLoggingConfig:
    """
    Configure evomachine logger levels and timezone formatting.

    Parameters
    ----------
    level
        Logging level for non-GUI loggers.
    gui_level
        Logging level for GUI loggers.
    peripheral_level
        Logging level for peripheral loggers.
    binding_level
        Logging level for external hardware binding loggers.
    binding_logger_names
        Package logger names configured for external hardware bindings.
    time_config
        TimeConfig used to format log timestamps.

    Returns
    -------
    EvoLoggingConfig
        Immutable logging configuration.
    """

    level: int = logging.INFO
    gui_level: int = logging.INFO
    peripheral_level: int = logging.DEBUG
    binding_level: int = logging.DEBUG
    binding_logger_names: tuple[str, ...] = ("syncboard", "asitiger")
    time_config: TimeConfig = field(default_factory=TimeConfig)

    def __post_init__(self) -> None:
        """
        Validate logging levels and time configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for field_name in ("level", "gui_level", "peripheral_level", "binding_level"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"EvoLoggingConfig: {field_name} must be int, received {type(value)}.")
        if not isinstance(self.binding_logger_names, tuple):
            raise TypeError(
                "EvoLoggingConfig: binding_logger_names must be tuple[str, ...], "
                f"received {type(self.binding_logger_names)}."
            )
        for logger_name in self.binding_logger_names:
            if not isinstance(logger_name, str):
                raise TypeError(
                    "EvoLoggingConfig: binding_logger_names must contain only str values, "
                    f"received {type(logger_name)}."
                )
            if not logger_name:
                raise ValueError("EvoLoggingConfig: binding_logger_names must not contain empty names.")
        if not isinstance(self.time_config, TimeConfig):
            raise TypeError(
                f"EvoLoggingConfig: time_config must be TimeConfig, received {type(self.time_config)}."
            )


DEFAULT_TIME_CONFIG = TimeConfig()
DEFAULT_LOGGING_CONFIG = EvoLoggingConfig(
    level=EVO_LOGGING_LEVEL,
    gui_level=EVO_GUI_LOGGING_LEVEL,
    peripheral_level=EVO_PERIPHERAL_LOGGING_LEVEL,
    binding_level=EVO_BINDING_LOGGING_LEVEL,
    time_config=DEFAULT_TIME_CONFIG,
)


def get_timezone(config: TimeConfig | None = None) -> ZoneInfo:
    """
    Return the timezone configured for evomachine timestamps.

    Parameters
    ----------
    config
        Optional TimeConfig. If None, DEFAULT_TIME_CONFIG is used.

    Returns
    -------
    ZoneInfo
        Timezone object for the configured IANA timezone name.
    """
    config = DEFAULT_TIME_CONFIG if config is None else config
    if not isinstance(config, TimeConfig):
        raise TypeError(f"get_timezone: config must be TimeConfig or None, received {type(config)}.")
    return ZoneInfo(config.timezone_name)


def now(config: TimeConfig | None = None) -> datetime:
    """
    Return the current timezone-aware datetime.

    Parameters
    ----------
    config
        Optional TimeConfig controlling the timezone.

    Returns
    -------
    datetime
        Current timezone-aware datetime.
    """
    return datetime.now(tz=get_timezone(config=config))


def ensure_timezone_aware(value: datetime, config: TimeConfig | None = None) -> datetime:
    """
    Return a timezone-aware datetime, interpreting naive values in the default timezone.

    Parameters
    ----------
    value
        Datetime to normalise.
    config
        Optional TimeConfig used when value is naive and as the output timezone.

    Returns
    -------
    datetime
        Timezone-aware datetime converted to the configured timezone.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"ensure_timezone_aware: value must be datetime, received {type(value)}.")
    timezone = get_timezone(config=config)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def format_timezone_id(value: datetime | None = None, config: TimeConfig | None = None) -> str:
    """
    Return a filename-safe IANA timezone identifier.

    Parameters
    ----------
    value
        Optional datetime whose timezone should be represented.
    config
        Optional TimeConfig used when value has no named timezone.

    Returns
    -------
    str
        Filename-safe timezone identifier.
    """
    timezone_name = None
    if value is not None:
        if not isinstance(value, datetime):
            raise TypeError(f"format_timezone_id: value must be datetime or None, received {type(value)}.")
        if hasattr(value.tzinfo, "key"):
            timezone_name = value.tzinfo.key
    if timezone_name is None:
        timezone_name = (DEFAULT_TIME_CONFIG if config is None else config).timezone_name
    return timezone_name.replace("/", "-")


def format_timestamp_for_filename(value: datetime, config: TimeConfig | None = None) -> str:
    """
    Return a filename-safe timestamp with timezone offset.

    Parameters
    ----------
    value
        Datetime to format.
    config
        Optional TimeConfig used for normalisation.

    Returns
    -------
    str
        Timestamp suitable for filenames.
    """
    value = ensure_timezone_aware(value=value, config=config)
    return value.strftime("%Y-%m-%d_%H-%M-%S-%f%z")


def format_timestamp_for_display(value: datetime, config: TimeConfig | None = None) -> str:
    """
    Return a display timestamp in the configured timezone.

    Parameters
    ----------
    value
        Datetime to format.
    config
        Optional TimeConfig used for normalisation.

    Returns
    -------
    str
        Human-readable timestamp.
    """
    value = ensure_timezone_aware(value=value, config=config)
    return value.strftime("%Y-%m-%d %H:%M:%S")


class EvoFormatter(logging.Formatter):
    """
    Logging formatter that renders timestamps in an evomachine timezone.

    Parameters
    ----------
    fmt
        Logging format string.
    time_config
        TimeConfig used to format record timestamps.

    Returns
    -------
    EvoFormatter
        Formatter with timezone-aware asctime values.
    """

    def __init__(self, fmt: str, time_config: TimeConfig | None = None):
        """
        Initialise a timezone-aware formatter.

        Parameters
        ----------
        fmt
            Logging format string.
        time_config
            Optional TimeConfig used for timestamp formatting.

        Returns
        -------
        None
        """
        super().__init__(fmt=fmt)
        self.time_config: TimeConfig = DEFAULT_TIME_CONFIG if time_config is None else time_config

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """
        Format one log record timestamp.

        Parameters
        ----------
        record
            Log record whose created timestamp is formatted.
        datefmt
            Optional strftime-compatible date format.

        Returns
        -------
        str
            Formatted timestamp.
        """
        record_time = datetime.fromtimestamp(record.created, tz=get_timezone(config=self.time_config))
        if datefmt is not None:
            return record_time.strftime(datefmt)
        return format_timestamp_for_display(value=record_time, config=self.time_config)


EVO_FORMATTER = EvoFormatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

consolidated_logger = logging.getLogger('consolidated_logger')
consolidated_logger.setLevel(EVO_LOGGING_LEVEL)

def _log_filename(logging_config: EvoLoggingConfig = DEFAULT_LOGGING_CONFIG) -> str:
    """
    Return a timezone-aware log filename.

    Parameters
    ----------
    logging_config
        Logging configuration used for timestamp and timezone formatting.

    Returns
    -------
    str
        Log filename.
    """
    timestamp = now(config=logging_config.time_config)
    return (
        f"evom_{format_timestamp_for_filename(value=timestamp, config=logging_config.time_config)}_"
        f"{format_timezone_id(value=timestamp, config=logging_config.time_config)}.log"
    )


# Add a file handler for consolidated logging
filename = _log_filename()
LOG_DIR = EVOMACHINE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(LOG_DIR / filename, maxBytes=1000000, backupCount=20)
file_handler.setFormatter(EVO_FORMATTER)
file_handler.setLevel(
    min(
        DEFAULT_LOGGING_CONFIG.level,
        DEFAULT_LOGGING_CONFIG.gui_level,
        DEFAULT_LOGGING_CONFIG.peripheral_level,
        DEFAULT_LOGGING_CONFIG.binding_level,
    )
)


def get_logger(
        name: str,
        is_gui: bool = False,
        logging_config: EvoLoggingConfig | None = None,
        is_peripheral: bool = False,
        is_binding: bool = False,
) -> logging.Logger:
    """
    Return a configured evomachine logger.

    Parameters
    ----------
    name
        Logger name.
    is_gui
        If True, use the GUI logging level.
    logging_config
        Optional logging configuration. If None, DEFAULT_LOGGING_CONFIG is used.
    is_peripheral
        If True, use the peripheral logging level.
    is_binding
        If True, use the external binding logging level.

    Returns
    -------
    logging.Logger
        Logger configured with stream and rotating file handlers.
    """
    global file_handler
    logging_config = DEFAULT_LOGGING_CONFIG if logging_config is None else logging_config
    if not isinstance(logging_config, EvoLoggingConfig):
        raise TypeError(
            f"get_logger: logging_config must be EvoLoggingConfig or None, received {type(logging_config)}."
        )
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    if is_binding:
        logger.setLevel(logging_config.binding_level)
    elif is_peripheral:
        logger.setLevel(logging_config.peripheral_level)
    elif is_gui:
        logger.setLevel(logging_config.gui_level)
    else:
        logger.setLevel(logging_config.level)

    formatter = EvoFormatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        time_config=logging_config.time_config,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(
        min(
            logging_config.level,
            logging_config.gui_level,
            logging_config.peripheral_level,
            logging_config.binding_level,
        )
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def configure_binding_loggers(logging_config: EvoLoggingConfig | None = None) -> tuple[logging.Logger, ...]:
    """
    Configure package loggers for external hardware bindings.

    Parameters
    ----------
    logging_config
        Optional logging configuration. If None, DEFAULT_LOGGING_CONFIG is used.

    Returns
    -------
    tuple[logging.Logger, ...]
        Configured binding package loggers.
    """
    logging_config = DEFAULT_LOGGING_CONFIG if logging_config is None else logging_config
    if not isinstance(logging_config, EvoLoggingConfig):
        raise TypeError(
            "configure_binding_loggers: logging_config must be EvoLoggingConfig or None, "
            f"received {type(logging_config)}."
        )
    return tuple(
        get_logger(name=logger_name, logging_config=logging_config, is_binding=True)
        for logger_name in logging_config.binding_logger_names
    )


def __getattr__(name: str):
    """Lazy exports for config classes owned by domain modules."""
    if name == "TigerAutofocusConfig":
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfig

        return TigerAutofocusConfig
    if name == "TigerAutofocusConfigFactory":
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory

        return TigerAutofocusConfigFactory
    if name == "SoftwareFocusConfig":
        from evomachine.softwarefocus import SoftwareFocusConfig

        return SoftwareFocusConfig
    if name == "SoftwareFocusConfigFactory":
        from evomachine.softwarefocus import SoftwareFocusConfigFactory

        return SoftwareFocusConfigFactory
    if name == "ImageProcessorConfig":
        from evomachine.image_processing_config import ImageProcessorConfig

        return ImageProcessorConfig
    if name == "ImageProcessorConfigFactory":
        from evomachine.image_processing_config import ImageProcessorConfigFactory

        return ImageProcessorConfigFactory
    if name in {"ImageConfigType", "ImageConfigTypeFactory", "ObjectiveConfigType", "ObjectiveConfigTypeFactory"}:
        from evomachine.peripherals import camera

        return getattr(camera, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

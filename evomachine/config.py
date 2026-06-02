from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

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

# EVO_FORMATTER = logging.Formatter('--->\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n<---')
EVO_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
EVO_LOGGING_LEVEL = logging.INFO
EVO_GUI_LOGGING_LEVEL = logging.INFO
# EVO_LOGGING_LEVEL = logging.DEBUG
# EVO_GUI_LOGGING_LEVEL = logging.DEBUG

consolidated_logger = logging.getLogger('consolidated_logger')
consolidated_logger.setLevel(EVO_LOGGING_LEVEL)

# Add a file handler for consolidated logging
filename = "evom_{}.log".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))
LOG_DIR = EVOMACHINE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(LOG_DIR / filename, maxBytes=1000000, backupCount=20)
file_handler.setFormatter(EVO_FORMATTER)
file_handler.setLevel(logging.INFO)


def get_logger(name: str, is_gui: bool = False) -> logging.Logger:
    global file_handler
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(EVO_LOGGING_LEVEL if not is_gui else EVO_GUI_LOGGING_LEVEL)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(EVO_FORMATTER)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def __getattr__(name: str):
    """Lazy compatibility exports for config classes moved to domain modules."""
    if name in {"ConfigCRISP", "TigerAutofocusConfig"}:
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfig

        return TigerAutofocusConfig
    if name in {"ConfigCRISPFactory", "TigerAutofocusConfigFactory"}:
        from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory

        return TigerAutofocusConfigFactory
    if name in {"ConfigFocus", "SoftwareFocusConfig"}:
        from evomachine.softwarefocus import SoftwareFocusConfig

        return SoftwareFocusConfig
    if name in {"ConfigFocusFactory", "SoftwareFocusConfigFactory"}:
        from evomachine.softwarefocus import SoftwareFocusConfigFactory

        return SoftwareFocusConfigFactory
    if name in {"ConfigImageProcessor", "ImageProcessorConfig"}:
        from evomachine.image_processing_config import ImageProcessorConfig

        return ImageProcessorConfig
    if name in {"ConfigImageProcessorFactory", "ImageProcessorConfigFactory"}:
        from evomachine.image_processing_config import ImageProcessorConfigFactory

        return ImageProcessorConfigFactory
    if name in {"ConfigCamera", "CameraSystemConfig"}:
        from evomachine.peripherals.camera import CameraSystemConfig

        return CameraSystemConfig
    if name in {"ConfigCameraFactory", "CameraSystemConfigFactory"}:
        from evomachine.peripherals.camera import CameraSystemConfigFactory

        return CameraSystemConfigFactory
    if name in {"ImageConfigType", "ImageConfigTypeFactory", "ObjectiveConfigType", "ObjectiveConfigTypeFactory"}:
        from evomachine.peripherals import camera

        return getattr(camera, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

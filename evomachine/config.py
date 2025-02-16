from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.evotypes import FilterWheelType, FocusAlgorithmType, LEDType, ImageConfigType, ObjectiveConfigType, \
    ImageConfigTypeFactory, ObjectiveConfigTypeFactory, ChamberOrientationType

# Path to large data storage to store logs and files
DATA_DIR = Path("/home/gabi/Projects/evomachine/evomachine_repo/data")
LOG_DIR = DATA_DIR / "Logs"
LOG_DIR.mkdir(parents=False, exist_ok=True)

# DeLTA/evomachine/asitiger/syncboard/dmdwindow lib install directory
EVOMACHINE_DIR: Path = Path(__file__).parent

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
file_handler = RotatingFileHandler(f"{LOG_DIR}/{filename}", maxBytes=1000000, backupCount=20)
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
    update_rate: int
    "The time in ms to wait between updates to the CRISP trajectory."
    objective_na: float
    "NA of the objective used to calculate dither steps. Can be different from the actual objective NA."

    user_input: bool | None = True
    "Ask for user input before configuring and locking CRISP autofocus."
    min_snr: int | None = 2
    "Minimum acceptable signal to noise ratio measured during calibration."
    min_error: int | None = 100
    "Minimum acceptable absolute error measured during calibration."
    pause_long: int | None = 5
    "Value of long pause in s between CRISP configuration steps."
    pause_short: int | None = 1
    "Value of short pause in s between CRISP configuration steps."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) -> int | float | bool | None:
        if attr_name == 'lock_range' or attr_name == 'objective_na':
            return float(attr_value_str)
        else:
            return int(attr_value_str)

    @staticmethod
    def attr_is_valid(attr_name: str, attr_value) -> bool:
        if attr_name == 'averaging':
            return isinstance(attr_value, int) and (attr_value >= 0) and (attr_value < 100)
        elif attr_name == 'led_intensity':
            return isinstance(attr_value, int) and (attr_value > 1) and (attr_value <= 100)
        elif attr_name == 'loop_gain':
            return isinstance(attr_value, int) and (attr_value >= 1) and (attr_value <= 100)
        elif attr_name == 'lock_range':
            return isinstance(attr_value, float) and (attr_value > 0) and (attr_value < 1)
        elif attr_name == 'objective_na':
            return isinstance(attr_value, float) and (attr_value > 0) and (attr_value < 10.0)
        elif attr_name == 'update_rate':
            return isinstance(attr_value, int)
        else:
            return False

    def __post_init__(self):
        if not self.attr_is_valid('led_intensity', self.led_intensity):
            raise TypeError(f"led_intensity must be an integer in the range (0,100]. Provided {self.led_intensity}.")
        if not self.attr_is_valid('loop_gain', self.loop_gain):
            raise TypeError(f"loop_gain must be an integer in the range [1,10]. Provided {self.loop_gain}.")
        if not self.attr_is_valid('averaging', self.averaging):
            raise TypeError(f"averaging must be an integer in the range [0,Inf). Provided {self.averaging}.")
        if not self.attr_is_valid('update_rate', self.update_rate):
            raise TypeError(f"update_rate must be an integer in the range [0,Inf). Provided {self.update_rate}.")
        if not self.attr_is_valid('lock_range', self.lock_range):
            raise TypeError(f"lock_range may lead to objective crashing into the sample. Provided {self.lock_range}.")

    def __str__(self):
        s = ["ConfigCRISP"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)

    def copy(self):
        return ConfigCRISP(**self.__dict__)


class ConfigCRISPFactory:
    @staticmethod
    def default_config() -> ConfigCRISP:
        return ConfigCRISP(
            led_intensity=70,
            loop_gain=10,
            averaging=5,
            update_rate=10,
            objective_na=0.65,
            lock_range=0.1,
        )


@dataclass
class ConfigFocus:
    exposure_time: float | int
    "Exposure time for focusing in ms."
    focus_channel: LEDType
    "LED channel to use while scanning. See LEDType for available channels."
    rel_range: int
    "Relative range for Z-movement of stage in 1/10 μm, e.g., stage will move current_position+-rel_range."
    step_size: int
    "Step size for Z-movement of stage in 1/10 μm, e.g., step_size=1 -> stage moves in 0.1 μm."
    brightness: float
    "Brightness value in (1,29) for LED brightness during focus."

    algorithm: FocusAlgorithmType = FocusAlgorithmType.STEEL
    "Algorithm used to focus. See FocusAlgorithmType for available algorithms."
    user_input: bool | None = True
    "Ask for user input before configuring and starting software focus."

    @staticmethod
    def get_attr_from_str(attr_name: str, attr_value_str: str) \
            -> int | float | bool | FocusAlgorithmType | LEDType | None:
        if attr_name == 'exposure_time':
            return float(attr_value_str)
        elif attr_name == 'user_input':
            return bool(attr_value_str)
        elif attr_name == 'algorithm':
            return FocusAlgorithmType.from_string(attr_value_str)
        elif attr_name == 'focus_channel':
            return LEDType(int(attr_value_str))
        elif attr_name == 'brightness':
            return float(attr_value_str)
        else:
            return int(attr_value_str)

    def attr_is_valid(self, attr_name: str, attr_value) -> bool:
        if attr_name == 'exposure_time':
            return (isinstance(attr_value, int) or isinstance(attr_value, float)) and attr_value >= 0.01
        elif attr_name == 'focus_channel':
            return isinstance(attr_value, LEDType)
        elif attr_name == 'rel_range':
            return isinstance(attr_value, int) and (attr_value > 0) and (attr_value < 2000)
        elif attr_name == 'step_size':
            return isinstance(attr_value, int) and (attr_value > 0) and (attr_value <= self.rel_range)
        elif attr_name == 'algorithm':
            return isinstance(attr_value, FocusAlgorithmType)
        elif attr_name == 'brightness':
            return (isinstance(attr_value, float) or isinstance(attr_value, int))\
                and (attr_value >= 0) and (attr_value <= 29)
        elif attr_name == 'user_input':
            return isinstance(attr_value, bool)
        else:
            return False

    def __post_init__(self):
        if not self.attr_is_valid('step_size', self.step_size):
            raise TypeError(f"step_size must be an int in [1, rel_range={self.rel_range}]. Provided {self.step_size}.")
        if not self.attr_is_valid('rel_range', self.rel_range):
            raise TypeError(f"rel_range must be an integer in the range [1, Inf]. Provided {self.rel_range}.")
        if not self.attr_is_valid('focus_channel', self.focus_channel):
            raise TypeError(f"focus_channel must be a led type.")
        if not self.attr_is_valid('exposure_time', self.exposure_time):
            raise TypeError(f"exposure_time must be an int in [0.01, Inf]. Provided {self.exposure_time}.")
        if not self.attr_is_valid('brightness', self.brightness):
            raise TypeError(f"brightness must be an int or float in [0, 29]. Provided {self.brightness}.")
        if not self.attr_is_valid('algorithm', self.algorithm):
            raise TypeError(f"algorithm must be an instance of FocusAlgorithmType. Provided {self.algorithm}.")

    def copy(self):
        return ConfigFocus(**self.__dict__)

    def __str__(self):
        s = ["ConfigFocus"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if i < len(self.__dict__) - 1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


class ConfigFocusFactory:
    @staticmethod
    def default_config() -> ConfigFocus:
        return ConfigFocus(
            exposure_time=200,
            focus_channel=LEDType.LED_450_NM,
            brightness=29,
            rel_range=50,
            step_size=5,
        )


@dataclass
class ConfigCamera:
    objective: ObjectiveConfigType
    "Objective type. See ObjectiveType."
    image: ImageConfigType
    "Image configuration. See ImageConfig."
    focus: ConfigFocus
    "Focus configuration. See ConfigFocus."
    autofocus: ConfigCRISP
    "Autofocus configuration. See ConfigCRISP."
    leds: list[LEDType]
    "Available LED channels. See LEDType."
    filters: list[FilterWheelType]
    "Available filter wheels. See FilterWheelType."
    path_to_save: Path
    "Path to save images."
    default_exposure_time: float | int = 500
    "Default exposure time in ms."
    default_focus_channel_id: int = 0
    "Default LED channel index in self.leds."
    cam_pxl_size: float = 6.5
    "Pixel size of camera in μm."

    def copy(self):
        return ConfigCamera(**self.__dict__)

    @property
    def pxl_size(self) -> float:
        """
        Returns the size of one pixel in micrometers.

        Returns
        -------
        pixel_size: float
        """
        return self.cam_pxl_size / self.objective.mag  # in μm

    @property
    def fov_size(self) -> float:
        """
        Returns the size of the field of view in micrometers.

        Returns
        -------
        fov_size: float
        """
        return self.cam_pxl_size / self.objective.mag * self.image.pxl_vert  # in μm

    def __post_init__(self):
        if not isinstance(self.objective, ObjectiveConfigType):
            raise TypeError(f"objective must be a ObjectiveType object. Provided {self.objective}.")
        if not isinstance(self.image, ImageConfigType):
            raise TypeError(f"image must be a ImageConfigType object. Provided {self.image}.")
        if not isinstance(self.focus, ConfigFocus):
            raise TypeError(f"focus must be a ConfigFocus object. Provided {self.focus}.")
        if not isinstance(self.autofocus, ConfigCRISP):
            raise TypeError(f"autofocus must be a ConfigCRISP object. Provided {self.autofocus}.")
        if not (isinstance(self.leds, list) and all(isinstance(led, LEDType) for led in self.leds))\
                or len(self.leds) == 0 or LEDType.NO_LED not in self.leds:
            raise ConfigError("Invalid LED list.", ErrorCode.ERROR_CONFIG)
        if not (isinstance(self.filters, list) and all(isinstance(f, FilterWheelType) for f in self.filters))\
                or len(self.filters) == 0:
            raise ConfigError("Invalid filter list.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.path_to_save, Path) and self.path_to_save.exists():
            raise ConfigError("Invalid path_to_save.", ErrorCode.ERROR_CONFIG)
        if not self.image.pxl_vert == self.image.pxl_horiz:
            raise ConfigError(f"Currently limited to square images.", ErrorCode.ERROR_FOCUS_CONFIG)
        if ((not (isinstance(self.default_exposure_time, int) or isinstance(self.default_exposure_time, float))) or
                self.default_exposure_time <= 0):
            raise TypeError(f"Invalid default_exposure_time {self.default_exposure_time}.")
        if not (isinstance(self.default_focus_channel_id, int) and 0 <= self.default_focus_channel_id < len(self.leds)):
            raise TypeError(f"Invalid default_focus_channel_id {self.default_focus_channel_id}.")


class ConfigCameraFactory:
    @staticmethod
    def get_available_leds() -> list[LEDType]:
        if USE_SYNC_BOARD:
            return [LEDType.NO_LED, LEDType.LED_385_NM, LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM,
                    LEDType.LED_645_NM, LEDType.LED_OVERHEAD]
        else:
            return [LEDType.NO_LED, LEDType.LED_405_NM, LEDType.LED_450_NM, LEDType.LED_505_NM, LEDType.LED_538_NM]

    @staticmethod
    def get_available_filters() -> list[FilterWheelType]:
        return [FilterWheelType.FILTER, FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_527nm,
                FilterWheelType.FILTER_592nm, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER]

    @staticmethod
    def default_oil_config(path_to_save: Path | None = None) -> ConfigCamera:
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_oil(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=ConfigCameraFactory.get_available_filters(), # [FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=EVOMACHINE_DIR.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

    @staticmethod
    def default_air_config(path_to_save: Path | None = None) -> ConfigCamera:
        return ConfigCamera(
            objective=ObjectiveConfigTypeFactory.default_air(),
            image=ImageConfigTypeFactory.pv_cam(),
            focus=ConfigFocusFactory.default_config(),
            autofocus=ConfigCRISPFactory.default_config(),
            leds=ConfigCameraFactory.get_available_leds(),
            filters=ConfigCameraFactory.get_available_filters(), # [FilterWheelType.FILTER, FilterWheelType.BLOCKING, FilterWheelType.NO_FILTER],
            path_to_save=EVOMACHINE_DIR.parent / "images/DEFAULT" if path_to_save is None else path_to_save,
        )

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
from typing import Any

import numpy as np

from evomachine.config import CAM_WIDTH_HEIGHT, DATA_DIR, DMD_WIDTH_HEIGHT, EVOMACHINE_DIR


@dataclass(frozen=True, kw_only=True)
class HardwareGuiRuntimeSettings:
    """Default hardware wiring used by scripts/launch_hardware_gui.py."""

    syncboard_port: str | None = None
    syncboard_hwid: str = "16C0:0483"
    tiger_port: str | None = None
    tiger_hwid: str = "10C4:EA60"
    kwr103_port: str | None = None
    kwr103_hwid: str = "0416:5011"
    camera_device: str = "Camera-1"
    readout_mode_property: str = "Port"
    camera_width: int = CAM_WIDTH_HEIGHT[0]
    camera_height: int = CAM_WIDTH_HEIGHT[1]
    camera_exposure_ms: float = 200.0
    stage_fov_step_size: float = 100.0
    stage_min_x_um: float = -8000.0
    stage_max_x_um: float = 8000.0
    stage_min_y_um: float = -19000.0
    stage_max_y_um: float = 19000.0
    stage_min_z_um: float = -1000.0
    stage_max_z_um: float = 1000.0
    filter_card_address: int = 8
    dmd_width: int = DMD_WIDTH_HEIGHT[0]
    dmd_height: int = DMD_WIDTH_HEIGHT[1]
    dmd_calibration_file: Path = EVOMACHINE_DIR / "evomachine" / "dmd_calibration_data.pkl"
    output_directory: Path = DATA_DIR
    use_dmd: bool = True
    use_kwr103: bool = True

    def __post_init__(self) -> None:
        for axis, minimum, maximum in (
            ("X", self.stage_min_x_um, self.stage_max_x_um),
            ("Y", self.stage_min_y_um, self.stage_max_y_um),
            ("Z", self.stage_min_z_um, self.stage_max_z_um),
        ):
            if not isfinite(minimum) or not isfinite(maximum):
                raise ValueError(
                    f"HardwareGuiRuntimeSettings: stage {axis} limits must be finite."
                )
            if minimum >= maximum:
                raise ValueError(
                    f"HardwareGuiRuntimeSettings: stage {axis} minimum must be below maximum."
                )
            if minimum > 0 or maximum < 0:
                raise ValueError(
                    f"HardwareGuiRuntimeSettings: stage {axis} limits must include the startup zero."
                )

    @classmethod
    def from_env(cls) -> "HardwareGuiRuntimeSettings":
        defaults = cls()
        return cls(
            syncboard_port=os.getenv("EVOMACHINE_GUI_SYNCBOARD_PORT") or defaults.syncboard_port,
            syncboard_hwid=os.getenv("EVOMACHINE_GUI_SYNCBOARD_HWID", defaults.syncboard_hwid),
            tiger_port=os.getenv("EVOMACHINE_GUI_TIGER_PORT") or defaults.tiger_port,
            tiger_hwid=os.getenv("EVOMACHINE_GUI_TIGER_HWID", defaults.tiger_hwid),
            kwr103_port=os.getenv("EVOMACHINE_GUI_KWR103_PORT") or defaults.kwr103_port,
            kwr103_hwid=os.getenv("EVOMACHINE_GUI_KWR103_HWID", defaults.kwr103_hwid),
            camera_device=os.getenv("EVOMACHINE_GUI_MICROMANAGER_CAMERA_DEVICE", defaults.camera_device),
            readout_mode_property=os.getenv(
                "EVOMACHINE_GUI_MICROMANAGER_READOUT_PROPERTY",
                defaults.readout_mode_property,
            ),
            camera_width=_env_int("EVOMACHINE_GUI_CAMERA_WIDTH", defaults.camera_width),
            camera_height=_env_int("EVOMACHINE_GUI_CAMERA_HEIGHT", defaults.camera_height),
            camera_exposure_ms=_env_float("EVOMACHINE_GUI_CAMERA_EXPOSURE_MS", defaults.camera_exposure_ms),
            stage_fov_step_size=_env_float("EVOMACHINE_GUI_STAGE_FOV_STEP_SIZE", defaults.stage_fov_step_size),
            stage_min_x_um=_env_float("EVOMACHINE_GUI_STAGE_MIN_X_UM", defaults.stage_min_x_um),
            stage_max_x_um=_env_float("EVOMACHINE_GUI_STAGE_MAX_X_UM", defaults.stage_max_x_um),
            stage_min_y_um=_env_float("EVOMACHINE_GUI_STAGE_MIN_Y_UM", defaults.stage_min_y_um),
            stage_max_y_um=_env_float("EVOMACHINE_GUI_STAGE_MAX_Y_UM", defaults.stage_max_y_um),
            stage_min_z_um=_env_float("EVOMACHINE_GUI_STAGE_MIN_Z_UM", defaults.stage_min_z_um),
            stage_max_z_um=_env_float("EVOMACHINE_GUI_STAGE_MAX_Z_UM", defaults.stage_max_z_um),
            filter_card_address=_env_int("EVOMACHINE_GUI_FILTER_CARD_ADDRESS", defaults.filter_card_address),
            dmd_width=_env_int("EVOMACHINE_GUI_DMD_WIDTH", defaults.dmd_width),
            dmd_height=_env_int("EVOMACHINE_GUI_DMD_HEIGHT", defaults.dmd_height),
            dmd_calibration_file=_env_path(
                "EVOMACHINE_GUI_DMD_CALIBRATION_FILE",
                defaults.dmd_calibration_file,
            ),
            output_directory=_env_path("EVOMACHINE_GUI_OUTPUT_DIR", defaults.output_directory),
            use_dmd=_env_bool("EVOMACHINE_GUI_USE_DMD", defaults.use_dmd),
            use_kwr103=_env_bool("EVOMACHINE_GUI_USE_KWR103", defaults.use_kwr103),
        )

    @property
    def camera_size(self) -> tuple[int, int]:
        return self.camera_width, self.camera_height

    @property
    def dmd_size(self) -> tuple[int, int]:
        return self.dmd_width, self.dmd_height

    @property
    def stage_bounds(self):
        """Return microscope-safe bounds in zeroed Tiger coordinates, in micrometres."""
        from evomachine.coordinates import Coordinate, CoordinateBounds

        return CoordinateBounds(
            low=Coordinate(self.stage_min_x_um, self.stage_min_y_um, self.stage_min_z_um),
            high=Coordinate(self.stage_max_x_um, self.stage_max_y_um, self.stage_max_z_um),
        )

    @property
    def tiger_stage_limits(self) -> dict[str, tuple[float, float]]:
        """Return the same bounds in the Tiger driver's tenths-of-a-micrometre units."""
        scale = 10.0
        return {
            "X": (self.stage_min_x_um * scale, self.stage_max_x_um * scale),
            "Y": (self.stage_min_y_um * scale, self.stage_max_y_um * scale),
            "Z": (self.stage_min_z_um * scale, self.stage_max_z_um * scale),
        }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return default if value is None else Path(value).expanduser()


def _find_serial_port_by_hwid_fragment(
        hwid_fragment: str,
        display_name: str,
        ports: list[Any] | None = None,
) -> str:
    if ports is None:
        from serial.tools import list_ports

        ports = list(list_ports.comports())

    matches = [port for port in ports if hwid_fragment in str(getattr(port, "hwid", ""))]
    if len(matches) == 1:
        return str(matches[0].device)

    available = ", ".join(
        f"{getattr(port, 'device', '<unknown>')} [{getattr(port, 'hwid', '')}]"
        for port in ports
    )
    if not matches:
        raise RuntimeError(
            f"No {display_name} serial port matched HWID fragment {hwid_fragment!r}. "
            f"Available ports: {available or 'none'}."
        )
    raise RuntimeError(
        f"Multiple {display_name} serial ports matched HWID fragment {hwid_fragment!r}. "
        "Set the exact port with the corresponding EVOMACHINE_GUI_*_PORT environment variable. "
        f"Matches: {available}."
    )


def _resolve_serial_port(explicit_port: str | None, hwid_fragment: str, display_name: str) -> str:
    return explicit_port or _find_serial_port_by_hwid_fragment(hwid_fragment, display_name)


def build_virtual_automaton():
    """Build a small virtual automaton for GUI development."""
    from multiprocessing import Event

    from evomachine.acquisition import FrameAcquisitionManager
    from evomachine.bindings.binding_types import BindingType
    from evomachine.bindings.virtual.dmd import VirtualDmdPeripheralController
    from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
    from evomachine.config import DATA_DIR
    from evomachine.coordinates import Coordinate
    from evomachine.filemanager import FileManager, FileNameConfig
    from evomachine.frame import FrameMetaDataFactory
    from evomachine.image_processing_config import ImageProcessorConfigFactory
    from evomachine.navigation import FocusNavigator
    from evomachine.peripherals.autofocus import AutofocusConfig, AutofocusFactory
    from evomachine.peripherals.camera import CameraConfig, CameraFactory, ImageConfigType, ObjectiveConfigTypeFactory
    from evomachine.peripherals.dmd import DmdConfig, DmdFactory
    from evomachine.peripherals.filterwheel import FilterWheelConfig, FilterWheelFactory
    from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager
    from evomachine.peripherals.stage import StageConfig, StageFactory
    from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig
    from evomachine.strategy import NoStrategy
    from evomachine.types import FilterWheelType, FocusAlgorithmType, LEDType
    from evomachine.automaton import Automaton

    controller = VirtualPeripheralController()
    dmd_controller = VirtualDmdPeripheralController()
    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.VIRTUAL,
            fov_step_size=100.0,
            initial_coordinate=Coordinate(0, 0, 0),
            check_alive=False,
        ),
        peripheral_controllers=controller,
    )
    camera = CameraFactory.create(
        CameraConfig(
            binding=BindingType.VIRTUAL,
            image=ImageConfigType(pxl_horiz=64, pxl_vert=48, pxl_dtype=np.dtype("uint16")),
            objective_config=ObjectiveConfigTypeFactory.default_air(),
            check_alive=False,
        ),
        peripheral_controllers=controller,
    )
    filter_wheel = FilterWheelFactory.create(
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=[
                FilterWheelType.FILTER,
                FilterWheelType.FILTER_465nm,
                FilterWheelType.FILTER_527nm,
                FilterWheelType.FILTER_592nm,
                FilterWheelType.NO_FILTER,
                FilterWheelType.BLOCKING,
            ],
            check_alive=False,
        ),
        peripheral_controllers=controller,
        current_filter_type=FilterWheelType.NO_FILTER,
    )
    led_source = LedFactory.create(
        LedConfig(
            binding=BindingType.VIRTUAL,
            available_leds=[
                LEDType.LED_385_NM,
                LEDType.LED_450_NM,
                LEDType.LED_515_NM,
                LEDType.LED_565_NM,
                LEDType.LED_645_NM,
                LEDType.LED_OVERHEAD_TIGER,
            ],
            led_to_internal={
                LEDType.LED_385_NM: "385",
                LEDType.LED_450_NM: "450",
                LEDType.LED_515_NM: "515",
                LEDType.LED_565_NM: "565",
                LEDType.LED_645_NM: "645",
                LEDType.LED_OVERHEAD_TIGER: "tiger_overhead",
            },
            check_alive=False,
        ),
        peripheral_controllers=controller,
    )
    led_manager = LedManager([led_source])
    dmd = DmdFactory.create(
        DmdConfig(
            binding=BindingType.VIRTUAL,
            check_alive=False,
        ),
        peripheral_controllers=dmd_controller,
    )
    autofocus = AutofocusFactory.create(
        AutofocusConfig(
            binding=BindingType.VIRTUAL,
            check_alive=False,
        ),
        peripheral_controllers=controller,
    )
    acq_mngr = FrameAcquisitionManager(
        camera=camera,
        led_manager=led_manager,
        filter_wheel=filter_wheel,
        stage=stage,
        dmd=dmd,
        file_manager=FileManager(FileNameConfig(directory=DATA_DIR / "gui_demo_acquisitions")),
    )
    software_focus = SoftwareFocus(
        acquisition_manager=acq_mngr,
        config=SoftwareFocusConfig(
            focus_frames=[
                FrameMetaDataFactory.default(
                    leds={LEDType.LED_450_NM: 29},
                    exposure=200,
                    fov_id=0,
                )
            ],
            acquisition_settings=None,
            rel_range=15,
            step_size=5,
            algorithm=FocusAlgorithmType.STEEL,
            algorithm_kwargs={"rowshift": 1, "colshift": 1, "normalise": True},
            cropping_box=None,
        ),
    )
    focus_nav = FocusNavigator(stage=stage, autofocus=autofocus, software_focus=software_focus)
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[
            LEDType.LED_385_NM,
            LEDType.LED_450_NM,
            LEDType.LED_515_NM,
            LEDType.LED_565_NM,
            LEDType.LED_645_NM,
        ],
        channels_seg=[LEDType.LED_450_NM],
    )
    automaton = Automaton(
        acq_mngr=acq_mngr,
        focus_nav=focus_nav,
        strategy=NoStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        run_timeout=0.01,
    )
    return automaton


def build_hardware_automaton(settings: HardwareGuiRuntimeSettings | None = None):
    """Build the default hardware automaton used by the Napari GUI launcher."""
    from multiprocessing import Event

    from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
    from evomachine.automaton import Automaton
    from evomachine.bindings.binding_types import BindingType
    from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
    from evomachine.filemanager import FileManager, FileNameConfig
    from evomachine.frame import FrameMetaDataFactory
    from evomachine.image_processing_config import ImageProcessorConfigFactory
    from evomachine.navigation import FocusNavigator
    from evomachine.peripherals.autofocus import AutofocusConfig, AutofocusFactory
    from evomachine.peripherals.camera import CameraConfig, CameraFactory, ImageConfigType, ObjectiveConfigTypeFactory
    from evomachine.peripherals.dmd import DmdConfig, DmdFactory
    from evomachine.peripherals.filterwheel import FilterWheelConfig, FilterWheelFactory
    from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager
    from evomachine.peripherals.peripheralcontrollers import PeripheralControllerFactory, SerialPeripheralControllerConfig
    from evomachine.peripherals.stage import StageConfig, StageFactory
    from evomachine.projection import ProjectionManager
    from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig
    from evomachine.strategy import NoStrategy
    from evomachine.types import FilterWheelType, FocusAlgorithmType, LEDType

    settings = settings or HardwareGuiRuntimeSettings.from_env()
    syncboard_port = _resolve_serial_port(settings.syncboard_port, settings.syncboard_hwid, "SyncBoard")
    tiger_port = _resolve_serial_port(settings.tiger_port, settings.tiger_hwid, "ASI Tiger")
    kwr103_port = (
        _resolve_serial_port(settings.kwr103_port, settings.kwr103_hwid, "KWR103")
        if settings.use_kwr103
        else None
    )

    syncboard_controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.SYNCBOARD,
            port=syncboard_port,
            initialise=False,
        ),
    )
    tiger_controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            port=tiger_port,
            initialise=False,
        ),
        card_address_filter_wheel=settings.filter_card_address,
        stage_limits=settings.tiger_stage_limits,
    )
    kwr103_controller = None
    if kwr103_port is not None:
        kwr103_controller = PeripheralControllerFactory.create(
            SerialPeripheralControllerConfig(
                binding=BindingType.KWR103,
                port=kwr103_port,
                initialise=False,
            ),
        )

    camera = CameraFactory.create(
        CameraConfig(
            binding=BindingType.MMC,
            image=ImageConfigType(
                pxl_horiz=settings.camera_width,
                pxl_vert=settings.camera_height,
                pxl_dtype=np.dtype("uint16"),
            ),
            default_exposure_time=settings.camera_exposure_ms,
            objective_config=ObjectiveConfigTypeFactory.default_oil(),
        ),
        camera_device=settings.camera_device,
        readout_mode_property=settings.readout_mode_property,
    )
    syncboard_led_source = LedFactory.create(
        LedConfig(
            binding=BindingType.SYNCBOARD,
            available_leds=[
                LEDType.LED_385_NM,
                LEDType.LED_450_NM,
                LEDType.LED_515_NM,
                LEDType.LED_565_NM,
                LEDType.LED_645_NM,
            ],
        ),
        peripheral_controllers=syncboard_controller,
    )
    tiger_led_source = LedFactory.create(
        LedConfig(
            binding=BindingType.ASI_TIGER,
            available_leds=[LEDType.LED_OVERHEAD_TIGER],
        ),
        peripheral_controllers=tiger_controller,
    )
    led_sources = [syncboard_led_source, tiger_led_source]
    if kwr103_controller is not None:
        kwr103_led_source = LedFactory.create(
            LedConfig(
                binding=BindingType.KWR103,
                available_leds=[LEDType.LED_OVERHEAD],
            ),
            peripheral_controllers=kwr103_controller,
        )
        led_sources.append(kwr103_led_source)
    led_manager = LedManager(led_sources)
    filter_wheel = FilterWheelFactory.create(
        FilterWheelConfig(
            binding=BindingType.ASI_TIGER,
            available_filters=[
                FilterWheelType.FILTER,
                FilterWheelType.FILTER_465nm,
                FilterWheelType.FILTER_527nm,
                FilterWheelType.FILTER_592nm,
                FilterWheelType.NO_FILTER,
                FilterWheelType.BLOCKING,
            ],
        ),
        peripheral_controllers=tiger_controller,
    )
    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.ASI_TIGER,
            fov_step_size=settings.stage_fov_step_size,
            coordinate_bounds=settings.stage_bounds,
            zero_on_initialise=True,
        ),
        peripheral_controllers=tiger_controller,
    )
    dmd = None
    if settings.use_dmd:
        dmd_controller = EmDmdWindowPeripheralController(debug_mode=False)
        dmd = DmdFactory.create(
            DmdConfig(
                binding=BindingType.EM_DMD_WINDOW,
                width_height_DMD=settings.dmd_size,
                width_height_CAM=settings.camera_size,
                calibration_file=settings.dmd_calibration_file,
            ),
            peripheral_controllers=dmd_controller,
        )
    autofocus = AutofocusFactory.create(
        AutofocusConfig(binding=BindingType.ASI_TIGER),
        peripheral_controllers=tiger_controller,
    )
    acq_mngr = FrameAcquisitionManager(
        camera=camera,
        led_manager=led_manager,
        filter_wheel=filter_wheel,
        dmd=dmd,
        stage=stage,
        file_manager=FileManager(
            FileNameConfig(directory=settings.output_directory),
        ),
        default_settings=FrameAcquisitionSettings(
            save=False,
            normalise=False,
            illuminate_dmd=True,
            clear_dmd_after=True,
            restore_leds_after=False,
            disable_leds_after=True,
        ),
    )
    software_focus = SoftwareFocus(
        acquisition_manager=acq_mngr,
        config=SoftwareFocusConfig(
            focus_frames=[
                FrameMetaDataFactory.default(
                    leds={LEDType.LED_515_NM: 29},
                    exposure=settings.camera_exposure_ms,
                    fov_id=0,
                )
            ],
            acquisition_settings=None,
            rel_range=100,
            step_size=5,
            algorithm=FocusAlgorithmType.STEEL,
            algorithm_kwargs={"rowshift": 1, "colshift": 1, "normalise": True},
            cropping_box=None,
        ),
    )
    focus_nav = FocusNavigator(stage=stage, autofocus=autofocus, software_focus=software_focus)
    proj_mngr = None
    if dmd is not None:
        proj_mngr = ProjectionManager(
            camera=camera,
            dmd=dmd,
            led_manager=led_manager,
            filter_wheel=filter_wheel,
        )
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[
            LEDType.LED_385_NM,
            LEDType.LED_450_NM,
            LEDType.LED_515_NM,
            LEDType.LED_565_NM,
            LEDType.LED_645_NM,
        ],
        channels_seg=[LEDType.LED_450_NM],
    )
    return Automaton(
        acq_mngr=acq_mngr,
        focus_nav=focus_nav,
        strategy=NoStrategy(cfg=cfg),
        cfg_processor=cfg,
        start_strategy_event=Event(),
        stop_strategy_event=Event(),
        stop_event=Event(),
        shutdown_event=Event(),
        proj_mngr=proj_mngr,
        run_timeout=0.01,
    )

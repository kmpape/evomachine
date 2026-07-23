from __future__ import annotations

import numpy as np


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

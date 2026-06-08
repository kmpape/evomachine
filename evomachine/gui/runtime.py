from __future__ import annotations

import numpy as np


def build_virtual_automaton():
    """Build a small virtual automaton for GUI development."""
    from multiprocessing import Event

    from evomachine.acquisition import FrameAcquisitionManager
    from evomachine.bindings.binding_types import BindingType
    from evomachine.bindings.virtual.dmd import VirtualDmdPeripheralController
    from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
    from evomachine.coordinates import Coordinate
    from evomachine.image_processing_config import ImageProcessorConfigFactory
    from evomachine.navigation import FocusNavigator
    from evomachine.peripherals.camera import CameraConfig, CameraFactory, ImageConfigType
    from evomachine.peripherals.dmd import DmdConfig, DmdFactory
    from evomachine.peripherals.leds import LedConfig, LedFactory, LedManager
    from evomachine.peripherals.stage import StageConfig, StageFactory
    from evomachine.strategy import NoStrategy
    from evomachine.types import LEDType
    from evomachine.automaton import Automaton

    controller = VirtualPeripheralController()
    controller.initialise()
    dmd_controller = VirtualDmdPeripheralController()
    dmd_controller.initialise()
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
            check_alive=False,
        ),
        peripheral_controllers=controller,
    )
    led_source = LedFactory.create(
        LedConfig(
            binding=BindingType.VIRTUAL,
            available_leds=[LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM],
            led_to_internal={
                LEDType.LED_450_NM: "450",
                LEDType.LED_515_NM: "515",
                LEDType.LED_565_NM: "565",
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
    acq_mngr = FrameAcquisitionManager(camera=camera, led_manager=led_manager, stage=stage, dmd=dmd)
    focus_nav = FocusNavigator(stage=stage)
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM],
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
    automaton.initialise(fovs={0: Coordinate(0, 0, 0)})
    return automaton

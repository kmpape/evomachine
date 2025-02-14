#!/usr/bin/env python
from nicegui import app, ui, run

import os
from multiprocessing import Event, Lock, Process, Queue
from pathlib import Path
import sys
import threading
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'asitiger'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sync_board'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evomachine'))

from evomachine.acquisition import TestCamera, EvoCamera, EvoCamerav3 # , EvoCamerav3
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD 
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
    import pygame
from evomachine.evotypes import LEDType

from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import AbstractStrategy   # TODO add dropdown in GUI
from strategies.strategy_image import ImageStrategy

#### Automaton ####

import multiprocessing
ctx = multiprocessing.get_context('spawn')

# Create queues and events for multiprocessing
process_queue: Queue = ctx.Queue()
gui_to_automaton_queue: Queue = ctx.Queue()
automaton_to_gui_queue: Queue = ctx.Queue()
shutdown_event: Event = ctx.Event()
start_strategy_event: Event = ctx.Event()
stop_strategy_event: Event = ctx.Event()
stop_event: Event = ctx.Event()
initialize_event: Event = ctx.Event()
request_lock: Lock = ctx.Lock()

automaton_process: Process = None

#### GUI ####

LED_Brightness = {ledType: 10 for ledType in LEDType}
LED_Status = {ledType: 'Off' for ledType in LEDType}
state = {'status': 'Not Initialized'}

def set_led(led: LEDType):
    print(type(led))
    print(led)
    print(f"Setting {led} to {LED_Status[led]} with brightness {LED_Brightness[led]}")
    # This is bad because this cam.set_led
    # is not an atomic request, it toggles all the LEDs (?)
    queue_manager.request(
        req_str='self.cam.set_led',
        kwargs_dict={'i_chan': led, 'brightness': LED_Brightness[led]},
    )


async def initialize_clicked():
    state['status'] = 'Initializing'
    automaton_process.start()
    await run.io_bound(initialize_event.wait)
    state['status'] = 'Initialized'
    
def shutdown():
    stop_event.set()
    stop_strategy_event.set()
    start_strategy_event.clear()
    shutdown_event.set()
    
    # Wait until automaton has shut down peripherals
    time.sleep(1)
    start_time = time.time()
    while shutdown_event.is_set():
        if time.time() > start_time + 60:
            print("Error shutting down peripherals.")
            return False
    
    automaton_process.terminate()
    
    state['status'] = 'Not Initialized'
    return True
    
async def shutdown_clicked():
    state['status'] = 'Shutting down'
    result = await run.cpu_bound(shutdown)
    if result:
        state['status'] = 'Not Initialized'
    else:
        state['status'] = 'Error shutting down!'
    
with ui.header().classes(replace='row items-center') as header:
    ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
    with ui.tabs() as tabs:
        ui.tab('Interactive')
        ui.tab('Strategy')

# with ui.footer(value=False) as footer:
#     ui.label('Footer')

with ui.left_drawer().classes('bg-blue-100') as left_drawer:
    ui.label('Side menu')

# with ui.page_sticky(position='bottom-right', x_offset=20, y_offset=20):
#     ui.button(on_click=footer.toggle, icon='contact_support').props('fab')

with ui.tab_panels(tabs, value='Interactive').classes('w-full'):
    with ui.tab_panel('Interactive'):
        # Card with LED controls
        with ui.card():
            ui.button("Initialize", on_click=initialize_clicked).bind_enabled_from(state, target_name='status', backward=lambda x: x == 'Not Initialized')
            ui.label("").bind_text_from(state, target_name='status')
            ui.button("Shutdown", on_click=shutdown_clicked).bind_enabled_from(state, target_name='status', backward=lambda x: x == 'Initialized')
            
        
        with ui.card():
            with ui.column():
                # Each LED has a toggle (on or off) and a value (0 - 100) which is chosen using a slider or input field
                for led in LEDType:
                    with ui.row():
                        ui.label(f'{led.name}')
                        ui.toggle(options=['Off', 'On'], value='Off', on_change=lambda led=led: set_led(led)) \
                            .bind_value(LED_Status, led) \
                            .bind_enabled_from(state, target_name='status', backward=lambda x: x == 'Initialized')
                        ui.number(min=0, max=100, step=1, value=10, on_change=lambda led=led: set_led(led))\
                            .bind_value(LED_Brightness, led) \
                            .bind_enabled_from(state, target_name='status', backward=lambda x: x == 'Initialized')
                        # ui.slider(min=0, max=100, step=1, value=10, on_change=set_led).bind_value(LED_Brightness, f'LED{led}')
                        
    with ui.tab_panel('Strategy'):
        ui.label('Content of B')




def create_automaton_process(
        camera_config: ConfigCamera,
        processor_config: ConfigImageProcessor,
        start_strategy_event: Event,
        stop_strategy_event: Event,
        stop_event: Event,
        shutdown_event: Event,
        initialize_event: Event,
        process_queue: Queue,
        gui_to_automaton_queue: Queue,
        automaton_to_gui_queue: Queue,
        strategy: AbstractStrategy,
):
    if USE_SYNC_BOARD:
        cam = EvoCamerav3(cfg_camera=camera_config)
        # cam = EvoCamerav3(cfg_camera=camera_config)
    else:
        cam = EvoCamera(cfg_camera=camera_config)
    if not USE_DMD_SOCKET:
        pygame.init()
    dmd = DMDControl()
    automaton = Automaton(
        camera=cam,
        cfg_processor=processor_config,
        dmd=dmd,
        strategy=strategy,
        start_strategy_event=start_strategy_event,
        stop_strategy_event=stop_strategy_event,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
        initialize_event=initialize_event,
        process_q=process_queue,
        gui_to_automaton_q=gui_to_automaton_queue,
        automaton_to_gui_q=automaton_to_gui_queue,
        run_timeout=0,
    )
    automaton.run()

import atexit
def cleanup():
    print("Launch GUI: Clean up callback")
    shutdown()

if __name__ in {'__main__', '__mp_main__'}:
    print(f"Launching evomachine GUI from {EVOMACHINE_DIR}.")

    # Provide strategy that will be loaded by GUI
    # save_path: str = "/media/hslab/Data/ImageData/Idris/2024-07-04"
    # save_path = "/home/hslab/Documents/Gabi/GUI_SaveDir"
    save_path = "/mnt/nvme1/data/Default"
    if not os.path.exists(save_path):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_folder, "DEFAULT")
        os.makedirs(save_path, exist_ok=True)

    # Create configurations (modify if needed)
    is_oil_objective = False
    camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
    camera_config.path_to_save = Path(save_path)

    processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM, LEDType.LED_645_NM],
        channels_seg=[LEDType.LED_565_NM],
    )
    processor_config.preproc_enabled = False
    processor_config.roi_enabled = False
    processor_config.seg_enabled = False
    processor_config.track_enabled = False
    processor_config.lineage_enabled = False
    
    strategy: AbstractStrategy = ImageStrategy(processor_config)
    camera_config.focus.focus_channel = strategy.imaging_channel  # NEED TO GIVE SF THE RIGHT CHANNEL
    

    # Test strategy and do not launch if test fails
    if not strategy.test_strategy():  # noqa
        print("Strategy test not passed. Cannot launch GUI.")
        sys.exit(1)

    
    queue_manager: QueueManager = QueueManager(
        process_q=process_queue,
        gui_to_automaton_q=gui_to_automaton_queue,
        automaton_to_gui_q=automaton_to_gui_queue,
        start_strategy_event=start_strategy_event,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
        request_lock=request_lock,
        queue_timeout=0,
        run_timeout=0,
    )

    # Queue manager used by the GUI
    queue_thread: threading.Thread = threading.Thread(
        target=queue_manager.run,
        name='QueueManagerThread',
        daemon=True,
    )
    queue_thread.start()

    automaton_process: Process = ctx.Process(
        target=create_automaton_process,
        name='AutomatonProcess',
        daemon=True,
        args=(
            camera_config,
            processor_config,
            start_strategy_event,
            stop_strategy_event,
            stop_event,
            shutdown_event,
            initialize_event,
            process_queue,
            gui_to_automaton_queue,
            automaton_to_gui_queue,
            strategy,
        ),
    )

    def exception_hook(exctype, value, traceback):
        print(f"An unhandled exception occurred (type={exctype}): {value}\n{traceback}")
        stop_event.set()
        stop_strategy_event.set()
        start_strategy_event.set()
        shutdown_event.set()
        time.sleep(5)
        sys.exit()

    atexit.register(cleanup)

    sys.excepthook = exception_hook
    try:
        ui.run(favicon='🔬')
    except Exception as e:
        print(f"launch_gui: Exception occurred {e}.")
        traceback.print_exc()
        stop_event.set()
        stop_strategy_event.set()
        start_strategy_event.set()
        shutdown_event.set()
        time.sleep(5)
        sys.exit()
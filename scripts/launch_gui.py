import os
from multiprocessing import Event, Lock, Process, Queue
from pathlib import Path
import sys
import threading
import pygame
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')

from evomachine.acquisition import TestCamera, EvoCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl

from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import AbstractStrategy, BasicStrategy   # TODO add dropdown in GUI
from strategies.strategy_2024_03_07 import JessStrategy

# TODO remove test code below
# from evomachine.evotypes import LEDType
# camera_config = ConfigCameraFactory.default_air_config()
# cam = EvoCamera(cfg_camera=camera_config)
# cam.set_led(i_chan=LEDType.LED_450_NM, brightness=100)
# pygame.init()
# dmd = DMDControl()

def create_automaton_process(
        camera_config: ConfigCamera,
        processor_config: ConfigImageProcessor,
        start_strategy_event: Event,
        stop_strategy_event: Event,
        stop_event: Event,
        shutdown_event: Event,
        process_queue: Queue,
        gui_to_automaton_queue: Queue,
        automaton_to_gui_queue: Queue,
        strategy: AbstractStrategy,
):
    cam = EvoCamera(cfg_camera=camera_config)
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
        process_q=process_queue,
        gui_to_automaton_q=gui_to_automaton_queue,
        automaton_to_gui_q=automaton_to_gui_queue,
        use_seg=False,
        run_timeout=0,
    )
    automaton.run()


if __name__ == '__main__':
    # Provide strategy that will be loaded by GUI
    save_path: str = "/media/hslab/Data/ImageData/DEFAULT"
    save_path: str = "/media/hslab/Data/ImageData/Idris/2024-03-22"
    strategy: AbstractStrategy = BasicStrategy(save_path=save_path)

    # Create configurations (modify if needed)
    is_oil_objective = False
    camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
    camera_config.path_to_save = Path(save_path)
    processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()
    processor_config.cfg_delta.whole_frame_drift = True  # FIXME set to False. Temporary until ROI ID works.

    # DO NOT MODIFY ANYTHING BELOW THIS LINE -----------------------------------

    # Test strategy and do not launch if test fails
    if not strategy.test_strategy():
        print("Strategy test not passed. Cannot launch GUI.")
        sys.exit(1)

    # Create queues and events for multiprocessing
    process_queue: Queue = Queue()
    gui_to_automaton_queue: Queue = Queue()
    automaton_to_gui_queue: Queue = Queue()
    shutdown_event: Event = Event()
    start_strategy_event: Event = Event()
    stop_strategy_event: Event = Event()
    stop_event: Event = Event()
    request_lock: Lock = Lock()

    # Queue manager used by the GUI
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
    queue_thread: threading.Thread = threading.Thread(
        target=queue_manager.run,
        name='QueueManagerThread',
        daemon=True,
    )
    queue_thread.start()

    automaton_process: Process = Process(
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
            process_queue,
            gui_to_automaton_queue,
            automaton_to_gui_queue,
            strategy,
        ),
    )
    automaton_process.start()

    app: QApplication = QApplication(sys.argv)
    app.setStyleSheet("* { background-color: lightgray; }")
    w: EvoGUI = EvoGUI(
        queue_manager=queue_manager,
        camera_config=camera_config,
        processor_config=processor_config,
        start_strategy_event=start_strategy_event,
        stop_strategy_event=stop_strategy_event,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
    )
    icon = QIcon(str(EVOMACHINE_DIR / 'guidir/em_logo.jpg'))
    w.setWindowIcon(icon)
    w.show()
    sys.exit(app.exec_())

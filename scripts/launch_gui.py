import os
from multiprocessing import Event, Lock, Process, Queue
from pathlib import Path
import sys
import threading
import time
import traceback
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'asitiger'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sync_board'))

from evomachine.acquisition import TestCamera, EvoCamera, EvoCamerav2  # noqa
from evomachine.automaton import Automaton  # noqa
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD  # noqa
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl  # noqa
else:
    from evomachine.dmd import DMDControl  # noqa
    import pygame  # noqa

from evomachine.guidir.newgui import EvoGUI  # noqa
from evomachine.guidir.queuemanager import QueueManager  # noqa
from evomachine.strategy import AbstractStrategy, BasicStrategy   # TODO add dropdown in GUI  # noqa
from strategies.strategy_2024_03_07 import JessStrategy  # noqa
from strategies.strategy_2024_04_25 import UVTestingStrategy  # noqa
from strategies.strategy_2024_04_30 import UVTestingStrategyv2  # noqa
from strategies.strategy_2024_05_01 import UVTestingStrategyv3  # noqa
from strategies.strategy_2024_05_10 import UVTestingStrategyv4  # noqa
from strategies.strategy_2024_05_28 import UVTestingStrategyv5  # noqa
from strategies.strategy_2024_05_31 import ROITestingStrategy  # noqa
from strategies.strategy_MagnetOnOff import MagnetOnOffStrategy, PROCESSOR_CONFIG

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
    if USE_SYNC_BOARD:
        cam = EvoCamerav2(cfg_camera=camera_config)
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
        process_q=process_queue,
        gui_to_automaton_q=gui_to_automaton_queue,
        automaton_to_gui_q=automaton_to_gui_queue,
        run_timeout=0,
    )
    automaton.run()


if __name__ == '__main__':
    print(f"Launching evomachine GUI from {EVOMACHINE_DIR}.")

    # Provide strategy that will be loaded by GUI
    save_path: str = "/media/hslab/Data/ImageData/Vicente/2024-06-25"
    if not os.path.exists(save_path):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_folder, "DEFAULT")
        os.makedirs(save_path, exist_ok=True)

    # Create configurations (modify if needed)
    is_oil_objective = False
    camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
    camera_config.path_to_save = Path(save_path)
    
    # processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()
    # processor_config.cfg_delta.whole_frame_drift = True

    processor_config = PROCESSOR_CONFIG

    # Provide strategy that will be loaded by GUI
    # strategy: AbstractStrategy = UVTestingStrategyv5(cfg=processor_config)
    # strategy: AbstractStrategy = BasicStrategy(cfg=processor_config, save_path=save_path)
    # strategy: AbstractStrategy = ROITestingStrategy(cfg=processor_config)
    strategy: AbstractStrategy = MagnetOnOffStrategy(cfg=processor_config)
    
    # DO NOT MODIFY ANYTHING BELOW THIS LINE -----------------------------------

    # Test strategy and do not launch if test fails
    if not strategy.test_strategy():  # noqa
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

    def exception_hook(exctype, value, traceback):
        print(f"An unhandled exception occurred (type={exctype}): {value}\n{traceback}")
        stop_event.set()
        stop_strategy_event.set()
        start_strategy_event.set()
        shutdown_event.set()
        time.sleep(5)
        sys.exit()

    sys.excepthook = exception_hook
    try:
        w.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"launch_gui: Exception occurred {e}.")
        traceback.print_exc()
        stop_event.set()
        stop_strategy_event.set()
        start_strategy_event.set()
        shutdown_event.set()
        time.sleep(5)
        sys.exit()

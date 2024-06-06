import datetime
from multiprocessing import Event, Lock, Process, Queue
import os
from pathlib import Path
import sys
import threading

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'asitiger'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sync_board'))

from evomachine.acquisition import TestCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
    import pygame

from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import AbstractStrategy, BasicStrategy


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
    def get_position(filename):
        parts = filename.split('_')
        position_str = parts[1][1:]
        return int(position_str)

    def get_time(filename):
        parts = filename.split('_')
        time_str = parts[-2] + '_' + parts[-1].split('.')[0] + '.' + parts[-1].split('.')[1]
        return datetime.datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S.%f')

    folder_path = str(EVOMACHINE_DIR.parent / "data")
    filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith('.tiff')]
    filenames = sorted(filenames, key=lambda x: (get_position(x), get_time(x)))
    pos_to_filename = {get_position(filename): index for index, filename in enumerate(filenames)}
    filenames = [str(EVOMACHINE_DIR.parent / "data") + "/" + f for f in filenames]
    print(filenames)
    cam = TestCamera(cfg_camera=camera_config, filenames=filenames, pos_to_filename=pos_to_filename)

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
    print(f"Launching evomachine GUI (Testcamera) from {EVOMACHINE_DIR}.")

    # Provide strategy that will be loaded by GUI
    save_path: str = "/media/hslab/Data/ImageData/DEFAULT"
    if not os.path.exists(save_path):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_folder, "DEFAULT")
        os.makedirs(save_path, exist_ok=True)

    # Create configurations (modify if needed)
    is_oil_objective = False
    camera_config: ConfigCamera = ConfigCameraFactory.default_air_config()
    camera_config.path_to_save = Path(save_path)
    processor_config: ConfigImageProcessor = ConfigImageProcessorFactory.default_config()

    # Provide strategy that will be loaded by GUI
    strategy: AbstractStrategy = BasicStrategy(save_path=save_path, cfg=processor_config)

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

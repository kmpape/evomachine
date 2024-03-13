import os
from multiprocessing import Event, Lock, Process, Queue
import sys
import threading
import pygame
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')

from evomachine.acquisition import TestCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import BasicStrategy   # TODO add dropdown in GUI


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
):
    cam = TestCamera(
        cfg_camera=camera_config,
        filenames=[EVOMACHINE_DIR.parent / "tests/data/LED450NM_P0_X119_Y0_Z79_2024-02-15_19:51:29.956548.tiff"],
    )
    pygame.init()
    dmd = DMDControl()
    automaton = Automaton(
        camera=cam,
        cfg_processor=processor_config,
        dmd=dmd,
        strategy=BasicStrategy(),
        start_strategy_event=start_strategy_event,
        stop_strategy_event=stop_strategy_event,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
        process_q=process_queue,
        gui_to_automaton_q=gui_to_automaton_queue,
        automaton_to_gui_q=automaton_to_gui_queue,
        use_segmentation=False,
        run_timeout=0,
    )
    automaton.run()


if __name__ == '__main__':
    # Create configurations
    is_oil_objective = False
    camera_config = ConfigCameraFactory.default_air_config()
    camera_config.focus.rel_range = 10
    processor_config = ConfigImageProcessorFactory.default_config()
    processor_config.cfg_delta.model_file_seg = "/home/idris/.cache/delta/models/unet_moma_seg.hdf5"
    processor_config.cfg_delta.model_file_rois = "/home/idris/.cache/delta/models/unet_momachambers_seg.hdf5"
    processor_config.cfg_delta.model_file_track = "/home/idris/.cache/delta/models/unet_moma_track.hdf5"

    # Create queues and events for multiprocessing
    process_queue = Queue()
    gui_to_automaton_queue = Queue()
    automaton_to_gui_queue = Queue()
    shutdown_event = Event()
    start_strategy_event = Event()
    stop_strategy_event = Event()
    stop_event = Event()
    request_lock = Lock()

    queue_manager = QueueManager(
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
    queue_thread = threading.Thread(
        target=queue_manager.run,
        name='QueueManagerThread',
        daemon=True,
    )
    queue_thread.start()

    automaton_process = Process(
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
        ),
    )
    automaton_process.start()

    app = QApplication(sys.argv)
    app.setStyleSheet("* { background-color: lightgray; }")
    app.setStyleSheet("""
    background-color: #262626;
    color: #FFFFFF;
    font-family: Titillium;
    font-size: 18px;
    """)
    w = EvoGUI(
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

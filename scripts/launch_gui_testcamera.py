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

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
from evomachine.acquisition_bkp import TestCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCamera, ConfigCameraFactory, ConfigImageProcessor, ConfigImageProcessorFactory, \
    EVOMACHINE_DIR, USE_DMD_SOCKET, USE_SYNC_BOARD, DATA_DIR
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd_pygame import DMDControl
    import pygame
from evomachine.types import LEDType, FilterWheelType
from evomachine.guidir.newgui import EvoGUI
from evomachine.guidir.queuemanager import QueueManager
from evomachine.strategy import AbstractStrategy, BasicStrategy
from strategies.strategy_UV_testing import UVStrategy

from delta.rttypes import TrackingSetting


# Utils for loading saved data
def get_position(filename) -> int:
    parts = filename.split('/')[-1].split('_')
    position_str = parts[1][1:]
    return int(position_str)


def get_led(filename) -> LEDType:
    parts = filename.split('/')[-1].split('_')
    led_str = "LED_"+parts[0][3:6]+"_NM"
    return LEDType.from_string(led_str)


def get_filter(filename) -> FilterWheelType:
    if "F" not in filename:
        return FilterWheelType.FILTER
    else:
        parts = filename.split('/')[-1].split('_')
        fw_int = int(parts[5][1:])
        return FilterWheelType(fw_int)


def get_time(filename, filename_format='new'):
    parts = filename.split('/')[-1].split('_')
    if filename_format != 'new':
        time_str = parts[-2] + '_' + parts[-1].split('.')[0] + '.' + parts[-1].split('.')[1]
        return datetime.datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S.%f')
    else:
        time_str = parts[-2] + '_' + parts[-1][:8] + '.' + parts[-1].split('-')[-1].split('.')[0]
        return datetime.datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S.%f')


def get_pos_to_filename(
        folder_path: str,
        nposmax: int = 1,
        nmax: int = 10,
        burn_in: int = 0,
        sel_led: LEDType | None = LEDType.LED_450_NM,
        sel_fw: FilterWheelType | None = FilterWheelType.FILTER_527nm,
) -> dict[int, str]:
    filenames = [folder_path + "/" + f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')]
    all_positions = list(set([get_position(f) for f in filenames]))
    all_leds = list(set([get_led(f) for f in filenames]))
    all_filters = list(set([get_filter(f) for f in filenames]))
    print(f"Found positions={all_positions}, leds={all_leds}, and filters={all_filters} in {folder_path}")
    if nposmax is None:
        nposmax = len(all_positions)
    else:
        nposmax = min(nposmax, len(all_positions))
        all_positions = all_positions[:nposmax]

    def discard(l: list[str]) -> list[str]:
        return l if burn_in is None else l[burn_in:]

    filenames_by_pos = {
        pos: {
            led: {
                fw: discard(sorted(
                    [f for f in filenames if get_position(f) == pos and get_led(f) == led and get_filter(f) == fw],
                    key=get_time))
                for fw in all_filters
            }
            for led in all_leds
        }
        for pos in all_positions
    }

    min_length = min(
        len(lst)
        for d1 in filenames_by_pos.values()
        for d2 in d1.values()
        for lst in d2.values()
    )
    print(f"Maximum number of filenames per position: {min_length}")
    nmax = min_length if nmax is None else min(nmax, min_length)
    for d1 in filenames_by_pos.values():
        for d2 in d1.values():
            for key in d2:
                d2[key] = d2[key][:nmax]
    return {pos: filenames_by_pos[pos][sel_led][sel_fw] for pos in filenames_by_pos.values()}


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
        timepart = parts[-1].split(".")[0]
        time_str = parts[-2] + '_' + ":".join(timepart.split('-')[:-1]) + '.' + timepart.split('-')[-1]
        return datetime.datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S.%f')

    # folder_path = str(EVOMACHINE_DIR.parent / "data")
    folder_path = str(IMAGE_DIR / "UV_by_ROI_2025-02-14")
    filenames = [filename for filename in os.listdir(folder_path)
                 if filename.lower().endswith('.tiff') and "preproc" not in filename]
    filenames = sorted(filenames, key=lambda x: (get_position(x), get_time(x)))
    pos_to_filename = {get_position(filename): index for index, filename in enumerate(filenames)}
    filenames = [folder_path + "/" + f for f in filenames]
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
    save_path: str = str(DATA_DIR)
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
        channels_seg=[LEDType.LED_450_NM],
    )
    processor_config.preproc_enabled = True
    processor_config.roi_enabled = True
    processor_config.seg_enabled = True
    processor_config.track_enabled = True
    processor_config.lineage_enabled = False
    processor_config.channels_seg = LEDType.LED_450_NM
    processor_config.tracking_setting = TrackingSetting.MOTHERONLY
    processor_config.cfg_delta.drift_correction = True

    # Provide strategy that will be loaded by GUI
    # strategy: AbstractStrategy = BasicStrategy(save_path=save_path, cfg=processor_config)
    strategy: AbstractStrategy = UVStrategy(cfg=processor_config)

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

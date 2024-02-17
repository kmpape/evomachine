import copy
from enum import auto, Enum
import logging
import numpy as np
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import delta
from delta.config import Config

from evomachine.acquisition import AbstractCamera
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigDevice, ConfigFocus, ConfigImage, ConfigLED, EVO_GUI_LOGGING_LEVEL, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ErrorCode, ErrorContainer, ConfigError
from evomachine.positionrt import PositionRT
from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


class AutomatonQueueDataType(Enum):
    FOCUS_CURVES = auto()
    FOCUS_FRAMES = auto()
    INFO_TEXT = auto()
    PROCESS_DATA = auto()


class Automaton(threading.Thread):
    def __init__(
            self,
            cfg_device: ConfigDevice,  # TODO add reference channel to config
            cfg_image: ConfigImage,
            cfg_delta: delta.config.Config,
            cfg_focus: ConfigFocus,
            camera: AbstractCamera,
            dmd: DMDControl,
            strategy: AbstractStrategy,
            data_queue: Optional[queue.Queue] = None,
            use_segmentation: bool = False,
    ):
        super().__init__(name="Automaton")

        # TODO temporary switch
        assert not use_segmentation
        self.use_segmentation = use_segmentation

        self._cfg_delta: delta.config.Config = cfg_delta
        "Delta configuration object for image segmentation."
        self._cfg_device: ConfigDevice = cfg_device
        "Device configuration object defining geometry."
        self._cfg_image: ConfigImage = cfg_image
        "Image configuration object defining size and data type."
        self._cfg_focus: ConfigFocus = cfg_focus
        "Focus configuration object."
        self._curr_pos_id: int = 0
        "Current position."
        self._curr_period: int = 0
        "Incremented after completing one round of imaging the whole device."
        self._curr_step: int = 0
        "Incremented every time a picture is taken."
        self._camera: AbstractCamera = camera
        "Camera object which can be a real camera or a class that reads from the disk."
        self._dmd: DMDControl = dmd
        "DMDControl object to project images."
        self._pos_processor: List[PositionRT] = []
        "List of Delta objects to process the images."
        self._all_frames: List[np.ndarray[(int, int, int, int), ConfigImage.pxl_dtype]] = []
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._ref_frames: List[np.ndarray[(int, int, int), ConfigImage.pxl_dtype]] = []
        "List indexed by i_pos w. reference image array: channels x pxl_vert x pxl_horiz."
        self._is_initialised: bool = False
        "Set to true after initialisation."
        self._position_list_is_initialised: bool = False
        "Set to true after initialise_position_list."
        self._data_queue: Union[queue.Queue, None] = data_queue
        "Queue for communication with the GUI."
        self._cropping_boxes: Union[None, List[delta.utils.CroppingBox]] = None
        "List of cropping boxes applied to each FoV. Current: first box used for focus routine."

        self.positions: Dict[int, Coordinate] = {}
        "Dictionary position coordinates (after focus) initialised in initialise_position_list()."
        self.focus_curves: Dict[int, Tuple[np.typing.Array, np.typing.Array]] = {}
        "Dictionary containing (Z coordinates for focus, focus scores) at each position."
        self.focus_stack: Union[None, np.typing.Array] = None
        "3D array with focus frame of each position (3rd dimension)."
        self.focus_prev_stack: Union[None, np.typing.Array] = None
        "3D array with frame before focus for each position (3rd dimension)."
        self.focus_prev_z_coords: Union[None, np.typing.Array] = None
        "1D array with z coordinate before focus for each position."

        self.next_commands: List[AutomatonCommand] = []
        "List of commands to be executed at the next timestep."
        self.last_commands: List[AutomatonCommand] = []
        "List of commands executed at the last timestep."

        self._stop_event = threading.Event()
        "Stops Automaton loop. Event can be set through stop()."
        self.error_container: ErrorContainer = ErrorContainer()
        "Container for errors."

        self._strategy: AbstractStrategy = strategy
        "Strategy object defining actions taken at each timestep."

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logging.warning(msg=msg)
        else:
            logging.warning("No errors for automaton found.")
        self._camera.check_status()

    def fill_queue(
            self,
            queue_data_type: AutomatonQueueDataType,
            queue_data: Any,
            logging_level: int = logging.INFO,
    ):
        if (self._data_queue is not None) and (logging_level >= EVO_GUI_LOGGING_LEVEL):
            self._data_queue.put((queue_data_type, copy.copy(queue_data)))

    def initialise(
            self,
            positions: Optional[Union[Dict[int, Coordinate], List[Coordinate]]] = None,
            cropping_boxes: Optional[List[delta.utils.CroppingBox]] = None,
    ):
        logger.info("Automaton.initialise: starting...")
        self._is_initialised = False
        
        # Initialise devices
        self._camera.initialise()
        self._camera.studio.live().set_live_mode(False)

        # Validate position list, get focus coordinates, and broadcast position list to camera
        if positions is not None:
            logger.info(f"Automaton.initialise: initialising {len(positions)} positions...")
            self.initialise_position_list(positions=positions, cropping_boxes=cropping_boxes)
        elif not self._position_list_is_initialised:
            raise ConfigError(message="Automaton.initialise: position list is not initialised.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)
        else:
            logger.info(f"Automaton.initialise: found {len(self.positions)} initialised positions...")

        if not self._camera.set_pos_id_to_coordinate(pos_id_to_coordinate=self.positions):
            raise ConfigError(message="Automaton.initialise: failed to pass position list to camera.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)

        # Allocate variables
        self._all_frames = [
            np.empty((2, self._cfg_device.num_chan, self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz),
                     dtype=self._cfg_image.pxl_dtype)
            for _ in self.positions.keys()
        ]
        self._ref_frames = [
            np.empty((self._cfg_device.num_chan, self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz),
                     dtype=self._cfg_image.pxl_dtype)
            for _ in self.positions.keys()
        ]
        self._pos_processor = [
            PositionRT(
                position_nb=i,
                config=self._cfg_delta,
                cfg_image=self._cfg_image,
                verbose=self._cfg_device.image_processing_verbosity
            )
            for i in self.positions.keys()
        ]

        # Take reference frames on each channel
        logger.info(f"Automaton.initialise: taking reference frames on all channels and all positions...")
        for i_pos in self.positions.keys():
            self._camera.move_to_pos(i_pos=i_pos)
            for i_chan in range(self._cfg_device.num_chan):
                self._ref_frames[i_pos][i_chan, :, :] = self._camera.get_frame(i_chan=i_chan)
            if self.use_segmentation:
                self._pos_processor[i_pos].initialise(self._ref_frames[i_pos])
            self.increment_pos()
        self._camera.reset_counter()

        assert self._curr_pos_id == 0
        assert self._curr_period == 1  # Note that each ROI keeps track of _curr_period as well

        # Initialise strategy
        logger.info(f"Automaton.initialise: initialising strategy...")
        self._initialise_strategy()

        self._is_initialised = True

        logger.info(f"Automaton.initialise: initialisation done.")

    def initialise_position_list(
            self,
            positions: Union[Dict[int, Coordinate], List[Coordinate]],
            cfg_focus: Optional[ConfigFocus] = None,
            cropping_boxes: Optional[List[delta.utils.CroppingBox]] = None,
    ):
        self._position_list_is_initialised = False

        cropping_indices = None
        if cropping_boxes is not None:
            if (not cropping_boxes) or (not all(isinstance(c, delta.utils.CroppingBox) for c in cropping_boxes)):
                raise ConfigError(f"Automaton.initialise_position_list: invalid cropping boxes {cropping_boxes}",
                                  ErrorCode.ERROR_NOT_INITIALISED)
            self._cropping_boxes = cropping_boxes
            box0 = self._cropping_boxes[0]
            cropping_indices = ((box0.xtl, box0.xbr), (box0.ytl, box0.ybr))

        self._camera.disable_led()
        self._camera.studio.live().set_live_mode(False)

        # FIXME any DMD call from this thread crashes pygame
        # self._dmd.display_full()

        cfg_focus = self._cfg_focus if cfg_focus is None else cfg_focus
        if isinstance(positions, list):
            positions = {i_pos: coord.copy() for i_pos, coord in enumerate(positions)}
        else:
            if not all(isinstance(key, int) and key >= 0 for key in positions.keys()):
                raise ConfigError(message="Automaton.initialise_position_list: dictionary must have integer keys > 0.",
                                  error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            positions = {i_pos: coord.copy() for i_pos, coord in positions.items()}

        # Check positions
        z_coord = self._camera.get_coordinates(['Z'])['Z']
        self.focus_prev_z_coords = np.zeros(len(positions))
        for i_pos, coord in positions.items():
            if not coord.has_z():
                coord.z = z_coord
            if self._camera.coordinate_is_out_of_bounds(coordinate=coord):
                msg = f"Automaton.initialise_position_list: {Coordinate} for position {i_pos} is out of bounds " \
                      f"({self._camera.get_stage_limits()})."
                raise ConfigError(message=msg, error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            self.focus_prev_z_coords[i_pos] = coord.z

        # Run software focus on each position
        self.focus_curves = {i_pos: None for i_pos in positions.keys()}
        self.positions = {i_pos: None for i_pos in positions.keys()}
        self.focus_prev_stack = np.zeros((self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz, len(positions)))
        self.focus_stack = np.zeros((self._cfg_image.pxl_vert, self._cfg_image.pxl_horiz, len(positions)))
        for i_pos, coord in positions.items():
            logger.info(f"Automaton.initialise_position_list: initialising position {i_pos+1} of {len(positions)}.")
            if self.stopped():
                logger.warning("Automaton.initialise_position_list: stopping initialisation.")
                return
            self._camera.move_to(coordinate=coord, block=True)
            self._camera.software_focus(
                cfg_focus=cfg_focus,
                user_input_override=True,
                countdown_override=True,
                cropping_indices=cropping_indices,
            )
            self.focus_curves[i_pos] = (self._camera.focus_Z_coords, self._camera.focus_scores)
            self.focus_prev_stack[:, :, i_pos] = self._camera.focus_prev_image
            self.focus_stack[:, :, i_pos] = self._camera.get_software_focus_z_frame()
            coord.z = self._camera.get_software_focus_z_coord()
            self.positions[i_pos] = coord

        self.fill_queue(
            queue_data_type=AutomatonQueueDataType.FOCUS_CURVES,
            queue_data=(self.focus_curves, self.focus_prev_stack, self.focus_stack, self.focus_prev_z_coords),
            logging_level=logging.INFO,
        )

        self._camera.disable_led()
        # TODO any DMD call from this thread crashes pygame
        # self._dmd.display_none()
        self._position_list_is_initialised = True

    def _initialise_strategy(self):
        self.next_commands = self._strategy.initialise(
            field_of_views=self.positions,
            region_of_interests={i_pos: [i_roi for i_roi in range(len(self._pos_processor[i_pos].rois))]
                                 for i_pos in self.positions.keys()}
        )

        # Grab configuration object overrides TODO
        if self._strategy.path_to_save is not None:
            if not self._strategy.path_to_save.exists():
                raise ConfigError(f"Automaton.initialise: path_to_save provided by strategy is invalid "
                                  f"({self._strategy.path_to_save}).", ErrorCode.ERROR_DEVICE_CONFIG)
            self._cfg_device.path_to_save = self._strategy.path_to_save

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_pos_id + 1 == len(self.positions))
                             else self._curr_period)
        self._curr_pos_id = (self._curr_pos_id + 1) % len(self.positions)

    def process(self):
        self.fill_queue(
            queue_data_type=AutomatonQueueDataType.INFO_TEXT,
            queue_data=f"At period {self._curr_period}.",
            logging_level=logging.DEBUG,
        )

        # Execute requested commands in the given order
        for cmd in self.next_commands:
            if self.stopped():
                logger.warning(f"Automaton.process: stopping process at {str(cmd)}.")
                return
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                time.sleep(cmd.command_args)  # TODO implement our own function that reacts to stop event

            elif cmd.command_type == AutomatonCommandType.STOP:
                logger.warning("Automaton.process: Received STOP command. Shutting down.")
                self.stop()
                cmd.command_execution_time = time.time()
                return

            elif cmd.command_type == AutomatonCommandType.IMAGE:
                if self._camera.get_exposure() != cmd.command_args['exposure_time']:
                    self._camera.set_exposure(exposure_time=cmd.command_args['exposure_time'])
                # TODO actuate DMD
                self._take_image(channels=cmd.command_args['channels'], brightness=cmd.command_args['brightness'])
                if not cmd.command_args['segment']:
                    channels_int = [c.value for c in cmd.command_args['channels']]
                    cmd.command_data = self._all_frames[self._curr_pos_id][1, channels_int, :, :]
                else:
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data = {roi_id: None
                                        for roi_id in range(len(self._pos_processor[self._curr_pos_id].rois))}
                if cmd.command_args['save']:
                    for i_chan in cmd.command_args['channels']:
                        self._camera.save_frame(
                            frame=self._all_frames[self._curr_pos_id][1, i_chan.value, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_pos_id,
                        )

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                # TODO any DMD call from this thread crashes pygame
                # self._dmd.display_image(img=cmd.command_args['image'])
                self._camera.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])
                self._camera.disable_led()

            cmd.command_execution_time = time.time()

        self.fill_queue(
            queue_data_type=AutomatonQueueDataType.PROCESS_DATA,
            queue_data=(self._curr_pos_id, self.next_commands),
            logging_level=logging.INFO,
        )

        new_errors = list(self.error_container.error_list)  # TODO extract new errors
        self.last_commands = self.next_commands
        self.next_commands = self._strategy.callback(
            fov_id=self._curr_pos_id,
            data=self.last_commands,
            errors=new_errors,
        )

    def run(self):
        has_stopped = True
        if not self.is_initialised():
            raise ConfigError(message="Automaton.run: not initialised.", error_code=ErrorCode.ERROR_NOT_INITIALISED)
        while True:
            while not self.stopped():
                self.process()
            if has_stopped:
                logger.warning("Automaton.run: halting execution.")
                has_stopped = False
            time.sleep(1)

    def set_strategy(self, strategy: AbstractStrategy):
        self._strategy = strategy
        self._initialise_strategy()

    def _move_to_pos(self, pos_id: Union[int, None] = -1):
        if pos_id is None:
            return
        elif pos_id == -1:
            self.increment_pos()
            self._camera.move_to_pos(i_pos=self._curr_pos_id)
        else:
            self._camera.move_to_pos(i_pos=pos_id)
            self._curr_pos_id = pos_id

    def _take_image(self, channels: Optional[List[Union[int, ConfigLED]]] = None, brightness: Union[int, List[int]] = 100):
        if (channels is None) or not channels:
            channels = list(range(self._cfg_device.num_chan))
        channels = [c.value if isinstance(c, ConfigLED) else c for c in channels]
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        for i, i_chan in enumerate(channels):
            self._all_frames[self._curr_pos_id][0, i_chan, :, :] = self._all_frames[self._curr_pos_id][1, i_chan, :, :]
            self._all_frames[self._curr_pos_id][1, i_chan, :, :] = self._camera.get_frame(
                i_chan=i_chan,
                brightness=brightness[i],
            )

    def _process_position(self):
        self._pos_processor[self._curr_pos_id].process_new_frame(
            new_frame=self._all_frames[self._curr_pos_id][1, :, :, :]  # NOTE: frame passed by reference
        )

    def get_period(self) -> int:
        return self._curr_period

    def get_pos_id(self) -> int:
        return self._curr_pos_id

    def get_frame(self, i_pos: int, i_chan: int) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        return self._all_frames[i_pos][1, i_chan, :, :]

    def is_initialised(self):
        return self._is_initialised

    def reset(self):
        self._position_list_is_initialised = False
        self._is_initialised = False
        self.positions = {}

    def sleep(self, duration: float):
        now = time.perf_counter()
        end = now + duration
        while (now < end) and not self.stopped():
            now = time.perf_counter()

    def restart(self):
        self._stop_event.clear()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()



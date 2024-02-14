import logging
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import delta
from delta.config import Config

from evomachine.acquisition import AbstractCamera
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigDevice, ConfigFocus, ConfigImage, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ErrorCode, ErrorContainer, ConfigError
from evomachine.positionrt import PositionRT

from evomachine.strategy import AbstractStrategy


logger = get_logger(name=__name__)


class Automaton(threading.Thread):
    def __init__(
            self,
            cfg_device: ConfigDevice,
            cfg_image: ConfigImage,
            cfg_delta: delta.config.Config,
            cfg_focus: ConfigFocus,
            camera: AbstractCamera,
            dmd: DMDControl,
            strategy: AbstractStrategy,
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

        self.positions: Dict[int, Coordinate] = {}
        "Dictionary position coordinates (after focus) initialised in initialise_position_list()."
        self.focus_curves: Dict[int, Tuple[np.typing.Array, np.typing.Array]] = {}
        "Dictionary containing (Z coordinates for focus, focus scores) at each position."

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

    def initialise(self, positions: Union[Dict[int, Coordinate], List[Coordinate]]):
        
        # Initialise devices
        self._camera.initialise()

        # Validate position list, get focus coordinates, and broadcast position list to camera
        self.initialise_position_list(positions=positions)
        if not self._camera.set_pos_id_to_coordinate(pos_id_to_coordinate=self.positions):
            raise ConfigError(message="Automaton.initialise: failed to set position list.",
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

    def initialise_position_list(
            self,
            positions: Union[Dict[int, Coordinate], List[Coordinate]],
            cfg_focus: Optional[ConfigFocus] = None,
    ):
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
        for i_pos, coord in positions.items():
            if not coord.has_z():
                coord.z = z_coord
            if self._camera.coordinate_is_out_of_bounds(coordinate=coord):
                msg = f"Automaton.initialise_position_list: {Coordinate} for position {i_pos} is out of bounds " \
                      f"({self._camera.get_stage_limits()})."
                raise ConfigError(message=msg, error_code=ErrorCode.ERROR_DEVICE_CONFIG)

        # Run software focus on each position
        self.focus_curves = {i_pos: None for i_pos in positions.keys()}
        self.positions = {i_pos: None for i_pos in positions.keys()}
        for i_pos, coord in positions.items():
            self._camera.move_to(coordinate=coord, block=True)
            self._camera.software_focus(cfg_focus=cfg_focus, user_input_override=True, countdown_override=True)
            self.focus_curves[i_pos] = (self._camera.focus_Z_coords, self._camera.focus_scores)
            coord.z = self._camera.get_software_focus_z_coord()
            self.positions[i_pos] = coord

        self._is_initialised = True

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logging.warning(msg=msg)
        else:
            logging.warning("No errors for automaton found.")
        self._camera.check_status()

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_pos_id + 1 == len(self.positions))
                             else self._curr_period)
        self._curr_pos_id = (self._curr_pos_id + 1) % len(self.positions)

    def process(self):
        # Execute requested commands in the given order
        for cmd in self.next_commands:
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                time.sleep(cmd.command_args)

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
                    cmd.command_data = self._all_frames[self._curr_pos_id][1, cmd.command_args['channels'], :, :]
                else:
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data = {roi_id: None
                                        for roi_id in range(len(self._pos_processor[self._curr_pos_id].rois))}
                if cmd.command_args['save']:
                    for i_chan in cmd.command_args['channels']:
                        self._camera.save_frame(frame=self._all_frames[self._curr_pos_id][1, i_chan, :, :])

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                self._dmd.display_image(img=cmd.command_args['image'])
                self._camera.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])
                self._camera.disable_led()

            cmd.command_execution_time = time.time()

        new_errors = list(self.error_container.error_list)  # TODO extract new errors
        self.last_commands = self.next_commands
        self.next_commands = self._strategy.callback(
            fov_id=self._curr_pos_id,
            data=self.last_commands,
            errors=new_errors,
        )

    def run(self):
        if not self.is_initialised():
            raise ConfigError(message="Automaton.run: not initialised.", error_code=ErrorCode.ERROR_NOT_INITIALISED)

        while not self.stopped():
            self.process()

    def _move_to_pos(self, pos_id: Union[int, None] = -1):
        if pos_id is None:
            return
        elif pos_id == -1:
            self.increment_pos()
            self._camera.move_to_pos(i_pos=self._curr_pos_id)
        else:
            self._camera.move_to_pos(i_pos=pos_id)
            self._curr_pos_id = pos_id

    def _take_image(self, channels: Optional[List[int]] = None, brightness: Union[int, List[int]] = 100):
        if (channels is None) or not channels:
            channels = list(range(self._cfg_device.num_chan))
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        for i_chan in channels:
            self._all_frames[self._curr_pos_id][0, i_chan, :, :] = self._all_frames[self._curr_pos_id][1, i_chan, :, :]
            self._all_frames[self._curr_pos_id][1, i_chan, :, :] = self._camera.get_frame(
                i_chan=i_chan,
                brightness=brightness[i_chan],
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

    def sleep(self, duration: float):
        now = time.perf_counter()
        end = now + duration
        while (now < end) and not self.stopped():
            now = time.perf_counter()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()



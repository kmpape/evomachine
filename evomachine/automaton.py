import copy
from enum import auto, Enum
import logging
from multiprocessing import Event, Queue
import numpy as np
import queue
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import delta
from delta.config import Config

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import ConfigFocus, ConfigImageProcessor, EVO_GUI_LOGGING_LEVEL, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ErrorCode, ErrorContainer, ConfigError
from evomachine.positionrt import PositionRT
from evomachine.strategy import AbstractStrategy
from evomachine.utils import EvoCroppingBox
from evomachine.evotypes import AutomatonCommandType, ImageConfigType, LEDType


logger = get_logger(name=__name__)


class Automaton:
    def __init__(
            self,
            camera: AbstractCamera,
            cfg_processor: ConfigImageProcessor,
            dmd: DMDControl,
            strategy: AbstractStrategy,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            process_q: Optional[Queue] = None,
            gui_to_automaton_q: Optional[Queue] = None,
            automaton_to_gui_q: Optional[Queue] = None,
            use_segmentation: bool = False,
            queue_timeout: float = 0,
            run_timeout: float = 0,
    ):
        # FIXME temporary switch
        assert not use_segmentation
        self.use_segmentation = use_segmentation

        self._cfg: ConfigImageProcessor = cfg_processor
        "Delta configuration object for image segmentation."
        self._channel_to_index: Dict[LEDType, int] = self._cfg.channel_to_index
        "Dictionary mapping LEDType to channel index in 3D arrays."
        self._curr_pos_id: int = 0
        "Current position."
        self._curr_period: int = 0
        "Incremented after completing one round of imaging the whole device."
        self._curr_step: int = 0
        "Incremented every time a picture is taken."
        self.cam: AbstractCamera = camera
        "Camera object which can be a real camera or a class that reads from the disk."
        self._dmd: DMDControl = dmd
        "DMDControl object to project images."
        self._devices_initialised: bool = False
        "Set to true after initialise_devices."
        self._pos_processor: List[PositionRT] = []
        "List of Delta objects to process the images."
        self._all_frames: List[np.ndarray] = []
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._ref_frames: List[np.ndarray] = []
        "List indexed by i_pos w. reference image array: channels x pxl_vert x pxl_horiz."
        self._fov_list_is_initialised: bool = False
        "Set to true after initialise_field_of_view_list."
        self._focus_is_initialised: bool = False
        "Set to true after initialise_position_list."
        self._position_processors_is_initialised: List[bool] = []
        "Set to true after initialise_position_processor."
        self._reference_frames_is_initialised: bool = False
        "Set to true after initialise_reference_frames."
        self._strategy_is_initialised: bool = False
        "Set to true after _initialise_strategy."

        self._fovs: Dict[int, Coordinate] = {}
        "Dictionary containing coordinates of field of views."
        self._cropping_boxes: Union[None, Dict[int, List[EvoCroppingBox]]] = None
        "List of cropping boxes applied to each FoV. One cropping box yields one position."
        self._fov_to_pos: Dict[int, List[int]] = {}
        "Dictionary mapping FoV to positions. Initialised in initialise_field_of_view_list()."
        self._pos_to_fov: Dict[int, int] = {}
        "Dictionary mapping position to FoV. Initialised in initialise_field_of_view_list()."
        self._pos_to_fov_index: Dict[int, int] = {}
        "Dictionary mapping position to index in _cropping_boxes. Initialised in initialise_field_of_view_list()."
        self._pos_to_roi: Dict[int, List[int]] = {}
        "Dictionary mapping position to RoI. Initialised in initialise_position_processor."

        self.focus_curves: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        "Dictionary containing (Z coordinates for focus, focus scores) at each position."
        self.focus_stack: Union[None, np.ndarray] = None
        "3D array with focus frame of each position (3rd dimension)."
        self.focus_prev_stack: Union[None, np.ndarray] = None
        "3D array with frame before focus for each position (3rd dimension)."
        self.focus_prev_z_coords: Union[None, np.ndarray] = None
        "1D array with z coordinate before focus for each position."

        self.next_commands: List[AutomatonCommand] = []
        "List of commands to be executed at the next timestep."
        self.last_commands: List[AutomatonCommand] = []
        "List of commands executed at the last timestep."

        self._start_strategy_event = start_strategy_event
        "Starts Automaton strategy loop."
        self._stop_strategy_event = stop_strategy_event
        "Stops Automaton strategy loop."
        self._stop_event = stop_event
        "Stops Automaton loop. Event can be set through stop()."
        self._shutdown_event = shutdown_event
        "Shuts down Automaton "

        self._process_q: Union[queue.Queue, None] = process_q
        "Queue for communication with the GUI."
        self._gui_to_automaton_q: Union[queue.Queue, None] = gui_to_automaton_q
        "Queue for communication with the GUI."
        self._automaton_to_gui_q: Union[queue.Queue, None] = automaton_to_gui_q
        "Queue for communication with the GUI."
        self.queue_timeout: float = queue_timeout
        "Timeout for polling all queues."
        self.run_timeout: float = run_timeout
        "Timeout after each iteration."

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
        self.cam.check_status()

    def fill_queue(
            self,
            queue_data_type: AutomatonCommandType,
            queue_data: AutomatonCommand,
            logging_level: int = logging.INFO,
    ):
        if (self._process_q is not None) and (logging_level >= EVO_GUI_LOGGING_LEVEL):
            self._process_q.put((queue_data_type, copy.copy(queue_data)))

    def initialise(
            self,
            field_of_views: Optional[Dict[int, Coordinate]] = None,
            cropping_boxes: Optional[Dict[int, List[EvoCroppingBox]]] = None,
    ):
        logger.info("Automaton.initialise: starting...")
        
        # Initialise devices
        if not self._devices_initialised:
            self.initialise_devices()

        # Initialise field of views
        if field_of_views is not None:
            logger.info(f"Automaton.initialise: initialising {len(field_of_views)} FoVs...")
            self.initialise_field_of_view_list(field_of_views=field_of_views, cropping_boxes=cropping_boxes)
        elif not self._fov_list_is_initialised:
            raise ConfigError(message="Automaton.initialise: FoV list is not initialised.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)
        else:
            logger.info(f"Automaton.initialise: found {len(self._fovs)} initialised positions...")

        # Initialise focus
        self.initialise_fov_focus()

        # Take reference frames on each channel
        logger.info(f"Automaton.initialise: taking reference frames on all channels and all positions...")
        self.initialise_reference_frames()

        # Initialise position processor
        if self.use_segmentation:
            self.initialise_position_processor()

        assert self._curr_pos_id == 0
        assert self._curr_period == 1  # Note that each ROI keeps track of _curr_period as well

        # Initialise strategy
        logger.info(f"Automaton.initialise: initialising strategy...")
        self._initialise_strategy()

        logger.info(f"Automaton.initialise: initialisation done.")

    def initialise_devices(self):
        logger.info("Automaton.initialise_devices")
        self.cam.initialise()
        if isinstance(self.cam, EvoCamera):
            self.cam.studio.live().set_live_mode(False)
        self.cam.set_exposure(exposure_time=self.cam.cfg.default_exposure_time)
        self._dmd.initialise()
        self._devices_initialised = True

    def initialise_position_processor(
            self,
            which: Optional[int] = None,
            rotation: Optional[float] = None,
            roi_boxes: Optional[Union[None, List[delta.utils.CroppingBox]]] = None,
    ):
        if not self._fov_list_is_initialised or not self._reference_frames_is_initialised:
            logging.warning("Automaton.initialise_position_processor: position list is not initialised.")
            raise ConfigError(message="Automaton.initialise_position_processor: position list is not initialised.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)
        if which is None:
            logger.info("Automaton.initialise_position_processor: initialising all position processors.")
            for i in range(len(self._pos_processor)):
                logger.debug(f"Automaton.initialise_position_processor: initialising position processor {i}.")
                if self.use_segmentation:
                    self._pos_processor[i].initialise(
                        reference=self.normalise_frame(self._ref_frames[i]),  # TODO need to crop frames
                        channel_rot=self._channel_to_index[self._cfg.channel_rot],
                        channel_roi=self._channel_to_index[self._cfg.channel_roi],
                        rotate=rotation,
                        roi_boxes=roi_boxes
                    )
                    self._position_processors_is_initialised[i] = True
                    self.fill_queue(
                        queue_data_type=AutomatonCommandType.ROI_DATA,
                        queue_data=CommandFactory.command_roi_data(
                            fov_id=i,
                            rotation=self._pos_processor[i].rotate,
                            roi_boxes=self._pos_processor[i].roi_boxes,
                        ),
                        logging_level=logging.INFO,
                    )
                else:
                    self._position_processors_is_initialised[i] = True
                    self.fill_queue(
                        queue_data_type=AutomatonCommandType.ROI_DATA,
                        queue_data=CommandFactory.command_roi_data(
                            fov_id=i,
                            rotation=0,
                            roi_boxes=[],
                        ),
                        logging_level=logging.INFO,
                    )
            self._pos_to_roi = {i_pos: [i_roi for i_roi in range(len(self._pos_processor[i_pos].rois))]
                                for i_pos in range(len(self._pos_processor))}
        else:
            logger.info(f"Automaton.initialise_position_processor: initialising position processor {which}.")
            if self.use_segmentation:
                self._pos_processor[which].initialise(self._ref_frames[which], rotate=rotation, roi_boxes=roi_boxes)
                self.fill_queue(
                    queue_data_type=AutomatonCommandType.ROI_DATA,
                    queue_data=CommandFactory.command_roi_data(
                        fov_id=which,
                        rotation=self._pos_processor[which].rotate,
                        roi_boxes=self._pos_processor[which].roi_boxes
                    ),
                    logging_level=logging.INFO,
                )
            else:
                self.fill_queue(
                    queue_data_type=AutomatonCommandType.ROI_DATA,
                    queue_data=CommandFactory.command_roi_data(fov_id=which, rotation=0, roi_boxes=[]),
                    logging_level=logging.INFO,
                )
            self._position_processors_is_initialised[which] = True
            self._pos_to_roi[which] = [i_roi for i_roi in range(len(self._pos_processor[which].rois))]

    def initialise_field_of_view_list(
            self,
            field_of_views: Dict[int, Coordinate],
            cropping_boxes: Optional[Dict[int, List[EvoCroppingBox]]] = None,
    ):
        self._fov_list_is_initialised = False
        self._focus_is_initialised = False
        self._position_processors_is_initialised = []
        self._strategy_is_initialised = False
        self._reference_frames_is_initialised = False

        self._fovs = field_of_views
        if cropping_boxes is not None:
            if not field_of_views.keys() == cropping_boxes.keys():
                raise ConfigError(f"Automaton.initialise_position_list: cropping box keys do not match field_of_views.",
                                  ErrorCode.ERROR_DEVICE_CONFIG)
            self._cropping_boxes = cropping_boxes
            pos_id = 0
            for fov_id, fov_boxes in self._cropping_boxes.items():
                self._fov_to_pos[fov_id] = [pos_id + i for i in range(len(fov_boxes))]
                for i in range(len(fov_boxes)):
                    self._pos_to_fov[pos_id] = fov_id
                    self._pos_to_fov_index[pos_id] = i
                    pos_id += 1
        else:
            self._cropping_boxes = {fov_id: [EvoCroppingBox.full(self.cam.cfg.image.shape)]
                                    for fov_id in field_of_views.keys()}
            self._fov_to_pos = {fov_id: [fov_id] for fov_id in field_of_views.keys()}
            self._pos_to_fov = {fov_id: fov_id for fov_id in field_of_views.keys()}
            self._pos_to_fov_index = {fov_id: 0 for fov_id in field_of_views.keys()}

        # Check positions
        z_coord = self.cam.get_coordinates(['Z'])['Z']
        self.focus_prev_z_coords = np.zeros(len(self._fovs))
        for i_fov, coord in enumerate(self._fovs.values()):
            if not coord.has_z():
                coord.z = z_coord
            if self.cam.coordinate_is_out_of_bounds(coordinate=coord):
                msg = f"Automaton.initialise_field_of_view_list: {Coordinate} for FoV {i_fov} is out of bounds " \
                      f"({self.cam.get_stage_limits()})."
                raise ConfigError(message=msg, error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            self.focus_prev_z_coords[i_fov] = coord.z

        # Allocate variables
        self._all_frames = [
            np.empty((2, len(self._cfg.channels), *self.cam.cfg.image.shape), dtype=self.cam.cfg.image.pxl_dtype)
            for _ in self._fovs
        ]
        self._ref_frames = [
            np.empty((len(self._cfg.channels), *self.cam.cfg.image.shape), dtype=self.cam.cfg.image.pxl_dtype)
            for _ in self._fovs
        ]
        self._pos_processor = []
        for pos_id, fov_ind in self._pos_to_fov_index.items():
            box = self._cropping_boxes[pos_id][fov_ind]
            pos_image_cfg = ImageConfigType(
                pxl_horiz=box.shape[1], pxl_vert=box.shape[0], pxl_dtype=self.cam.cfg.image.pxl_dtype,
            )
            self._pos_processor.append(
                PositionRT(
                    position_nb=pos_id,
                    config=self._cfg.cfg_delta,
                    processor_config=self._cfg,
                    image_config=pos_image_cfg,
                )
            )
        if not self.cam.set_pos_id_to_coordinate(pos_id_to_coordinate=self._fovs):
            raise ConfigError(message="Automaton.initialise: failed to pass position list to camera.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)
        logger.info(f"Automaton.initialise_field_of_view_list: initialised {len(self._fovs)} FoVs with"
                    f" {len(self._pos_to_fov)} positions.")
        self._position_processors_is_initialised = [False for _ in self._pos_to_fov.keys()]
        self._fov_list_is_initialised = True
        self.fill_queue(
            queue_data_type=AutomatonCommandType.FOV_DATA,
            queue_data=CommandFactory.command_fov_data(
                fovs=self._fovs,
                fov_to_pos=self._fov_to_pos,
                pos_to_fov_index=self._pos_to_fov_index,
                cropping_boxes=self._cropping_boxes,
            ),
            logging_level=logging.INFO,
        )

    def initialise_fov_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
    ):
        if not self._fov_list_is_initialised:
            raise ConfigError("Automaton.initialise_fov_focus: FoV list is not initialised.", ErrorCode.ERROR_CONFIG)

        self.cam.disable_led()
        self.cam.disable_live_mode()

        self._dmd.display_full()

        cfg_focus = self.cam.cfg.focus if cfg_focus is None else cfg_focus
        self.focus_curves = {i_fov: None for i_fov in self._fovs.keys()}
        self.focus_prev_stack = np.zeros((*self.cam.cfg.image.shape, len(self._fovs)))
        self.focus_stack = np.zeros((*self.cam.cfg.image.shape, len(self._fovs)))
        logger.info(f"Automaton.initialise_fov_focus with configuration {cfg_focus}")
        for i_fov, coord in self._fovs.items():
            logger.info(f"Automaton.initialise_fov_focus: initialising FoV {i_fov+1} of {len(self._fovs)}.")
            if self.stopped():
                logger.warning("Automaton.initialise_position_list: stopping initialisation.")
                return
            self.cam.move_to(coordinate=coord, block=True)
            self.cam.software_focus(
                cfg_focus=cfg_focus,
                user_input_override=True,
                countdown_override=True,
                cropping_box=None,
            )
            self.focus_curves[i_fov] = (self.cam.focus_Z_coords, self.cam.focus_scores)
            self.focus_prev_stack[:, :, i_fov] = self.cam.focus_prev_image
            self.focus_stack[:, :, i_fov] = self.cam.get_software_focus_z_frame()
            coord.z = self.cam.get_software_focus_z_coord()

        cmd = CommandFactory.command_focus_data(
            focus_curves=self.focus_curves,
            focus_prev_stack=self.focus_prev_stack,
            focus_stack=self.focus_stack,
            focus_prev_z_coords=self.focus_prev_z_coords,
            fovs=self._fovs,
        )
        self.fill_queue(queue_data_type=AutomatonCommandType.FOCUS_DATA, queue_data=cmd, logging_level=logging.INFO)

        self.cam.disable_led()
        # TODO any DMD call from this thread crashes pygame
        # self._dmd.display_none()
        self._focus_is_initialised = True

    def initialise_reference_frames(self):
        logger.info(f"Automaton.initialise_reference_frames: "
                    f"Imaging {len(self._fovs.keys())} on {self._channel_to_index.keys()}.")
        for i_fov in self._fovs.keys():
            self.cam.move_to_pos(i_pos=i_fov)
            for channel_type, ind in self._channel_to_index.items():
                self._ref_frames[i_fov][ind, :, :] = self.cam.get_frame(i_chan=channel_type)
            self.increment_pos()
        self.cam.reset_counter()
        self._reference_frames_is_initialised = True
        cmd = CommandFactory.command_ref_data(ref_frames=self._ref_frames)
        self.fill_queue(queue_data_type=AutomatonCommandType.REF_DATA, queue_data=cmd, logging_level=logging.INFO)

    def _initialise_strategy(self):
        if not self._fov_list_is_initialised and not all(self._position_processors_is_initialised):
            raise ConfigError(message="Automaton._initialise_strategy: not initialised.",
                              error_code=ErrorCode.ERROR_NOT_INITIALISED)
        self.next_commands = self._strategy.initialise(
            field_of_views=self._fovs,
            positions=self._fov_to_pos,
            region_of_interests=self._pos_to_roi,
            config_camera=self.cam.cfg,
        )

        # Grab configuration object overrides TODO
        if self._strategy.path_to_save is not None:
            if not self._strategy.path_to_save.exists():
                raise ConfigError(f"Automaton.initialise: path_to_save provided by strategy is invalid "
                                  f"({self._strategy.path_to_save}).", ErrorCode.ERROR_DEVICE_CONFIG)
            self.cam.cfg.path_to_save = self._strategy.path_to_save

        self._strategy_is_initialised = True

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_pos_id + 1 == len(self._fovs))
                             else self._curr_period)
        self._curr_pos_id = (self._curr_pos_id + 1) % len(self._fovs)

    def normalise_frame(self, frame: np.ndarray, channels: Optional[int] = None) -> np.ndarray:
        # TODO implement normalisation
        norm_frame = frame.copy()
        if channels is None:
            channels = list(range(frame.shape[0]))
        for c in channels:
            norm_frame[c, :, :] = (norm_frame[c, :, :] - norm_frame[c, :, :].min()) / np.ptp(norm_frame[c, :, :])
        return norm_frame

    def override_parameter(self, fov_id: int, pos_id: int, param_name: str, param_value: Any):
        logger.info(f"Automaton.override_parameter: setting {param_name} to {param_value} for "
                    f"FoV {fov_id} and pos {pos_id}.")
        avail: List[str] = ["z_pos", "rotation"]
        if param_name not in avail:
            raise ConfigError(f"Automaton.override_parameter: {param_name} is not an available override. {avail}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if fov_id not in self._fovs.keys():
            raise ConfigError(f"Automaton.override_parameter: {fov_id} is not a valid FoV id. {self._fovs.keys()}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if pos_id not in self._pos_to_fov.keys():
            raise ConfigError(f"Automaton.override_parameter: {pos_id} is not a valid position id. {self._pos_to_fov.keys()}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if param_name == "z_pos":
            tmp_fov = copy.deepcopy(self._fovs)
            tmp_fov[fov_id].z = float(param_value)
            if not self.cam.set_pos_id_to_coordinate(pos_id_to_coordinate=tmp_fov):
                raise ConfigError(message="Automaton.override_parameter: failed to pass position list to camera.",
                                  error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            logger.info(f"Automaton.override_parameter: changing Z from {self._fovs[fov_id].z} to {tmp_fov[fov_id].z}.")
            self._fovs[fov_id].z = float(param_value)
        elif param_name == "rotation":
            logger.info(f"Automaton.override_parameter: changing rotation from {param_value} "
                        f"to {self._pos_processor[pos_id].rotate}.")
            self._pos_processor[pos_id].rotate = float(param_value)

    def _gui_process(self):
        # logger.debug("Polling _gui_to_automaton_q and filling _automaton_to_gui_q")
        while not self._gui_to_automaton_q.empty():
            try:
                req_id, req_str, kwargs_dict = self._gui_to_automaton_q.get(timeout=self.queue_timeout)
                logger.debug(f"Automaton received {req_id} and {req_str} from _gui_to_automaton_q")
                req_args = ",".join(f"{k}=kwargs_dict['{k}']" for k in kwargs_dict.keys())
                try:
                    req_ans = eval(f"{req_str}({req_args})")
                    self._automaton_to_gui_q.put((req_id, req_ans))
                except Exception as e:
                    logger.error(f"Automaton._gui_process: failed to execute {req_str}({req_args}). "
                                 f"Received {str(e)}.")
                    traceback.print_exc()
                    self._automaton_to_gui_q.put((req_id, e))
            except queue.Empty:
                pass

    def _process(self):
        self.fill_queue(AutomatonCommandType.INFO_TEXT,
                        CommandFactory.command_info_text(f"At period {self._curr_period}."),
                        logging.DEBUG)

        # Execute requested commands in the given order
        for cmd in self.next_commands:
            if self.stopped():
                logger.warning(f"Automaton.process: stopping process at {str(cmd)}.")
                return
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                self.sleep(cmd.command_args)  # TODO implement our own function that reacts to stop event

            elif cmd.command_type == AutomatonCommandType.STOP:
                logger.warning("Automaton.process: Received STOP command. Shutting down.")
                self.stop()
                cmd.command_execution_time = time.time()
                return

            elif cmd.command_type == AutomatonCommandType.IMAGE:
                if self.cam.get_exposure() != cmd.command_args['exposure_time']:
                    self.cam.set_exposure(exposure_time=cmd.command_args['exposure_time'])
                # TODO actuate DMD
                self._take_image(channels=cmd.command_args['channels'], brightness=cmd.command_args['brightness'])
                if not cmd.command_args['segment']:
                    channels_int = [self._channel_to_index[c] for c in cmd.command_args['channels']]
                    cmd.command_data = self._all_frames[self._curr_pos_id][1, channels_int, :, :]
                else:
                    # TODO make sure segmentation channel is in cmd.command_args['channels']
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data = {roi_id: None
                                        for roi_id in range(len(self._pos_processor[self._curr_pos_id].rois))}
                if cmd.command_args['save']:
                    for i_chan in cmd.command_args['channels']:
                        self.cam.save_frame(
                            frame=self._all_frames[self._curr_pos_id][1, i_chan.value, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_pos_id,
                        )

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                # TODO any DMD call from this thread crashes pygame
                # self._dmd.display_image(img=cmd.command_args['image'])
                self.cam.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])  # TODO disable with timer
                self.cam.disable_led()

            cmd.command_execution_time = time.time()
            cmd.fov_id = self._curr_pos_id

            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=cmd,
                logging_level=logging.INFO,
            )

        new_errors = list(self.error_container.error_list)  # TODO extract new errors
        self.last_commands = self.next_commands
        self.next_commands = self._strategy.callback(
            fov_id=self._curr_pos_id,
            data=self.last_commands,
            errors=new_errors,
        )

    def _finalise_process(self):
        self.fill_queue(AutomatonCommandType.INFO_TEXT,
                        CommandFactory.command_info_text(f"At period {self._curr_period}. Finalising."),
                        logging.INFO)

        self.last_commands = self.next_commands
        self.next_commands = self._strategy.finalise()

        # FIXME with current events loop below will return
        for cmd in self.next_commands:
            if self.stopped():
                logger.warning(f"Automaton.process: stopping process at {str(cmd)}.")
                return
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                self.sleep(cmd.command_args)  # TODO implement our own function that reacts to stop event

            elif cmd.command_type == AutomatonCommandType.STOP:
                logger.warning("Automaton.process: Received STOP command. Shutting down.")
                self.stop()
                cmd.command_execution_time = time.time()
                return

            elif cmd.command_type == AutomatonCommandType.IMAGE:
                if self.cam.get_exposure() != cmd.command_args['exposure_time']:
                    self.cam.set_exposure(exposure_time=cmd.command_args['exposure_time'])
                # TODO actuate DMD
                self._take_image(channels=cmd.command_args['channels'], brightness=cmd.command_args['brightness'])
                if not cmd.command_args['segment']:
                    channels_int = [self._channel_to_index[c] for c in cmd.command_args['channels']]
                    cmd.command_data = self._all_frames[self._curr_pos_id][1, channels_int, :, :]
                else:
                    # TODO make sure segmentation channel is in cmd.command_args['channels']
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data = {roi_id: None
                                        for roi_id in range(len(self._pos_processor[self._curr_pos_id].rois))}
                if cmd.command_args['save']:
                    for i_chan in cmd.command_args['channels']:
                        self.cam.save_frame(
                            frame=self._all_frames[self._curr_pos_id][1, i_chan.value, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_pos_id,
                        )

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                # TODO any DMD call from this thread crashes pygame
                # self._dmd.display_image(img=cmd.command_args['image'])
                self.cam.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])
                self.cam.disable_led()

            cmd.command_execution_time = time.time()
            cmd.fov_id = self._curr_pos_id

            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=cmd,
                logging_level=logging.INFO,
            )

    def run(self):
        logger.info("Automaton.run: starting...")
        while not self.has_shutdown():

            logger.info("Automaton.run: Starting GUI loop.")
            if not self._devices_initialised:
                self.initialise_devices()
            has_stopped = True
            while not self.strategy_has_started():
                while (not self.stopped()) and (not self.strategy_has_started()):
                    self._gui_process()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
                if has_stopped:
                    logger.warning("Automaton.run: halting GUI execution.")
                    has_stopped = False
            logger.info("Automaton.run: Leaving GUI loop.")

            if not self.strategy_has_stopped():
                if not self._strategy_is_initialised:
                    self._initialise_strategy()
                if not self.is_initialised():
                    logger.error(f"Automaton.run: Attempt to start strategy before intialisation. Current status:"
                                 f"_strategy_is_initialised={self._strategy_is_initialised}, "
                                 f"_reference_frames_is_initialised={self._reference_frames_is_initialised}, "
                                 f"_fov_list_is_initialised={self._fov_list_is_initialised}, "
                                 f"_position_processors_is_initialised={all(self._position_processors_is_initialised)}")
                    raise ConfigError(message="Automaton.run strategy: not initialised.",
                                      error_code=ErrorCode.ERROR_NOT_INITIALISED)
                logger.info(f"Automaton.run: Starting strategy loop. Moving to fov {self._curr_pos_id}.")
                self._move_to_pos(pos_id=self._curr_pos_id)
                self._dmd.display_full()  # FIXME temporary statement

            has_stopped = True
            while not self.strategy_has_stopped():
                while (not self.stopped()) and (not self.strategy_has_stopped()):
                    self._process()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
                if has_stopped:
                    logger.warning("Automaton.run strategy: halting execution.")
                    has_stopped = False
            logger.info(f"Automaton.run: Leaving strategy loop. Current fov {self._curr_pos_id}.")

            if self.is_initialised():
                logger.info("Automaton.run: finalising strategy.")
                self._finalise_process()
        logger.info("Automaton.run: shutting down.")
        self._dmd.finalise()

    def set_strategy(self, strategy: AbstractStrategy):
        self._strategy = strategy
        self._initialise_strategy()

    def _move_to_pos(self, pos_id: Union[int, None] = -1):
        if pos_id is None:
            return
        elif pos_id == -1:
            self.increment_pos()
            self.cam.move_to_pos(i_pos=self._curr_pos_id)
        else:
            self.cam.move_to_pos(i_pos=pos_id)
            self._curr_pos_id = pos_id

    def _take_image(self, channels: Optional[List[LEDType]] = None, brightness: Union[int, List[int]] = 100):
        if (channels is None) or not channels:
            channels = self._cfg.channels
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        for i, channel in enumerate(channels):
            i_chan = self._channel_to_index[channel]
            self._all_frames[self._curr_pos_id][0, i_chan, :, :] = self._all_frames[self._curr_pos_id][1, i_chan, :, :]
            self._all_frames[self._curr_pos_id][1, i_chan, :, :] = self.cam.get_frame(
                i_chan=channel,
                brightness=brightness[i],
            )

    def _process_position(self):
        self._pos_processor[self._curr_pos_id].process_new_frame(
            new_frame=self._all_frames[self._curr_pos_id][1, :, :, :],
        )

    def get_channel_to_index(self) -> Dict[LEDType, int]:
        return {key: value for key, value in self._channel_to_index.items()}

    def get_period(self) -> int:
        return self._curr_period

    def get_pos_id(self) -> int:
        return self._curr_pos_id

    def get_frame(self, i_pos: int, channel: LEDType) -> np.ndarray:
        return self._all_frames[i_pos][1, self._channel_to_index[channel], :, :]

    def is_initialised(self):
        return self._strategy_is_initialised and self._reference_frames_is_initialised \
            and self._fov_list_is_initialised and all(self._position_processors_is_initialised)

    def reset(self):
        self._fov_list_is_initialised = False
        self._focus_is_initialised = False
        self._strategy_is_initialised = False
        self._position_processors_is_initialised = []
        self._reference_frames_is_initialised = False

    def sleep(self, duration: float):
        now = time.perf_counter()
        end = now + duration
        while (now < end) and not self.stopped():
            now = time.perf_counter()

    def restart(self):
        self._stop_event.clear()

    def stop(self):
        self._stop_event.set()

    def start_strategy(self):
        return self._start_strategy_event.set()

    def strategy_has_started(self) -> bool:
        return self._start_strategy_event.is_set()

    def stop_strategy(self):
        return self._stop_strategy_event.set()

    def strategy_has_stopped(self) -> bool:
        return self._stop_strategy_event.is_set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def shutdown(self):
        self._stop_strategy_event.set()
        self._start_strategy_event.set()
        self._stop_event.set()
        self._shutdown_event.set()

    def has_shutdown(self) -> bool:
        return self._shutdown_event.is_set()



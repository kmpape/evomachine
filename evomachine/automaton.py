import copy
import logging
from multiprocessing import Event, Queue
import numpy as np
import pickle
import queue
import skimage
import time
import tensorrt  # noqa
import tensorflow as tf
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import delta
from delta.rt import PositionRT

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import ConfigFocus, ConfigImageProcessor, EVO_GUI_LOGGING_LEVEL, get_logger, EVOMACHINE_DIR,\
    USE_DMD_SOCKET
from evomachine.coordinates import Coordinate
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMDControl
else:
    from evomachine.dmd import DMDControl
from evomachine.exceptions import ErrorCode, ErrorContainer, ConfigError
from evomachine.strategy import AbstractStrategy
from evomachine.utils import EvoCroppingBox, normalise_frame, rotation_correction
from evomachine.evotypes import AutomatonCommandType, DMDCalibConfigType, LEDType, FocusAlgorithmType, \
    FocusStatusType, FilterWheelType


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
            use_seg: bool = False,
            queue_timeout: float = 0,
            run_timeout: float = 0,
    ):
        self.use_seg = True
        "Temporary switch to disable image processing."
        self._cfg: ConfigImageProcessor = cfg_processor
        "Delta configuration object for image segmentation."
        self._channel_to_index: Dict[LEDType, int] = self._cfg.channel_to_index
        "Dictionary mapping LEDType to channel index in 3D arrays."
        self._curr_fov_id: int = 0
        "Current position."
        self._curr_period: int = 0
        "Incremented after completing one round of imaging the whole device."
        self._curr_step: int = 0
        "Incremented every time a picture is taken."
        self.cam: AbstractCamera = camera
        "Camera object which can be a real camera or a class that reads from the disk."
        self._mmc_live_mode_is_on: bool = True
        "Flag for live mode for EvoCamera that uses MMC."
        self._dmd: DMDControl = dmd
        "DMDControl object to project images."
        self._pos_processor: List[PositionRT] = []
        "List of Delta objects to process the images."
        self._all_frames_raw: List[np.ndarray] = []
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._all_frames: List[np.ndarray] = []
        "List indexed by i_pos w. image array: prev/current x channels x pxl_vert x pxl_horiz."
        self._ref_frames: List[np.ndarray] = []
        "List indexed by i_pos w. reference image array: channels x pxl_vert x pxl_horiz. In camera format."
        self._use_autofocus: bool = False
        "If true, only X & Y coordinates are used in the position list."
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
        self._num_refocus: int = 0
        "Counter for refocusing after loosing autofocus. See ConfigImageProcessor for max_refocus_trials."

        if self.use_seg:
            self.roi_model = delta.model.unet_rois(input_size=(*self._cfg.cfg_delta.target_size_rois, 1))  # noqa
            logger.info(f"Automaton: Loading model with weights from {self._cfg.cfg_delta.model_file_rois}")
            self.roi_model.load_weights(self._cfg.cfg_delta.model_file_rois)
        else:
            self.roi_model = None
        self.seg_model: Optional[tf.keras.Model] = None
        "Delta segmentation model."
        self.tracking_model: Optional[tf.keras.Model] = None
        "Delta tracking model."

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

        self.dmd_calibration_data: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
        "Tuple containing calibration data: [((r_dmd, c_dmd), (r_cam, c_cam), (r_max_val, c_max_val)), ...]."

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

    def act_on_halt(self):
        """
        This is called whenever the strategy loop is interrupted.
        """
        self.cam.autofocus_unlock()
        self.cam.disable_led()
        self._dmd.display_full()

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
            use_autofocus: bool = False
    ):
        logger.info("Automaton.initialise: starting...")

        self.set_cam_live_mode(status=False)
        
        # Initialise devices
        if not self.devices_is_initialised():
            self.initialise_devices()

        # Initialise field of views
        self._use_autofocus = use_autofocus
        if field_of_views is not None:
            logger.info(f"Automaton.initialise: initialising {len(field_of_views)} FoVs...")
            self.initialise_field_of_view_list(
                field_of_views=field_of_views,
                cropping_boxes=cropping_boxes,
                use_autofocus=use_autofocus,
            )
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
        if self.use_seg:
            self.initialise_position_processor()

        assert self._curr_fov_id == 0
        assert self._curr_period == 1  # Note that each ROI keeps track of _curr_period as well

        # Initialise strategy
        logger.info(f"Automaton.initialise: initialising strategy...")
        self._initialise_strategy()

        logger.info(f"Automaton.initialise: initialisation done.")

    def set_cam_live_mode(self, status: bool = False):
        if not self.cam.is_initialised():
            logger.warning(f"Automaton.set_mmc_live_mode: Camera not initialised.")
            return
        if not isinstance(self.cam, EvoCamera):
            logger.warning(f"Automaton.set_mmc_live_mode: Camera is not of type EvoCamera.")
            return
        self.cam.studio.live().set_live_mode(status)  # noqa
        self._mmc_live_mode_is_on = status

    def initialise_devices(self):
        logger.info("Automaton.initialise_devices")
        self.cam.initialise()
        if isinstance(self.cam, EvoCamera):
            if self.cam.is_initialised():
                self.set_cam_live_mode(False)
                self.cam.set_exposure(exposure_time=self.cam.cfg.default_exposure_time)
            else:
                logger.error("Automaton.initialise_devices: Camera or Tiger not initialised.")
        self._dmd.initialise()
        if not self._dmd.is_initialised():
            logger.error("Automaton.initialise_devices: DMD not initialised.")

    def devices_is_initialised(self) -> bool:
        return self.cam.is_initialised() and self._dmd.is_initialised()

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
        self._create_position_processor(which=which)
        position_list = list(range(len(self._pos_processor))) if which is None else [which]
        logger.info(f"Automaton.initialise_position_processors: "
                    f"initialising position processors (use_seg={self.use_seg}): {position_list}.")
        for i_pos in position_list:
            logger.debug(f"Automaton.initialise_position_processor: initialising position processor {i_pos}.")
            if self.use_seg:
                if rotation is None:
                    this_rotation = rotation_correction(
                        img=self._ref_frames[i_pos][self._channel_to_index[self._cfg.channel_rot], :, :],
                    )
                else:
                    this_rotation = rotation
                logger.info(f"Automaton.initialise_position_processor: Rotating pos {i_pos} "
                            f"by {this_rotation} degrees.")

                self._pos_processor[i_pos].initialise(
                    reference=self._ref_frames[i_pos],  #  normalise_frame(self._ref_frames[i_pos]),  # TODO need to crop frames
                    channel_rot=self._channel_to_index[self._cfg.channel_rot],
                    channel_roi=self._channel_to_index[self._cfg.channel_roi],
                    rotate=this_rotation,
                    roi_boxes=roi_boxes,
                    seg_model=self.seg_model,
                    tracking_model=self.tracking_model,
                    roi_model=self.roi_model,
                )
                if not self._pos_processor[i_pos].roi_boxes:
                    logger.warning(f"Initialised position {i_pos} but found no RoIs.")
                else:
                    logger.info(f"Found {len(self._pos_processor[i_pos].roi_boxes)} RoIs for position {i_pos}.")
                self._position_processors_is_initialised[i_pos] = True
                self.fill_queue(
                    queue_data_type=AutomatonCommandType.ROI_DATA,
                    queue_data=CommandFactory.command_roi_data(
                        fov_id=i_pos,
                        rotation=self._pos_processor[i_pos].rotate,
                        roi_boxes=self._pos_processor[i_pos].roi_boxes,
                    ),
                    logging_level=logging.INFO,
                )
                logger.info(f"Automaton.initialise_position_processor: Initialised position {i_pos}.")
            else:
                self._position_processors_is_initialised[i_pos] = True
                self.fill_queue(
                    queue_data_type=AutomatonCommandType.ROI_DATA,
                    queue_data=CommandFactory.command_roi_data(
                        fov_id=i_pos,
                        rotation=0,
                        roi_boxes=[],
                    ),
                    logging_level=logging.INFO,
                )
            self._pos_to_roi[i_pos] = [i_roi for i_roi in range(len(self._pos_processor[i_pos].rois))]

    def initialise_field_of_view_list(
            self,
            field_of_views: Dict[int, Coordinate],
            cropping_boxes: Optional[Dict[int, List[EvoCroppingBox]]] = None,
            use_autofocus: bool = False,
    ):
        self._fov_list_is_initialised = False
        self._focus_is_initialised = False
        self._position_processors_is_initialised = []
        self._strategy_is_initialised = False
        self._reference_frames_is_initialised = False
        self._use_autofocus = use_autofocus
        if use_autofocus:
            logger.info("Automaton.initialise_field_of_view_list: Using autofocus.")

        self._fovs = field_of_views
        if cropping_boxes is not None:
            if not field_of_views.keys() == cropping_boxes.keys():
                raise ConfigError(f"Automaton.initialise_field_of_view_list: cropping box keys do not match "
                                  f"field_of_views.", ErrorCode.ERROR_DEVICE_CONFIG)
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
            if (not use_autofocus) and (not coord.has_z()):
                coord.z = z_coord
            elif use_autofocus:
                coord.z = None
            if self.cam.coordinate_is_out_of_bounds(coordinate=coord):
                msg = f"Automaton.initialise_field_of_view_list: {Coordinate} for FoV {i_fov} is out of bounds " \
                      f"({self.cam.get_stage_limits()})."
                raise ConfigError(message=msg, error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            self.focus_prev_z_coords[i_fov] = coord.z if not use_autofocus else z_coord

        # Allocate variables
        self._all_frames = [
            np.empty((2, len(self._cfg.channels), *self.cam.cfg.image.shape), dtype=np.float32)
            for _ in self._fovs
        ]
        self._all_frames_raw = [
            np.empty((2, len(self._cfg.channels), *self.cam.cfg.image.shape), dtype=self.cam.cfg.image.pxl_dtype)
            for _ in self._fovs
        ]
        self._ref_frames = [
            np.empty((len(self._cfg.channels), *self.cam.cfg.image.shape), dtype=self.cam.cfg.image.pxl_dtype)
            for _ in self._fovs
        ]
        if not self.cam.set_pos_id_to_coordinate(pos_id_to_coordinate=self._fovs, use_autofocus=use_autofocus):
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
            use_autofocus: bool = False,
    ):
        if not self._fov_list_is_initialised:
            raise ConfigError("Automaton.initialise_fov_focus: FoV list is not initialised.", ErrorCode.ERROR_CONFIG)
        if not self.devices_is_initialised():
            raise ConfigError("Automaton.initialise_fov_focus: Devices not initialised.", ErrorCode.ERROR_CONFIG)
        self.set_cam_live_mode(False)
        self._use_autofocus = use_autofocus

        cfg_focus = self.cam.cfg.focus if cfg_focus is None else cfg_focus
        self.focus_curves = {i_fov: None for i_fov in self._fovs.keys()}
        self.focus_prev_stack = np.zeros((*self.cam.cfg.image.shape, len(self._fovs)))
        self.focus_stack = np.zeros((*self.cam.cfg.image.shape, len(self._fovs)))
        if self._use_autofocus:
            logger.info(f"Automaton.initialise_fov_focus: Using autofocus instead.")
        else:
            self.cam.disable_led()
            self.cam.disable_live_mode()
            self._dmd.display_full()
            logger.info(f"Automaton.initialise_fov_focus with configuration {cfg_focus}")
        for i_fov, coord in self._fovs.items():
            if self._use_autofocus:
                coord.z = None
            else:
                logger.info(f"Automaton.initialise_fov_focus: initialising FoV {i_fov+1} of {len(self._fovs)}.")
                if self.stopped():
                    logger.warning("Automaton.initialise_position_list: stopping initialisation.")
                    return
                self.cam.move_to(coordinate=coord, block=True)
                self._run_software_focus(cfg_focus=cfg_focus, curr_fov_id=i_fov)
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
        self._focus_is_initialised = True

    def _run_software_focus(self, cfg_focus: ConfigFocus, curr_fov_id: int):
        self.cam.software_focus(
            cfg_focus=cfg_focus,
            user_input_override=True,
            countdown_override=True,
            cropping_box=None,
        )
        self.focus_curves[curr_fov_id] = (self.cam.focus_Z_coords, self.cam.focus_scores)
        self.focus_prev_stack[:, :, curr_fov_id] = self.cam.focus_prev_image
        self.focus_stack[:, :, curr_fov_id] = self.cam.get_software_focus_z_frame()

    def initialise_reference_frames(self):
        if not self.devices_is_initialised():
            raise ConfigError("Automaton.initialise_fov_focus: Devices not initialised.", ErrorCode.ERROR_CONFIG)
        self.set_cam_live_mode(False)
        logger.info(f"Automaton.initialise_reference_frames: "
                    f"Imaging {len(self._fovs.keys())} FoVs on {self._channel_to_index.keys()}.")
        for i_fov in self._fovs.keys():
            self.cam.move_to_pos(i_pos=i_fov)
            for channel_type, ind in self._channel_to_index.items():
                if not channel_type == LEDType.LED_385_NM:
                    self._ref_frames[i_fov][ind, :, :] = self.cam.get_frame(i_chan=channel_type)
            self.increment_pos()
        self.cam.reset_counter()
        self._reference_frames_is_initialised = True
        norm_frames = {i_fov: normalise_frame(frame) for i_fov, frame in zip(self._fovs.keys(), self._ref_frames)}
        cmd = CommandFactory.command_ref_data(ref_frames=norm_frames)
        self.fill_queue(queue_data_type=AutomatonCommandType.REF_DATA, queue_data=cmd, logging_level=logging.INFO)
        self.cam.disable_led()

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

    def _create_position_processor(self, which: Optional[int] = None):
        if which is None:
            logger.debug("Creating position procesors.")
            self._pos_processor = []
            for pos_id, fov_ind in self._pos_to_fov_index.items():
                self._pos_processor.append(
                    PositionRT(
                        position_nb=pos_id,
                        config=self._cfg.cfg_delta,
                        use_track_moma=self._cfg.use_track_RT,
                    )
                )
        else:
            if (which >= len(self._pos_processor)) or (which not in self._pos_to_fov):
                raise ConfigError(message=f"Cannot recreate processor {which}.",
                                  error_code=ErrorCode.ERROR_NOT_INITIALISED)
            self._pos_processor[which] = PositionRT(
                    position_nb=which,
                    config=self._cfg.cfg_delta,
                    use_track_moma=self._cfg.use_track_RT,
            )

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_fov_id + 1 == len(self._fovs))
                             else self._curr_period)
        self._curr_fov_id = (self._curr_fov_id + 1) % len(self._fovs)

    def manage_autofocus(self, curr_fov_id: int, debug_mode: bool = True) -> None:
        """
        Checks the autofocus status and refocuses if the autofocus is lost and self._cfg.refocus is True. Throws an
        error if the software focus fails.
        """
        if not self._use_autofocus:
            logger.warning(f"manage_autofocus: no autofocus to manage as autofocus is disabled.")
            return
        if debug_mode:
            curr_pos = self.cam.get_coordinates(['Z'])
            logger.info(f"manage_autofocus: At FoV ID {curr_fov_id} and coordinate {curr_pos}.")
        is_locked = self.cam.autofocus_is_locked()
        if not is_locked:
            logger.warning(f"manage_autofocus: lost autofocus lock on fov_id={curr_fov_id}.")
            max_num_trials_reached = False
            if self._cfg.refocus:
                if self._num_refocus < self._cfg.max_refocus_trials:
                    max_num_trials_reached = True
                    logger.error(f"manage_autofocus: Max. number ({self._cfg.max_refocus_trials}) of refocusing trials "
                                 f"reached. Halting execution.")
                    self.shutdown()
                else:
                    self._num_refocus += 1
                    self.cam.autofocus_unlock()  # just to be sure
                    self._run_software_focus(cfg_focus=self.cam.cfg.focus, curr_fov_id=curr_fov_id)
                    if self.cam.get_software_focus_status() == FocusStatusType.IN_FOCUS:
                        is_success = self.cam.autofocus_initialise()
                        if is_success:
                            self.cam.autofocus_lock()
                            logger.info(f"manage_autofocus: successfully refocused and locked autofocus.")
                        else:
                            logger.error(f"manage_autofocus: Error initialising autofocus. Halting execution.")
                            self.shutdown()
                    else:
                        logger.error(f"manage_autofocus: Received bad FocusStatusType="
                                     f"{self.cam.get_software_focus_status()}. Halting execution.")
                        self.shutdown()
            else:
                logger.warning(f"manage_autofocus: Refocusing disabled. Halting execution.")
                self.shutdown()
            self.fill_queue(
                queue_data_type=AutomatonCommandType.AUTOFOCUS_DATA,
                queue_data=CommandFactory.command_autofocus(
                    is_locked=self.cam.autofocus_is_locked(),
                    refocusing=self._cfg.refocus,
                    max_num_trials_reached=max_num_trials_reached,
                    software_focus_status=self.cam.get_software_focus_status(),
                ),
                logging_level=logging.ERROR,
            )
        else:
            logger.debug(f"manage_autofocus: autofocus is locked.")

    def override_parameter(self, fov_id: int, pos_id: int, param_name: str, param_value: Any):
        logger.info(f"Automaton.override_parameter: setting {param_name} to {param_value} for "
                    f"FoV {fov_id} and pos {pos_id}.")
        avail: List[str] = ["z_pos", "rotation", "cols_s_e"]
        if param_name not in avail:
            raise ConfigError(f"Automaton.override_parameter: {param_name} is not an available override. {avail}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if fov_id not in self._fovs.keys():
            raise ConfigError(f"Automaton.override_parameter: {fov_id} is not a valid FoV id. {self._fovs.keys()}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if pos_id not in self._pos_to_fov.keys():
            raise ConfigError(f"Automaton.override_parameter: {pos_id} is not a valid position id. "
                              f"{self._pos_to_fov.keys()}",
                              ErrorCode.ERROR_DEVICE_CONFIG)
        if param_name == "z_pos":
            if self._use_autofocus:
                logger.error("Cannot set Z coordinate when using autofocus. Returning.")
                return
            tmp_fov = copy.deepcopy(self._fovs)
            tmp_fov[fov_id].z = float(param_value)
            if not self.cam.set_pos_id_to_coordinate(pos_id_to_coordinate=tmp_fov, use_autofocus=self._use_autofocus):
                raise ConfigError(message="Automaton.override_parameter: failed to pass position list to camera.",
                                  error_code=ErrorCode.ERROR_DEVICE_CONFIG)
            logger.info(f"Automaton.override_parameter: changing Z from {self._fovs[fov_id].z} to {tmp_fov[fov_id].z}.")
            self._fovs[fov_id].z = float(param_value)
        elif param_name == "rotation":
            logger.info(f"Automaton.override_parameter: changing rotation from {self._pos_processor[pos_id].rotate} "
                        f"to {param_value}.")
            self._pos_processor[pos_id].rotate = float(param_value)
            self._pos_processor[pos_id].initialise(
                reference=normalise_frame(self._ref_frames[pos_id]),  # TODO need to crop frames
                channel_rot=self._channel_to_index[self._cfg.channel_rot],
                channel_roi=self._channel_to_index[self._cfg.channel_roi],
                rotate=self._pos_processor[pos_id].rotate,
                seg_model=self.seg_model,
                tracking_model=self.tracking_model,
                roi_model=self.roi_model,
            )

    def _gui_process(self):
        """
        Main GUI loop. Reads requested commands from the gui_to_automaton_queue and executes them.
        See QueueManager for implementation details.

        Returns
        -------
        Returns nothing but fills the automaton_to_gui_queue.
        """

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
        """
        Main experiment loop. Executes a list of commands provided by the strategy.

        Returns
        -------
        Returns nothing, put fills the process queue that is emptied by the GUI.
        """
        self.fill_queue(AutomatonCommandType.INFO_TEXT,
                        CommandFactory.command_info_text(f"At period {self._curr_period}."),
                        logging.DEBUG)

        # Execute requested commands in the given order
        for cmd in self.next_commands:
            logger.info(f"Automaton._process: Executing {cmd}.")

            if self.stopped():
                logger.warning(f"Automaton.process: stopping process at {str(cmd)}.")
                return
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)
                self.manage_autofocus(curr_fov_id=self._curr_fov_id)

            elif cmd.command_type == AutomatonCommandType.LIVE_MODE:
                self.set_cam_live_mode(cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                self.sleep(
                    duration=cmd.command_args['duration'],
                    set_live_mode=cmd.command_args['set_live_mode'],
                    channel=cmd.command_args['channel'],
                    brightness=cmd.command_args['brightness'],
                )

            elif cmd.command_type == AutomatonCommandType.STOP:
                logger.warning("Automaton.process: Received STOP command. Shutting down.")
                self.stop()
                cmd.command_execution_time = time.time()
                return

            elif cmd.command_type == AutomatonCommandType.IMAGE:
                if self._mmc_live_mode_is_on:  # noqa
                    logger.warning("Automaton._process: Camera live mode is on for IMAGE. Disabling.")
                    self.set_cam_live_mode(False)
                self._dmd.display_full()
                time.sleep(0.5)  # TODO
                if self.cam.get_exposure() != cmd.command_args['exposure_time']:
                    self.cam.set_exposure(exposure_time=cmd.command_args['exposure_time'])
                self._take_image(channels=cmd.command_args['channels'], brightness=cmd.command_args['brightness'])
                self.cam.disable_led()
                channels_int = [self._channel_to_index[c] for c in cmd.command_args['channels']]
                cmd.command_data = [self._all_frames[self._curr_fov_id][1, channels_int, :, :]]
                if cmd.command_args['segment']:
                    # TODO make sure segmentation channel is in cmd.command_args['channels']
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data.append(
                        {roi_id: None for roi_id in range(len(self._pos_processor[self._curr_fov_id].rois))}
                    )
                if cmd.command_args['save']:
                    for i_chan, channel_index in zip(cmd.command_args['channels'], channels_int):
                        self.cam.save_frame(
                            frame=self._all_frames_raw[self._curr_fov_id][1, channel_index, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_fov_id,
                        )
                self._dmd.display_none()

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                self._dmd.display_image(img=cmd.command_args['image'])
                time.sleep(0.5)  # TODO
                # TODO allow for NONE LED to actuate LED separately
                self.cam.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])  # TODO disable with timer
                self.cam.disable_led()

            cmd.command_execution_time = time.time()
            cmd.fov_id = self._curr_fov_id

            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=cmd,
                logging_level=logging.INFO,
            )

        new_errors = list(self.error_container.error_list)  # TODO extract new errors
        self.last_commands = self.next_commands
        self.next_commands = self._strategy.callback(
            fov_id=self._curr_fov_id,
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

            elif cmd.command_type == AutomatonCommandType.LIVE_MODE:
                self.set_cam_live_mode(cmd.command_args)

            elif cmd.command_type == AutomatonCommandType.WAIT:
                self.sleep(
                    duration=cmd.command_args['duration'],
                    set_live_mode=cmd.command_args['set_live_mode'],
                    channel=cmd.command_args['channel'],
                    brightness=cmd.command_args['brightness'],
                )

            elif cmd.command_type == AutomatonCommandType.STOP:
                logger.warning("Automaton.process: Received STOP command. Shutting down.")
                self.stop()
                cmd.command_execution_time = time.time()
                return

            elif cmd.command_type == AutomatonCommandType.IMAGE:
                if self._mmc_live_mode_is_on:  # noqa
                    logger.warning("Automaton._process: Camera live mode is on for IMAGE. Disabling.")
                    self.set_cam_live_mode(False)
                self._dmd.display_full()
                time.sleep(0.5)  # TODO
                if self.cam.get_exposure() != cmd.command_args['exposure_time']:
                    self.cam.set_exposure(exposure_time=cmd.command_args['exposure_time'])
                self._take_image(channels=cmd.command_args['channels'], brightness=cmd.command_args['brightness'])
                self.cam.disable_led()
                channels_int = [self._channel_to_index[c] for c in cmd.command_args['channels']]
                cmd.command_data = [self._all_frames[self._curr_fov_id][1, channels_int, :, :]]
                if cmd.command_args['segment']:
                    # TODO make sure segmentation channel is in cmd.command_args['channels']
                    self._process_position()
                    # TODO fill with segmentation data
                    cmd.command_data.append(
                        {roi_id: None for roi_id in range(len(self._pos_processor[self._curr_fov_id].rois))}
                    )
                if cmd.command_args['save']:
                    for i_chan, channel_index in zip(cmd.command_args['channels'], channels_int):
                        self.cam.save_frame(
                            frame=self._all_frames_raw[self._curr_fov_id][1, channel_index, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_fov_id,
                        )
                self._dmd.display_none()

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                self._dmd.display_image(img=cmd.command_args['image'])
                self.cam.set_led(i_chan=cmd.command_args['channel'], brightness=cmd.command_args['brightness'])
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])
                self.cam.disable_led()

            cmd.command_execution_time = time.time()
            cmd.fov_id = self._curr_fov_id

            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=cmd,
                logging_level=logging.INFO,
            )

    def run(self):
        """
        Main overall loop that contains the GUI loop and the experiment loop:

        - Main loop: exit when _shutdown_event.set()
            - GUI loop: exit when _start_strategy_event.set() or _shutdown_event.set()
            - Experiment loop: exit when _stop_strategy_event.set() or _shutdown_event.set()

        Both inner loops are halted (!= exit) when _stop_event.set()

        Returns
        -------
        Returns nothing, but fills queues.
        """

        logger.info("Automaton.run: starting...")
        while not self.has_shutdown():

            logger.info("Automaton.run: Starting GUI loop.")
            if not self.devices_is_initialised():
                self.initialise_devices()
            has_stopped = True
            while (not self.strategy_has_started()) and (not self.has_shutdown()):
                while (not self.stopped()) and (not self.strategy_has_started()) and (not self.has_shutdown()):
                    try:
                        self._gui_process()
                    except Exception as e:
                        logger.error(f"Automaton.run: Shutting down. Exception during GUI process: {e}.")
                        traceback.print_exc()
                        self.shutdown()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
                if has_stopped:
                    logger.warning("Automaton.run: halting GUI execution.")
                    has_stopped = False
            logger.info("Automaton.run: Leaving GUI loop and disabling MMC live mode.")
            self.set_cam_live_mode(status=False)

            if (not self.strategy_has_stopped()) and (not self.has_shutdown()):
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
                logger.info(f"Automaton.run: Starting strategy loop. Moving to fov {self._curr_fov_id}.")
                self._move_to_pos(pos_id=self._curr_fov_id)
                self._dmd.display_full()  # FIXME temporary statement

            self._num_refocus = 0
            has_stopped = True
            while (not self.strategy_has_stopped()) and (not self.has_shutdown()):
                while (not self.stopped()) and (not self.strategy_has_stopped()):
                    try:
                        self._process()
                    except Exception as e:
                        logger.error(f"Automaton.run: Shutting down. Exception during GUI process: {e}.")
                        traceback.print_exc()
                        self.shutdown()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
                if has_stopped:
                    logger.warning("Automaton.run strategy: halting execution.")
                    has_stopped = False
            logger.info(f"Automaton.run: Leaving strategy loop. Current fov {self._curr_fov_id}.")

            if self.is_initialised():
                logger.info("Automaton.run: finalising strategy.")
                self._finalise_process()
                self.act_on_halt()
        logger.info("Automaton.run: Shutting down.")
        self._dmd.finalise()
        self.cam.finalise()
        time.sleep(2)

    def set_strategy(self, strategy: AbstractStrategy):
        self._strategy = strategy
        self._initialise_strategy()

    def _move_to_pos(self, pos_id: Union[int, None] = -1):
        if pos_id is None:
            return
        elif pos_id == -1:
            self.increment_pos()
            self.cam.move_to_pos(i_pos=self._curr_fov_id)
        else:
            self.cam.move_to_pos(i_pos=pos_id)
            self._curr_fov_id = pos_id

    def _take_image(self, channels: Optional[List[LEDType]] = None, brightness: Union[int, List[int]] = 100):
        if (channels is None) or not channels:
            channels = self._cfg.channels
        if isinstance(brightness, int):
            brightness = [brightness for _ in channels]
        for i, channel in enumerate(channels):
            i_chan = self._channel_to_index[channel]
            fov_id = self._curr_fov_id
            self._all_frames_raw[fov_id][0, i_chan, :, :] = self._all_frames_raw[fov_id][1, i_chan, :, :]
            self._all_frames[fov_id][0, i_chan, :, :] = self._all_frames[fov_id][1, i_chan, :, :]

            self._all_frames_raw[fov_id][1, i_chan, :, :] = self.cam.get_frame(
                i_chan=channel,
                brightness=brightness[i],
                normalise=False,
            )
            if self._pos_processor[fov_id].rotate is not None:
                # TODO this should only be done once. Already rotated in preprocess image
                self._all_frames[fov_id][1, i_chan, :, :] = normalise_frame(
                    skimage.transform.rotate(
                        self._all_frames_raw[fov_id][1, i_chan, :, :],
                        self._pos_processor[fov_id].rotate,
                        resize=True,
                    )  # TODO use delta affine transform and store transformation matrix in position
                )
            else:
                self._all_frames[self._curr_fov_id][1, i_chan, :, :] = normalise_frame(
                    self._all_frames_raw[self._curr_fov_id][1, i_chan, :, :]
                )

    def _process_position(self):
        self._pos_processor[self._curr_fov_id].process_new_frame(
            new_frame=normalise_frame(self._all_frames_raw[self._curr_fov_id][1, :, :, :]),
            seg_model=self.seg_model,
            tracking_model=self.tracking_model,
        )

    def get_channel_to_index(self) -> Dict[LEDType, int]:
        return {key: value for key, value in self._channel_to_index.items()}

    def get_period(self) -> int:
        return self._curr_period

    def get_pos_id(self) -> int:
        return self._curr_fov_id

    def get_frame(self, i_pos: int, channel: LEDType) -> np.ndarray:
        return self._all_frames[i_pos][1, self._channel_to_index[channel], :, :]

    def get_strategy_name(self) -> str:
        return self._strategy.name()

    def is_initialised(self):
        return self.devices_is_initialised() and self._strategy_is_initialised and \
            self._reference_frames_is_initialised and self._fov_list_is_initialised and \
            all(self._position_processors_is_initialised)

    def reset(self):
        self._fov_list_is_initialised = False
        self._focus_is_initialised = False
        self._strategy_is_initialised = False
        self._position_processors_is_initialised = []
        self._reference_frames_is_initialised = False

    def sleep(
            self,
            duration: float,
            set_live_mode: bool = False,
            channel: LEDType = LEDType.LED_450_NM,
            brightness: float | int = 10,
    ):
        now = time.perf_counter()
        end = now + duration
        if set_live_mode:
            self._dmd.display_full()
            self.cam.set_led(i_chan=channel, brightness=brightness)
            self.set_cam_live_mode(status=True)
        while (now < end) and (not self.stopped()) and (not self.strategy_has_stopped()) and (not self.has_shutdown()):
            now = time.perf_counter()
        if set_live_mode:
            self._dmd.display_none()
            self.cam.disable_led()
            self.set_cam_live_mode(status=False)

    def restart(self):
        self._stop_event.clear()

    def stop(self):
        self._stop_event.set()

    def software_focus(
            self,
            cfg_focus: Optional[ConfigFocus] = None,
            focus_channel_override: Optional[LEDType] = None,
            rel_range_override: Optional[int] = None,
            cropping_box: Optional[EvoCroppingBox] = None,
            algorithm_override: Optional[FocusAlgorithmType] = None,
            user_input_override: bool = False,
            countdown_override: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        self.cam.software_focus(
            cfg_focus=cfg_focus,
            focus_channel_override=focus_channel_override,
            rel_range_override=rel_range_override,
            cropping_box=cropping_box,
            algorithm_override=algorithm_override,
            user_input_override=user_input_override,
            countdown_override=countdown_override,
        )
        focus_z_coords = self.cam.focus_Z_coords
        focus_scores = self.cam.focus_scores
        focus_stack = self.cam.focus_stack
        prev_image = self.cam.focus_prev_image
        prev_z = self.cam.focus_curr_pos['Z']
        new_z = self.cam.get_software_focus_z_coord()

        return focus_z_coords, focus_scores, focus_stack, prev_image, prev_z, new_z

    def start_strategy(self):
        return self._start_strategy_event.set()

    def strategy_has_started(self) -> bool:
        return self._start_strategy_event.is_set()

    def stop_strategy(self):
        self._stop_strategy_event.set()

    def strategy_has_stopped(self) -> bool:
        return self._stop_strategy_event.is_set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def shutdown(self):
        self.act_on_halt()
        self._stop_strategy_event.set()
        self._start_strategy_event.set()
        self._stop_event.set()
        self._shutdown_event.set()

    def has_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def dmd_calibrate(
            self,
            cfg: DMDCalibConfigType,
            filename: str | None = None
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Calibrates DMD by scanning DMD coordinates and measuring CAM coordinates (on image). See DMDCAlibConfigType for
        parameters. Note that this routine hard-codes some points that MUST be within the image. Change if needed.

        Parameters
        ----------
        cfg : DMDCalibConfigType
            Calibration parameters.
        filename : str
            Filename for saving calibration. Uses and overwrites default file otherwise.

        Returns
        -------
        results : [((row, col), (img_row_max.argmax(), img_col_max.argmax()), (img_row_max.max(), img_col_max.max()))]
            Calibration values with row/col DMD coordinates and remaining coordinates are CAM coordinates.
        """
        if not self.devices_is_initialised():
            logging.error("Automaton.dmd_calibrate: Devices not initialised. Returning.")
            return []
        if filename is None:
            filename = str(EVOMACHINE_DIR / "dmd_calibration_data.pkl")
        logger.info(f"Starting DMD calibration with config {cfg} and filename {filename}.")

        # Build grid to scan
        if cfg.end_col >= self._dmd.width_height_DMD[0] or cfg.end_col >= self._dmd.width_height_DMD[1]:
            logger.error(f"dmd_calibrate: Invalid row or column ranges for config: {cfg}")
        col_range = np.arange(cfg.start_col, cfg.end_col + cfg.step, cfg.step, dtype=np.dtype('int'))
        row_range = np.arange(cfg.start_row, cfg.end_row + cfg.step, cfg.step, dtype=np.dtype('int'))
        if col_range[-1] == self._dmd.width_height_DMD[1]:
            col_range[-1] -= 1
        if row_range[-1] == self._dmd.width_height_DMD[0]:
            row_range[-1] -= 1
        cols, rows = np.meshgrid(col_range, row_range)

        # Set camera configuration
        self.cam.set_exposure(exposure_time=cfg.exposure)
        if isinstance(self.cam, EvoCamera):
            self.set_cam_live_mode(False)
        self.cam.set_led(i_chan=cfg.channel, brightness=cfg.brightness)
        self.cam.set_filter_wheel(FilterWheelType.NO_FILTER)

        # Get minimum intensity for points to be considered
        max_intensity = 0
        for i_row in range(3):
            for i_col in range(3):
                if self.stopped():
                    logger.warning("dmd_calibrate: Stop event encountered. Aborting DMD calibration.")
                    return []
                row = (self._dmd.width_height_DMD[0] * (i_row + 1)) // 4
                col = (self._dmd.width_height_DMD[1] * (i_col + 1)) // 4
                self._dmd.display_circle(row=row, col=col, radius=cfg.line_width)  # these points should be on screen
                self.sleep(cfg.delay)
                test_img = self.cam.get_frame(i_chan=None, normalise=False)
                max_intensity += test_img.max()  # noqa
                logger.debug(f"Init image ({row}, {col}): {test_img.max()}")  # noqa
        max_intensity = float(max_intensity) / 9
        self._dmd.display_circle(row=0, col=0, radius=cfg.line_width)
        self.sleep(cfg.delay)
        test_img_none = self.cam.get_frame(i_chan=None, normalise=False)
        max_intensity_none = test_img_none.max()  # noqa
        if max_intensity_none >= 0.9*max_intensity:
            logger.error(f"dmd_calibrate: max off-screen intensity is high. off_screen={max_intensity} > "
                         f"0.9*on_screen={0.9*max_intensity}. "
                         f"Please verify. Aborting calibration.")
            return []
        min_intensity = max_intensity_none + 0.5 * (max_intensity - max_intensity_none)
        logger.info(f"dmd_calibrate: Max. on-screen intensity={max_intensity}, "
                    f"max off-screen intensity={max_intensity_none} => min req. intensity={min_intensity}.")

        # Get calibration points
        results = []
        for i, (col, row) in enumerate(zip(cols.flatten(), rows.flatten())):
            if i % 50 == 0:
                logger.info(f"At {i+1} of {len(cols.flatten())}")
            if self.stopped():
                logger.warning("dmd_calibrate: Stop event encountered. Aborting DMD calibration.")
                return []
            if not USE_DMD_SOCKET:
                self._dmd.display_none(update_display=False)
            self._dmd.display_circle(row=row, col=col, radius=cfg.line_width)
            self.sleep(duration=cfg.delay)
            img = self.cam.get_frame(
                i_chan=None,
                normalise=False
            )
            img_max = img.max()  # noqa
            img_col_max = img.max(axis=0)  # noqa
            img_row_max = img.max(axis=1)  # noqa
            if img_max >= min_intensity:
                results.append(((row, col),
                                (img_row_max.argmax(), img_col_max.argmax()),
                                (img_row_max.max(), img_col_max.max())))
            else:
                logger.debug(f"dmd_calibrate: DMD point (r{row},c{col}) off screen with intensity "
                             f"{img_max} < {min_intensity}.")

        self.cam.set_filter_wheel(FilterWheelType.FILTER)
        self.cam.disable_led()
        self._dmd.display_none()

        self.dmd_calibration_data = results

        if filename is not None:
            with open(filename, 'wb') as file:
                pickle.dump(results, file)

        logger.info(f"dmd_calibrate: Saved calibration data under {filename}.")

        return results



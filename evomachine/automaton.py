import copy
import cv2
from datetime import datetime
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

from evomachine.acquisition import AbstractCamera, EvoCamera, EvoCamerav3
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
from evomachine.utils import EvoCroppingBox, normalise_frame, rotation_correction, multipos_rotation_correction, \
    combine_channels, channel_extend_img
from evomachine.evotypes import AutomatonCommandType, DMDCalibConfigType, LEDType, FocusAlgorithmType, \
    FocusStatusType, FilterWheelType, MagnetModeType, AutoFocusStatusType


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
            queue_timeout: float = 0,
            run_timeout: float = 0,
    ):
        """
        Main runtime object that controls communication with the GUI and executes a strategy.

        Parameters
        ----------
        camera : AbstractCamera
            Actuation for imaging, LEDs, stage, filter wheel, and autofocus.
        cfg_processor : ConfigImageProcessor
            Image processing configuration.
        dmd : DMDControl
            Actuation for DMD and camera image <-> DMD image mapping.
        strategy : AbstractStrategy
            Strategy to be executed after initialisation.
        start_strategy_event : multiprocessing.Event
            Event to break GUI loop and start strategy loop.
        stop_strategy_event : multiprocessing.Event
            Event to break strategy loop and start GUI loop.
        stop_event : multiprocessing.Event
            Event to stop current execution of GUI/strategy commands.
        shutdown_event : multiprocessing.Event
            Event to exit both GUI and strategy loop. Also calls finalise on all devices.
        process_q : multiprocessing.Queue
            Queue filled by the Automaton during strategy loop.
        gui_to_automaton_q : multiprocessing.Queue
            Queue filled by the GUI during GUI loop.
        automaton_to_gui_q : multiprocessing.Queue
            Queue filled by the Automaton during GUI loop.
        queue_timeout : float
            Timeout for polling gui_to_automaton_q.
        run_timeout : float
            Run timeout applied to both GUI and strategy loops.
        """
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
        self._position_processors_is_initialised: list[bool] = []
        "Set to true after initialise_position_processor."
        self._reference_frames_is_initialised: bool = False
        "Set to true after initialise_reference_frames."
        self._strategy_is_initialised: bool = False
        "Set to true after _initialise_strategy."
        self._num_refocus: int = 0
        "Counter for refocusing after loosing autofocus. See ConfigImageProcessor for max_refocus_trials."
        self._multipos_rotation: float | None = None
        "Rotation correction across all positions."

        self.roi_model: tf.keras.Model | None = None
        "Delta RoI ID model."
        self.seg_model: tf.keras.Model | None = None
        "Delta segmentation model."
        self.tracking_model: tf.keras.Model | None = None
        "Delta tracking model."
        if self._cfg.roi_enabled:
            self.roi_model = delta.model.unet_rois(
                input_size=(*self._cfg.cfg_delta.target_size_rois, 1),  # noqa
                conv_kernel_size=5,
            )
            logger.info(f"Automaton: Loading RoI model with weights from {self._cfg.cfg_delta.model_file_rois}")
            self.roi_model.load_weights(self._cfg.cfg_delta.model_file_rois)
        if self._cfg.seg_enabled:
            self.seg_model = delta.model.unet_seg(input_size=(*self._cfg.cfg_delta.target_size_seg, 1))  # noqa
            logger.info(f"Automaton: Loading seg model with weights from {self._cfg.cfg_delta.model_file_seg}")
            self.seg_model.load_weights(self._cfg.cfg_delta.model_file_seg)
        if self._cfg.track_enabled and not self._cfg.use_track_RT:
            self.tracking_model = delta.model.unet_track(input_size=(*self._cfg.cfg_delta.target_size_track, 1))  # noqa
            logger.info(f"Automaton: Loading tracking model with weights from {self._cfg.cfg_delta.model_file_track}")
            self.tracking_model.load_weights(self._cfg.cfg_delta.model_file_track)
        self._use_delta: bool = self._cfg.preproc_enabled or self._cfg.seg_enabled or self._cfg.roi_enabled

        self._fovs: Dict[int, Coordinate] = {}
        "Dictionary containing coordinates of field of views. If AF is ON, Z coordinate is None."
        self._fovs_full_coords: Dict[int, Coordinate] = {}
        "Dictionary including Z coordinates. Initialised in initialise_reference_frames."
        self._fovs_coords_timeseries: dict[int, list[tuple[float, float, bool]]] = {}
        "Dictionary indexed by FoV with a list (Z pos, time, autofocus_on), used in manage_autofocus."
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
        logger.warning(f"act_on_halt: disabling autofocus and LEDS.")
        try:
            self.cam.autofocus_unlock()
        except Exception as e:
            msg = f"Automaton.act_on_halt: error unlocking autofocus: {e}"
            logger.error(msg)
        try:
            self.cam.disable_led()
        except Exception as e:
            msg = f"Automaton.act_on_halt: error unlocking disabling led: {e}"
            logger.error(msg)
        try:
            self._dmd.display_full()
        except Exception as e:
            msg = f"Automaton.act_on_halt: error setting dmd: {e}"
            logger.error(msg)

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logger.warning(msg=msg)
        else:
            logger.warning("No errors for automaton found.")
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
            field_of_views: dict[int, Coordinate] | None = None,
            cropping_boxes: dict[int, list[EvoCroppingBox]] | None = None,
            use_autofocus: bool = False
    ):
        """
        Initialises automaton for strategy execution. Can be called more than once.

        Parameters
        ----------
        field_of_views : dict[int, Coordinate] | None
            Dictionary with fov_id as keys and Coordinate as values. Cannot be None the first time this is called.
        cropping_boxes : dict[int, list[EvoCroppingBox]] | None
            Optional cropping boxes to split one field of view into several positions.
        use_autofocus : bool
            Flag indicating if autofocus is used. Uses software focus for each field of view to determine Z once
            otherwise.
        Returns
        -------

        """
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
        self.initialise_position_processor()

        assert self._curr_fov_id == 0
        assert self._curr_period == 1  # Note that each ROI keeps track of _curr_period as well

        # Initialise strategy
        logger.warn(f"Automaton.initialise: _NOT_ initialising strategy...")
        # self._initialise_strategy()

        logger.info(f"Automaton.initialise: initialisation done.")

    def set_cam_live_mode(self, status: bool = False):
        if isinstance(self.cam, EvoCamerav3):
            logger.warning(f"Automaton.set_mmc_live_mode: Using pvc.")
            return
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
        if isinstance(self.cam, EvoCamerav3):
            if self.cam.is_initialised():
                self.cam.cam.exp_time = self.cam.cfg.default_exposure_time
        # elif...
        elif isinstance(self.cam, EvoCamera):
            if self.cam.is_initialised():
                self.set_cam_live_mode(False)
                self.cam.set_exposure(exposure_time=self.cam.cfg.default_exposure_time)
            else:
                logger.error("Automaton.initialise_devices: Camera or Tiger not initialised.")
        self._dmd.initialise()
        if not self._dmd.is_initialised():
            logger.error("Automaton.initialise_devices: DMD not initialised.")
            while not self._dmd.is_initialised():
                logger.warning("Retrying DMD initialisation in 5 seconds")
                time.sleep(5)
                self._dmd.initialise()

    def devices_is_initialised(self) -> bool:
        return self.cam.is_initialised() and self._dmd.is_initialised()

    def initialise_position_processor(
            self,
            which: int | None = None,
            rotation: float | None = None,
            roi_boxes: list[delta.utils.CroppingBox] | None = None,
    ):
        if not self._fov_list_is_initialised or not self._reference_frames_is_initialised:
            logger.warning("Automaton.initialise_position_processor: position list is not initialised.")
            raise ConfigError(message="Automaton.initialise_position_processor: position list is not initialised.",
                              error_code=ErrorCode.ERROR_DEVICE_CONFIG)
        self._create_position_processor(which=which)
        position_list = list(range(len(self._pos_processor))) if which is None else [which]
        logger.info(f"Automaton.initialise_position_processors: Positions {position_list} with {self._cfg}.")
        for i_pos in position_list:
            logger.debug(f"Automaton.initialise_position_processor: initialising position processor {i_pos}.")
            if self._cfg.preproc_enabled or self._cfg.roi_enabled or self._cfg.seg_enabled:
                if rotation is None:
                    if self._multipos_rotation is None:
                        this_rotation = multipos_rotation_correction(
                            imgs=[combine_channels(self._ref_frames[i], self._channel_to_index, self._cfg.channels_seg)  # noqa
                                  for i in list(range(len(self._pos_processor)))],  # self._ref_frames[i][self._channel_to_index[self._cfg.channel_rot], :, :]
                        )
                    else:
                        this_rotation = self._multipos_rotation
                else:
                    this_rotation = rotation
                logger.info(f"Automaton.initialise_position_processor: Rotating pos {i_pos} "
                            f"by {this_rotation} degrees.")
                ref = channel_extend_img(
                    img=self._ref_frames[i_pos],
                    channel_dict=self._channel_to_index,
                    channels=self._cfg.channels_seg,
                    ind=0,
                )
                self._pos_processor[i_pos].initialise(
                    reference=ref,
                    rotate=this_rotation,
                    roi_boxes=roi_boxes,
                    seg_model=self.seg_model,
                    tracking_model=self.tracking_model,
                    roi_model=self.roi_model,
                    lineage_enabled=self._cfg.lineage_enabled,
                    roi_min_area=self._cfg.roi_min_area,
                    roi_max_area=self._cfg.roi_max_area,
                    roi_max_height=self._cfg.roi_max_height,
                    adjust_cropping_boxes=True,
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
                if self._cfg.seg_enabled:
                    self.fill_queue(
                        queue_data_type=AutomatonCommandType.SEG_DATA,
                        queue_data=CommandFactory.command_seg_data(
                            fov_id=i_pos,
                            seg_masks=self._pos_processor[i_pos].get_seg(frame=0),
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
        logger.info(f"Automaton.initialise_position_processor: initialisation done.")

    def initialise_field_of_view_list(
            self,
            field_of_views: dict[int, Coordinate],
            cropping_boxes: dict[int, list[EvoCroppingBox]] | None = None,
            use_autofocus: bool = False,
    ):
        self._fov_list_is_initialised = False
        self._focus_is_initialised = False
        self._position_processors_is_initialised = []
        self._strategy_is_initialised = False
        self._reference_frames_is_initialised = False
        self._use_autofocus = use_autofocus
        if use_autofocus:
            if not self.cam.autofocus_is_locked():
                msg = "Automaton.initialise_field_of_view_list: autofocus is not locked. Lock autofocus or uncheck" \
                      "autofocus and re-run initialisation."
                raise RuntimeError(msg)
            else:
                logger.info("Automaton.initialise_field_of_view_list: Using autofocus.")

        self._fovs = field_of_views
        self._fovs_full_coords = copy.deepcopy(field_of_views)
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
        logger.info(f"Automaton.initialise_field_of_view_list: initialisation done.")

    def initialise_fov_focus(self, cfg_focus: ConfigFocus | None = None, use_autofocus: bool = False):
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
        logger.info(f"Automaton.initialise_fov_focus: initialisation done.")

    def _run_software_focus(self, cfg_focus: ConfigFocus, curr_fov_id: int | None):
        self._dmd.display_full()
        self.cam.software_focus(
            cfg_focus=cfg_focus,
            user_input_override=True,
            countdown_override=True,
            cropping_box=cfg_focus.cropping_box,
        )
        if curr_fov_id is not None:
            self.focus_curves[curr_fov_id] = (self.cam.focus_Z_coords, self.cam.focus_scores)
            self.focus_prev_stack[:, :, curr_fov_id] = self.cam.focus_prev_image
            self.focus_stack[:, :, curr_fov_id] = self.cam.get_software_focus_z_frame()
        self._dmd.display_none()

    def initialise_reference_frames(self, save_references_frames: bool = True):
        if not self.devices_is_initialised():
            raise ConfigError("Automaton.initialise_fov_focus: Devices not initialised.", ErrorCode.ERROR_CONFIG)
        self.set_cam_live_mode(False)
        logger.info(f"Automaton.initialise_reference_frames: "
                    f"Imaging {len(self._fovs.keys())} FoVs on {self._channel_to_index.keys()}.")
        # if self._use_autofocus:
            # Set _curr_fov_id to force autofocus toggle during first move
            # self._curr_fov_id = None DISABLED BECAUSE NOT NECESSARY FOR NOW
        for i_fov in self._fovs.keys():
            self._move_to_pos(pos_id=i_fov)
            # self._fovs_full_coords[i_fov].z = self.cam.get_coordinates(['Z'])['Z']
            for channel_type, ind in self._channel_to_index.items():
                if not channel_type == LEDType.LED_385_NM:
                    self._ref_frames[i_fov][ind, :, :] = self.cam.get_frame(i_chan=channel_type, reset_led=False)
                    if save_references_frames:
                        self.cam.save_frame(
                            frame=self._ref_frames[i_fov][ind, :, :],
                            i_channel=channel_type,
                            i_pos=i_fov,
                            filename_suffix="_ref",
                        )
            self.increment_pos()
        self._move_to_pos(pos_id=0)
        self.cam.reset_counter()
        self._reference_frames_is_initialised = True
        norm_frames = {i_fov: normalise_frame(frame) for i_fov, frame in zip(self._fovs.keys(), self._ref_frames)}
        cmd = CommandFactory.command_ref_data(ref_frames=norm_frames)
        self.fill_queue(queue_data_type=AutomatonCommandType.REF_DATA, queue_data=cmd, logging_level=logging.INFO)
        self.cam.disable_led()
        logger.info(f"Automaton.initialise_reference_frames: initialisation done.")

    def _initialise_strategy(self):
        if not self._fov_list_is_initialised and not all(self._position_processors_is_initialised):
            raise ConfigError(message="Automaton._initialise_strategy: not initialised.",
                              error_code=ErrorCode.ERROR_NOT_INITIALISED)
        self.next_commands = self._strategy.initialise(
            field_of_views=self._fovs,
            positions=self._fov_to_pos,
            region_of_interests=self._pos_to_roi,
            config_camera=self.cam.cfg,
            pos_processors=self._pos_processor,
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
                        roi_input_size=self._cfg.delta_roi_preprocess_target_size,
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
                    roi_input_size=self._cfg.delta_roi_preprocess_target_size,
            )

    def increment_pos(self) -> None:
        self._curr_period = ((self._curr_period + 1) if (self._curr_fov_id + 1 == len(self._fovs))
                             else self._curr_period)
        self._curr_fov_id = (self._curr_fov_id + 1) % len(self._fovs)

    def manage_autofocus(
            self,
            curr_fov_id: int,
            refocus_on_all_positions: bool | None = None,
            debug_mode: bool = True,
    ) -> None:
        """
        Checks the autofocus status and refocuses if the autofocus is lost and self._cfg.refocus is True. The logic in
        case of autofocus loss is as follows:
        if self._cfg.refocus:
            if self._num_refocus > self._cfg.max_refocus_trials:
                Throw error and shut down.
            if self._cfg.refocus_using_software_focus:
                if self._cfg.refocus_on_all_positions: (overridden by refocus_on_all_positions)
                    Try software_focus and reinitialise autofocus on all recorded positions until successful.
                else:
                    Try software_focus and reinitialise autofocus on current position.
            else:
                Move back to previously recorded Z coordinate and reinitialise autofocus.
        else:
            Throw error and shut down.

        If any of the above fails, or if autofocus is lost when self._cfg.refocus is False, this function will throw an
        error and trigger a shutdown.

        Parameters
        ----------
        curr_fov_id: int
            The current ID of the field of view.
        refocus_on_all_positions: bool | None
            Overrides self._cfg.refocus_on_all_positions.
        debug_mode: bool
            Provides additional print statements

        Returns
        -------

        Assigns
        -------
        self._fovs_full_coords[curr_fov_id].z
        self._fovs_coords_timeseries[curr_fov_id][-1]

        """
        if not (self.devices_is_initialised() and self._fov_list_is_initialised):
            logger.error(f"manage_autofocus: Automaton not initialised. Returning.")
            return

        if not self._use_autofocus:
            self._fovs_full_coords[self._curr_fov_id].z = self.cam.get_coordinates(['Z'])['Z']
            logger.warning(f"manage_autofocus: no autofocus to manage as autofocus is disabled.")
            return
        old_z_coord = self._fovs_full_coords[curr_fov_id].z
        if debug_mode:
            curr_z_coord = self.cam.get_coordinates(['Z'])['Z']
            logger.info(f"manage_autofocus: At FoV ID {curr_fov_id} and Z coordinate {curr_z_coord}. "
                        f"Previous recorded Z coordinate was {old_z_coord}.")
        if self.cam.autofocus_get_status() == AutoFocusStatusType.OUT_OF_FOCUS:
            wait_s = 10
            logger.warning(f"manage_autofocus: received autofocus status {AutoFocusStatusType.OUT_OF_FOCUS}. Waiting "
                           f"{wait_s} seconds before proceeding.")
            time.sleep(wait_s)
        is_locked = self.cam.autofocus_is_locked()
        if not is_locked:
            logger.warning(f"manage_autofocus: lost autofocus lock on fov_id={curr_fov_id} and Z coordinate "
                           f"self.cam.get_coordinates(['Z'])['Z'].")
            max_num_trials_reached = False
            if self._cfg.refocus:
                if self._num_refocus > self._cfg.max_refocus_trials:
                    max_num_trials_reached = True
                    logger.error(f"manage_autofocus: Max. number ({self._cfg.max_refocus_trials}) of refocusing trials "
                                 f"reached. Halting execution.")
                    self.shutdown()
                else:
                    self._num_refocus += 1
                    logger.info(f"manage_autofocus: refocusing at trial {self._num_refocus}.")

                    # Unlock autofocus to be sure
                    self.cam.autofocus_unlock()

                    if not self._cfg.refocus_using_software_focus:
                        # Find previous position
                        prev_pos = curr_fov_id-1 if curr_fov_id-1 >= 0 else list(self._fovs_full_coords.keys())[-1]
                        logger.info(f"manage_autofocus: moving back to pos ID = {prev_pos} at coordinates "
                                    f"{self._fovs_full_coords[prev_pos]}.")

                        # Move to previous position
                        self.cam.move_to(coordinate=self._fovs_full_coords[prev_pos], block=True)

                        # Run autofocus configuration and lock autofocus if successful.
                        is_success = self.cam.autofocus_initialise(
                            user_input=False,
                        )
                        if is_success:
                            self.cam.autofocus_lock()
                            time.sleep(3)  # give the tiger box some time to update the autofocus status
                            logger.info(f"manage_autofocus: Successfully locked on previous position. Moving back.")
                            self._move_to_pos(curr_fov_id, do_manage_autofocus=False)
                            self._fovs_full_coords[curr_fov_id].z = self.cam.get_coordinates(['Z'])['Z']
                            logger.info(f"manage_autofocus: successfully re-initialised autofocus. "
                                        f"Old Z coordinate was {old_z_coord}. "
                                        f"New is {self._fovs_full_coords[curr_fov_id].z}.")
                        else:
                            logger.error(f"manage_autofocus: Error initialising autofocus. Halting execution.")
                            self.shutdown()
                    else:
                        if refocus_on_all_positions is not None:
                            num_iter: int = len(self._fovs) if refocus_on_all_positions else 1
                        else:
                            num_iter: int = len(self._fovs) if self._cfg.refocus_on_all_positions else 1
                        msg_tmp = f"all ({num_iter})" if num_iter > 1 else "one"
                        msg = f"manage_autofocus: Trying to refocus on {msg_tmp} positions."
                        logger.info(msg)

                        # Try software focus on every required FoV
                        next_fov_id = curr_fov_id
                        software_focus_success: bool = False
                        for i_sf_trial in range(num_iter):
                            # Move to previously recorded Z coordinate (X and Y should be current)
                            logger.info(f"manage_autofocus: At iteration {i_sf_trial+1} of {num_iter} and "
                                        f"fov_id={next_fov_id}. Moving back to {self._fovs_full_coords[next_fov_id]}.")
                            self.cam.move_to(coordinate=self._fovs_full_coords[next_fov_id], block=True)
                            this_old_z_coord = self._fovs_full_coords[next_fov_id].z

                            # Run software focus and lock autofocus if successful.
                            self._run_software_focus(cfg_focus=self.cam.cfg.focus, curr_fov_id=next_fov_id)
                            software_focus_success = self.cam.get_software_focus_status() == FocusStatusType.IN_FOCUS
                            if software_focus_success:
                                auto_focus_success = self.cam.autofocus_initialise(
                                    user_input=False,
                                )
                                if auto_focus_success:
                                    self.cam.autofocus_lock()
                                    self._fovs_full_coords[next_fov_id].z = self.cam.get_coordinates(['Z'])['Z']
                                    logger.info(f"manage_autofocus: successfully refocused and locked autofocus on "
                                                f"fov_id={next_fov_id}. Old Z coordinate was {this_old_z_coord}. "
                                                f"New is {self._fovs_full_coords[next_fov_id].z}.")
                                    if next_fov_id != curr_fov_id:
                                        logger.info(f"manage_autofocus: moving back to fov_id={curr_fov_id} from "
                                                    f"fov_id={next_fov_id} before proceeding.")
                                        self._move_to_pos(pos_id=curr_fov_id, do_manage_autofocus=False)
                                    break
                                else:
                                    logger.error(f"manage_autofocus: Error initialising autofocus. Halting execution.")
                                    self.shutdown()
                            else:
                                # Get next FoV and retry
                                old_fov_id = next_fov_id
                                next_fov_id = self.get_next_pos_id(current_pos=next_fov_id)
                                msg = f"manage_autofocus: software focus unsuccessful on fov_id={old_fov_id}."
                                logger.warning(msg)

                        if not software_focus_success:
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

        # Record data
        # 1. Always record last Z coord
        self._fovs_full_coords[curr_fov_id].z = self.cam.get_coordinates(['Z'])['Z']
        # 2. Record timeseries for debugging
        if curr_fov_id not in self._fovs_coords_timeseries.keys():
            self._fovs_coords_timeseries[curr_fov_id] = []
        self._fovs_coords_timeseries[curr_fov_id].append(
            (self._fovs_full_coords[curr_fov_id].z, time.time(), is_locked)
        )

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
                reference=channel_extend_img(
                    img=self._ref_frames[pos_id],
                    channel_dict=self._channel_to_index,
                    channels=self._cfg.channels_seg,
                    ind=0,
                ),
                rotate=self._pos_processor[pos_id].rotate,
                seg_model=self.seg_model,
                tracking_model=self.tracking_model,
                roi_model=self.roi_model,
                lineage_enabled=self._cfg.lineage_enabled,
                roi_min_area=self._cfg.roi_min_area,
                roi_max_area=self._cfg.roi_max_area,
                roi_max_height=self._cfg.roi_max_height,
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

    def _process(self, finalise: bool = False):
        """
        Main experiment loop. Executes a list of commands provided by the strategy. The commands for the next iteration
        are obtained from Strategy.callback() at the end of this function, or from Strategy.initialise() before the
        first call of this function. If finalise=True, the commands are obtained from Strategy.finalise().

        Returns
        -------
        Returns nothing, put fills the process queue that is emptied by the GUI.
        """
        if not finalise:
            self.fill_queue(AutomatonCommandType.INFO_TEXT,
                            CommandFactory.command_info_text(f"At period {self._curr_period}."),
                            logging.DEBUG)
        else:
            self.fill_queue(AutomatonCommandType.INFO_TEXT,
                            CommandFactory.command_info_text(f"At period {self._curr_period}. Finalising."),
                            logging.INFO)
            self.last_commands = self.next_commands
            self.next_commands = self._strategy.finalise()

        # Execute requested commands in the given order
        for cmd in self.next_commands:
            logger.info(f"Automaton._process: Executing {cmd} with args {cmd.command_args}.")

            if self.stopped():
                logger.warning(f"Automaton.process: stopping process at {str(cmd)}.")
                return
            cmd.command_data = None  # Overwritten by AutomatonCommandType.IMAGE

            if cmd.command_type == AutomatonCommandType.MOVE:
                self._move_to_pos(pos_id=cmd.command_args)
                # self.manage_autofocus(curr_fov_id=self._curr_fov_id)
                if self._cfg.refocus and self._cfg.refocus_on_all_positions:
                    # If we refocused on a different position we might've lost autofocus again moving back. In this
                    # case, refocus on the current position.
                    self.manage_autofocus(curr_fov_id=self._curr_fov_id, refocus_on_all_positions=False)

            elif cmd.command_type == AutomatonCommandType.SAVE_STATE:
                self.save_state(filename_suffix=cmd.command_args)

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

                if cmd.command_args['pattern'] is None:
                    self._dmd.display_full()
                else:
                    self._dmd.display_image(img=cmd.command_args['pattern'])
                time.sleep(0.5)  # TODO: implement feedback

                if cmd.command_args['filter_wheel'] is not None:
                    if self.cam.get_filter_wheel() != cmd.command_args['filter_wheel']:
                        self.cam.set_filter_wheel(filter_type=cmd.command_args['filter_wheel'])
                        self.sleep(duration=1)  # TODO: implement feedback

                if self.cam.get_exposure() != cmd.command_args['exposure_time']:
                    self.cam.set_exposure(exposure_time=cmd.command_args['exposure_time'])

                self._take_image(
                    channels=cmd.command_args['channels'],
                    brightness=cmd.command_args['brightness'],
                    reset_led=cmd.command_args['reset_led'],
                )
                if not cmd.command_args['force_led']:
                    self.cam.disable_led()

                self._process_position(do_segment=cmd.command_args['segment'], channels=cmd.command_args['channels'])
                channels_int = [self._channel_to_index[c] for c in cmd.command_args['channels']]
                if self._cfg.preproc_enabled and self._pos_processor[self._curr_fov_id].rois:
                    # Remove channel from channel extension for segmentation
                    tmp = self._pos_processor[self._curr_fov_id].preproc_frame[1:, :, :]
                    cmd.command_data = {
                        'img': [tmp[channels_int, :, :]],
                    }
                else:
                    cmd.command_data = {
                        'img': [self._all_frames[self._curr_fov_id][1, channels_int, :, :]],
                    }
                if self._cfg.seg_enabled and cmd.command_args['segment']:
                    cmd.command_data['seg'] = self._pos_processor[self._curr_fov_id].get_seg(frame=1)
                if self._cfg.seg_enabled and cmd.command_args['segment'] and self._cfg.track_enabled:
                    cmd.command_data['cells'] = [r.lineage.cells for r in self._pos_processor[self._curr_fov_id].rois]
                if cmd.command_args['save']:
                    for i_chan, channel_index in zip(cmd.command_args['channels'], channels_int):
                        self.cam.save_frame(
                            frame=self._all_frames_raw[self._curr_fov_id][1, channel_index, :, :],
                            i_channel=i_chan,
                            i_pos=self._curr_fov_id,
                            filter_wheel=cmd.command_args['filter_wheel'],
                        )
                        if self._cfg.preproc_enabled and self._pos_processor[self._curr_fov_id].rois:
                            self.cam.save_frame(
                                frame=self._pos_processor[self._curr_fov_id].preproc_frame[channel_index, :, :],
                                i_channel=i_chan,
                                i_pos=self._curr_fov_id,
                                filter_wheel=cmd.command_args['filter_wheel'],
                                filename_suffix="_preproc",
                            )
                self._dmd.display_none()

            elif cmd.command_type == AutomatonCommandType.PROJECT:
                # TODO need assert whether DMD image is being displayed
                self._dmd.display_image(img=cmd.command_args['image'])
                time.sleep(0.5)  # TODO
                # TODO allow for NONE LED to actuate LED separately
                self.cam.set_led(
                    i_chan=cmd.command_args['channel'],
                    brightness=cmd.command_args['brightness'],
                    duration=cmd.command_args['duration']*1000.0,
                )
                # TODO need to block movement and implement the sleep statement as countdown w. callback
                self.sleep(duration=cmd.command_args['duration'])  # TODO disable with timer
                self.cam.disable_led()

            elif cmd.command_type == AutomatonCommandType.PROJECT_ROI:
                pos_id = cmd.command_args['pos_id']
                roi_boxes = [self._pos_processor[pos_id].roi_boxes[r] for r in cmd.command_args['roi_ids']]
                pattern = self._dmd.pattern_from_roi_boxes(
                    boxes=roi_boxes,
                    fill_x=cmd.command_args['fill_x'],
                    fill_y=cmd.command_args['fill_y'],
                )
                # TODO need assert whether DMD image is being displayed
                self._dmd.display_image(img=pattern)
                time.sleep(0.5)  # TODO
                self.cam.set_led(
                    i_chan=cmd.command_args['channel'],
                    brightness=cmd.command_args['brightness'],
                    duration=cmd.command_args['duration']*1000.0,
                )
                self.sleep(duration=cmd.command_args['duration'])  # TODO disable with timer
                self.cam.disable_led()

            elif cmd.command_type == AutomatonCommandType.MAGNET:
                enable = cmd.command_args['enable']
                value = cmd.command_args['value']
                mode = cmd.command_args['mode']
                
                if enable is not None:  # TODO Fix bugs below
                    self.cam.syncboard.enable_magnet(enable)
                
                if mode == MagnetModeType.CURRENT_SET:
                    self.cam.syncboard.set_magnet_current(value)
                elif mode == MagnetModeType.FIELD_SET:
                    self.cam.syncboard.set_magnet_field(value)

            elif cmd.command_type == AutomatonCommandType.CALIBRATE_MAGNET:
                self.cam.syncboard.calibrate_magnet()

            elif cmd.command_type == AutomatonCommandType.CALIBRATE_HALL:
                hall_id = cmd.command_args
                self.cam.syncboard.calibrate_hall(hall_id)

            elif cmd.command_type == AutomatonCommandType.READ_HALL:
                hall_id = cmd.command_args
                value = self.cam.syncboard.read_hall(hall_id)
                logger.info(f"Automaton._process: Hall sensor {hall_id} read value: {value}.")

            cmd.command_execution_time = time.time()
            cmd.fov_id = self._curr_fov_id

            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=cmd,
                logging_level=logging.INFO,
            )
        if not finalise:
            new_errors = list(self.error_container.error_list)  # TODO extract new errors
            self.last_commands = self.next_commands
            self.next_commands = self._strategy.callback(
                fov_id=self._curr_fov_id,
                data=self.last_commands,
                errors=new_errors,
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
                        logger.error(f"Automaton.run: Shutting down. Exception during GUI process:\n {e}.")
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
                self.save_state(filename_suffix='initialise')
                logger.info(f"Automaton.run: Starting strategy loop. Moving to fov {self._curr_fov_id}.")
                self.cam.disable_led()
                self._move_to_pos(pos_id=self._curr_fov_id)
                self._dmd.display_full()  # FIXME temporary statement

            self._num_refocus = 0
            has_stopped = True
            while (not self.strategy_has_stopped()) and (not self.has_shutdown()):
                while (not self.stopped()) and (not self.strategy_has_stopped()):
                    try:
                        self._process()
                    except Exception as e:
                        logger.error(f"Automaton.run: Shutting down. Exception during strategy process:\n {e}.")
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
                try:
                    self._process(finalise=True)
                except Exception as e:
                    logger.error(f"Automaton.run: Exception during GUI process finalisation: {e}.")
                    traceback.print_exc()
                    self.act_on_halt()
                try:
                    self.save_state(filename_suffix='finalise')
                except Exception as e:
                    logger.error(f"Automaton.run: Exception during save_state for finalisation: {e}")
        logger.info("Automaton.run: Finalising devices.")
        try:
            self._dmd.finalise()
        except Exception as e:
            logger.error(f"Automaton.run: Exception finalising dmd: {e}")
        try:
            self.cam.autofocus_unlock()
        except Exception as e:
            logger.error(f"Automaton.run: Exception unlocking autofocus: {e}")
        try:
            self.cam.finalise()
        except Exception as e:
            logger.error(f"Automaton.run: Exception finalising cam: {e}")
        time.sleep(3)
        logger.info("Automaton.run: Shutting down.")
        # Clear shutdown event for GUI
        self._shutdown_event.clear()

    def set_strategy(self, strategy: AbstractStrategy):
        self._strategy = strategy
        self._initialise_strategy()

    def _move_to_pos(self, pos_id: int | None = -1, do_manage_autofocus: bool = True):
        """
        Move to position pos_id. Implements logic for toggling autofocus if the channel_ids for the current and new
        position are different. Flag do_manage_autofocus serves to avoid infinite recursive calls as _move_to_pos is
        also used in manage_autofocus.

        Parameters
        ----------
        pos_id : int
            Position ID to move to.
        do_manage_autofocus : bool
            Call self.manage_autofocus() after move.

        Returns
        -------

        """
        old_pos_id = self._curr_fov_id
        if pos_id is None:
            return
        elif pos_id == -1:
            self.increment_pos()
            pos_id = self._curr_fov_id
        else:
            self._curr_fov_id = pos_id

        toggle_autofocus = (old_pos_id is None and self._use_autofocus) or (
                (old_pos_id is not None) and
                (self._fovs[old_pos_id].get_channel_id() is not None) and
                (self._fovs[self._curr_fov_id].get_channel_id() is not None) and
                self._use_autofocus and
                (self._fovs[old_pos_id].get_channel_id() != self._fovs[self._curr_fov_id].get_channel_id())
        )
        if not toggle_autofocus:
            self.cam.move_to_pos(i_pos=pos_id)
        else:
            msg = f"Toggling autofocus to move from pos_id {old_pos_id} " \
                  f"({self._fovs_full_coords[old_pos_id] if old_pos_id is not None else '?'}) to " \
                  f"{pos_id} ({self._fovs_full_coords[pos_id]})."
            logger.info(msg)
            self.cam.autofocus_unlock()
            self.cam.move_to(coordinate=self._fovs_full_coords[pos_id], block=True)
            self._run_software_focus(cfg_focus=self.cam.cfg.focus, curr_fov_id=pos_id)
            time.sleep(1)  # give the stage some time to move to new position
            # Note: locking and unlocking does not seem to work
            is_success = self.cam.autofocus_initialise(
                user_input=False,
            )
            msg = f"Re-initialised autofocus after move. Successful: {is_success}"
            logger.info(msg)
            time.sleep(3)  # give the tiger box some time to update the autofocus status
            self.cam.autofocus_lock()

        if do_manage_autofocus:
            # Note: even when autofocus disabled, manage_autofocus records the current Z coordinate
            self.manage_autofocus(curr_fov_id=self._curr_fov_id)

    def _take_image(
            self,
            channels: list[LEDType] | None = None,
            brightness: int | float | list[int] | list[float] = 100,
            reset_led: bool = False,
            disable_led: bool = False,
    ):
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
                reset_led=reset_led,
                disable_led=disable_led,
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

    def _process_position(self, channels: list[LEDType], do_segment: bool = True):
        if self._cfg.preproc_enabled and self._pos_processor[self._curr_fov_id].rois:
            # TODO: channel extend images for segmentation
            img = channel_extend_img(
                img=self._all_frames_raw[self._curr_fov_id][1, :, :, :],
                channel_dict=self._channel_to_index,
                channels=self._cfg.channels_seg,
                ind=0,
            )
            channel_inds = [self._channel_to_index[c]+1 for c in channels]  # Add 1 for channel_extend_img
            if 0 not in channel_inds:
                channel_inds = [0] + channel_inds
            self._pos_processor[self._curr_fov_id].process_new_frame(
                new_frame=img,  # normalise_frame(self._all_frames_raw[self._curr_fov_id][1, :, :, :]),
                seg_model=self.seg_model if do_segment else None,
                tracking_model=self.tracking_model,
                channel_inds=channel_inds,
            )

    def get_channel_to_index(self) -> Dict[LEDType, int]:
        return {key: value for key, value in self._channel_to_index.items()}

    def get_next_pos_id(self, current_pos: int) -> int:
        return (current_pos + 1) % len(self._fovs)

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

    def save_state(self, filename_suffix: str = ''):
        exclude = [
            'cam', '_dmd', '_position_processors_is_initialised', 'roi_model',
            'seg_model', 'tracking_model', '_use_delta', '_start_strategy_event', '_stop_strategy_event',
            '_stop_event', '_shutdown_event', '_process_q', '_gui_to_automaton_q', '_automaton_to_gui_q',
            'roi_model', 'seg_model', 'tracking_model', '_pos_processor', 'dmd', 'dmd_calibration_data',
        ]

        add = {
            'strategy_name': self._strategy.__class__.__name__,
        }
        if self._pos_processor is not None and len(self._pos_processor) > 0:
            roi_boxes = {
                i: proc.roi_boxes for i, proc in enumerate(self._pos_processor)
            }
            add['roi_boxes'] = roi_boxes

        to_save: dict = {
            k: v for k, v in self.__dict__.items() if k not in exclude
        }
        for k, v in add.items():
            to_save[k] = v
        filename = self.cam.get_filename()
        for ending in ['.tiff', '.tif', '.png', '.jpg', '.jpeg']:
            if ending in filename:
                filename = filename.replace(ending, '')
                break
        if filename_suffix != '':
            filename = str(self.cam.cfg.path_to_save) + '/' + filename + f'_automatonstate_{filename_suffix}.pkl'
        else:
            filename = str(self.cam.cfg.path_to_save) + '/' + filename + f'_automatonstate.pkl'
        logger.info(f"save_state: Saving state under {filename}")
        # TODO below should be covered by test strategy or another check at startup
        try:
            with open(filename, 'wb') as file:
                pickle.dump(to_save, file)
        except Exception as e:
            logger.warning(f"save_state: Error saving state: {e}.")
            traceback.print_exc()

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
        self._dmd.finalise()
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
            logger.error("Automaton.dmd_calibrate: Devices not initialised. Returning.")
            return []
        if filename is None:
            datestr = datetime.today().strftime('%Y-%m-%d')
            filename = str(EVOMACHINE_DIR / f"dmd_calibration_data_{datestr}.pkl")
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
        last_filter_type = self.cam.get_filter_wheel()
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

        self.cam.set_filter_wheel(last_filter_type)
        self.cam.disable_led()
        self._dmd.display_none()

        self.dmd_calibration_data = results

        if filename is not None:
            with open(filename, 'wb') as file:
                pickle.dump(results, file)

        logger.info(f"dmd_calibrate: Saved calibration data under {filename}.")

        return results

    def project_roi(self, fill_y: float = 0.1):
        if not self._strategy.proj_imgs:
            self._curr_fov_id = 0
            self._initialise_strategy()
            self.next_commands = self._strategy.callback(
                fov_id=self._curr_fov_id,
                data=self.last_commands,
                errors=[],
            )
            self.next_commands = [cmd for cmd in self.next_commands if cmd.command_type != AutomatonCommandType.WAIT]
            self._process(finalise=False)
            time.sleep(5)
            self._strategy.make_projection_images()

        if not self._position_processors_is_initialised or not self._pos_processor:
            logger.warning(f"Cannot project ROI as position processor not initialised.")
            return
        if not self._pos_processor[0].roi_boxes:
            logger.warning(f"No ROI boxes available to project onto.")
            return
        if fill_y > 0:
            # boxes_to_project = [b for i, b in enumerate(self._pos_processor[0].roi_boxes) if i % 2 == 0]
            boxes_to_project = [self._pos_processor[0].roi_boxes[iroi] for iroi in self._strategy.is_not_red_id[0]]
        else:
            boxes_to_project = self._pos_processor[0].roi_boxes
            fill_y = abs(fill_y)
        pattern = self._dmd.pattern_from_roi_boxes(
            boxes=boxes_to_project,
            fill_x=self._strategy.fill_x,
            fill_y=self._strategy.fill_y,
            invert=True,
        )
        self._dmd.display_image(img=pattern)

        # cam_img = self._dmd.get_zero_array(img_size=self._dmd.width_height_CAM)
        # b = self._pos_processor[0].roi_boxes[0]
        # start_col = b.xtl
        # end_col = b.xbr
        # cam_img[:, start_col: end_col] = 255
        # pattern = self._dmd.img_to_dmd_array(cam_img)
        # self._dmd.display_image(img=pattern)
        # img = self._dmd.get_zero_array()
        # for col in range(30):
        #     for row in range(30):
        #         cv2.circle(img, (col*100, row*100), 4, color=255, thickness=-1)  # noqa
        #         # self._dmd.display_circle(row=row*200, col=col*200, radius=4)
        # self._dmd.display_image(img)

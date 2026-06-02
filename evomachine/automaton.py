from __future__ import annotations

import copy
import logging
from multiprocessing import Event, Queue
import queue
import time
import traceback
from typing import Any

import numpy as np

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import EVO_GUI_LOGGING_LEVEL, get_logger
from evomachine.coordinates import Coordinate
from evomachine.frame import Frame, FrameMetaData
from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.peripherals.dmd import DmdCalibrationConfig
from evomachine.navigation import FocusNavigator
from evomachine.projection import ProjectionManager
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType
from evomachine.utils import normalise_frame

logger = get_logger(name=__name__)


class Automaton:
    """Execute strategies using explicit peripherals and runtime manager objects."""

    def __init__(
            self,
            camera: Any,  # TODO(CODEX): make automaton take one FrameAcquisitionManager and access through it if necessary
            stage: Any,
            led_manager: Any,
            acquisition_manager: FrameAcquisitionManager,
            focus_navigator: FocusNavigator,
            strategy: AbstractStrategy,
            cfg_processor: ImageProcessorConfig,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            filter_wheel: Any | None = None,
            dmd: Any | None = None,
            autofocus: Any | None = None,
            photodiode: Any | None = None,
            projection_manager: ProjectionManager | None = None,  #  TODO(CODEX): rename ProjectionManager class as DmdCalibrator
            process_q: Queue | None = None,
            gui_to_automaton_q: Queue | None = None,
            automaton_to_gui_q: Queue | None = None,
            queue_timeout: float = 0,
            run_timeout: float = 0,
    ):
        """
        Initialise the automaton with explicit peripherals and managers.
        # TODO(CODEX): explain how the automaton works here, e.g. gui vs strategy process

        Parameters
        ----------
        camera
            Camera peripheral used by the acquisition manager.
        stage
            Stage peripheral used by the focus navigator.
        led_manager
            LED manager used for waits and projections.
        acquisition_manager
            FrameAcquisitionManager used for image commands.
        focus_navigator
            FocusNavigator used for MOVE commands and focus recovery.
        strategy
            Strategy that produces AutomatonCommand objects.
        cfg_processor
            Image-processing configuration.
        start_strategy_event
            Event that starts strategy execution.
        stop_strategy_event
            Event that stops strategy execution.
        stop_event
            Event that halts the current loop.
        shutdown_event
            Event that exits all loops.
        filter_wheel
            Optional filter wheel peripheral.
        dmd
            Optional DMD peripheral.
        autofocus
            Optional autofocus peripheral.
        photodiode
            Optional photodiode peripheral.
        projection_manager
            Optional ProjectionManager used for DMD calibration.
        process_q
            Optional process-to-GUI queue.
        gui_to_automaton_q
            Optional GUI-to-automaton queue.
        automaton_to_gui_q
            Optional automaton-to-GUI queue.
        queue_timeout
            Queue polling timeout in seconds.
        run_timeout
            Delay between loop iterations in seconds.

        Returns
        -------
        None
        """
        self._require_methods(camera, "camera", ("initialise", "is_initialised", "stop", "finalise"))
        self._require_methods(stage, "stage", ("initialise", "is_initialised", "stop", "finalise"))
        self._require_methods(led_manager, "led_manager", ("initialise", "is_initialised", "set_led", "disable_led"))
        if not isinstance(acquisition_manager, FrameAcquisitionManager):
            raise TypeError("Automaton.__init__: acquisition_manager must be FrameAcquisitionManager.")
        if not isinstance(focus_navigator, FocusNavigator):
            raise TypeError("Automaton.__init__: focus_navigator must be FocusNavigator.")
        if not isinstance(strategy, AbstractStrategy):
            raise TypeError("Automaton.__init__: strategy must be AbstractStrategy.")
        if projection_manager is not None and not isinstance(projection_manager, ProjectionManager):
            raise TypeError("Automaton.__init__: projection_manager must be ProjectionManager or None.")
        # TODO(CODEX) add "description strings" and type annotations everywhere
        self.camera = camera
        self.stage = stage
        self.led_manager = led_manager
        self.filter_wheel = filter_wheel
        self.dmd = dmd
        self.autofocus = autofocus
        self.photodiode = photodiode
        self.acquisition_manager = acquisition_manager
        self.focus_navigator = focus_navigator
        self.projection_manager = projection_manager
        self._strategy = strategy
        self._cfg = cfg_processor
        self._channel_to_index: dict[LEDType, int] = self._cfg.channel_to_index
        self._curr_fov_id: int = 0
        self._curr_period: int = 0
        self._curr_step: int = 0
        self._fovs: dict[int, Coordinate] = {}
        self._fov_to_roi: dict[int, list[int]] = {}
        self._cropping_boxes: dict[int, list[Any]] = {}
        self._fov_processors: list[Any] = []
        self._all_frames_raw: list[np.ndarray] = []
        self._all_frames: list[np.ndarray] = []
        self._ref_frames: list[np.ndarray] = []
        self._fov_list_is_initialised = False
        self._strategy_is_initialised = False
        self._reference_frames_is_initialised = False
        self._fov_processors_is_initialised: list[bool] = []
        self.next_commands: list[AutomatonCommand] = []
        self.last_commands: list[AutomatonCommand] = []
        self._start_strategy_event = start_strategy_event
        self._stop_strategy_event = stop_strategy_event
        self._stop_event = stop_event
        self._shutdown_event = shutdown_event
        self._process_q: Queue | None = process_q
        self._gui_to_automaton_q: Queue | None = gui_to_automaton_q
        self._automaton_to_gui_q: Queue | None = automaton_to_gui_q
        self.queue_timeout = self._validate_non_negative_float(queue_timeout, "queue_timeout")
        self.run_timeout = self._validate_non_negative_float(run_timeout, "run_timeout")

    # TODO(CODEX): Move _require_methods and _validate_non_negative_float to utils. check if other classes redefine similar functions and reuse.
    @staticmethod
    def _require_methods(obj: Any, name: str, methods: tuple[str, ...]) -> None:
        """
        Raise TypeError if an object does not expose required methods.

        Parameters
        ----------
        obj
            Object to inspect.
        name
            Dependency name used in error messages.
        methods
            Method names that must be callable.

        Returns
        -------
        None
        """
        missing = [method for method in methods if not callable(getattr(obj, method, None))]
        if missing:
            raise TypeError(f"Automaton.__init__: {name} is missing callable methods {missing}.")

    @staticmethod
    def _validate_non_negative_float(value: float, name: str) -> float:
        """
        Return a validated non-negative float.

        Parameters
        ----------
        value
            Candidate numeric value.
        name
            Field name used in error messages.

        Returns
        -------
        float
            Validated value.
        """
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"Automaton.__init__: {name} must be numeric, received {type(value)}.")
        if value < 0:
            raise ValueError(f"Automaton.__init__: {name} must be non-negative, received {value}.")
        return float(value)

    def initialise(
            self,
            fovs: dict[int, Coordinate],
            cropping_boxes: dict[int, list[Any]] | None = None,
            use_autofocus: bool = False,
    ) -> None:
        """
        Initialise devices, field-of-view state, focus navigation, and strategy.

        Parameters
        ----------
        fovs
            Mapping from fov ID to Coordinate.
        cropping_boxes
            Optional mapping from FoV ID to cropping boxes.
        use_autofocus
            Whether registered stage fovs should omit Z for autofocus.

        Returns
        -------
        None
        """
        if not isinstance(fovs, dict) or not fovs:
            raise TypeError("Automaton.initialise: fovs must be a non-empty dict[int, Coordinate].")
        if not all(isinstance(key, int) and isinstance(value, Coordinate) for key, value in fovs.items()):
            raise TypeError("Automaton.initialise: fovs must map int to Coordinate.")
        self.initialise_devices()
        self._fovs = {fov_id: coordinate.copy() for fov_id, coordinate in fovs.items()}
        self._cropping_boxes = {} if cropping_boxes is None else copy.copy(cropping_boxes)
        self._fov_to_roi = {fov_id: [] for fov_id in self._fovs}
        self.focus_navigator.initialise_fovs(
            fov_id_to_coordinate=self._fovs,
            use_autofocus=use_autofocus,
        )
        self._fov_processors_is_initialised = [True for _ in self._fovs]
        self._fov_list_is_initialised = True
        self._reference_frames_is_initialised = True
        first_fov_id = next(iter(self._fovs))
        self._curr_fov_id = first_fov_id
        self._initialise_strategy()

    def initialise_devices(self) -> None:
        """
        Initialise every configured peripheral.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for device in self._iter_peripherals():
            if not device.is_initialised():
                device.initialise()
        if not self.devices_is_initialised():
            raise RuntimeError("Automaton.initialise_devices: not all devices initialised.")

    def devices_is_initialised(self) -> bool:
        """
        Return whether all configured peripherals are initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when all configured peripherals report initialised.
        """
        return all(device.is_initialised() for device in self._iter_peripherals())

    def _iter_peripherals(self) -> list[Any]:
        """
        Return configured peripherals in deterministic shutdown order.

        Parameters
        ----------
        None

        Returns
        -------
        list[Any]
            Configured peripheral objects.
        """
        return [
            device for device in (
                self.camera,
                self.stage,
                self.led_manager,
                self.filter_wheel,
                self.dmd,
                self.autofocus,
                self.photodiode,
            )
            if device is not None
        ]

    def _initialise_strategy(self) -> None:
        """
        Initialise the current strategy.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self._fov_list_is_initialised:
            raise RuntimeError("Automaton._initialise_strategy: field of views are not initialised.")
        if self.dmd is None:
            raise RuntimeError("Automaton._initialise_strategy: dmd is required.")
        self._strategy.command_factory.update_region_of_interests(region_of_interests=self._fov_to_roi)
        self.next_commands = self._strategy.initialise(
            fovs=self._fovs,
            region_of_interests=self._fov_to_roi,
            config_camera=getattr(self.camera, "cfg", None),
            fov_processors=self._fov_processors,
            dmd=self.dmd,
        )
        self._strategy_is_initialised = True

    def _gui_process(self) -> None:
        """
        Process queued GUI requests using the current legacy request format.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._gui_to_automaton_q is None or self._automaton_to_gui_q is None:
            return
        while not self._gui_to_automaton_q.empty():
            try:
                req_id, req_str, kwargs_dict = self._gui_to_automaton_q.get(timeout=self.queue_timeout)
                req_args = ",".join(f"{key}=kwargs_dict['{key}']" for key in kwargs_dict)
                try:
                    req_ans = eval(f"{req_str}({req_args})")
                    self._automaton_to_gui_q.put((req_id, req_ans))
                except Exception as error:
                    logger.error(f"Automaton._gui_process: failed to execute {req_str}({req_args}): {error}.")
                    traceback.print_exc()
                    self._automaton_to_gui_q.put((req_id, error))
            except queue.Empty:
                pass

    def _process(self, finalise: bool = False) -> None:
        """
        Execute current strategy commands and request the next command list.

        Parameters
        ----------
        finalise
            If True, execute commands returned by strategy.finalise().

        Returns
        -------
        None
        """
        if not self._strategy_is_initialised:
            raise RuntimeError("Automaton._process: strategy is not initialised.")
        if finalise:
            self.last_commands = self.next_commands
            self.next_commands = self._strategy.finalise()
        for command in self.next_commands:
            if self.stopped():
                return
            command.command_data = None
            if command.command_type == AutomatonCommandType.MOVE:
                command.command_data = self._execute_move(command=command)
            elif command.command_type == AutomatonCommandType.IMAGE:
                command.command_data = self._execute_image(command=command)
            elif command.command_type == AutomatonCommandType.PROJECT:
                command.command_data = self._execute_project(command=command)
            elif command.command_type == AutomatonCommandType.PROJECT_ROI:
                command.command_data = self._execute_project_roi(command=command)
            elif command.command_type == AutomatonCommandType.WAIT:
                self.sleep(**command.command_args)
            elif command.command_type == AutomatonCommandType.STOP:
                self.stop()
            elif command.command_type == AutomatonCommandType.SAVE_STATE:
                self.save_state(filename_suffix=command.command_args)
            elif command.command_type == AutomatonCommandType.LIVE_MODE:
                self.set_cam_live_mode(status=command.command_args)
            else:
                raise RuntimeError(f"Automaton._process: unsupported command type {command.command_type}.")
            command.command_execution_time = time.time()
            command.fov_id = self._curr_fov_id
            self.fill_queue(
                queue_data_type=AutomatonCommandType.PROCESS_DATA,
                queue_data=command,
                logging_level=logging.INFO,
            )
        if not finalise:
            self.last_commands = self.next_commands
            self.next_commands = self._strategy.callback(
                fov_id=self._curr_fov_id,
                data=self.last_commands,
                errors=[],
            )

    def _execute_move(self, command: AutomatonCommand) -> Any:
        """
        Execute one MOVE command through FocusNavigator.

        Parameters
        ----------
        command
            MOVE command to execute.

        Returns
        -------
        Any
            FocusNavigatorResult returned by FocusNavigator.move().
        """
        target_fov_id = command.command_args
        if target_fov_id is None:
            return None
        if target_fov_id == -1:
            target_fov_id = self.get_next_fov_id(current_fov=self._curr_fov_id)
        result = self.focus_navigator.move(fov_id=target_fov_id)
        self._curr_fov_id = target_fov_id
        if self._fovs and target_fov_id == next(iter(self._fovs)):
            self._curr_period += 1
        return result

    def _execute_image(self, command: AutomatonCommand) -> dict[str, Any]:
        """
        Execute one IMAGE command through FrameAcquisitionManager.

        Parameters
        ----------
        command
            IMAGE command containing FrameMetaData.

        Returns
        -------
        dict[str, Any]
            Acquired image data and metadata for strategy callbacks and GUI queues.
        """
        frame_metadata = command.command_args["frame_metadata"]
        metadata_items = frame_metadata if isinstance(frame_metadata, list) else [frame_metadata]
        for metadata in metadata_items:
            if not isinstance(metadata, FrameMetaData):
                raise TypeError("Automaton._execute_image: frame_metadata entries must be FrameMetaData.")
            metadata.callback_id = self._strategy.callback_counter
            if metadata.fov_id < 0:
                metadata.fov_id = self._curr_fov_id
        frame = self.acquisition_manager.take_frame(
            frame_metadata=frame_metadata,
            settings=FrameAcquisitionSettings(save=command.command_args["save"]),
        )
        self._store_frame(frame=frame)
        channel_indices = [
            self._metadata_channel_index(metadata)
            for metadata in frame.frame_metadata
            if self._metadata_channel_index(metadata) is not None
        ]
        command_data = {
            "img": [frame.array],
            "frame_metadata": frame.frame_metadata,
            "saved_paths": frame.saved_paths,
        }
        if command.command_args["segment"] and channel_indices:
            command_data["seg"] = {}
        return command_data

    def _store_frame(self, frame: Frame) -> None:
        """
        Store acquired frame data in the automaton frame buffers.

        Parameters
        ----------
        frame
            Frame returned by FrameAcquisitionManager.

        Returns
        -------
        None
        """
        self._ensure_frame_buffers(frame=frame)
        for frame_index, metadata in enumerate(frame.frame_metadata):
            channel_index = self._metadata_channel_index(metadata=metadata)
            if channel_index is None:
                continue
            fov_id = metadata.fov_id if metadata.fov_id >= 0 else self._curr_fov_id
            self._all_frames_raw[fov_id][0, channel_index, :, :] = self._all_frames_raw[fov_id][1, channel_index, :, :]
            self._all_frames[fov_id][0, channel_index, :, :] = self._all_frames[fov_id][1, channel_index, :, :]
            self._all_frames_raw[fov_id][1, channel_index, :, :] = frame.array[frame_index]
            self._all_frames[fov_id][1, channel_index, :, :] = normalise_frame(frame.array[frame_index])

    def _ensure_frame_buffers(self, frame: Frame) -> None:
        """
        Allocate frame buffers when image shape is first known.

        Parameters
        ----------
        frame
            Frame whose image shape and dtype determine buffer allocation.

        Returns
        -------
        None
        """
        if self._all_frames_raw:
            return
        if not self._fovs:
            raise RuntimeError("Automaton._ensure_frame_buffers: field of views are not initialised.")
        frame_shape = frame.array.shape[-2:]
        dtype = frame.array.dtype
        num_channels = len(self._channel_to_index)
        num_fovs = max(self._fovs) + 1
        # TODO(CODEX) refactor these arrays below to:
        # - store the latest nsteps frames (replacing 2) and add this variable as class attribute for now. reference frame remains 2D.
        # - refactor these arrays to be arrays of size num_fovs x nsteps x num_channels x *frame_shape, and store the latest frame in index 0 for easier access and appending new frames at index 1 before shifting. this will simplify the code and make it more efficient when we want to add more than 2 frames in the future.
        self._all_frames_raw = [
            np.zeros((2, num_channels, *frame_shape), dtype=dtype)
            for _ in range(num_fovs)
        ]
        self._all_frames = [
            np.zeros((2, num_channels, *frame_shape), dtype=np.float32)
            for _ in range(num_fovs)
        ]
        self._ref_frames = [
            np.zeros((num_channels, *frame_shape), dtype=dtype)
            for _ in range(num_fovs)
        ]

    def _metadata_channel_index(self, metadata: FrameMetaData) -> int | None:
        """
        Return the image channel index associated with one FrameMetaData.

        Parameters
        ----------
        metadata
            Frame metadata to inspect.

        Returns
        -------
        int | None
            Channel index, or None when no LED channel is present.
        """
        if metadata.leds is None or len(metadata.leds) == 0:
            return None
        led_type = next(iter(metadata.leds))
        return self._channel_to_index[led_type]

    def _execute_project(self, command: AutomatonCommand) -> bool:
        """
        Execute one PROJECT command.

        Parameters
        ----------
        command
            PROJECT command to execute.

        Returns
        -------
        bool
            True after projection completes.
        """
        if self.dmd is None:
            raise RuntimeError("Automaton._execute_project: dmd is required.")
        args = command.command_args
        self.dmd.display_image(img=args["image"])
        self.led_manager.set_led(
            led_type=args["channel"],
            brightness=args["brightness"],
            duration=args["duration"] * 1000.0,
        )
        self.sleep(duration=args["duration"])
        self.led_manager.disable_led()
        return True

    def _execute_project_roi(self, command: AutomatonCommand) -> np.ndarray:
        """
        Execute one PROJECT_ROI command.

        Parameters
        ----------
        command
            PROJECT_ROI command to execute.

        Returns
        -------
        np.ndarray
            DMD pattern displayed for the ROI projection.
        """
        if self.dmd is None:
            raise RuntimeError("Automaton._execute_project_roi: dmd is required.")
        args = command.command_args
        fov_id = args["fov_id"]
        if fov_id >= len(self._fov_processors):
            raise KeyError(f"Automaton._execute_project_roi: unknown fov ID {fov_id}.")
        processor = self._fov_processors[fov_id]
        roi_boxes = [processor.roi_boxes[roi_id] for roi_id in args["roi_ids"]]
        pattern = self.dmd.pattern_from_roi_boxes(
            boxes=roi_boxes,
            fill_x=args["fill_x"],
            fill_y=args["fill_y"],
            invert=args["invert"],
            warp=True,
        )
        self.dmd.display_image(img=pattern)
        self.led_manager.set_led(
            led_type=args["channel"],
            brightness=args["brightness"],
            duration=args["duration"] * 1000.0,
        )
        self.sleep(duration=args["duration"], set_live_mode=args["set_live_mode"])
        self.led_manager.disable_led()
        return pattern

    def fill_queue(
            self,
            queue_data_type: AutomatonCommandType,
            queue_data: AutomatonCommand,
            logging_level: int = logging.INFO,
    ) -> None:
        """
        Put copied command data onto the process queue when configured.

        Parameters
        ----------
        queue_data_type
            Queue message type.
        queue_data
            Command payload.
        logging_level
            Minimum GUI logging level filter.

        Returns
        -------
        None
        """
        if self._process_q is not None and logging_level >= EVO_GUI_LOGGING_LEVEL:
            self._process_q.put((queue_data_type, copy.copy(queue_data)))

    def run(self) -> None:
        """
        Run the GUI and strategy loops until shutdown is requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        while not self.has_shutdown():
            while not self.strategy_has_started() and not self.has_shutdown():
                self._gui_process()
                if self.run_timeout > 0:
                    self.sleep(duration=self.run_timeout)
            while not self.strategy_has_stopped() and not self.has_shutdown():
                if not self.stopped():
                    self._process()
                if self.run_timeout > 0:
                    self.sleep(duration=self.run_timeout)

    def set_cam_live_mode(self, status: bool = False) -> None:
        """
        Set camera live mode when the camera exposes a compatible method.

        Parameters
        ----------
        status
            Desired live mode state.

        Returns
        -------
        None
        """
        method_name = "enable_live_mode" if status else "disable_live_mode"
        live_mode = getattr(self.camera, method_name, None)
        if callable(live_mode):
            live_mode()

    def override_parameter(self, fov_id: int, param_name: str, param_value: Any) -> None:
        """
        Placeholder for legacy GUI parameter overrides.

        Parameters
        ----------
        fov_id
            Field-of-view identifier from the GUI request.
        param_name
            Parameter name requested by the GUI.
        param_value
            Replacement parameter value requested by the GUI.

        Returns
        -------
        None
        """
        raise NotImplementedError("Automaton.override_parameter: GUI parameter overrides are not refactored yet.")

    def dmd_calibrate(
            self,
            cfg: DmdCalibrationConfig,
            filename: str | None = None,
    ) -> tuple[list, np.ndarray, np.ndarray, Any] | tuple[None, None, None, None]:
        """
        Calibrate DMD projection through ProjectionManager.

        Parameters
        ----------
        cfg
            DMD calibration configuration.
        filename
            Optional calibration filename.

        Returns
        -------
        tuple[list, np.ndarray, np.ndarray, Any] | tuple[None, None, None, None]
            ProjectionManager calibration result.
        """
        if self.projection_manager is None:
            raise RuntimeError("Automaton.dmd_calibrate: projection_manager is required.")
        return self.projection_manager.dmd_calibrate(cfg=cfg, filename=filename)

    def sleep(
            self,
            duration: float,
            set_live_mode: bool = False,
            channel: LEDType = LEDType.LED_450_NM,
            brightness: float | int = 10,
    ) -> None:
        """
        Sleep while respecting stop/shutdown events.

        Parameters
        ----------
        duration
            Duration in seconds.
        set_live_mode
            If True, enable LED/live mode during the wait when available.
        channel
            LED channel to enable during live-mode waits.
        brightness
            LED brightness for live-mode waits.

        Returns
        -------
        None
        """
        if duration < 0:
            raise ValueError(f"Automaton.sleep: duration must be non-negative, received {duration}.")
        end = time.perf_counter() + duration
        if set_live_mode:
            self.led_manager.set_led(led_type=channel, brightness=brightness)
            self.set_cam_live_mode(status=True)
        while time.perf_counter() < end and not self.stopped() and not self.has_shutdown():
            time.sleep(min(0.01, max(0.0, end - time.perf_counter())))
        if set_live_mode:
            self.led_manager.disable_led()
            self.set_cam_live_mode(status=False)

    def save_state(self, filename_suffix: str = "") -> None:
        """
        Placeholder for future state persistence.

        Parameters
        ----------
        filename_suffix
            Requested filename suffix.

        Returns
        -------
        None
        """
        logger.warning("Automaton.save_state: state persistence is not refactored yet.")

    def act_on_halt(self) -> None:
        """
        Stop active peripherals after a halt request.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.acquisition_manager.stop()
        if self.autofocus is not None and callable(getattr(self.autofocus, "unlock", None)):
            self.autofocus.unlock()

    def shutdown(self) -> None:
        """
        Stop and finalise peripherals, then set shutdown events.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.act_on_halt()
        for device in reversed(self._iter_peripherals()):
            finalise = getattr(device, "finalise", None)
            if callable(finalise):
                finalise()
        self._stop_strategy_event.set()
        self._start_strategy_event.set()
        self._stop_event.set()
        self._shutdown_event.set()

    def start_strategy(self) -> None:
        """
        Set the strategy-start event.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._start_strategy_event.set()

    def strategy_has_started(self) -> bool:
        """
        Return whether the strategy-start event is set.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when strategy execution has been requested.
        """
        return self._start_strategy_event.is_set()

    def stop_strategy(self) -> None:
        """
        Set the strategy-stop event.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._stop_strategy_event.set()

    def strategy_has_stopped(self) -> bool:
        """
        Return whether the strategy-stop event is set.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when strategy execution should stop.
        """
        return self._stop_strategy_event.is_set()

    def stop(self) -> None:
        """
        Set the stop event.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._stop_event.set()

    def stopped(self) -> bool:
        """
        Return whether the stop event is set.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when execution is halted.
        """
        return self._stop_event.is_set()

    def restart(self) -> None:
        """
        Clear the stop event.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._stop_event.clear()

    def has_shutdown(self) -> bool:
        """
        Return whether shutdown has been requested.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when shutdown event is set.
        """
        return self._shutdown_event.is_set()

    def is_initialised(self) -> bool:
        """
        Return whether automaton runtime state is ready for strategy execution.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when devices, fovs, and strategy are initialised.
        """
        return (
            self.devices_is_initialised()
            and self._strategy_is_initialised
            and self._reference_frames_is_initialised
            and self._fov_list_is_initialised
            and all(self._fov_processors_is_initialised)
        )

    def get_channel_to_index(self) -> dict[LEDType, int]:
        """
        Return a copy of the LED channel index mapping.

        Parameters
        ----------
        None

        Returns
        -------
        dict[LEDType, int]
            Channel-to-index mapping.
        """
        return dict(self._channel_to_index)

    # TODO(CODEX): can this function be inlined as it is only used once currently? also can keys[(keys.index(current_fov) + 1) % len(keys)] be improved?
    def get_next_fov_id(self, current_fov: int) -> int:
        """
        Return the next fov ID, wrapping at the end.

        Parameters
        ----------
        current_fov
            Current fov ID.

        Returns
        -------
        int
            Next fov ID.
        """
        keys = list(self._fovs)
        if current_fov not in keys:
            raise KeyError(f"Automaton.get_next_fov_id: unknown fov ID {current_fov}.")
        return keys[(keys.index(current_fov) + 1) % len(keys)]


    # TODO(CODEX): I think that this should be removed, self._curr_period. We should have a callback counter, and maybe a command counter
    def get_period(self) -> int:
        """
        Return the current acquisition period counter.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current period.
        """
        return self._curr_period

    # TODO(CODEX): this should be extracted from the navigator, which should give UNKNOWN_FOV_ID when the current fov is not known (if the coordinates don't match the current position)
    def get_fov_id(self) -> int:
        """
        Return the current fov ID.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current fov ID.
        """
        return self._curr_fov_id

    # TODO(CODEX): this should have a time id argument and return most recent one by default
    def get_frame(self, fov_id: int, channel: LEDType) -> np.ndarray:
        """
        Return the latest normalised frame for one fov and LED channel.

        Parameters
        ----------
        fov_id
            FoV ID.
        channel
            LED channel.

        Returns
        -------
        np.ndarray
            Latest normalised frame.
        """
        return self._all_frames[fov_id][1, self._channel_to_index[channel], :, :]

    def get_strategy_name(self) -> str:
        """
        Return the current strategy class name.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Strategy name.
        """
        return self._strategy.name()

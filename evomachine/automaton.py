from __future__ import annotations

from collections import deque
import copy
from multiprocessing import Event
from pathlib import Path
import time
from typing import Any, ClassVar

import numpy as np

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger
from evomachine.coordinates import Coordinate
from evomachine.frame import Frame, FrameMetaData
from evomachine.gui.protocol import GuiRequestProcessor
from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.navigation import FocusNavigator
from evomachine.peripherals.autofocus import Autofocus
from evomachine.peripherals.camera import Camera
from evomachine.peripherals.dmd import Dmd, DmdCalibrationConfig
from evomachine.peripherals.filterwheel import FilterWheel
from evomachine.peripherals.leds import LedManager
from evomachine.peripherals.photodiode import Photodiode
from evomachine.peripherals.stage import Stage
from evomachine.projection import ProjectionManager
from evomachine.runtime_errors import (
    CommandExecutionError,
    LifecycleSection,
    UnexpectedRuntimeError,
)
from evomachine.softwarefocus import SoftwareFocus
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType
from evomachine.utils import normalise_frame

logger = get_logger(name=__name__)


class Automaton:
    """Execute strategies through acquisition, focus, and projection managers."""

    _COMMAND_REQUIREMENTS: ClassVar[dict[AutomatonCommandType, tuple[str, ...]]] = {
        AutomatonCommandType.MOVE: ("focus_nav", "_stage"),
        AutomatonCommandType.UPDATE_FOV_CONFIG: ("focus_nav", "_stage"),
        AutomatonCommandType.IMAGE: ("acq_mngr", "_camera", "_led_mngr"),
        AutomatonCommandType.LIVE_MODE: ("acq_mngr", "_camera", "_led_mngr"),
        AutomatonCommandType.PROJECT: ("_dmd", "_led_mngr"),
        AutomatonCommandType.PROJECT_ROI: ("_dmd", "_led_mngr"),
        AutomatonCommandType.WAIT: (),
        AutomatonCommandType.STOP: (),
        AutomatonCommandType.TERMINATE_STRATEGY: (),
        AutomatonCommandType.ABORT_STRATEGY: (),
        AutomatonCommandType.SAVE_STATE: (),
    }
    _REQUIREMENT_LABELS: ClassVar[dict[str, str]] = {
        "acq_mngr": "acquisition manager",
        "focus_nav": "focus navigator",
        "_camera": "camera",
        "_stage": "stage",
        "_led_mngr": "LED manager",
        "_dmd": "DMD",
    }

    def __init__(
            self,
            acq_mngr: FrameAcquisitionManager,
            focus_nav: FocusNavigator,
            strategy: AbstractStrategy | None,
            cfg_processor: ImageProcessorConfig,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            proj_mngr: ProjectionManager | None = None,
            run_timeout: float = 0,
            frame_history_limit: int | None = 2,
            gui_request_processor: GuiRequestProcessor | None = None,
            gui_request_budget: int = 16,
    ):
        """
        Initialise the automaton with runtime manager objects.

        Parameters
        ----------
        acq_mngr
            FrameAcquisitionManager used for camera, LED, filter wheel, and DMD
            commands.
        focus_nav
            FocusNavigator used for stage, autofocus, and software-focus work.
        strategy
            Strategy that produces AutomatonCommand objects, or None for
            device-only startup.
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
        proj_mngr
            Optional ProjectionManager used for DMD calibration and photodiode
            workflows.
        run_timeout
            Delay between loop iterations in seconds.
        frame_history_limit
            Maximum number of raw Frame objects retained per FoV, or None to
            retain all frames in memory.
        gui_request_processor
            Optional callback used by a typed GUI RPC server to process a
            bounded number of pending GUI jobs on the automaton thread.
        gui_request_budget
            Maximum number of typed GUI jobs processed per automaton loop tick.

        Returns
        -------
        None
        """
        if not isinstance(acq_mngr, FrameAcquisitionManager):
            raise TypeError("Automaton.__init__: acq_mngr must be FrameAcquisitionManager.")
        if not isinstance(focus_nav, FocusNavigator):
            raise TypeError("Automaton.__init__: focus_nav must be FocusNavigator.")
        if strategy is not None and not isinstance(strategy, AbstractStrategy):
            raise TypeError("Automaton.__init__: strategy must be AbstractStrategy or None.")
        if proj_mngr is not None and not isinstance(proj_mngr, ProjectionManager):
            raise TypeError("Automaton.__init__: proj_mngr must be ProjectionManager or None.")
        self.acq_mngr: FrameAcquisitionManager = acq_mngr
        "FrameAcquisitionManager used for camera, LED, filter wheel, and DMD acquisition commands."
        self.focus_nav: FocusNavigator = focus_nav
        "FocusNavigator used for stage movement, autofocus, and software-focus work."
        self.proj_mngr: ProjectionManager | None = proj_mngr
        "Optional ProjectionManager used for DMD calibration and photodiode workflows."
        self._camera: Camera = self.acq_mngr.camera
        "Camera cached from acq_mngr for strategy requirement validation and lifecycle handling."
        self._led_mngr: LedManager = self.acq_mngr.led_manager
        "LED manager cached from acq_mngr for illumination, projection, and lifecycle handling."
        self._filt_wheel: FilterWheel | None = self.acq_mngr.filter_wheel
        "Optional filter wheel cached from acq_mngr for lifecycle and consistency checks."
        self._dmd: Dmd | None = self.acq_mngr.dmd
        "Optional DMD cached from acq_mngr for projection commands."
        self._stage: Stage = self.focus_nav.stage
        "Stage cached from focus_nav for lifecycle and requirement validation."
        self._autofocus: Autofocus | None = self.focus_nav.autofocus
        "Optional autofocus cached from focus_nav for halt cleanup and lifecycle handling."
        self._swfocus: SoftwareFocus | None = self.focus_nav.software_focus
        "Optional SoftwareFocus cached from focus_nav for halt cleanup."
        self._photodiode: Photodiode | None = self.proj_mngr.photodiode if self.proj_mngr is not None else None
        "Optional photodiode cached from proj_mngr for lifecycle handling."
        self._validate_manager_device_consistency()
        self._strategy: AbstractStrategy | None = strategy
        "Strategy currently installed on the automaton, or None for device-only startup."
        self._cfg: ImageProcessorConfig = cfg_processor
        "Image-processing configuration used by strategy commands and frame channel lookup."
        self._channel_to_index: dict[LEDType, int] = self._cfg.channel_to_index
        "Mapping from LED channel type to image channel index."
        self._curr_step: int = 0
        "Current strategy step counter reserved for future strategy bookkeeping."
        self._fovs: dict[int, Coordinate] = {}
        "Registered field-of-view coordinates keyed by FoV ID."
        self._fov_to_roi: dict[int, list[int]] = {}
        "ROI IDs keyed by FoV ID for strategy command validation and projection."
        self._cropping_boxes: dict[int, list[Any]] = {}
        "Cropping boxes keyed by FoV ID."
        self._fov_processors: dict[int, Any] = {}
        "Runtime FoV processors keyed by FoV ID."
        self._skip_image_fov_id: int | None = None
        "FoV ID whose next image command should be skipped after failed focus handling."
        self._skip_image_reason: str | None = None
        "Human-readable reason for skipping an image command."

        self.frame_history_limit: int | None = self._validate_frame_history_limit(
            frame_history_limit=frame_history_limit,
        )
        "Maximum number of raw Frame objects retained per FoV, or None to retain all."
        self._all_frames: dict[int, deque[Frame]] = {}
        "Raw acquired Frame objects keyed by FoV ID, newest frame first."

        self._fov_list_is_initialised: bool = False
        "True after FoV state has been registered."
        self._strategy_is_initialised: bool = False
        "True after the current strategy has been initialised."
        self._strategy_is_finalised: bool = False
        "True after finalisation has been requested for the current strategy."
        self._fov_processors_is_initialised: dict[int, bool] = {}
        "Initialisation status of each FoV processor keyed by FoV ID."
        self.next_commands: list[AutomatonCommand] = []
        "Strategy commands scheduled for the next processing step."
        self.last_commands: list[AutomatonCommand] = []
        "Strategy commands executed during the previous processing step."
        self._command_section: LifecycleSection = "initialise"
        "Lifecycle section that produced next_commands."
        self._runtime_failure_history: list[CommandExecutionError | UnexpectedRuntimeError] = []
        "Detailed failures observed during the current Automaton lifetime."
        self._start_strategy_event: Event = start_strategy_event
        "Event set when strategy execution should begin."
        self._stop_strategy_event: Event = stop_strategy_event
        "Event set when strategy execution should stop."
        self._stop_event: Event = stop_event
        "Event set when current automaton work should halt."
        self._shutdown_event: Event = shutdown_event
        "Event set when all automaton loops should exit."
        self.run_timeout: float = self._validate_non_negative_float(run_timeout, "run_timeout")
        "Delay between run-loop ticks in seconds."
        if gui_request_processor is not None and not callable(gui_request_processor):
            raise TypeError(
                f"Automaton.__init__: gui_request_processor must be callable or None, "
                f"received {type(gui_request_processor)}."
            )
        if not isinstance(gui_request_budget, int) or isinstance(gui_request_budget, bool):
            raise TypeError(
                f"Automaton.__init__: gui_request_budget must be int, received {type(gui_request_budget)}."
            )
        if gui_request_budget < 1:
            raise ValueError(f"Automaton.__init__: gui_request_budget must be positive, received {gui_request_budget}.")
        self.gui_request_processor: GuiRequestProcessor | None = gui_request_processor
        "Optional callable that processes pending GUI requests on the automaton thread."
        self.gui_request_budget: int = gui_request_budget
        "Maximum number of GUI requests processed per automaton loop tick."

    def gui_set_request_processor(
            self,
            processor: GuiRequestProcessor | None,
            budget: int | None = None,
    ) -> None:
        """
        Install or clear the typed GUI request processor used by the run loop.

        Parameters
        ----------
        processor
            Callable that processes up to the supplied number of pending GUI
            requests, or None to clear the hook.
        budget
            Optional replacement per-loop processing budget.

        Returns
        -------
        None
        """
        if processor is not None and not callable(processor):
            raise TypeError(
                f"Automaton.gui_set_request_processor: processor must be callable or None, received {type(processor)}."
            )
        if budget is not None:
            if not isinstance(budget, int) or isinstance(budget, bool):
                raise TypeError(f"Automaton.gui_set_request_processor: budget must be int, received {type(budget)}.")
            if budget < 1:
                raise ValueError(f"Automaton.gui_set_request_processor: budget must be positive, received {budget}.")
            self.gui_request_budget = budget
        self.gui_request_processor = processor

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

    @staticmethod
    def _validate_frame_history_limit(frame_history_limit: int | None) -> int | None:
        """
        Return a validated per-FoV frame history limit.

        Parameters
        ----------
        frame_history_limit
            Positive frame count to retain per FoV, or None to retain all.

        Returns
        -------
        int | None
            Validated limit.
        """
        if frame_history_limit is None:
            return None
        if not isinstance(frame_history_limit, int) or isinstance(frame_history_limit, bool):
            raise TypeError(
                "Automaton.__init__: frame_history_limit must be int or None, "
                f"received {type(frame_history_limit)}."
            )
        if frame_history_limit < 1:
            raise ValueError(
                "Automaton.__init__: frame_history_limit must be positive or None, "
                f"received {frame_history_limit}."
            )
        return frame_history_limit

    def _validate_manager_device_consistency(self) -> None:
        """
        Raise if managers expose different objects for a shared device role.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        acquisition_stage = getattr(self.acq_mngr, "stage", None)
        if acquisition_stage is not None and acquisition_stage is not self._stage:
            raise ValueError(
                "Automaton.__init__: acq_mngr.stage and focus_nav.stage "
                "must refer to the same object."
            )
        if self.proj_mngr is None:
            return
        self._validate_shared_manager_device("camera", self._camera, getattr(self.proj_mngr, "camera", None))
        self._validate_shared_manager_device("led_manager", self._led_mngr, getattr(self.proj_mngr, "led_manager", None))
        self._validate_shared_manager_device(
            "filter_wheel",
            self._filt_wheel,
            getattr(self.proj_mngr, "filter_wheel", None),
        )
        self._validate_shared_manager_device("dmd", self._dmd, getattr(self.proj_mngr, "dmd", None))

    @staticmethod
    def _validate_shared_manager_device(device_name: str, acq_device: Any, proj_device: Any) -> None:
        """
        Raise if two managers expose different non-None objects for one device.

        Parameters
        ----------
        device_name
            Attribute name to compare on both managers.
        acq_device
            Device exposed through FrameAcquisitionManager.
        proj_device
            Device exposed through ProjectionManager.

        Returns
        -------
        None
        """
        if acq_device is not None and proj_device is not None and acq_device is not proj_device:
            raise ValueError(
                f"Automaton.__init__: acq_mngr.{device_name} and proj_mngr.{device_name} "
                "must refer to the same object."
            )

    def _registered_strategy_commands(self) -> set[AutomatonCommandType]:
        """
        Return validated command types declared by the current strategy.

        Parameters
        ----------
        None

        Returns
        -------
        set[AutomatonCommandType]
            Command types declared by the current strategy.
        """
        if self._strategy is None:
            return set()
        command_types = self._strategy.register_automaton_commands()
        if not isinstance(command_types, set):
            raise TypeError(
                f"Automaton: strategy.register_automaton_commands() must return set[AutomatonCommandType], "
                f"received {type(command_types)}."
            )
        invalid_command_types = [
            command_type
            for command_type in command_types
            if not isinstance(command_type, AutomatonCommandType)
        ]
        if invalid_command_types:
            raise TypeError(
                "Automaton: strategy.register_automaton_commands() returned non-AutomatonCommandType "
                f"entries: {invalid_command_types}."
            )
        unsupported_command_types = command_types - set(self._COMMAND_REQUIREMENTS)
        if unsupported_command_types:
            unsupported_names = ", ".join(sorted(command_type.name for command_type in unsupported_command_types))
            raise RuntimeError(f"Automaton: unsupported strategy command types: {unsupported_names}.")
        return set(command_types)

    def _validate_strategy_command_requirements(self) -> None:
        """
        Raise if the current strategy declares commands requiring missing devices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        missing_requirements: set[str] = set()
        for command_type in self._registered_strategy_commands():
            for attribute_name in self._COMMAND_REQUIREMENTS[command_type]:
                if getattr(self, attribute_name) is None:
                    device_name = self._REQUIREMENT_LABELS.get(attribute_name, attribute_name)
                    missing_requirements.add(f"{command_type.name}: {device_name}")
        if missing_requirements:
            missing = ", ".join(sorted(missing_requirements))
            strategy_name = self.get_strategy_name() or "<none>"
            raise RuntimeError(
                f"Automaton: strategy {strategy_name} requires missing device(s): {missing}."
            )

    def _validate_commands_are_registered(self, commands: list[AutomatonCommand], source: str) -> None:
        """
        Raise if a strategy emitted a command type it did not declare. 
        NOTE: the strategy test run does not enforce every branch of the strategy code to be covered, 
        so this is not a guarantee that undeclared commands will never be emitted during actual runs. 

        Parameters
        ----------
        commands
            Command list returned by the strategy.
        source
            Human-readable strategy hook name used in error messages.

        Returns
        -------
        None
        """
        registered_commands = self._registered_strategy_commands()
        undeclared_commands = [
            command.command_type
            for command in commands
            if command.command_type not in registered_commands
        ]
        if undeclared_commands:
            names = ", ".join(sorted({command_type.name for command_type in undeclared_commands}))
            raise RuntimeError(
                f"Automaton: strategy {source} emitted undeclared command type(s): {names}."
            )

    def _require_runtime_devices(self, command_type: AutomatonCommandType, attribute_names: tuple[str, ...]) -> None:
        """
        Raise if a runtime command path needs a missing device.

        Parameters
        ----------
        command_type
            Command currently being executed.
        attribute_names
            Private/public attributes that must not be None.

        Returns
        -------
        None
        """
        missing_devices = [
            self._REQUIREMENT_LABELS.get(attribute_name, attribute_name)
            for attribute_name in attribute_names
            if getattr(self, attribute_name) is None
        ]
        if missing_devices:
            missing = ", ".join(missing_devices)
            raise RuntimeError(f"Automaton.{command_type.name}: missing required device(s): {missing}.")

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
        if self._strategy is not None:
            self._validate_strategy_command_requirements()
        self.initialise_devices()
        self._fovs = {fov_id: coordinate.copy() for fov_id, coordinate in fovs.items()}
        self._cropping_boxes = {} if cropping_boxes is None else copy.copy(cropping_boxes)
        self._fov_to_roi = {fov_id: [] for fov_id in self._fovs}
        self._fov_processors = {}
        self._fov_processors_is_initialised = {fov_id: True for fov_id in self._fovs}
        self._fov_list_is_initialised = True
        self._skip_image_fov_id = None
        self._skip_image_reason = None
        fov_configs = None
        if self._strategy is not None:
            self._initialise_strategy()
            fov_configs = self._strategy.initial_fov_configs()
        self.focus_nav.initialise_fovs(
            fov_id_to_coordinate=self._fovs,
            use_autofocus=use_autofocus,
            fov_configs=fov_configs or None,
        )

    def set_strategy(self, strategy: AbstractStrategy) -> None:
        """
        Set or replace the strategy before strategy execution has started.

        Parameters
        ----------
        strategy
            Strategy to install.

        Returns
        -------
        None
        """
        if self.strategy_has_started():
            raise RuntimeError("Automaton.set_strategy: cannot set strategy after start_strategy_event is set.")
        if not isinstance(strategy, AbstractStrategy):
            raise TypeError(f"Automaton.set_strategy: strategy must be AbstractStrategy, received {type(strategy)}.")
        self._strategy = strategy
        self._strategy_is_initialised = False
        self._strategy_is_finalised = False
        self.next_commands = []
        self.last_commands = []
        self._command_section = "initialise"
        if self._fov_list_is_initialised:
            self._initialise_strategy()
            for fov_id, fov_config in self._strategy.initial_fov_configs().items():
                self.focus_nav.update_fov_config(fov_id=fov_id, fov_config=fov_config)

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
        Return lifecycle-capable manager-owned devices in deterministic order.

        Parameters
        ----------
        None

        Returns
        -------
        list[Any]
            Configured peripheral objects.
        """
        ordered_devices = [
            self._camera,
            self._stage,
            self._led_mngr,
            self._filt_wheel,
            self._dmd,
            self._autofocus,
        ]
        if self.proj_mngr is not None:
            ordered_devices.extend([
                getattr(self.proj_mngr, "camera", None),
                getattr(self.proj_mngr, "dmd", None),
                getattr(self.proj_mngr, "led_manager", None),
                getattr(self.proj_mngr, "filter_wheel", None),
                self._photodiode,
            ])
        unique_devices: list[Any] = []
        seen_ids: set[int] = set()
        for device in ordered_devices:
            if device is None or id(device) in seen_ids:
                continue
            if callable(getattr(device, "initialise", None)) and callable(getattr(device, "is_initialised", None)):
                unique_devices.append(device)
                seen_ids.add(id(device))
        return unique_devices

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
        if self._strategy is None:
            raise RuntimeError("Automaton._initialise_strategy: strategy is required.")
        self._validate_strategy_command_requirements()
        self._strategy_is_finalised = False
        self._command_section = "initialise"
        self._strategy.command_factory.update_region_of_interests(region_of_interests=self._fov_to_roi)
        self.next_commands = self._strategy.initialise(
            fovs=self._fovs,
            region_of_interests=self._fov_to_roi,
            fov_processors=self._fov_processors,
            dmd=self._dmd,
        )
        self._validate_commands_are_registered(commands=self.next_commands, source="initialise")
        self._strategy_is_initialised = True

    def gui_process_requests(self) -> None:
        """
        Process a bounded batch of typed GUI requests without owning transport code.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.gui_request_processor is not None:
            self.gui_request_processor(self.gui_request_budget)

    @property
    def runtime_failure_history(
        self,
    ) -> tuple[CommandExecutionError | UnexpectedRuntimeError, ...]:
        """Return detailed runtime failures in observation order."""
        return tuple(self._runtime_failure_history)

    def _process(self, finalise: bool = False) -> None:
        """Execute one batch under the Automaton's fail-safe runtime boundary."""
        section: LifecycleSection = "finalise" if finalise else self._command_section
        try:
            self._process_commands(finalise=finalise)
        except BaseException as error:
            self._fail_safe_abort(error, section=section)
            raise

    def _process_commands(self, finalise: bool = False) -> None:
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
        if self._strategy is None:
            raise RuntimeError("Automaton._process: strategy is required.")
        if not self._strategy_is_initialised:
            raise RuntimeError("Automaton._process: strategy is not initialised.")
        if not finalise and (self.stopped() or self.strategy_has_stopped()):
            return
        if finalise:
            if self._strategy_is_finalised:
                return
            self._strategy_is_finalised = True
            self.last_commands = self.next_commands
            self._command_section = "finalise"
            self.next_commands = self._strategy.finalise()
            self._validate_commands_are_registered(commands=self.next_commands, source="finalise")
        else:
            self._validate_commands_are_registered(commands=self.next_commands, source="next_commands")
        completed_commands: list[AutomatonCommand] = []
        command_errors: list[Exception] = []
        for command in self.next_commands:
            if self.stopped():
                return
            command.command_data = None
            try:
                if command.command_type == AutomatonCommandType.MOVE:
                    command.command_data = self._execute_move(command=command)
                elif command.command_type == AutomatonCommandType.UPDATE_FOV_CONFIG:
                    command.command_data = self._execute_update_fov_config(command=command)
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
                    command.command_execution_time = time.time()
                    command.fov_id = self.get_fov_id()
                    return
                elif command.command_type == AutomatonCommandType.TERMINATE_STRATEGY:
                    if not finalise:
                        self._process_commands(finalise=True)
                    self.stop_strategy()
                    self.stop()
                    command.command_execution_time = time.time()
                    command.fov_id = self.get_fov_id()
                    return
                elif command.command_type == AutomatonCommandType.ABORT_STRATEGY:
                    self.stop_strategy()
                    self.stop()
                    self.act_on_halt()
                    command.command_execution_time = time.time()
                    command.fov_id = self.get_fov_id()
                    return
                elif command.command_type == AutomatonCommandType.SAVE_STATE:
                    self.save_state(filename_suffix=command.command_args)
                elif command.command_type == AutomatonCommandType.LIVE_MODE:
                    self.set_cam_live_mode(status=command.command_args)
                else:
                    raise RuntimeError(
                        f"Automaton._process: unsupported command type {command.command_type}."
                    )
            except Exception as error:
                command.command_execution_time = time.time()
                command.fov_id = self.get_fov_id()
                failure = CommandExecutionError(
                    command_id=command.command_id,
                    command_type=command.command_type,
                    command_args=self._snapshot_command_args(command.command_args),
                    lifecycle_section=self._command_section,
                    original_error=error,
                )
                self._runtime_failure_history.append(failure)
                if finalise or command.command_type in {
                    AutomatonCommandType.TERMINATE_STRATEGY,
                    AutomatonCommandType.ABORT_STRATEGY,
                }:
                    raise failure from error
                command_errors.append(failure)
                break
            command.command_execution_time = time.time()
            command.fov_id = self.get_fov_id()
            completed_commands.append(command)
        if not finalise:
            self.last_commands = completed_commands
            self.next_commands = self._strategy.callback(
                fov_id=self.get_fov_id(),
                data=self.last_commands,
                errors=command_errors,
            )
            self._command_section = "step"
            self._validate_commands_are_registered(commands=self.next_commands, source="callback")

    @staticmethod
    def _snapshot_command_args(command_args: Any) -> Any:
        """Copy diagnostic arguments without allowing copy failures to mask command errors."""
        try:
            return copy.copy(command_args)
        except Exception:
            return repr(command_args)

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
            FocusNavigatorFovRecord returned by FocusNavigator.move().
        """
        target_fov_id = command.command_args
        if target_fov_id is None:
            return None
        if target_fov_id == -1:
            target_fov_id = self.focus_nav.get_next_fov_id(fov_id=self.get_fov_id())
        result = self.focus_nav.move(fov_id=target_fov_id)
        if getattr(result, "skipped", False):
            self._skip_image_fov_id = target_fov_id
            self._skip_image_reason = getattr(result, "skip_reason", None)
        else:
            self._skip_image_fov_id = None
            self._skip_image_reason = None
        return result

    def _execute_update_fov_config(self, command: AutomatonCommand) -> Any:
        """
        Execute one UPDATE_FOV_CONFIG command through FocusNavigator.

        Parameters
        ----------
        command
            UPDATE_FOV_CONFIG command to execute.

        Returns
        -------
        Any
            Updated FocusNavigatorFovRecord.
        """
        args = command.command_args
        if not isinstance(args, dict):
            raise TypeError("Automaton._execute_update_fov_config: command_args must be dict.")
        return self.focus_nav.update_fov_config(
            fov_id=args["fov_id"],
            fov_config=args["fov_config"],
        )

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
        if self._strategy is None:
            raise RuntimeError("Automaton._execute_image: strategy is required.")
        for metadata in metadata_items:
            if not isinstance(metadata, FrameMetaData):
                raise TypeError("Automaton._execute_image: frame_metadata entries must be FrameMetaData.")
            metadata.callback_id = self._strategy.callback_counter
            if metadata.fov_id < 0:
                metadata.fov_id = self.get_fov_id()
        if any(metadata.fov_id == self._skip_image_fov_id for metadata in metadata_items):
            return {
                "skipped": True,
                "skip_reason": self._skip_image_reason,
                "frame_metadata": metadata_items,
                "saved_paths": [None for _ in metadata_items],
            }
        frame = self.acq_mngr.take_frame(
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
        fov_id = frame.fov_id if frame.fov_id >= 0 else self.get_fov_id()
        if fov_id not in self._fovs:
            raise KeyError(f"Automaton._store_frame: unknown fov ID {fov_id}.")
        if fov_id not in self._all_frames:
            self._all_frames[fov_id] = deque(maxlen=self.frame_history_limit)
        self._all_frames[fov_id].appendleft(frame)

    def _frame_channel_arrays(self, frame: Frame, channel: LEDType) -> list[np.ndarray]:
        """
        Return raw image arrays from a stored Frame for one channel.

        Parameters
        ----------
        frame
            Stored frame to inspect.
        channel
            LED channel to match against frame metadata.

        Returns
        -------
        list[np.ndarray]
            Matching raw image arrays in newest-first order within the Frame.
        """
        channel_index = self._channel_to_index[channel]
        matching_arrays: list[np.ndarray] = []
        for frame_index in reversed(range(len(frame.frame_metadata))):
            metadata = frame.frame_metadata[frame_index]
            metadata_channel_index = self._metadata_channel_index(metadata=metadata)
            if metadata_channel_index == channel_index:
                matching_arrays.append(frame.array[frame_index])
        return matching_arrays

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
        self._require_runtime_devices(AutomatonCommandType.PROJECT, ("_dmd", "_led_mngr"))
        assert self._dmd is not None
        args = command.command_args
        self._dmd.display_image(img=args["image"])
        self._led_mngr.set_led(
            led_type=args["channel"],
            brightness=args["brightness"],
            duration=args["duration"] * 1000.0,
        )
        self.sleep(duration=args["duration"])
        self._led_mngr.disable_led()
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
        self._require_runtime_devices(AutomatonCommandType.PROJECT_ROI, ("_dmd", "_led_mngr"))
        assert self._dmd is not None
        args = command.command_args
        fov_id = args["fov_id"]
        if fov_id not in self._fov_processors:
            raise KeyError(f"Automaton._execute_project_roi: unknown fov ID {fov_id}.")
        processor = self._fov_processors[fov_id]
        roi_boxes = [processor.roi_boxes[roi_id] for roi_id in args["roi_ids"]]
        pattern = self._dmd.pattern_from_roi_boxes(
            boxes=roi_boxes,
            fill_x=args["fill_x"],
            fill_y=args["fill_y"],
            invert=args["invert"],
            warp=True,
        )
        self._dmd.display_image(img=pattern)
        self._led_mngr.set_led(
            led_type=args["channel"],
            brightness=args["brightness"],
            duration=args["duration"] * 1000.0,
        )
        self.sleep(duration=args["duration"], set_live_mode=args["set_live_mode"])
        self._led_mngr.disable_led()
        return pattern

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
        try:
            while not self.has_shutdown():
                while not self.strategy_has_started() and not self.has_shutdown():
                    self.gui_process_requests()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
                while not self.strategy_has_stopped() and not self.has_shutdown():
                    self.gui_process_requests()
                    if not self.stopped():
                        self._process()
                    if self.run_timeout > 0:
                        self.sleep(duration=self.run_timeout)
        except BaseException as error:
            self._fail_safe_abort(error, section=self._command_section)
            raise

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
        self._require_runtime_devices(AutomatonCommandType.LIVE_MODE, ("acq_mngr", "_camera"))
        self.acq_mngr.set_camera_live_mode(status=status)

    def dmd_calibrate(
            self,
            cfg: DmdCalibrationConfig,
            filename: str | Path | None = None,
    ) -> None:
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
        None
            ProjectionManager saves calibration data and updates the DMD
            calibration state.
        """
        if self.proj_mngr is None:
            raise RuntimeError("Automaton.dmd_calibrate: proj_mngr is required.")
        self.proj_mngr.dmd_calibrate(cfg=cfg, filename=filename)

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
            self._require_runtime_devices(AutomatonCommandType.WAIT, ("acq_mngr", "_camera", "_led_mngr"))
            self._led_mngr.set_led(led_type=channel, brightness=brightness)
            self.set_cam_live_mode(status=True)
        while time.perf_counter() < end and not self.stopped() and not self.has_shutdown():
            time.sleep(min(0.01, max(0.0, end - time.perf_counter())))
        if set_live_mode:
            self._led_mngr.disable_led()
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
        operations = [("acquisition manager", self.acq_mngr.stop)]
        if self._swfocus is not None and callable(getattr(self._swfocus, "stop", None)):
            operations.append(("software focus", self._swfocus.stop))
        if self._autofocus is not None and callable(getattr(self._autofocus, "unlock", None)):
            operations.append(("autofocus", self._autofocus.unlock))

        failures = []
        for label, operation in operations:
            try:
                operation()
            except Exception as error:
                failures.append(RuntimeError(f"Failed to halt {label}: {error}"))
        if failures:
            raise ExceptionGroup("One or more peripherals failed to halt.", failures)

    def _fail_safe_abort(
        self,
        error: BaseException,
        *,
        section: LifecycleSection,
    ) -> None:
        """Record an unexpected failure and attempt an idempotent emergency halt."""
        already_recorded = any(
            failure is error
            or (
                isinstance(failure, UnexpectedRuntimeError)
                and failure.original_error is error
            )
            for failure in self._runtime_failure_history
        )
        if not already_recorded:
            self._runtime_failure_history.append(
                UnexpectedRuntimeError(
                    lifecycle_section=section,
                    original_error=error,
                )
            )

        already_stopped = self.strategy_has_stopped() and self.stopped()
        self.stop_strategy()
        self.stop()
        if already_stopped:
            return
        try:
            self.act_on_halt()
        except Exception as halt_error:
            self._runtime_failure_history.append(
                UnexpectedRuntimeError(
                    lifecycle_section=section,
                    original_error=halt_error,
                )
            )

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
        if self._strategy is None:
            raise RuntimeError("Automaton.start_strategy: strategy is required.")
        if not self._strategy_is_initialised:
            raise RuntimeError("Automaton.start_strategy: strategy is not initialised.")
        self._validate_strategy_command_requirements()
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
            and self._strategy is not None
            and self._strategy_is_initialised
            and self._fov_list_is_initialised
            and all(self._fov_processors_is_initialised.values())
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

    def get_next_fov_id(self, current_fov: int) -> int:
        """
        Return the next fov ID using FocusNavigator ordering.

        Parameters
        ----------
        current_fov
            Current fov ID.

        Returns
        -------
        int
            Next fov ID.
        """
        return self.focus_nav.get_next_fov_id(fov_id=current_fov)

    def get_fov_id(self) -> int:
        """
        Return the current fov ID known by FocusNavigator.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current fov ID.
        """
        return self.focus_nav.get_current_fov_id()

    def get_frame(self, fov_id: int, channel: LEDType, time_id: int = 0) -> np.ndarray:
        """
        Return one normalised frame for one fov, LED channel, and time index.

        Parameters
        ----------
        fov_id
            FoV ID.
        channel
            LED channel.
        time_id
            Frame history index. 0 is the most recent frame and 1 is the
            previous frame.

        Returns
        -------
        np.ndarray
            Normalised frame.
        """
        if fov_id not in self._all_frames:
            raise KeyError(f"Automaton.get_frame: unknown fov ID {fov_id}.")
        if not isinstance(time_id, int) or isinstance(time_id, bool):
            raise TypeError(f"Automaton.get_frame: time_id must be int, received {type(time_id)}.")
        if time_id < 0:
            raise IndexError(f"Automaton.get_frame: time_id {time_id} is out of range.")
        matching_frames = [
            frame_array
            for frame in self._all_frames[fov_id]
            for frame_array in self._frame_channel_arrays(frame=frame, channel=channel)
        ]
        if time_id >= len(matching_frames):
            raise IndexError(f"Automaton.get_frame: time_id {time_id} is out of range.")
        return normalise_frame(matching_frames[time_id])

    def get_strategy_name(self) -> str | None:
        """
        Return the current strategy class name.

        Parameters
        ----------
        None

        Returns
        -------
        str | None
            Strategy name, or None when no strategy is set.
        """
        if self._strategy is None:
            return None
        return self._strategy.name()

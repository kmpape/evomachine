"""Map the initial microscopy domain pack onto existing EvoMachine behavior."""

from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np

from autostrat.language.model import QuantityValue, ValidatedCommandCall, ValidatedValue

from evomachine.commands import AutomatonCommand
from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.bindings.software_focus.software_focus_algorithms import (
    LaplacianVarianceFocusAlgorithm,
)
from evomachine.frame import FrameMetaDataFactory
from evomachine.navigation import FocusNavigatorFovRecord
from evomachine.strategy_generation.interfaces import (
    CommandAdapter,
    CommandBuildContext,
    ObservationProvider,
    RuntimeErrorProvider,
)
from evomachine.runtime_errors import CommandExecutionError
from evomachine.strategy_generation.runtime import (
    ActiveRuntimeError,
    StrategyInterpretationError,
)
from evomachine.types import AutomatonCommandType, FilterWheelType, LEDType


_LED_BY_DOMAIN_VALUE = {
    "385nm": LEDType.LED_385_NM,
    "450nm": LEDType.LED_450_NM,
    "515nm": LEDType.LED_515_NM,
    "565nm": LEDType.LED_565_NM,
    "645nm": LEDType.LED_645_NM,
}

_FILTER_BY_DOMAIN_VALUE = {
    "465nm": FilterWheelType.FILTER_465nm,
    "527nm": FilterWheelType.FILTER_527nm,
    "592nm": FilterWheelType.FILTER_592nm,
    "no_filter": FilterWheelType.NO_FILTER,
    "blocking": FilterWheelType.BLOCKING,
}

_COMMAND_TYPES = {
    "move_fov": AutomatonCommandType.MOVE,
    "image": AutomatonCommandType.IMAGE,
    "project": AutomatonCommandType.PROJECT,
    "wait": AutomatonCommandType.WAIT,
}


class MicroscopyCommandAdapter(CommandAdapter):
    """Build one existing AutomatonCommand for each microscopy domain call."""

    def __init__(
        self,
        *,
        segment_images: bool,
        save_images: bool,
    ) -> None:
        if not isinstance(segment_images, bool):
            raise TypeError("segment_images must be a bool.")
        if not isinstance(save_images, bool):
            raise TypeError("save_images must be a bool.")
        self._segment_images = segment_images
        self._save_images = save_images

    def command_type(self, call: ValidatedCommandCall) -> AutomatonCommandType:
        try:
            return _COMMAND_TYPES[call.name]
        except KeyError as error:
            raise StrategyInterpretationError(
                f"Unsupported microscopy command {call.name!r}."
            ) from error

    def build(
        self,
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        if call.name == "move_fov":
            return self._build_move_fov(call, context)
        if call.name == "image":
            return self._build_image(call, context)
        if call.name == "project":
            return self._build_project(call, context)
        if call.name == "wait":
            return self._build_wait(call, context)
        raise StrategyInterpretationError(f"Unsupported microscopy command {call.name!r}.")

    @staticmethod
    def _build_move_fov(
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        target = call.arguments["target"]
        if not context.fovs:
            raise StrategyInterpretationError("move_fov requires at least one application-supplied FOV.")
        if target == "first_fov":
            fov_id = next(iter(context.fovs))
        elif target == "next_fov":
            if context.current_fov_id < 0:
                raise StrategyInterpretationError(
                    "move_fov(target=next_fov) requires an established current FOV."
                )
            fov_id = -1
        else:
            raise StrategyInterpretationError(f"Unsupported move_fov target {target!r}.")
        return context.command_factory.command_move(fov_id=fov_id)

    def _build_image(
        self,
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        exposure = call.arguments["exposure"]
        led = call.arguments["led"]
        led_brightness = call.arguments["led_brightness"]
        filter_value = call.arguments["filter"]
        if not isinstance(exposure, QuantityValue) or exposure.unit != "ms":
            raise StrategyInterpretationError("image exposure must be a quantity in ms.")
        if not isinstance(led, str) or led not in _LED_BY_DOMAIN_VALUE:
            raise StrategyInterpretationError(f"Unsupported image LED {led!r}.")
        brightness = self._brightness(led_brightness, argument="image led_brightness")
        if not isinstance(filter_value, str) or filter_value not in _FILTER_BY_DOMAIN_VALUE:
            raise StrategyInterpretationError(f"Unsupported image filter {filter_value!r}.")
        metadata = FrameMetaDataFactory.default(
            leds={_LED_BY_DOMAIN_VALUE[led]: brightness},
            filter_wheel=_FILTER_BY_DOMAIN_VALUE[filter_value],
            exposure=exposure.magnitude,
        )
        return context.command_factory.command_image(
            frame_metadata=metadata,
            segment=self._segment_images,
            save=self._save_images,
        )

    def _build_project(
        self,
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        led = call.arguments["illumination_led"]
        brightness_value = call.arguments["illumination_brightness"]
        duration = call.arguments["duration"]
        if not isinstance(led, str) or led not in _LED_BY_DOMAIN_VALUE:
            raise StrategyInterpretationError(f"Unsupported projection LED {led!r}.")
        brightness = self._brightness(
            brightness_value,
            argument="project illumination_brightness",
        )
        if not isinstance(duration, QuantityValue) or duration.unit != "s":
            raise StrategyInterpretationError("project duration must be a quantity in s.")
        full_field_pattern = np.full(DMD_WIDTH_HEIGHT, 255, dtype=np.uint8)
        return context.command_factory.command_project(
            channel=_LED_BY_DOMAIN_VALUE[led],
            image=full_field_pattern,
            duration=duration.magnitude,
            brightness=brightness,
        )

    @staticmethod
    def _brightness(value: ValidatedValue, *, argument: str) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value <= 100:
            raise StrategyInterpretationError(f"{argument} must be a number in [0, 100].")
        return float(value)

    @staticmethod
    def _build_wait(
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        duration = call.arguments["duration"]
        if not isinstance(duration, QuantityValue) or duration.unit != "s":
            raise StrategyInterpretationError("wait duration must be a quantity in s.")
        return context.command_factory.command_wait(
            duration=duration.magnitude,
            set_live_mode=False,
        )


class MicroscopyObservationProvider(ObservationProvider):
    """Expose lifecycle, focus, and latest-image measurements to the strategy."""

    def __init__(self) -> None:
        self._focus_algorithm = LaplacianVarianceFocusAlgorithm()
        self._started_at: float | None = None
        self._latest: dict[str, ValidatedValue] = {}

    def observe(
        self,
        *,
        fov_id: int,
        completed_commands: list[AutomatonCommand],
        step_count: int,
    ) -> dict[str, ValidatedValue]:
        if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 0:
            raise StrategyInterpretationError("step_count must be a non-negative integer.")
        if not isinstance(fov_id, int) or isinstance(fov_id, bool) or fov_id < -1:
            raise StrategyInterpretationError("fov_id must be -1 or a non-negative integer.")

        if fov_id == -1 and step_count == 0 and not completed_commands:
            self._started_at = time.monotonic()
            self._latest.clear()
        if self._started_at is None:
            self._started_at = time.monotonic()

        for command in completed_commands:
            if command.command_type is AutomatonCommandType.MOVE:
                self._observe_move(command)
            elif command.command_type is AutomatonCommandType.IMAGE:
                self._observe_image(command)

        observations: dict[str, ValidatedValue] = dict(self._latest)
        observations["step_count"] = step_count
        observations["elapsed_time"] = QuantityValue(
            magnitude=max(0.0, time.monotonic() - self._started_at),
            unit="s",
        )
        if fov_id >= 0:
            observations["current_fov_id"] = fov_id
        return observations

    def _observe_move(self, command: AutomatonCommand) -> None:
        result = command.command_data
        if not isinstance(result, FocusNavigatorFovRecord):
            raise StrategyInterpretationError(
                "Completed move command did not contain a FocusNavigatorFovRecord."
            )
        self._latest.update(
            {
                "hardware_autofocus_locked": result.is_locked,
                "focus_recovery_attempted": result.refocusing,
                "software_focus_status": result.software_focus_status.name.lower(),
                "fov_imaging_skipped": result.skipped,
                "focus_recovery_exhausted": result.max_refocus_trials_reached,
            }
        )

    def _observe_image(self, command: AutomatonCommand) -> None:
        result = command.command_data
        if not isinstance(result, dict):
            raise StrategyInterpretationError("Completed image command did not contain image data.")
        if result.get("skipped") is True:
            self._latest["fov_imaging_skipped"] = True
            return
        images = result.get("img")
        if not isinstance(images, list) or not images:
            raise StrategyInterpretationError("Completed image command did not contain an image array.")
        image = np.asarray(images[-1])
        while image.ndim > 2:
            image = image[-1]
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.number):
            raise StrategyInterpretationError("Completed image data must be a numeric two-dimensional array.")

        camera_max = self._camera_max(image)
        image_float = image.astype(np.float64)
        if not np.all(np.isfinite(image_float)):
            raise StrategyInterpretationError("Completed image data contains non-finite values.")
        clipped = np.clip(image_float, 0.0, camera_max)
        contrast = float(np.percentile(clipped, 99) - np.percentile(clipped, 1)) / camera_max
        focus_score = 0.0
        if min(image.shape) >= 3:
            focus_score = self._focus_algorithm.score_image(image)
        if not np.isfinite(focus_score) or focus_score < 0:
            raise StrategyInterpretationError("Focus-score calculation returned an invalid value.")
        self._latest.update(
            {
                "mean_intensity": float(clipped.mean()) / camera_max,
                "contrast_score": contrast,
                "saturation_fraction": float(np.mean(clipped >= 0.98 * camera_max)),
                "focus_score": focus_score,
            }
        )

    @staticmethod
    def _camera_max(image: np.ndarray) -> float:
        if np.issubdtype(image.dtype, np.integer):
            maximum = float(np.iinfo(image.dtype).max)
        elif np.issubdtype(image.dtype, np.floating):
            maximum = 1.0
        else:
            raise StrategyInterpretationError(f"Unsupported image dtype {image.dtype}.")
        return maximum


class MicroscopyRuntimeErrorProvider(RuntimeErrorProvider):
    """Classify Automaton failures into the microscopy domain's stable error names."""

    _DEVICE_NOT_READY_MARKERS = (
        "not initialised",
        "not initialized",
        "not alive",
        "not available",
        "missing required device",
    )
    _COMMUNICATION_MARKERS = (
        "communication",
        "connection",
        "disconnected",
        "serial",
        "socket",
    )

    def classify(
        self,
        *,
        errors: list[Exception],
        command_origins: Mapping[int, ValidatedCommandCall],
    ) -> dict[str, ActiveRuntimeError]:
        if not errors:
            return {}
        if len(errors) > 1:
            raise StrategyInterpretationError(
                "Microscopy runtime classification requires at most one execution error."
            )
        name, failed_call, command_id, occurred_at, cause = self._classify_one(
            errors[0],
            command_origins,
        )
        return {
            name: ActiveRuntimeError(
                name=name,
                failed_call=failed_call,
                original_error=cause,
                command_id=command_id,
                occurred_at=occurred_at,
            )
        }

    @classmethod
    def _classify_one(
        cls,
        supplied_error: Exception,
        command_origins: Mapping[int, ValidatedCommandCall],
    ) -> tuple[str, ValidatedCommandCall | None, int | None, float, Exception]:
        if not isinstance(supplied_error, CommandExecutionError):
            return "runtime_failure", None, None, time.time(), supplied_error

        failed_call = command_origins.get(supplied_error.command_id)
        if failed_call is None:
            return (
                "runtime_failure",
                None,
                supplied_error.command_id,
                supplied_error.occurred_at,
                supplied_error.original_error,
            )

        cause = supplied_error.original_error
        detail = f"{type(cause).__module__}.{type(cause).__name__}: {cause}".lower()
        if any(marker in detail for marker in cls._DEVICE_NOT_READY_MARKERS):
            name = "device_not_ready"
        elif isinstance(cause, (ConnectionError, TimeoutError)) or any(
            marker in detail for marker in cls._COMMUNICATION_MARKERS
        ):
            name = "communication_failed"
        elif supplied_error.command_type is AutomatonCommandType.MOVE:
            name = "movement_failed"
        elif supplied_error.command_type is AutomatonCommandType.IMAGE:
            name = "image_acquisition_failed"
        elif supplied_error.command_type is AutomatonCommandType.PROJECT:
            name = "projection_failed"
        else:
            name = "runtime_failure"
        return (
            name,
            failed_call,
            supplied_error.command_id,
            supplied_error.occurred_at,
            cause,
        )


__all__ = [
    "MicroscopyCommandAdapter",
    "MicroscopyObservationProvider",
    "MicroscopyRuntimeErrorProvider",
]

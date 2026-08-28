"""Map the initial microscopy domain pack onto existing EvoMachine behavior."""

from __future__ import annotations

from collections.abc import Mapping
import time

from autostrat.language.model import QuantityValue, ValidatedCommandCall, ValidatedValue

from evomachine.commands import AutomatonCommand
from evomachine.frame import FrameMetaDataFactory
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
from evomachine.types import AutomatonCommandType, LEDType


_LED_BY_DOMAIN_VALUE = {
    "385nm": LEDType.LED_385_NM,
    "450nm": LEDType.LED_450_NM,
    "515nm": LEDType.LED_515_NM,
    "565nm": LEDType.LED_565_NM,
    "645nm": LEDType.LED_645_NM,
}

_COMMAND_TYPES = {
    "move_fov": AutomatonCommandType.MOVE,
    "image": AutomatonCommandType.IMAGE,
    "wait": AutomatonCommandType.WAIT,
}


class MicroscopyCommandAdapter(CommandAdapter):
    """Build one existing AutomatonCommand for each microscopy domain call."""

    def __init__(
        self,
        *,
        image_brightness: int | float,
        segment_images: bool,
        save_images: bool,
    ) -> None:
        if (
            not isinstance(image_brightness, int | float)
            or isinstance(image_brightness, bool)
            or not 0 <= image_brightness <= 100
        ):
            raise ValueError("image_brightness must be a number in [0, 100].")
        if not isinstance(segment_images, bool):
            raise TypeError("segment_images must be a bool.")
        if not isinstance(save_images, bool):
            raise TypeError("save_images must be a bool.")
        self._image_brightness = float(image_brightness)
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
        if not isinstance(exposure, QuantityValue) or exposure.unit != "ms":
            raise StrategyInterpretationError("image exposure must be a quantity in ms.")
        if not isinstance(led, str) or led not in _LED_BY_DOMAIN_VALUE:
            raise StrategyInterpretationError(f"Unsupported image LED {led!r}.")
        metadata = FrameMetaDataFactory.default(
            leds={_LED_BY_DOMAIN_VALUE[led]: self._image_brightness},
            exposure=exposure.magnitude,
        )
        return context.command_factory.command_image(
            frame_metadata=metadata,
            segment=self._segment_images,
            save=self._save_images,
        )

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
    """Expose simple scalar state available on every applicable lifecycle call."""

    def observe(
        self,
        *,
        fov_id: int,
        completed_commands: list[AutomatonCommand],
        errors: list[Exception],
        step_count: int,
    ) -> dict[str, ValidatedValue]:
        del completed_commands, errors
        if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 0:
            raise StrategyInterpretationError("step_count must be a non-negative integer.")
        if not isinstance(fov_id, int) or isinstance(fov_id, bool) or fov_id < -1:
            raise StrategyInterpretationError("fov_id must be -1 or a non-negative integer.")

        observations: dict[str, ValidatedValue] = {"step_count": step_count}
        if fov_id >= 0:
            observations["current_fov_id"] = fov_id
        return observations


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
        active: dict[str, ActiveRuntimeError] = {}
        for supplied_error in errors:
            name, failed_call, command_id, occurred_at, cause = self._classify_one(
                supplied_error,
                command_origins,
            )
            active[name] = ActiveRuntimeError(
                name=name,
                failed_call=failed_call,
                original_error=supplied_error,
                message=str(supplied_error),
                command_id=command_id,
                exception_type=type(cause).__name__,
                occurred_at=occurred_at,
            )
        return active

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

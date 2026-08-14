"""Deterministic semantic validation for parsed strategy DSL programs."""

from math import isfinite
from typing import Final

from evomachine.dsl.errors import DSLValidationError
from evomachine.dsl.model import (
    CaptureImage,
    Move,
    ParsedStrategy,
    Wait,
)


MIN_EXPOSURE_MS: Final[float] = 1.0
MAX_EXPOSURE_MS: Final[float] = 1000.0
MIN_WAIT_SECONDS: Final[float] = 0.0
MAX_WAIT_SECONDS: Final[float] = 10.0


def validate_strategy(parsed_strategy: ParsedStrategy) -> ParsedStrategy:
    """Validate parsed command parameters and return the same strategy.

    Raises
    ------
    DSLValidationError
        If a command is unsupported or a parameter is outside its accepted range.
    """
    sections = {
        "initialise": parsed_strategy.initialise,
        "callback": parsed_strategy.callback,
        "finalise": parsed_strategy.finalise,
    }
    for section_name, commands in sections.items():
        for command in commands:
            if isinstance(command, Move):
                continue
            if isinstance(command, CaptureImage):
                if (
                        not isfinite(command.exposure_ms)
                        or command.exposure_ms < MIN_EXPOSURE_MS
                        or command.exposure_ms > MAX_EXPOSURE_MS
                ):
                    raise DSLValidationError(
                        f"{section_name}: image exposure must be between "
                        f"{MIN_EXPOSURE_MS:g} and {MAX_EXPOSURE_MS:g} ms; "
                        f"received {command.exposure_ms}."
                    )
                continue
            if isinstance(command, Wait):
                if (
                        not isfinite(command.duration_seconds)
                        or command.duration_seconds <= MIN_WAIT_SECONDS
                        or command.duration_seconds > MAX_WAIT_SECONDS
                ):
                    raise DSLValidationError(
                        f"{section_name}: wait duration must be greater than "
                        f"{MIN_WAIT_SECONDS:g} and at most {MAX_WAIT_SECONDS:g} seconds; "
                        f"received {command.duration_seconds}."
                    )
                continue
            raise DSLValidationError(
                f"{section_name}: unsupported parsed command {type(command).__name__}."
            )
    return parsed_strategy

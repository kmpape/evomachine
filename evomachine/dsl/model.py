"""Immutable intermediate representation for parsed strategy DSL programs."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum

from evomachine.types import LEDType


@dataclass(frozen=True)
class ParsedCommand(ABC):
    """Base type for commands in the parsed DSL representation."""


class MoveTarget(Enum):
    """Define the field-of-view selection modes supported by ``Move``."""

    FIRST_FOV = "first_fov"
    NEXT_FOV = "next_fov"


@dataclass(frozen=True)
class Move(ParsedCommand):
    """Represent movement to a field of view selected at interpretation time."""

    target: MoveTarget


@dataclass(frozen=True)
class CaptureImage(ParsedCommand):
    """Represent an image acquisition and its illumination settings."""

    exposure_ms: float
    led_type: LEDType | None = None


@dataclass(frozen=True)
class Wait(ParsedCommand):
    """Represent a wait with duration measured in seconds."""

    duration_seconds: float


@dataclass(frozen=True)
class ParsedStrategy:
    """Store parsed commands for each strategy lifecycle section."""

    initialise: tuple[ParsedCommand, ...]
    callback: tuple[ParsedCommand, ...]
    finalise: tuple[ParsedCommand, ...]

"""Typed failures crossing the Automaton-to-strategy runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal

from evomachine.types import AutomatonCommandType


LifecycleSection = Literal["initialise", "step", "finalise"]


@dataclass(slots=True)
class CommandExecutionError(RuntimeError):
    """Preserve one failed Automaton command and its original exception."""

    command_id: int
    command_type: AutomatonCommandType
    command_args: Any
    lifecycle_section: LifecycleSection
    original_error: Exception
    occurred_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        message = (
            f"{self.command_type.name} command {self.command_id} failed during "
            f"{self.lifecycle_section}: {type(self.original_error).__name__}: "
            f"{self.original_error}"
        )
        RuntimeError.__init__(self, message)


@dataclass(slots=True)
class UnexpectedRuntimeError(RuntimeError):
    """Record an unexpected failure outside normal command execution."""

    lifecycle_section: LifecycleSection
    original_error: BaseException
    occurred_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        message = (
            f"Unexpected runtime failure during {self.lifecycle_section}: "
            f"{type(self.original_error).__name__}: {self.original_error}"
        )
        RuntimeError.__init__(self, message)


__all__ = ["CommandExecutionError", "LifecycleSection", "UnexpectedRuntimeError"]

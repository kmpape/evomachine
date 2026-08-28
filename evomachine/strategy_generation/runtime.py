"""Runtime values used while interpreting validated AutoStrat programs."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping

from autostrat.language.model import ControlActionName, ValidatedCommandCall, ValidatedValue


class StrategyInterpretationError(RuntimeError):
    """Raised when a validated program cannot be evaluated against runtime state."""


@dataclass(frozen=True, slots=True)
class ActiveRuntimeError:
    """Describe one strategy-visible error active for the current lifecycle call."""

    name: str
    failed_call: ValidatedCommandCall | None = None
    original_error: Exception | None = None
    message: str = ""
    command_id: int | None = None
    exception_type: str | None = None
    occurred_at: float = field(default_factory=time.time)
    retry_attempt: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ActiveRuntimeError.name must be a non-empty string.")
        if self.command_id is not None and self.command_id < 0:
            raise ValueError("ActiveRuntimeError.command_id must be non-negative or None.")
        if self.retry_attempt < 0:
            raise ValueError("ActiveRuntimeError.retry_attempt must be non-negative.")


@dataclass(frozen=True, slots=True)
class StrategyRuntimeContext:
    """Provide one immutable-by-convention snapshot to the conditional interpreter."""

    observations: Mapping[str, ValidatedValue] = field(default_factory=dict)
    errors: Mapping[str, ActiveRuntimeError] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", dict(self.observations))
        object.__setattr__(self, "errors", dict(self.errors))
        for name, error in self.errors.items():
            if name != error.name:
                raise ValueError(
                    f"Runtime error key {name!r} must match ActiveRuntimeError.name {error.name!r}."
                )


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    """Return command calls selected by the interpreter and optional control flow."""

    calls: tuple[ValidatedCommandCall, ...] = ()
    action: ControlActionName | None = None
    retry_error: ActiveRuntimeError | None = None
    inspected_errors: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.action not in {None, "continue", "retry", "terminate", "abort"}:
            raise ValueError(f"Unsupported interpretation action {self.action!r}.")
        if self.action == "retry" and self.retry_error is None:
            raise ValueError("A retry result requires the active runtime error being retried.")
        if self.action != "retry" and self.retry_error is not None:
            raise ValueError("retry_error is only valid for a retry result.")


__all__ = [
    "ActiveRuntimeError",
    "InterpretationResult",
    "StrategyInterpretationError",
    "StrategyRuntimeContext",
]

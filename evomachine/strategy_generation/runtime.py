"""Runtime values used while interpreting validated AutoStrat programs."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from autostrat.language.evaluator import ExecutionContext
from autostrat.language.model import ValidatedCommandCall


class StrategyInterpretationError(RuntimeError):
    """Raised when a validated program cannot be evaluated against runtime state."""


@dataclass(frozen=True, slots=True)
class ActiveRuntimeError:
    """Describe one strategy-visible error active for the current lifecycle call."""

    name: str
    failed_call: ValidatedCommandCall | None = None
    original_error: Exception | None = None
    command_id: int | None = None
    occurred_at: float = field(default_factory=time.time)
    retry_attempt: int = 0
    statement_path: str = ""
    collection_context: ExecutionContext = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ActiveRuntimeError.name must be a non-empty string.")
        if self.command_id is not None and self.command_id < 0:
            raise ValueError("ActiveRuntimeError.command_id must be non-negative or None.")
        if self.retry_attempt < 0:
            raise ValueError("ActiveRuntimeError.retry_attempt must be non-negative.")

    @property
    def message(self) -> str:
        """Return the original exception message without storing a duplicate value."""
        return "" if self.original_error is None else str(self.original_error)

    @property
    def exception_type(self) -> str | None:
        """Return the original exception type without storing a duplicate value."""
        return None if self.original_error is None else type(self.original_error).__name__


__all__ = [
    "ActiveRuntimeError",
    "StrategyInterpretationError",
]

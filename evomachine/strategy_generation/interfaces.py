"""Dependency-injection interfaces for the AutoStrat/EvoMachine boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from autostrat.language.evaluator import ExecutionContext
from autostrat.language.model import ValidatedCommandCall, ValidatedCommandTemplate, ValidatedValue

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.coordinates import Coordinate
from evomachine.delta_processing import MicroscopyState
from evomachine.strategy_generation.runtime import ActiveRuntimeError, StrategyInterpretationError
from evomachine.types import AutomatonCommandType


@dataclass(frozen=True, slots=True)
class CommandBuildContext:
    """Expose application state required to build one AutomatonCommand."""

    command_factory: CommandFactory
    fovs: Mapping[int, Coordinate]
    current_fov_id: int
    selections: ExecutionContext = ()
    processing_state: MicroscopyState | None = None


class CommandAdapter(ABC):
    """Translate validated domain calls into application-owned commands."""

    @abstractmethod
    def command_type(
        self, call: ValidatedCommandCall | ValidatedCommandTemplate
    ) -> AutomatonCommandType:
        """Return the Automaton command type produced for a validated call."""

    @abstractmethod
    def build(
        self,
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        """Build one application command from a validated domain call."""


class ObservationProvider(ABC):
    """Produce global values at section entry and after each completed command."""

    @abstractmethod
    def observe(
        self,
        *,
        fov_id: int,
        completed_commands: list[AutomatonCommand],
        step_count: int,
    ) -> Mapping[str, ValidatedValue]:
        """Refresh global values; step_count counts completed DSL steps, not commands."""

    def invalidate(self) -> None:
        """Discard cached measurements after an abandoned failed command."""

    def reset(self) -> None:
        """Begin a new experiment without carrying observations from an earlier run."""
        self.invalidate()

    def bind_processing_state(self, state) -> None:
        """Optionally bind the application's shared measurement state."""


class RuntimeErrorProvider(ABC):
    """Classify application exceptions into domain-declared runtime errors."""

    @abstractmethod
    def classify(
        self,
        *,
        errors: list[Exception],
        command_origins: Mapping[int, ValidatedCommandCall],
    ) -> Mapping[str, ActiveRuntimeError]:
        """Return zero or one active strategy-facing error keyed by its domain name."""


class EmptyObservationProvider(ObservationProvider):
    """Provide no observations until a deployment wires concrete metrics."""

    def observe(
        self,
        *,
        fov_id: int,
        completed_commands: list[AutomatonCommand],
        step_count: int,
    ) -> Mapping[str, ValidatedValue]:
        del fov_id, completed_commands, step_count
        return {}


class EmptyRuntimeErrorProvider(RuntimeErrorProvider):
    """Fail closed if errors arrive before a deployment defines classifications."""

    def classify(
        self,
        *,
        errors: list[Exception],
        command_origins: Mapping[int, ValidatedCommandCall],
    ) -> Mapping[str, ActiveRuntimeError]:
        del command_origins
        if errors:
            raise StrategyInterpretationError(
                "Runtime errors were supplied, but no RuntimeErrorProvider is configured."
            )
        return {}


__all__ = [
    "CommandAdapter",
    "CommandBuildContext",
    "CollectionProvider",
    "EmptyObservationProvider",
    "EmptyRuntimeErrorProvider",
    "ObservationProvider",
    "RuntimeErrorProvider",
]


class CollectionProvider:
    """Application-owned records. Iteration order and scopes are interpreted by AutoStrat."""

    def bind(self, *, fovs, region_of_interests, fov_processors) -> None:
        pass

    def items(self, collection: str, context: ExecutionContext) -> tuple[int | str, ...]:
        raise StrategyInterpretationError(f"No provider for collection {collection!r}")

    def observe(self, context: ExecutionContext) -> Mapping[str, ValidatedValue]:
        return {}

    def invalidate(self, context: ExecutionContext) -> None:
        """Discard measurements affected by a failed command in this selected context."""

    def bind_processing_state(self, state) -> None:
        """Optionally bind the application's shared measurement state."""

"""An AbstractStrategy implementation backed by a validated AutoStrat program."""

from __future__ import annotations

from collections.abc import Iterable

from autostrat.domain import DomainPack
from autostrat.language.model import (
    ControlAction,
    ValidatedCommandCall,
    ValidatedIfStatement,
    ValidatedStrategyProgram,
    ValidatedStatement,
)
from autostrat.pipeline import VerifiedStrategy

from evomachine.commands import AutomatonCommand
from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.strategy import AbstractStrategy
from evomachine.strategy_generation.interfaces import (
    CommandAdapter,
    CommandBuildContext,
    EmptyObservationProvider,
    EmptyRuntimeErrorProvider,
    ObservationProvider,
    RuntimeErrorProvider,
)
from evomachine.strategy_generation.interpreter import ConditionalInterpreter
from evomachine.strategy_generation.runtime import (
    StrategyInterpretationError,
    StrategyRuntimeContext,
)
from evomachine.types import AutomatonCommandType


class AutoStratStrategy(AbstractStrategy):
    """Execute one immutable, validated AutoStrat program through injected adapters."""

    def __init__(
        self,
        cfg: ImageProcessorConfig,
        *,
        verified: VerifiedStrategy,
        domain: DomainPack,
        command_adapter: CommandAdapter,
        observation_provider: ObservationProvider | None = None,
        runtime_error_provider: RuntimeErrorProvider | None = None,
    ) -> None:
        super().__init__(cfg=cfg)
        if not isinstance(verified, VerifiedStrategy):
            raise TypeError("verified must be a VerifiedStrategy.")
        if not isinstance(domain, DomainPack):
            raise TypeError("domain must be a DomainPack.")
        if verified.domain_id != domain.metadata.id or verified.domain_version != domain.metadata.version:
            raise ValueError("Verified strategy and domain pack metadata do not match.")
        if not isinstance(command_adapter, CommandAdapter):
            raise TypeError("command_adapter must be a CommandAdapter.")

        self.verified = verified
        self.domain = domain
        self.command_adapter = command_adapter
        self.observation_provider = observation_provider or EmptyObservationProvider()
        self.runtime_error_provider = runtime_error_provider or EmptyRuntimeErrorProvider()
        self._interpreter = ConditionalInterpreter()
        if not isinstance(self.observation_provider, ObservationProvider):
            raise TypeError("observation_provider must be an ObservationProvider.")
        if not isinstance(self.runtime_error_provider, RuntimeErrorProvider):
            raise TypeError("runtime_error_provider must be a RuntimeErrorProvider.")
        self._command_origins: dict[int, ValidatedCommandCall] = {}
        self._runtime_context = StrategyRuntimeContext()
        self._current_fov_id = -1

    @property
    def source(self) -> str:
        """Return the accepted DSL source used by this strategy."""
        return self.verified.source

    @property
    def program(self) -> ValidatedStrategyProgram:
        """Return the accepted, deterministically validated program."""
        return self.verified.program

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return every command type that any validated branch may emit."""
        command_types = {
            self.command_adapter.command_type(call)
            for call in self._all_command_calls(
                (*self.program.initialise, *self.program.step, *self.program.finalise)
            )
        }
        if self._contains_action(
            (*self.program.initialise, *self.program.step, *self.program.finalise),
            "terminate",
        ):
            command_types.add(AutomatonCommandType.TERMINATE_STRATEGY)
        if self._contains_action(
            (*self.program.initialise, *self.program.step, *self.program.finalise),
            "abort",
        ):
            command_types.add(AutomatonCommandType.ABORT_STRATEGY)
        return command_types

    def _initialise(self) -> list[AutomatonCommand]:
        self._runtime_context = self._build_runtime_context(
            fov_id=-1,
            completed_commands=[],
            errors=[],
        )
        return self._commands_for(self.program.initialise, current_fov_id=-1)

    def _callback(
        self,
        fov_id: int,
        data: list[AutomatonCommand],
        errors: list[Exception],
    ) -> list[AutomatonCommand]:
        self._current_fov_id = fov_id
        self._runtime_context = self._build_runtime_context(
            fov_id=fov_id,
            completed_commands=data,
            errors=errors,
        )
        return self._commands_for(self.program.step, current_fov_id=fov_id)

    def finalise(self) -> list[AutomatonCommand]:
        return self._commands_for(
            self.program.finalise,
            current_fov_id=self._current_fov_id,
        )

    def _build_runtime_context(
        self,
        *,
        fov_id: int,
        completed_commands: list[AutomatonCommand],
        errors: list[Exception],
    ) -> StrategyRuntimeContext:
        observations = self.observation_provider.observe(
            fov_id=fov_id,
            completed_commands=completed_commands,
            errors=errors,
            step_count=self.callback_counter,
        )
        active_errors = self.runtime_error_provider.classify(
            errors=errors,
            command_origins=self._command_origins,
        )
        unknown_observations = set(observations) - set(self.domain.observations)
        if unknown_observations:
            raise StrategyInterpretationError(
                f"Observation provider returned undeclared values: {sorted(unknown_observations)!r}."
            )
        unknown_errors = set(active_errors) - set(self.domain.runtime_errors)
        if unknown_errors:
            raise StrategyInterpretationError(
                f"Runtime error provider returned undeclared errors: {sorted(unknown_errors)!r}."
            )
        return StrategyRuntimeContext(observations=observations, errors=active_errors)

    def _commands_for(
        self,
        statements: tuple[ValidatedStatement, ...],
        *,
        current_fov_id: int,
    ) -> list[AutomatonCommand]:
        interpreted = self._interpreter.interpret(statements, self._runtime_context)
        calls = list(interpreted.calls)
        if interpreted.action == "retry":
            retry_error = interpreted.retry_error
            if retry_error is None or retry_error.failed_call is None:
                raise StrategyInterpretationError(
                    "Interpreter returned retry without an associated failed command."
                )
            calls.append(retry_error.failed_call)

        context = CommandBuildContext(
            command_factory=self.command_factory,
            fovs=self.fovs,
            current_fov_id=current_fov_id,
        )
        commands = []
        for call in calls:
            expected_type = self.command_adapter.command_type(call)
            command = self.command_adapter.build(call, context)
            if not isinstance(command, AutomatonCommand):
                raise TypeError("CommandAdapter.build() must return an AutomatonCommand.")
            if command.command_type is not expected_type:
                raise ValueError(
                    f"Command adapter declared {expected_type.name} for {call.name!r} "
                    f"but built {command.command_type.name}."
                )
            self._command_origins[command.command_id] = call
            commands.append(command)

        if interpreted.action == "terminate":
            commands.append(self.command_factory.command_terminate_strategy())
        elif interpreted.action == "abort":
            commands.append(self.command_factory.command_abort_strategy())
        return commands

    @classmethod
    def _all_command_calls(
        cls,
        statements: Iterable[ValidatedStatement],
    ) -> tuple[ValidatedCommandCall, ...]:
        calls = []
        for statement in statements:
            if isinstance(statement, ValidatedCommandCall):
                calls.append(statement)
            elif isinstance(statement, ValidatedIfStatement):
                calls.extend(cls._all_command_calls(statement.body))
                calls.extend(cls._all_command_calls(statement.else_body))
        return tuple(calls)

    @classmethod
    def _contains_action(
        cls,
        statements: Iterable[ValidatedStatement],
        action: str,
    ) -> bool:
        for statement in statements:
            if isinstance(statement, ControlAction) and statement.action == action:
                return True
            if isinstance(statement, ValidatedIfStatement):
                if cls._contains_action(statement.body, action) or cls._contains_action(
                    statement.else_body, action
                ):
                    return True
        return False


__all__ = ["AutoStratStrategy"]

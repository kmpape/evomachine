"""An AbstractStrategy implementation backed by a validated AutoStrat program."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from autostrat.domain import DomainPack, RecoveryAction
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
    ActiveRuntimeError,
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
        self._command_tails: dict[int, tuple[ValidatedCommandCall, ...]] = {}
        self._command_tail_actions: dict[int, RecoveryAction | None] = {}
        self._retry_counts: dict[tuple[str, int], int] = {}
        self._failure_history: list[ActiveRuntimeError] = []
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

    @property
    def failure_history(self) -> tuple[ActiveRuntimeError, ...]:
        """Return every classified runtime failure in observation order."""
        return tuple(self._failure_history)

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
        policy_actions = {
            action
            for definition in self.domain.runtime_errors.values()
            for action in (definition.default_action, definition.retry_exhausted_action)
            if action is not None
        }
        if "terminate" in policy_actions:
            command_types.add(AutomatonCommandType.TERMINATE_STRATEGY)
        if "abort" in policy_actions:
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
        self._discard_pending_batch()
        self._runtime_context = StrategyRuntimeContext(
            observations=self._runtime_context.observations,
        )
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
        if len(errors) > 1:
            raise StrategyInterpretationError(
                "A stopped Automaton command batch cannot report more than one execution error."
            )
        observations = self.observation_provider.observe(
            fov_id=fov_id,
            completed_commands=completed_commands,
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
        if len(active_errors) > 1:
            raise StrategyInterpretationError(
                "Runtime error provider returned more than one active error for one execution error."
            )
        self._validate_error_origins(active_errors)
        self._clear_successful_retries(completed_commands, errors)
        active_with_attempts = {
            name: replace(
                error,
                retry_attempt=self._retry_counts.get(self._retry_key(error), 0),
                remaining_calls=self._command_tails.get(error.command_id, ()),
                remaining_action=self._command_tail_actions.get(error.command_id),
            )
            for name, error in active_errors.items()
        }
        self._failure_history.extend(active_with_attempts.values())
        self._discard_pending_batch()
        return StrategyRuntimeContext(observations=observations, errors=active_with_attempts)

    def _commands_for(
        self,
        statements: tuple[ValidatedStatement, ...],
        *,
        current_fov_id: int,
    ) -> list[AutomatonCommand]:
        interpreted = self._interpreter.interpret(statements, self._runtime_context)
        calls = list(interpreted.calls)
        action = interpreted.action
        action_error = interpreted.action_error
        if self._runtime_context.errors and action is None:
            error_name = next(iter(self._runtime_context.errors))
            raise StrategyInterpretationError(
                f"Active runtime error {error_name!r} was not handled by the validated strategy."
            )
        if self._runtime_context.errors:
            calls = []

        if action == "retry":
            if action_error is None or action_error.failed_call is None:
                raise StrategyInterpretationError(
                    "Retry requires an active error associated with a failed command."
                )
            calls = []
            action = self._apply_retry_policy(action_error, calls)
        elif action == "continue" and action_error is not None:
            self._retry_counts.pop(self._retry_key(action_error), None)
            calls = list(action_error.remaining_calls)
            action = action_error.remaining_action or "continue"
        elif action is not None:
            self._clear_active_retry_counts()

        context = CommandBuildContext(
            command_factory=self.command_factory,
            fovs=self.fovs,
            current_fov_id=current_fov_id,
        )
        built_commands: list[tuple[ValidatedCommandCall, AutomatonCommand]] = []
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
            built_commands.append((call, command))

        commands = [command for _, command in built_commands]
        for index, (call, command) in enumerate(built_commands):
            self._command_origins[command.command_id] = call
            self._command_tails[command.command_id] = tuple(calls[index + 1:])
            self._command_tail_actions[command.command_id] = action if action != "retry" else None

        if action == "terminate":
            commands.append(self.command_factory.command_terminate_strategy())
        elif action == "abort":
            commands.append(self.command_factory.command_abort_strategy())
        return commands

    def _apply_retry_policy(
        self,
        error: ActiveRuntimeError,
        calls: list[ValidatedCommandCall],
    ) -> RecoveryAction:
        """Retry one failed call or return its automatic exhaustion action."""
        definition = self.domain.runtime_errors[error.name]
        if "retry" not in definition.allowed_actions:
            raise StrategyInterpretationError(
                f"Runtime error {error.name!r} does not permit retry."
            )
        if error.failed_call is None:
            raise StrategyInterpretationError(
                f"Runtime error {error.name!r} has no failed command to retry."
            )
        key = self._retry_key(error)
        attempts = self._retry_counts.get(key, 0)
        if attempts < definition.max_retries:
            self._retry_counts[key] = attempts + 1
            calls.append(error.failed_call)
            calls.extend(error.remaining_calls)
            return error.remaining_action or "retry"
        exhausted_action = definition.retry_exhausted_action
        if exhausted_action is None:
            raise StrategyInterpretationError(
                f"Retryable runtime error {error.name!r} has no exhaustion action."
            )
        self._retry_counts.pop(key, None)
        if exhausted_action == "continue":
            calls.extend(error.remaining_calls)
            return error.remaining_action or "continue"
        return exhausted_action

    def _validate_error_origins(
        self,
        active_errors: Mapping[str, ActiveRuntimeError],
    ) -> None:
        """Require command-linked classifications to match the failed call's declaration."""
        for name, error in active_errors.items():
            if error.failed_call is None:
                continue
            command = self.domain.commands.get(error.failed_call.name)
            if command is None or name not in command.runtime_errors:
                raise StrategyInterpretationError(
                    f"Runtime error {name!r} is not declared for failed command "
                    f"{error.failed_call.name!r}."
                )

    def _clear_successful_retries(
        self,
        completed_commands: list[AutomatonCommand],
        errors: list[Exception],
    ) -> None:
        failed_ids = {
            error.command_id
            for error in errors
            if hasattr(error, "command_id")
        }
        completed_call_ids = {
            id(self._command_origins[command.command_id])
            for command in completed_commands
            if command.command_id not in failed_ids and command.command_id in self._command_origins
        }
        for key in tuple(self._retry_counts):
            if key[1] in completed_call_ids:
                self._retry_counts.pop(key, None)

    def _discard_pending_batch(self) -> None:
        """Discard bookkeeping for the one batch whose callback is now being handled."""
        self._command_origins.clear()
        self._command_tails.clear()
        self._command_tail_actions.clear()

    def _clear_active_retry_counts(self) -> None:
        for error in self._runtime_context.errors.values():
            self._retry_counts.pop(self._retry_key(error), None)

    @staticmethod
    def _retry_key(error: ActiveRuntimeError) -> tuple[str, int]:
        failed_call_id = id(error.failed_call) if error.failed_call is not None else -1
        return error.name, failed_call_id

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

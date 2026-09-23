"""Sequential AutoStrat execution through the existing Automaton command boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from autostrat.domain import DomainPack
from autostrat.language.evaluator import StrategyExecution
from autostrat.language.model import (
    ControlAction,
    ValidatedCommandCall,
    ValidatedCommandTemplate,
    ValidatedIfStatement,
    ValidatedLoopStatement,
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
    CollectionProvider,
    EmptyObservationProvider,
    EmptyRuntimeErrorProvider,
    ObservationProvider,
    RuntimeErrorProvider,
)
from evomachine.strategy_generation.runtime import ActiveRuntimeError, StrategyInterpretationError
from evomachine.types import AutomatonCommandType


class _Host:
    def __init__(self, strategy):
        self.strategy = strategy

    def items(self, collection, context):
        return self.strategy.collection_provider.items(collection, context)

    def observe(self, context):
        global_values = self.strategy._observations
        scoped = self.strategy.collection_provider.observe(context)
        overlap = set(global_values) & set(scoped)
        if overlap:
            raise StrategyInterpretationError(
                f"Duplicate global/scoped observations: {sorted(overlap)}"
            )
        return {**global_values, **scoped}


class AutoStratStrategy(AbstractStrategy):
    """One interpreter run; each hardware callback resumes, rather than restarts, its section."""

    def __init__(
        self,
        cfg: ImageProcessorConfig,
        *,
        verified: VerifiedStrategy,
        domain: DomainPack,
        command_adapter: CommandAdapter,
        observation_provider: ObservationProvider | None = None,
        runtime_error_provider: RuntimeErrorProvider | None = None,
        collection_provider: CollectionProvider | None = None,
    ) -> None:
        super().__init__(cfg=cfg)
        if not isinstance(verified, VerifiedStrategy):
            raise TypeError("verified must be a VerifiedStrategy.")
        if not isinstance(domain, DomainPack):
            raise TypeError("domain must be a DomainPack.")
        if (
            verified.domain_id != domain.metadata.id
            or verified.domain_version != domain.metadata.version
        ):
            raise ValueError("Verified strategy and domain pack metadata do not match.")
        if not isinstance(command_adapter, CommandAdapter):
            raise TypeError("command_adapter must be a CommandAdapter.")

        self.verified = verified
        self.domain = domain
        self.command_adapter = command_adapter
        self.observation_provider = observation_provider or EmptyObservationProvider()
        self.runtime_error_provider = runtime_error_provider or EmptyRuntimeErrorProvider()
        self.collection_provider = collection_provider or CollectionProvider()
        self._execution = StrategyExecution(domain, verified.program)
        self._host = _Host(self)
        if not isinstance(self.observation_provider, ObservationProvider):
            raise TypeError("observation_provider must be an ObservationProvider.")
        if not isinstance(self.runtime_error_provider, RuntimeErrorProvider):
            raise TypeError("runtime_error_provider must be a RuntimeErrorProvider.")
        if not isinstance(self.collection_provider, CollectionProvider):
            raise TypeError("collection_provider must be a CollectionProvider.")
        self._command_origins: dict[int, ValidatedCommandCall] = {}
        self._retry_counts: dict[tuple[str, int], int] = {}
        self._failure_history: list[ActiveRuntimeError] = []
        self._observations = {}
        self._pending_event = None
        self._pending_id = None
        self._steps_completed = 0
        self._next_section = "initialise"
        self._current_fov_id = -1
        self._stopped = False

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
            for action in (definition.action, definition.exhausted_action)
            if action is not None
        }
        if "terminate" in policy_actions:
            command_types.add(AutomatonCommandType.TERMINATE_STRATEGY)
        if "abort" in policy_actions:
            command_types.add(AutomatonCommandType.ABORT_STRATEGY)
        return command_types

    @property
    def pending_section(self):
        return self._execution.section

    def _initialise(self):
        self._stopped = False
        self._execution.reset()
        self._command_origins.clear()
        self._retry_counts.clear()
        self._failure_history.clear()
        self._pending_event = None
        self._pending_id = None
        self._steps_completed = 0
        self._current_fov_id = -1
        self.collection_provider.bind(
            fovs=self.fovs,
            region_of_interests=self.region_of_interests,
            fov_processors=self.fov_processors,
        )
        self.collection_provider.bind_processing_state(self.processing_state)
        self.observation_provider.bind_processing_state(self.processing_state)
        self.observation_provider.reset()
        self._refresh([])
        self._execution.start("initialise")
        return self._advance()

    def callback(self, fov_id, data, errors):
        commands = self._callback(fov_id, data, errors)
        if not self.is_valid_command_list(commands):
            raise RuntimeError("AutoStrat callback returned invalid commands")
        self.callback_counter = self._steps_completed
        return commands

    def _callback(self, fov_id, data, errors):
        if self._stopped:
            return []
        self._current_fov_id = fov_id
        # Recovery precedes all observation collection and resumed DSL evaluation.
        if errors:
            return self._recover(errors)
        self._complete(data)
        self._refresh(data)
        if not self._execution.active:
            self._execution.start(self._next_section)
        return self._advance()

    def _refresh(self, data):
        self._observations = self.observation_provider.observe(
            fov_id=self._current_fov_id, completed_commands=data, step_count=self._steps_completed
        )

    def _complete(self, data):
        if self._pending_event is None:
            if data:
                raise StrategyInterpretationError("Unexpected completed commands")
            return
        if len(data) != 1 or data[0].command_id != self._pending_id:
            raise StrategyInterpretationError("Pending command completion is missing or mismatched")
        call = self._pending_event.call
        for key in tuple(self._retry_counts):
            if key[1] == id(call):
                self._retry_counts.pop(key)
        self._execution.acknowledge()
        self._pending_event = None
        self._pending_id = None
        self._command_origins.clear()

    def _advance(self):
        event = self._execution.resume(self._host)
        if event.call is not None:
            self._pending_event = event
            return self._build_pending()
        if event.action is not None:
            return self._control(event.action)
        if event.section == "step":
            self._steps_completed += 1
            self.callback_counter = self._steps_completed
        self._next_section = "step"
        return []

    def _build_pending(self):
        event = self._pending_event
        call = event.call
        context = CommandBuildContext(
            self.command_factory,
            self.fovs,
            self._current_fov_id,
            event.context,
            self.processing_state,
        )
        expected = self.command_adapter.command_type(call)
        command = self.command_adapter.build(call, context)
        if not isinstance(command, AutomatonCommand) or command.command_type is not expected:
            raise StrategyInterpretationError("Adapter returned the wrong command type")
        self._command_origins.clear()
        self._command_origins[command.command_id] = call
        self._pending_id = command.command_id
        return [command]

    def _control(self, action):
        self._stopped = True
        self._execution.cancel()
        self._pending_event = None
        self._pending_id = None
        self._command_origins.clear()
        if action == "terminate":
            return [self.command_factory.command_terminate_strategy()]
        return [self.command_factory.command_abort_strategy()]

    def _recover(self, errors):
        if len(errors) != 1:
            raise StrategyInterpretationError(
                "A stopped command can report only one execution error"
            )
        active = self.runtime_error_provider.classify(
            errors=errors, command_origins=self._command_origins
        )
        if len(active) != 1:
            raise StrategyInterpretationError("Expected one active error classification")
        name, error = next(iter(active.items()))
        if name not in self.domain.runtime_errors or name != error.name:
            raise StrategyInterpretationError("Undeclared or mismatched runtime error")
        if error.failed_call is not None:
            if self._pending_event is None or error.failed_call is not self._pending_event.call:
                raise StrategyInterpretationError("Error does not belong to the pending command")
            if name not in self.domain.commands[error.failed_call.name].runtime_errors:
                raise StrategyInterpretationError("Error is not declared for the failed command")
        key = (name, id(error.failed_call))
        attempts = self._retry_counts.get(key, 0)
        self._failure_history.append(
            replace(
                error,
                retry_attempt=attempts,
                statement_path=self._pending_event.path if self._pending_event else "",
                collection_context=self._pending_event.context if self._pending_event else (),
            )
        )
        policy = self.domain.runtime_errors[name]
        action = policy.action
        if action == "retry":
            if self._pending_event is None or error.failed_call is None:
                raise StrategyInterpretationError("Cannot retry without a pending command")
            if attempts < policy.max_retries:
                self._retry_counts[key] = attempts + 1
                return self._build_pending()
            action = policy.exhausted_action
        self._retry_counts.pop(key, None)
        if action in {"terminate", "abort"}:
            return self._control(action)
        if self._pending_event is None:
            raise StrategyInterpretationError("Cannot continue without a pending command")
        self.observation_provider.invalidate()
        self.collection_provider.invalidate(self._pending_event.context)
        self._observations = {}
        self._execution.acknowledge()
        self._pending_event = None
        self._pending_id = None
        self._command_origins.clear()
        self._refresh([])
        return self._advance()

    def finalise(self):
        self._execution.cancel()
        self._pending_event = None
        self._pending_id = None
        self._command_origins.clear()
        self._refresh([])
        self._execution.start("finalise")
        return self._advance()

    def resume_finalise(self, fov_id, data):
        self._current_fov_id = fov_id
        self._complete(data)
        self._refresh(data)
        return self._advance() if self._execution.active else []

    @classmethod
    def _all_command_calls(
        cls,
        statements: Iterable[ValidatedStatement],
    ) -> tuple[ValidatedCommandTemplate, ...]:
        calls = []
        for statement in statements:
            if isinstance(statement, ValidatedCommandTemplate):
                calls.append(statement)
            elif isinstance(statement, ValidatedLoopStatement):
                calls.extend(cls._all_command_calls(statement.body))
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
            if isinstance(statement, ValidatedLoopStatement) and cls._contains_action(
                statement.body, action
            ):
                return True
            if isinstance(statement, ValidatedIfStatement):
                if cls._contains_action(statement.body, action) or cls._contains_action(
                    statement.else_body, action
                ):
                    return True
        return False


__all__ = ["AutoStratStrategy"]

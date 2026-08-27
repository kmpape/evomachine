"""Focused tests for the generic AutoStrat integration infrastructure."""

from __future__ import annotations

from threading import current_thread

import pytest

from autostrat import load_domain_pack
from autostrat.generation import GeneratedStrategy, Prompt
from autostrat.language.model import (
    ControlAction,
    ReferenceExpression,
    ValidatedCommandCall,
    ValidatedComparisonExpression,
    ValidatedIfStatement,
)
from autostrat.language.parser import parse_strategy
from autostrat.language.validator import validate_strategy
from autostrat.pipeline import StrategyAttempt, VerifiedStrategy
from autostrat.verification import SemanticVerdict

from evomachine.commands import AutomatonCommand
from evomachine.coordinates import Coordinate
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.strategy import AbstractStrategy
from evomachine.strategy_generation import (
    ActiveRuntimeError,
    AutoStratStrategy,
    CommandAdapter,
    CommandBuildContext,
    ConditionalInterpreter,
    EmptyRuntimeErrorProvider,
    StrategyGenerationService,
    StrategyInterpretationError,
    StrategyRuntimeContext,
)
from evomachine.types import AutomatonCommandType, LEDType


class FakeCommandAdapter(CommandAdapter):
    """Map every test call to a WAIT command without defining microscopy behavior."""

    def command_type(self, call: ValidatedCommandCall) -> AutomatonCommandType:
        del call
        return AutomatonCommandType.WAIT

    def build(
        self,
        call: ValidatedCommandCall,
        context: CommandBuildContext,
    ) -> AutomatonCommand:
        del call
        return context.command_factory.command_wait(duration=0)


def _domain():
    return load_domain_pack("evomachine/domain_packs/microscopy")


def _verified(source: str) -> VerifiedStrategy:
    domain = _domain()
    program = validate_strategy(parse_strategy(source), domain)
    generated = GeneratedStrategy(
        source=source,
        program=program,
        prompt=Prompt(messages=(), recipe_name="test"),
    )
    attempt = StrategyAttempt(
        number=1,
        generated=generated,
        verdict=SemanticVerdict(accepted=True, issues=()),
    )
    return VerifiedStrategy(
        request="test request",
        accepted=attempt,
        attempts=(attempt,),
        domain_id=domain.metadata.id,
        domain_version=domain.metadata.version,
    )


def _cfg():
    return ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM],
        channels_seg=[LEDType.LED_450_NM],
    )


def test_interpreter_selects_nested_branch_from_runtime_snapshot() -> None:
    capture = ValidatedCommandCall(name="capture")
    statements = (
        ValidatedIfStatement(
            condition=ValidatedComparisonExpression(
                left=ReferenceExpression(namespace="observation", name="focus_score"),
                operator="<",
                right=0.5,
            ),
            body=(
                ValidatedIfStatement(
                    condition=ReferenceExpression(namespace="observation", name="recovery_enabled"),
                    body=(capture,),
                ),
            ),
        ),
    )

    result = ConditionalInterpreter().interpret(
        statements,
        StrategyRuntimeContext(
            observations={"focus_score": 0.2, "recovery_enabled": True},
        ),
    )

    assert result.calls == (capture,)
    assert result.action is None


def test_interpreter_retries_the_call_owned_by_the_active_error() -> None:
    failed_call = ValidatedCommandCall(name="capture")
    active_error = ActiveRuntimeError(name="camera_failed", failed_call=failed_call)
    statements = (
        ValidatedIfStatement(
            condition=ReferenceExpression(namespace="error", name="camera_failed"),
            body=(ControlAction(action="retry"),),
        ),
    )

    result = ConditionalInterpreter().interpret(
        statements,
        StrategyRuntimeContext(errors={"camera_failed": active_error}),
    )

    assert result.action == "retry"
    assert result.retry_error is active_error


def test_unconfigured_error_provider_does_not_silently_discard_errors() -> None:
    provider = EmptyRuntimeErrorProvider()

    with pytest.raises(StrategyInterpretationError, match="no RuntimeErrorProvider"):
        provider.classify(errors=[RuntimeError("camera failed")], command_origins={})


def test_verified_program_is_wrapped_as_an_abstract_strategy() -> None:
    verified = _verified(
        "initialise\n"
        "    wait(duration=1s)\n"
        "step\n"
        "finalise\n"
    )
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=verified,
        domain=_domain(),
        command_adapter=FakeCommandAdapter(),
    )

    commands = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )

    assert isinstance(strategy, AbstractStrategy)
    assert strategy.source == verified.source
    assert strategy.program is verified.program
    assert [command.command_type for command in commands] == [AutomatonCommandType.WAIT]
    assert strategy.register_automaton_commands() == {AutomatonCommandType.WAIT}


def test_generation_service_runs_pipeline_on_its_worker() -> None:
    verified = _verified("initialise\nstep\nfinalise\n")

    class FakePipeline:
        thread_name: str | None = None

        def run(self, request: str) -> VerifiedStrategy:
            assert request == "build a strategy"
            self.thread_name = current_thread().name
            return verified

    pipeline = FakePipeline()
    with StrategyGenerationService(
        pipeline=pipeline,
        domain=_domain(),
        command_adapter=FakeCommandAdapter(),
    ) as service:
        strategy = service.submit("build a strategy", _cfg()).result(timeout=5)

    assert strategy.source == verified.source
    assert isinstance(strategy, AbstractStrategy)
    assert pipeline.thread_name is not None
    assert pipeline.thread_name.startswith("strategy-generation")

"""Focused tests for the generic AutoStrat integration infrastructure."""

from __future__ import annotations

import asyncio
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
    MicroscopyCommandAdapter,
    MicroscopyObservationProvider,
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
        thread_names: list[str] = []

        def run(self, request: str) -> VerifiedStrategy:
            assert request == "build a strategy"
            self.thread_names.append(current_thread().name)
            assert not asyncio.get_event_loop().is_running()
            return verified

    pipeline = FakePipeline()
    with StrategyGenerationService(
        pipeline=pipeline,
        domain=_domain(),
        command_adapter=FakeCommandAdapter(),
    ) as service:
        submitted_strategy = service.submit("build a strategy", _cfg()).result(timeout=5)

        async def build_from_running_loop():
            return service.build("build a strategy", _cfg())

        notebook_strategy = asyncio.run(build_from_running_loop())

    assert submitted_strategy.source == verified.source
    assert notebook_strategy.source == verified.source
    assert isinstance(submitted_strategy, AbstractStrategy)
    assert isinstance(notebook_strategy, AbstractStrategy)
    assert len(pipeline.thread_names) == 2
    assert all(name.startswith("strategy-generation") for name in pipeline.thread_names)


def test_microscopy_domain_exposes_only_implemented_initial_behavior() -> None:
    domain = _domain()

    assert set(domain.commands) == {"move_fov", "image", "wait"}
    assert set(domain.observations) == {"current_fov_id", "step_count"}
    assert domain.runtime_errors == {}


def test_microscopy_adapter_builds_existing_automaton_commands() -> None:
    verified = _verified(
        "initialise\n"
        "    move_fov(target=first_fov)\n"
        "    image(exposure=25ms, led=515nm)\n"
        "    wait(duration=3s)\n"
        "step\n"
        "finalise\n"
    )
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=verified,
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            image_brightness=12,
            segment_images=False,
            save_images=True,
        ),
        observation_provider=MicroscopyObservationProvider(),
    )

    commands = strategy.initialise(
        fovs={4: Coordinate(1, 2, 3)},
        region_of_interests={4: []},
        fov_processors={},
        dmd=None,
    )

    assert [command.command_type for command in commands] == [
        AutomatonCommandType.MOVE,
        AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
    ]
    assert commands[0].command_args == 4
    image_args = commands[1].command_args
    assert image_args["frame_metadata"].exposure == 25
    assert image_args["frame_metadata"].leds == {LEDType.LED_515_NM: 12}
    assert image_args["segment"] is False
    assert image_args["save"] is True
    assert commands[2].command_args["duration"] == 3
    assert commands[2].command_args["set_live_mode"] is False


def test_microscopy_observations_drive_step_conditionals() -> None:
    verified = _verified(
        "initialise\n"
        "step\n"
        "    if observation.current_fov_id == 4:\n"
        "        move_fov(target=next_fov)\n"
        "    if observation.step_count == 0:\n"
        "        wait(duration=1s)\n"
        "finalise\n"
    )
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=verified,
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            image_brightness=10,
            segment_images=False,
            save_images=True,
        ),
        observation_provider=MicroscopyObservationProvider(),
    )
    strategy.initialise(
        fovs={4: Coordinate(1, 2, 3)},
        region_of_interests={4: []},
        fov_processors={},
        dmd=None,
    )

    first_step = strategy.callback(fov_id=4, data=[], errors=[])
    second_step = strategy.callback(fov_id=4, data=[], errors=[])

    assert [command.command_type for command in first_step] == [
        AutomatonCommandType.MOVE,
        AutomatonCommandType.WAIT,
    ]
    assert first_step[0].command_args == -1
    assert [command.command_type for command in second_step] == [AutomatonCommandType.MOVE]


def test_autostrat_maps_terminate_and_abort_to_distinct_lifecycle_commands() -> None:
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified(
            "initialise\n"
            "step\n"
            "    if observation.step_count == 0:\n"
            "        abort\n"
            "    else:\n"
            "        terminate\n"
            "finalise\n"
        ),
        domain=_domain(),
        command_adapter=FakeCommandAdapter(),
        observation_provider=MicroscopyObservationProvider(),
    )
    strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )

    first_step = strategy.callback(fov_id=0, data=[], errors=[])
    second_step = strategy.callback(fov_id=0, data=[], errors=[])

    assert first_step[-1].command_type is AutomatonCommandType.ABORT_STRATEGY
    assert second_step[-1].command_type is AutomatonCommandType.TERMINATE_STRATEGY
    assert strategy.register_automaton_commands() == {
        AutomatonCommandType.ABORT_STRATEGY,
        AutomatonCommandType.TERMINATE_STRATEGY,
    }


def test_move_fov_requires_valid_application_context() -> None:
    verified = _verified(
        "initialise\n"
        "    move_fov(target=first_fov)\n"
        "step\n"
        "finalise\n"
    )
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=verified,
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            image_brightness=10,
            segment_images=False,
            save_images=True,
        ),
    )

    with pytest.raises(StrategyInterpretationError, match="requires at least one"):
        strategy.initialise(
            fovs={},
            region_of_interests={},
            fov_processors={},
            dmd=None,
        )

    next_fov_strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified(
            "initialise\n"
            "    move_fov(target=next_fov)\n"
            "step\n"
            "finalise\n"
        ),
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            image_brightness=10,
            segment_images=False,
            save_images=True,
        ),
    )
    with pytest.raises(StrategyInterpretationError, match="established current FOV"):
        next_fov_strategy.initialise(
            fovs={4: Coordinate(1, 2, 3)},
            region_of_interests={4: []},
            fov_processors={},
            dmd=None,
        )

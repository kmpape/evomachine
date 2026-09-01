"""Focused tests for the generic AutoStrat integration infrastructure."""

from __future__ import annotations

from threading import current_thread

import numpy as np
import pytest

from autostrat import load_domain_pack
from autostrat.generation import GeneratedStrategy, Prompt
from autostrat.language.model import (
    ControlAction,
    QuantityValue,
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
from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.coordinates import Coordinate
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.navigation import FocusNavigatorFovRecord, FovConfig
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
    MicroscopyRuntimeErrorProvider,
    StrategyGenerationService,
    StrategyInterpretationError,
    StrategyRuntimeContext,
)
from evomachine.types import AutomatonCommandType, FilterWheelType, FocusStatusType, LEDType
from evomachine.runtime_errors import CommandExecutionError


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


ERROR_HANDLERS = (
    "    if error.device_not_ready:\n"
    "        abort\n"
    "    if error.movement_failed:\n"
    "        retry\n"
    "    if error.image_acquisition_failed:\n"
    "        retry\n"
    "    if error.projection_failed:\n"
    "        retry\n"
    "    if error.communication_failed:\n"
    "        retry\n"
    "    if error.runtime_failure:\n"
    "        abort\n"
)


def _domain():
    return load_domain_pack("evomachine/domain_packs/microscopy")


def _verified(source: str) -> VerifiedStrategy:
    domain = _domain()
    source = source.replace("step\n", f"step\n{ERROR_HANDLERS}", 1)
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
    assert result.action_error is active_error
    assert result.inspected_errors == frozenset({"camera_failed"})


def test_unconfigured_error_provider_does_not_silently_discard_errors() -> None:
    provider = EmptyRuntimeErrorProvider()

    with pytest.raises(StrategyInterpretationError, match="no RuntimeErrorProvider"):
        provider.classify(errors=[RuntimeError("camera failed")], command_origins={})


@pytest.mark.parametrize(
    ("command_type", "cause", "expected_name"),
    [
        (AutomatonCommandType.MOVE, RuntimeError("stage is not initialised"), "device_not_ready"),
        (AutomatonCommandType.MOVE, RuntimeError("motion rejected"), "movement_failed"),
        (AutomatonCommandType.IMAGE, RuntimeError("no frame returned"), "image_acquisition_failed"),
        (AutomatonCommandType.PROJECT, RuntimeError("DMD rejected pattern"), "projection_failed"),
        (AutomatonCommandType.IMAGE, ConnectionError("socket closed"), "communication_failed"),
    ],
)
def test_microscopy_runtime_errors_are_classified_with_diagnostics(
    command_type: AutomatonCommandType,
    cause: Exception,
    expected_name: str,
) -> None:
    call_name = {
        AutomatonCommandType.IMAGE: "image",
        AutomatonCommandType.MOVE: "move_fov",
        AutomatonCommandType.PROJECT: "project",
    }[command_type]
    call = ValidatedCommandCall(name=call_name)
    supplied = CommandExecutionError(
        command_id=7,
        command_type=command_type,
        command_args={"example": True},
        lifecycle_section="step",
        original_error=cause,
    )

    active = MicroscopyRuntimeErrorProvider().classify(
        errors=[supplied],
        command_origins={7: call},
    )[expected_name]

    assert active.failed_call is call
    assert active.command_id == 7
    assert active.exception_type == type(cause).__name__
    assert str(cause) in active.message


def test_image_retry_exhaustion_automatically_continues() -> None:
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified(
            "initialise\n"
            "    image(exposure=25ms, led=450nm, led_brightness=10, filter=465nm)\n"
            "    wait(duration=1s)\n"
            "step\n"
            "    image(exposure=25ms, led=450nm, led_brightness=10, filter=465nm)\n"
            "    wait(duration=1s)\n"
            "finalise\n"
        ),
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            segment_images=False,
            save_images=False,
        ),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )
    command = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )[0]

    returned_commands = []
    for _ in range(3):
        failure = CommandExecutionError(
            command_id=command.command_id,
            command_type=command.command_type,
            command_args=command.command_args,
            lifecycle_section="step",
            original_error=RuntimeError("camera returned no frame"),
        )
        returned_commands = strategy.callback(fov_id=0, data=[], errors=[failure])
        if returned_commands:
            command = returned_commands[0]

    assert [command.command_type for command in returned_commands] == [
        AutomatonCommandType.WAIT
    ]
    assert [failure.retry_attempt for failure in strategy.failure_history] == [0, 1, 2]


def test_retry_preserves_interrupted_batch_tail_and_discards_old_tracking() -> None:
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified(
            "initialise\n"
            "    wait(duration=1s)\n"
            "    image(exposure=25ms, led=450nm, led_brightness=10, filter=465nm)\n"
            "    wait(duration=2s)\n"
            "    if observation.step_count == 0:\n"
            "        terminate\n"
            "step\n"
            "finalise\n"
        ),
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            segment_images=False,
            save_images=False,
        ),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )
    original = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )
    failure = CommandExecutionError(
        command_id=original[1].command_id,
        command_type=original[1].command_type,
        command_args=original[1].command_args,
        lifecycle_section="initialise",
        original_error=RuntimeError("camera returned no frame"),
    )

    resumed = strategy.callback(fov_id=0, data=[original[0]], errors=[failure])

    assert [command.command_type for command in resumed] == [
        AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
        AutomatonCommandType.TERMINATE_STRATEGY,
    ]
    assert resumed[1].command_args["duration"] == 2
    assert not ({command.command_id for command in original} & set(strategy._command_origins))
    assert set(strategy._command_origins) == {
        command.command_id
        for command in resumed
        if command.command_type is not AutomatonCommandType.TERMINATE_STRATEGY
    }


def test_movement_retry_exhaustion_terminates_without_leaking_into_finalise() -> None:
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified(
            "initialise\n"
            "    move_fov(target=first_fov)\n"
            "step\n"
            "finalise\n"
            "    wait(duration=1s)\n"
        ),
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            segment_images=False,
            save_images=False,
        ),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )
    command = strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )[0]

    returned_commands = []
    for _ in range(3):
        failure = CommandExecutionError(
            command_id=command.command_id,
            command_type=command.command_type,
            command_args=command.command_args,
            lifecycle_section="step",
            original_error=RuntimeError("stage rejected movement"),
        )
        returned_commands = strategy.callback(fov_id=0, data=[], errors=[failure])
        if returned_commands[0].command_type is AutomatonCommandType.MOVE:
            command = returned_commands[0]

    assert [command.command_type for command in returned_commands] == [
        AutomatonCommandType.TERMINATE_STRATEGY
    ]
    assert [command.command_type for command in strategy.finalise()] == [
        AutomatonCommandType.WAIT
    ]


def test_unclassified_runtime_failure_preserves_diagnostics_and_aborts() -> None:
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=_verified("initialise\nstep\nfinalise\n"),
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
            segment_images=False,
            save_images=False,
        ),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )
    strategy.initialise(
        fovs={0: Coordinate(0, 0, 0)},
        region_of_interests={0: []},
        fov_processors={},
        dmd=None,
    )

    commands = strategy.callback(
        fov_id=0,
        data=[],
        errors=[RuntimeError("unexpected adapter failure")],
    )

    assert [command.command_type for command in commands] == [
        AutomatonCommandType.ABORT_STRATEGY
    ]
    failure = strategy.failure_history[-1]
    assert failure.name == "runtime_failure"
    assert failure.failed_call is None
    assert failure.exception_type == "RuntimeError"
    assert "unexpected adapter failure" in failure.message


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
    assert strategy.register_automaton_commands() == {
        AutomatonCommandType.WAIT,
        AutomatonCommandType.TERMINATE_STRATEGY,
        AutomatonCommandType.ABORT_STRATEGY,
    }


def test_generation_service_distinguishes_blocking_build_from_worker_submission() -> None:
    verified = _verified("initialise\nstep\nfinalise\n")

    class FakePipeline:
        thread_names: list[str] = []

        def run(self, request: str) -> VerifiedStrategy:
            assert request == "build a strategy"
            self.thread_names.append(current_thread().name)
            return verified

    pipeline = FakePipeline()
    with StrategyGenerationService(
        pipeline=pipeline,
        domain=_domain(),
        command_adapter=FakeCommandAdapter(),
    ) as service:
        blocking_strategy = service.build("build a strategy", _cfg())
        submitted_strategy = service.submit("build a strategy", _cfg()).result(timeout=5)

    assert blocking_strategy.source == verified.source
    assert submitted_strategy.source == verified.source
    assert isinstance(blocking_strategy, AbstractStrategy)
    assert isinstance(submitted_strategy, AbstractStrategy)
    assert len(pipeline.thread_names) == 2
    assert pipeline.thread_names[0] == current_thread().name
    assert pipeline.thread_names[1].startswith("strategy-generation")


def test_microscopy_domain_exposes_runtime_error_policies() -> None:
    domain = _domain()

    assert set(domain.commands) == {"move_fov", "image", "project", "wait"}
    assert set(domain.observations) == {
        "current_fov_id",
        "step_count",
        "elapsed_time",
        "mean_intensity",
        "contrast_score",
        "saturation_fraction",
        "focus_score",
        "hardware_autofocus_locked",
        "focus_recovery_attempted",
        "software_focus_status",
        "fov_imaging_skipped",
        "focus_recovery_exhausted",
    }
    assert set(domain.runtime_errors) == {
        "device_not_ready",
        "movement_failed",
        "image_acquisition_failed",
        "projection_failed",
        "communication_failed",
        "runtime_failure",
    }
    assert domain.commands["image"].runtime_errors == (
        "device_not_ready",
        "image_acquisition_failed",
        "communication_failed",
    )
    assert domain.runtime_errors["image_acquisition_failed"].retry_exhausted_action == "continue"
    assert domain.runtime_errors["projection_failed"].retry_exhausted_action == "continue"


def test_microscopy_adapter_builds_existing_automaton_commands() -> None:
    verified = _verified(
        "initialise\n"
        "    move_fov(target=first_fov)\n"
        "    image(exposure=25ms, led=515nm, led_brightness=12, filter=527nm)\n"
        "    project(illumination_led=385nm, illumination_brightness=20, duration=2s)\n"
        "    wait(duration=3s)\n"
        "step\n"
        "finalise\n"
    )
    strategy = AutoStratStrategy(
        cfg=_cfg(),
        verified=verified,
        domain=_domain(),
        command_adapter=MicroscopyCommandAdapter(
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
        AutomatonCommandType.PROJECT,
        AutomatonCommandType.WAIT,
    ]
    assert commands[0].command_args == 4
    image_args = commands[1].command_args
    assert image_args["frame_metadata"].exposure == 25
    assert image_args["frame_metadata"].leds == {LEDType.LED_515_NM: 12}
    assert image_args["frame_metadata"].filter_wheel is FilterWheelType.FILTER_527nm
    assert image_args["segment"] is False
    assert image_args["save"] is True
    project_args = commands[2].command_args
    assert project_args["channel"] is LEDType.LED_385_NM
    assert project_args["brightness"] == 20
    assert project_args["duration"] == 2
    assert project_args["image"].shape == DMD_WIDTH_HEIGHT
    assert project_args["image"].dtype == np.uint8
    assert np.all(project_args["image"] == 255)
    assert commands[3].command_args["duration"] == 3
    assert commands[3].command_args["set_live_mode"] is False


def test_microscopy_provider_exposes_latest_image_and_focus_results() -> None:
    provider = MicroscopyObservationProvider()
    provider.observe(fov_id=-1, completed_commands=[], errors=[], step_count=0)
    move = AutomatonCommand(
        command_id=1,
        command_type=AutomatonCommandType.MOVE,
        command_args=0,
        command_creation_time=0,
        command_data=FocusNavigatorFovRecord(
            fov_id=0,
            coordinate=Coordinate(0, 0, 0),
            fov_config=FovConfig(),
            is_locked=True,
            refocusing=True,
            software_focus_status=FocusStatusType.IN_FOCUS,
        ),
    )
    image_array = np.array(
        [
            [0, 0, 0, 0],
            [0, 64, 128, 0],
            [0, 128, 255, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    image = AutomatonCommand(
        command_id=2,
        command_type=AutomatonCommandType.IMAGE,
        command_args={},
        command_creation_time=0,
        command_data={"img": [image_array[np.newaxis, ...]]},
    )

    observations = provider.observe(
        fov_id=0,
        completed_commands=[move, image],
        errors=[],
        step_count=1,
    )

    assert observations["hardware_autofocus_locked"] is True
    assert observations["focus_recovery_attempted"] is True
    assert observations["software_focus_status"] == "in_focus"
    assert observations["fov_imaging_skipped"] is False
    assert observations["focus_recovery_exhausted"] is False
    assert observations["mean_intensity"] == pytest.approx(float(image_array.mean()) / 255)
    assert 0 <= observations["contrast_score"] <= 1
    assert observations["saturation_fraction"] == pytest.approx(1 / image_array.size)
    assert observations["focus_score"] >= 0
    assert isinstance(observations["elapsed_time"], QuantityValue)
    assert observations["elapsed_time"].unit == "s"


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

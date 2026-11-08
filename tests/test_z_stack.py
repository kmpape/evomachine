"""Strategy Z-stack command validation and acquisition integration tests."""

import pytest

from evomachine.delta_processing import FovMeasurements
from evomachine.runtime_errors import CommandExecutionError
from evomachine.types import AutomatonCommandType
from tests.test_acquisition import FakeStage
from tests.test_automaton import FakeStrategy, make_automaton
from tests.test_commands import make_factory, make_metadata


class StackStrategy(FakeStrategy):
    """Test strategy declaring Z_STACK in addition to the fake's commands."""

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return the set of command types emitted by this test strategy."""
        return super().register_automaton_commands() | {AutomatonCommandType.Z_STACK}


@pytest.fixture
def stack_automaton():
    """Return an initialised Automaton with a movable fake stage and stack strategy."""
    automaton, manager, navigator, *_ = make_automaton()
    stage = FakeStage()
    manager.stage = stage
    navigator.stage = stage
    automaton._stage = stage
    automaton.set_strategy(StackStrategy(automaton._cfg))
    navigator.current_fov_id = 0
    return automaton, manager, stage


@pytest.mark.parametrize("name,value,error", [
    ("frame_metadata", None, TypeError),
    ("frame_metadata", [make_metadata()], TypeError),
    ("save", 1, TypeError),
    ("relative_range", True, TypeError),
    ("step_size", False, TypeError),
    ("relative_range", "10", TypeError),
    ("step_size", None, TypeError),
    ("relative_range", 0, ValueError),
    ("step_size", -1, ValueError),
    ("relative_range", float("inf"), ValueError),
    ("step_size", float("nan"), ValueError),
    ("step_size", 11, ValueError),
])
def test_z_stack_validation(name, value, error):
    """Reject invalid argument values with the expected exception type.

    Args:
        name: Command argument to replace.
        value: Invalid argument value.
        error: Expected exception class.
    """
    arguments = dict(frame_metadata=make_metadata(), relative_range=10, step_size=3, save=False)
    arguments[name] = value
    with pytest.raises(error, match=name):
        make_factory().command_z_stack(**arguments)


@pytest.mark.parametrize("relative_range,step_size,offsets", [
    (10, 3, [0, 3, 6, 9, -3, -6, -9]),
    (4, 2, [0, 2, 4, -2, -4]),
    (2, 2, [0, 2, -2]),
    (0.3, 0.1, [0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3]),
    (0.35, 0.1, [0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3]),
])
def test_z_stack_execution(stack_automaton, relative_range, step_size, offsets):
    """Acquire expected offsets and expose the complete stack through callback data.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
        relative_range: Half-range in stage units.
        step_size: Requested spacing in stage units.
        offsets: Expected centre-first relative capture positions.
    """
    automaton, manager, stage = stack_automaton
    metadata = make_metadata()
    command = automaton._strategy.command_factory.command_z_stack(
        metadata, relative_range, step_size, save=True,
    )
    assert command.command_type is AutomatonCommandType.Z_STACK
    stage.coordinate.z = 25  # Changed after command construction.
    automaton.processing_state.fovs[0] = FovMeasurements(valid=True, projection_targets={0})
    automaton._fov_to_roi[0] = [0, 1]
    processor = object()
    automaton._fov_processors[0] = processor
    automaton.next_commands = [command]
    automaton._execute_strategy_batch()
    assert not automaton._strategy.received_errors
    data = command.command_data
    assert [item.coordinate.z for item in data["frame_metadata"]] == pytest.approx(
        [25 + offset for offset in offsets]
    )
    assert all(item.fov_id == 0 and item.callback_id == 0 for item in data["frame_metadata"])
    assert all(
        item.leds == metadata.leds and item.exposure == metadata.exposure
        for item in data["frame_metadata"]
    )
    assert all(settings.save for _, settings in manager.calls)
    assert data["img"][0].shape == (len(offsets), 3, 4)
    assert data["saved_paths"] == [None] * len(offsets)
    assert stage.coordinate.z == 25
    assert automaton._latest_frame.array is data["img"][0]
    assert automaton._latest_frame_sequence == 1
    assert not automaton.processing_state.fovs[0].valid
    assert not automaton.processing_state.fovs[0].projection_targets
    assert automaton._fov_to_roi[0] == [0, 1]
    assert automaton._fov_processors[0] is processor
    assert automaton._delta_processor._roi_model is None
    assert automaton._delta_processor._seg_model is None


def test_z_stack_skipped_fov(stack_automaton):
    """Skip acquisition for a focus-failed FOV, returning the IMAGE skip payload.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
    """
    automaton, manager, _ = stack_automaton
    automaton._skip_image_fov_id = 0
    automaton._skip_image_reason = "focus failed"
    command = make_factory().command_z_stack(make_metadata(), 2, 1)
    data = automaton._execute_z_stack(command)
    assert data["skipped"] and data["skip_reason"] == "focus failed"
    assert not manager.calls
    assert automaton._latest_frame is None


def test_z_stack_requires_acquisition_stage(stack_automaton):
    """Reject missing acquisition-stage hardware during registration and execution.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
    """
    automaton, manager, _ = stack_automaton
    manager.stage = None
    with pytest.raises(RuntimeError, match="acquisition manager stage"):
        automaton._validate_strategy_command_requirements()
    with pytest.raises(RuntimeError, match="acquisition manager stage"):
        automaton._execute_z_stack(make_factory().command_z_stack(make_metadata(), 2, 1))


def test_z_stack_failure_restores_z(stack_automaton, monkeypatch):
    """Route capture failures through command errors and restore the initial Z.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
        monkeypatch: Pytest fixture replacing frame capture with a failure.
    """
    automaton, manager, stage = stack_automaton
    stage.coordinate.z = 12
    original_capture = manager.take_frame
    calls = 0

    def fail_capture(**kwargs):
        """Capture keyword-configured frames, raising on the second plane."""
        nonlocal calls
        calls += 1
        if calls == 2:
            assert stage.coordinate.z == 13
            raise RuntimeError("capture failed")
        return original_capture(**kwargs)

    monkeypatch.setattr(manager, "take_frame", fail_capture)
    automaton.next_commands = [make_factory().command_z_stack(make_metadata(), 2, 1)]
    automaton._execute_strategy_batch()
    error, = automaton._strategy.received_errors
    assert isinstance(error, CommandExecutionError)
    assert error.command_type is AutomatonCommandType.Z_STACK
    assert str(error.original_error) == "capture failed"
    assert stage.coordinate.z == 12
    assert automaton._latest_frame is None


def test_z_stack_cancel_preserves_manager_behavior(stack_automaton, monkeypatch):
    """Forward the stop event and retain a partial stack without a return move.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
        monkeypatch: Pytest fixture wrapping frame capture to request cancellation.
    """
    automaton, manager, stage = stack_automaton
    original_capture = manager.take_frame
    calls = 0

    def capture_and_cancel(**kwargs):
        """Capture using acquisition keyword arguments and set stop after plane two."""
        nonlocal calls
        frame = original_capture(**kwargs)
        calls += 1
        if calls == 2:
            automaton._stop_event.set()
        return frame

    monkeypatch.setattr(manager, "take_frame", capture_and_cancel)
    data = automaton._execute_z_stack(make_factory().command_z_stack(make_metadata(), 2, 1))
    assert data["img"][0].shape[0] == 2
    assert stage.coordinate.z == 1
    assert automaton._stop_event.is_set()


def test_z_stack_requires_current_z(stack_automaton):
    """Fail before capture when hardware cannot provide the current Z coordinate.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
    """
    automaton, manager, stage = stack_automaton
    stage.coordinate.z = None
    with pytest.raises(RuntimeError, match="current stage Z"):
        automaton._execute_z_stack(make_factory().command_z_stack(make_metadata(), 2, 1))
    assert not manager.calls


@pytest.mark.parametrize("attribute", ["_camera", "_led_mngr", "_stage"])
def test_z_stack_missing_devices(stack_automaton, attribute):
    """Check registration rejects missing required Z-stack peripherals.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
        attribute: Required device attribute removed from the automaton.
    """
    automaton, _, _ = stack_automaton
    setattr(automaton, attribute, None)
    with pytest.raises(RuntimeError, match="Z_STACK"):
        automaton._validate_strategy_command_requirements()


def test_z_stack_default_save_and_enum_compatibility(stack_automaton):
    """Preserve existing command values and forward the default no-save setting.

    Args:
        stack_automaton: Automaton, acquisition manager and fake stage fixture.
    """
    automaton, manager, _ = stack_automaton
    assert AutomatonCommandType.SELECT_PROJECTION_ROI.value == 25
    assert AutomatonCommandType.Z_STACK.value == 26
    automaton._execute_z_stack(make_factory().command_z_stack(make_metadata(), 2, 1))
    assert all(not settings.save for _, settings in manager.calls)

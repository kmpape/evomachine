"""Translate parsed DSL commands into executable automaton commands."""

from collections.abc import Mapping, Sequence
from typing import Final

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.coordinates import Coordinate
from evomachine.dsl.errors import DSLInterpretationError
from evomachine.dsl.model import (
    CaptureImage,
    Move,
    MoveTarget,
    ParsedCommand,
    Wait,
)
from evomachine.frame import FrameMetaDataFactory
from evomachine.types import LEDType


NEXT_FOV_ID: Final[int] = -1


def interpret_commands(
        commands: Sequence[ParsedCommand],
        *,
        command_factory: CommandFactory,
        fovs: Mapping[int, Coordinate],
        imaging_channel: LEDType,
        imaging_brightness: int | float,
) -> list[AutomatonCommand]:
    """Map one lifecycle section to fresh automaton commands.

    This function constructs commands only; hardware execution remains owned by
    ``Automaton``. ``NEXT_FOV_ID`` is resolved by the navigation layer.
    """
    automaton_commands: list[AutomatonCommand] = []
    for command in commands:
        if isinstance(command, Move):
            if command.target is MoveTarget.FIRST_FOV:
                if not fovs:
                    raise DSLInterpretationError(
                        "Cannot interpret 'move first_fov' because no FoVs are registered."
                    )
                target_fov_id = next(iter(fovs))
            elif command.target is MoveTarget.NEXT_FOV:
                target_fov_id = NEXT_FOV_ID
            else:
                raise DSLInterpretationError(f"Unsupported move target: {command.target}.")
            automaton_commands.append(command_factory.command_move(fov_id=target_fov_id))
            continue
        if isinstance(command, CaptureImage):
            metadata = FrameMetaDataFactory.default(
                leds={command.led_type or imaging_channel: imaging_brightness},
                exposure=command.exposure_ms,
                fov_id=NEXT_FOV_ID,
            )
            automaton_commands.append(
                command_factory.command_image(
                    frame_metadata=metadata,
                    segment=False,
                    save=False,
                )
            )
            continue
        if isinstance(command, Wait):
            automaton_commands.append(
                command_factory.command_wait(
                    duration=command.duration_seconds,
                    set_live_mode=False,
                )
            )
            continue
        raise DSLInterpretationError(f"Unsupported parsed command: {type(command).__name__}.")
    return automaton_commands

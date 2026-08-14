"""Adapter exposing a parsed DSL program through ``AbstractStrategy``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evomachine.commands import AutomatonCommand
from evomachine.dsl import validator as dsl_validator
from evomachine.dsl.interpreter import interpret_commands
from evomachine.dsl.model import (
    CaptureImage,
    Move,
    ParsedCommand,
    ParsedStrategy,
    Wait,
)
from evomachine.dsl.parser import DSLParser
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evomachine.image_processing_config import ImageProcessorConfig


class DSLStrategy(AbstractStrategy):
    """Parse and execute DSL text through the standard strategy lifecycle."""

    def __init__(self, cfg: ImageProcessorConfig, dsl_text: str) -> None:
        super().__init__(cfg=cfg)
        self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_brightness: int = 10
        self.parsed_strategy: ParsedStrategy = DSLParser().parse(dsl_text=dsl_text)
        self.validated_strategy: ParsedStrategy = dsl_validator.validate_strategy(
            self.parsed_strategy
        )

    def _interpret(self, commands: Sequence[ParsedCommand]) -> list[AutomatonCommand]:
        """Interpret one lifecycle section using current strategy runtime state."""
        return interpret_commands(
            commands,
            command_factory=self.command_factory,
            fovs=self.fovs,
            imaging_channel=self.imaging_channel,
            imaging_brightness=self.imaging_brightness,
        )

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return command types that this parsed strategy can actually emit."""
        type_mapping = {
            Move: AutomatonCommandType.MOVE,
            CaptureImage: AutomatonCommandType.IMAGE,
            Wait: AutomatonCommandType.WAIT,
        }
        parsed_commands = (
            *self.validated_strategy.initialise,
            *self.validated_strategy.callback,
            *self.validated_strategy.finalise,
        )
        return {type_mapping[type(command)] for command in parsed_commands}

    def _initialise(self) -> list[AutomatonCommand]:
        """Interpret and return commands from the DSL initialise section."""
        return self._interpret(self.validated_strategy.initialise)

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Interpret and return commands from the DSL callback section."""
        return self._interpret(self.validated_strategy.callback)

    def finalise(self) -> list[AutomatonCommand]:
        """Interpret and return commands from the DSL finalise section."""
        return self._interpret(self.validated_strategy.finalise)

from __future__ import annotations

import numpy as np

from evomachine.commands import AutomatonCommand
from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.config_types import FrameMetaDataFactory, ImageProcessorConfig
from evomachine.strategy import AbstractStrategy
from evomachine.types import LEDType


class DmdProjectFullFovStrategy(AbstractStrategy):
    """
    Image FoVs and project a full-FoV DMD pattern.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    DmdProjectFullFovStrategy
        Strategy that schedules full-pattern DMD projections.
    """

    def __init__(self, cfg: ImageProcessorConfig):
        super().__init__(cfg=cfg)
        self.image_channel: LEDType = LEDType.LED_565_NM
        self.projection_channel: LEDType = LEDType.LED_385_NM
        self.exposure_ms: int = 100
        self.image_brightness: int = 10
        self.projection_brightness: int = 29
        self.projection_duration_s: float = 0.5
        self.wait_after_cycle_s: float = 30.0
        self.pattern: np.ndarray = np.ones(DMD_WIDTH_HEIGHT, dtype=np.uint8)

    def _cycle_commands(self, project: bool) -> list[AutomatonCommand]:
        """
        Build one full-FoV imaging/projection cycle.

        Parameters
        ----------
        project
            Whether to include projection commands.

        Returns
        -------
        list[AutomatonCommand]
            Commands for the cycle.
        """
        commands: list[AutomatonCommand] = []
        for fov_id in self.fovs:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
            metadata = FrameMetaDataFactory.default(
                leds={self.image_channel: self.image_brightness},
                exposure=self.exposure_ms,
                fov_id=fov_id,
            )
            commands.append(self.command_factory.command_image(frame_metadata=metadata, segment=False, save=True))
            if project:
                commands.append(
                    self.command_factory.command_project(
                        channel=self.projection_channel,
                        image=self.pattern,
                        duration=self.projection_duration_s,
                        brightness=self.projection_brightness,
                    )
                )
        commands.append(self.command_factory.command_wait(duration=self.wait_after_cycle_s))
        return commands

    def _initialise(self) -> list[AutomatonCommand]:
        """Return an initial imaging-only cycle."""
        return self._cycle_commands(project=False)

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Return repeated imaging plus projection commands."""
        return self._cycle_commands(project=True)

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []

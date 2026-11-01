"""
Timelapse: image every registered FoV once per period, for as long as the strategy runs.

Edit the constants below, then select ``TimelapseStrategy`` in the GUI. Register the FoVs
with *FoV Initialise* first; the strategy visits them in registration order. One round is
"move to each FoV and take one image"; the next round starts ``PERIOD_MIN`` minutes after
the previous round started, however long the round itself took. If a round takes longer
than the period, the next one starts immediately.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from evomachine.commands import AutomatonCommand
from evomachine.frame import FrameMetaDataFactory
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType

if TYPE_CHECKING:
    from evomachine.image_processing_config import ImageProcessorConfig

PERIOD_MIN: float = 5.0
"""Minutes between the starts of consecutive rounds."""

LED: LEDType = LEDType.LED_OVERHEAD_TIGER
"""Illumination for every image: the overhead light on the ASI Tiger LED output."""

BRIGHTNESS: int = 10
"""LED brightness in percent (0-100)."""

EXPOSURE_MS: int = 100
"""Camera exposure per image, in milliseconds."""


class TimelapseStrategy(AbstractStrategy):
    """
    Image every registered FoV once per ``PERIOD_MIN`` minutes.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    TimelapseStrategy
        Strategy that repeats move-and-image rounds on a fixed schedule.
    """

    def __init__(self, cfg: ImageProcessorConfig):
        super().__init__(cfg=cfg)
        self.period_s: float = PERIOD_MIN * 60.0
        self.led: LEDType = LED
        self.brightness: int = BRIGHTNESS
        self.exposure_ms: int = EXPOSURE_MS
        self._round_started_at: float | None = None

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return every command type this strategy may emit."""
        return {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.WAIT,
        }

    def _round_commands(self) -> list[AutomatonCommand]:
        """
        Build one round: visit and image every FoV, then wait out the rest of the period.

        Returns
        -------
        list[AutomatonCommand]
            Move and image commands per FoV followed by one wait command.
        """
        commands: list[AutomatonCommand] = []
        for fov_id in self.fovs:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
            commands.append(
                self.command_factory.command_image(
                    frame_metadata=[
                        FrameMetaDataFactory.default(
                            leds={self.led: self.brightness},
                            exposure=self.exposure_ms,
                            fov_id=fov_id,
                        )
                    ],
                    segment=False,
                    save=True,
                )
            )
        commands.append(
            self.command_factory.command_wait(
                duration=self._seconds_left_in_period(),
                set_live_mode=False,
            )
        )
        return commands

    def _seconds_left_in_period(self) -> float:
        """
        Return how long to wait so the next round starts one period after this one did.

        The wait is computed when the round's commands are built, which is before the
        moves and images run. The automaton executes those first, then the wait, so
        the wait is shortened by the time the previous round took in total.

        Returns
        -------
        float
            Non-negative wait in seconds.
        """
        now = time.monotonic()
        if self._round_started_at is None:
            self._round_started_at = now
            return self.period_s
        elapsed = now - self._round_started_at
        self._round_started_at = now
        return max(0.0, self.period_s - elapsed)

    def _initialise(self) -> list[AutomatonCommand]:
        """Return the first round."""
        self._round_started_at = None
        return self._round_commands()

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Return the next round, or re-raise the first error of the previous one."""
        if errors:
            raise errors[0]
        return self._round_commands()

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []

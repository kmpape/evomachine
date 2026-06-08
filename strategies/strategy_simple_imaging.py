from __future__ import annotations

from evomachine.commands import AutomatonCommand
from evomachine.config_types import FrameMetaDataFactory, ImageProcessorConfig
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType


class SimpleImagingStrategy(AbstractStrategy):
    """
    Image every configured FoV on one or more LED channels.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    SimpleImagingStrategy
        Strategy that repeatedly images all FoVs.
    """

    def __init__(self, cfg: ImageProcessorConfig):
        super().__init__(cfg=cfg)
        self.imaging_channels: list[LEDType] = [LEDType.LED_565_NM]
        self.initial_segment: bool = True
        self.exposure_ms: int = 100
        self.brightness: int = 10
        self.period_s: float = 180.0

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return every command type this strategy may emit."""
        return {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.WAIT,
        }

    def _imaging_commands(self, segment: bool) -> list[AutomatonCommand]:
        """
        Build one full-FoV imaging cycle.

        Parameters
        ----------
        segment
            Whether image commands should request segmentation.

        Returns
        -------
        list[AutomatonCommand]
            Move, image, and wait commands.
        """
        commands: list[AutomatonCommand] = []
        channels = self.imaging_channels
        if segment:
            channels = list(dict.fromkeys([*self.imaging_channels, *self.cfg.channels_seg]))
        for fov_id in self.fovs:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
            metadata = [
                FrameMetaDataFactory.default(
                    leds={channel: self.brightness},
                    exposure=self.exposure_ms,
                    fov_id=fov_id,
                )
                for channel in channels
            ]
            commands.append(self.command_factory.command_image(frame_metadata=metadata, segment=segment, save=True))
            commands.append(
                self.command_factory.command_wait(
                    duration=self.period_s / max(len(self.fovs), 1),
                    set_live_mode=False,
                )
            )
        return commands

    def _initialise(self) -> list[AutomatonCommand]:
        """Return the first imaging cycle."""
        return self._imaging_commands(segment=self.initial_segment)

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Return repeated imaging commands."""
        return self._imaging_commands(segment=False)

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []


class MCherryGfpImagingStrategy(SimpleImagingStrategy):
    """
    Image every FoV on mCherry/GFP-like channels.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    MCherryGfpImagingStrategy
        Multichannel imaging strategy.
    """

    def __init__(self, cfg: ImageProcessorConfig):
        super().__init__(cfg=cfg)
        self.imaging_channels = [LEDType.LED_565_NM, LEDType.LED_515_NM]

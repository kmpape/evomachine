from __future__ import annotations

from evomachine.commands import AutomatonCommand
from evomachine.config_types import FrameMetaDataFactory, ImageProcessorConfig
from evomachine.strategy import AbstractStrategy
from evomachine.types import LEDType


class DmdProjectByRoiStrategy(AbstractStrategy):
    """
    Image FoVs and project UV patterns onto detected ROI boxes.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    DmdProjectByRoiStrategy
        Strategy that combines ROI imaging and ROI-targeted projection.
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
        self.fill_x: float = 1.0
        self.fill_y: float = 1.0
        self.invert: bool = False

    def _roi_ids_for_fov(self, fov_id: int) -> list[int]:
        """
        Return ROI IDs for one FoV.

        Parameters
        ----------
        fov_id
            FoV ID whose ROI IDs should be used.

        Returns
        -------
        list[int]
            ROI IDs registered for that FoV.
        """
        return list(self.region_of_interests.get(fov_id, []))

    def _cycle_commands(self, segment: bool) -> list[AutomatonCommand]:
        """
        Build one ROI imaging/projection cycle.

        Parameters
        ----------
        segment
            Whether image commands should request segmentation.

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
            commands.append(self.command_factory.command_image(frame_metadata=metadata, segment=segment, save=True))
            roi_ids = self._roi_ids_for_fov(fov_id=fov_id)
            if roi_ids:
                commands.append(
                    self.command_factory.command_project_roi(
                        channel=self.projection_channel,
                        fov_id=fov_id,
                        roi_ids=roi_ids,
                        duration=self.projection_duration_s,
                        brightness=self.projection_brightness,
                        fill_x=self.fill_x,
                        fill_y=self.fill_y,
                        invert=self.invert,
                    )
                )
        commands.append(self.command_factory.command_wait(duration=self.wait_after_cycle_s))
        return commands

    def _initialise(self) -> list[AutomatonCommand]:
        """Return an initial segmentation cycle."""
        return self._cycle_commands(segment=True)

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Update ROI registrations from segmentation data and return another cycle."""
        for command in data:
            if isinstance(command.command_data, dict) and "seg" in command.command_data:
                self.command_factory.update_region_of_interests(region_of_interests=self.region_of_interests)
        return self._cycle_commands(segment=False)

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []

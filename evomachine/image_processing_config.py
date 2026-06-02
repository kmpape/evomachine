from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import delta
from delta.rttypes import TrackingSetting

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.types import ChamberOrientationType, LEDType


def _get_evomachine_dir() -> Path:
    from evomachine.config import EVOMACHINE_DIR

    return EVOMACHINE_DIR


@dataclass
class ImageProcessorConfig:
    cfg_delta: delta.config.Config
    channels: list[LEDType]
    channels_seg: list[LEDType]
    preproc_enabled: bool = True
    roi_enabled: bool = True
    roi_min_area: int | None = 1500
    roi_max_area: int | None = 7000
    roi_max_height: int | None = 40
    seg_enabled: bool = False
    track_enabled: bool = False
    lineage_enabled: bool = False
    tracking_setting: TrackingSetting = TrackingSetting.MOTHERONLY
    delta_roi_preprocess_target_size: tuple[int, int] = (3200, 3200)
    image_processing_verbosity: int = 0
    refocus: bool = True
    refocus_using_software_focus: bool = True
    refocus_on_all_fovs: bool = False
    max_refocus_trials: int = 10
    chamber_orientation: ChamberOrientationType = ChamberOrientationType.HORIZONTAL

    def copy(self) -> "ImageProcessorConfig":
        return ImageProcessorConfig(**self.__dict__)

    def updated(self, **kwargs) -> "ImageProcessorConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"ImageProcessorConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return ImageProcessorConfig(**values)

    def update_from_mapping(self, updates: dict) -> "ImageProcessorConfig":
        if not isinstance(updates, dict):
            raise TypeError("ImageProcessorConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)

    @property
    def channel_to_index(self) -> dict[LEDType, int]:
        return {channel: index for index, channel in enumerate(self.channels)}

    def __post_init__(self) -> None:
        if not isinstance(self.cfg_delta, delta.config.Config):
            raise TypeError("cfg_delta must be a delta.config.Config object.")
        if not (
                isinstance(self.channels, list)
                and all(isinstance(channel, LEDType) for channel in self.channels)
        ) or len(self.channels) == 0 or LEDType.NO_LED in self.channels:
            raise ConfigError("Invalid channel list.", ErrorCode.ERROR_CONFIG)
        if not (
                isinstance(self.channels_seg, list)
                and all(isinstance(channel, LEDType) for channel in self.channels_seg)
        ) or len(self.channels_seg) == 0 or LEDType.NO_LED in self.channels_seg:
            raise ConfigError("Invalid channels_seg list.", ErrorCode.ERROR_CONFIG)
        if not all(channel in self.channels for channel in self.channels_seg):
            raise ConfigError("All channels_seg must be contained in channels.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.tracking_setting, TrackingSetting):
            raise TypeError("Tracking setting must have type TrackingSetting.")
        if not isinstance(self.image_processing_verbosity, int) or self.image_processing_verbosity < 0:
            raise TypeError("image_processing_verbosity must be an integer >= 0.")
        if self.refocus and self.max_refocus_trials < 1:
            raise TypeError(f"Refocus is True but max_refocus_trials={self.max_refocus_trials}")
        for field_name in ("preproc_enabled", "roi_enabled", "seg_enabled", "track_enabled", "lineage_enabled"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")
        if self.seg_enabled and not self.preproc_enabled:
            raise TypeError("preproc_enabled must be true if seg_enabled is true.")
        if self.track_enabled and ((not self.preproc_enabled) or (not self.seg_enabled)):
            raise TypeError("preproc_enabled and seg_enabled must be true if track_enabled is true.")
        if self.lineage_enabled and ((not self.preproc_enabled) or (not self.seg_enabled) or (not self.track_enabled)):
            raise TypeError("preproc_enabled, seg_enabled, and track_enabled must be true if lineage_enabled is true.")

    def __str__(self) -> str:
        lines = ["ImageProcessorConfig"]
        for index, (key, value) in enumerate(self.__dict__.items()):
            if key == "cfg_delta":
                value = str(value).replace("\n", "\n\t")
            lines.append(f"{' └─ ' if index == len(self.__dict__) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)


class ImageProcessorConfigFactory:
    @staticmethod
    def default_config(
            channels: list[LEDType] | None = None,
            channels_seg: list[LEDType] | None = None,
    ) -> ImageProcessorConfig:
        evomachine_dir = _get_evomachine_dir()
        default_channels = [LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM, LEDType.LED_645_NM]
        cfg_delta = delta.config.Config.default("mothermachine")
        cfg_delta.whole_frame_drift = False
        cfg_delta.target_size_rois = (200, 800)
        cfg_delta.tolerable_resizing_rois = 0
        cfg_delta.model_file_rois = evomachine_dir.parent / "delta_models/evo_roi_mixed_200x800_CKS5_2025-02-11.keras"
        cfg_delta.target_size_seg = (64, 512)
        cfg_delta.model_file_seg = evomachine_dir.parent / "delta_models/evo_seg_64x512_kernel5_levels5_2025-05-29_realdata.keras"
        cfg_delta.target_size_seg = (64, 256)
        cfg_delta.model_file_track = evomachine_dir.parent / "delta_models/evo_track_64x256_2024-08-30.keras"
        return ImageProcessorConfig(
            cfg_delta=cfg_delta,
            channels=default_channels if channels is None else channels,
            channels_seg=default_channels if channels_seg is None else channels_seg,
            roi_enabled=False,
            seg_enabled=False,
            preproc_enabled=False,
        )

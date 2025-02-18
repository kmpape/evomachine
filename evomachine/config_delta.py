import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
import delta
from evomachine.config import *

@dataclass
class ConfigImageProcessor:
    cfg_delta: delta.config.Config
    "Delta configuration object. Contains paths to RoI ID, segmentation, and tracking models."
    channels: list[LEDType]
    "List of channels to be imaged. Used for taking reference frames. Do not include any UV here. The list must " \
    "include channels_seg."  # noqa
    channels_seg: list[LEDType]
    "Channel(s) used for segmentation. If a list is provided, channels are averaged."
    preproc_enabled: bool = True
    "Enable image preprocessing."
    roi_enabled: bool = True
    "Enable ROI identification (preproc_enabled must be true)."
    roi_min_area: int | None = 1500
    "Min. area of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    roi_max_area: int | None = 7000
    "Max. area of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    roi_max_height: int | None = 40
    "Max. height of RoIs to be considered in pixels (original image size). Operation is not applied if None."
    seg_enabled: bool = False
    "Enable image segmentation (preproc_enabled must be true)."
    track_enabled: bool = False
    "Enable tracking (preproc_enabled and seg_enabled must be true)."
    lineage_enabled: bool = False
    "Enable lineage computations(preproc_enabled, seg_enabled, and track_enabled must be true)."
    use_track_RT: bool = False
    "Use special tracking function for tracking in trenches."
    delta_roi_preprocess_target_size: tuple[int, int] = (3200, 3200)
    "Size of microscope images just before being input to DeLTA. This can be different from cfg_delta.target_size_rois."
    image_processing_verbosity: int = 0
    "Lowest verbosity is 0."
    refocus: bool = True
    "Refocus after autofocus loss (otherwise shuts down). Set refocus_using_software_focus=False to avoid using autofocus."
    refocus_using_software_focus: bool = True
    "Use software focus to refocus after autofocus loss. refocus must be True."
    refocus_on_all_positions: bool = False
    "Try refocusing on all recorded positions. refocus and refocus_using_software_focus must be True."
    max_refocus_trials: int = 10
    "Maximum number of refocusing trials before stopping execution."
    chamber_orientation: ChamberOrientationType = ChamberOrientationType.HORIZONTAL
    "Orientation of chambers."

    def copy(self):
        return ConfigImageProcessor(**self.__dict__)

    @property
    def channel_to_index(self) -> dict[LEDType, int]:
        return {c: i for i, c in enumerate(self.channels)}

    def __post_init__(self):
        if not isinstance(self.cfg_delta, delta.config.Config):
            raise TypeError("cfg_delta must be a delta.config.Config object.")
        if not (isinstance(self.channels, list) and all(isinstance(channel, LEDType) for channel in self.channels))\
                or len(self.channels) == 0 or LEDType.NO_LED in self.channels:
            raise ConfigError("Invalid channel list.", ErrorCode.ERROR_CONFIG)
        if not (isinstance(self.channels_seg, list) and all(isinstance(ch, LEDType) for ch in self.channels_seg))\
                or len(self.channels_seg) == 0 or LEDType.NO_LED in self.channels_seg:
            raise ConfigError("Invalid channels_seg list.", ErrorCode.ERROR_CONFIG)
        if not all([ch_seg in self.channels for ch_seg in self.channels_seg]):
            raise ConfigError("All channels_seg must be contained in channels.", ErrorCode.ERROR_CONFIG)
        if not isinstance(self.use_track_RT, bool):
            raise TypeError("use_track_RT must be a boolean.")
        if not isinstance(self.image_processing_verbosity, int) or self.image_processing_verbosity < 0:
            raise TypeError("image_processing_verbosity must be an integer >= 0.")
        if self.refocus and self.max_refocus_trials < 1:
            raise TypeError(f"Refocus is True but max_refocus_trials={self.max_refocus_trials}")
        if not isinstance(self.preproc_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.roi_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.seg_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.track_enabled, bool):
            raise TypeError("preproc_enabled must be a boolean.")
        if not isinstance(self.lineage_enabled, bool):
            raise TypeError("lineage_enabled must be a boolean.")
        if self.seg_enabled and not self.preproc_enabled:
            raise TypeError("preproc_enabled must be true if seg_enabled is true.")
        if self.track_enabled and ((not self.preproc_enabled) or (not self.seg_enabled)):
            print(f"self.track_enabled = {self.track_enabled} self.preproc_enabled = {self.preproc_enabled} self.seg_enabled = {self.seg_enabled}")
            raise TypeError("preproc_enabled and seg_enabled must be true if track_enabled is true.")
        if self.lineage_enabled and ((not self.preproc_enabled) or (not self.seg_enabled) or (not self.track_enabled)):
            raise TypeError("preproc_enabled and seg_enabled must be true if track_enabled is true.")

    def __str__(self):
        s = ["ConfigImageProcessor"]
        for i, (k, v) in enumerate(self.__dict__.items()):
            if k == 'cfg_delta':
                tmp = str(v)
                v = tmp.replace("\n", "\n\t")
            if i < len(self.__dict__)-1:
                s.append(f" ├─ {k}: {v}")
            else:
                s.append(f" └─ {k}: {v}")
        return "\n".join(s)


class ConfigImageProcessorFactory:
    @staticmethod
    def default_config(
            channels: list[LEDType] | None = None,
            channels_seg: list[LEDType] | None = None
    ) -> ConfigImageProcessor:
        default_channels = [LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM, LEDType.LED_645_NM]
        cfg_delta = delta.config.Config.default("mothermachine")
        cfg_delta.whole_frame_drift = False
        # cfg_delta.target_size_rois = (1024, 1024)
        # cfg_delta.target_size_rois = (1600, 1600)
        cfg_delta.target_size_rois = (200, 800)
        cfg_delta.tolerable_resizing_rois = 0
        # cfg_delta.model_file_rois = EVOMACHINE_DIR.parent / "delta_models/evo_roi_2024-05-08.keras"
        # cfg_delta.model_file_rois = EVOMACHINE_DIR.parent / "delta_models/evo_roi_M9_2024-12-10.keras"
        cfg_delta.model_file_rois = EVOMACHINE_DIR.parent / "delta_models/evo_roi_mixed_200x800_CKS5_2025-02-11.keras"
        cfg_delta.target_size_seg = (250, 64)
        cfg_delta.model_file_seg = EVOMACHINE_DIR.parent / "delta_models/evo_seg_2024-06-27.keras"
        # cfg_delta.model_file_track = EVOMACHINE_DIR.parent / "delta_models/unet_moma_track.hdf5"
        return ConfigImageProcessor(
            cfg_delta=cfg_delta,
            channels=default_channels if channels is None else channels,
            channels_seg=default_channels if channels_seg is None else channels_seg,
            roi_enabled=False,
            seg_enabled=False,
            preproc_enabled=False,
        )


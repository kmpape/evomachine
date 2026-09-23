"""Synchronous DeLTA processing and experiment-owned, per-trench measurement state.

The backend detects trenches explicitly. It never chooses a treatment threshold or
an imaging interval; those decisions belong to the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import copy
from typing import Any, Callable

import numpy as np

from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.types import ChamberOrientationType


class DeltaProcessingError(RuntimeError):
    """Processing failed; a partially updated lineage must not be retried in place."""


class TargetedProjectionError(RuntimeError):
    """Targeted exposure failed or its completion is uncertain; do not auto-retry."""


@dataclass(frozen=True)
class MotherMeasurement:
    frame_id: int
    time: float
    length: float | None = None
    growth_rate: float | None = None

    @property
    def valid(self) -> bool:
        return self.length is not None


@dataclass
class TrenchState:
    selected: bool = True
    measurement: MotherMeasurement | None = None
    previous: MotherMeasurement | None = None
    treatment_count: int = 0
    last_treatment_time: float | None = None
    comparison_count: int = 0
    length_increase_count: int = 0

    def update(self, measurement: MotherMeasurement) -> None:
        previous = self.measurement
        self.previous = previous
        self.measurement = measurement
        if (
            self.last_treatment_time is not None
            and previous is not None
            and previous.time > self.last_treatment_time
            and previous.valid
            and measurement.valid
        ):
            self.comparison_count += 1
            self.length_increase_count += int(measurement.length > previous.length)

    def treated(self, when: float) -> None:
        self.treatment_count += 1
        self.last_treatment_time = when
        self.comparison_count = 0
        self.length_increase_count = 0


@dataclass
class FovMeasurements:
    rois: dict[int, TrenchState] = field(default_factory=dict)
    valid: bool = False
    measurement_time: float | None = None
    projection_targets: set[int] = field(default_factory=set)


@dataclass
class MicroscopyState:
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    fovs: dict[int, FovMeasurements] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def invalidate(self, fov_id: int) -> None:
        fov = self.fovs.get(fov_id)
        if fov is not None:
            fov.valid = False
            fov.projection_targets.clear()


class DeltaProcessor:
    """Lazy model loading, injectable backend for tests, stable mother-only trenches.

    Only one segmentation-channel plane is accepted initially. Automatic rotation
    and whole-frame drift are disabled: ROI boxes must stay in camera coordinates
    for the existing camera-to-DMD transform. The acquisition must stay registered.
    """

    def __init__(
        self,
        cfg: ImageProcessorConfig,
        *,
        position_factory=None,
        model_loader=None,
        selected_targets: dict[int, set[int]] | None = None,
    ):
        self.cfg = cfg
        if selected_targets is not None and any(
            type(fov_id) is not int or fov_id < 0 or any(type(i) is not int or i < 0 for i in ids)
            for fov_id, ids in selected_targets.items()
        ):
            raise ValueError(
                "Treatment selections must contain non-negative integer FOV/trench IDs"
            )
        self._position_factory = position_factory
        self._model_loader = model_loader
        self._seg_model: Any = None
        self._roi_model: Any = None
        self.selected_targets = (
            None
            if selected_targets is None
            else {fov_id: frozenset(ids) for fov_id, ids in selected_targets.items()}
        )
        self._failed: set[int] = set()

    def reset(self) -> None:
        self._failed.clear()

    def process(
        self,
        *,
        fov_id: int,
        image: np.ndarray,
        acquired_at: float,
        processors: dict,
        roi_ids: dict[int, list[int]],
        state: MicroscopyState,
        roi_boxes=None,
        detect_rois: bool = False,
        segment: bool = True,
    ) -> dict[int, np.ndarray]:
        if not detect_rois and not segment:
            state.invalidate(fov_id)
            return {}
        state.invalidate(fov_id)
        if fov_id in self._failed:
            raise DeltaProcessingError("DeLTA state is invalid; start a new experiment")
        try:
            redetection = detect_rois and fov_id in processors
            if detect_rois:
                processors.pop(fov_id, None)
                roi_ids[fov_id] = []
                state.fovs[fov_id] = FovMeasurements()
            elif fov_id not in processors:
                raise ValueError("Segmentation requires existing ROIs; use detect_rois=true first")
            if image.ndim != 3 or image.shape[0] != 1:
                raise ValueError("Processed imaging requires one segmentation-channel plane")
            if self.cfg.chamber_orientation is not ChamberOrientationType.HORIZONTAL:
                raise ValueError("This integration requires horizontal mother-machine trenches")
            from delta.rt import PositionRT
            from delta.rttypes import TrackingSetting

            config = copy.deepcopy(self.cfg.cfg_delta)
            config.rotation_correction = False
            config.whole_frame_drift = False
            config.drift_correction = False
            loader = self._model_loader or config.model
            if segment and self._seg_model is None:
                self._seg_model = loader("seg")
            if fov_id not in processors:
                if roi_boxes is None and self._roi_model is None:
                    self._roi_model = loader("rois")
                factory = self._position_factory or PositionRT
                position = factory(
                    position_nb=fov_id,
                    config=config,
                    tracking_setting=TrackingSetting.MOTHERONLY,
                    roi_input_size=self.cfg.delta_roi_preprocess_target_size,
                )
                position.initialise(
                    reference=image,
                    seg_model=self._seg_model if segment else None,
                    tracking_model=None,
                    roi_model=self._roi_model,
                    roi_boxes=roi_boxes,
                    lineage_enabled=segment,
                    roi_min_area=self.cfg.roi_min_area,
                    roi_max_area=self.cfg.roi_max_area,
                    roi_max_height=self.cfg.roi_max_height,
                )
                if not segment:
                    # DeLTA allocates these stacks with np.empty. Without initial
                    # segmentation, garbage labels can look like existing cells
                    # when tracking starts on a later acquisition.
                    for roi in position.rois:
                        for mask in (*roi.seg_stack, *roi.label_stack):
                            mask.fill(0)
                processors[fov_id] = position
                ids = list(range(len(position.rois)))
                if self.selected_targets is not None and not redetection:
                    unknown = self.selected_targets.get(fov_id, frozenset()) - set(ids)
                    if unknown:
                        raise ValueError(
                            f"Supplied target IDs were not detected: {sorted(unknown)}"
                        )
                roi_ids[fov_id] = ids
                state.fovs.setdefault(fov_id, FovMeasurements()).rois = {
                    roi_id: TrenchState(
                        selected=self.selected_targets is None
                        or (not redetection and roi_id in self.selected_targets.get(fov_id, set()))
                    )
                    for roi_id in ids
                }
            else:
                position = processors[fov_id]
                if position.rois:
                    position.process_new_frame(
                        new_frame=image,
                        channel_inds=[0],
                        seg_model=self._seg_model,
                        tracking_model=None,
                        lineage_enabled=True,
                    )
            fov = state.fovs[fov_id]
            if not segment:
                fov.valid = True
                return {}
            frame_id = position.get_frame_id()
            for roi_id, trench in fov.rois.items():
                cell = position.rois[roi_id].lineage.cells.get(1)
                length = growth = None
                if cell is not None and frame_id in cell.frames:
                    features = cell.features(frame_id)
                    if math.isfinite(features.length) and features.length > 0:
                        length = float(features.length)
                        if math.isfinite(features.growthrate_length):
                            growth = float(features.growthrate_length)
                trench.update(MotherMeasurement(frame_id, acquired_at, length, growth))
            masks = position.get_seg() if position.rois else {}
            fov.measurement_time = acquired_at
            fov.valid = True
            return masks
        except Exception as error:
            self._failed.add(fov_id)
            state.invalidate(fov_id)
            raise DeltaProcessingError(
                f"DeLTA processing failed for FOV {fov_id}: {error}"
            ) from error

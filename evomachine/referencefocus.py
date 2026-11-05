"""Acquisition-backed reference focus; calibration is accepted after plot review."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import threading
from typing import Any

import numpy as np

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.coordinates import Coordinate
from evomachine.focus_registration import (
    FocusShiftResult,
    ReferenceProfile,
    ShiftSearchConfig,
    estimate_focus_shift,
    fit_reference_profile,
    score_focus_stack,
)
from evomachine.frame import Frame, FrameMetaData
from evomachine.types import FocusAlgorithmType, FocusStatusType


class ReferenceFocusCancelled(RuntimeError):
    """A stopped scan must not produce a focus correction."""


class ReferenceFocusOutOfRange(ValueError):
    """A complete scan would exceed the stage's coordinate bounds."""


@dataclass(frozen=True)
class ReferenceProfileFocusConfig:
    """Whole-image acquisition and scan settings; all Z distances are in micrometres."""

    focus_frame: FrameMetaData
    calibration_range: float = 100.0
    calibration_step: float = 5.0
    sample_offsets: tuple[float, ...] = tuple(range(-40, 41, 10))
    recovery_offsets: tuple[float, ...] = tuple(range(-50, 51, 5))
    search: ShiftSearchConfig = field(default_factory=ShiftSearchConfig)
    recovery_search: ShiftSearchConfig = field(
        default_factory=lambda: ShiftSearchConfig(shift_min=-50, shift_max=50)
    )
    algorithm: FocusAlgorithmType = FocusAlgorithmType.SQUARED_GRAD_AVG
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    smoothing: float | None = None
    acquisition_settings: FrameAcquisitionSettings = field(default_factory=FrameAcquisitionSettings)

    def __post_init__(self) -> None:
        if not np.isfinite(self.calibration_range) or not np.isfinite(self.calibration_step):
            raise ValueError("Calibration range and step must be finite.")
        if not 0 < self.calibration_step <= self.calibration_range / 2:
            raise ValueError("Calibration requires a positive range and at least five planes.")
        for offsets in (self.sample_offsets, self.recovery_offsets):
            values = np.asarray(offsets, dtype=float)
            if (
                values.ndim != 1
                or values.size < 5
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError("Scan offsets must contain at least five increasing finite Zs.")


@dataclass
class ReferenceFocusScan:
    """Captured whole images and their scores for one scan."""

    z_coordinates: np.ndarray
    focus_scores: np.ndarray
    frame: Frame


@dataclass
class ReferenceProfileFocusResult:
    scans: tuple[ReferenceFocusScan, ...]
    matches: tuple[FocusShiftResult, ...]
    focus_status: FocusStatusType
    failure_reason: str | None = None

    @property
    def recovery_attempted(self) -> bool:
        return bool(self.matches and not self.matches[0].accepted)


class ReferenceProfileFocus:
    """Capture, inspect and explicitly accept a chip reference, then track its Z.

    ``calibrate()`` returns a candidate for ``plot_reference_profile``. Call
    ``accept_reference()`` after inspecting the plot, before running autofocus.
    Changing configuration clears calibration. One frame is acquired per Z;
    all scoring uses the whole image. Acquisition settings can enable saving.
    """

    def __init__(
        self, acquisition_manager: FrameAcquisitionManager, config: ReferenceProfileFocusConfig
    ):
        if acquisition_manager.stage is None:
            raise ValueError("Reference-profile focus requires a stage.")
        self.acquisition_manager = acquisition_manager
        self._stop_event = threading.Event()
        self.update_config(config)

    @property
    def config(self) -> ReferenceProfileFocusConfig:
        """Return a snapshot; use update_config so changes invalidate calibration."""
        return deepcopy(self._config)

    def update_config(self, config: ReferenceProfileFocusConfig) -> None:
        self._config = deepcopy(config)
        self.clear_reference()

    def clear_reference(self) -> None:
        self.reference: ReferenceProfile | None = None
        self.pending_reference: ReferenceProfile | None = None
        self.calibration_scan: ReferenceFocusScan | None = None
        self.last_result: ReferenceProfileFocusResult | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def _check_cancelled(self) -> None:
        if self._stop_event.is_set():
            raise ReferenceFocusCancelled("Reference-profile focus cancelled.")

    def _begin(self, stop_event: threading.Event | None) -> float:
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        self._check_cancelled()
        z = self.acquisition_manager.stage.get_coordinates(query_hardware=True).z
        if z is None or not np.isfinite(z):
            raise RuntimeError("Reference-profile focus requires a finite stage Z.")
        return float(z)

    def calibrate(self, stop_event: threading.Event | None = None) -> ReferenceProfile:
        centre = self._begin(stop_event)
        self.clear_reference()
        config = self._config
        positive = np.r_[
            np.arange(0, config.calibration_range, config.calibration_step),
            config.calibration_range,
        ]
        offsets = np.r_[-positive[:0:-1], positive]
        self.calibration_scan = self._scan(
            centre,
            offsets,
            self.acquisition_manager.stage.get_fov_id(),
        )
        self.pending_reference = fit_reference_profile(
            self.calibration_scan.z_coordinates,
            self.calibration_scan.focus_scores,
            focus_z=centre,
            smoothing=config.smoothing,
        )
        self._check_cancelled()
        return self.pending_reference

    def accept_reference(self) -> ReferenceProfile:
        if self.pending_reference is None:
            raise RuntimeError("Capture and inspect a reference before accepting it.")
        self._check_cancelled()
        self.reference = self.pending_reference
        self.pending_reference = None
        return self.reference

    def _scan(self, centre: float, offsets, fov_id: int) -> ReferenceFocusScan:
        self._check_cancelled()
        z = centre + np.asarray(offsets, dtype=float)
        targets = [Coordinate(None, None, float(value)) for value in z]
        stage = self.acquisition_manager.stage
        if any(stage.coordinate_is_out_of_bounds(target) for target in targets):
            raise ReferenceFocusOutOfRange("Reference focus scan exceeds the stage Z bounds.")
        config = self._config
        try:
            frame = self.acquisition_manager.take_z_stack(
                frame_metadata=replace(config.focus_frame, fov_id=fov_id),
                z_coordinates=targets,
                settings=config.acquisition_settings,
                stop_event=self._stop_event,
            )
        except RuntimeError:
            self._check_cancelled()
            raise
        self._check_cancelled()
        if frame.array.shape[0] != z.size:
            raise RuntimeError("Reference focus scan returned an incomplete image stack.")
        scores = score_focus_stack(
            frame.array, algorithm=config.algorithm, algorithm_kwargs=config.algorithm_kwargs
        )
        return ReferenceFocusScan(z, scores, frame)

    def run(
        self, fov_id: int, stop_event: threading.Event | None = None
    ) -> ReferenceProfileFocusResult:
        if self.reference is None:
            raise RuntimeError("Reference calibration must be accepted before focusing.")
        centre = self._begin(stop_event)
        self.last_result = None
        scans, matches = [], []
        config = self._config
        for offsets, search in (
            (config.sample_offsets, config.search),
            (config.recovery_offsets, config.recovery_search),
        ):
            try:
                scan = self._scan(centre, offsets, fov_id)
            except ReferenceFocusOutOfRange as error:
                self.last_result = ReferenceProfileFocusResult(
                    tuple(scans),
                    tuple(matches),
                    FocusStatusType.OUT_OF_RANGE,
                    str(error),
                )
                return self.last_result
            scans.append(scan)
            match = estimate_focus_shift(
                self.reference,
                scan.z_coordinates,
                scan.focus_scores,
                current_z=centre,
                config=search,
            )
            matches.append(match)
            self._check_cancelled()
            if match.accepted:
                stage = self.acquisition_manager.stage
                target = Coordinate(None, None, match.focus_z)
                if stage.coordinate_is_out_of_bounds(target):
                    self.last_result = ReferenceProfileFocusResult(
                        tuple(scans),
                        tuple(matches),
                        FocusStatusType.OUT_OF_RANGE,
                        "Estimated focal Z exceeds the stage bounds.",
                    )
                    return self.last_result
                stage.move(target=target, block=True)
                self._check_cancelled()
                break
        accepted = matches[-1].accepted
        self.last_result = ReferenceProfileFocusResult(
            tuple(scans),
            tuple(matches),
            FocusStatusType.IN_FOCUS if accepted else FocusStatusType.BAD_FOCUS_CURVE,
            None if accepted else "; ".join(matches[-1].rejection_reasons),
        )
        return self.last_result

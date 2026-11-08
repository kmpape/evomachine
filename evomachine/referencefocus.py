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
from evomachine.bindings.software_focus.software_focus_algorithms import (
    create_software_focus_algorithm,
)


class ReferenceFocusCancelled(RuntimeError):
    """A stopped scan must not produce a focus correction."""


class ReferenceFocusOutOfRange(ValueError):
    """A complete scan would exceed the stage's coordinate bounds."""


@dataclass(frozen=True)
class SamplingPlan:
    """Symmetric image positions in µm, optionally denser around the scan centre.

    Outer and central grids are merged without duplicates. Their steps control
    image acquisition; the separate ShiftSearchConfig step controls only the
    numerical Pearson search and does not acquire additional images.
    """

    half_range: float = 20.0
    step: float = 5.0
    central_range: float | None = None
    central_step: float = 0.5

    def offsets(self) -> tuple[float, ...]:
        """Return increasing offsets including zero and both range endpoints."""
        values = (self.half_range, self.step, self.central_step)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Sampling range and steps must be finite and greater than zero.")
        if self.central_range is not None and not 0 < self.central_range <= self.half_range:
            raise ValueError("The denser centre must fit inside the outer sampling range.")
        if self.central_range is not None and self.central_step >= self.step:
            raise ValueError("The central image step must be smaller than the outer image step.")
        if self.half_range / self.step > 2500 or (
            self.central_range is not None and self.central_range / self.central_step > 2500
        ):
            raise ValueError("Sampling would acquire too many images; increase the Z step.")
        positive = np.r_[np.arange(0, self.half_range, self.step), self.half_range]
        if self.central_range is not None:
            positive = np.r_[
                positive, np.arange(0, self.central_range, self.central_step), self.central_range
            ]
        offsets = np.unique(np.round(np.r_[-positive, positive], 9))
        offsets[offsets == 0] = 0.0
        if offsets.size < 5:
            raise ValueError("Use a smaller image step to acquire at least five Z positions.")
        return tuple(float(value) for value in offsets)


@dataclass(frozen=True)
class ReferenceProfileFocusConfig:
    """Whole-image acquisition and scan settings; all Z distances are in micrometres."""

    focus_frame: FrameMetaData
    calibration_range: float = 40.0
    calibration_step: float = 1.0
    sample_offsets: tuple[float, ...] = SamplingPlan().offsets()
    recovery_offsets: tuple[float, ...] = SamplingPlan(central_range=5).offsets()
    search: ShiftSearchConfig = field(
        default_factory=lambda: ShiftSearchConfig(shift_min=-20, shift_max=20, shift_step=0.1)
    )
    recovery_search: ShiftSearchConfig = field(
        default_factory=lambda: ShiftSearchConfig(shift_min=-20, shift_max=20, shift_step=0.1)
    )
    algorithm: FocusAlgorithmType = FocusAlgorithmType.SQUARED_GRAD_AVG
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    smoothing: float | None = 0.0
    acquisition_settings: FrameAcquisitionSettings = field(default_factory=FrameAcquisitionSettings)
    allow_peak_reset: bool = False

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
    centre_z: float


@dataclass
class ReferenceProfileFocusResult:
    scans: tuple[ReferenceFocusScan, ...]
    matches: tuple[FocusShiftResult, ...]
    focus_status: FocusStatusType
    failure_reason: str | None = None
    reference_reset: bool = False
    correction_applied: bool = False

    @property
    def recovery_attempted(self) -> bool:
        return bool(self.matches and not self.matches[0].accepted)


class ReferenceProfileFocus:
    """Capture, inspect and explicitly accept a chip reference, then track its Z.

    ``calibrate()`` returns a candidate for ``plot_reference_profile``. Call
    ``accept_reference()`` after inspecting the plot, before running autofocus.
    Scan/search edits retain calibration. One frame is acquired per Z;
    all scoring uses the whole image. Acquisition settings can enable saving.
    """

    def __init__(
        self, acquisition_manager: FrameAcquisitionManager, config: ReferenceProfileFocusConfig
    ):
        if acquisition_manager.stage is None:
            raise ValueError("Reference-profile focus requires a stage.")
        self.acquisition_manager = acquisition_manager
        self._stop_event = threading.Event()
        self.process_events = None
        self.update_config(config)

    @property
    def config(self) -> ReferenceProfileFocusConfig:
        """Return a snapshot; use update_config to preserve compatible calibration."""
        return deepcopy(self._config)

    def update_config(self, config: ReferenceProfileFocusConfig) -> None:
        """Keep references compatible with the changed settings.

        Scan/search, saving and future calibration range/step edits retain the
        accepted profile. Smoothing/metric changes refit cached images into a
        candidate requiring acceptance. Imaging changes retain the old curve
        for inspection but require a new capture before it can be accepted.
        Compare resolved scorer parameters so omitted defaults do not cause
        a refit when configuration is round-tripped through the GUI.
        """
        previous = getattr(self, "_config", None)
        if previous is not None:
            old_scorer = create_software_focus_algorithm(
                previous.algorithm, **previous.algorithm_kwargs
            )
            new_scorer = create_software_focus_algorithm(
                config.algorithm, **config.algorithm_kwargs
            )
            scoring_changed = type(old_scorer) is not type(new_scorer) or vars(old_scorer) != vars(
                new_scorer
            )
            capture_changed = any(
                getattr(previous.focus_frame, key) != getattr(config.focus_frame, key)
                for key in ("exposure", "filter_wheel", "leds")
            ) or not np.array_equal(
                previous.focus_frame.dmd_pattern, config.focus_frame.dmd_pattern
            )
            capture_changed |= (
                previous.acquisition_settings.normalise != config.acquisition_settings.normalise
            )
            profile = self.pending_reference or self.reference
            if (
                profile is not None
                and not capture_changed
                and not self.needs_calibration
                and (scoring_changed or previous.smoothing != config.smoothing)
            ):
                scores = score_focus_stack(
                    self.calibration_scan.frame.array,
                    algorithm=config.algorithm,
                    algorithm_kwargs=config.algorithm_kwargs,
                )
                fitted = fit_reference_profile(
                    self.calibration_scan.z_coordinates,
                    scores,
                    focus_z=profile.focus_z,
                    smoothing=config.smoothing,
                )
                self.calibration_scan.focus_scores = scores
                self.pending_reference, self.reference = fitted, None
                self._stop_event = threading.Event()
                self.last_result = None
                self.reference_message = (
                    "Reference refitted from cached images. Inspect and accept it again."
                )
            elif profile is not None and capture_changed:
                self.pending_reference, self.reference = profile, None
                self.needs_calibration = True
                self.reference_message = "Imaging settings changed. Curve retained; capture a new reference before focusing."
            elif profile is not None and not self.needs_calibration:
                self.reference_message = (
                    "Reference retained. Updated settings apply to the next scan."
                )
        self._config = deepcopy(config)
        if previous is None:
            self.clear_reference()

    def clear_reference(self) -> None:
        self.reference: ReferenceProfile | None = None
        self.pending_reference: ReferenceProfile | None = None
        self.calibration_scan: ReferenceFocusScan | None = None
        self.last_result: ReferenceProfileFocusResult | None = None
        self.needs_calibration = False
        self.reference_message: str | None = None

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
        """Capture a candidate around current Z, which defines the manual zero.

        Zero need not be the score maximum. A completed scan restores its
        starting Z; cancellation leaves the stopped position. The caller must
        inspect the returned curve and call accept_reference before focusing.
        """
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
        if self.needs_calibration:
            raise RuntimeError(
                "Imaging settings changed. Capture a new reference before accepting it."
            )
        if self.pending_reference is None:
            raise RuntimeError("Capture and inspect a reference before accepting it.")
        self._check_cancelled()
        self.reference = self.pending_reference
        self.pending_reference = None
        self.reference_message = None
        return self.reference

    def refit_reference(self, smoothing: float | None) -> ReferenceProfile:
        if self.needs_calibration:
            raise RuntimeError(
                "Capture a new reference with the current imaging settings before refitting."
            )
        if self.calibration_scan is None:
            raise RuntimeError("Capture a reference before refitting its spline.")
        previous = self.pending_reference or self.reference
        profile = fit_reference_profile(
            self.calibration_scan.z_coordinates,
            self.calibration_scan.focus_scores,
            focus_z=previous.focus_z,
            smoothing=smoothing,
        )
        self._config = replace(self._config, smoothing=smoothing)
        self._stop_event = threading.Event()
        self.last_result = None
        self.pending_reference = profile
        self.reference = None
        self.reference_message = "Spline refitted. Inspect and accept the reference again."
        return profile

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
                step_callback=self.process_events,
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
        return ReferenceFocusScan(z, scores, frame, centre)

    def run(
        self, fov_id: int, stop_event: threading.Event | None = None, *, apply: bool = True
    ) -> ReferenceProfileFocusResult:
        """Try sparse then recovery sampling, correcting Z only for an accepted match.

        apply=False still moves through the sampling positions and acquires
        images, without applying a correction or rebuilding the reference.
        Completed scans restore starting Z; cancellation leaves the stopped
        position. Results retain scans and rejected matches for diagnostics.
        Peak resetting requires both apply and explicit opt-in.
        """
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
                if apply:
                    stage.move(target=target, block=True)
                self._check_cancelled()
                break
        accepted = matches[-1].accepted
        if not accepted and config.allow_peak_reset and apply:
            reset = self._reset_at_peak(centre, fov_id, scans, matches)
            if reset is not None:
                self.last_result = reset
                return reset
        self.last_result = ReferenceProfileFocusResult(
            tuple(scans),
            tuple(matches),
            FocusStatusType.IN_FOCUS if accepted else FocusStatusType.BAD_FOCUS_CURVE,
            None if accepted else "; ".join(matches[-1].rejection_reasons),
            correction_applied=accepted and apply,
        )
        return self.last_result

    def _reset_at_peak(self, centre, fov_id, scans, matches):
        """One opt-in attempt; callers fork shared references before using this."""
        from evomachine.softwarefocus import SoftwareFocus
        from evomachine.types import FocusCurveType

        config = self._config
        positive = np.r_[
            np.arange(0, config.calibration_range, config.calibration_step),
            config.calibration_range,
        ]
        offsets = np.r_[-positive[:0:-1], positive]
        try:
            locating = self._scan(centre, offsets, fov_id)
            scans.append(locating)
            scores = locating.focus_scores
            if SoftwareFocus._get_focus_curve_type(
                scores
            ) != FocusCurveType.HAS_GLOBAL_MAXIMUM or np.std(
                scores
            ) < config.search.min_relative_variation * np.mean(abs(scores)):
                return None
            peak_z = float(locating.z_coordinates[np.argmax(scores)])
            calibration = self._scan(peak_z, offsets, fov_id)
            scans.append(calibration)
            profile = fit_reference_profile(
                calibration.z_coordinates,
                calibration.focus_scores,
                focus_z=peak_z,
                smoothing=config.smoothing,
            )
        except (ReferenceFocusOutOfRange, ValueError):
            return None
        self._check_cancelled()
        self.acquisition_manager.stage.move(target=Coordinate(None, None, peak_z), block=True)
        self._check_cancelled()
        self.reference = profile
        self.pending_reference = None
        self.calibration_scan = calibration
        self.needs_calibration = False
        self.reference_message = None
        return ReferenceProfileFocusResult(
            tuple(scans),
            tuple(matches),
            FocusStatusType.IN_FOCUS,
            reference_reset=True,
            correction_applied=True,
        )

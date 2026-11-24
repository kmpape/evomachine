"""Focus configuration transport and independent manual/strategy state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import threading
from uuid import uuid4

import numpy as np

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.softwarefocus import (
    FocusScoringConfig, ReferenceFocusSelectionConfig, SoftwareFocus,
    SoftwareFocusConfig, SCORING_CONFIG_TYPES,
)
from evomachine.types import FilterWheelType, FocusScoringType, FocusSelectionType, LEDType


def frame_payload(frame):
    """Serialize one ``FrameMetaData`` object for GUI transport."""
    return dict(
        exposure=frame.exposure,
        filter_wheel=frame.filter_wheel.name if frame.filter_wheel else None,
        leds=None if frame.leds is None else {key.name: value for key, value in frame.leds.items()},
    )


def frame_from_payload(payload, base):
    """Apply a GUI frame mapping to a base ``FrameMetaData`` value."""
    if not isinstance(payload, dict):
        raise ValueError("Focus frame settings must be an object.")
    values = dict(payload)
    if "leds" in values and values["leds"] is not None:
        values["leds"] = {LEDType[key]: value for key, value in values["leds"].items()}
    if "filter_wheel" in values:
        values["filter_wheel"] = (
            FilterWheelType[values["filter_wheel"]] if values["filter_wheel"] else None
        )
    if set(values) - {"exposure", "filter_wheel", "leds"}:
        raise ValueError("Unknown focus-frame setting.")
    return replace(base, **values)


def settings_payload(settings):
    """Serialize acquisition settings, using defaults when settings are absent."""
    return dict(FrameAcquisitionSettings().__dict__ if settings is None else settings.__dict__)


def software_payload(config):
    """Serialize a typed unified ``SoftwareFocusConfig`` for the GUI."""
    scoring_type, scoring_config = config.scoring
    selection_type, selection_config = config.selection
    return dict(
        scoring_type=scoring_type.name,
        scoring_config=asdict(scoring_config),
        selection_type=selection_type.name,
        selection_config=asdict(selection_config),
        focus_frames=[frame_payload(frame) for frame in config.focus_frames],
        acquisition_settings=settings_payload(config.acquisition_settings),
        cropping_boxes=[
            dict(x_left=box.xtl, y_top=box.ytl, x_right=box.xbr, y_bottom=box.ybr)
            for box in config.cropping_boxes
        ],
    )


def reference_payload(config):
    """Serialize reference-profile selection settings from a config or engine config."""
    if isinstance(config, SoftwareFocusConfig):
        config = config.selection[1]
    if not isinstance(config, ReferenceFocusSelectionConfig):
        raise TypeError("Reference payload requires reference-profile selection settings.")
    payload = dict(
        calibration_range=config.calibration_range,
        calibration_step=config.calibration_step,
        sample_offsets=list(config.sample_offsets),
        recovery_offsets=list(config.recovery_offsets),
        smoothing=config.smoothing,
        allow_peak_reset=config.allow_peak_reset,
        share_reference_across_fovs=config.share_reference_across_fovs,
    )
    for prefix in ("sparse", "recovery"):
        payload.update(
            {f"{prefix}_{key}": value for key, value in config.matching_settings(prefix).items()}
        )
    return payload


def reference_from_payload(payload, base):
    """Validate GUI reference settings and apply them to a typed base config."""
    if not isinstance(payload, dict):
        raise ValueError("Reference focus settings must be an object.")
    values = dict(payload)
    unknown = set(values) - {item.name for item in fields(ReferenceFocusSelectionConfig)}
    if unknown:
        raise ValueError(f"Unknown reference setting(s): {', '.join(sorted(unknown))}.")
    for key in ("sample_offsets", "recovery_offsets"):
        if key in values:
            values[key] = tuple(values[key])
    for key in ("allow_peak_reset", "share_reference_across_fovs"):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"{key} must be boolean.")
    config = replace(base, **values)
    if config.smoothing is not None and (not np.isfinite(config.smoothing) or config.smoothing < 0):
        raise ValueError("Smoothing must be nonnegative or automatic.")
    return config


def curve_payload(profile):
    """Serialize measured and fitted points for an optional reference profile."""
    if profile is None:
        return None
    grid = np.linspace(profile.z_offsets[0], profile.z_offsets[-1], 1001)
    return dict(
        focus_z=profile.focus_z,
        z=profile.z_offsets.tolist(),
        scores=profile.focus_scores.tolist(),
        fit_z=grid.tolist(),
        fit_scores=profile.evaluate(grid).tolist(),
        smoothing=profile.smoothing,
    )


def _match_payload(scan):
    """Serialize optional match diagnostics from one focus scan."""
    return dict(
        accepted=bool(scan and scan.accepted),
        rejection_reasons=list(scan.rejection_reasons) if scan else [],
        shift=scan.candidate_shift if scan else None,
        correlation=scan.peak_correlation if scan else None,
        peak_margin=scan.peak_margin if scan else None,
        peak_width=scan.peak_width if scan else None,
        shifts=scan.shifts.tolist() if scan else [],
        correlations=[float(x) if np.isfinite(x) else None for x in scan.correlations]
        if scan
        else [],
    )


def _attempt_payload(profile, scan, label):
    """Expose raw samples and optional display alignment/scaling.

    Offsets are relative to the scan centre; subtracting the candidate shift
    aligns them with reference zero. A positive gain and baseline fit rescales
    scores for plotting only. Neither transform changes matching or stage Z.
    """
    offsets = scan.z_coordinates - scan.centre_z
    aligned = offsets - scan.candidate_shift if scan.candidate_shift is not None else offsets
    display_scores = scan.focus_scores
    scaled = False
    if (
        profile is not None
        and aligned.min() >= profile.z_offsets[0]
        and aligned.max() <= profile.z_offsets[-1]
    ):
        expected = profile.evaluate(aligned)
        scale, baseline = np.linalg.lstsq(
            np.column_stack([expected, np.ones(expected.size)]), scan.focus_scores, rcond=None
        )[0]
        if scale > 1e-12 and np.ptp(expected) > 1e-12:
            display_scores = (scan.focus_scores - baseline) / scale
            scaled = True
    return dict(
        label=label,
        offsets=offsets.tolist(),
        aligned_offsets=aligned.tolist(),
        scores=scan.focus_scores.tolist(),
        display_scores=display_scores.tolist(),
        scaled=scaled,
        **_match_payload(scan),
    )


def reference_status(focus):
    """Return reference state and diagnostics for a unified focus engine."""
    result = focus.last_result
    match = result.scans[-1] if result and result.scans else None
    profile = focus.pending_reference or focus.reference
    return dict(
        config=software_payload(focus.config),
        accepted=focus.reference is not None,
        needs_calibration=focus.needs_calibration,
        message=focus.reference_message,
        curve=curve_payload(profile),
        result=None
        if result is None
        else dict(
            status=result.focus_status.name,
            failure_reason=result.failure_reason,
            recovery_attempted=result.recovery_attempted,
            reference_reset=result.reference_reset,
            correction_applied=result.correction_applied,
            method="peak reset"
            if result.reference_reset
            else "recovery"
            if len(result.scans) > 1
            else "sparse",
            **_match_payload(match),
            attempts=[
                _attempt_payload(profile, scan, "Sparse" if index == 0 else "Recovery")
                for index, scan in enumerate(result.scans)
            ],
        ),
    )


@dataclass
class ReferenceReview:
    """Track one strategy-owned reference candidate awaiting a GUI decision."""
    identifier: str
    fov_id: int
    curve: dict
    decision: bool | None = None
    event: threading.Event = field(default_factory=threading.Event)


class FocusGuiState:
    """Keep manual experiments separate from strategy focus configuration/results.

    Hardware remains shared. Strategy review waits service queued GUI requests
    on the strategy thread at safe boundaries; decisions identify the current
    candidate and never execute hardware operations on the socket thread.
    """

    def __init__(self, facade):
        """Attach independent manual focus and strategy review state to a facade."""
        self.facade = facade
        self.manual_software = None
        self.review: ReferenceReview | None = None
        self.lock = threading.Lock()
        acquisition = getattr(facade.automaton, "acq_mngr", None)
        if acquisition is not None:
            acquisition.step_callback = self.process_events
        navigator = getattr(facade.automaton, "focus_nav", None)
        if navigator is not None:
            navigator.review_reference = self.review_strategy_reference
            software = getattr(navigator, "software_focus", None)
            if software is not None:
                software.process_events = self.process_events

    def process_events(self):
        """Service queued GUI requests at acquisition-safe strategy boundaries."""
        if getattr(self.facade.automaton, "_strategy_processing", False):
            self.facade.automaton.gui_process_requests()
            if self.facade.automaton.strategy_has_stopped():
                self.facade.automaton._stop_event.set()

    def invalidate_hardware_focus(self):
        """Invalidate cached hardware-focus state when the navigator supports it."""
        navigator = getattr(self.facade.automaton, "focus_nav", None)
        invalidate = getattr(navigator, "invalidate_hardware_focus", None)
        if callable(invalidate):
            invalidate()

    def prepare_manual_scan(self):
        """Disable hardware autofocus before a manual software-focus scan."""
        autofocus = getattr(self.facade.automaton.focus_nav, "autofocus", None)
        if autofocus is not None:
            if autofocus.is_locked():
                autofocus.unlock()
            autofocus.disable()
        self.invalidate_hardware_focus()

    def reference(self):
        """Return the single manual focus engine in reference-profile mode."""
        focus = self.facade.gui_software_focus()
        if focus.config.selection[0] is not FocusSelectionType.REFERENCE_PROFILE:
            focus.update_config(replace(
                focus.config,
                selection=(FocusSelectionType.REFERENCE_PROFILE, ReferenceFocusSelectionConfig()),
            ))
        focus.process_events = self.process_events
        return focus

    def review_strategy_reference(self, profile, fov_id, stop_event):
        """Wait for this strategy candidate's acceptance, rejection or cancellation."""
        review = ReferenceReview(uuid4().hex, fov_id, curve_payload(profile))
        with self.lock:
            self.review = review
        try:
            while not review.event.wait(0.05):
                if stop_event is not None and stop_event.is_set():
                    return False
                if (
                    getattr(self.facade.automaton, "_strategy_processing", False)
                    and self.facade.automaton.strategy_has_stopped()
                ):
                    return False
                if getattr(self.facade.automaton, "_strategy_processing", False):
                    self.facade.automaton.gui_process_requests()
            if review.decision is False and getattr(
                self.facade.automaton, "_strategy_processing", False
            ):
                self.facade.automaton.stop()
            return bool(review.decision)
        finally:
            with self.lock:
                if self.review is review:
                    self.review = None

    def decide(self, identifier, accepted):
        """Resolve the current strategy review by identifier with a Boolean decision."""
        if not isinstance(accepted, bool):
            raise ValueError("Reference acceptance must be boolean.")
        with self.lock:
            if self.review is None or self.review.identifier != identifier:
                raise ValueError("This reference review is no longer current.")
            self.review.decision = accepted
            self.review.event.set()

    def strategy_status(self):
        """Serialize the current strategy reference review, if one is pending."""
        with self.lock:
            review = self.review
            payload = (
                None
                if review is None
                else dict(id=review.identifier, fov_id=review.fov_id, curve=review.curve)
            )
        return dict(review=payload)


def strategy_reference_decide(facade, payload):
    """Apply a GUI decision to the current strategy reference candidate."""
    facade.focus_state.decide(payload["id"], payload["accepted"])
    return {"strategy_focus": facade.focus_state.strategy_status()}


def focus_defaults(facade, payload):
    """Return authoritative strategy focus and autofocus defaults for GUI editors."""
    facade.focus_state.reference()
    return {
        "focus_defaults": dict(
            software_focus=software_payload(facade.gui_strategy_software_focus().default_config),
            reference_profile=reference_payload(
                replace(
                    facade.gui_strategy_software_focus().config,
                    selection=(FocusSelectionType.REFERENCE_PROFILE, ReferenceFocusSelectionConfig()),
                )
            ),
            autofocus=facade._gui_autofocus_config_payload(
                getattr(
                    getattr(facade.automaton.focus_nav, "autofocus", None), "tiger_config", None
                )
            ),
        )
    }

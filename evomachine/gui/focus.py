"""Focus configuration transport and independent manual/strategy state."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import threading
from uuid import uuid4

import numpy as np

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.focus_registration import ShiftSearchConfig
from evomachine.referencefocus import ReferenceProfileFocus, ReferenceProfileFocusConfig
from evomachine.types import FilterWheelType, FocusAlgorithmType, LEDType


def frame_payload(frame):
    return dict(
        exposure=frame.exposure,
        filter_wheel=frame.filter_wheel.name if frame.filter_wheel else None,
        leds=None if frame.leds is None else {key.name: value for key, value in frame.leds.items()},
    )


def frame_from_payload(payload, base):
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
    return dict(FrameAcquisitionSettings().__dict__ if settings is None else settings.__dict__)


def software_payload(config):
    return dict(
        rel_range=config.rel_range,
        step_size=config.step_size,
        algorithm=config.algorithm.name,
        algorithm_kwargs=dict(config.algorithm_kwargs),
        focus_frames=[frame_payload(frame) for frame in config.focus_frames],
        acquisition_settings=settings_payload(config.acquisition_settings),
    )


def reference_payload(config):
    return dict(
        focus_frame=frame_payload(config.focus_frame),
        calibration_range=config.calibration_range,
        calibration_step=config.calibration_step,
        sample_offsets=list(config.sample_offsets),
        recovery_offsets=list(config.recovery_offsets),
        search=dict(vars(config.search)),
        recovery_search=dict(vars(config.recovery_search)),
        algorithm=config.algorithm.name,
        algorithm_kwargs=dict(config.algorithm_kwargs),
        smoothing=config.smoothing,
        allow_peak_reset=config.allow_peak_reset,
        acquisition_settings=settings_payload(config.acquisition_settings),
    )


def reference_from_payload(payload, base):
    if not isinstance(payload, dict):
        raise ValueError("Reference focus settings must be an object.")
    values = dict(payload)
    unknown = set(values) - {item.name for item in fields(ReferenceProfileFocusConfig)}
    if unknown:
        raise ValueError(f"Unknown reference setting(s): {', '.join(sorted(unknown))}.")
    if "focus_frame" in values:
        values["focus_frame"] = frame_from_payload(values["focus_frame"], base.focus_frame)
    for key in ("sample_offsets", "recovery_offsets"):
        if key in values:
            values[key] = tuple(values[key])
    for key in ("search", "recovery_search"):
        if key in values:
            values[key] = ShiftSearchConfig(**values[key])
    if "algorithm" in values:
        values["algorithm"] = FocusAlgorithmType[values["algorithm"]]
    if "acquisition_settings" in values:
        values["acquisition_settings"] = FrameAcquisitionSettings(**values["acquisition_settings"])
    if "allow_peak_reset" in values and not isinstance(values["allow_peak_reset"], bool):
        raise ValueError("allow_peak_reset must be boolean.")
    config = replace(base, **values)
    from evomachine.bindings.software_focus.software_focus_algorithms import (
        create_software_focus_algorithm,
    )

    create_software_focus_algorithm(config.algorithm, **config.algorithm_kwargs)
    if config.smoothing is not None and (not np.isfinite(config.smoothing) or config.smoothing < 0):
        raise ValueError("Smoothing must be nonnegative or automatic.")
    return config


def curve_payload(profile):
    if profile is None:
        return None
    grid = np.linspace(profile.z_offsets[0], profile.z_offsets[-1], 501)
    return dict(
        focus_z=profile.focus_z,
        z=profile.z_offsets.tolist(),
        scores=profile.focus_scores.tolist(),
        fit_z=grid.tolist(),
        fit_scores=profile.evaluate(grid).tolist(),
        smoothing=profile.smoothing,
    )


def reference_status(focus):
    result = focus.last_result
    match = result.matches[-1] if result and result.matches else None
    return dict(
        config=reference_payload(focus.config),
        accepted=focus.reference is not None,
        curve=curve_payload(focus.pending_reference or focus.reference),
        result=None
        if result is None
        else dict(
            status=result.focus_status.name,
            failure_reason=result.failure_reason,
            recovery_attempted=result.recovery_attempted,
            reference_reset=result.reference_reset,
            correction_applied=result.correction_applied,
            shift=match.candidate_shift if match else None,
            correlation=match.peak_correlation if match else None,
            peak_margin=match.peak_margin if match else None,
            peak_width=match.peak_width if match else None,
            shifts=match.shifts.tolist() if match else [],
            correlations=[float(x) if np.isfinite(x) else None for x in match.correlations]
            if match
            else [],
        ),
    )


@dataclass
class ReferenceReview:
    identifier: str
    fov_id: int
    curve: dict
    decision: bool | None = None
    event: threading.Event = field(default_factory=threading.Event)


class FocusGuiState:
    def __init__(self, facade):
        self.facade = facade
        self.manual_software = None
        self.manual_reference = None
        self.review: ReferenceReview | None = None
        self.lock = threading.Lock()
        acquisition = getattr(facade.automaton, "acq_mngr", None)
        if acquisition is not None:
            acquisition.step_callback = self.process_events
        navigator = getattr(facade.automaton, "focus_nav", None)
        if navigator is not None:
            navigator.review_reference = self.review_strategy_reference
            reference = getattr(navigator, "reference_focus", None)
            if reference is not None:
                reference.process_events = self.process_events

    def process_events(self):
        if getattr(self.facade.automaton, "_strategy_processing", False):
            self.facade.automaton.gui_process_requests()
            if self.facade.automaton.strategy_has_stopped():
                self.facade.automaton._stop_event.set()

    def invalidate_hardware_focus(self):
        navigator = getattr(self.facade.automaton, "focus_nav", None)
        invalidate = getattr(navigator, "invalidate_hardware_focus", None)
        if callable(invalidate):
            invalidate()

    def prepare_manual_scan(self):
        autofocus = getattr(self.facade.automaton.focus_nav, "autofocus", None)
        if autofocus is not None:
            if autofocus.is_locked():
                autofocus.unlock()
            autofocus.disable()
        self.invalidate_hardware_focus()

    def reference(self):
        if self.manual_reference is None:
            nav = self.facade.automaton.focus_nav
            base = getattr(nav, "reference_focus", None)
            if base is None:
                software = self.facade.gui_strategy_software_focus()
                base = ReferenceProfileFocus(
                    software.acquisition_manager,
                    ReferenceProfileFocusConfig(
                        focus_frame=software.default_config.focus_frames[0]
                    ),
                )
                nav.reference_focus = base
                base.process_events = self.process_events
            self.manual_reference = ReferenceProfileFocus(base.acquisition_manager, base.config)
        return self.manual_reference

    def review_strategy_reference(self, profile, fov_id, stop_event):
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
        if not isinstance(accepted, bool):
            raise ValueError("Reference acceptance must be boolean.")
        with self.lock:
            if self.review is None or self.review.identifier != identifier:
                raise ValueError("This reference review is no longer current.")
            self.review.decision = accepted
            self.review.event.set()

    def strategy_status(self):
        with self.lock:
            review = self.review
            payload = (
                None
                if review is None
                else dict(id=review.identifier, fov_id=review.fov_id, curve=review.curve)
            )
        return dict(review=payload)


def manual_reference_status(facade, payload):
    return {"reference_focus": reference_status(facade.focus_state.reference())}


def manual_reference_configure(facade, payload):
    focus = facade.focus_state.reference()
    focus.update_config(reference_from_payload(payload["config"], focus.config))
    return manual_reference_status(facade, payload)


def manual_reference_run(facade, payload):
    if not facade.automaton.devices_is_initialised():
        raise RuntimeError("Initialise devices before running reference focus.")
    focus = facade.focus_state.reference()
    action = payload.get("action", "run")
    if action not in {"calibrate", "run", "test"}:
        raise ValueError("Unknown reference operation.")

    def run(cancel_event, report):
        report(0, "Capturing reference focus images.")
        facade.focus_state.prepare_manual_scan()
        if action == "calibrate":
            focus.calibrate(stop_event=cancel_event)
        else:
            focus.run(
                focus.acquisition_manager.stage.get_fov_id(),
                stop_event=cancel_event,
                apply=action == "run",
            )
        report(1, "Inspect the reference curve and focus diagnostics.")
        return {"reference_focus": reference_status(focus)}

    return {"operation": facade.gui_operations.start("reference_focus", run)}


def manual_reference_edit(facade, payload):
    focus = facade.focus_state.reference()
    action = payload.get("action")
    if action == "accept":
        focus.accept_reference()
    elif action == "clear":
        focus.clear_reference()
    elif action == "refit":
        focus.refit_reference(payload.get("smoothing"))
    else:
        raise ValueError("Unknown reference action.")
    return manual_reference_status(facade, payload)


def manual_reference_cancel(facade, payload):
    facade.focus_state.reference().stop()
    return {"operation": facade.gui_operations.cancel("reference_focus")}


def manual_reference_operation(facade, payload):
    return {
        "operation": facade.gui_operations.status("reference_focus")
        or {"kind": "reference_focus", "state": "idle"}
    }


def strategy_reference_decide(facade, payload):
    facade.focus_state.decide(payload["id"], payload["accepted"])
    return {"strategy_focus": facade.focus_state.strategy_status()}


def focus_defaults(facade, payload):
    facade.focus_state.reference()
    return {
        "focus_defaults": dict(
            software_focus=software_payload(facade.gui_strategy_software_focus().default_config),
            reference_profile=reference_payload(facade.automaton.focus_nav.reference_focus.config),
            autofocus=facade._gui_autofocus_config_payload(
                getattr(
                    getattr(facade.automaton.focus_nav, "autofocus", None), "tiger_config", None
                )
            ),
        )
    }

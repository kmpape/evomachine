from dataclasses import replace
import threading

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionManager
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.frame import FrameMetaData
from evomachine.navigation import FocusNavigator, FocusNavigatorConfig
from evomachine.softwarefocus import (
    ReferenceFocusSelectionConfig, SoftwareFocus, SoftwareFocusConfig,
    SquaredGradientScoringConfig,
)
from evomachine.types import FocusMode, FocusScoringType, FocusSelectionType, FocusStatusType
from tests.test_acquisition import FakeLedManager
from tests.test_automaton import make_automaton
from tests.test_focusnavigator import FakeAutofocus, FakeSoftwareFocus, FakeStage


class FocusCamera:
    def __init__(self, stage):
        self.stage = stage
        self.focus_z = 120.25
        self.flat_frames = 0
        self.cancel_event = None
        self.fail = False

    def set_exposure(self, exposure_time):
        pass

    def get_frame(self, normalise=False):
        if self.fail:
            raise RuntimeError("Camera failed")
        z = self.stage.get_coordinates().z - self.focus_z
        score = 1 + np.exp(-((z / 14) ** 2)) + 2 * np.exp(-(((z - 32) / 9) ** 2))
        if self.flat_frames:
            score = 1
            self.flat_frames -= 1
        if self.cancel_event is not None:
            self.cancel_event.set()
        return np.tile([0, np.sqrt(score), 0], (4, 1))


@pytest.fixture
def setup():
    stage = FakeStage(Coordinate(0, 0, 120.25))
    camera = FocusCamera(stage)
    acquisition = FrameAcquisitionManager(camera, FakeLedManager(), stage=stage)
    reference = SoftwareFocus(
        acquisition,
        SoftwareFocusConfig(
            focus_frames=(FrameMetaData(frame_id=-1, leds=None, filter_wheel=None, exposure=20),),
            scoring=(FocusScoringType.SQUARED_GRAD_AVG, SquaredGradientScoringConfig()),
            selection=(FocusSelectionType.REFERENCE_PROFILE, ReferenceFocusSelectionConfig(smoothing=0)),
        ),
    )
    peak_focus = reference
    crisp = FakeAutofocus()
    navigator = FocusNavigator(
        stage,
        autofocus=crisp,
        software_focus=reference,
        config=FocusNavigatorConfig(
            focus_mode=FocusMode.REFERENCE_PROFILE,
        ),
    )
    navigator.initialise_fovs({0: Coordinate(0, 0, 999), 1: Coordinate(100, 0, -999)})
    return navigator, reference, camera, stage, peak_focus, crisp


def _calibrate(navigator):
    navigator.software_focus.update_config(
        _reference_config(navigator.software_focus.config, share_reference_across_fovs=True)
    )
    profile = navigator.calibrate_reference()
    navigator.accept_reference()
    return profile


def _reference_config(config: SoftwareFocusConfig, **updates) -> SoftwareFocusConfig:
    """Return a unified config with updated nested reference settings."""
    selection = replace(config.selection[1], **updates)
    return replace(config, selection=(FocusSelectionType.REFERENCE_PROFILE, selection))


def test_reference_curves_are_acquired_per_fov_by_default(setup):
    """Default strategy navigation must calibrate an independent curve per FoV."""
    navigator, reference, _, _, _, _ = setup
    reference.update_config(
        _reference_config(reference.config, share_reference_across_fovs=False)
    )
    navigator.initialise_fovs(
        {0: Coordinate(0, 0, 120.25), 1: Coordinate(100, 0, 120.25)}
    )
    reviewed = []
    navigator.review_reference = lambda profile, fov_id, event: (
        reviewed.append((fov_id, profile)) or True
    )
    navigator.prepare_run()

    assert not navigator.move(0).skipped
    assert not navigator.move(1).skipped
    first = navigator.reference_for_fov(0)
    second = navigator.reference_for_fov(1)
    assert first is not second
    assert first.reference is reviewed[0][1]
    assert second.reference is reviewed[1][1]
    assert [fov_id for fov_id, _ in reviewed] == [0, 1]


def test_reference_curve_can_be_shared_explicitly(setup):
    """The opt-in shared mode must calibrate once and reuse the same curve."""
    navigator, reference, _, _, _, _ = setup
    reference.update_config(
        _reference_config(reference.config, share_reference_across_fovs=True)
    )
    navigator.initialise_fovs(
        {0: Coordinate(0, 0, 120.25), 1: Coordinate(100, 0, 120.25)}
    )
    reviewed = []
    navigator.review_reference = lambda profile, fov_id, event: (
        reviewed.append((fov_id, profile)) or True
    )
    navigator.prepare_run()

    assert not navigator.move(0).skipped
    assert not navigator.move(1).skipped
    assert navigator.reference_for_fov(0) is reference
    assert navigator.reference_for_fov(1) is reference
    assert [fov_id for fov_id, _ in reviewed] == [0]


def test_calibration_preserves_manual_z_and_requires_acceptance(setup):
    navigator, reference, _, stage, _, _ = setup
    reference.update_config(
        _reference_config(reference.config, share_reference_across_fovs=True)
    )
    profile = navigator.calibrate_reference()
    assert profile.focus_z == stage.get_coordinates().z == 120.25
    assert reference.calibration_scan.frame.array.shape == (81, 4, 3)
    assert profile.z_offsets[np.argmax(profile.focus_scores)] != 0
    moves = len(stage.moves)
    with pytest.raises(RuntimeError, match="accept"):
        navigator.move(1)
    assert len(stage.moves) == moves
    navigator.accept_reference()
    navigator.validate_ready()


def test_arrival_uses_current_z_and_reference_focus_only(setup):
    navigator, reference, camera, stage, _, crisp = setup
    _calibrate(navigator)
    camera.focus_z += 7.5
    stage.moves.clear()
    result = navigator.move(1)
    assert stage.moves[0].z is None
    assert not result.skipped
    assert result.software_focus_status == FocusStatusType.IN_FOCUS
    assert result.coordinate.z == pytest.approx(camera.focus_z, abs=0.5)
    assert reference.last_result.scans[0].accepted
    assert not crisp.is_locked()


def test_recovery_scan_can_resolve_failed_sparse_match(setup):
    navigator, reference, camera, _, _, _ = setup
    _calibrate(navigator)
    camera.focus_z += 7.5
    camera.flat_frames = 9
    result = navigator.move(1)
    assert not result.skipped
    assert result.refocusing
    assert len(reference.last_result.scans) == 2
    assert result.coordinate.z == pytest.approx(camera.focus_z, abs=0.5)


def test_sparse_and_recovery_matching_thresholds_are_independent(setup):
    """A sparse-only rejection threshold must not alter recovery acceptance."""
    navigator, reference, camera, _, _, _ = setup
    _calibrate(navigator)
    reference.update_config(
        _reference_config(
            reference.config,
            sparse_min_relative_variation=10,
            recovery_min_relative_variation=0.01,
        )
    )
    camera.focus_z += 7.5
    result = navigator.move(1)
    assert not result.skipped
    assert len(reference.last_result.scans) == 2
    assert not reference.last_result.scans[0].accepted
    assert reference.last_result.scans[1].accepted


def test_failure_skips_fov_and_next_move_can_focus(setup):
    navigator, reference, camera, stage, _, _ = setup
    _calibrate(navigator)
    selection = reference.config.selection[1]
    camera.flat_frames = len(selection.sample_offsets) + len(
        selection.recovery_offsets
    )
    result = navigator.move(1)
    assert result.skipped
    assert len(reference.last_result.scans) == 2
    assert stage.get_coordinates().z == 120.25
    assert result.coordinate.z == -999  # Failed scans do not overwrite stored focal Z.
    assert not navigator.move(0).skipped


def test_cancelled_scan_does_not_apply_a_correction_or_recover(setup):
    navigator, reference, camera, stage, _, _ = setup
    _calibrate(navigator)
    camera.focus_z += 7.5
    camera.cancel_event = threading.Event()
    stage.moves.clear()
    result = navigator.move(1, stop_event=camera.cancel_event)
    assert result.skipped
    assert result.skip_reason == "Focus navigation cancelled."
    assert len(stage.moves) == 2  # XY arrival and one scan position; no later movement.
    assert reference.last_result is None


def test_scan_bounds_are_checked_before_scanning(setup):
    navigator, _, _, stage, _, _ = setup
    _calibrate(navigator)
    stage._coordinate_bounds = CoordinateBounds(
        low=Coordinate(-1000, -1000, 110),
        high=Coordinate(1000, 1000, 130),
    )
    stage.moves.clear()
    result = navigator.move(1)
    assert result.skipped
    assert result.software_focus_status == FocusStatusType.OUT_OF_RANGE
    assert len(stage.moves) == 1


def test_camera_failure_is_not_treated_as_a_bad_match(setup):
    navigator, _, camera, stage, _, _ = setup
    _calibrate(navigator)
    camera.fail = True
    with pytest.raises(RuntimeError, match="Camera failed"):
        navigator.move(1)
    assert stage.get_coordinates().z == 120.25


def test_smoothing_change_refits_cached_data_for_review(setup):
    navigator, reference, _, _, _, _ = setup
    _calibrate(navigator)
    scan = reference.calibration_scan
    reference.update_config(_reference_config(reference.config, smoothing=10))
    assert reference.calibration_scan is scan
    assert reference.pending_reference is not None
    with pytest.raises(RuntimeError, match="accept"):
        navigator.validate_ready()
    reference.accept_reference()
    navigator.validate_ready()


def test_scan_and_search_edits_preserve_accepted_reference(setup):
    navigator, reference, _, stage, _, _ = setup
    profile = _calibrate(navigator)
    moves = len(stage.moves)
    reference.update_config(
        _reference_config(
            reference.config,
            calibration_range=50,
            calibration_step=2,
            sample_offsets=ReferenceFocusSelectionConfig.sampling_offsets(step=10),
            sparse_min_correlation=0.95,
        )
    )
    assert reference.reference is profile
    assert reference.calibration_scan is not None
    assert len(stage.moves) == moves
    navigator.validate_ready()


def test_imaging_changes_retain_curve_but_require_new_capture(setup):
    navigator, reference, _, _, _, _ = setup
    profile = _calibrate(navigator)
    reference.update_config(
        replace(reference.config, focus_frames=(replace(reference.config.focus_frames[0], exposure=30),))
    )
    assert reference.reference is None
    assert reference.pending_reference is None
    assert not reference.needs_calibration
    with pytest.raises(RuntimeError, match="Capture and inspect"):
        reference.accept_reference()


def test_mode_defaults_to_standard_and_can_be_changed(setup):
    navigator, _, _, stage, _, _ = setup
    assert FocusNavigatorConfig().focus_mode == FocusMode.STANDARD
    with pytest.raises(ValueError, match="use_autofocus"):
        FocusNavigatorConfig(focus_mode=FocusMode.REFERENCE_PROFILE, use_autofocus=True)
    navigator.update_config(FocusNavigatorConfig())
    assert stage._fov_id_to_coordinate[1].z == -999
    navigator.update_config(FocusNavigatorConfig(focus_mode=FocusMode.REFERENCE_PROFILE))
    assert stage._fov_id_to_coordinate[1].z is None


def test_strategy_start_requires_acceptance_and_stop_cancels_reference_focus(setup):
    navigator, reference, _, _, _, _ = setup
    automaton, *_ = make_automaton()
    automaton.focus_nav = navigator
    with pytest.raises(RuntimeError, match="accept"):
        automaton.start_strategy()
    assert not automaton.strategy_has_started()
    _calibrate(navigator)
    automaton.start_strategy()
    assert automaton.strategy_has_started()
    automaton.stop()
    assert reference._stop_event.is_set()


def test_strategy_initial_reference_uses_saved_z_and_requests_review(setup):
    navigator, reference, camera, stage, _, _ = setup
    navigator._fov_states[1].coordinate.z = 123.75
    camera.focus_z = 123.75
    reviewed = []
    navigator.review_reference = lambda profile, fov_id, event: (
        reviewed.append((profile.focus_z, fov_id)) or True
    )
    result = navigator.move(1)
    assert reviewed == [(123.75, 1)]
    assert navigator.reference_for_fov(1).reference.focus_z == 123.75
    assert reference.reference is None
    assert not result.skipped


def test_standard_saved_z_initialisation_calibrates_crisp_without_peak_search(setup):
    navigator, software, _, stage, _, crisp = setup
    navigator._fov_states[0].coordinate.z = 123.75
    navigator.update_config(
        FocusNavigatorConfig(use_autofocus=True, initialise_from_first_fov_z=True)
    )
    result = navigator.move(0)
    assert stage.get_coordinates().z == 123.75
    assert result.is_locked
    assert software.last_result is None
    assert "initialise:None" in crisp.history


def test_mixed_modes_preserve_global_policy(setup):
    from evomachine.navigation import FovConfig

    navigator, _, _, stage, _, _ = setup
    _calibrate(navigator)
    navigator._fov_states[0].coordinate.z = 120.25
    navigator.update_config(FocusNavigatorConfig())
    navigator.update_fov_config(1, FovConfig(focus_mode=FocusMode.REFERENCE_PROFILE))
    assert not navigator.move(0).skipped
    result = navigator.move(1)
    assert result.software_focus_status == FocusStatusType.IN_FOCUS
    assert navigator.config.focus_mode == FocusMode.STANDARD
    assert not navigator.move(0).skipped
    assert stage.get_coordinates().z == 120.25


def test_peak_reset_is_opt_in_and_local_to_failed_fov(setup):
    navigator, reference, camera, _, _, _ = setup
    reference.update_config(_reference_config(reference.config, allow_peak_reset=True))
    original = _calibrate(navigator)
    selection = reference.config.selection[1]
    camera.flat_frames = len(selection.sample_offsets) + len(
        selection.recovery_offsets
    )
    result = navigator.move(1)
    assert not result.skipped
    assert navigator.last_software_focus.last_result.reference_reset
    assert len(navigator.last_software_focus.last_result.scans) == 2
    assert reference.reference is original
    assert navigator.reference_for_fov(1).reference.focus_z != original.focus_z
    assert navigator.reference_for_fov(0).reference is original

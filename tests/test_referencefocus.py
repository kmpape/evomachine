from dataclasses import replace
import threading

import numpy as np
import pytest

from evomachine.acquisition import FrameAcquisitionManager
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.frame import FrameMetaData
from evomachine.navigation import FocusNavigator, FocusNavigatorConfig
from evomachine.referencefocus import ReferenceProfileFocus, ReferenceProfileFocusConfig
from evomachine.types import FocusMode, FocusStatusType
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
    reference = ReferenceProfileFocus(
        acquisition,
        ReferenceProfileFocusConfig(
            focus_frame=FrameMetaData(frame_id=-1, leds=None, filter_wheel=None, exposure=20),
            smoothing=0,
        ),
    )
    peak_focus = FakeSoftwareFocus(stage)
    crisp = FakeAutofocus()
    navigator = FocusNavigator(
        stage,
        autofocus=crisp,
        software_focus=peak_focus,
        reference_focus=reference,
        config=FocusNavigatorConfig(
            focus_mode=FocusMode.REFERENCE_PROFILE,
        ),
    )
    navigator.initialise_fovs({0: Coordinate(0, 0, 999), 1: Coordinate(100, 0, -999)})
    return navigator, reference, camera, stage, peak_focus, crisp


def _calibrate(navigator):
    profile = navigator.calibrate_reference()
    navigator.accept_reference()
    return profile


def test_calibration_preserves_manual_z_and_requires_acceptance(setup):
    navigator, reference, _, stage, _, _ = setup
    profile = navigator.calibrate_reference()
    assert profile.focus_z == stage.get_coordinates().z == 120.25
    assert reference.calibration_scan.frame.array.shape == (41, 4, 3)
    assert profile.z_offsets[np.argmax(profile.focus_scores)] != 0
    moves = len(stage.moves)
    with pytest.raises(RuntimeError, match="accept"):
        navigator.move(1)
    assert len(stage.moves) == moves
    navigator.accept_reference()
    navigator.validate_ready()


def test_arrival_uses_current_z_and_reference_focus_only(setup):
    navigator, reference, camera, stage, peak_focus, crisp = setup
    _calibrate(navigator)
    camera.focus_z += 7.5
    stage.moves.clear()
    result = navigator.move(1)
    assert stage.moves[0].z is None
    assert not result.skipped
    assert result.software_focus_status == FocusStatusType.IN_FOCUS
    assert result.coordinate.z == pytest.approx(camera.focus_z, abs=0.5)
    assert reference.last_result.matches[0].accepted
    assert peak_focus.runs == []
    assert not crisp.is_locked()


def test_recovery_scan_can_resolve_failed_sparse_match(setup):
    navigator, reference, camera, _, _, _ = setup
    _calibrate(navigator)
    camera.focus_z += 7.5
    camera.flat_frames = 9
    result = navigator.move(1)
    assert not result.skipped
    assert result.refocusing
    assert len(reference.last_result.matches) == 2
    assert result.coordinate.z == pytest.approx(camera.focus_z, abs=0.5)


def test_failure_skips_fov_and_next_move_can_focus(setup):
    navigator, reference, camera, stage, _, _ = setup
    _calibrate(navigator)
    camera.flat_frames = 30
    result = navigator.move(1)
    assert result.skipped
    assert len(reference.last_result.matches) == 2
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


def test_configuration_changes_require_recalibration(setup):
    navigator, reference, _, _, _, _ = setup
    _calibrate(navigator)
    reference.update_config(replace(reference.config, smoothing=10))
    with pytest.raises(RuntimeError, match="accept"):
        navigator.validate_ready()


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
    navigator.review_reference = lambda profile, fov_id, event: reviewed.append((profile.focus_z, fov_id)) or True
    result = navigator.move(1)
    assert reviewed == [(123.75, 1)]
    assert reference.reference.focus_z == 123.75
    assert not result.skipped


def test_standard_saved_z_initialisation_calibrates_crisp_without_peak_search(setup):
    navigator, _, _, stage, software, crisp = setup
    navigator._fov_states[0].coordinate.z = 123.75
    navigator.update_config(FocusNavigatorConfig(use_autofocus=True, initialise_from_first_fov_z=True))
    result = navigator.move(0)
    assert stage.get_coordinates().z == 123.75
    assert result.is_locked
    assert software.runs == []
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
    reference.update_config(replace(reference.config, allow_peak_reset=True))
    original = _calibrate(navigator)
    camera.flat_frames = 30
    result = navigator.move(1)
    assert not result.skipped
    assert navigator.last_reference_focus.last_result.reference_reset
    assert reference.reference is original
    assert navigator.reference_for_fov(1).reference.focus_z != original.focus_z
    assert navigator.reference_for_fov(0).reference is original

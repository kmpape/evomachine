import numpy as np
import pytest

from evomachine.softwarefocus import (
    FocusScan,
    ReferenceFocusSelectionConfig,
    SoftwareFocus,
    LaplacianVarianceScoringConfig,
    SquaredGradientScoringConfig,
    SteelScoringConfig,
    BandpassFFTScoringConfig,
)
from evomachine.types import FocusScoringType


def _shape(z):
    return 1 + np.exp(-((z / 14) ** 2)) + 2 * np.exp(-(((z - 32) / 9) ** 2))


@pytest.fixture
def reference():
    """Return a fitted reference profile with two differently sized peaks."""
    z = np.arange(-100, 101, 5, dtype=float)
    return SoftwareFocus._fit_reference_profile(
        120 + z, _shape(z), focus_z=120, smoothing=0
    )


def _match(reference, z, scores, *, current_z=0, **settings):
    """Match numerical samples through the consolidated scan model."""
    scan = FocusScan(np.asarray(z), np.asarray(scores), None, current_z)
    values = {
        "shift_min": -30.0,
        "shift_max": 30.0,
        "shift_step": 0.5,
        "min_correlation": 0.90,
        "min_peak_margin": 0.05,
        "boundary_margin": 1.0,
        "min_relative_variation": 0.01,
    } | settings
    ReferenceFocusSelectionConfig.validate_matching_settings("test", values)
    return SoftwareFocus._match_scan(
        reference, scan, current_z=current_z, settings=values
    )


def test_manual_focus_is_zero_even_when_another_peak_is_higher(reference):
    assert reference.focus_z == 120
    assert reference.z_offsets[np.argmax(reference.focus_scores)] != 0
    assert reference.evaluate(0) == pytest.approx(_shape(0))


def test_automatic_smoothing_and_override():
    z = np.arange(-100, 101, 5, dtype=float)
    scores = _shape(z) + np.random.default_rng(7).normal(0, 0.08, z.size)
    automatic = SoftwareFocus._fit_reference_profile(z, scores, focus_z=0)
    interpolated = SoftwareFocus._fit_reference_profile(z, scores, focus_z=0, smoothing=0)
    assert not np.allclose(automatic.evaluate(z), scores)
    np.testing.assert_allclose(interpolated.evaluate(z), scores)


@pytest.mark.parametrize("shift", [-12.5, 0, 7.5])
def test_recovers_shift_with_noise_scale_and_baseline(reference, shift):
    offsets = np.arange(-40, 41, 10)
    noise = np.random.default_rng(1).normal(0, 0.01, offsets.size)
    scores = 3.2 * _shape(offsets - shift) + 5 + noise
    result = _match(reference, 260 + offsets, scores, current_z=260)
    assert result.accepted, result.rejection_reasons
    assert result.focus_z == pytest.approx(260 + shift, abs=0.5)


@pytest.mark.parametrize("scores", [np.ones(9), 100 + np.linspace(0, 0.1, 9)])
def test_rejects_insufficient_score_variation(reference, scores):
    result = _match(reference, np.arange(-40, 41, 10), scores)
    assert not result.accepted


def test_rejects_linear_profile_with_no_identifiable_shift():
    z = np.arange(-100, 101, 5)
    reference = SoftwareFocus._fit_reference_profile(z, 200 + z, focus_z=0, smoothing=0)
    offsets = np.arange(-40, 41, 10)
    result = _match(reference, offsets, 200 + offsets)
    assert not result.accepted


def test_rejects_ambiguous_repeated_peaks():
    z = np.arange(-100, 101)
    reference = SoftwareFocus._fit_reference_profile(
        z, 2 + np.cos(z * np.pi / 10), focus_z=0, smoothing=0
    )
    offsets = np.arange(-16, 17, 4)
    scores = 2 + np.cos((offsets - 3) * np.pi / 10)
    result = _match(reference, offsets, scores)
    assert not result.accepted
    assert result.peak_margin < 0.05


@pytest.mark.parametrize("extent,shift", [(40, 30), (90, 10)])
def test_rejects_peaks_at_search_or_reference_boundaries(reference, extent, shift):
    offsets = np.arange(-extent, extent + 1, 10)
    result = _match(reference, offsets, _shape(offsets - shift))
    assert result.candidate_shift == shift
    assert not result.accepted


def test_rejects_poor_match(reference):
    offsets = np.arange(-40, 41, 10)
    result = _match(reference, offsets, [1, 6, 2, 8, 3, 9, 1, 5, 2])
    assert not result.accepted


def test_does_not_extrapolate_reference(reference):
    offsets = np.arange(-90, 91, 10)
    result = _match(reference, offsets, _shape(offsets))
    assert result.accepted
    assert np.all(np.isnan(result.correlations[np.abs(result.shifts) > 10]))
    with pytest.raises(ValueError, match="outside"):
        reference.evaluate([-101, 0])


def test_default_gradient_scoring_avoids_unsigned_overflow():
    image = np.tile(np.array([0, 255, 0], dtype=np.uint8), (3, 1))
    scores = SoftwareFocus._score_focus_stack(
        np.stack([image, np.zeros_like(image)]),
        scoring=(FocusScoringType.SQUARED_GRAD_AVG, SquaredGradientScoringConfig()),
    )
    np.testing.assert_allclose(scores, [255**2, 0])


@pytest.mark.parametrize("scoring_type", list(FocusScoringType))
def test_existing_metrics_are_selectable(scoring_type):
    image = np.random.default_rng(4).integers(0, 256, (32, 32), dtype=np.uint16)
    configs = {
        FocusScoringType.LAPLACIAN_VAR: LaplacianVarianceScoringConfig(),
        FocusScoringType.SQUARED_GRAD_AVG: SquaredGradientScoringConfig(),
        FocusScoringType.STEEL: SteelScoringConfig(rowshift=1, colshift=1),
        FocusScoringType.BANDPASS_FFT: BandpassFFTScoringConfig(),
    }
    scores = SoftwareFocus._score_focus_stack(
        np.stack([image, np.zeros_like(image)]), scoring=(scoring_type, configs[scoring_type])
    )
    assert scores[0] > scores[1] == 0

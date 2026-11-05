import numpy as np
import pytest
from matplotlib import pyplot as plt

from evomachine.focus_registration import (
    estimate_focus_shift,
    fit_reference_profile,
    plot_reference_profile,
    score_focus_stack,
)
from evomachine.types import FocusAlgorithmType


def _shape(z):
    return 1 + np.exp(-((z / 14) ** 2)) + 2 * np.exp(-(((z - 32) / 9) ** 2))


@pytest.fixture
def reference():
    z = np.arange(-100, 101, 5, dtype=float)
    return fit_reference_profile(120 + z, _shape(z), focus_z=120, smoothing=0)


def test_manual_focus_is_zero_even_when_another_peak_is_higher(reference):
    assert reference.focus_z == 120
    assert reference.z_offsets[np.argmax(reference.focus_scores)] != 0
    assert reference.evaluate(0) == pytest.approx(_shape(0))


def test_automatic_smoothing_and_override():
    z = np.arange(-100, 101, 5, dtype=float)
    scores = _shape(z) + np.random.default_rng(7).normal(0, 0.08, z.size)
    automatic = fit_reference_profile(z, scores, focus_z=0)
    interpolated = fit_reference_profile(z, scores, focus_z=0, smoothing=0)
    assert not np.allclose(automatic.evaluate(z), scores)
    np.testing.assert_allclose(interpolated.evaluate(z), scores)


@pytest.mark.parametrize("shift", [-12.5, 0, 7.5])
def test_recovers_shift_with_noise_scale_and_baseline(reference, shift):
    offsets = np.arange(-40, 41, 10)
    noise = np.random.default_rng(1).normal(0, 0.01, offsets.size)
    scores = 3.2 * _shape(offsets - shift) + 5 + noise
    result = estimate_focus_shift(reference, 260 + offsets, scores, current_z=260)
    assert result.accepted, result.rejection_reasons
    assert result.focus_z == pytest.approx(260 + shift, abs=0.5)


@pytest.mark.parametrize("scores", [np.ones(9), 100 + np.linspace(0, 0.1, 9)])
def test_rejects_insufficient_score_variation(reference, scores):
    result = estimate_focus_shift(reference, np.arange(-40, 41, 10), scores, current_z=0)
    assert not result.accepted


def test_rejects_linear_profile_with_no_identifiable_shift():
    z = np.arange(-100, 101, 5)
    reference = fit_reference_profile(z, 200 + z, focus_z=0, smoothing=0)
    offsets = np.arange(-40, 41, 10)
    result = estimate_focus_shift(reference, offsets, 200 + offsets, current_z=0)
    assert not result.accepted


def test_rejects_ambiguous_repeated_peaks():
    z = np.arange(-100, 101)
    reference = fit_reference_profile(z, 2 + np.cos(z * np.pi / 10), focus_z=0, smoothing=0)
    offsets = np.arange(-16, 17, 4)
    scores = 2 + np.cos((offsets - 3) * np.pi / 10)
    result = estimate_focus_shift(reference, offsets, scores, current_z=0)
    assert not result.accepted
    assert result.peak_margin < 0.05


@pytest.mark.parametrize("extent,shift", [(40, 30), (90, 10)])
def test_rejects_peaks_at_search_or_reference_boundaries(reference, extent, shift):
    offsets = np.arange(-extent, extent + 1, 10)
    result = estimate_focus_shift(reference, offsets, _shape(offsets - shift), current_z=0)
    assert result.candidate_shift == shift
    assert not result.accepted


def test_rejects_poor_match(reference):
    offsets = np.arange(-40, 41, 10)
    result = estimate_focus_shift(reference, offsets, [1, 6, 2, 8, 3, 9, 1, 5, 2], current_z=0)
    assert not result.accepted


def test_does_not_extrapolate_reference(reference):
    offsets = np.arange(-90, 91, 10)
    result = estimate_focus_shift(reference, offsets, _shape(offsets), current_z=0)
    assert result.accepted
    assert np.all(np.isnan(result.correlations[np.abs(result.shifts) > 10]))
    with pytest.raises(ValueError, match="outside"):
        reference.evaluate([-101, 0])


def test_default_gradient_scoring_avoids_unsigned_overflow():
    image = np.tile(np.array([0, 255, 0], dtype=np.uint8), (3, 1))
    scores = score_focus_stack(np.stack([image, np.zeros_like(image)]))
    np.testing.assert_allclose(scores, [255**2, 0])


@pytest.mark.parametrize("algorithm", list(FocusAlgorithmType))
def test_existing_metrics_are_selectable(algorithm):
    image = np.random.default_rng(4).integers(0, 256, (32, 32), dtype=np.uint16)
    kwargs = {"rowshift": 1, "colshift": 1} if algorithm == FocusAlgorithmType.STEEL else {}
    scores = score_focus_stack(
        np.stack([image, np.zeros_like(image)]), algorithm=algorithm, algorithm_kwargs=kwargs
    )
    assert scores[0] > scores[1] == 0


def test_preview_can_be_saved_without_displaying(reference, monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda **kwargs: pytest.fail("Unexpected display"))
    figure = plot_reference_profile(reference)
    try:
        path = tmp_path / "reference.png"
        figure.savefig(path)
        assert path.stat().st_size > 0
    finally:
        plt.close(figure)

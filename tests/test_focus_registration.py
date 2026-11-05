import numpy as np
import pytest
from matplotlib import pyplot as plt

from evomachine.focus_registration import (
    ShiftSearchConfig,
    estimate_focus_shift,
    fit_reference_profile,
    plot_reference_profile,
    score_focus_stack,
)
from evomachine.types import FocusAlgorithmType


def _shape(z):
    """An asymmetric profile whose largest peak is away from selected zero."""
    return 1 + np.exp(-((z / 14) ** 2)) + 2 * np.exp(-(((z - 32) / 9) ** 2))


@pytest.fixture
def reference():
    offsets = np.arange(-100, 101, 5, dtype=float)
    return fit_reference_profile(120 + offsets, _shape(offsets), focus_z=120, smoothing=0)


def test_selected_zero_is_preserved_instead_of_global_maximum(reference):
    assert reference.focus_z == 120
    assert reference.z_offsets[np.argmax(reference.focus_scores)] != 0
    assert reference.evaluate(0) == pytest.approx(_shape(0))
    np.testing.assert_allclose(reference.evaluate(reference.z_offsets), reference.focus_scores)
    assert not reference.z_offsets.flags.writeable
    assert not reference.focus_scores.flags.writeable


def test_automatic_smoothing_and_override():
    offsets = np.arange(-100, 101, 5, dtype=float)
    scores = _shape(offsets) + np.random.default_rng(7).normal(0, 0.08, offsets.size)
    automatic = fit_reference_profile(offsets, scores, focus_z=0)
    interpolated = fit_reference_profile(offsets, scores, focus_z=0, smoothing=0)
    smoother = fit_reference_profile(offsets, scores, focus_z=0, smoothing=500)
    np.testing.assert_allclose(interpolated.evaluate(offsets), scores)
    assert np.all(np.isfinite(automatic.evaluate(offsets)))
    assert not np.allclose(automatic.evaluate(offsets), scores)

    def roughness(profile):
        return np.linalg.norm(np.diff(profile.evaluate(offsets), n=2))

    assert roughness(smoother) < roughness(interpolated)


@pytest.mark.parametrize("shift", [-12.5, 0, 7.5, 18])
@pytest.mark.parametrize("scale,baseline", [(1, 0), (3.2, 5)])
def test_recovers_signed_shift_with_scale_and_baseline(reference, shift, scale, baseline):
    offsets = np.arange(-40, 41, 10)
    measured = scale * _shape(offsets - shift) + baseline
    result = estimate_focus_shift(reference, 260 + offsets, measured, current_z=260)
    assert result.accepted, result.rejection_reasons
    assert result.candidate_shift == pytest.approx(shift, abs=0.5)
    assert result.focus_z == pytest.approx(260 + shift, abs=0.5)
    assert result.peak_correlation > 0.99
    assert result.peak_width >= 0
    assert result.shifts.shape == result.correlations.shape


def test_sparse_measurements_with_noise(reference):
    offsets = np.arange(-40, 41, 10)
    measured = _shape(offsets - 7.5) + np.random.default_rng(1).normal(0, 0.01, offsets.size)
    result = estimate_focus_shift(reference, 260 + offsets, measured, current_z=260)
    assert result.accepted, result.rejection_reasons
    assert result.focus_z == pytest.approx(267.5, abs=0.5)


@pytest.mark.parametrize("scores", [np.ones(9), 100 + np.linspace(0, 0.1, 9)])
def test_low_variation_cannot_produce_accepted_focus(reference, scores):
    result = estimate_focus_shift(reference, np.arange(-40, 41, 10), scores, current_z=0)
    assert not result.accepted
    assert result.focus_z is None
    assert "variation" in result.rejection_reasons[0]


def test_linear_reference_is_not_identifiable():
    z = np.linspace(-100, 100, 41)
    reference = fit_reference_profile(z, 200 + z, focus_z=0, smoothing=0)
    offsets = np.arange(-40, 41, 10)
    result = estimate_focus_shift(reference, offsets, 200 + offsets, current_z=0)
    assert not result.accepted
    assert any("flat" in reason for reason in result.rejection_reasons)


def test_repeated_profile_peaks_are_ambiguous():
    z = np.arange(-100, 101, 1)
    reference = fit_reference_profile(z, 2 + np.cos(z * np.pi / 10), focus_z=0, smoothing=0)
    offsets = np.arange(-16, 17, 4)
    result = estimate_focus_shift(
        reference,
        offsets,
        2 + np.cos((offsets - 3) * np.pi / 10),
        current_z=0,
    )
    assert not result.accepted
    assert result.peak_margin < 0.05
    assert any("ambiguous" in reason for reason in result.rejection_reasons)


def test_boundary_solution_is_not_accepted(reference):
    offsets = np.arange(-40, 41, 10)
    result = estimate_focus_shift(reference, offsets, _shape(offsets - 30), current_z=0)
    assert result.candidate_shift == 30
    assert not result.accepted
    assert any("boundary" in reason for reason in result.rejection_reasons)


def test_poor_match_is_not_accepted(reference):
    offsets = np.arange(-40, 41, 10)
    scores = np.array([1, 6, 2, 8, 3, 9, 1, 5, 2])
    result = estimate_focus_shift(reference, offsets, scores, current_z=0)
    assert not result.accepted
    assert any("threshold" in reason for reason in result.rejection_reasons)


def test_out_of_domain_candidates_remain_unsupported(reference):
    offsets = np.arange(-90, 91, 10)
    result = estimate_focus_shift(reference, offsets, _shape(offsets), current_z=0)
    assert result.accepted, result.rejection_reasons
    assert np.all(np.isnan(result.correlations[np.abs(result.shifts) > 10]))
    outside = estimate_focus_shift(reference, offsets + 500, _shape(offsets), current_z=0)
    assert not outside.accepted
    assert np.all(np.isnan(outside.correlations))
    with pytest.raises(ValueError, match="outside"):
        reference.evaluate([-101, 0])


def test_truncated_valid_domain_boundary_is_rejected(reference):
    offsets = np.arange(-90, 91, 10)
    result = estimate_focus_shift(reference, offsets, _shape(offsets - 10), current_z=0)
    assert result.candidate_shift == 10
    assert not result.accepted
    assert any("boundary" in reason for reason in result.rejection_reasons)


def test_search_supports_nonuniform_samples_and_nondivisible_grid(reference):
    offsets = np.array([20, -37, 0, 38, -15, 7, -29])
    result = estimate_focus_shift(
        reference,
        260 + offsets,
        _shape(offsets - 6),
        current_z=260,
        config=ShiftSearchConfig(shift_min=-10, shift_max=10, shift_step=3),
    )
    assert result.accepted, result.rejection_reasons
    assert result.shifts[-1] == 10
    assert result.candidate_shift == pytest.approx(6, abs=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"shift_min": 30},
        {"shift_step": 0},
        {"shift_step": 40},
        {"shift_step": 1e-12},
        {"shift_max": np.nan},
        {"min_correlation": 2},
        {"min_peak_margin": -1},
        {"boundary_margin": 30},
        {"min_relative_variation": -1},
    ],
)
def test_invalid_search_configuration(changes):
    with pytest.raises(ValueError):
        ShiftSearchConfig(**changes)


@pytest.mark.parametrize(
    "z,scores,focus_z,smoothing",
    [
        ([0, 1, 2, 3], [1, 2, 3, 4], 0, None),
        ([0, 1, 1, 3, 4], [1, 2, 3, 4, 5], 0, None),
        ([4, 3, 2, 1, 0], [1, 2, 3, 4, 5], 0, None),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, np.nan], 0, None),
        ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], 0, None),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], 5, None),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], 0, -1),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], 0, np.inf),
    ],
)
def test_invalid_reference_measurements(z, scores, focus_z, smoothing):
    with pytest.raises(ValueError):
        fit_reference_profile(z, scores, focus_z=focus_z, smoothing=smoothing)


@pytest.mark.parametrize(
    "z,scores",
    [
        ([0, 1, 2, 3], [1, 2, 3, 4]),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, np.inf]),
        ([0, 1, 2, 3, 3], [1, 2, 3, 4, 5]),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
    ],
)
def test_invalid_local_measurements(reference, z, scores):
    with pytest.raises(ValueError):
        estimate_focus_shift(reference, z, scores, current_z=0)


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


@pytest.mark.parametrize(
    "stack",
    [np.zeros((3, 3)), np.zeros((0, 3, 3)), np.full((1, 3, 3), np.nan), np.zeros((1, 2, 2))],
)
def test_invalid_image_stack(stack):
    with pytest.raises(ValueError):
        score_focus_stack(stack)


def test_preview_returns_figure_without_showing_or_matching(reference, monkeypatch, tmp_path):
    def unexpected_show(*args, **kwargs):
        pytest.fail("Preview must leave displaying/continuing to the caller.")

    monkeypatch.setattr(plt, "show", unexpected_show)
    figure = plot_reference_profile(reference)
    try:
        assert len(figure.axes[0].lines) == 3
        assert list(figure.axes[0].lines[2].get_xdata()) == [0, 0]
        path = tmp_path / "reference.png"
        figure.savefig(path)
        assert path.stat().st_size > 0
    finally:
        plt.close(figure)

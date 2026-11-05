"""Reference-profile fitting and registration, independent of stage movement.

Z values are in micrometres. Fit a reference about the manually selected focus,
inspect it with ``plot_reference_profile`` and ``plt.show()``, then explicitly
call ``estimate_focus_shift`` with subsequent measurements. Nothing runs or
accepts a reference automatically when its plot is closed.
Use the same image metric, ROI and acquisition settings for both sets of scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, make_smoothing_spline
from scipy.signal import find_peaks

from evomachine.bindings.software_focus.software_focus_algorithms import (
    create_software_focus_algorithm,
)
from evomachine.types import FocusAlgorithmType


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _vector(values: ArrayLike, name: str, minimum_size: int = 1) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values.")
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size < minimum_size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1D array with at least {minimum_size} values.")
    return result


def _unit_variation(values: NDArray[np.float64]) -> NDArray[np.float64] | None:
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return None
    centred = values / scale - np.mean(values / scale)
    norm = float(np.linalg.norm(centred))
    return None if norm <= 1e-12 else centred / norm


@dataclass(frozen=True)
class ReferenceProfile:
    """Measured scores and fitted spline relative to a manually selected focal Z.

    Construct with ``fit_reference_profile``. ``smoothing=None`` requests GCV;
    nonnegative overrides use SciPy's lambda convention (larger is smoother).
    """

    z_offsets: NDArray[np.float64]
    focus_scores: NDArray[np.float64]
    focus_z: float
    smoothing: float | None
    _spline: BSpline = field(repr=False, compare=False)

    def evaluate(self, z_offsets: ArrayLike) -> NDArray[np.float64]:
        """Evaluate relative Zs inside the measured domain, without extrapolation."""
        offsets = np.asarray(z_offsets, dtype=np.float64)
        if not np.all(np.isfinite(offsets)):
            raise ValueError("Reference evaluation requires finite Z offsets.")
        if np.any(offsets < self.z_offsets[0]) or np.any(offsets > self.z_offsets[-1]):
            raise ValueError("Z offsets lie outside the measured reference range.")
        return np.asarray(self._spline(offsets), dtype=np.float64)


def score_focus_stack(
    images: ArrayLike,
    *,
    algorithm: FocusAlgorithmType = FocusAlgorithmType.SQUARED_GRAD_AVG,
    algorithm_kwargs: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """Score a Z-first ``(planes, height, width)`` stack with an existing metric.

    Supply cropped images if an ROI is wanted. Convert each plane to float before
    scoring to avoid unsigned subtraction/squaring overflow in gradient metrics.
    Existing autofocus defaults and implementations are unaffected.
    """
    stack = np.asarray(images)
    if stack.ndim != 3 or stack.shape[0] == 0 or min(stack.shape[1:]) < 3:
        raise ValueError("images must have shape (planes, height, width), with height/width >= 3.")
    if not np.issubdtype(stack.dtype, np.number) or np.iscomplexobj(stack):
        raise ValueError("images must contain real numeric values.")
    scorer = create_software_focus_algorithm(algorithm, **(algorithm_kwargs or {}))
    scores = []
    for image in stack:
        image = image.astype(np.float64)
        if not np.all(np.isfinite(image)):
            raise ValueError("images must contain finite values.")
        scores.append(scorer.score_image(image))
    return _vector(scores, "focus scores")


def fit_reference_profile(
    z_coordinates: ArrayLike,
    focus_scores: ArrayLike,
    *,
    focus_z: float,
    smoothing: float | None = None,
) -> ReferenceProfile:
    """Fit a cubic smoothing spline, keeping the selected focus as relative zero.

    Supply at least five strictly increasing absolute Zs and their scores.
    The selected focus must lie inside that range; it need not be a score peak.
    """
    z = _vector(z_coordinates, "z_coordinates", 5)
    scores = _vector(focus_scores, "focus_scores", 5)
    focus_z = _finite_number(focus_z, "focus_z")
    if z.shape != scores.shape or np.any(np.diff(z) <= 0):
        raise ValueError("Zs and scores must have equal lengths, with strictly increasing Zs.")
    if not z[0] <= focus_z <= z[-1]:
        raise ValueError("focus_z must lie inside the measured Z range.")
    if _unit_variation(scores) is None:
        raise ValueError("Reference scores contain insufficient variation.")
    if smoothing is not None:
        smoothing = _finite_number(smoothing, "smoothing")
        if smoothing < 0:
            raise ValueError("smoothing must be nonnegative or None.")
    offsets = z - focus_z
    spline = make_smoothing_spline(offsets, scores, lam=smoothing)
    spline.extrapolate = False
    offsets.setflags(write=False)
    scores.setflags(write=False)
    return ReferenceProfile(offsets, scores, focus_z, smoothing, spline)


@dataclass(frozen=True)
class ShiftSearchConfig:
    """Grid and provisional acceptance thresholds; tune against microscope data.

    ``min_relative_variation`` compares score standard deviation to mean absolute
    score. This confidence check depends on the baseline, whereas Pearson's
    correlation objective is invariant to positive scaling and baseline changes.
    Set the variation threshold to zero to reject only numerically flat scores.
    """

    shift_min: float = -30.0
    shift_max: float = 30.0
    shift_step: float = 0.5
    min_correlation: float = 0.90
    min_peak_margin: float = 0.05
    boundary_margin: float = 1.0
    min_relative_variation: float = 0.01

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _finite_number(value, name)
        span = self.shift_max - self.shift_min
        if span <= 0 or not 0 < self.shift_step <= span / 2:
            raise ValueError("Shift bounds must increase and include at least three grid points.")
        if span / self.shift_step > 100_000:
            raise ValueError("Shift grid must contain at most 100001 regularly spaced points.")
        if not -1 <= self.min_correlation <= 1 or not 0 <= self.min_peak_margin <= 2:
            raise ValueError("Correlation threshold must be in [-1, 1]; peak margin in [0, 2].")
        if not 0 <= self.boundary_margin < span / 2 or self.min_relative_variation < 0:
            raise ValueError("Boundary margin must be in [0, half-span); variation must be >= 0.")


@dataclass(frozen=True)
class FocusShiftResult:
    """Diagnostic candidate and accepted focal Z; rejected results have no focal Z.

    ``peak_margin`` compares distinct local maxima, including endpoint maxima.
    ``peak_width`` is the contiguous width within 0.01 of the best correlation.
    Unsupported or locally flat reference candidates have NaN correlations.
    """

    candidate_shift: float | None
    focus_z: float | None
    peak_correlation: float | None
    peak_margin: float | None
    peak_width: float | None
    rejection_reasons: tuple[str, ...]
    shifts: NDArray[np.float64]
    correlations: NDArray[np.float64]

    @property
    def accepted(self) -> bool:
        return self.focus_z is not None


def estimate_focus_shift(
    reference: ReferenceProfile,
    z_coordinates: ArrayLike,
    focus_scores: ArrayLike,
    *,
    current_z: float,
    config: ShiftSearchConfig | None = None,
) -> FocusShiftResult:
    """Estimate delta from ``S(z) ~ a F(z-current_z-delta) + b``, for positive a.

    Supply at least five distinct sample Zs. This only calculates a result: it
    never moves a stage. An accepted focal Z equals ``current_z + delta``.
    Every candidate uses all measurements, with no extrapolation or padding.
    """
    config = ShiftSearchConfig() if config is None else config
    z = _vector(z_coordinates, "z_coordinates", 5)
    scores = _vector(focus_scores, "focus_scores", 5)
    current_z = _finite_number(current_z, "current_z")
    if z.shape != scores.shape or np.unique(z).size != z.size:
        raise ValueError("Sample Zs and scores must have equal lengths and distinct Zs.")
    count = int(np.floor((config.shift_max - config.shift_min) / config.shift_step))
    shifts = config.shift_min + np.arange(count + 1) * config.shift_step
    if not np.isclose(shifts[-1], config.shift_max, rtol=0, atol=1e-12):
        shifts = np.append(shifts, config.shift_max)
    correlations = np.full(shifts.shape, np.nan)

    def rejected(reason: str) -> FocusShiftResult:
        return FocusShiftResult(None, None, None, None, None, (reason,), shifts, correlations)

    measured = _unit_variation(scores)
    if measured is None or np.std(scores) < config.min_relative_variation * np.mean(abs(scores)):
        return rejected("Insufficient measured score variation.")
    offsets = z - current_z
    for index, shift in enumerate(shifts):
        positions = offsets - shift
        if positions.min() < reference.z_offsets[0] or positions.max() > reference.z_offsets[-1]:
            continue
        expected = _unit_variation(reference.evaluate(positions))
        if expected is not None:
            correlations[index] = np.clip(np.dot(measured, expected), -1, 1)
    valid = np.flatnonzero(np.isfinite(correlations))
    if valid.size < 3:
        return rejected("Insufficient valid shift candidates inside the reference range.")
    best = int(valid[np.argmax(correlations[valid])])
    peak = float(correlations[best])
    landscape = np.where(np.isfinite(correlations), correlations, -np.inf)
    # Padding makes maxima at the valid search endpoints visible as competitors.
    maxima = find_peaks(np.r_[-np.inf, landscape, -np.inf])[0] - 1
    competitors = maxima[(maxima != best) & np.isfinite(landscape[maxima])]
    margin = float(peak - landscape[competitors].max()) if competitors.size else None
    left = right = best
    while left > valid[0] and correlations[left - 1] >= peak - 0.01:
        left -= 1
    while right < valid[-1] and correlations[right + 1] >= peak - 0.01:
        right += 1
    reasons = []
    if peak < config.min_correlation:
        reasons.append("Correlation is below the acceptance threshold.")
    if np.ptp(correlations[valid]) <= 1e-6:
        reasons.append("Correlation landscape is flat; shift is not identifiable.")
    if margin is not None and margin < config.min_peak_margin:
        reasons.append("Competing correlation peaks are ambiguous.")
    if (
        min(shifts[best] - shifts[valid[0]], shifts[valid[-1]] - shifts[best])
        <= config.boundary_margin
    ):
        reasons.append("Best shift is at or near the valid search boundary.")
    shift = float(shifts[best])
    return FocusShiftResult(
        shift,
        None if reasons else current_z + shift,
        peak,
        margin,
        float(shifts[right] - shifts[left]),
        tuple(reasons),
        shifts,
        correlations,
    )


def plot_reference_profile(reference: ReferenceProfile, *, ax: Axes | None = None) -> Figure:
    """Plot measurements, fit and selected zero; return without displaying/blocking.

    Call ``matplotlib.pyplot.show(block=True)`` to inspect before matching, or
    save the returned figure. Plotting never starts registration or accepts a fit.
    """
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
    grid = np.linspace(reference.z_offsets[0], reference.z_offsets[-1], 501)
    ax.plot(reference.z_offsets, reference.focus_scores, "o", label="Measured scores")
    ax.plot(grid, reference.evaluate(grid), label="Cubic smoothing spline")
    ax.axvline(0, color="black", linestyle="--", label="Manually selected focus")
    smoothing = "automatic (GCV)" if reference.smoothing is None else str(reference.smoothing)
    ax.set(
        xlabel="Z relative to selected focus (µm)",
        ylabel="Focus score",
        title=f"Reference profile — smoothing: {smoothing}",
    )
    ax.legend()
    return ax.figure

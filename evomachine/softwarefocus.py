"""Unified acquisition-backed maximum and reference-profile software focus."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
import threading
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, make_smoothing_spline
from scipy.signal import find_peaks
from delta.utils import CroppingBox

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.coordinates import Coordinate
from evomachine.frame import Frame, FrameMetaData
from evomachine.types import FocusCurveType, FocusScoringType, FocusSelectionType, FocusStatusType, LEDType, UNKNOWN_FOV_ID
from evomachine.bindings.software_focus.software_focus_algorithms import (
    create_focus_scorer,
)


@dataclass(frozen=True)
class FocusScoringConfig:
    """Base configuration for a focus image scorer."""

    def kwargs(self) -> dict[str, Any]:
        """Return constructor keywords for the configured scorer."""
        return asdict(self)


@dataclass(frozen=True)
class LaplacianVarianceScoringConfig(FocusScoringConfig):
    """Configuration for Laplacian-variance scoring, which has no parameters."""


@dataclass(frozen=True)
class SquaredGradientScoringConfig(FocusScoringConfig):
    """Configuration for mean-squared horizontal-gradient scoring."""

    threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the optional gradient threshold through the scorer."""
        create_focus_scorer(FocusScoringType.SQUARED_GRAD_AVG, **self.kwargs())


@dataclass(frozen=True)
class SteelScoringConfig(FocusScoringConfig):
    """Configuration for shifted row/column Steel scoring."""

    rowshift: int = 25
    colshift: int = 50
    normalise: bool = False

    def __post_init__(self) -> None:
        """Validate pixel shifts and normalization through the scorer."""
        create_focus_scorer(FocusScoringType.STEEL, **self.kwargs())


@dataclass(frozen=True)
class BandpassFFTScoringConfig(FocusScoringConfig):
    """Configuration for cell-scale FFT bandpass scoring."""

    cell_radius: float = 20.0
    downsample: int = 4
    order: int = 4
    saturation: float | None = None
    normalize_brightness: bool = True

    def __post_init__(self) -> None:
        """Validate FFT scoring parameters through the scorer."""
        create_focus_scorer(FocusScoringType.BANDPASS_FFT, **self.kwargs())


SCORING_CONFIG_TYPES: dict[FocusScoringType, type[FocusScoringConfig]] = {
    FocusScoringType.LAPLACIAN_VAR: LaplacianVarianceScoringConfig,
    FocusScoringType.SQUARED_GRAD_AVG: SquaredGradientScoringConfig,
    FocusScoringType.STEEL: SteelScoringConfig,
    FocusScoringType.BANDPASS_FFT: BandpassFFTScoringConfig,
}


@dataclass(frozen=True)
class FocusSelectionConfig:
    """Base configuration for selecting a focal Z from focus scores."""


@dataclass(frozen=True)
class MaximumFocusSelectionConfig(FocusSelectionConfig):
    """Regular Z scan used to select the unique internal score maximum."""

    rel_range: float = 50.0
    step_size: float = 5.0

    def __post_init__(self) -> None:
        """Validate the maximum-focus scan range and spacing."""
        if not 0 < self.rel_range < 2000:
            raise ValueError("rel_range must lie in (0, 2000) µm.")
        if not 0 < self.step_size <= self.rel_range:
            raise ValueError("step_size must lie in (0, rel_range].")


class FocusCancelled(RuntimeError):
    """A stopped scan must not produce a focus correction."""


class FocusOutOfRange(ValueError):
    """A complete scan would exceed the stage's coordinate bounds."""


def _finite_number(value: float, name: str) -> float:
    """Return a finite real as a float, or reject an invalid scalar.

    Args:
        value: Scalar to validate.
        name: Human-readable field name used in validation errors.

    Returns:
        The validated value converted to ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _vector(values: ArrayLike, name: str, minimum_size: int = 1) -> NDArray[np.float64]:
    """Return a copied, finite, one-dimensional float vector.

    Args:
        values: Array-like values to validate.
        name: Human-readable field name used in validation errors.
        minimum_size: Smallest accepted number of values.

    Returns:
        A validated one-dimensional ``float64`` array.
    """
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values.")
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size < minimum_size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1D array with at least {minimum_size} values.")
    return result


def _unit_variation(values: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Normalise centred variation for correlation without numerical overflow.

    Args:
        values: Finite one-dimensional values.

    Returns:
        Unit-length centred variation, or ``None`` when effectively flat.
    """
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return None
    centred = values / scale - np.mean(values / scale)
    norm = float(np.linalg.norm(centred))
    return None if norm <= 1e-12 else centred / norm


@dataclass(frozen=True)
class ReferenceProfile:
    """Measured scores and fitted spline relative to a selected focal Z.

    Args:
        z_offsets: Measured Z positions relative to ``focus_z`` in micrometres.
        focus_scores: Focus scores measured at ``z_offsets``.
        focus_z: Absolute manually selected focal Z in micrometres.
        smoothing: SciPy smoothing lambda, or ``None`` for GCV.
        _spline: Fitted non-extrapolating cubic smoothing spline.
    """

    z_offsets: NDArray[np.float64]
    focus_scores: NDArray[np.float64]
    focus_z: float
    smoothing: float | None
    _spline: BSpline = field(repr=False, compare=False)

    def evaluate(self, z_offsets: ArrayLike) -> NDArray[np.float64]:
        """Evaluate finite relative Z positions inside the measured domain.

        Args:
            z_offsets: Relative Z positions in micrometres.

        Returns:
            Spline focus scores at the requested positions.
        """
        offsets = np.asarray(z_offsets, dtype=np.float64)
        if not np.all(np.isfinite(offsets)):
            raise ValueError("Reference evaluation requires finite Z offsets.")
        if np.any(offsets < self.z_offsets[0]) or np.any(offsets > self.z_offsets[-1]):
            raise ValueError("Z offsets lie outside the measured reference range.")
        return np.asarray(self._spline(offsets), dtype=np.float64)

@dataclass(frozen=True)
class ReferenceFocusSelectionConfig(FocusSelectionConfig):
    """Reference acquisition, matching, and recovery settings.

    Calibration always measures one regular, symmetric reference grid, so its
    positions are described compactly by ``calibration_range`` and
    ``calibration_step``. Routine and recovery matching may instead use sparse,
    irregular, or mixed-density grids. Their fully resolved positions are
    therefore stored explicitly in ``sample_offsets`` and ``recovery_offsets``.

    Matching never extrapolates the fitted reference. The calibration half-range
    should cover the largest acquisition offset plus the largest candidate shift.
    With the defaults this is ``40 µm = 20 µm + 20 µm``.

    Args:
        calibration_range: Half-range of calibration scans in micrometres.
        calibration_step: Calibration image spacing in micrometres.
        sample_offsets: Sparse scan positions relative to the current Z.
        recovery_offsets: Recovery scan positions relative to the current Z.
        sparse_*: Numerical search bounds and acceptance thresholds for the
            sparse scan.
        recovery_*: Independent equivalents used for the recovery scan.
        smoothing: Spline smoothing lambda, or ``None`` for GCV.
        allow_peak_reset: Whether a failed recovery may rebuild at a new peak.
        share_reference_across_fovs: Whether all FoVs use one measured curve.
    """

    # Reference calibration acquisition
    calibration_range: float = 40.0
    """Calibration half-range in µm; 40 acquires from -40 to +40 around focus."""
    calibration_step: float = 1.0
    """Regular image spacing in µm used only to measure the reference profile."""

    # Sparse and recovery image acquisition
    sample_offsets: tuple[float, ...] = field(
        default_factory=lambda: ReferenceFocusSelectionConfig.sampling_offsets()
    )
    """Explicit sparse-scan positions in µm relative to the current stage Z."""
    recovery_offsets: tuple[float, ...] = field(
        default_factory=lambda: ReferenceFocusSelectionConfig.sampling_offsets(central_range=5.0)
    )
    """Explicit recovery positions, normally denser near the current stage Z."""

    # Sparse-scan numerical matching
    sparse_shift_min: float = -20.0
    """Lowest candidate sparse-scan correction in µm."""
    sparse_shift_max: float = 20.0
    """Highest candidate sparse-scan correction in µm."""
    sparse_shift_step: float = 0.1
    """Numerical sparse-search resolution in µm; this acquires no extra images."""
    sparse_min_correlation: float = 0.90
    """Minimum accepted Pearson correlation for sparse matching."""
    sparse_min_peak_margin: float = 0.05
    """Minimum separation between the best and competing sparse-match peaks."""
    sparse_boundary_margin: float = 1.0
    """Distance in µm inside the usable sparse-search boundary required for acceptance."""
    sparse_min_relative_variation: float = 0.01
    """Minimum measured score variation relative to its mean absolute score."""

    # Recovery-scan numerical matching
    recovery_shift_min: float = -20.0
    """Lowest candidate recovery-scan correction in µm."""
    recovery_shift_max: float = 20.0
    """Highest candidate recovery-scan correction in µm."""
    recovery_shift_step: float = 0.1
    """Numerical recovery-search resolution in µm; it does not acquire images."""
    recovery_min_correlation: float = 0.90
    """Minimum accepted Pearson correlation for recovery matching."""
    recovery_min_peak_margin: float = 0.05
    """Minimum separation between the best and competing recovery peaks."""
    recovery_boundary_margin: float = 1.0
    """Distance in µm inside the recovery-search boundary required for acceptance."""
    recovery_min_relative_variation: float = 0.01
    """Minimum recovery-score variation relative to its mean absolute score."""

    # Reference fitting
    smoothing: float | None = 0.0
    """Nonnegative spline smoothing lambda, or None for automatic GCV smoothing."""

    # Acquisition behavior and reference ownership
    allow_peak_reset: bool = False
    """Allow one failed recovery to rebuild the affected FoV reference at a new peak."""
    share_reference_across_fovs: bool = False
    """Reuse one measured curve across FoVs instead of acquiring one curve per FoV."""

    @staticmethod
    def sampling_offsets(
        half_range: float = 20.0,
        step: float = 5.0,
        central_range: float | None = None,
        central_step: float = 0.5,
    ) -> tuple[float, ...]:
        """Generate symmetric increasing acquisition offsets.

        Args:
            half_range: Outermost absolute Z offset in micrometres.
            step: Spacing of the outer acquisition grid in micrometres.
            central_range: Optional half-range of a denser central grid.
            central_step: Spacing of the central grid in micrometres.

        Returns:
            Unique increasing offsets containing zero and both endpoints.
        """
        values = (half_range, step, central_step)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Sampling range and steps must be finite and greater than zero.")
        if central_range is not None and not 0 < central_range <= half_range:
            raise ValueError("The denser centre must fit inside the outer sampling range.")
        if central_range is not None and central_step >= step:
            raise ValueError("The central image step must be smaller than the outer image step.")
        if half_range / step > 2500 or (
            central_range is not None and central_range / central_step > 2500
        ):
            raise ValueError("Sampling would acquire too many images; increase the Z step.")
        positive = np.r_[np.arange(0, half_range, step), half_range]
        if central_range is not None:
            positive = np.r_[
                positive, np.arange(0, central_range, central_step), central_range
            ]
        offsets = np.unique(np.round(np.r_[-positive, positive], 9))
        offsets[offsets == 0] = 0.0
        if offsets.size < 5:
            raise ValueError("Use a smaller image step to acquire at least five Z positions.")
        return tuple(float(value) for value in offsets)

    def __post_init__(self) -> None:
        """Validate acquisition grids and both independent matching policies."""
        if not isinstance(self.allow_peak_reset, bool):
            raise TypeError("allow_peak_reset must be boolean.")
        if not isinstance(self.share_reference_across_fovs, bool):
            raise TypeError("share_reference_across_fovs must be boolean.")
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
        self.validate_matching_settings("sparse", self.matching_settings("sparse"))
        self.validate_matching_settings("recovery", self.matching_settings("recovery"))

    def matching_settings(self, prefix: str) -> dict[str, float]:
        """Return one prefixed matching policy with unprefixed field names.

        Args:
            prefix: Either ``"sparse"`` or ``"recovery"``.

        Returns:
            Matching parameters used internally by the numerical search.
        """
        if prefix not in {"sparse", "recovery"}:
            raise ValueError("Matching prefix must be 'sparse' or 'recovery'.")
        names = (
            "shift_min", "shift_max", "shift_step", "min_correlation",
            "min_peak_margin", "boundary_margin", "min_relative_variation",
        )
        return {name: getattr(self, f"{prefix}_{name}") for name in names}

    @staticmethod
    def validate_matching_settings(prefix: str, values: dict[str, float]) -> None:
        """Validate a sparse or recovery matching-policy mapping.

        Args:
            prefix: Human-readable policy name for validation errors.
            values: The seven unprefixed matching values.

        Returns:
            ``None`` after successful validation.
        """
        for name, value in values.items():
            _finite_number(value, f"{prefix}_{name}")
        span = values["shift_max"] - values["shift_min"]
        step = values["shift_step"]
        if span <= 0 or not 0 < step <= span / 2:
            raise ValueError(
                f"{prefix.capitalize()} shift minimum ({values['shift_min']:g}) must be below "
                f"maximum ({values['shift_max']:g}); step ({step:g}) must be positive and no "
                "larger than half that range."
            )
        if span / step > 100_000:
            raise ValueError("Shift grid must contain at most 100001 regularly spaced points.")
        if not -1 <= values["min_correlation"] <= 1 or not 0 <= values["min_peak_margin"] <= 2:
            raise ValueError("Correlation threshold must be in [-1, 1]; peak margin in [0, 2].")
        if not 0 <= values["boundary_margin"] < span / 2 or values["min_relative_variation"] < 0:
            raise ValueError("Boundary margin must be in [0, half-span); variation must be >= 0.")


@dataclass(frozen=True)
class SoftwareFocusConfig:
    """Shared acquisition, scoring, cropping, and Z-selection configuration.

    Args:
        focus_frames: One or more metadata records acquired at every sampled Z.
        acquisition_settings: Capture and saving behaviour used by every scan.
        scoring: Selected scoring type paired with its typed configuration.
        selection: Selected Z-selection policy paired with its typed configuration.
        cropping_boxes: Shared image-coordinate regions scored for every focus frame.
    """

    focus_frames: tuple[FrameMetaData, ...]
    acquisition_settings: FrameAcquisitionSettings = field(default_factory=FrameAcquisitionSettings)
    scoring: tuple[FocusScoringType, FocusScoringConfig] = field(
        default_factory=lambda: (
            FocusScoringType.STEEL,
            SteelScoringConfig(rowshift=25, colshift=0),
        )
    )
    selection: tuple[FocusSelectionType, FocusSelectionConfig] = field(
        default_factory=lambda: (
            FocusSelectionType.MAXIMUM,
            MaximumFocusSelectionConfig(),
        )
    )
    cropping_boxes: tuple[CroppingBox, ...] = ()

    def __post_init__(self) -> None:
        """Validate typed policy pairs and crop geometry."""
        if not self.focus_frames or not all(isinstance(frame, FrameMetaData) for frame in self.focus_frames):
            raise TypeError("focus_frames must be a non-empty tuple of FrameMetaData.")
        if not isinstance(self.acquisition_settings, FrameAcquisitionSettings):
            raise TypeError("acquisition_settings must be FrameAcquisitionSettings.")
        scoring_type, scoring_config = self.scoring
        expected_scoring = SCORING_CONFIG_TYPES.get(scoring_type)
        if expected_scoring is None or not isinstance(scoring_config, expected_scoring):
            expected_name = expected_scoring.__name__ if expected_scoring else "a supported scoring config"
            raise TypeError(f"{scoring_type!r} requires {expected_name}.")
        selection_type, selection_config = self.selection
        expected_selection = {
            FocusSelectionType.MAXIMUM: MaximumFocusSelectionConfig,
            FocusSelectionType.REFERENCE_PROFILE: ReferenceFocusSelectionConfig,
        }.get(selection_type)
        if expected_selection is None or not isinstance(selection_config, expected_selection):
            expected_name = expected_selection.__name__ if expected_selection else "a supported selection config"
            raise TypeError(f"{selection_type!r} requires {expected_name}.")
        if not isinstance(self.cropping_boxes, tuple) or not all(
            isinstance(box, CroppingBox) for box in self.cropping_boxes
        ):
            raise TypeError("cropping_boxes must be a tuple of CroppingBox objects.")
        for box in self.cropping_boxes:
            if min(box.xtl, box.ytl) < 0 or box.xbr <= box.xtl or box.ybr <= box.ytl:
                raise ValueError("Focus crops must be nonnegative rectangles with positive area.")

    def copy(self) -> SoftwareFocusConfig:
        """Return an independent configuration copy."""
        return deepcopy(self)


class SoftwareFocusConfigFactory:
    """Construct standard software-focus configurations."""

    @staticmethod
    def default_config() -> SoftwareFocusConfig:
        """Return the standard maximum-focus configuration."""
        return SoftwareFocusConfig(
            focus_frames=(FrameMetaData(
                frame_id=-1,
                leds={LEDType.LED_450_NM: 29},
                filter_wheel=None,
                exposure=200,
            ),),
            cropping_boxes=(CroppingBox(xtl=200, xbr=3000, ytl=300, ybr=2900),),
        )


@dataclass
class FocusScan:
    """Captured images, scores, selection, and optional matching diagnostics."""

    z_coordinates: np.ndarray
    focus_scores: np.ndarray
    frame: Frame
    centre_z: float
    frames: tuple[Frame, ...] = ()
    frames_per_z: int = 1
    display_stack: np.ndarray | None = None
    selected_z: float | None = None
    candidate_shift: float | None = None
    focus_z: float | None = None
    peak_correlation: float | None = None
    peak_margin: float | None = None
    peak_width: float | None = None
    rejection_reasons: tuple[str, ...] = ()
    shifts: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    correlations: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    @property
    def accepted(self) -> bool:
        """Return whether matching produced an accepted absolute focal Z."""
        return self.focus_z is not None


@dataclass
class SoftwareFocusResult:
    """Ordered focus scans and the resulting focus state."""

    scans: tuple[FocusScan, ...]
    focus_status: FocusStatusType
    previous_coordinate: Coordinate | None = None
    previous_image: np.ndarray | None = None
    best_coordinate: Coordinate | None = None
    best_frame: np.ndarray | None = None
    curve_status: FocusCurveType = FocusCurveType.UNKNOWN
    failure_reason: str | None = None
    reference_reset: bool = False
    correction_applied: bool = False

    @property
    def recovery_attempted(self) -> bool:
        """Return whether a recovery matching scan was acquired."""
        return len(self.scans) > 1

    @property
    def z_coordinates(self) -> np.ndarray:
        """Return Z coordinates from the final scan, or an empty vector."""
        return self.scans[-1].z_coordinates if self.scans else np.empty(0)

    @property
    def focus_scores(self) -> np.ndarray:
        """Return scores from the final scan, or an empty vector."""
        return self.scans[-1].focus_scores if self.scans else np.empty(0)

    @property
    def focus_stack(self) -> np.ndarray:
        """Return the first acquired stack from the final scan."""
        if not self.scans:
            return np.empty((0, 0, 0))
        scan = self.scans[-1]
        return scan.display_stack if scan.display_stack is not None else scan.frame.array


class SoftwareFocus:
    """Acquire and score Z scans using maximum or reference-profile selection.

    ``calibrate()`` returns a candidate for GUI review. Call
    ``accept_reference()`` after inspection, before autofocus.
    Scan/search edits retain compatible calibration. Every configured focus
    frame is acquired at each Z and its scores are averaged. Crops are shared
    across frames; an empty crop tuple scores each whole image.
    """

    def __init__(
        self, acquisition_manager: FrameAcquisitionManager, config: SoftwareFocusConfig
    ):
        """Create reference focus around a stage-backed acquisition manager.

        Args:
            acquisition_manager: Manager used to acquire Z stacks and move Z.
            config: Complete acquisition and matching configuration.
        """
        if acquisition_manager.stage is None:
            raise ValueError("Reference-profile focus requires a stage.")
        self.acquisition_manager = acquisition_manager
        self._stop_event = threading.Event()
        self._reference_by_fov: dict[int, SoftwareFocus] = {}
        self._fov_results: dict[int, SoftwareFocusResult] = {}
        self._fov_config: dict[int, SoftwareFocusConfig] = {}
        self.process_events = None
        self.update_config(config)
        self.default_config = config

    def initialise_fovs(
        self,
        fov_ids: list[int],
        fov_configs: dict[int, SoftwareFocusConfig] | None = None,
    ) -> None:
        """Register per-FoV state without moving hardware or acquiring images.

        Args:
            fov_ids: Integer FoV identifiers to register.
            fov_configs: Optional per-FoV configuration overrides.
        """
        if not isinstance(fov_ids, list) or not all(
            isinstance(fov_id, int) and not isinstance(fov_id, bool) for fov_id in fov_ids
        ):
            raise TypeError("fov_ids must be a list of integers.")
        self._fov_results = {}
        self._fov_config = {}
        self.clear_fov_references()
        for fov_id in fov_ids:
            self._fov_results[fov_id] = SoftwareFocusResult((), FocusStatusType.UNKNOWN)
        for fov_id, config in (fov_configs or {}).items():
            if fov_id not in self._fov_results:
                raise KeyError(f"Unknown focus FoV ID {fov_id}.")
            self._fov_config[fov_id] = config

    def get_fov_result(self, fov_id: int) -> SoftwareFocusResult:
        """Return the latest registered result for one FoV."""
        if fov_id not in self._fov_results:
            raise KeyError(f"Unknown focus FoV ID {fov_id}.")
        return self._fov_results[fov_id]

    def score_image(
        self,
        img: np.ndarray,
        config: SoftwareFocusConfig | None = None,
        cropping_box: CroppingBox | list[CroppingBox] | tuple[CroppingBox, ...] | None = None,
    ) -> float:
        """Score one image using a configuration and optional explicit crops.

        Args:
            img: Two-dimensional image to score.
            config: Optional configuration; the active default is used when omitted.
            cropping_box: Optional one-or-many crop override.

        Returns:
            Whole-image score or mean score across the selected crops.
        """
        active = self._config if config is None else config
        scoring_type, scoring_config = active.scoring
        scorer = create_focus_scorer(scoring_type, **scoring_config.kwargs())
        if cropping_box is None:
            boxes = active.cropping_boxes
        elif isinstance(cropping_box, CroppingBox):
            boxes = (cropping_box,)
        else:
            boxes = tuple(cropping_box)
        if not boxes:
            return scorer.score_image(img)
        return float(np.mean([scorer.score_image(box.crop(img)) for box in boxes]))

    @property
    def config(self) -> SoftwareFocusConfig:
        """Return a snapshot; use update_config to preserve compatible calibration."""
        return deepcopy(self._config)

    @property
    def reference_config(self) -> ReferenceFocusSelectionConfig:
        """Return the active reference-selection settings or reject maximum mode."""
        selection_type, selection_config = self._config.selection
        if selection_type is not FocusSelectionType.REFERENCE_PROFILE:
            raise RuntimeError("Reference-profile operation requires REFERENCE_PROFILE selection.")
        if not isinstance(selection_config, ReferenceFocusSelectionConfig):
            raise TypeError("Reference-profile selection has an incompatible configuration.")
        return selection_config

    def reference_for_fov(
        self,
        fov_id: int,
        config: SoftwareFocusConfig | None = None,
    ) -> SoftwareFocus:
        """Return the shared reference or an owned reference for one FoV.

        Args:
            fov_id: Registered field-of-view identifier.
            config: Optional FoV-specific configuration. The owner's current
                configuration is copied when omitted.

        Returns:
            The reference-focus instance responsible for the requested FoV.
        """
        if not isinstance(fov_id, int) or isinstance(fov_id, bool):
            raise TypeError("Reference FoV ID must be an integer.")
        if fov_id in self._reference_by_fov:
            return self._reference_by_fov[fov_id]
        if (
            self._config.selection[0] is FocusSelectionType.REFERENCE_PROFILE
            and self.reference_config.share_reference_across_fovs
        ):
            return self
        if fov_id not in self._reference_by_fov:
            child = SoftwareFocus(
                self.acquisition_manager,
                self._config if config is None else config,
            )
            child.process_events = self.process_events
            self._reference_by_fov[fov_id] = child
        return self._reference_by_fov[fov_id]

    def set_reference_for_fov(self, fov_id: int, focus: SoftwareFocus) -> None:
        """Store an explicitly local reference-focus instance for one FoV.

        Args:
            fov_id: Registered field-of-view identifier.
            focus: Local reference-focus instance to retain.

        Returns:
            ``None`` after replacing the owned instance.
        """
        if not isinstance(focus, SoftwareFocus):
            raise TypeError("FoV reference must be a SoftwareFocus instance.")
        self._reference_by_fov[fov_id] = focus

    def remove_reference_for_fov(self, fov_id: int) -> None:
        """Discard any owned reference-focus instance for one FoV.

        Args:
            fov_id: Field-of-view identifier to remove.

        Returns:
            ``None`` whether or not the FoV had an owned instance.
        """
        self._reference_by_fov.pop(fov_id, None)

    def clear_fov_references(self) -> None:
        """Discard all measured FoV-specific reference-focus instances."""
        self._reference_by_fov.clear()

    def iter_fov_references(self) -> tuple[SoftwareFocus, ...]:
        """Return a stable snapshot of all owned FoV-specific instances."""
        return tuple(self._reference_by_fov.values())

    @staticmethod
    def _score_focus_stack(
        images: ArrayLike,
        *,
        scoring: tuple[FocusScoringType, FocusScoringConfig],
        cropping_boxes: tuple[CroppingBox, ...] = (),
    ) -> NDArray[np.float64]:
        """Score every whole image in a Z-first stack.

        Args:
            images: Numeric array with shape ``(planes, height, width)``.
            scoring: Scoring type paired with its typed configuration.
            cropping_boxes: Shared image regions; empty scores each whole image.

        Returns:
            One finite focus score per plane.
        """
        stack = np.asarray(images)
        if stack.ndim != 3 or stack.shape[0] == 0 or min(stack.shape[1:]) < 3:
            raise ValueError(
                "images must have shape (planes, height, width), with height/width >= 3."
            )
        if not np.issubdtype(stack.dtype, np.number) or np.iscomplexobj(stack):
            raise ValueError("images must contain real numeric values.")
        scoring_type, scoring_config = scoring
        scorer = create_focus_scorer(scoring_type, **scoring_config.kwargs())
        scores = []
        for image in stack:
            image = image.astype(np.float64)
            if not np.all(np.isfinite(image)):
                raise ValueError("images must contain finite values.")
            scores.append(
                float(np.mean([scorer.score_image(box.crop(image)) for box in cropping_boxes]))
                if cropping_boxes
                else scorer.score_image(image)
            )
        return _vector(scores, "focus scores")

    @classmethod
    def _score_acquired_stack(
        cls,
        images: np.ndarray,
        frames_per_z: int,
        config: SoftwareFocusConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average scorer outputs and images across focus frames at each Z.

        Args:
            images: Acquisition-order stack with focus frames interleaved per Z.
            frames_per_z: Number of configured frame metadata records.
            config: Shared scoring and crop configuration.

        Returns:
            Per-Z scores and a Z-first mean image stack.
        """
        if frames_per_z < 1 or images.shape[0] % frames_per_z:
            raise ValueError("Acquired focus stack is not divisible by frames_per_z.")
        grouped = images.reshape((-1, frames_per_z, *images.shape[1:]))
        scores = [
            np.mean([
                cls._score_focus_stack(
                    image[np.newaxis, ...],
                    scoring=config.scoring,
                    cropping_boxes=config.cropping_boxes,
                )[0]
                for image in group
            ])
            for group in grouped
        ]
        return np.asarray(scores, dtype=float), np.mean(grouped.astype(float), axis=1)

    @staticmethod
    def _fit_reference_profile(
        z_coordinates: ArrayLike,
        focus_scores: ArrayLike,
        *,
        focus_z: float,
        smoothing: float | None = None,
    ) -> ReferenceProfile:
        """Fit a non-extrapolating cubic spline around the selected focal Z.

        Args:
            z_coordinates: At least five increasing absolute Z positions.
            focus_scores: Focus scores corresponding to the Z positions.
            focus_z: Absolute Z used as relative zero.
            smoothing: Nonnegative SciPy lambda, or ``None`` for GCV.

        Returns:
            An immutable fitted reference profile.
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

    @staticmethod
    def _match_scan(
        reference: ReferenceProfile,
        scan: FocusScan,
        *,
        current_z: float,
        settings: dict[str, float],
    ) -> FocusScan:
        """Populate a scan with Pearson shift-search diagnostics.

        Args:
            reference: Accepted calibration profile.
            scan: Acquired scan to match and update in place.
            current_z: Absolute scan centre in micrometres.
            settings: Validated unprefixed matching settings.

        Returns:
            The same scan with candidate, acceptance, and landscape fields set.
        """
        z = _vector(scan.z_coordinates, "z_coordinates", 5)
        scores = _vector(scan.focus_scores, "focus_scores", 5)
        current_z = _finite_number(current_z, "current_z")
        if z.shape != scores.shape or np.unique(z).size != z.size:
            raise ValueError("Sample Zs and scores must have equal lengths and distinct Zs.")
        count = int(np.floor((settings["shift_max"] - settings["shift_min"]) / settings["shift_step"]))
        shifts = settings["shift_min"] + np.arange(count + 1) * settings["shift_step"]
        if not np.isclose(shifts[-1], settings["shift_max"], rtol=0, atol=1e-12):
            shifts = np.append(shifts, settings["shift_max"])
        correlations = np.full(shifts.shape, np.nan)

        def reject(reason: str) -> FocusScan:
            """Store an early rejection while retaining the evaluated grid."""
            scan.rejection_reasons = (reason,)
            scan.shifts = shifts
            scan.correlations = correlations
            return scan

        measured = _unit_variation(scores)
        if measured is None:
            return reject("Insufficient measured score variation.")
        low_variation = np.std(scores) < settings["min_relative_variation"] * np.mean(abs(scores))
        offsets = z - current_z
        in_range = 0
        for index, shift in enumerate(shifts):
            positions = offsets - shift
            if positions.min() < reference.z_offsets[0] or positions.max() > reference.z_offsets[-1]:
                continue
            in_range += 1
            expected = _unit_variation(reference.evaluate(positions))
            if expected is not None:
                correlations[index] = np.clip(np.dot(measured, expected), -1, 1)
        valid = np.flatnonzero(np.isfinite(correlations))
        if valid.size < 3:
            hint = (
                "Widen reference calibration or reduce the image-sampling range."
                if in_range < 3
                else "Reference scores are flat at these sample positions."
            )
            return reject(
                f"Only {valid.size} usable shift candidates ({in_range} within the reference); "
                "at least 3 are needed. "
                f"Reference: {reference.z_offsets[0]:g} to {reference.z_offsets[-1]:g} µm; "
                f"sample offsets: {offsets.min():g} to {offsets.max():g} µm. {hint}"
            )
        best = int(valid[np.argmax(correlations[valid])])
        peak = float(correlations[best])
        landscape = np.where(np.isfinite(correlations), correlations, -np.inf)
        maxima = find_peaks(np.r_[-np.inf, landscape, -np.inf])[0] - 1
        competitors = maxima[(maxima != best) & np.isfinite(landscape[maxima])]
        margin = float(peak - landscape[competitors].max()) if competitors.size else None
        left = right = best
        while left > valid[0] and correlations[left - 1] >= peak - 0.01:
            left -= 1
        while right < valid[-1] and correlations[right + 1] >= peak - 0.01:
            right += 1
        reasons = ["Insufficient measured score variation."] if low_variation else []
        if peak < settings["min_correlation"]:
            reasons.append(
                f"Best correlation {peak:.3f} is below the required "
                f"{settings['min_correlation']:.3f}."
            )
        if np.ptp(correlations[valid]) <= 1e-6:
            reasons.append("Correlation landscape is flat; shift is not identifiable.")
        if margin is not None and margin < settings["min_peak_margin"]:
            reasons.append(
                f"Competing peaks are ambiguous: margin {margin:.3f}, required "
                f"{settings['min_peak_margin']:.3f}."
            )
        if min(shifts[best] - shifts[valid[0]], shifts[valid[-1]] - shifts[best]) <= settings[
            "boundary_margin"
        ]:
            reasons.append(
                f"Best shift {shifts[best]:g} µm is near the usable search boundary "
                f"({shifts[valid[0]]:g} to {shifts[valid[-1]]:g} µm)."
            )
        shift = float(shifts[best])
        scan.candidate_shift = shift
        scan.focus_z = None if reasons else current_z + shift
        scan.peak_correlation = peak
        scan.peak_margin = margin
        scan.peak_width = float(shifts[right] - shifts[left])
        scan.rejection_reasons = tuple(reasons)
        scan.shifts = shifts
        scan.correlations = correlations
        return scan

    def update_config(self, config: SoftwareFocusConfig, fov_id: int | None = None) -> None:
        """Keep references compatible with the changed settings.

        Scan/search, saving and future calibration range/step edits retain the
        accepted profile. Smoothing/metric changes refit cached images into a
        candidate requiring acceptance. Imaging changes retain the old curve
        for inspection but require a new capture before it can be accepted.
        Compare resolved scorer parameters so omitted defaults do not cause
        a refit when configuration is round-tripped through the GUI.
        """
        if not isinstance(config, SoftwareFocusConfig):
            raise TypeError("config must be SoftwareFocusConfig.")
        image_config = getattr(self.acquisition_manager.camera, "image", None)
        image_shape = getattr(image_config, "shape", None)
        for box in config.cropping_boxes:
            if image_shape is not None and (box.xbr > image_shape[1] or box.ybr > image_shape[0]):
                raise ValueError(
                    f"Focus crop ({box.xtl}, {box.ytl}, {box.xbr}, {box.ybr}) exceeds "
                    f"the camera image bounds {image_shape[1]}×{image_shape[0]}."
                )
        if fov_id is not None:
            if fov_id not in self._fov_results:
                raise KeyError(f"Unknown focus FoV ID {fov_id}.")
            self._fov_config[fov_id] = config
            local_focus = self._reference_by_fov.get(fov_id)
            if local_focus is not None:
                local_focus.update_config(config)
            return
        previous = getattr(self, "_config", None)
        if previous is not None:
            scoring_changed = previous.scoring != config.scoring
            capture_changed = (
                previous.focus_frames != config.focus_frames
                or previous.acquisition_settings != config.acquisition_settings
                or previous.cropping_boxes != config.cropping_boxes
            )
            profile = self.pending_reference or self.reference
            old_reference = previous.selection[1] if previous.selection[0] is FocusSelectionType.REFERENCE_PROFILE else None
            new_reference = config.selection[1] if config.selection[0] is FocusSelectionType.REFERENCE_PROFILE else None
            sharing_enabled = (
                isinstance(new_reference, ReferenceFocusSelectionConfig)
                and new_reference.share_reference_across_fovs
                and (
                    not isinstance(old_reference, ReferenceFocusSelectionConfig)
                    or not old_reference.share_reference_across_fovs
                )
            )
            if sharing_enabled:
                self.clear_fov_references()
            smoothing_changed = (
                isinstance(old_reference, ReferenceFocusSelectionConfig)
                and isinstance(new_reference, ReferenceFocusSelectionConfig)
                and old_reference.smoothing != new_reference.smoothing
            )
            if profile is not None and smoothing_changed and not scoring_changed and not capture_changed:
                scores, _ = self._score_acquired_stack(
                    self.calibration_scan.frame.array,
                    self.calibration_scan.frames_per_z,
                    config,
                )
                fitted = self._fit_reference_profile(
                    self.calibration_scan.z_coordinates,
                    scores,
                    focus_z=profile.focus_z,
                    smoothing=new_reference.smoothing,
                )
                self.calibration_scan.focus_scores = scores
                self.pending_reference, self.reference = fitted, None
                self._stop_event = threading.Event()
                self.last_result = None
                self.reference_message = (
                    "Reference refitted from cached images. Inspect and accept it again."
                )
            elif profile is not None and (capture_changed or scoring_changed):
                self.clear_reference()
                self.reference_message = "Scoring or imaging settings changed; acquire a new reference."
            elif profile is not None and not self.needs_calibration:
                self.reference_message = (
                    "Reference retained. Updated settings apply to the next scan."
                )
        self._config = deepcopy(config)
        self.default_config = config
        for local_fov_id, local_focus in self._reference_by_fov.items():
            if local_fov_id not in self._fov_config:
                local_focus.update_config(config)
        if previous is None:
            self.clear_reference()

    def clear_reference(self) -> None:
        """Discard accepted, pending, calibration, and result state."""
        self.reference: ReferenceProfile | None = None
        self.pending_reference: ReferenceProfile | None = None
        self.calibration_scan: FocusScan | None = None
        self.last_result: SoftwareFocusResult | None = None
        self.needs_calibration = False
        self.reference_message: str | None = None

    def stop(self) -> None:
        """Request cancellation at the next safe acquisition boundary."""
        self._stop_event.set()
        for focus in self._reference_by_fov.values():
            focus.stop()

    def _check_cancelled(self) -> None:
        """Raise when the active scan has been cancelled."""
        if self._stop_event.is_set():
            raise FocusCancelled("Software focus cancelled.")

    def _begin(self, stop_event: threading.Event | None) -> float:
        """Start an operation and return its finite current stage Z."""
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
        config = self.reference_config
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
        self.pending_reference = self._fit_reference_profile(
            self.calibration_scan.z_coordinates,
            self.calibration_scan.focus_scores,
            focus_z=centre,
            smoothing=config.smoothing,
        )
        self._check_cancelled()
        return self.pending_reference

    def accept_reference(self) -> ReferenceProfile:
        """Accept the reviewed pending profile for subsequent matching."""
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
        """Refit cached calibration scores and require renewed acceptance.

        Args:
            smoothing: Nonnegative spline lambda, or ``None`` for GCV.

        Returns:
            The pending refitted profile.
        """
        if self.needs_calibration:
            raise RuntimeError(
                "Capture a new reference with the current imaging settings before refitting."
            )
        if self.calibration_scan is None:
            raise RuntimeError("Capture a reference before refitting its spline.")
        previous = self.pending_reference or self.reference
        profile = self._fit_reference_profile(
            self.calibration_scan.z_coordinates,
            self.calibration_scan.focus_scores,
            focus_z=previous.focus_z,
            smoothing=smoothing,
        )
        selection = replace(self.reference_config, smoothing=smoothing)
        self._config = replace(
            self._config,
            selection=(FocusSelectionType.REFERENCE_PROFILE, selection),
        )
        self._stop_event = threading.Event()
        self.last_result = None
        self.pending_reference = profile
        self.reference = None
        self.reference_message = "Spline refitted. Inspect and accept the reference again."
        return profile

    def _scan(self, centre: float, offsets, fov_id: int) -> FocusScan:
        """Acquire and score one bounds-checked Z scan.

        Args:
            centre: Absolute centre Z in micrometres.
            offsets: Relative acquisition positions in micrometres.
            fov_id: FoV identifier stored in acquired frame metadata.

        Returns:
            The acquired scan before matching diagnostics are populated.
        """
        self._check_cancelled()
        z = centre + np.asarray(offsets, dtype=float)
        targets = [Coordinate(None, None, float(value)) for value in z]
        stage = self.acquisition_manager.stage
        out_of_bounds = getattr(stage, "coordinate_is_out_of_bounds", None)
        if callable(out_of_bounds) and any(out_of_bounds(target) for target in targets):
            raise FocusOutOfRange("Software focus scan exceeds the stage Z bounds.")
        config = self._config
        try:
            frame = self.acquisition_manager.take_z_stack(
                frame_metadata=[replace(metadata, fov_id=fov_id) for metadata in config.focus_frames],
                z_coordinates=targets,
                settings=config.acquisition_settings,
                stop_event=self._stop_event,
                step_callback=self.process_events,
            )
        except RuntimeError:
            self._check_cancelled()
            raise
        self._check_cancelled()
        frames_per_z = len(config.focus_frames)
        if frame.array.shape[0] != z.size * frames_per_z:
            raise RuntimeError("Reference focus scan returned an incomplete image stack.")
        scores, display_stack = self._score_acquired_stack(frame.array, frames_per_z, config)
        return FocusScan(
            z, scores, frame, centre,
            frames=(frame,), frames_per_z=frames_per_z, display_stack=display_stack,
        )

    def run(
        self,
        fov_id: int | None = None,
        stop_event: threading.Event | None = None,
        *,
        apply: bool = True,
    ) -> SoftwareFocusResult:
        """Run the configured maximum or reference-profile focus policy.

        Args:
            fov_id: Optional registered FoV identifier; the stage FoV is used when omitted.
            stop_event: Optional cooperative cancellation event.
            apply: Whether an accepted focal Z should be applied to the stage.

        Returns:
            Unified result containing the ordered acquired scans.
        """
        resolved_fov = self.acquisition_manager.stage.get_fov_id() if fov_id is None else fov_id
        if resolved_fov in self._fov_results:
            child = self.reference_for_fov(resolved_fov, self._fov_config.get(resolved_fov))
            if child is not self:
                result = child.run(resolved_fov, stop_event, apply=apply)
                self._fov_results[resolved_fov] = result
                return result
        selection_type = self._config.selection[0]
        try:
            if selection_type is FocusSelectionType.MAXIMUM:
                result = self._run_maximum(resolved_fov, stop_event, apply=apply)
            else:
                result = self._run_reference(resolved_fov, stop_event, apply=apply)
        except FocusCancelled:
            result = SoftwareFocusResult((), FocusStatusType.UNKNOWN, failure_reason="Software focus cancelled.")
        self._fov_results[resolved_fov] = result
        return result

    def _run_maximum(
        self,
        fov_id: int,
        stop_event: threading.Event | None,
        *,
        apply: bool,
    ) -> SoftwareFocusResult:
        """Acquire a regular scan and select its unique internal score maximum."""
        selection = self._config.selection[1]
        if not isinstance(selection, MaximumFocusSelectionConfig):
            raise TypeError("MAXIMUM selection requires MaximumFocusSelectionConfig.")
        centre = self._begin(stop_event)
        previous = self.acquisition_manager.stage.get_coordinates(query_hardware=True)
        previous_frame = self.acquisition_manager.take_frame(
            frame_metadata=[replace(metadata, fov_id=fov_id) for metadata in self._config.focus_frames],
            settings=self._config.acquisition_settings,
        )
        previous_image = np.mean(previous_frame.array.astype(float), axis=0)
        offsets = np.arange(-selection.rel_range, selection.rel_range, selection.step_size)
        try:
            scan = self._scan(centre, offsets, fov_id)
        except FocusOutOfRange as error:
            return SoftwareFocusResult(
                scans=(),
                focus_status=FocusStatusType.OUT_OF_RANGE,
                previous_coordinate=previous,
                previous_image=previous_image,
                failure_reason=str(error),
            )
        curve_status = self._focus_curve_type(scan.focus_scores)
        accepted = curve_status is FocusCurveType.HAS_GLOBAL_MAXIMUM
        best_index = int(np.argmax(scan.focus_scores)) if accepted else None
        if best_index is not None:
            scan.selected_z = float(scan.z_coordinates[best_index])
            scan.focus_z = scan.selected_z
        target_z = scan.selected_z if accepted and apply else centre
        self.acquisition_manager.stage.move(Coordinate(None, None, target_z), block=True)
        best_coordinate = Coordinate(previous.x, previous.y, scan.selected_z) if accepted else None
        best_frame = scan.display_stack[best_index] if best_index is not None else None
        return SoftwareFocusResult(
            scans=(scan,),
            focus_status=FocusStatusType.IN_FOCUS if accepted else FocusStatusType.BAD_FOCUS_CURVE,
            previous_coordinate=previous,
            previous_image=previous_image,
            best_coordinate=best_coordinate,
            best_frame=best_frame,
            curve_status=curve_status,
            failure_reason=None if accepted else "Focus scores do not contain one internal maximum.",
            correction_applied=bool(accepted and apply),
        )

    @staticmethod
    def _focus_curve_type(scores: ArrayLike) -> FocusCurveType:
        """Classify whether a score curve has one safe internal maximum."""
        values = np.asarray(scores)
        if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
            return FocusCurveType.UNKNOWN
        maximum_indices = np.flatnonzero(values == np.max(values))
        if maximum_indices.size != 1:
            return FocusCurveType.HAS_MAXIMA
        if maximum_indices[0] in (0, values.size - 1):
            return FocusCurveType.HAS_BOUNDARY_MAXIMUM
        return FocusCurveType.HAS_GLOBAL_MAXIMUM

    def _run_reference(
        self, fov_id: int, stop_event: threading.Event | None = None, *, apply: bool = True
    ) -> SoftwareFocusResult:
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
        scans: list[FocusScan] = []
        config = self.reference_config
        for prefix, offsets in (
            ("sparse", config.sample_offsets),
            ("recovery", config.recovery_offsets),
        ):
            try:
                scan = self._scan(centre, offsets, fov_id)
            except FocusOutOfRange as error:
                self.last_result = SoftwareFocusResult(
                    scans=tuple(scans),
                    focus_status=FocusStatusType.OUT_OF_RANGE,
                    failure_reason=str(error),
                )
                return self.last_result
            scans.append(scan)
            self._match_scan(
                self.reference,
                scan,
                current_z=centre,
                settings=config.matching_settings(prefix),
            )
            self._check_cancelled()
            if scan.accepted:
                stage = self.acquisition_manager.stage
                target = Coordinate(None, None, scan.focus_z)
                if stage.coordinate_is_out_of_bounds(target):
                    self.last_result = SoftwareFocusResult(
                        scans=tuple(scans),
                        focus_status=FocusStatusType.OUT_OF_RANGE,
                        failure_reason="Estimated focal Z exceeds the stage bounds.",
                    )
                    return self.last_result
                if apply:
                    stage.move(target=target, block=True)
                self._check_cancelled()
                break
        accepted = scans[-1].accepted
        if not accepted and config.allow_peak_reset and apply:
            reset = self._reset_at_peak(centre, fov_id, scans)
            if reset is not None:
                self.last_result = reset
                return reset
        self.last_result = SoftwareFocusResult(
            scans=tuple(scans),
            focus_status=FocusStatusType.IN_FOCUS if accepted else FocusStatusType.BAD_FOCUS_CURVE,
            failure_reason=None if accepted else "; ".join(scans[-1].rejection_reasons),
            correction_applied=accepted and apply,
        )
        return self.last_result

    def _reset_at_peak(self, centre, fov_id, scans):
        """Try one opt-in peak reset without adding reset scans to match results.

        Args:
            centre: Absolute Z around which to locate a new peak.
            fov_id: FoV identifier stored in acquired images.
            scans: Existing sparse and recovery matching scans.

        Returns:
            A successful reset result, or ``None`` when no safe peak is found.
        """
        config = self.reference_config
        positive = np.r_[
            np.arange(0, config.calibration_range, config.calibration_step),
            config.calibration_range,
        ]
        offsets = np.r_[-positive[:0:-1], positive]
        try:
            locating = self._scan(centre, offsets, fov_id)
            scores = locating.focus_scores
            if not self._has_valid_peak(scores) or np.std(
                scores
            ) < config.sparse_min_relative_variation * np.mean(abs(scores)):
                return None
            peak_z = float(locating.z_coordinates[np.argmax(scores)])
            calibration = self._scan(peak_z, offsets, fov_id)
            profile = self._fit_reference_profile(
                calibration.z_coordinates,
                calibration.focus_scores,
                focus_z=peak_z,
                smoothing=config.smoothing,
            )
        except (FocusOutOfRange, ValueError):
            return None
        self._check_cancelled()
        self.acquisition_manager.stage.move(target=Coordinate(None, None, peak_z), block=True)
        self._check_cancelled()
        self.reference = profile
        self.pending_reference = None
        self.calibration_scan = calibration
        self.needs_calibration = False
        self.reference_message = None
        return SoftwareFocusResult(
            scans=tuple(scans),
            focus_status=FocusStatusType.IN_FOCUS,
            reference_reset=True,
            correction_applied=True,
        )

    @staticmethod
    def _has_valid_peak(scores: ArrayLike) -> bool:
        """Return whether scores have one internal global maximum.

        Args:
            scores: Ordered focus scores from a Z scan.

        Returns:
            ``True`` only for at least three scores with a unique maximum that
            is not at either scan boundary.
        """
        values = np.asarray(scores)
        if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
            return False
        maximum = np.max(values)
        indices = np.flatnonzero(values == maximum)
        return indices.size == 1 and 0 < indices[0] < values.size - 1

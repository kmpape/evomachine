from __future__ import annotations

import numpy as np
from delta.utils import CroppingBox

from evomachine.bindings.software_focus.software_focus_algorithms import (
    DEFAULT_SQUARED_GRAD_THRESHOLD,
    LaplacianVarianceFocusAlgorithm,
    SoftwareFocusAlgorithm,
    SquaredGradientAverageFocusAlgorithm,
    SteelFocusAlgorithm,
    create_software_focus_algorithm as _create_software_focus_algorithm,
)
from evomachine.config_types import SoftwareFocusConfig
from evomachine.types import FocusAlgorithmType, FocusCurveType


def get_focus_score_is_good(focus_curve: np.ndarray) -> bool:
    """
    Return whether a focus score curve has a single non-boundary maximum.

    Parameters
    ----------
    focus_curve
        Focus score values ordered by scanned Z position.

    Returns
    -------
    bool
        True when the curve is classified as having a global maximum.
    """
    return get_focus_curve_type(focus_curve=focus_curve) == FocusCurveType.HAS_GLOBAL_MAXIMUM


def get_focus_curve_type(focus_curve: np.ndarray) -> FocusCurveType:
    """
    Classify the shape of a focus score curve.

    Parameters
    ----------
    focus_curve
        Focus score values ordered by scanned Z position.

    Returns
    -------
    FocusCurveType
        Classification of the curve maximum pattern.
    """
    if focus_curve.size < 3:
        return FocusCurveType.UNKNOWN

    max_indices = np.where(focus_curve == np.max(focus_curve))[0]
    num_maxima = len(max_indices)

    if 0 in max_indices or len(focus_curve) - 1 in max_indices:
        return FocusCurveType.HAS_BOUNDARY_MAXIMUM
    if num_maxima == 1:
        return FocusCurveType.HAS_GLOBAL_MAXIMUM
    if num_maxima > 1:
        return FocusCurveType.HAS_MAXIMA
    return FocusCurveType.UNKNOWN


def create_software_focus_algorithm(
        algorithm: FocusAlgorithmType,
        config: SoftwareFocusConfig | None = None,
        threshold: float | None = None,
        rowshift: int | None = None,
        colshift: int | None = None,
        normalise_score: bool = False,
) -> SoftwareFocusAlgorithm:
    """
    Create a software focus algorithm using explicit values or config defaults.

    Parameters
    ----------
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Optional row shift for the Steel algorithm. If None, config or legacy
        defaults are used.
    colshift
        Optional column shift for the Steel algorithm. If None, config or
        legacy defaults are used.
    normalise_score
        If True, normalise the Steel score by image area.

    Returns
    -------
    SoftwareFocusAlgorithm
        Focus scoring algorithm instance.
    """
    if config is not None and not isinstance(config, SoftwareFocusConfig):
        raise TypeError(
            f"create_software_focus_algorithm: config must be SoftwareFocusConfig or None, received {type(config)}."
        )
    resolved_rowshift = rowshift if rowshift is not None else (config.rowshift_px if config is not None else 25)
    resolved_colshift = colshift if colshift is not None else (config.colshift_px if config is not None else 50)
    return _create_software_focus_algorithm(
        algorithm=algorithm,
        threshold=threshold,
        rowshift=resolved_rowshift,
        colshift=resolved_colshift,
        normalise=normalise_score,
    )


def get_roi_focus_score(
        img: np.ndarray,
        algorithm: FocusAlgorithmType,
        boxes: list[CroppingBox],
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        config: SoftwareFocusConfig | None = None,
) -> float:
    """
    Return a summed focus score for cropped regions of one image.

    Parameters
    ----------
    img
        Source image array containing all regions.
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    boxes
        CroppingBox objects selecting regions to score.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Row shift for the Steel algorithm.
    colshift
        Column shift for the Steel algorithm.
    normalise_score
        If True, normalise the Steel score by image area.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.

    Returns
    -------
    float
        Sum of focus scores across the provided regions.
    """
    scorer = create_software_focus_algorithm(
        algorithm=algorithm,
        config=config,
        threshold=threshold,
        rowshift=rowshift,
        colshift=colshift,
        normalise_score=normalise_score,
    )
    return scorer.score_rois(img=img, boxes=boxes)


def get_focus_score(
        img: np.ndarray,
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        config: SoftwareFocusConfig | None = None,
) -> float:
    """
    Return a focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Row shift for the Steel algorithm.
    colshift
        Column shift for the Steel algorithm.
    normalise_score
        If True, normalise the Steel score by image area.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.

    Returns
    -------
    float
        Focus score for the provided image.
    """
    scorer = create_software_focus_algorithm(
        algorithm=algorithm,
        config=config,
        threshold=threshold,
        rowshift=rowshift,
        colshift=colshift,
        normalise_score=normalise_score,
    )
    return scorer.score_image(img=img)


def get_focus_score_laplacian_var(img: np.ndarray) -> float:
    """
    Return a Laplacian-variance focus score for one image.

    Parameters
    ----------
    img
        Image array to score.

    Returns
    -------
    float
        Variance of the squared inner Laplacian image.
    """
    return LaplacianVarianceFocusAlgorithm().score_image(img=img)


def get_focus_score_squared_gradient(img: np.ndarray, threshold: float | None = None) -> float:
    """
    Return a squared-gradient focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    threshold
        Optional squared-gradient threshold.

    Returns
    -------
    float
        Mean squared horizontal gradient after thresholding.
    """
    return SquaredGradientAverageFocusAlgorithm(threshold=threshold).score_image(img=img)


def get_focus_score_steel(
        img: np.ndarray,
        rowshift: int,
        colshift: int,
        normalise: bool = False,
) -> float:
    """
    Return a Steel focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    rowshift
        Pixel shift along image rows.
    colshift
        Pixel shift along image columns.
    normalise
        If True, divide the score by twice the image area.

    Returns
    -------
    float
        Shifted-difference Steel focus score.
    """
    return SteelFocusAlgorithm(rowshift=rowshift, colshift=colshift, normalise=normalise).score_image(img=img)

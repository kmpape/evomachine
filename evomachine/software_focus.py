import cv2
import numpy as np

from delta.utils import CroppingBox

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.types import FocusAlgorithmType, FocusCurveType


DEFAULT_SQUARED_GRAD_THRESHOLD = 0

def get_focus_score_is_good(focus_curve: np.array) -> bool:  # noqa
    return get_focus_curve_type(focus_curve=focus_curve) == FocusCurveType.HAS_GLOBAL_MAXIMUM


def get_focus_curve_type(focus_curve: np.array) -> FocusCurveType:
    if focus_curve.size < 3:
        return FocusCurveType.UNKNOWN  # Not enough data to analyze properly

    max_indices = np.where(focus_curve == np.max(focus_curve))[0]
    num_maxima = len(max_indices)

    # Check for boundary maximum
    if 0 in max_indices or len(focus_curve) - 1 in max_indices:
        return FocusCurveType.HAS_BOUNDARY_MAXIMUM

    # Check for single global maximum
    if num_maxima == 1:
        return FocusCurveType.HAS_GLOBAL_MAXIMUM

    # Check for multiple maxima
    if num_maxima > 1:
        return FocusCurveType.HAS_MAXIMA

    return FocusCurveType.UNKNOWN


def get_roi_focus_score(
        img: np.array,
        algorithm: FocusAlgorithmType,
        boxes: list[CroppingBox],
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
) -> float:
    total_score = 0
    for box in boxes:
        if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
            total_score += get_focus_score_laplacian_var(img=box.crop(img))
        elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
            total_score += get_focus_score_squared_gradient(img=box.crop(img), threshold=threshold)
        elif algorithm == FocusAlgorithmType.STEEL:
            total_score += get_focus_score_steel(img=box.crop(img), rowshift=rowshift, colshift=colshift, normalise=normalise_score)
        else:
            raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)
    return total_score


def get_focus_score(
        img: np.array,
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
) -> float:
    if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
        return get_focus_score_laplacian_var(img=img)
    elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return get_focus_score_squared_gradient(img=img, threshold=threshold)
    elif algorithm == FocusAlgorithmType.STEEL:
        return get_focus_score_steel(img=img, rowshift=rowshift, colshift=colshift, normalise=normalise_score)
    else:
        raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)


def get_focus_score_laplacian_var(img: np.array) -> float:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return (lap[1:-1, 1:-1]**2).var()


def get_focus_score_squared_gradient(img: np.array, threshold: float | None = None) -> float:
    threshold = DEFAULT_SQUARED_GRAD_THRESHOLD if threshold is None else threshold
    tmp = abs(img[:, 1:] - img[:, :-1])
    tmp[tmp < threshold] = 0
    return (tmp**2).mean()


def get_focus_score_steel(img: np.array, rowshift: int, colshift: int, normalise=False):
    img_float = img.astype(float)
    img_row = np.multiply(
        np.abs(img_float - np.roll(img_float, -rowshift, axis=0)),
        np.abs(img_float - np.roll(img_float, +rowshift, axis=0)),
    )
    img_col = np.multiply(
        np.abs(img_float - np.roll(img_float, -colshift, axis=1)),
        np.abs(img_float - np.roll(img_float, +colshift, axis=1)),
    )
    if normalise:
        return (img_row.sum() + img_col.sum()) / float(2 * img.shape[0] * img.shape[1])
    else:
        return img_row.sum() + img_col.sum()

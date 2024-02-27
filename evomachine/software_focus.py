import cv2
import numpy as np
from typing import Optional

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.evotypes import FocusAlgorithmType


DEFAULT_SQUARED_GRAD_THRESHOLD = 0.1


def get_focus_score(img: np.array, algorithm: FocusAlgorithmType, threshold: Optional[float] = None) -> float:
    if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
        return get_focus_score_laplacian_var(img=img)
    elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return get_focus_score_squared_gradient(img=img, threshold=threshold)
    elif algorithm == FocusAlgorithmType.STEEL:
        return get_focus_score_steel(img=img)
    else:
        raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)


def get_focus_score_laplacian_var(img: np.array) -> float:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return (lap[1:-1, 1:-1]**2).var()


def get_focus_score_squared_gradient(img: np.array, threshold: Optional[float] = None) -> float:
    threshold = DEFAULT_SQUARED_GRAD_THRESHOLD if threshold is None else threshold
    tmp = abs(img[:, 1:] - img[:, :-1])
    tmp[tmp < threshold] = 0
    return (tmp**2).mean()


def get_focus_score_steel(img: np.array, rowshift: int = 25, colshift: int = 50):
    img_row = np.multiply(
        np.abs(img - np.roll(img, -rowshift, axis=0)),
        np.abs(img - np.roll(img, +rowshift, axis=0)),
    )
    img_col = np.multiply(
        np.abs(img - np.roll(img, -colshift, axis=1)),
        np.abs(img - np.roll(img, +colshift, axis=1)),
    )
    return img_row.sum() + img_col.sum()

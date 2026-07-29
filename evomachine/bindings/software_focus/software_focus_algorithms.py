from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from evomachine.types import FocusAlgorithmType
from evomachine.utils import EvoCroppingBox as CroppingBox


DEFAULT_SQUARED_GRAD_THRESHOLD = 0


class SoftwareFocusAlgorithm(ABC):
    """Base class for software focus scoring algorithms."""

    @abstractmethod
    def score_image(self, img: np.ndarray) -> float:
        """
        Return a focus score for one image.

        Parameters
        ----------
        img
            Image array to score. Larger returned values indicate sharper
            focus according to the concrete algorithm.

        Returns
        -------
        float
            Focus score for the provided image.
        """
        raise NotImplementedError

    def score_rois(self, img: np.ndarray, boxes: list[CroppingBox]) -> float:
        """
        Return the summed focus score over cropped image regions.

        Parameters
        ----------
        img
            Source image array containing all regions.
        boxes
            CroppingBox objects that select the regions to score.

        Returns
        -------
        float
            Sum of focus scores for all cropped regions.
        """
        if not isinstance(boxes, list):
            raise TypeError(f"SoftwareFocusAlgorithm.score_rois: boxes must be list, received {type(boxes)}.")
        return sum(self.score_image(img=box.crop(img)) for box in boxes)


class LaplacianVarianceFocusAlgorithm(SoftwareFocusAlgorithm):
    """Focus algorithm based on the variance of the squared Laplacian."""

    def score_image(self, img: np.ndarray) -> float:
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
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return float((lap[1:-1, 1:-1]**2).var())


class SquaredGradientAverageFocusAlgorithm(SoftwareFocusAlgorithm):
    """Focus algorithm based on the mean squared horizontal gradient."""

    def __init__(self, threshold: float | None = None):
        """
        Initialise a squared-gradient focus algorithm.

        Parameters
        ----------
        threshold
            Gradient values below this threshold are ignored. If None, the
            default threshold is used.

        Returns
        -------
        None
        """
        self.threshold: float = DEFAULT_SQUARED_GRAD_THRESHOLD if threshold is None else self._validate_threshold(threshold)

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        """
        Return a validated gradient threshold.

        Parameters
        ----------
        threshold
            Candidate gradient threshold.

        Returns
        -------
        float
            Validated threshold as a float.
        """
        if not isinstance(threshold, int | float) or isinstance(threshold, bool):
            raise TypeError(
                f"SquaredGradientAverageFocusAlgorithm: threshold must be numeric, received {type(threshold)}."
            )
        return float(threshold)

    def score_image(self, img: np.ndarray) -> float:
        """
        Return a squared-gradient focus score for one image.

        Parameters
        ----------
        img
            Image array to score.

        Returns
        -------
        float
            Mean squared horizontal gradient after thresholding.
        """
        tmp = abs(img[:, 1:] - img[:, :-1])
        tmp[tmp < self.threshold] = 0
        return float((tmp**2).mean())


class SteelFocusAlgorithm(SoftwareFocusAlgorithm):
    """Focus algorithm based on shifted row and column image differences."""

    def __init__(
            self,
            rowshift: int = 25,
            colshift: int = 50,
            normalise: bool = False,
    ):
        """
        Initialise a Steel focus algorithm.

        Parameters
        ----------
        rowshift
            Pixel shift along image rows.
        colshift
            Pixel shift along image columns.
        normalise
            If True, divide the summed score by twice the image area.

        Returns
        -------
        None
        """
        self.rowshift: int = self._validate_shift(rowshift=rowshift, name="rowshift")
        self.colshift: int = self._validate_shift(rowshift=colshift, name="colshift")
        if not isinstance(normalise, bool):
            raise TypeError(f"SteelFocusAlgorithm: normalise must be bool, received {type(normalise)}.")
        self.normalise: bool = normalise

    @staticmethod
    def _validate_shift(rowshift: int, name: str) -> int:
        """
        Return a validated pixel shift.

        Parameters
        ----------
        rowshift
            Candidate shift value.
        name
            Field name used in exception messages.

        Returns
        -------
        int
            Validated shift value.
        """
        if not isinstance(rowshift, int) or isinstance(rowshift, bool):
            raise TypeError(f"SteelFocusAlgorithm: {name} must be int, received {type(rowshift)}.")
        return rowshift

    def score_image(self, img: np.ndarray) -> float:
        """
        Return a Steel focus score for one image.

        Parameters
        ----------
        img
            Image array to score.

        Returns
        -------
        float
            Shifted-difference Steel focus score.
        """
        img_float = img.astype(float)
        img_row = np.multiply(
            np.abs(img_float - np.roll(img_float, -self.rowshift, axis=0)),
            np.abs(img_float - np.roll(img_float, +self.rowshift, axis=0)),
        )
        img_col = np.multiply(
            np.abs(img_float - np.roll(img_float, -self.colshift, axis=1)),
            np.abs(img_float - np.roll(img_float, +self.colshift, axis=1)),
        )
        score = float(img_row.sum() + img_col.sum())
        if self.normalise:
            return score / float(2 * img.shape[0] * img.shape[1])
        return score


def create_software_focus_algorithm(
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise: bool = False,
) -> SoftwareFocusAlgorithm:
    """
    Create a focus scoring algorithm for a FocusAlgorithmType.

    Parameters
    ----------
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Row shift for the Steel algorithm.
    colshift
        Column shift for the Steel algorithm.
    normalise
        If True, normalise the Steel score by image area.

    Returns
    -------
    SoftwareFocusAlgorithm
        Focus scoring algorithm instance.
    """
    if not isinstance(algorithm, FocusAlgorithmType):
        raise TypeError(
            f"create_software_focus_algorithm: algorithm must be FocusAlgorithmType, received {type(algorithm)}."
        )
    if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
        return LaplacianVarianceFocusAlgorithm()
    if algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return SquaredGradientAverageFocusAlgorithm(threshold=threshold)
    if algorithm == FocusAlgorithmType.STEEL:
        return SteelFocusAlgorithm(rowshift=rowshift, colshift=colshift, normalise=normalise)
    raise ValueError(f"create_software_focus_algorithm: unsupported focus algorithm {algorithm}.")

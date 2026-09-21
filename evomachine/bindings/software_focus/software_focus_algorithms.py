from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from delta.utils import CroppingBox

from evomachine.types import FocusAlgorithmType


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


class BandpassFFTFocusAlgorithm(SoftwareFocusAlgorithm):
    """Focus algorithm based on the variance of a cell-scale FFT bandpass."""

    def __init__(
            self,
            cell_radius: float = 20.0,
            downsample: int = 4,
            order: int = 4,
            saturation: float | None = None,
            normalize_brightness: bool = True,
    ):
        """
        Initialise a bandpass-FFT focus algorithm.

        Parameters
        ----------
        cell_radius
            Approximate in-focus cell radius in full-resolution pixels.
        downsample
            Integer stride applied before the FFT (speed knob); 1 disables it.
        order
            Butterworth filter order controlling passband roll-off.
        saturation
            Upper intensity clip applied before filtering (camera full scale,
            e.g. 4095 for a 12-bit sensor). None disables clipping.
        normalize_brightness
            If True, normalise the score by mean intensity squared for exposure
            invariance.

        Returns
        -------
        None
        """
        self.cell_radius: float = self._validate_positive(cell_radius, "cell_radius")
        self.downsample: int = self._validate_positive_int(downsample, "downsample")
        self.order: int = self._validate_positive_int(order, "order")
        self.saturation: float | None = (
            None if saturation is None else self._validate_positive(saturation, "saturation")
        )
        if not isinstance(normalize_brightness, bool):
            raise TypeError(
                f"BandpassFFTFocusAlgorithm: normalize_brightness must be bool, "
                f"received {type(normalize_brightness)}."
            )
        self.normalize_brightness: bool = normalize_brightness

    @staticmethod
    def _validate_positive(value: float, name: str) -> float:
        """
        Return a validated strictly-positive number.

        Parameters
        ----------
        value
            Candidate value.
        name
            Field name used in exception messages.

        Returns
        -------
        float
            Validated value as a float.
        """
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise TypeError(
                f"BandpassFFTFocusAlgorithm: {name} must be a positive number, received {value!r}."
            )
        return float(value)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """
        Return a validated strictly-positive integer.

        Parameters
        ----------
        value
            Candidate value.
        name
            Field name used in exception messages.

        Returns
        -------
        int
            Validated integer.
        """
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TypeError(
                f"BandpassFFTFocusAlgorithm: {name} must be a positive int, received {value!r}."
            )
        return value

    def score_image(self, img: np.ndarray) -> float:
        """
        Return a bandpass-FFT focus score for one image; higher is sharper.

        Band-pass the image to the cell spatial scale (features roughly 0.5x-2x
        ``cell_radius``) and measure the variance of that band. Sharp cells put
        energy in the passband; defocus, illumination gradients, and large bright
        blobs are low-frequency and fall outside it, while noise is high-frequency
        and also excluded. The radial passband and the sign-agnostic variance make
        the score rotation- and inversion-invariant.

        Parameters
        ----------
        img
            2D greyscale image array (any numeric dtype).

        Returns
        -------
        float
            Variance of the cell-scale bandpass of the image.
        """
        image = np.asarray(img, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"expected a 2D greyscale image, got shape {image.shape}")

        if self.downsample > 1:
            image = image[:: self.downsample, :: self.downsample]

        # Clamp over-bright artifacts to the sensor ceiling so they cannot inject
        # an unphysically tall edge or inflate the brightness scale.
        if self.saturation is not None:
            image = np.minimum(image, self.saturation)

        # Brightness scale for exposure invariance, taken after clipping.
        scale = image.mean()

        # Work in the downsampled coordinate system.
        radius_ds = self.cell_radius / max(self.downsample, 1)
        # Cutoffs in cycles/pixel: pass features between ~0.5x and ~2x cell size.
        f_hi = 1.0 / radius_ds  # reject features SMALLER than ~0.5x cell (noise)
        f_lo = 1.0 / (4.0 * radius_ds)  # reject features LARGER than ~2x cell (blobs)

        # Radial frequency magnitude for the (possibly rectangular) image.
        fy = np.fft.fftfreq(image.shape[0])[:, None]
        fx = np.fft.fftfreq(image.shape[1])[None, :]
        f = np.sqrt(fx * fx + fy * fy)
        f[0, 0] = 1e-12  # avoid divide-by-zero at DC for the highpass term

        # Butterworth bandpass = lowpass(f_hi) * highpass(f_lo).
        lowpass = 1.0 / (1.0 + (f / f_hi) ** (2 * self.order))
        highpass = 1.0 / (1.0 + (f_lo / f) ** (2 * self.order))
        bandpass = lowpass * highpass

        filtered = np.fft.ifft2(np.fft.fft2(image) * bandpass).real
        score = float(filtered.var())

        if self.normalize_brightness and scale > 0:
            score /= scale * scale

        return score


def create_software_focus_algorithm(
        algorithm: FocusAlgorithmType,
        **kwargs,
) -> SoftwareFocusAlgorithm:
    """
    Create a focus scoring algorithm for a FocusAlgorithmType.

    Parameters
    ----------
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    **kwargs
        Algorithm-specific keyword arguments forwarded to the selected
        algorithm's constructor (e.g. ``rowshift``/``colshift``/``normalise``
        for STEEL, ``threshold`` for SQUARED_GRAD_AVG, or ``cell_radius``/
        ``downsample``/``order``/``saturation``/``normalize_brightness`` for
        BANDPASS_FFT). Passing a keyword the selected algorithm does not accept
        raises TypeError from its constructor.

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
        return LaplacianVarianceFocusAlgorithm(**kwargs)
    if algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return SquaredGradientAverageFocusAlgorithm(**kwargs)
    if algorithm == FocusAlgorithmType.STEEL:
        return SteelFocusAlgorithm(**kwargs)
    if algorithm == FocusAlgorithmType.BANDPASS_FFT:
        return BandpassFFTFocusAlgorithm(**kwargs)
    raise ValueError(f"create_software_focus_algorithm: unsupported focus algorithm {algorithm}.")

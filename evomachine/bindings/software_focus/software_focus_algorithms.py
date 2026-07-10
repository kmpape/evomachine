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


def bandpass_fft_focus_score(
        image: np.ndarray,
        cell_radius: float = 20.0,
        downsample: int = 4,
        order: int = 4,
        saturation: float | None = None,
        normalize_brightness: bool = True,
) -> float:
    """Return a focus score for a single greyscale image; higher is sharper.

    Band-pass the image to the cell spatial scale (features roughly 0.5x-2x
    ``cell_radius``) and measure the variance of that band. Sharp cells put
    energy in the passband; defocus, illumination gradients, and large bright
    blobs are low-frequency and fall outside it, while noise is high-frequency
    and also excluded. The radial passband and the sign-agnostic variance make
    the score rotation- and inversion-invariant. Pure NumPy.

    Parameters
    ----------
    image
        2D greyscale array (any numeric dtype).
    cell_radius
        Approximate in-focus cell radius in full-resolution pixels. The broad
        passband tolerates ~50% error.
    downsample
        Integer stride applied before the FFT (speed knob); 1 disables it.
    order
        Butterworth filter order controlling passband roll-off steepness.
    saturation
        Upper intensity clip applied before filtering (camera full scale, e.g.
        4095 for a 12-bit sensor). ``None`` disables clipping. This is the key
        resilience knob against over-bright blobs/artifacts.
    normalize_brightness
        If True, divide the score by mean intensity squared so the ranking is
        invariant to global brightness/exposure.

    Returns
    -------
    float
        Variance of the cell-scale bandpass of the image.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D greyscale image, got shape {img.shape}")

    if downsample > 1:
        img = img[::downsample, ::downsample]

    # Clamp over-bright artifacts to the sensor ceiling so they cannot inject an
    # unphysically tall edge or inflate the brightness scale.
    if saturation is not None:
        img = np.minimum(img, saturation)

    # Brightness scale for exposure invariance, taken after clipping.
    scale = img.mean()

    # Work in the downsampled coordinate system.
    radius_ds = cell_radius / max(downsample, 1)
    # Cutoffs in cycles/pixel: pass features between ~0.5x and ~2x cell size.
    f_hi = 1.0 / radius_ds  # reject features SMALLER than ~0.5x cell (noise)
    f_lo = 1.0 / (4.0 * radius_ds)  # reject features LARGER than ~2x cell (blobs)

    # Radial frequency magnitude for the (possibly rectangular) image.
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1e-12  # avoid divide-by-zero at DC for the highpass term

    # Butterworth bandpass = lowpass(f_hi) * highpass(f_lo).
    lowpass = 1.0 / (1.0 + (f / f_hi) ** (2 * order))
    highpass = 1.0 / (1.0 + (f_lo / f) ** (2 * order))
    bandpass = lowpass * highpass

    filtered = np.fft.ifft2(np.fft.fft2(img) * bandpass).real
    score = float(filtered.var())

    if normalize_brightness and scale > 0:
        score /= scale * scale

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
        Return a bandpass-FFT focus score for one image.

        Parameters
        ----------
        img
            Image array to score.

        Returns
        -------
        float
            Variance of the cell-scale bandpass of the image.
        """
        return bandpass_fft_focus_score(
            img,
            cell_radius=self.cell_radius,
            downsample=self.downsample,
            order=self.order,
            saturation=self.saturation,
            normalize_brightness=self.normalize_brightness,
        )


def create_software_focus_algorithm(
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise: bool = False,
        cell_radius: float = 20.0,
        downsample: int = 4,
        order: int = 4,
        saturation: float | None = None,
        normalize_brightness: bool = True,
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
    cell_radius
        In-focus cell radius (full-res px) for the bandpass-FFT algorithm.
    downsample
        Integer stride before the FFT for the bandpass-FFT algorithm.
    order
        Butterworth filter order for the bandpass-FFT algorithm.
    saturation
        Upper intensity clip for the bandpass-FFT algorithm (camera full scale,
        e.g. 4095 for a 12-bit sensor). None disables clipping.
    normalize_brightness
        If True, normalise the bandpass-FFT score for exposure invariance.

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
    if algorithm == FocusAlgorithmType.BANDPASS_FFT:
        return BandpassFFTFocusAlgorithm(
            cell_radius=cell_radius,
            downsample=downsample,
            order=order,
            saturation=saturation,
            normalize_brightness=normalize_brightness,
        )
    raise ValueError(f"create_software_focus_algorithm: unsupported focus algorithm {algorithm}.")

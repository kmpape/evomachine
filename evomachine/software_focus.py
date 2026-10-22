import cv2
import numpy as np
from typing import Optional

from delta.utils import CroppingBox

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.evotypes import FocusAlgorithmType, FocusCurveType


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
        threshold: Optional[float] = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        cell_radius: float = 20.0,
        downsample: int = 4,
        order: int = 4,
        saturation: float | None = None,
        normalize_brightness: bool = True,
) -> float:
    total_score = 0
    for box in boxes:
        if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
            total_score += get_focus_score_laplacian_var(img=box.crop(img))
        elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
            total_score += get_focus_score_squared_gradient(img=box.crop(img), threshold=threshold)
        elif algorithm == FocusAlgorithmType.STEEL:
            total_score += get_focus_score_steel(img=box.crop(img), rowshift=rowshift, colshift=colshift, normalise=normalise_score)
        elif algorithm == FocusAlgorithmType.BANDPASS_FFT:
            total_score += get_focus_score_bandpass_fft(
                img=box.crop(img),
                cell_radius=cell_radius,
                downsample=downsample,
                order=order,
                saturation=saturation,
                normalize_brightness=normalize_brightness,
            )
        else:
            raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)
    return total_score


def get_focus_score(
        img: np.array,
        algorithm: FocusAlgorithmType,
        threshold: Optional[float] = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        cell_radius: float = 20.0,
        downsample: int = 4,
        order: int = 4,
        saturation: float | None = None,
        normalize_brightness: bool = True,
) -> float:
    if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
        return get_focus_score_laplacian_var(img=img)
    elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return get_focus_score_squared_gradient(img=img, threshold=threshold)
    elif algorithm == FocusAlgorithmType.STEEL:
        return get_focus_score_steel(img=img, rowshift=rowshift, colshift=colshift, normalise=normalise_score)
    elif algorithm == FocusAlgorithmType.BANDPASS_FFT:
        return get_focus_score_bandpass_fft(
            img=img,
            cell_radius=cell_radius,
            downsample=downsample,
            order=order,
            saturation=saturation,
            normalize_brightness=normalize_brightness,
        )
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


def get_focus_score_bandpass_fft(
        img: np.ndarray,
        cell_radius: float = 20.0,
        downsample: int = 4,
        order: int = 4,
        saturation: float | None = None,
        normalize_brightness: bool = True,
) -> float:
    """
    Return the variance of the cell-scale FFT bandpass of an image.

    Parameters
    ----------
    img
        Two-dimensional greyscale image array with any numeric dtype.
    cell_radius
        Approximate in-focus cell radius in full-resolution pixels.
    downsample
        Positive integer stride applied before the FFT.
    order
        Positive integer Butterworth filter order controlling passband roll-off.
    saturation
        Optional positive upper intensity clip applied before filtering.
    normalize_brightness
        Divide the score by the squared mean intensity when True.

    Returns
    -------
    float
        Variance of the bandpass-filtered image; higher values indicate sharper focus.
    """
    if not isinstance(cell_radius, (int, float)) or isinstance(cell_radius, bool) or cell_radius <= 0:
        raise TypeError(f"get_focus_score_bandpass_fft: cell_radius must be a positive number, received {cell_radius!r}.")
    if not isinstance(downsample, int) or isinstance(downsample, bool) or downsample <= 0:
        raise TypeError(f"get_focus_score_bandpass_fft: downsample must be a positive int, received {downsample!r}.")
    if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
        raise TypeError(f"get_focus_score_bandpass_fft: order must be a positive int, received {order!r}.")
    if saturation is not None and (
            not isinstance(saturation, (int, float)) or isinstance(saturation, bool) or saturation <= 0
    ):
        raise TypeError(f"get_focus_score_bandpass_fft: saturation must be a positive number, received {saturation!r}.")
    if not isinstance(normalize_brightness, bool):
        raise TypeError(
            "get_focus_score_bandpass_fft: normalize_brightness must be bool, "
            f"received {type(normalize_brightness)}."
        )

    image = np.asarray(img, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D greyscale image, got shape {image.shape}")

    if downsample > 1:
        image = image[::downsample, ::downsample]

    if saturation is not None:
        image = np.minimum(image, saturation)

    scale = image.mean()
    radius_ds = float(cell_radius) / downsample
    f_hi = 1.0 / radius_ds
    f_lo = 1.0 / (4.0 * radius_ds)

    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    frequency[0, 0] = 1e-12

    lowpass = 1.0 / (1.0 + (frequency / f_hi) ** (2 * order))
    highpass = 1.0 / (1.0 + (f_lo / frequency) ** (2 * order))
    bandpass = lowpass * highpass

    filtered = np.fft.ifft2(np.fft.fft2(image) * bandpass).real
    score = float(filtered.var())
    if normalize_brightness and scale > 0:
        score /= scale * scale
    return score

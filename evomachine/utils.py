from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
import serial.tools.list_ports
import skimage
import time
from typing import Any

import delta.utils

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def channel_extend_img(
        img: np.ndarray,
        channel_dict: dict[Any, int],
        channels: list[Any] | Any,
        ind: int = 0,
) -> np.ndarray:
    """
    Takes a 3D images and inserts a new frame obtained from combine_channels.

    Parameters
    ----------
    img : np.ndarray
        3D image.
    channel_dict : dict[Any, int]
        Dictionary mapping channels to their respective index in img.
    channels : list[Any]
        List of channels to combine.
    ind : int, optional
        Index of where the combined image will be inserted.
    Returns
    -------

    """
    if not isinstance(channels, list):
        channels = [channels]
    img2d = combine_channels(img=img, channel_dict=channel_dict, channels=channels)
    return combine_images(img3d=img, img2d=img2d, ind=ind)  # noqa


def combine_images(img3d: np.ndarray, img2d: np.ndarray, ind: int = 0) -> np.ndarray:
    """
    Combines a 3D and a 2D image by putting the 2D image at position ind of the first 3D axis.

    Parameters
    ----------
    img3d:  3D image
    img2d:  2D image
    ind:    Index of where img2d will be placed

    Returns
    -------
    img3d_new:  New 3D image.
    """
    if img2d.shape != img3d.shape[1:]:
        raise ValueError(f"Shape of img2d must match the height and width of img3d slices. "
                         f"Expected shape {img3d.shape[1:]}, got {img2d.shape}")
    n = img3d.shape[0]
    ind = n if ind == -1 else ind
    if ind < 0 or ind > n:
        raise ValueError(f"Index `ind` must be between 0 and {n}, but got {ind}.")
    img3d_new = np.empty((n + 1, *img3d.shape[1:]), dtype=img3d.dtype)
    ind_old = [i for i in range(n+1) if i != ind]
    img3d_new[ind_old] = img3d
    img3d_new[ind] = img2d
    return img3d_new


def combine_channels(
        img: np.ndarray,
        channel_dict: dict[Any, int],
        channels: list[Any],
) -> np.ndarray:
    """
    
    Parameters
    ----------
    img:            3D numpy array with channels in the first dimension
    channel_dict:   Dictionary mapping channels to first dimension indices
    channels:       Channels to combine

    Returns
    -------
    combined_img:   2D numpy array of the same type with averaged values
    """
    if len(channels) == 1:
        return img[channel_dict[channels[0]]]
    else:
        tmp = img.astype(float)
        tmp = tmp[[channel_dict[ch] for ch in channels]]
        tmp = np.mean(tmp, axis=0)
        return tmp.astype(img.dtype)


def list_serial_ports(starts_with: str | None = None) -> list[str]:
    ports = serial.tools.list_ports.comports()
    if starts_with is not None:
        return [port.device for port in ports if port.device.startswith(starts_with)]
    else:
        return [port.device for port in ports]


def get_psu_port() -> str:
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.product == "KORAD USB Mode" and port.manufacturer == "Nuvoton":
            return port.device


# Data Class to hold rotation parameters
@dataclass
class RotationParameters:
    cutoff_frequency_ratio: float = 0.04
    min_exposure: float = 0.05
    max_exposure: float = 0.1
    hough_threshold: float = 0.7


def multipos_rotation_correction(imgs: list[np.ndarray], params: RotationParameters = RotationParameters()) -> float:
    rotations = [rotation_correction(img=img.astype(float), params=params) for img in imgs]
    rotations_counter = Counter([r for r in rotations if r is not None])
    if not rotations_counter.most_common():
        return 0.0
    else:
        return rotations_counter.most_common()[0][0]


def rotation_correction(img: np.ndarray, params: RotationParameters = RotationParameters()) -> float:
    """

    Parameters
    ----------
    img: np.ndarray
        Image (2D) of any type.
    params: RotationParameters
        TODO

    Returns
    -------
    angle: float
        Returns the correction angle in degrees. Apply to image e.g. via skimage.transform.rotate(img, angle, resize=True).
    """
    img = skimage.transform.resize(img, (img.shape[0] // 4, img.shape[1] // 4))
    rs = skimage.filters.butterworth(img, cutoff_frequency_ratio=params.cutoff_frequency_ratio, high_pass=False)
    rs = skimage.exposure.equalize_hist(rs)
    rs = skimage.filters.frangi(rs, sigmas=(4,))
    rs = skimage.util.crop(rs, 100)
    rs = skimage.exposure.rescale_intensity(rs, out_range=(0, 1))
    rs = (rs > params.min_exposure) & (rs < params.max_exposure)
    hspace, angles, distances = skimage.transform.hough_line(rs)
    _, angles, distances = skimage.transform.hough_line_peaks(hspace, angles, distances, threshold=params.hough_threshold*np.max(hspace))
    bin_edges = np.linspace(-np.pi/2, np.pi/2, 200)
    bins = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    hist, _ = np.histogram(angles, bins=bin_edges)
    angle = bins[np.argmax(hist)]
    return np.rad2deg(angle)


def normalise_frame(
        frame: np.ndarray,
        channels: list[int] | None = None,
        dtype: np.dtype | None = None
) -> np.ndarray:
    """
    Normalises frame by datatype or range.

    Parameters
    ----------
    frame: np.ndarray
        Either a 2D or a 3D image (of shape (channels, X, Y))
    channels: Optional[List[int]]
        For 3D array, provide a list of integers to apply normalisation to (i, X, Y) for i in channels.
    dtype: Optional[np.dtype]
        If none, frame normalised as (frame-frame.min())/(frame.max()-frame.min())
    Returns
    -------
    norm_frame: np.ndarray
        Array of doubles with values in [0,1].
    """
    norm_frame = frame.astype(float)
    if len(frame.shape) == 2:
        if dtype is None:
            norm_frame = (norm_frame - norm_frame.min()) / np.ptp(norm_frame)
        else:
            depth = {np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('uint32'): 32}[dtype]
            f = float(2 ** depth - 1)
            norm_frame = norm_frame / f
    else:
        if channels is None:
            channels = list(range(frame.shape[0]))
        if dtype is None:
            for c in channels:
                norm_frame[c, :, :] = (norm_frame[c, :, :] - norm_frame[c, :, :].min()) / np.ptp(norm_frame[c, :, :])
        else:
            depth = {np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('uint32'): 32}[dtype]
            f = float(2**depth - 1)
            for c in channels:
                norm_frame[c, :, :] = norm_frame[c, :, :] / f
    return norm_frame


@dataclass
class EvoCroppingBox:
    """
    Class describing a box to cut out. Taken and extended from delta.utils.CroppingBox.

    Attributes
    ----------
        xtl : int
            Top-left corner X coordinate.
        ytl : int
            Top-left corner Y coordinate.
        xbr : int
            Bottom-right corner X coordinate.
        ybr : int
            Bottom-right corner Y coordinate.
    """

    xtl: int
    ytl: int
    xbr: int
    ybr: int
    is_none: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.ybr - self.ytl, self.xbr - self.xtl

    @staticmethod
    def full(image: np.ndarray | tuple[int, int]) -> 'EvoCroppingBox':
        """
        Return a cropping box set to the full size of the image.

        Arguments
        ---------
            image : np.ndarray
                Image to use as reference for the bounding box.

        Returns
        -------
            box : CroppingBox
                Cropping box adjusted to the full size of the image.
        """
        shape = image.shape[:2] if isinstance(image, np.ndarray) else image
        return EvoCroppingBox(xtl=0, ytl=0, xbr=shape[1], ybr=shape[0])

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop an image according to the cropping box.

        Pads with zeros if a part of the box falls outside the image.

        Arguments
        ---------
            image : np.ndarray
                Image to crop.

        Returns
        -------
            patch : np.ndarray
                Patch cropped from the image.
        """
        if image.ndim != 2:
            raise ValueError("`image` must have 2 dimensions.")
        cropped = image[
            max(self.ytl, 0): min(self.ybr, image.shape[0]),
            max(self.xtl, 0): min(self.xbr, image.shape[1]),
        ]
        padding = (
            (max(-self.ytl, 0), max(self.ybr - image.shape[0], 0)),
            (max(-self.xtl, 0), max(self.xbr - image.shape[1], 0)),
        )
        return np.pad(cropped, padding)

    def frame(self, image: np.ndarray, thickness: int = 1, value: Any = 0) -> np.ndarray:
        """
        Add a frame to an image according to the cropping box.

        Arguments
        ---------
            image : np.ndarray
                Image to crop.
            thickness : int
                Thickness of the frame.
            value : Any
                Value to use for the frame.

        Returns
        -------
            framed : np.ndarray
                Image with frame
        """
        framed = image.copy()
        framed[max(self.ytl, 0): min(self.ybr, image.shape[0]),
               max(self.xtl, 0): max(self.xtl, 0) + thickness] = value
        framed[max(self.ytl, 0): min(self.ybr, image.shape[0]),
               min(self.xbr, image.shape[1]) - thickness: min(self.xbr, image.shape[1])] = value
        framed[max(self.ytl, 0): max(self.ytl, 0) + thickness,
               max(self.xtl, 0): min(self.xbr, image.shape[1])] = value
        framed[min(self.ybr, image.shape[0]) - thickness: min(self.ybr, image.shape[0]),
               max(self.xtl, 0): min(self.xbr, image.shape[1])] = value

        return framed

    @staticmethod
    def from_dict(d: dict[str, int]) -> 'EvoCroppingBox':
        return EvoCroppingBox(xtl=d['xtl'], ytl=d['ytl'], xbr=d['xbr'], ybr=d['ybr'])

    def patch(self, image: np.ndarray, patch: np.ndarray):
        """
        Apply a patch on an image at the position specified by the box.

        Parts of the box may fall outside the image.

        Arguments
        ---------
            image : np.ndarray
                Image to patch.
            patch : np.ndarray
                Patch to apply.

        Returns
        -------
            image : np.ndarray
                The patched image.
        """
        if image.ndim != 2 or patch.ndim != 2:
            raise ValueError("`image` and `patch` must have 2 dimensions.")
        if patch.shape != (self.ybr - self.ytl, self.xbr - self.xtl):
            raise ValueError(
                "`patch` must have the same dimensions as the cropping box."
            )
        if (
            self.ybr <= 0
            or image.shape[0] <= self.ytl
            or self.xbr <= 0
            or image.shape[1] <= self.xtl
        ):
            raise ValueError("`box` must fall at least partially inside the image.")
        image[
            max(self.ytl, 0): min(self.ybr, image.shape[0]),
            max(self.xtl, 0): min(self.xbr, image.shape[1]),
        ] = patch[
            max(-self.ytl, 0): min(self.ybr, image.shape[0]) - self.ytl,
            max(-self.xtl, 0): min(self.xbr, image.shape[1]) - self.xtl,
        ]
        return image

    def to_delta_cropping_box(self) -> delta.utils.CroppingBox:
        return delta.utils.CroppingBox(xtl=self.xtl, ytl=self.ytl, xbr=self.xbr, ybr=self.ybr)

    @staticmethod
    def none_box() -> 'EvoCroppingBox':
        return EvoCroppingBox(xtl=0, ytl=0, xbr=0, ybr=0, is_none=True)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.xbr == other.xbr and self.xtl == other.xtl and self.ybr == other.ybr and self.ytl == other.ytl
        else:
            return False


class Timer:
    def __init__(self, timer_level: int = 0, name: str = "", enabled: bool = True):
        self.timer_level: int = timer_level
        self.name: str = name
        self.enabled: bool = enabled
        self.timings: dict[str, list] = {}

    def start(self, func_name: str, level: int) -> None:
        if (not self.enabled) or (level > self.timer_level):
            return
        if not (func_name in self.timings):
            self.timings[func_name] = []
        start_time = time.perf_counter()
        self.timings[func_name].append([start_time, None])

    def stop(self, func_name: str, level: int) -> None:
        if (not self.enabled) or (level > self.timer_level):
            return
        end_time = time.perf_counter()
        self.timings[func_name][-1][1] = end_time

    def get_timings(self) -> dict[str, dict]:
        timings = {}
        for func_name, start_end in self.timings.items():
            n_calls = len(start_end)
            elapsed = [end-start for start, end in start_end]
            timings[func_name] = {'n_calls': n_calls, 'avg': sum(elapsed)/n_calls, 'median': np.median(elapsed),
                                  'min': min(elapsed), 'max': max(elapsed)}
        return timings

    def get_timings_per_call(self) -> dict[str, list]:
        timings = {}
        for func_name, start_end in self.timings.items():
            elapsed = [end-start for start, end in start_end]
            timings[func_name] = elapsed
        return timings

    def display_timings(self, scaling: int = 1) -> None:
        if not self.enabled:
            return
        # if scaling != 1:
        #     logger.info(f"\nTimings {self.name} (timer_level {self.timer_level}) (scaling {scaling}):\n")
        # else:
        #     logger.info(f"\nTimings {self.name} (timer_level {self.timer_level}):\n")
        data = [[func_name, _data['n_calls'], _data['avg']*scaling, _data['median']*scaling,
                 _data['min']*scaling, _data['max']*scaling]
                for func_name, _data in self.get_timings().items()]
        headers = ["name", "n_calls", "avg", "median", "min", "max"]
        df = pd.DataFrame(data, columns=headers)
        # logger.info(f"{df}")
        # logger.info("---\n")

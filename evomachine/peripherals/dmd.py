from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import pickle as pkl
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import skimage.color
import skimage.io

from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig
from evomachine.bindings.binding_types import BindingType
from evomachine.config import CAM_WIDTH_HEIGHT, DMD_WIDTH_HEIGHT
from evomachine.types import LEDType

logger = logging.getLogger(__name__)

ARR_TYPE = np.uint8


@dataclass
class DmdCalibrationConfig:
    channel: LEDType | list[LEDType]
    brightness: float | int
    exposure: float | int
    line_width: int
    step: int
    delay: float | int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    on_mothermachine: bool

    def __post_init__(self) -> None:
        if not ((0 <= self.start_row) and (self.start_row < self.end_row) and (self.end_row < 2716)):
            raise ValueError("Indices must be within DMD boundaries.")
        if not ((0 <= self.start_col) and (self.start_col < self.end_col) and (self.end_col < 1600)):
            raise ValueError("Indices must be within DMD boundaries.")

    def copy(self) -> "DmdCalibrationConfig":
        return DmdCalibrationConfig(**self.__dict__)

    def updated(self, **kwargs: Any) -> "DmdCalibrationConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"DmdCalibrationConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return DmdCalibrationConfig(**values)

    def update_from_mapping(self, updates: dict[str, Any]) -> "DmdCalibrationConfig":
        if not isinstance(updates, dict):
            raise TypeError("DmdCalibrationConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)

    def __str__(self) -> str:
        lines = ["DmdCalibrationConfig"]
        for index, (key, value) in enumerate(self.__dict__.items()):
            lines.append(f"{' └─ ' if index == len(self.__dict__) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)


class DmdCalibrationConfigFactory:
    @staticmethod
    def default(channel: LEDType | list[LEDType] = LEDType.LED_450_NM) -> DmdCalibrationConfig:
        return DmdCalibrationConfig(
            channel=channel,
            brightness=29,
            exposure=100,
            line_width=5,
            step=150,
            delay=0.75,
            start_row=200,
            end_row=2500,
            start_col=0,
            end_col=1599,
            on_mothermachine=True,
        )

    @staticmethod
    def thin_fluo_slide(channel: LEDType = LEDType.LED_565_NM) -> DmdCalibrationConfig:
        return DmdCalibrationConfig(
            channel=channel,
            brightness=29,
            exposure=100,
            line_width=5,
            step=150,
            delay=0.5,
            start_row=200,
            end_row=2200,
            start_col=0,
            end_col=1599,
            on_mothermachine=False,
        )

    @staticmethod
    def fluo_slide(channel: LEDType = LEDType.LED_450_NM) -> DmdCalibrationConfig:
        return DmdCalibrationConfig(
            channel=channel,
            brightness=0.4,
            exposure=50,
            line_width=2,
            step=50,
            delay=0.5,
            start_row=200,
            end_row=2200,
            start_col=0,
            end_col=1599,
            on_mothermachine=False,
        )


@dataclass(kw_only=True)
class DmdConfig(PeripheralConfig):
    """Configuration for creating a DMD wrapper from a peripheral controller."""

    width_height_DMD: tuple[int, int] = DMD_WIDTH_HEIGHT
    width_height_CAM: tuple[int, int] = CAM_WIDTH_HEIGHT
    display_offset: tuple[int, int] = (0, 0)
    monitor_index: int | None = None
    calibration_file: Path | None = None

    def __post_init__(self) -> None:
        """
        Validate DMD configuration after dataclass construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        super().__post_init__()
        self.width_height_DMD = self._validate_size(
            value=self.width_height_DMD,
            field_name="width_height_DMD",
        )
        self.width_height_CAM = self._validate_size(
            value=self.width_height_CAM,
            field_name="width_height_CAM",
        )
        self.display_offset = self._validate_size(
            value=self.display_offset,
            field_name="display_offset",
            allow_zero=True,
        )
        if self.monitor_index is not None:
            if not isinstance(self.monitor_index, int) or isinstance(self.monitor_index, bool):
                raise TypeError(f"DmdConfig: monitor_index must be int or None, received {type(self.monitor_index)}.")
            if self.monitor_index < 0:
                raise ValueError(f"DmdConfig: monitor_index must be non-negative, received {self.monitor_index}.")
        if isinstance(self.calibration_file, str):
            self.calibration_file = Path(self.calibration_file)
        if self.calibration_file is not None and not isinstance(self.calibration_file, Path):
            raise TypeError(
                f"DmdConfig: calibration_file must be Path, str, or None, received {type(self.calibration_file)}."
            )

    @staticmethod
    def _validate_size(
            value: tuple[int, int],
            field_name: str,
            allow_zero: bool = False,
    ) -> tuple[int, int]:
        """
        Return a validated two-integer size or offset tuple.

        Parameters
        ----------
        value
            Candidate two-value tuple.
        field_name
            Field name used in error messages.
        allow_zero
            If True, zero values are accepted.

        Returns
        -------
        tuple[int, int]
            Validated tuple.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"DmdConfig: {field_name} must be tuple[int, int], received {type(value)}.")
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise TypeError(f"DmdConfig: {field_name} entries must be int.")
        if allow_zero:
            if not all(item >= 0 for item in value):
                raise ValueError(f"DmdConfig: {field_name} entries must be non-negative.")
        elif not all(item > 0 for item in value):
            raise ValueError(f"DmdConfig: {field_name} entries must be positive.")
        return value


class Dmd(Peripheral):
    """  TODO(CODEX): Modify doc if needed with implemented changes.
    Class for communicating with the DMD. After calling initialise(), communicate with the DMD using following
    functions:
    - display_full():           Full illumination
    - display_none():           No illumination
    - display_fov_full():       Display full illumination on entire FoV.
    - display_line_horiz(...):  Display a horizontal line. Uses DMD coordinates.
    - display_line_vert(...):   Display a vertical line. Uses DMD coordinates.
    - display_on_fov(...):      Display a number of rectangles on FoV. Uses image coordinates.


    Note:
    The DMD has width DMD_WIDTH_HEIGHT[0] and height DMD_WIDTH_HEIGHT[1]. In this class, the images are allocated as
    an array with the number of rows corresponding to the width and columns corresponding to the height.

    _____________________________________________________
    | (width,0)                                   (0,0) |
    |                                                   |
    | SCREEN AS SEEN ON A SURFACE BEFORE THE MICROSCOPE |
    |                                                   |
    | (width,height)                         (0,height) |
    |___________________________________________________|


    Example:
        Line 1 produced by display_line_horiz(100)
        Line 2 produced by display_line_vert(100)

    -> Monitor view:
    _____________________________________________________
    | 1111112111111111111111111111111111111111111111111 |
    |       2                                           |
    |       2                                           |
    |       2                                           |
    |       2                                           |
    |_______2___________________________________________|

    -> Camera view:
    _____________________________________________________
    | 2222221222222222222222222222222222222222222222222 |
    |       1                                           |
    |       1                                           |
    |       1                                           |
    |       1                                           |
    |_______1___________________________________________|

    """

    DEFAULT_NAME: str = "DMD"
    DEFAULT_LINE_WIDTH: int = 5
    DEFAULT_SQUARE_WIDTH: int = 20
    EXTENSIONS = ["png", "tiff", "tif"]

    def __init__(
            self,
            peripheral_ctrl: PeripheralController,
            name: str = "",
            check_initialised: bool = True,
            check_alive: bool = True,
            width_height_DMD: tuple[int, int] = DMD_WIDTH_HEIGHT,
            width_height_CAM: tuple[int, int] = CAM_WIDTH_HEIGHT,
            calibration_file: Path | None = None,
    ):
        self.peripheral_ctrl: PeripheralController = peripheral_ctrl
        self.name: str = name or self.DEFAULT_NAME
        self.check_initialised: bool = check_initialised
        self.check_alive: bool = check_alive
        self.width_height_DMD: tuple[int, int] = width_height_DMD
        self.width_height_CAM: tuple[int, int] = width_height_CAM
        self.default_line_width: int = self.DEFAULT_LINE_WIDTH
        self._is_full_display: bool = False
        self._loaded_img: np.ndarray | None = None
        self._calib_file: Path = calibration_file or Path(__file__).resolve().parent / "dmd_calibration_data_2025-08-14_v2.pkl"
        self._calib_data: list | None = None
        self._homography_mat: np.ndarray | None = None
        self._homography_mat_inv: np.ndarray | None = None
        self.config: DmdConfig | None = None
        self.calibrate(filepath=self._calib_file)

    @staticmethod
    def load_calibration_data(
            filepath: Path,
    ) -> tuple[list, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Load calibration data and compute image-to-DMD and DMD-to-image homographies."""
        if not filepath.exists():
            logger.error(f"Dmd.load_calibration_data: file {filepath} not found.")
            return None, None, None
        logger.info(f"Dmd.load_calibration_data: loading calibration data from {filepath}.")
        with open(str(filepath), "rb") as file:
            calib_data = pkl.load(file)

        dmd_points = np.array([(c_dmd, r_dmd) for ((r_dmd, c_dmd), _, _) in calib_data])
        cam_points = np.array([(c_cam, r_cam) for (_, (r_cam, c_cam), _) in calib_data])
        homography_mat, _ = cv2.findHomography(srcPoints=cam_points, dstPoints=dmd_points)
        homography_mat_inv, _ = cv2.findHomography(srcPoints=dmd_points, dstPoints=cam_points)
        return calib_data, homography_mat, homography_mat_inv

    def initialise(self, force: bool = False) -> None:
        """Initialise the underlying DMD peripheral controller."""
        self.peripheral_ctrl.initialise(force=force)

    def finalise(self, force: bool = False) -> None:
        """Shutdown the underlying DMD peripheral controller."""
        self.peripheral_ctrl.shutdown(force=force)

    def stop(self) -> None:
        """Blank the DMD display."""
        self.display_none()

    def is_alive(self) -> bool:
        """Return whether the underlying controller reports alive."""
        return self.peripheral_ctrl.is_alive()

    def is_initialised(self) -> bool:
        """Return whether the underlying controller reports initialised."""
        return self.peripheral_ctrl.is_initialised()

    def is_calibrated(self) -> bool:
        """Return whether both homography matrices are available."""
        return self._homography_mat is not None and self._homography_mat_inv is not None

    def is_full_display(self) -> bool:
        """Return whether the most recent display state was full white."""
        return self._is_full_display

    def calibrate(self, filepath: Path | None = None) -> None:
        """Load calibration data from filepath or the configured calibration file."""
        filepath = self._calib_file if filepath is None else Path(filepath)
        self._calib_data, self._homography_mat, self._homography_mat_inv = self.load_calibration_data(filepath=filepath)
        self._calib_file = filepath

    def get_calibration_data(self) -> tuple[list, np.ndarray, np.ndarray, Path] | tuple[None, None, None, Path]:
        """Return loaded calibration data, homographies, and the calibration filename."""
        return self._calib_data, self._homography_mat, self._homography_mat_inv, self._calib_file

    def get_calibration_filename(self) -> Path:
        """Return the configured calibration filename."""
        return self._calib_file

    def img_to_dmd_coords(self, img_row: int, img_col: int) -> tuple[int, int]:
        """Transform camera image coordinates to DMD coordinates."""
        if not self.is_calibrated():
            msg = "img_to_dmd_coords: no calibration data provided."
            logger.error(msg)
            raise RuntimeError(msg)
        point_cam = np.array([[[img_col, img_row]]]).astype(float)
        point_dmd = cv2.perspectiveTransform(point_cam, self._homography_mat)
        return int(np.round(point_dmd[0][0][1])), int(np.round(point_dmd[0][0][0]))

    def img_to_dmd_array(self, img: np.ndarray) -> np.ndarray:
        """Transform a camera-sized pattern to a DMD-sized pattern."""
        if not self.is_calibrated():
            msg = "img_to_dmd_array: no calibration data provided."
            logger.error(msg)
            raise RuntimeError(msg)
        if img.shape != self.width_height_CAM:
            msg = f"img_to_dmd_array: Expected image of shape {self.width_height_CAM} but received {img.shape}."
            logger.error(msg)
            raise ValueError(msg)
        return cv2.warpPerspective(
            img, self._homography_mat, self.width_height_DMD[::-1], flags=cv2.INTER_NEAREST
        ).astype(img.dtype)

    def dmd_to_img_coords(self, img_row: int, img_col: int) -> tuple[int, int]:
        """Transform DMD coordinates to camera image coordinates."""
        if not self.is_calibrated():
            msg = "dmd_to_img_coords: no calibration data provided."
            logger.error(msg)
            raise RuntimeError(msg)
        point_dmd = np.array([[[img_col, img_row]]]).astype(float)
        point_cam = cv2.perspectiveTransform(point_dmd, self._homography_mat_inv)
        return int(np.round(point_cam[0][0][1])), int(np.round(point_cam[0][0][0]))

    def dmd_to_img_array(self, img: np.ndarray) -> np.ndarray:
        """Transform a DMD-sized pattern to a camera-sized pattern."""
        if not self.is_calibrated():
            msg = "dmd_to_img_array: no calibration data provided."
            logger.error(msg)
            raise RuntimeError(msg)
        if img.shape != self.width_height_DMD:
            msg = f"dmd_to_img_array: Expected image of shape {self.width_height_DMD} but received {img.shape}."
            logger.error(msg)
            raise ValueError(msg)
        return cv2.warpPerspective(
            img, self._homography_mat_inv, self.width_height_CAM, flags=cv2.INTER_NEAREST
        ).astype(img.dtype)

    def patches_from_roi_groups(
            self,
            roi_boxes_group_ids: list[list[int]],
            roi_boxes: list[Any],
            xshift: int = 0,
            yshift: int = 0,
    ) -> list[Any]:
        """Create black patch boxes between grouped ROI columns and image borders."""
        from delta.utils import CroppingBox

        black_patches = []
        for i, group_ids in enumerate(roi_boxes_group_ids):
            if i == 0:
                trench = roi_boxes[group_ids[0]]
                black_patches.append(CroppingBox(
                    xtl=0,
                    ytl=0 + yshift,
                    xbr=trench.xtl - xshift,
                    ybr=self.width_height_CAM[1] - 1 - yshift,
                ))
            else:
                group_ids_left = roi_boxes_group_ids[i - 1]
                trench_left = roi_boxes[group_ids_left[0]]
                trench_right = roi_boxes[group_ids[0]]
                black_patches.append(CroppingBox(
                    xtl=trench_left.xbr + xshift,
                    ytl=0 + yshift,
                    xbr=trench_right.xtl - xshift,
                    ybr=self.width_height_CAM[1] - 1 - yshift,
                ))
                if i == len(roi_boxes_group_ids) - 1:
                    trench = roi_boxes[group_ids[0]]
                    black_patches.append(CroppingBox(
                        xtl=trench.xbr + xshift,
                        ytl=0 + yshift,
                        xbr=self.width_height_CAM[0] - 1,
                        ybr=self.width_height_CAM[1] - 1 - yshift,
                    ))
        return black_patches

    def pattern_from_roi_boxes(
            self,
            boxes: list[Any],
            fill_x: float = 1.0,
            fill_y: float = 1.0,
            invert: bool = False,
            warp: bool = True,
            drift: tuple[int, int] | None = None,
            black_patches: list[Any] | None = None,
            border_px: int = 2,
    ) -> np.ndarray:
        """Create a DMD pattern from camera-coordinate ROI boxes."""
        fill_color = 255 if (not invert) else 0
        cam_img = self.get_one_array(img_size=self.width_height_CAM) * (0 if (not invert) else 255)
        for box in boxes:
            shift_x = int(np.round(0.5 * (1 - fill_x) * (box.xbr - box.xtl), 0))
            shift_y = int(np.round(0.5 * (1 - fill_y) * (box.ybr - box.ytl), 0))
            start_row = max(box.ytl + shift_y, 0)
            end_row = min(box.ybr + 1 - shift_y, cam_img.shape[0] - 1)
            start_col = max(box.xtl + shift_x, 0)
            end_col = min(box.xbr + 1 - shift_x, cam_img.shape[1] - 1)
            cam_img[start_row:end_row, start_col:end_col] = fill_color
        if black_patches is not None:
            for box in black_patches:
                start_row = max(box.ytl, 0)
                end_row = min(box.ybr + 1, cam_img.shape[0] - 1)
                start_col = max(box.xtl, 0)
                end_col = min(box.xbr + 1, cam_img.shape[1] - 1)
                cam_img[start_row:end_row, start_col:end_col] = 0
        if drift is not None:
            from delta.imgops import correct_drift

            cam_img = correct_drift(cam_img, drift)
        if border_px > 0:
            cam_img[0:border_px + 1, :] = 0
            cam_img[-border_px:, :] = 0
            cam_img[:, 0:border_px + 1] = 0
            cam_img[:, -border_px:] = 0
        return self.img_to_dmd_array(cam_img) if warp else cam_img

    def _check_ready(self) -> None:
        """Raise if display actions are not allowed by current readiness checks."""
        if self.check_initialised and not self.is_initialised():
            raise RuntimeError(f"{self.name}: DMD peripheral controller is not initialised.")
        if self.check_alive and not self.is_alive():
            raise RuntimeError(f"{self.name}: DMD peripheral controller is not alive.")

    def _normalise_display_image(self, img: np.ndarray) -> np.ndarray:
        """Validate a DMD image and return a 2D uint8 array ready for display transport."""
        if img.dtype != ARR_TYPE:
            raise TypeError(f"Image must be of type {ARR_TYPE}. Received {img.dtype}.")
        if img.shape == self.width_height_DMD:
            return img
        if img.shape == (*self.width_height_DMD, 3):
            logger.warning(f"{self.name}.display_image: B/W image expected. Sending image[:,:,0] instead.")
            return img[:, :, 0]
        raise ValueError(f"{self.name}.display_image: provided image of shape={img.shape}, "
                         f"but DMD shape={self.width_height_DMD}.")

    @abstractmethod
    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        """Display a DMD-sized uint8 image through a binding-specific backend."""
        raise NotImplementedError

    def display_none(self) -> None:
        """Display a black screen."""
        self.display_image(self.get_zero_array())

    def display_full(self, force_display: bool = False) -> None:
        """Display a white screen."""
        if not force_display and self.is_full_display():
            return
        self.display_image(self.get_one_array(), _is_full_display=True)

    def get_zero_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a black uint8 image."""
        return np.zeros(img_size or self.width_height_DMD, dtype=ARR_TYPE)

    def get_one_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a white uint8 image."""
        return np.ones(img_size or self.width_height_DMD, dtype=ARR_TYPE) * 255

    @staticmethod
    def _make_half_line_width(line_width: int, at_pos: int, length: int) -> tuple[int, int]:
        """Return inclusive/exclusive slice bounds for a centered line width."""
        if line_width == 1:
            return at_pos, min(at_pos + line_width, length)
        if line_width % 2 == 0:
            return max(0, at_pos + 1 - int(line_width / 2)), min(length, at_pos + int(line_width / 2))
        return max(0, at_pos - int(line_width / 2)), min(length, at_pos + int(line_width / 2))

    def get_calibration_image(self, lw: int = 5, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a calibration image with boxes and a crosshair."""
        img = self.get_zero_array(img_size=img_size)
        mid_row, mid_col = img.shape[0] // 2, img.shape[1] // 2
        cv2.line(img, (mid_col, 0), (mid_col, img.shape[0]), 255, lw)
        cv2.line(img, (0, mid_row), (img.shape[1], mid_row), 255, lw)
        box_sizes = [5, 10, 20, 40, 80, 160, 320]
        box_sizes_rev = box_sizes[::-1]
        bigshifts_x = [400, -400]
        shift = 20
        for bigshift_x in bigshifts_x:
            for idx, box_size in enumerate(box_sizes):
                start_x = mid_col - shift - box_size + bigshift_x
                start_y = mid_row + shift * (idx + 1) + sum(box_sizes[:idx + 1])
                cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
                start_y = mid_row - shift * (idx + 1) - sum(box_sizes[:idx + 1])
                cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
            for idx, box_size in enumerate(box_sizes_rev):
                start_x = mid_col + shift + bigshift_x
                start_y = mid_row + shift * (idx + 1) + sum(box_sizes_rev[:idx + 1])
                cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
                start_y = mid_row - shift * (idx + 1) - sum(box_sizes_rev[:idx + 1])
                cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
        return img

    def display_calibration_image(self, lw: int = 5) -> None:
        """Display a calibration image."""
        self.display_image(self.get_calibration_image(lw=lw))

    def get_checkerboard(
            self,
            square_size: int | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a checkerboard image."""
        if not square_size:
            square_size = self.DEFAULT_SQUARE_WIDTH
        img = self.get_zero_array(img_size=img_size)
        for i in range(0, img.shape[0], square_size * 2):
            img[i:i + square_size, :] = 255
        for j in range(square_size, img.shape[1], square_size * 2):
            img[:, j:j + square_size] = 255
        return img

    def get_circles(self, col_range: np.ndarray, row_range: np.ndarray, radius: int) -> np.ndarray:
        """Return a DMD image with filled circles at grid coordinates."""
        cols, rows = np.meshgrid(col_range, row_range)
        img = self.get_zero_array()
        for col, row in zip(cols.flatten(), rows.flatten()):
            cv2.circle(img, (int(col), int(row)), radius, color=255, thickness=-1)
        return img

    def display_circles(
            self,
            start_col: int,
            end_col: int,
            start_row: int,
            end_row: int,
            step_row: int,
            step_col: int,
            radius: int,
    ) -> None:
        """Display a regular grid of filled circles."""
        col_range = np.arange(start_col, end_col + step_col, step_col, dtype=np.dtype("int"))
        row_range = np.arange(start_row, end_row + step_row, step_row, dtype=np.dtype("int"))
        self.display_image(self.get_circles(col_range=col_range, row_range=row_range, radius=radius))

    def display_checkerboard(self, square_size: int | None = None) -> None:
        """Display a checkerboard image."""
        self.display_image(self.get_checkerboard(square_size=square_size))

    def display_circle(self, row: int, col: int, radius: int = 1) -> None:
        """Display one filled circle."""
        img = self.get_zero_array()
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)
        self.display_image(img)

    def display_half(self) -> None:
        """Display a central horizontal white band."""
        img = self.get_zero_array()
        img[img.shape[0] // 4:img.shape[0] * 3 // 4, :] = 255
        self.display_image(img)

    def display_line_vert(self, at_pos: int, line_width: int | None = None) -> None:
        """Display a vertical DMD line."""
        if not line_width:
            line_width = self.DEFAULT_LINE_WIDTH
        img = self.get_zero_array()
        row_start, row_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos,
            length=self.width_height_DMD[0] - 1,
        )
        img[row_start:row_end, :] = 255
        self.display_image(img)

    def display_line_horiz(self, at_pos: int, line_width: int | None = None) -> None:
        """Display a horizontal DMD line."""
        if not line_width:
            line_width = self.DEFAULT_LINE_WIDTH
        img = self.get_zero_array()
        col_start, col_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos,
            length=self.width_height_DMD[1] - 1,
        )
        img[:, col_start:col_end] = 255
        self.display_image(img)

    def display_crosshair(
            self,
            at_pos: tuple[int, int] | None = None,
            line_width: int | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> None:
        """Display a crosshair at a DMD position or at image center."""
        if img_size is None:
            img_size = self.width_height_DMD
        if at_pos is None:
            at_pos = (img_size[0] // 2, img_size[1] // 2)
        if line_width is None:
            line_width = self.DEFAULT_LINE_WIDTH
        img = self.get_zero_array(img_size=img_size)
        row_start, row_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos[0],
            length=img_size[0] - 1,
        )
        col_start, col_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos[1],
            length=img_size[1] - 1,
        )
        img[row_start:row_end, :] = 255
        img[:, col_start:col_end] = 255
        self.display_image(img)

    @staticmethod
    def _make_text(
            text: str,
            img_fraction: float,
            path_to_font: str,
            img_size: tuple[int, int],
    ) -> np.ndarray:
        """Return a DMD text image."""
        image_pil = Image.fromarray(np.transpose(np.zeros(img_size, dtype=np.uint8)))
        img_height, img_width = img_size
        font_size = 2
        font = ImageFont.truetype(path_to_font, font_size)
        while font.getlength(text) < img_fraction * image_pil.size[0]:
            font_size += 1
            font = ImageFont.truetype(path_to_font, font_size)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(path_to_font, font_size)
        draw.text((int(img_width / 2), int(img_height / 2)), text, fill=255, font=font, anchor="mm", align="center")
        return np.transpose(np.array(image_pil))

    def display_text(
            self,
            text: str | None = "Hello, World!",
            img_fraction: float | None = 0.5,
            path_to_font: str | None = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            img_size: tuple[int, int] | None = None,
    ) -> None:
        """Display text centered in a DMD image."""
        if not Path(path_to_font).exists():
            logger.error("Dmd.display_text: Path to font does not exist.")
            return
        if img_size is None:
            img_size = self.width_height_DMD
        img = self._make_text(text=text, img_fraction=img_fraction, path_to_font=path_to_font, img_size=img_size)
        self.display_image(img=img)

    def display_loaded_image(self) -> None:
        """Display the image previously loaded by load_image()."""
        if self._loaded_img is None:
            raise RuntimeError("display_loaded_image: No image loaded. Use load_image() first.")
        self.display_image(img=self._loaded_img)

    def load_image(self, filename: str, display_image: bool = True) -> np.ndarray:
        """Load an image file, convert/map it to DMD coordinates, and optionally display it."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"load_image: Provided filename {filename} does not exist.")
        extension = filename.split(".")[-1].lower()
        if extension not in self.EXTENSIONS:
            raise TypeError(f"load_image: File type {extension} not supported. Supported file types: {self.EXTENSIONS}.")

        img = skimage.io.imread(filename)
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img_range = np.max(img) - np.min(img)
                if img_range == 0:
                    img = np.zeros(img.shape, dtype=np.uint8)
                else:
                    img = (img - np.min(img)) / img_range
                    img = (img * 255).astype(np.uint8)
        elif img.ndim == 3:
            logger.info("load_image: Converting image using rgb2gray.")
            img = skimage.color.rgb2gray(img)
            img = (img * 255).astype(np.uint8)
        else:
            raise ValueError(f"load_image: Unsupported image format: {img}")

        if img.shape == self.width_height_CAM:
            logger.info("load_image: Mapping image using img_to_dmd_array.")
            img = self.img_to_dmd_array(img)
        elif img.shape != self.width_height_DMD:
            raise ValueError(f"load_image: Provided image {img.shape} is not of size "
                             f"{self.width_height_CAM} or {self.width_height_DMD}")
        self._loaded_img = img
        if display_image:
            self.display_image(img=img)
        return img


class DmdFactory:
    """Factory for DMD wrappers backed by configured peripheral controllers."""

    @staticmethod
    def create(
            config: DmdConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
            **binding_options: Any,
    ) -> Dmd:
        if not isinstance(config, DmdConfig):
            raise TypeError(f"DmdFactory.create: expected DmdConfig, received {type(config)}.")
        dmd_options = {
            "width_height_DMD": config.width_height_DMD,
            "width_height_CAM": config.width_height_CAM,
            "calibration_file": config.calibration_file,
        }

        if config.binding == BindingType.EM_DMD_WINDOW:
            if config.display_offset != (0, 0) or config.monitor_index is not None:
                raise NotImplementedError(
                    "DmdFactory.create: EM_DMD_WINDOW does not support display_offset or monitor_index."
                )
            from evomachine.bindings.em_dmd_window.dmd import EmDmdWindowDmd
            from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=EmDmdWindowPeripheralController,
                action="DmdFactory.create",
            )
            dmd = EmDmdWindowDmd(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or EmDmdWindowDmd.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **dmd_options,
                **binding_options,
            )
            dmd.config = config.copy()
            return dmd
        if config.binding == BindingType.PYGAME:
            from evomachine.bindings.pygame.dmd import PygameDmd
            from evomachine.bindings.pygame.peripheralcontroller import PygameDmdPeripheralController

            controller_options = {
                "size": config.width_height_DMD,
                "display_offset": config.display_offset,
                "monitor_index": config.monitor_index,
            }
            for key in list(controller_options):
                binding_options.pop(key, None)
            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=PygameDmdPeripheralController,
                action="DmdFactory.create",
            )
            peripheral_ctrl.configure_display(**controller_options)
            dmd = PygameDmd(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or PygameDmd.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **dmd_options,
                **binding_options,
            )
            dmd.config = config.copy()
            return dmd
        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.dmd import VirtualDmd
            from evomachine.bindings.virtual.dmd import VirtualDmdPeripheralController

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualDmdPeripheralController,
                action="DmdFactory.create",
            )
            dmd = VirtualDmd(
                peripheral_ctrl=peripheral_ctrl,
                name=config.name or VirtualDmd.DEFAULT_NAME,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
                **dmd_options,
                **binding_options,
            )
            dmd.config = config.copy()
            return dmd
        raise ValueError(f"DmdFactory.create: unsupported binding {config.binding}.")

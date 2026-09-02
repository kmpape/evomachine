from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import pickle as pkl
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import field_validator
import skimage.color
import skimage.io

from evomachine.bindings.binding_types import BindingType
from evomachine.config_models import EvoConfig
from evomachine.config import CAM_WIDTH_HEIGHT, DMD_WIDTH_HEIGHT, EVOMACHINE_DIR, get_logger
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral, PeripheralConfig
from evomachine.types import LEDType

logger = get_logger(name=__name__, is_peripheral=True)

ARR_TYPE = np.uint8
DMD_PATTERN_ALIASES = {"clear": "empty"}
DMD_BUILT_IN_PATTERNS = frozenset({
    "empty",
    "clear",
    "full",
    "rectangle",
    "checkerboard",
    "crosshair",
    "circle",
})


class DmdCalibrationConfig(EvoConfig):
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

    def model_post_init(self, __context) -> None:
        if not ((0 <= self.start_row) and (self.start_row < self.end_row) and (self.end_row < 2716)):
            raise ValueError("Indices must be within DMD boundaries.")
        if not ((0 <= self.start_col) and (self.start_col < self.end_col) and (self.end_col < 1600)):
            raise ValueError("Indices must be within DMD boundaries.")


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


class DmdConfig(PeripheralConfig):
    """Configuration for creating a DMD wrapper from a peripheral controller."""

    width_height_DMD: tuple[int, int] = DMD_WIDTH_HEIGHT
    width_height_CAM: tuple[int, int] = CAM_WIDTH_HEIGHT
    display_offset: tuple[int, int] = (0, 0)
    monitor_index: int | None = None
    calibration_file: Path | None = None

    @field_validator("width_height_DMD", "width_height_CAM", "display_offset", mode="before")
    @classmethod
    def _validate_size_field(cls, value: object, info) -> object:
        return cls._validate_size(
            value=value,
            field_name=info.field_name,
            allow_zero=info.field_name == "display_offset",
        )

    @field_validator("monitor_index", mode="before")
    @classmethod
    def _validate_monitor_index_type(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"DmdConfig: monitor_index must be int or None, received {type(value)}.")
        return value

    @field_validator("calibration_file", mode="before")
    @classmethod
    def _validate_calibration_file_type(cls, value: object) -> object:
        if value is not None and not isinstance(value, Path | str):
            raise TypeError(
                f"DmdConfig: calibration_file must be Path, str, or None, received {type(value)}."
            )
        return value

    def model_post_init(self, __context) -> None:
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
        super().model_post_init(__context)
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
            value: tuple[int, int] | object,
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


class DmdShapeConfig(EvoConfig):
    """Configuration for built-in DMD shape patterns."""

    rectangle_row: int | None = None
    rectangle_col: int | None = None
    rectangle_height: int | None = None
    rectangle_width: int | None = None
    checkerboard_box_size: int | None = None
    crosshair_row: int | None = None
    crosshair_col: int | None = None
    crosshair_width: int = 1
    circle_row: int | None = None
    circle_col: int | None = None
    circle_radius: int | None = None

    @field_validator(
        "rectangle_row",
        "rectangle_col",
        "rectangle_height",
        "rectangle_width",
        "checkerboard_box_size",
        "crosshair_row",
        "crosshair_col",
        "crosshair_width",
        "circle_row",
        "circle_col",
        "circle_radius",
        mode="before",
    )
    @classmethod
    def _validate_optional_int_type(cls, value: object, info) -> object:
        if value is None:
            return value
        if isinstance(value, bool):
            raise TypeError(f"DmdShapeConfig: {info.field_name} must be int or None, received bool.")
        if not isinstance(value, int):
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"DmdShapeConfig: {info.field_name} must be int or None, received {type(value)}."
                ) from exc
        return value

    def model_post_init(self, __context) -> None:
        for field_name in (
                "rectangle_row",
                "rectangle_col",
                "crosshair_row",
                "crosshair_col",
                "circle_row",
                "circle_col",
        ):
            self._validate_optional_int(field_name=field_name, minimum=0)
        for field_name in (
                "rectangle_height",
                "rectangle_width",
                "checkerboard_box_size",
                "crosshair_width",
                "circle_radius",
        ):
            self._validate_optional_int(field_name=field_name, minimum=1)

    def _validate_optional_int(self, field_name: str, minimum: int) -> None:
        value = getattr(self, field_name)
        if value is None:
            return
        if isinstance(value, bool):
            raise TypeError(f"DmdShapeConfig: {field_name} must be int or None, received bool.")
        if not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"DmdShapeConfig: {field_name} must be int or None, received {type(value)}.") from exc
            setattr(self, field_name, value)
        if value < minimum:
            raise ValueError(f"DmdShapeConfig: {field_name} must be >= {minimum}, received {value}.")


@dataclass
class DmdCalibrationData:
    """
    DMD calibration point correspondences used to compute homography transforms.

    ProjectionManager produces the stored point records. Dmd loads those records
    into this dataclass and uses them to compute the homography matrices.
    """

    dmd_points: list[tuple[int, int]]
    """DMD calibration points as (row, column) pixel coordinates."""

    cam_points: list[tuple[int, int]]
    """Camera-detected calibration points as (row, column) pixel coordinates."""

    cfg: DmdCalibrationConfig | None = None
    """Configuration used to produce the calibration data, if available."""

    path: Path | None = None
    """File path this calibration data was loaded from or saved to, if available."""

    @classmethod
    def from_stored_data(
            cls,
            stored_data: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]],
            cfg: DmdCalibrationConfig | None = None,
            path: Path | None = None,
    ) -> "DmdCalibrationData":
        """
        Convert stored calibration point records to DmdCalibrationData.

        Pickle file entries are stored as:
        ((dmd_row, dmd_col), (cam_row, cam_col), (row_intensity, col_intensity)).
        The intensity values are not required for homography calculation.
        """
        return cls(
            dmd_points=[dmd_point for dmd_point, _, _ in stored_data],
            cam_points=[cam_point for _, cam_point, _ in stored_data],
            cfg=cfg,
            path=path,
        )


@dataclass(frozen=True)
class LoadedDmdImageInfo:
    """Describe how a successfully loaded custom pattern was interpreted."""

    filename: Path
    source_shape: tuple[int, int]
    coordinate_space: Literal["camera", "dmd"]


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
    DEFAULT_SQUARE_WIDTH: int = 200
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
        self._loaded_img_info: LoadedDmdImageInfo | None = None
        default_calibration_file = EVOMACHINE_DIR / "calibration_data" / "dmd" / "dmd_calibration_data_2025-08-14_v2.pkl"
        packaged_calibration_file = EVOMACHINE_DIR / "evomachine" / "dmd_calibration_data.pkl"
        if not default_calibration_file.exists() and packaged_calibration_file.exists():
            default_calibration_file = packaged_calibration_file
        self._calib_file: Path = calibration_file or default_calibration_file
        self._calib_data: DmdCalibrationData | None = None
        self._homography_mat: np.ndarray | None = None
        self._homography_mat_inv: np.ndarray | None = None
        self.config: DmdConfig | None = None

        self._calib_data = self.load_calibration_data(self._calib_file)
        if self._calib_data is not None:
            self.calibrate()

    @staticmethod
    def load_calibration_data(
            path: Path,
    ) -> DmdCalibrationData | None:
        """
        Load raw calibration point correspondences from a pickle file.

        Parameters
        ----------
        path
            Pickle file containing the raw calibration point list.

        Returns
        -------
        DmdCalibrationData | None
            Loaded calibration data, or None when the file is missing.
        """
        if not path.exists():
            logger.error(f"Dmd.load_calibration_data: file {path} not found.")
            return None

        logger.info(f"Dmd.load_calibration_data: loading calibration data from {path}.")
        with open(str(path), "rb") as file:
            loaded_data = pkl.load(file)
        return DmdCalibrationData.from_stored_data(
            stored_data=loaded_data,
            path=path,
        )

    def calibrate(self) -> None:
        """
        Compute homography matrices from the loaded calibration data.

        Returns
        -------
        None
        """

        if self._calib_data is None or not self._calib_data.dmd_points or not self._calib_data.cam_points:
            raise ValueError("Dmd.calibrate: no calibration data found.")
        if len(self._calib_data.dmd_points) != len(self._calib_data.cam_points):
            raise ValueError(
                "Dmd.calibrate: dmd_points and cam_points must contain the same number of points."
            )
        if len(self._calib_data.dmd_points) < 4:
            raise ValueError("Dmd.calibrate: at least four point correspondences are required.")

        dmd_points = np.array([(col, row) for row, col in self._calib_data.dmd_points])
        cam_points = np.array([(col, row) for row, col in self._calib_data.cam_points])
        homography_mat, _ = cv2.findHomography(srcPoints=cam_points, dstPoints=dmd_points)
        homography_mat_inv, _ = cv2.findHomography(srcPoints=dmd_points, dstPoints=cam_points)
        if homography_mat is None or homography_mat_inv is None:
            raise ValueError("Dmd.calibrate: could not compute homography matrices from calibration data.")

        self._homography_mat = homography_mat
        self._homography_mat_inv = homography_mat_inv
        if self._calib_data.path is not None:
            self._calib_file = self._calib_data.path

    def calibrate_from_path(self, path: Path | None) -> None:
        """Load calibration data from a file and compute homography matrices."""
        if path is None:
            raise ValueError("Dmd.calibrate_from_path: path must not be None.")
        calibration_data = self.load_calibration_data(path)
        if calibration_data is None:
            raise FileNotFoundError(f"Dmd.calibrate_from_path: no calibration data found at {path}.")
        self._calib_file = path
        self._calib_data = calibration_data
        self.calibrate()

    def initialise(self, force: bool = False) -> None:
        """Initialise the underlying DMD peripheral controller."""
        logger.debug("Dmd.initialise: initialising %s with force=%s.", self.name, force)
        self.peripheral_ctrl.initialise(force=force)

    def finalise(self, force: bool = False) -> None:
        """Shutdown the underlying DMD peripheral controller."""
        logger.debug("Dmd.finalise: finalising %s with force=%s.", self.name, force)
        self.peripheral_ctrl.shutdown(force=force)

    def stop(self) -> None:
        """Blank the DMD display."""
        logger.debug("Dmd.stop: blanking %s.", self.name)
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
            logger.warning("%s: DMD peripheral controller is not initialised.", self.name)
            raise RuntimeError(f"{self.name}: DMD peripheral controller is not initialised.")
        if self.check_alive and not self.is_alive():
            logger.warning("%s: DMD peripheral controller is not alive.", self.name)
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
        logger.debug("Dmd.display_none: displaying black screen on %s.", self.name)
        self.display_image(self.get_zero_array())

    def display_full(self, force_display: bool = False) -> None:
        """Display a white screen."""
        if not force_display and self.is_full_display():
            logger.debug("Dmd.display_full: %s already full display; skipping.", self.name)
            return
        logger.debug("Dmd.display_full: displaying full screen on %s with force_display=%s.", self.name, force_display)
        self.display_image(self.get_one_array(), _is_full_display=True)

    def get_zero_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a black uint8 image."""
        return np.zeros(img_size or self.width_height_DMD, dtype=ARR_TYPE)

    def get_one_array(self, img_size: tuple[int, int] | None = None) -> np.ndarray:
        """Return a white uint8 image."""
        return np.ones(img_size or self.width_height_DMD, dtype=ARR_TYPE) * 255

    def get_pattern(
            self,
            pattern: str,
            config: DmdShapeConfig | None = None,
            warp: bool = True,
    ) -> np.ndarray:
        """Return a built-in pattern in calibrated DMD or native DMD coordinates."""
        if not isinstance(warp, bool):
            raise TypeError(f"Dmd.get_pattern: warp must be bool, received {type(warp)}.")
        pattern = DMD_PATTERN_ALIASES.get(pattern, pattern)
        if pattern not in DMD_BUILT_IN_PATTERNS:
            raise ValueError(f"Unsupported DMD pattern {pattern!r}.")
        config = config or DmdShapeConfig()
        img_size = self.width_height_CAM if warp else self.width_height_DMD
        if pattern == "empty":
            pattern_array = self.get_zero_array(img_size=img_size)
        elif pattern == "full":
            pattern_array = self.get_one_array(img_size=img_size)
        elif pattern == "rectangle":
            pattern_array = self.get_rectangle(config=config, img_size=img_size)
        elif pattern == "checkerboard":
            pattern_array = self.get_checkerboard(
                square_size=config.checkerboard_box_size,
                img_size=img_size,
            )
        elif pattern == "crosshair":
            pattern_array = self.get_crosshair(
                at_pos=self._shape_position_or_none(config.crosshair_row, config.crosshair_col),
                line_width=config.crosshair_width,
                img_size=img_size,
            )
        elif pattern == "circle":
            pattern_array = self.get_circle(
                row=config.circle_row,
                col=config.circle_col,
                radius=config.circle_radius,
                img_size=img_size,
            )
        else:
            raise ValueError(f"Unsupported DMD pattern {pattern!r}.")
        return self.img_to_dmd_array(pattern_array) if warp else pattern_array

    @staticmethod
    def _shape_position_or_none(row: int | None, col: int | None) -> tuple[int, int] | None:
        if row is None and col is None:
            return None
        if row is None or col is None:
            raise ValueError("DMD shape position requires both row and col.")
        return row, col

    @staticmethod
    def _centered_slice(center: int, width: int, length: int) -> tuple[int, int]:
        width = max(1, min(width, length))
        start = center - width // 2
        end = start + width
        if start < 0:
            end -= start
            start = 0
        if end > length:
            start = max(0, start - (end - length))
            end = length
        return start, end

    @staticmethod
    def _validate_position(row: int, col: int, img_shape: tuple[int, int], action: str) -> None:
        if not (0 <= row < img_shape[0]):
            raise ValueError(f"{action}: row {row} outside image bounds 0-{img_shape[0] - 1}.")
        if not (0 <= col < img_shape[1]):
            raise ValueError(f"{action}: col {col} outside image bounds 0-{img_shape[1] - 1}.")

    @staticmethod
    def _validate_rectangle(row: int, col: int, height: int, width: int, img_shape: tuple[int, int]) -> None:
        Dmd._validate_position(row=row, col=col, img_shape=img_shape, action="Dmd.get_rectangle")
        if row + height > img_shape[0]:
            raise ValueError(f"Dmd.get_rectangle: row + height {row + height} exceeds image rows {img_shape[0]}.")
        if col + width > img_shape[1]:
            raise ValueError(f"Dmd.get_rectangle: col + width {col + width} exceeds image cols {img_shape[1]}.")

    def get_rectangle(
            self,
            config: DmdShapeConfig | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a filled rectangle pattern."""
        config = config or DmdShapeConfig()
        img = self.get_zero_array(img_size=img_size)
        height = config.rectangle_height or max(1, img.shape[0] // 2)
        width = config.rectangle_width or max(1, img.shape[1] // 2)
        row = config.rectangle_row if config.rectangle_row is not None else max(0, (img.shape[0] - height) // 2)
        col = config.rectangle_col if config.rectangle_col is not None else max(0, (img.shape[1] - width) // 2)
        self._validate_rectangle(row=row, col=col, height=height, width=width, img_shape=img.shape)
        img[row:row + height, col:col + width] = 255
        return img

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
        if square_size <= 0:
            raise ValueError(f"Dmd.get_checkerboard: square_size must be positive, received {square_size}.")
        img = self.get_zero_array(img_size=img_size)
        row_tiles = np.arange(img.shape[0])[:, None] // square_size
        col_tiles = np.arange(img.shape[1])[None, :] // square_size
        img[(row_tiles + col_tiles) % 2 == 0] = 255
        return img

    def get_circles(
            self,
            col_range: np.ndarray,
            row_range: np.ndarray,
            radius: int,
            img_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a DMD image with filled circles at grid coordinates."""
        if radius <= 0:
            raise ValueError(f"Dmd.get_circles: radius must be positive, received {radius}.")
        cols, rows = np.meshgrid(col_range, row_range)
        img = self.get_zero_array(img_size=img_size)
        for col, row in zip(cols.flatten(), rows.flatten()):
            cv2.circle(img, (int(col), int(row)), radius, color=255, thickness=-1)
        return img

    def get_circle(
            self,
            row: int | None = None,
            col: int | None = None,
            radius: int | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a DMD image with one filled circle."""
        img = self.get_zero_array(img_size=img_size)
        row = img.shape[0] // 2 if row is None else row
        col = img.shape[1] // 2 if col is None else col
        radius = max(1, min(img.shape) // 8) if radius is None else radius
        if radius <= 0:
            raise ValueError(f"Dmd.get_circle: radius must be positive, received {radius}.")
        self._validate_position(row=row, col=col, img_shape=img.shape, action="Dmd.get_circle")
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)
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

    def display_rectangle(self, config: DmdShapeConfig | None = None) -> None:
        """Display a filled rectangle."""
        self.display_image(self.get_rectangle(config=config))

    def display_circle(self, row: int, col: int, radius: int = 1) -> None:
        """Display one filled circle."""
        self.display_image(self.get_circle(row=row, col=col, radius=radius))

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

    def get_crosshair(
            self,
            at_pos: tuple[int, int] | None = None,
            line_width: int | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a DMD crosshair image at a position or at image center."""
        if img_size is None:
            img_size = self.width_height_DMD
        if at_pos is None:
            at_pos = (img_size[0] // 2, img_size[1] // 2)
        if line_width is None:
            line_width = self.DEFAULT_LINE_WIDTH
        self._validate_position(row=at_pos[0], col=at_pos[1], img_shape=img_size, action="Dmd.get_crosshair")
        img = self.get_zero_array(img_size=img_size)
        row_start, row_end = self._centered_slice(center=at_pos[0], width=line_width, length=img_size[0])
        col_start, col_end = self._centered_slice(center=at_pos[1], width=line_width, length=img_size[1])
        img[row_start:row_end, :] = 255
        img[:, col_start:col_end] = 255
        return img

    def display_crosshair(
            self,
            at_pos: tuple[int, int] | None = None,
            line_width: int | None = None,
            img_size: tuple[int, int] | None = None,
    ) -> None:
        """Display a crosshair at a DMD position or at image center."""
        self.display_image(self.get_crosshair(at_pos=at_pos, line_width=line_width, img_size=img_size))

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

    def get_loaded_image_info(self) -> LoadedDmdImageInfo | None:
        """Return metadata for the most recently loaded custom pattern."""
        return self._loaded_img_info

    def load_image(self, filename: str, display_image: bool = True) -> np.ndarray:
        """Load an image file, convert/map it to DMD coordinates, and optionally display it."""
        self._loaded_img = None
        self._loaded_img_info = None
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

        source_shape = tuple(img.shape)
        if source_shape == self.width_height_CAM:
            logger.info("load_image: Mapping image using img_to_dmd_array.")
            img = self.img_to_dmd_array(img)
            coordinate_space = "camera"
        elif source_shape == self.width_height_DMD:
            coordinate_space = "dmd"
        else:
            raise ValueError(f"load_image: Provided image {img.shape} is not of size "
                             f"{self.width_height_CAM} or {self.width_height_DMD}")
        self._loaded_img = img
        self._loaded_img_info = LoadedDmdImageInfo(
            filename=Path(filename),
            source_shape=source_shape,
            coordinate_space=coordinate_space,
        )
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

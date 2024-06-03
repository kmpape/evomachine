import cv2
from enum import Enum
import logging
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import skimage
import subprocess
import sys
from typing import List, Optional, Union, Tuple

import numpy as np
import pickle as pkl
import pygame
import pygame.locals
import screeninfo

from evomachine.config import get_logger, EVOMACHINE_DIR
from evomachine.exceptions import DMDError, ErrorCode, ErrorContainer

from delta.utils import CroppingBox

logger = get_logger(name=__name__)


DMD_WIDTH_HEIGHT = (2716, 1600)  # Provide images with img.shape == DMD_WIDTH_HEIGHT
CAM_WIDTH_HEIGHT = (3200, 3200)

class DMDColor(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    @classmethod
    def get_all_values(cls) -> List[Tuple[int, int, int]]:
        return [member.value for member in cls]

    @classmethod
    def get_name(cls, value_to_find) -> str:
        for member in cls:
            if member.value == value_to_find:
                return str(member.name)
        return ""


class DMDControl:
    EXTENSIONS = ['png', 'tiff', 'tif']
    "Accepted file extensions for loading images."
    DEFAULT_LINE_WIDTH: int = 5
    "Line width used for calibration and displaying lines. Use odd values."

    def __init__(self, debug_mode: bool = False):
        """

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
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self._dmd_is_alive: bool = False
        "Flag set in initialise."
        self.offset_DMD: Tuple[int, int] = (0, 0)
        "Offset to display PyGame window. Set in initialise."
        self.width_height_DMD: Tuple[int, int] = DMD_WIDTH_HEIGHT
        "Size of DMD."
        self.width_height_CAM: tuple[int, int] = CAM_WIDTH_HEIGHT
        "Size of camera."
        self.surface: Union[None, pygame.surface] = None
        "PyGame object to display images. Initialised in initialise."
        self.default_line_width: int = 5
        "Line width used for calibration and displaying lines. Use odd values."
        self._calib_data: Optional[List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = None
        "List containing calibration data."
        self._calib_file: Path = EVOMACHINE_DIR / 'dmd_calibration_data.pkl'
        "Path to calibration file."
        self._homography_mat: Optional[np.ndarray] = None
        "Homography matrix for mapping image to DMD coordinates."
        self._is_display_full: bool = False
        "Internal flag queried through is_full_display() that is set to True when displaying a full white screen."
        self.debug_mode: bool = debug_mode
        "Flag to set test environment or use DMD functions without displaying on the actual DMD window."

    def _load_calibration_data(self, filepath: Path | None = None) -> bool:
        if filepath is None:  # noqa
            filepath = self._calib_file
        if not filepath.exists():
            logger.error(f"DMDControl._load_calibration_data: file {filepath} not found.")
            return False
        logger.info(f"DMDControl._load_calibration_data: loading calibration data from {filepath}.")
        with open(str(filepath), 'rb') as f:
            self._calib_data = pkl.load(f)

        dmd_points = np.array([(c_dmd, r_dmd) for ((r_dmd, c_dmd), _, _) in self._calib_data])
        cam_points = np.array([(c_cam, r_cam) for (_, (r_cam, c_cam), _) in self._calib_data])
        self._homography_mat, _ = cv2.findHomography(srcPoints=cam_points, dstPoints=dmd_points)

        points_cam = np.array([[[0, 0], [3199, 3199]]], dtype=np.float32)
        points_dmd = cv2.perspectiveTransform(points_cam.reshape(-1, 1, 2), self._homography_mat)  # noqa
        logger.info(f"DMDControl._load_calibration_data: mapping point ("
                    f"{int(points_cam[0][0][0])},{int(points_cam[0][0][1])}) to "
                    f"{int(points_dmd[0][0][0])},{int(points_dmd[0][0][1])}) and "
                    f"{int(points_cam[0][1][0])},{int(points_cam[0][1][1])}) to "
                    f"{int(points_dmd[1][0][0])},{int(points_dmd[1][0][1])}).")
        return True

    def initialise(self, is_test: bool = False):
        if self._dmd_is_alive:
            logger.warning("DMDControl.initialise: DMD already initialised.")
            return
        if self.debug_mode:
            if not self._load_calibration_data():
                logger.info("DMDControl.initialise: no calibration data loaded.")
            self._dmd_is_alive = True
            return
        monitors = screeninfo.get_monitors()
        mon_info = "\n".join(m.__str__() for m in monitors)
        has_two_monitors = len(monitors) == 2
        if (not is_test) and has_two_monitors:
            has_one_primary = any(m.is_primary for m in monitors) and any(not m.is_primary for m in monitors)
            if has_one_primary:
                mon_dmd = [m for m in monitors if (not m.is_primary)][0]
                self.offset_DMD = (mon_dmd.x, mon_dmd.y)
                is_correct_size = all(x1 == x2
                                      for (x1, x2) in zip(self.width_height_DMD, (mon_dmd.width, mon_dmd.height)))
                if is_correct_size:
                    pygame.init()
                    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.offset_DMD[0]}, {self.offset_DMD[1]}"
                    self.surface = pygame.display.set_mode(
                        size=self.width_height_DMD,
                        flags=pygame.NOFRAME | pygame.FULLSCREEN,
                    )
                    window_id = pygame.display.get_wm_info()["window"]
                    subprocess.Popen(["wmctrl", "-i", "-r", str(window_id), "-b", "add,above"])
                    self._dmd_is_alive = True
                    self.display_none()
                    if not self._load_calibration_data():
                        logger.info("DMDControl.initialise: no calibration data loaded.")
                    logging.info(f"DMD: initialised at pos={self.offset_DMD} with size={self.width_height_DMD}.")
                else:  # Wrong DMD size (or wrong monitor selected)
                    msg = f"DMDControl.initialise: incorrect DMD size: {mon_dmd}."
                    logger.warning(msg)
                    self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))
            else:  # No primary monitor found
                msg = f"DMDControl.initialise: No primary monitor found: {mon_info}."
                logger.warning(msg)
                self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))
        else:  # Wrong number of monitors
            msg = f"DMDControl.initialise: found {len(monitors)}  monitor(s) (instead of 2). {mon_info}."
            logger.warning(msg)
            self.surface = pygame.display.set_mode(
                size=(300, 300),
                flags=pygame.NOFRAME,
            )
            self._dmd_is_alive = True
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))

    def is_initialised(self) -> bool:
        return self._dmd_is_alive

    def finalise(self):
        if self.debug_mode:
            return
        if not self._dmd_is_alive:
            logger.warning("DMDControl.finalise: DMD not initialised.")
            return
        DMDControl.close_window()
        self._dmd_is_alive = False

    @staticmethod
    def close_window():
        pygame.quit()

    def display_image(
            self,
            img: np.ndarray[(int, int), int],
            update_display: Optional[bool] = True,
    ):
        if self.debug_mode:
            return
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_image: DMD not initialised. Try running DMDControl.initialise.")
            return
        if img.shape == self.width_height_DMD:
            self.surface.blit(pygame.surfarray.make_surface(np.repeat(img[:, :, np.newaxis], 3, axis=-1)), (0, 0))
        elif img.shape == (*self.width_height_DMD, 3):
            self.surface.blit(pygame.surfarray.make_surface(img), (0, 0))
        else:
            logger.error(f"DMDControl.display_image: provided image of shape={img.shape}, "
                         f"but DMD shape={(*self.width_height_DMD, 3)}.")
            return
        if update_display:
            pygame.display.update()
        self._is_display_full = False

    def display_full(
            self,
            update_display: Optional[bool] = True,
            color: Optional[DMDColor] = DMDColor.WHITE,
            force_display: bool = False,
    ):
        if self.debug_mode:
            return
        if not force_display and self.is_display_full():
            return
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_full: DMD not initialised. Try running DMDControl.initialise.")
            return
        self.surface.fill(color.value)
        if update_display:
            pygame.display.update()
        self._is_display_full = color == DMDColor.WHITE

    def display_none(self, update_display: Optional[bool] = True):
        self.display_full(update_display=update_display, color=DMDColor.BLACK)

    def display_checkerboard(
            self,
            square_size: int | None = None,
    ):
        """
        Display a checkerboard with squares of size square_size.

        Parameters
        ----------
        square_size: int             Thickness of line (see _make_half_line_width)

        """
        if not square_size:
            square_size = DMDControl.DEFAULT_LINE_WIDTH
        img = self.get_zero_array()
        for i in range(0, img.shape[0], square_size * 2):
            img[i:i + square_size, :] = 255
        for j in range(square_size, img.shape[1], square_size * 2):
            img[:, j:j + square_size] = 255
        self.display_image(img)

    def display_circle(
            self,
            row: int,
            col: int,
            radius: int = 1,
    ):
        img = np.zeros(self.width_height_DMD, dtype=np.uint8)
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)  # noqa
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        self.display_image(img=img)

    def display_line(
            self,
            start_pos: Tuple[int, int],
            end_pos: Tuple[int, int],
            line_width: Optional[Union[int, None]] = None,
            update_display: Optional[bool] = True,
            color: Optional[DMDColor] = DMDColor.WHITE,
    ):
        if self.debug_mode:
            return
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_line: DMD not initialised. Try running DMDControl.initialise.")
            return
        if not line_width:
            line_width = DMDControl.DEFAULT_LINE_WIDTH
        pygame.draw.line(
            surface=self.surface,
            color=color.value,
            start_pos=start_pos,
            end_pos=end_pos,
            width=line_width,
        )
        if update_display:
            pygame.display.update()
        self._is_display_full = False

    def display_line_horiz(
            self,
            at_pos: int,
            line_width: Optional[Union[int, None]] = None,
            update_display: Optional[bool] = True,
            color: Optional[DMDColor] = DMDColor.WHITE,
    ):
        self.display_line(
            start_pos=(0, at_pos),
            end_pos=(self.width_height_DMD[0], at_pos),
            line_width=line_width,
            update_display=update_display,
            color=color,
        )

    def display_line_vert(
            self,
            at_pos: int,
            line_width: Optional[Union[int, None]] = None,
            update_display: Optional[bool] = True,
            color: Optional[DMDColor] = DMDColor.WHITE,
    ):
        self.display_line(
            start_pos=(at_pos, 0),
            end_pos=(at_pos, self.width_height_DMD[1]),
            line_width=line_width,
            update_display=update_display,
            color=color,
        )

    def display_crosshair(self, line_width: int = 1, update_display: Optional[bool] = True):
        img = np.zeros((*self.width_height_DMD, 3), dtype=int)
        center = (np.floor_divide(self.width_height_DMD[0], 2), np.floor_divide(self.width_height_DMD[1], 2))
        img[center[0] - int(np.floor(line_width / 2)): center[0] + int(np.ceil(line_width / 2)), :, :] = 255
        img[:, center[1] - int(np.floor(line_width / 2)): center[1] + int(np.ceil(line_width / 2)), :] = 255
        self.display_image(img=img, update_display=update_display)

    def display_text(
            self,
            text: Optional[str] = "Hello, World!",
            img_fraction: Optional[float] = 0.5,
            path_to_font: Optional[str] = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ):
        image_pil = Image.fromarray(np.transpose(np.zeros(self.width_height_DMD, dtype=np.uint8)))  # noqa
        img_height, img_width = self.width_height_DMD
        font_size = 2
        font = ImageFont.truetype(path_to_font, font_size)
        while font.getlength(text) < img_fraction * image_pil.size[0]:
            font_size += 1
            font = ImageFont.truetype(path_to_font, font_size)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(path_to_font, font_size)
        draw.text((int(img_width / 2), int(img_height / 2)), text, fill=255, font=font, anchor='mm')
        img = np.transpose(np.array(image_pil))
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        self.display_image(img=img)

    @staticmethod
    def get_zero_array():
        return np.zeros(DMD_WIDTH_HEIGHT, dtype=np.uint8)

    def is_display_full(self) -> bool:
        return self._is_display_full

    def img_to_dmd_coords(self, img_row: int, img_col: int) -> tuple[int, int] | None:
        """
        Transform coordinates on the camera to coordinates on the DMD.

        Parameters
        ----------
        img_row : int
            Camera Y coordinate.
        img_col : int
            Camera X coordinate.
        Returns
        -------
        dmd_row_col : tuple[int, int]
            DMD coordinates as (row, col).
        """
        if self._homography_mat is None:
            logger.error(f"img_to_dmd_coords: no calibration data provided.")
            return None
        point_cam = np.array([[[img_col, img_row]]]).astype(float)
        point_dmd = cv2.perspectiveTransform(point_cam, self._homography_mat)
        return int(np.round(point_dmd[0][0][1])), int(np.round(point_dmd[0][0][0]))

    def img_to_dmd_array(self, img: np.array) -> np.ndarray | None:
        """
        Transform a 3200 x 3200 camera pattern to a DMD pattern.

        Example: Project a square of size 100 in the top left corner of your image

        pattern_img = self.get_zero_array((3200, 3200))
        pattern_img[0:101, 0:101] = 255
        pattern_dmd = self.img_to_dmd_array(pattern_img)
        self.display_image(pattern_dmd)

        Parameters
        ----------
        img : np.ndarray
            2D image array.
        Returns
        -------
        dmd_img : np.ndarray
            2D image array to be displayed on the DMD using display_image().
        """
        if self._homography_mat is None:
            logger.error(f"img_to_dmd_coords: no calibration data provided.")
            return None
        if img.shape != (3200, 3200):
            logger.error(f"img_to_dmd_array: Expected image of shape (3200,3200) but received {img.shape}.")
            return self.get_zero_array()
        return cv2.warpPerspective(img, self._homography_mat, self.width_height_DMD[::-1]).astype(img.dtype)

    def pattern_from_roi_boxes(self, boxes: list[CroppingBox]) -> np.ndarray:
        """
        Creates a pattern from a list of cropping boxes (Image coordinates) and returns a warped DMD pattern.

        Parameters
        ----------
        boxes : list[CroppingBox]
            Cropping boxes to display pattern on.

        Returns
        -------
        warped_image : np.ndarray
            Warped image ready to be projected via DMD.
        """
        cam_img = self.get_zero_array()
        for b in boxes:
            cam_img[b.ytl: b.ybr+1, b.xtl: b.xbr+1] = 255
        return self.img_to_dmd_array(cam_img)

    def load_image(self, filename: str):
        if not os.path.exists(filename):  # noqa
            raise FileNotFoundError(f"load_image: Provided filename {filename} does not exist.")
        if not filename.split('.')[-1].lower() in self.EXTENSIONS:
            raise TypeError(f"load_image: File type {filename.split('.')[-1].lower()} not supported. "
                            f"Supported file types: {self.EXTENSIONS}.")
        img = skimage.io.imread(filename)
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
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
            raise ValueError(f"load_image: Provided image {img.shape} is not of size {self.width_height_CAM} or "
                             f"{self.width_height_DMD}")
        self.display_image(img=img)

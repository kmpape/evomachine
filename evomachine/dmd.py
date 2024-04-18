import cv2
from enum import Enum
import logging
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
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

logger = get_logger(name=__name__)


DMD_WIDTH_HEIGHT = (2716, 1600)  # Provide images with img.shape == DMD_WIDTH_HEIGHT

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
    # FIXME any DMD call from outside of the main thread yields to a crash

    DEFAULT_LINE_WIDTH: int = 5
    "Line width used for calibration and displaying lines. Use odd values."

    def __init__(self):
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
        "Size of DMD. Double-checked in initialise."
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

        # self.initialise()  # TODO remove this

    def _load_calibration_data(self, filepath: Optional[Path] = None) -> bool:
        if filepath is None:
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
        points_dmd = cv2.perspectiveTransform(points_cam.reshape(-1, 1, 2), self._homography_mat)
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
        if not self._dmd_is_alive:
            logger.warning("DMDControl.finalise: DMD not initialised.")
            return
        DMDControl.close_window()
        self._dmd_is_alive = False

    @staticmethod
    def close_window():
        pygame.quit()

    def display_image(self, img: np.ndarray[(int, int), int], update_display: Optional[bool] = True):
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

    def display_full(
            self,
            update_display: Optional[bool] = True,
            color: Optional[DMDColor] = DMDColor.WHITE,
    ):
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_full: DMD not initialised. Try running DMDControl.initialise.")
            return
        self.surface.fill(color.value)
        if update_display:
            pygame.display.update()

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
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)
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
        image_pil = Image.fromarray(np.transpose(np.zeros(self.width_height_DMD, dtype=np.uint8)))
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


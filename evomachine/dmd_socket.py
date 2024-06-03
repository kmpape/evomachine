import cv2
import logging
import numpy as np
import os
from pathlib import Path
import pickle as pkl
from PIL import Image, ImageFont, ImageDraw
import screeninfo
import skimage
import subprocess
import socket
import time
from typing import Dict, List, Optional, Union, Tuple
from threading import Thread

from evomachine.config import get_logger, EVOMACHINE_DIR
from evomachine.exceptions import DMDError, ErrorCode, ErrorContainer

from delta.utils import CroppingBox

logger = get_logger(name=__name__)


# NOTE: If modified, these parameters must also be modified in the C code.
DMD_WIDTH_HEIGHT = (2716, 1600)
CAM_WIDTH_HEIGHT = (3200, 3200)
PORT = 12345
HOST = '127.0.0.1'
MAX_BYTE_SIZE = 65482
NUM_CHUNKS = 97
CHUNK_ROWS = int(DMD_WIDTH_HEIGHT[0] / NUM_CHUNKS)
ARR_TYPE = np.uint8
EM_DMD_PROGRAM_PATH = EVOMACHINE_DIR.resolve().parent.parent / "em_dmd_window/Release/evomachine_dmd_window"
# /home/hslab/workspace_python/evomachine_v0/evomachine/scripts/../evomachine.
# TODO make test version that opens on the same screen


class DMDControl:
    DEFAULT_LINE_WIDTH: int = 5
    "Line width used for calibration and displaying lines. Use odd values."
    EXTENSIONS = ['png', 'tiff', 'tif']
    "Accepted file extensions for loading images."

    def __init__(self, debug_mode: bool = False):
        """
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
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self._is_initialised: bool = False
        "Flag set in initialise."
        self.s: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        "Socket to connect with C program."
        self.default_line_width: int = 5
        "Line width used for calibration and displaying lines. Use odd values."
        self._process: subprocess.Popen | None = None
        "Process for C program."
        self._output_thread: Thread | None = None
        "Thread to display output from C program."
        self._calib_data: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] | None = None
        "List containing calibration data."
        self._calib_file: Path = EVOMACHINE_DIR / 'dmd_calibration_data.pkl'
        "Path to calibration file."
        self._homography_mat: np.ndarray | None = None
        "Homography matrix for mapping image to DMD coordinates."
        self._homography_mat_inv: np.ndarray | None = None
        "Homography matrix for mapping DMD to image coordinates."
        self.debug_mode: bool = debug_mode
        "Flag to set test environment or use DMD functions without displaying on the actual DMD window."
        self.width_height_DMD: tuple[int, int] = DMD_WIDTH_HEIGHT
        "Size of DMD."
        self.width_height_CAM: tuple[int, int] = CAM_WIDTH_HEIGHT
        "Size of camera."
        self._is_full_display: bool = False
        "Internal flag queried through is_full_display() that is set to True when displaying a full white screen."
        self._loaded_img: np.ndarray | None = None
        "Image loaded through load_image."

    def _load_calibration_data(self, filepath: Optional[Path] = None) -> bool:
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
        self._homography_mat_inv, _ = cv2.findHomography(srcPoints=dmd_points, dstPoints=cam_points)

        points_cam = np.array([[[0, 0], [3199, 3199]]], dtype=np.float32)
        points_dmd = cv2.perspectiveTransform(points_cam.reshape(-1, 1, 2), self._homography_mat)   # noqa
        logger.info(f"DMDControl._load_calibration_data: mapping point "
                    f"({int(points_cam[0][0][0])},{int(points_cam[0][0][1])}) to "
                    f"({int(points_dmd[0][0][0])},{int(points_dmd[0][0][1])}) and "
                    f"({int(points_cam[0][1][0])},{int(points_cam[0][1][1])}) to "
                    f"({int(points_dmd[1][0][0])},{int(points_dmd[1][0][1])}).")
        return True

    def _launch_dmd_window(self):
        if self.debug_mode:
            return
        self._process = subprocess.Popen(
            [str(EM_DMD_PROGRAM_PATH.resolve())],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(1)

    def _send_image(self, img: np.ndarray):
        """
        Sends an image over the socket to the C program. Note that we are allocating the image
        as width (rows) x height (columns), so the transpose is sent here.

        Parameters
        ----------
        img: np.ndarray     Image must be of ARR_TYPE and of size DMD_WIDTH_HEIGHT.
        """
        if self.debug_mode:
            return
        self.s.sendall(img.transpose().tobytes())

    def _connect_socket(self):
        """
        This function opens a socket. Note that after calling s.close(), e.g. after a restart, re-opening a socket
        throws an error. The error is therefore caught once.
        """
        if self.debug_mode:
            return
        try:
            self.s.connect((HOST, PORT))
        except OSError as e:
            logger.info(f"Received error {str(e)} on opening socket. Retrying once.")
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((HOST, PORT))

    def _connection_test(self) -> bool:
        """
        Hard-coded enumeration test required after launching the C program.
        """
        if self.debug_mode:
            return True
        try:
            test_arr = np.zeros(DMD_WIDTH_HEIGHT, dtype=np.uint8)  # ROW MAJOR FORMAT
            for i in range(DMD_WIDTH_HEIGHT[0]):
                test_arr[i, :] = i % 255
            self.s.sendall(test_arr.tobytes())
            return True
        except ConnectionResetError as e:
            msg = f"Error connection test: {e}"
            logger.error(msg)
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_SOCKET))
            return False

    def get_calibration_data(self) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        return self._calib_data

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
        Transform a camera pattern of size width_height_CAM to a DMD pattern of size width_height_DMD.

        Example: Project a square of size 100 in the top left corner of your image

        pattern_img = self.get_zero_array(width_height_CAM)
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
            logger.error(f"img_to_dmd_array: no calibration data provided.")
            return None
        if img.shape != self.width_height_CAM:
            logger.error(f"img_to_dmd_array: Expected image of shape {self.width_height_CAM} but received {img.shape}.")
            return None
        return cv2.warpPerspective(img, self._homography_mat, self.width_height_DMD[::-1]).astype(img.dtype)

    def dmd_to_img_coords(self, img_row: int, img_col: int) -> tuple[int, int] | None:
        """
        Transform coordinates on the DMD to coordinates on the camera. Note that DMD coordinates lying outside of the
        camera display will yield coordinates outside of the range [0, 3200).

        Parameters
        ----------
        img_row : int
            DMD Y coordinate.
        img_col : int
            DMD X coordinate.
        Returns
        -------
        cam_row_col : tuple[int, int]
            Camera coordinates as (row, col).
        """
        if self._homography_mat_inv is None:
            logger.error(f"dmd_to_img_coords: no calibration data provided.")
            return None
        point_dmd = np.array([[[img_col, img_row]]]).astype(float)
        point_cam = cv2.perspectiveTransform(point_dmd, self._homography_mat_inv)
        return int(np.round(point_cam[0][0][1])), int(np.round(point_cam[0][0][0]))

    def dmd_to_img_array(self, img: np.array) -> np.ndarray | None:
        """
        Transform a width_height_DMD DMD pattern to a camera pattern.

        Parameters
        ----------
        img : np.ndarray
            2D image array of size width_height_DMD.
        Returns
        -------
        cam_img : np.ndarray
            3200 x 3200 camera array.
        """
        if self._homography_mat_inv is None:
            logger.error(f"dmd_to_img_array: no calibration data provided.")
            return None
        if img.shape != self.width_height_DMD:
            logger.error(f"dmd_to_img_array: Expected image of shape {self.width_height_DMD} but received {img.shape}.")
            return None
        return cv2.warpPerspective(img, self._homography_mat_inv, self.width_height_CAM).astype(img.dtype)

    def pattern_from_roi_boxes(self, boxes: list[CroppingBox], fill_x: float = 1.0, fill_y: float = 1.0) -> np.ndarray:
        """
        Creates a pattern from a list of cropping boxes (Image coordinates) and returns a warped DMD pattern.

        Parameters
        ----------
        boxes : list[CroppingBox]
            Cropping boxes to display pattern on.
        fill_x : float
            If fill_x=1.0, the entire cropping box is filled in horizontal direction. If 0.0 < fill_x < 1.0, it fills
            a fill percentage of the cropping box.
        fill_y : float
            Same as fill_x but in vertical direction.

        Returns
        -------
        warped_image : np.ndarray
            Warped image ready to be projected via DMD.
        """
        cam_img = self.get_zero_array(img_size=self.width_height_CAM)
        for b in boxes:
            shift_x = int(np.round(0.5 * (1-fill_x) * (b.xbr - b.xtl), 0))
            shift_y = int(np.round(0.5 * (1-fill_y) * (b.ybr - b.ytl), 0))
            start_row = max(b.ytl+shift_y, 0)
            end_row = min(b.ybr+1-shift_y, cam_img.shape[0]-1)
            start_col = max(b.xtl+shift_x, 0)
            end_col = min(b.xbr+1-shift_x, cam_img.shape[1]-1)
            cam_img[start_row: end_row, start_col: end_col] = 255
        return self.img_to_dmd_array(cam_img)

    def initialise(self, is_test: bool = False):
        if self.debug_mode:
            logger.debug(f"DMDControl.initialise: initialising DMD (debug_mode={self.debug_mode})")
            if not self._load_calibration_data():
                logger.info("DMDControl.initialise: no calibration data loaded.")
            self._is_initialised = True
            return
        else:
            logger.info(f"DMDControl.initialise: initialising DMD.")
        try:
            self._launch_dmd_window()
        except Exception as e:
            msg = f"Error launch DMD window: {e}"
            logger.error(msg)
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_SOCKET))
            return
        monitors = screeninfo.get_monitors()
        mon_info = "\n".join(m.__str__() for m in monitors)
        # TODO removed all screeninfo checks after switching DMD to X Screen 1 (not recognised by screeninfo)
        # has_two_monitors = len(monitors) == 2
        has_two_monitors = True
        if (not is_test) and has_two_monitors:
            # has_one_primary = any(m.is_primary for m in monitors) and any(not m.is_primary for m in monitors)
            has_one_primary = True
            if has_one_primary:
                mon_dmd = [m for m in monitors if (not m.is_primary)][0]
                # is_correct_size = all(x1 == x2
                #                       for (x1, x2) in zip(DMD_WIDTH_HEIGHT, (mon_dmd.width, mon_dmd.height)))
                is_correct_size = True
                if is_correct_size:
                    try:
                        self._connect_socket()
                        if self._connection_test():
                            self._is_initialised = True
                            logger.info(f"DMD: initialised with size={DMD_WIDTH_HEIGHT}.")
                        if not self._load_calibration_data():
                            logger.info("DMDControl.initialise: no calibration data loaded.")
                    except ConnectionError as e:
                        msg = f"Error connection to DMD C socket: {e}"
                        logger.error(msg)
                        self.error_container.add_error(new_error=DMDError(message=msg,
                                                                          error_code=ErrorCode.ERROR_SOCKET))
                else:  # Wrong DMD size (or wrong monitor selected)
                    msg = f"DMDControl.initialise: incorrect DMD size: {mon_dmd}."
                    logger.error(msg)
                    self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))
            else:  # No primary monitor found
                msg = f"DMDControl.initialise: No primary monitor found: {mon_info}."
                logger.error(msg)
                self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))
        else:  # Wrong number of monitors
            msg = f"DMDControl.initialise: found {len(monitors)}  monitor(s) (instead of 2). {mon_info}."
            logger.error(msg)
            # FIXME this is allowed for the TestCamera
            self._is_initialised = True
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))

    def is_full_display(self) -> bool:
        return self._is_full_display

    def is_initialised(self) -> bool:
        return self._is_initialised

    def finalise(self):
        """
        Closes connection with the C program and the program itself.
        """
        if self.debug_mode:
            return
        if not self._is_initialised:
            logger.warning("DMDControl.finalise: DMD not initialised. Attempting to close connection anyway.")
        # self._output_thread.join()
        self.s.close()
        time.sleep(0.5)
        if (self._process is not None) and (self._process.poll() is None):
            self._process.terminate()
        self._is_initialised = False

    def display_none(self):
        """
        Displays a black screen.
        """
        self.display_image(np.zeros(DMD_WIDTH_HEIGHT, dtype=ARR_TYPE))

    def display_full(self, force_display: bool = False):
        """
        Displays a white screen.

        """
        if not force_display and self.is_full_display():
            return
        self.display_image(np.ones(DMD_WIDTH_HEIGHT, dtype=ARR_TYPE)*255, _is_full_display=True)

    def display_image(self, img: np.ndarray[(int, int), ARR_TYPE], _is_full_display: bool = False):
        """

        Parameters
        ----------
        img : np.ndarray
            Image must be of ARR_TYPE.
        _is_full_display : bool
            Internal flag. Do not modify.

        Returns
        -------

        """
        if not self._is_initialised:
            logger.error(f"DMDControl.display_image: DMD not initialised. Try running DMDControl.initialise.")
            return
        if img.dtype != ARR_TYPE:
            logger.error(f"Image must be of type {ARR_TYPE}. Received {img.dtype}. Returning.")
            return
        if img.shape == DMD_WIDTH_HEIGHT:
            self._send_image(img)
        elif img.shape == (*DMD_WIDTH_HEIGHT, 3):
            logger.warning(f"DMDControl.display_image: B/W image expected. Sending image[:,:,0] instead.")
            self._send_image(img[:, :, 0])
            self._is_full_display = _is_full_display
        else:
            logger.error(f"DMDControl.display_image: provided image of shape={img.shape}, "
                         f"but DMD shape={DMD_WIDTH_HEIGHT}.")

    @staticmethod
    def get_zero_array(img_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if img_size is None:
            img_size = DMD_WIDTH_HEIGHT
        return np.zeros(img_size, dtype=ARR_TYPE)

    @staticmethod
    def _make_half_line_width(line_width: int, at_pos: int, length: int) -> Tuple[int, int]:
        if line_width == 1:
            return at_pos, min(at_pos+line_width, length)
        elif line_width % 2 == 0:
            return max(0, at_pos+1-int(line_width/2)), min(length, at_pos+int(line_width/2))
        else:
            return max(0, at_pos-int(line_width/2)), min(length, at_pos+int(line_width/2))

    def display_calibration_image(self, lw: int = 5):
        img = self.get_zero_array()
        mid_row, mid_col = img.shape[0]//2, img.shape[1]//2
        cv2.line(img, (mid_col, 0), (mid_col, img.shape[0]), 255, lw)  # noqa
        cv2.line(img, (0, mid_row), (img.shape[1], mid_row), 255, lw)  # noqa
        box_sizes = [5, 10, 20, 40, 80, 160, 320]
        box_sizes_rev = box_sizes[::-1]
        shift = 20
        for idx, box_size in enumerate(box_sizes):
            start_x = mid_col - shift - box_size
            start_y = mid_row + shift * (idx+1) + sum(box_sizes[:idx+1])  # noqa
            cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
            start_y = mid_row - shift * (idx+1) - sum(box_sizes[:idx+1])
            cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
        for idx, box_size in enumerate(box_sizes_rev):
            start_x = mid_col + shift
            start_y = mid_row + shift * (idx+1) + sum(box_sizes_rev[:idx+1])  # noqa
            cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
            start_y = mid_row - shift * (idx+1) - sum(box_sizes_rev[:idx+1])
            cv2.rectangle(img, (start_x, start_y), (start_x + box_size, start_y + box_size), 255, -1)
        self.display_image(img)

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
        img = self.get_zero_array()
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)  # noqa
        self.display_image(img)

    def display_half(self):
        img = self.get_zero_array()
        img[img.shape[0]//4:img.shape[0]*3//4, :] = 255
        self.display_image(img)

    def display_line_vert(
            self,
            at_pos: int,
            line_width: Optional[Union[int, None]] = None,
    ):
        """

        Parameters
        ----------
        at_pos: int                 Line position (row)
        line_width: int             Thickness of line (see _make_half_line_width)

        """
        if not line_width:
            line_width = DMDControl.DEFAULT_LINE_WIDTH
        img = self.get_zero_array()
        row_start, row_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos,
            length=DMD_WIDTH_HEIGHT[0]-1,
        )
        img[row_start:row_end, :] = 255
        self.display_image(img)

    def display_line_horiz(
            self,
            at_pos: int,
            line_width: Optional[Union[int, None]] = None,
    ):
        """

        Parameters
        ----------
        at_pos: int                 Line position (column)
        line_width: int             Thickness of line (see _make_half_line_width)

        """
        if not line_width:
            line_width = DMDControl.DEFAULT_LINE_WIDTH
        img = self.get_zero_array()
        col_start, col_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos,
            length=DMD_WIDTH_HEIGHT[1]-1,
        )
        img[:, col_start:col_end] = 255
        self.display_image(img)

    def display_crosshair(
            self,
            at_pos: Optional[Tuple[int, int]] = None,
            line_width: Optional[Union[int, None]] = None,
            img_size: Optional[Tuple[int, int]] = None,
    ):
        """

        Parameters
        ----------
        at_pos: Tuple[int, int]     Tuple with crosshair position (row, column and NOT x, y)
        line_width: int             Thickness of line (see _make_half_line_width)
        img_size: Tuple[int, int]   Note that changing the image size here requires to change it in the C program too
        """
        if img_size is None:
            img_size = DMD_WIDTH_HEIGHT
        img = self.get_zero_array(img_size=img_size)
        row_start, row_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos[0],
            length=img_size[0]-1,
        )
        col_start, col_end = self._make_half_line_width(
            line_width=line_width,
            at_pos=at_pos[1],
            length=img_size[1]-1,
        )
        img[row_start:row_end, :] = 255
        img[:, col_start:col_end] = 255
        self.display_image(img)

    @staticmethod
    def _make_text(
            text: str,
            img_fraction: float,
            path_to_font: str,
            img_size: Tuple[int, int],
    ) -> np.ndarray:
        image_pil = Image.fromarray(np.transpose(np.zeros(img_size, dtype=np.uint8)))
        img_height, img_width = img_size
        font_size = 2
        font = ImageFont.truetype(path_to_font, font_size)
        while font.getlength(text) < img_fraction * image_pil.size[0]:
            font_size += 1
            font = ImageFont.truetype(path_to_font, font_size)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(path_to_font, font_size)
        draw.text((int(img_width / 2), int(img_height / 2)), text, fill=255, font=font, anchor='mm', align='center')
        return np.transpose(np.array(image_pil))

    def display_text(
            self,
            text: Optional[str] = "Hello, World!",
            img_fraction: Optional[float] = 0.5,
            path_to_font: Optional[str] = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            img_size: Optional[Tuple[int, int]] = None,
    ):
        if not Path(path_to_font).exists():
            logger.error("DMDControl.display_test: Path to font does not exist.")
            return
        if img_size is None:
            img_size = DMD_WIDTH_HEIGHT
        img = self._make_text(text=text, img_fraction=img_fraction, path_to_font=path_to_font, img_size=img_size)
        self.display_image(img=img)

    def display_loaded_image(self):
        if self._loaded_img is None:
            logger.error(f"display_loaded_image: No image loaded. Use load_image to load an image.")
            return
        self.display_image(img=self._loaded_img)

    def load_image(self, filename: str, display_image: bool = True):
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
        self._loaded_img = img
        if display_image:
            self.display_image(img=img)

import cv2
import logging
import numpy as np
from pathlib import Path
import pickle as pkl
from PIL import Image, ImageFont, ImageDraw
import screeninfo
import subprocess
import socket
import time
from typing import Dict, List, Optional, Union, Tuple
from threading import Thread

from evomachine.config import get_logger, EVOMACHINE_DIR
from evomachine.exceptions import DMDError, ErrorCode, ErrorContainer

logger = get_logger(name=__name__)


DMD_WIDTH_HEIGHT = (2716, 1600)  # Provide images with img.shape == DMD_WIDTH_HEIGHT
PORT = 12345
HOST = '127.0.0.1'
MAX_BYTE_SIZE = 65482
NUM_CHUNKS = 97
CHUNK_ROWS = int(DMD_WIDTH_HEIGHT[0] / NUM_CHUNKS)
ARR_TYPE = np.uint8
EM_DMD_PROGRAM_PATH = EVOMACHINE_DIR / "C/evomachine_dmd_window"  # TODO test version that opens on the same screen


class DMDControl:
    DEFAULT_LINE_WIDTH: int = 5
    "Line width used for calibration and displaying lines. Use odd values."

    def __init__(self):
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
        self._process: Union[subprocess.Popen, None] = None
        "Process for C program."
        self._output_thread: Union[Thread, None] = None
        "Thread to display output from C program."
        self._calib_data: Optional[List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = None
        "List containing calibration data."
        self._calib_file: Path = EVOMACHINE_DIR / 'dmd_calibration_data.pkl'
        "Path to calibration file."
        self._homography_mat: Optional[np.ndarray] = None
        "Homography matrix for mapping image to DMD coordinates."

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
        logger.info(f"DMDControl._load_calibration_data: mapping point "
                    f"({int(points_cam[0][0][0])},{int(points_cam[0][0][1])}) to "
                    f"({int(points_dmd[0][0][0])},{int(points_dmd[0][0][1])}) and "
                    f"({int(points_cam[0][1][0])},{int(points_cam[0][1][1])}) to "
                    f"({int(points_dmd[1][0][0])},{int(points_dmd[1][0][1])}).")
        return True

    def _launch_dmd_window(self):
        def read_output(pipe):
            for line in iter(pipe.readline, b''):
                print(line.decode('utf-8').strip())
        self._process = subprocess.Popen([str(EM_DMD_PROGRAM_PATH)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(1)
        # self._output_thread = Thread(target=read_output, args=(self._process.stdout,), daemon=True)
        # self._output_thread.start()

    def _send_image(self, img: np.ndarray):
        """
        Sends an image over the socket to the C program. Note that we are allocating the image
        as width (rows) x height (columns), so the transpose is sent here.

        Parameters
        ----------
        img: np.ndarray     Image must be of ARR_TYPE and of size DMD_WIDTH_HEIGHT.
        """
        self.s.sendall(img.transpose().tobytes())

    def _connect_socket(self):
        """
        This function opens a socket. Note that after calling s.close(), e.g. after a restart, re-opening a socket throws
        an error. The error is therefore caught once.
        """
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

    def img_to_dmd_coords(self, img_row: int, img_col: int) -> Tuple[int, int]:
        point_cam = np.array([[[img_col, img_row]]])
        point_dmd = cv2.perspectiveTransform(point_cam, self._homography_mat)
        return int(np.round(point_dmd[0][0][1])), int(np.round(point_dmd[0][0][0]))

    def initialise(self, is_test: bool = False):
        try:
            self._launch_dmd_window()
        except Exception as e:
            msg = f"Error launch DMD window: {e}"
            logger.error(msg)
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_SOCKET))
            return
        monitors = screeninfo.get_monitors()
        mon_info = "\n".join(m.__str__() for m in monitors)
        has_two_monitors = len(monitors) == 2
        if (not is_test) and has_two_monitors:
            has_one_primary = any(m.is_primary for m in monitors) and any(not m.is_primary for m in monitors)
            if has_one_primary:
                mon_dmd = [m for m in monitors if (not m.is_primary)][0]
                is_correct_size = all(x1 == x2
                                      for (x1, x2) in zip(DMD_WIDTH_HEIGHT, (mon_dmd.width, mon_dmd.height)))
                if is_correct_size:
                    try:
                        self._connect_socket()
                        if self._connection_test():
                            self._is_initialised = True
                            # self.display_none()
                            logging.info(f"DMD: initialised with size={DMD_WIDTH_HEIGHT}.")
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

    def is_initialised(self) -> bool:
        return self._is_initialised

    def finalise(self):
        """
        Closes connection with the C program and the program itself.
        """
        if not self._is_initialised:
            logger.warning("DMDControl.finalise: DMD not initialised.")
            return
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

    def display_full(self):
        """
        Displays a white screen.

        """
        self.display_image(np.ones(DMD_WIDTH_HEIGHT, dtype=ARR_TYPE)*255)

    def display_image(self, img: np.ndarray[(int, int), ARR_TYPE]):
        """

        Parameters
        ----------
        img: np.ndarray     Image must be of ARR_TYPE.

        Returns
        -------

        """
        if not self._is_initialised:
            logger.error(f"DMDControl.display_image: DMD not initialised. Try running DMDControl.initialise.")
            return
        if img.shape == DMD_WIDTH_HEIGHT:
            self._send_image(img)
        elif img.shape == (*DMD_WIDTH_HEIGHT, 3):
            logger.warning(f"DMDControl.display_image: B/W image expected. Sending image[:,:,0] instead.")
            self._send_image(img[:, :, 0])
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

    def display_circle(
            self,
            row: int,
            col: int,
            radius: int = 1,
    ):
        img = self.get_zero_array()
        cv2.circle(img, (col, row), radius, color=255, thickness=-1)
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

    def warp_image_to_dmd(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self._homography_mat, DMD_WIDTH_HEIGHT)

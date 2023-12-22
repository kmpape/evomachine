from enum import Enum
import logging
import os
from PIL import Image, ImageFont, ImageDraw
import subprocess
import sys
from typing import List, Optional, Union, Tuple

import numpy as np
import pygame
import pygame.locals
import screeninfo

from evomachine.exceptions import DMDError, ErrorCode, ErrorContainer

formatter = logging.Formatter('--->\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n<---')
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


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

        """
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self._dmd_is_alive: bool = False
        "Flag set in initialise."
        self.offset_DMD: Tuple[int, int] = (0, 0)
        "Offset to display PyGame window. Set in initialise."
        self.width_height_DMD: Tuple[int, int] = (2716, 1600)
        "Size of DMD. Double-checked in initialise."
        self.surface: Union[None, pygame.surface] = None
        "PyGame object to display images. Initialised in initialise."
        self.default_line_width: int = 5
        "Line width used for calibration and displaying lines. Use odd values."

        self.initialise()

    def initialise(self):
        monitors = screeninfo.get_monitors()
        mon_info = "\n".join(m.__str__() for m in monitors)
        has_two_monitors = len(monitors) == 2
        if has_two_monitors:
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
            self.error_container.add_error(new_error=DMDError(message=msg, error_code=ErrorCode.ERROR_MONITORS))

    def finalise(self):
        self.display_none()
        DMDControl.close_window()

    @staticmethod
    def close_window():
        pygame.quit()

    def display_image(self, img: np.ndarray[(int, int), int], update_display: Optional[bool] = True):
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_image: DMD not initialised. Try running DMDControl.initialise.")
            return
        if img.shape == (*self.width_height_DMD, 3):
            self.surface.blit(pygame.surfarray.make_surface(img), (0, 0))
            if update_display:
                pygame.display.update()
        else:
            logger.error(f"DMDControl.display_image: provided image of shape={img.shape}, "
                         f"but DMD shape={(*self.width_height_DMD, 3)}.")

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

    def moving_rectangles(self):  # TODO
        clock = pygame.time.Clock()
        rect1 = pygame.locals.Rect(0, 0, 200, self.width_height_DMD[1])
        rect2 = pygame.locals.Rect(self.width_height_DMD[0] - 200, 0, 200, self.width_height_DMD[1])

        v = 10

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.locals.QUIT:
                        pygame.quit()
                        sys.exit()
                self.surface.fill((0, 0, 0))

                if rect1.left < 0 or rect1.left > self.width_height_DMD[0] - 200:
                    v *= -1

                rect1.move_ip(v, 0)
                rect2.move_ip(-v, 0)

                pygame.draw.rect(self.surface, (255, 255, 255), rect1)
                pygame.draw.rect(self.surface, (255, 255, 255), rect2)

                pygame.display.update()
                clock.tick(50)  # Framerate: 50 Hz
        except KeyboardInterrupt:
            pass


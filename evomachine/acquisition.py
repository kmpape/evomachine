from datetime import datetime
import logging
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Union, Tuple

import cv2
import matplotlib.pyplot as plt
from pycromanager import Core, Studio
import pygame
import pygame.locals
import screeninfo

import asitiger.tigercontroller
import delta

from evomachine.config import ConfigDevice, ConfigImage
from evomachine.exceptions import CameraError, ErrorCode, ErrorContainer, StageError, TigerError, DMDError

formatter = logging.Formatter('--->\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n<---')
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class AbstractCamera:
    def __init__(self, cfg_device: ConfigDevice):
        self.error_container: ErrorContainer = ErrorContainer()
        "Deque to store all errors."
        self.cfg_device: ConfigDevice = cfg_device
        "Device configuration object."
        self._step: int = -1
        "Increments each time an image is taken."
        self._curr_pos: int = 0
        "Current position equalling 0 or i_pos passed to move_to_pos."

        self.cfg_device.check_config()

    def initialise(self):
        self._step = -1
        self._initialise()

    def check_status(self):
        if len(self.error_container) > 0:
            msg = "\n".join([str(e) for e in self.error_container.error_list])
            logger.warning(msg=msg)
        else:
            logger.warning("No errors for acquisition found.")

    def move_to_pos(self, i_pos: int) -> None:
        if i_pos not in range(self.cfg_device.num_pos):
            raise StageError("Position index {} out of range".format(i_pos),
                             ErrorCode.ERROR_STAGE_COORDINATES)
        self._curr_pos = i_pos
        success = self._move_stage_to_pos(i_pos=i_pos)
        if not success:
            raise StageError("Fault moving to position={}.".format(i_pos), ErrorCode.ERROR_STAGE_MOVEMENT)

    def get_frame(
            self,
            i_chan: int,
            i_period: Union[int, None] = None,
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:  # TODO: check frame data type
        self._step += 1
        return self._take_frame(i_chan=i_chan, i_period=i_period)

    def display_frame(
            self,
            i_chan: int,
            i_period: Union[int, None] = None,
            path_to_save: Union[str, None] = None,
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        frame = self.get_frame(i_chan=i_chan, i_period=i_period)
        cmap = plt.cm.gray
        norm = plt.Normalize(vmin=frame.min(), vmax=frame.max())
        image = cmap(norm(frame))
        plt.imshow(image)
        plt.show()
        if path_to_save:
            filename = "evom_pos{:02d}_chan{}_{}.png".format(
                self._curr_pos,
                i_chan,
                datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
            )
            plt.imsave(path_to_save+filename, image)

        return frame

    def get_pos(self) -> int:
        return self._curr_pos

    def autofocus(self):
        raise NotImplementedError()

    def _initialise(self) -> None:
        raise NotImplementedError()

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        raise NotImplementedError()

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        raise NotImplementedError()


class DeltaCamera(AbstractCamera):
    """
    A class to mock the acquisition of frames.
    """
    def __init__(self, cfg_device: ConfigDevice):
        super().__init__(cfg_device=cfg_device)

        self.all_frames: List[np.ndarray[(int, int, int, int), np.float32]] = [
            np.empty((1, 1, 1, 1)) for _ in range(cfg_device.num_periods)
        ]
        "List of all frames by position."
        self._curr_period: int = -1
        "Incremented after completing one round of imaging the whole device."

        delta_reader: delta.utils.XPReader = \
            delta.utils.XPReader(self.cfg_device.path_to_images / "Position{p}Channel{c}Frames{t}.tif")
        for i_pos, i_delta_pos in enumerate(delta_reader.positions, start=0):
            self.all_frames[i_pos] = delta_reader.getframes(position=i_delta_pos)

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        return True

    def _initialise(self) -> None:
        self._curr_period = -1

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> np.ndarray[(int, int), 'ConfigImage.pxl_dtype']:
        return self.all_frames[self._curr_pos][i_period, i_chan, :, :]


class EvoCamera(AbstractCamera):
    """
    EvoMachine acquisition class.
    """
    def __init__(self, cfg_device: ConfigDevice):
        super().__init__(cfg_device=cfg_device)

        self.tiger: Union[asitiger.tigercontroller.TigerController, None] = None
        "Object for serial communication with ASI tiger."
        self._tiger_is_alive: bool = False
        "Flag set in _initialise."
        self.current_channel: int = -1
        "Current LED channel set"
        self.channel_settings: Dict[int, Dict] = {
            0: {"X": 100, "Y": 0, "Z": 0, "F": 0},
            1: {"X": 0, "Y": 100, "Z": 0, "F": 0},
            2: {"X": 0, "Y": 0, "Z": 100, "F": 0},
            3: {"X": 0, "Y": 0, "Z": 0, "F": 100},
            -1: {"X": 0, "Y": 0, "Z": 0, "F": 0},
        }
        "LED intensity for i_chan=0,...,3."
        self.card_address_led: int = 7
        "LED card address on ASI tiger."
        self.card_address_fw: int = 8
        "Filter wheel card address on ASI tiger."
        self.filter_wheel_settings: Dict[int, str] = {0: "TBD", 1: "TBD", 2: "TBD"}
        "Available filter wheels."

        self.mmc: Union[Core, None] = None
        "Micromanager Core object for taking images."
        self.studio: Union[Studio, None] = None
        "Micromanager Studio object for additional functions."
        self._mmc_is_alive: bool = False
        "Flag set in _initialise."

        self._initialise()  # Must be called before using EvoCamera

    def _initialise(self) -> None:
        try:
            self.tiger: asitiger.tigercontroller.TigerController = \
                asitiger.tigercontroller.TigerController.from_serial_port(port=self.cfg_device.tiger_port)
        except Exception as e:
            self._tiger_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to Tiger: {e}.")
            self.error_container.add_error(
                new_error=TigerError(message=str(e), error_code=ErrorCode.ERROR_TIGER_SERIAL_CONNECTION)
            )
        if not self._get_tiger_is_alive():
            self._tiger_is_alive = False
            logger.warning("EvoCamera._initialise: Tiger is not alive.")
            self.error_container.add_error(
                new_error=TigerError(message="Tiger is not alive.", error_code=ErrorCode.ERROR_TIGER_NOT_ALIVE)
            )
        else:
            self._tiger_is_alive = True
        try:
            self.mmc = Core()
            self.studio = Studio()
            self._mmc_is_alive = True
        except Exception as e:
            self._mmc_is_alive = False
            logger.warning(f"EvoCamera._initialise: Error connecting to MMC: {e}.")
            self.error_container.add_error(
                new_error=CameraError(message=str(e), error_code=ErrorCode.ERROR_MMC_NOT_ALIVE)
            )
        self._disable_channels()

    def _get_tiger_is_alive(self) -> bool:
        if not self.tiger:
            return False
        try:
            _ = self.tiger.status()
            return True
        except ValueError:
            return False

    def _set_channel(self, i_chan: int):
        if i_chan not in self.channel_settings.keys():
            logger.error(msg=f"EvoCamera._set_channel: i_chan={i_chan} not in channels={self.channel_settings.keys()}.")
            return
        if self._tiger_is_alive:
            self.tiger.led(led_brightnesses=self.channel_settings[i_chan], card_address=self.card_address_led)
            self.current_channel = i_chan
        else:
            logger.error(msg=f"EvoCamera._set_channel: Tiger is not alive. Check ASI Tiger box and serial connection.")

    def _set_filter_wheel(self, i_pos: int):
        if i_pos not in self.filter_wheel_settings.keys():
            logger.error(msg=f"EvoCamera._set_filter_wheel: i_pos={i_pos} not in wheels={self.filter_wheel_settings}.")
            return
        if self._tiger_is_alive:
            self.tiger.filter_wheel(position=i_pos, card_address=self.card_address_fw)
        else:
            logger.error(msg=f"EvoCamera._set_filter_wheel: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")

    def _disable_channels(self):
        self._set_channel(i_chan=-1)

    def _move_stage_to_pos(
            self,
            i_pos: int,
    ) -> bool:
        pos = {
            'X': self.cfg_device.coord_pos[i_pos][0],
            'Y': self.cfg_device.coord_pos[i_pos][1],
            'Z': self.cfg_device.coord_pos[i_pos][2],
        }
        return self._move_stage_to_coord(pos)

    def _move_stage_to_coord(
            self,
            coordinates: Dict[str, int],
    ) -> bool:
        answer = None
        if self._tiger_is_alive:
            answer = self.tiger.move(coordinates=coordinates)
        else:
            logger.error(msg=f"EvoCamera._move_stage_to_coord: Tiger is not alive. "
                             f"Check ASI Tiger box and serial connection.")
        return True if isinstance(answer, str) else False

    def _take_frame(
            self,
            i_chan: int,
            i_period: Union[int, None],
    ) -> Union[None, np.ndarray[(int, int), 'ConfigImage.pxl_dtype']]:
        if not self._mmc_is_alive:
            logger.error(msg=f"EvoCamera._take_frame: MMC is not alive. Check Camera and Micro-Manager.")
            return None
        self._set_channel(i_chan=i_chan)
        self.mmc.snap_image()
        self._disable_channels()
        tagged_image = self.mmc.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        return pixels

    def autofocus(self, focus_channel: int = 0, focus_exposure: int = 100):
        if not all([self._mmc_is_alive, self._tiger_is_alive]):
            logger.error(f"EvoCamera.autofocus: Device(s) not alive. "
                         f"Tiger={self._tiger_is_alive}, MMC={self._mmc_is_alive}.")
            return

        # Settings
        FOCUS_MIN = -76000
        FOCUS_MAX = -73000
        FOCUS_STEP = 50
        FOCUS_SMALL_STEP = 1
        FOCUS_SMALL_DIFF = 50

        # Prepare
        old_channel = self.current_channel
        self.mmc.set_exposure(focus_exposure)
        self.studio.live().set_live_mode(False)

        # Focus
        coordinates = range(FOCUS_MIN, FOCUS_MAX, FOCUS_STEP)
        best_coordinate: int = 0
        for raw_fine in range(2):
            best_focus_score = 0
            best_focus_position = 0
            for ipos, zcoord in enumerate(coordinates):
                self._move_stage_to_coord({'Z': zcoord})
                image_raw = self._take_frame(i_chan=focus_channel, i_period=None)
                laplacian = cv2.Laplacian(image_raw, cv2.CV_64F)
                focus_score = laplacian.var()
                # print(f"{ipos}/{len(coordinates)}: Z={zcoord}, Score={focus_score}\n")
                if focus_score > best_focus_score:
                    best_focus_position = ipos
                    best_focus_score = focus_score
            best_coordinate = coordinates[best_focus_position]
            coordinates = range(best_coordinate - FOCUS_SMALL_DIFF, best_coordinate + FOCUS_SMALL_DIFF, FOCUS_SMALL_STEP)
        self._move_stage_to_coord({'Z': best_coordinate})
        self._set_channel(i_chan=old_channel)


class DMDControl:
    def __init__(self):
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
                    self.surface = pygame.display.set_mode(size=self.width_height_DMD, flags=pygame.NOFRAME)
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

    def display_image(self, img: np.ndarray[(int, int), int]):
        if not self._dmd_is_alive:
            logger.error(f"DMDControl.display_image: DMD not initialised. Try running DMDControl.initialise.")
            return
        if img.shape == (*self.width_height_DMD, 3):
            self.surface.blit(pygame.surfarray.make_surface(img), (0, 0))
            pygame.display.update()
        else:
            logger.error(f"DMDControl.display_image: provided image of shape={img.shape}, "
                         f"but DMD shape={(*self.width_height_DMD, 3)}.")

    def display_full(self):
        self.display_image(img=np.ones((*self.width_height_DMD, 3), dtype=int) * 255)

    def display_none(self):
        self.display_image(img=np.zeros((*self.width_height_DMD, 3), dtype=int))

    def display_crosshair(self, line_width: int = 1):
        img = np.zeros((*self.width_height_DMD, 3), dtype=int)
        center = (np.floor_divide(self.width_height_DMD[0], 2), np.floor_divide(self.width_height_DMD[1], 2))
        img[center[0] - int(np.floor(line_width / 2)): center[0] + int(np.ceil(line_width / 2)), :, :] = 255
        img[:, center[1] - int(np.floor(line_width / 2)): center[1] + int(np.ceil(line_width / 2)), :] = 255
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

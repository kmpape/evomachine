from __future__ import annotations

import os
from typing import Any

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.peripherals.peripherals import PeripheralController


class FakeSurface:
    """Pygame surface-like object that records blit calls."""

    def __init__(self):
        """Initialise fake blit recording."""
        self.blits = []

    def blit(self, surface, position) -> None:
        """Record a fake blit call."""
        self.blits.append((surface, position))


class FakePygame:
    """Small pygame-like module used by DMD tests."""

    NOFRAME = 1
    FULLSCREEN = 2

    def __init__(self):
        """Initialise fake pygame display state."""
        self.updated = False
        self.quit_called = False
        self.modes = []
        self.surfarray = type("FakeSurfArray", (), {"make_surface": lambda _, img: ("surface", img.copy())})()
        self.display = type(
            "FakeDisplay",
            (),
            {"update": lambda _: self.update(), "set_mode": lambda _, size, flags=0: self.set_mode(size, flags)},
        )()

    def init(self) -> None:
        """Accept fake pygame initialisation."""
        return

    def set_mode(self, size, flags=0) -> FakeSurface:
        """Return a fake surface for the requested display mode."""
        self.modes.append((size, flags))
        return FakeSurface()

    def update(self) -> None:
        """Record a fake display update."""
        self.updated = True

    def quit(self) -> None:
        """Record a fake pygame shutdown."""
        self.quit_called = True


class PygameDmdPeripheralController(PeripheralController):
    """Peripheral controller for a pygame DMD display window."""

    DEFAULT_NAME: str = "pygame DMD Peripheral Controller"

    def __init__(
            self,
            name: str = DEFAULT_NAME,
            debug_mode: bool = False,
            size: tuple[int, int] = DMD_WIDTH_HEIGHT,
            display_offset: tuple[int, int] = (0, 0),
            monitor_index: int | None = None,
            surface: Any | None = None,
            pygame_module: Any | None = None,
    ):
        self.debug_mode: bool = debug_mode
        self.size: tuple[int, int] = size
        self.display_offset: tuple[int, int] = display_offset
        self.monitor_index: int | None = monitor_index
        self.surface: Any | None = surface
        self._pygame: Any | None = pygame_module
        super().__init__(name=name)

    @classmethod
    def from_default(
            cls,
            name: str = DEFAULT_NAME,
            **dmd_options: Any,
    ) -> PygameDmdPeripheralController:
        """Create a pygame-backed DMD peripheral controller."""
        return cls(name=name, **dmd_options)

    def configure_display(
            self,
            size: tuple[int, int],
            display_offset: tuple[int, int] = (0, 0),
            monitor_index: int | None = None,
    ) -> None:
        """
        Update pygame display placement configuration.

        Parameters
        ----------
        size
            Window/fullscreen size in pixels.
        display_offset
            SDL window position offset as an (x, y) tuple.
        monitor_index
            Optional SDL fullscreen display index.

        Returns
        -------
        None
        """
        self.size = size
        self.display_offset = display_offset
        self.monitor_index = monitor_index

    def get_pygame(self) -> Any:
        """Return the pygame module, importing it lazily."""
        if self._pygame is None:
            import pygame

            self._pygame = pygame
        return self._pygame

    def display_array(self, img, update_display: bool = True) -> None:
        """Blit a 2D or 3D DMD array to the pygame surface."""
        if self.debug_mode:
            return
        pygame = self.get_pygame()
        if img.ndim == 2:
            img = img[:, :, None].repeat(3, axis=2)
        self.surface.blit(pygame.surfarray.make_surface(img), (0, 0))
        if update_display:
            pygame.display.update()

    def _initialise(self, force: bool = False) -> bool:
        if self.debug_mode:
            return True
        pygame = self.get_pygame()
        pygame.init()
        if self.surface is None:
            flags = getattr(pygame, "NOFRAME", 0) | getattr(pygame, "FULLSCREEN", 0)
            try:
                os.environ["SDL_VIDEO_WINDOW_POS"] = f"{self.display_offset[0]},{self.display_offset[1]}"
                if self.monitor_index is not None:
                    os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(self.monitor_index)
                self.surface = pygame.display.set_mode(size=self.size, flags=flags)
            except Exception:
                self.surface = pygame.display.set_mode(size=(300, 300), flags=getattr(pygame, "NOFRAME", 0))
        return self.surface is not None

    def _check_is_alive(self) -> bool:
        return self.debug_mode or self.surface is not None

    def _stop(self) -> None:
        return

    def _shutdown(self, force: bool = False) -> None:
        if not self.debug_mode:
            self.get_pygame().quit()
        self.surface = None

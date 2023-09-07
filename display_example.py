import pygame
import sys
import os
from pygame.locals import *

SAMSUNG_SCREEN_WIDTH = 2560
SAMSUNG_SCREEN_HEIGHT = 1440
DMD_SCREEN_WIDTH  = 2716
DMD_SCREEN_HEIGHT = 1600


def main():
    pygame.init()
    clock = pygame.time.Clock()

    # Offset the window so that it is on the DMD
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{SAMSUNG_SCREEN_WIDTH},{SAMSUNG_SCREEN_HEIGHT}"

    # Setup display to fill the DMD
    surface = pygame.display.set_mode((DMD_SCREEN_WIDTH,DMD_SCREEN_HEIGHT), flags=pygame.NOFRAME)
    
    rect1 = Rect(0, 0, 200, DMD_SCREEN_HEIGHT)
    rect2 = Rect(DMD_SCREEN_WIDTH-200, 0, 200, DMD_SCREEN_HEIGHT)
    
    v = 10
    
    while True:
        for event in pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()
        surface.fill((0, 0, 0))
        
        
        if rect1.left < 0 or rect1.left > DMD_SCREEN_WIDTH-200:
            v *= -1
        
        rect1.move_ip(v, 0)
        rect2.move_ip(-v, 0)
        
        pygame.draw.rect(surface, (255, 255, 255), rect1)
        pygame.draw.rect(surface, (255, 255, 255), rect2)
        
        pygame.display.update()
        clock.tick(50) # Framerate: 50 Hz
        
if __name__ == '__main__':
    main()
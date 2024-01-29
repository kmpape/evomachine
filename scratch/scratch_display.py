import sys
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
from evomachine.acquisition import EvoCamera, DMDControl
from evomachine.config import DEVICE_CONFIG_EVO_TEST
import pygame
import sys
import os
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

cam = EvoCamera(DEVICE_CONFIG_EVO_TEST)
dmd = DMDControl()
tig = cam.tiger
cam.initialise()
cam._set_channel(-1)


cmap = plt.cm.jet
norm = plt.Normalize(vmin=frame.min(), vmax=frame.max())
image = cmap(norm(frame))
plt.imshow(image)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

text = "Hello, World!"
path_to_font = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
img_width, img_height = 400, 200
img = np.zeros((img_height, img_width), dtype=np.uint8)
img_fraction = 0.50
font_size = 2
font = ImageFont.truetype(path_to_font, font_size)
while font.getlength(text) < img_fraction*image.size[0]:
    font_size += 1
    font = ImageFont.truetype(path_to_font, font_size)

bbox = font.getbbox(text)
image_pil = Image.fromarray(img)
draw = ImageDraw.Draw(image_pil)
font = ImageFont.truetype(path_to_font, 35)
draw.text((int(img_width/2), int(img_height/2)), text, fill=255, font=font, anchor='mm')
img = np.array(image_pil)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

img_width, img_height = 400, 200
img = np.zeros((img_height, img_width), dtype=np.uint8)
image = Image.fromarray(img)
draw = ImageDraw.Draw(image)
txt = "Hello World"
fontsize = 1
img_fraction = 0.50
font = ImageFont.truetype(path_to_font, fontsize)
while font.getsize(txt)[0] < img_fraction*image.size[0]:
    fontsize += 1
    font = ImageFont.truetype(path_to_font, fontsize)

fontsize -= 1
font = ImageFont.truetype(path_to_font, fontsize)
print('final font size', fontsize)
draw.text((10, 25), txt, font=font, fill=255)
img = np.array(image_pil)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


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
    draw.text((int(img_height / 2), int(img_width / 2)), text, fill=255, font=font, anchor='mm')
    img = np.rot90(np.array(image_pil), k=1)
    # img = np.array(image_pil)
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    if False == True:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    self.display_image(img=img)

def set_chan(i):
    cam._set_channel(i)


def set_fw(i):
    cam._set_filter_wheel(i)

PATH_TO_SAVE = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/images/2023-09-22/"

SAMSUNG_SCREEN_WIDTH = 2560
SAMSUNG_SCREEN_HEIGHT = 1440
DMD_SCREEN_WIDTH = 2716
DMD_SCREEN_HEIGHT = 1600

def main():
    pygame.init()
    clock = pygame.time.Clock()

    # Offset the window so that it is on the DMD
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{SAMSUNG_SCREEN_WIDTH},{SAMSUNG_SCREEN_HEIGHT}"

    # Setup display to fill the DMD
    surface = pygame.display.set_mode((DMD_SCREEN_WIDTH, DMD_SCREEN_HEIGHT), flags=pygame.NOFRAME)
    
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
        clock.tick(50)  # Framerate: 50 Hz
        
if __name__ == '__main__':
    main()
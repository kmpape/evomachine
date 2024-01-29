import pygame

# Initialize Pygame
pygame.init()

# Define the width and height of the surface
width = 640
height = 480

# Create a 1-bit black and white surface
depth = 1
bw_surface = pygame.Surface((width, height), depth=8)

# Set one color to 0 (black) and the other to 1 (white)
black = (0, 0, 0, 255)
white = (255, 255, 255, 255)

# Fill the surface with a pattern (e.g., checkerboard)
for y in range(0, height, 20):
    for x in range(0, width, 20):
        pygame.draw.rect(bw_surface, black if (x // 20 + y // 20) % 2 == 0 else white, (x, y, 20, 20))

# Create a Pygame window and display the surface
window = pygame.display.set_mode((width, height))
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.blit(bw_surface, (0, 0))
    pygame.display.flip()

pygame.quit()

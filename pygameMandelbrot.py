import pygame
import numpy as np
from numba import jit, prange

# Parameters
width, height = 800, 600
max_iter = 256
zoom_factor = 0.9

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Viewer")

# Font setup for FPS display
font = pygame.font.SysFont("Arial", 18)


@jit(nopython=True, parallel=True)
def mandelbrot(Re, Im, max_iter):
    rows, cols = Re.shape
    c = Re + 1j * Im
    z = np.zeros_like(c, dtype=np.complex128)
    divtime = np.full(c.shape, max_iter, dtype=np.int32)

    for i in range(max_iter):
        z = z * z + c
        mask = np.abs(z) > 2
        for row in range(rows):
            for col in range(cols):
                if mask[row, col] and divtime[row, col] == max_iter:
                    divtime[row, col] = i
                    z[row, col] = 2

    return divtime


def create_mandelbrot_image(width, height, x_centre, y_centre, zoom, max_iter):
    aspect_ratio = width / height
    x_min = x_centre - 1.5 / zoom * aspect_ratio
    x_max = x_centre + 1.5 / zoom * aspect_ratio
    y_min = y_centre - 1.5 / zoom
    y_max = y_centre + 1.5 / zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    Re, Im = np.meshgrid(x, y)

    divtime = mandelbrot(Re, Im, max_iter)

    # Normalize the values to the range [0, 255]
    divtime = 255 - (divtime * 255 / max_iter).astype(np.uint8)

    return divtime


def draw_mandelbrot(screen, divtime):
    surface = pygame.surfarray.make_surface(np.transpose(divtime))
    screen.blit(surface, (0, 0))


def draw_fps(screen, clock):
    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, pygame.Color('white'))
    screen.blit(fps_text, (10, 10))


def main():
    x_centre, y_centre = -0.743643887037151, 0.131825904205330
    zoom = 1.0

    clock = pygame.time.Clock()
    running = True
    while running:
        divtime = create_mandelbrot_image(width, height, x_centre, y_centre, zoom, max_iter)
        draw_mandelbrot(screen, divtime)
        draw_fps(screen, clock)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    y_centre -= 0.1 / zoom
                elif event.key == pygame.K_DOWN:
                    y_centre += 0.1 / zoom
                elif event.key == pygame.K_LEFT:
                    x_centre -= 0.1 / zoom
                elif event.key == pygame.K_RIGHT:
                    x_centre += 0.1 / zoom
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    zoom *= zoom_factor
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    zoom /= zoom_factor

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

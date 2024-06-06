import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import jit, prange


def mandelbrot(c, max_iter):
    z = np.zeros(c.shape, dtype=np.complex128)
    divtime = max_iter + np.zeros(c.shape, dtype=np.int32)

    for i in range(max_iter):
        z = z * z + c
        diverge = np.abs(z) > 2
        divtime[diverge & (divtime == max_iter)] = i
        z[diverge] = 2

    return divtime

def create_mandelbrot_image(width, height, x_centre, y_centre, zoom, max_iter, output_filename):
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for x in tqdm(range(width), desc="Generating Mandelbrot", unit="pixel"):
        for y in range(height):
            re = x_min + (x / width) * (x_max - x_min)
            im = y_min + (y / height) * (y_max - y_min)
            c = complex(re, im)
            colour_value = mandelbrot(c, max_iter)
            colour = 255 - int(colour_value * 255 / max_iter)
            pixels[x, y] = (colour, colour, colour)

    image.save(output_filename)

width = 8000
height = 8000
x_min = -2.0
x_max = 1.0
y_min = -1.5
y_max = 1.5
max_iter = 256
output_filename = "high_res_mandelbrot.png"

create_mandelbrot_image(width, height, x_min, x_max, y_min, y_max, max_iter, output_filename)
print(f"Image saved as {output_filename}")
import numpy as np
from numba import jit
from PIL import Image
from tqdm import tqdm
import os

@jit(nopython=True, parallel=True)
def mandelbrot(Re, Im, max_iter):
    rows, cols = Re.shape
    c = Re + 1j * Im
    z = np.zeros_like(c, dtype=np.complex128)
    divtime = np.full(c.shape, max_iter, dtype=np.int32)

    for i in range(max_iter):
        z = z * z + c
        for row in range(rows):
            for col in range(cols):
                if divtime[row, col] == max_iter and np.abs(z[row, col]) > 2:
                    divtime[row, col] = i
                    z[row, col] = 2

    return divtime

def create_mandelbrot_frame(width, height, x_centre, y_centre, zoom, max_iter, filename):
    aspect_ratio = width/height
    x_min = x_centre - 1.5 / zoom * aspect_ratio
    x_max = x_centre + 1.5 / zoom * aspect_ratio
    y_min = y_centre - 1.5 / zoom
    y_max = y_centre + 1.5 / zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    Re, Im = np.meshgrid(x, y)

    divtime = mandelbrot(Re, Im, max_iter)

    divtime = 255 - (divtime * 255 / max_iter).astype(np.uint8)

    image = Image.fromarray(divtime)
    image.save(filename)

# Parameters for the Mandelbrot frames
width = 1920
height = 1080
x_centre = -0.743643887037151 # Centre x-axis value (real part of the complex plane)
y_centre = 0.131825904205330  # Centre y-axis value (imaginary part of the complex plane)
max_iter = 1000
num_frames = 60
zoom_factor = 1.1

output_folder = "mandelbrot_frames"
os.makedirs(output_folder, exist_ok=True)

zoom = 1.0
for i in tqdm(range(num_frames), desc="Generating frames"):
    filename = os.path.join(output_folder, f"frame_{i:04d}.png")
    create_mandelbrot_frame(width, height, x_centre, y_centre, zoom, max_iter, filename)
    zoom *= zoom_factor

print(f"Frames saved in {output_folder}")

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

def create_mandelbrot_image(width, height, x_min, x_max, y_min, y_max, max_iter, output_filename):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    divtime = np.zeros(C.shape, dtype=np.int32)
    chunk_size = 100

    for i in tqdm(range(0, width, chunk_size), desc="Generating Mandelbrot"):
        end = min(i + chunk_size, width)
        divtime_chunk = mandelbrot(C[i:end], max_iter)
        divtime[i:end] = divtime_chunk

    divtime = 255 - (divtime * 255 / max_iter).astype(np.uint8)

    image = Image.fromarray(divtime)
    image.save(output_filename)

width = 16000
height = 16000
x_min = -2.0
x_max = 1.0
y_min = -1.5
y_max = 1.5
max_iter = 1000
output_filename = "high_res_mandelbrot.png"

create_mandelbrot_image(width, height, x_min, x_max, y_min, y_max, max_iter, output_filename)
print(f"Image saved as {output_filename}")
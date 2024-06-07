import cv2
from tqdm import tqdm
import os

def compile_video_from_frames(input_folder, output_filename, fps):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    frame_files = [os.path.join(input_folder, f) for f in frame_files]
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Compiling video"):
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()

input_folder = "mandelbrot_frames"
output_filename = "mandelbrot_zoom.mp4"
fps = 60

compile_video_from_frames(input_folder, output_filename, fps)
print(f"Video saved as {output_filename}")


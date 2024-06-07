import cv2
import os

def compile_video_from_frames(input_folder, output_filename, fps):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

input_folder = "mandelbrot_frames"
output_filename = "mandelbrot_zoom.mp4"
fps = 2

compile_video_from_frames(input_folder, output_filename, fps)
print(f"Video saved as {output_filename}")
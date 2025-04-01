from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np
import os
import torch
import cv2
import multiprocessing as mp
import shutil

MODEL_PATH = os.path.join(os.getcwd(), 'Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth')
VIDEO_PATH = 'assets/bocchi.mp4'
OUTPUT_PATH = 'output/upscaled_bocchi.mp4'
FRAME_DIR = 'output_frames'
UPSCALE_FACTOR = 2
NUM_WORKERS = torch.cuda.device_count()  # Use all available GPUs

# Load the ESRGAN Model
def load_model(device):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=128,
        tile_pad=10,
        pre_pad=2,
        half=True,
        device=device
    )

# Enhance a single frame
def enhance_frame(frame_path):
    device = f'cuda:{mp.current_process()._identity[0] % NUM_WORKERS}' if torch.cuda.is_available() else 'cpu'
    restorer = load_model(device)
    
    img = Image.open(frame_path)
    img = np.array(img)
    
    result = restorer.enhance(img, outscale=UPSCALE_FACTOR)
    result_img = Image.fromarray(result[0])
    result_img.save(frame_path.replace('output_frames', 'enhanced_frames'))

# Extract frames from video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1

    cap.release()
    return frame_id  # Total frames extracted

# Reassemble enhanced frames into video
def create_video(input_folder, output_video, fps=30):
    frame_list = sorted(os.listdir(input_folder))
    first_frame = cv2.imread(os.path.join(input_folder, frame_list[0]))
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for frame_file in frame_list:
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        out.write(frame)

    out.release()

if __name__ == '__main__':
    shutil.rmtree('output_frames', ignore_errors=True)
    shutil.rmtree('enhanced_frames', ignore_errors=True)

    os.makedirs('output_frames', exist_ok=True)
    os.makedirs('enhanced_frames', exist_ok=True)

    # Step 1: Extract frames
    num_frames = extract_frames(VIDEO_PATH, FRAME_DIR)

    # Step 2: Parallelize enhancement
    frame_paths = [os.path.join(FRAME_DIR, f'frame_{i:05d}.jpg') for i in range(num_frames)]
    
    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(enhance_frame, frame_paths)

    # Step 3: Reassemble frames into a video
    create_video('enhanced_frames', OUTPUT_PATH, fps=30)

    print(f"Video processing completed: {OUTPUT_PATH}")
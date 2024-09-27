import cv2
import numpy as np

def preprocess_video(video, target_size=(28, 28)):
    processed_frames = []
    for frame in video:
        resized = cv2.resize(frame, target_size)
        normalized = resized / 255.0
        processed_frames.append(normalized)
    return np.array(processed_frames).transpose(0, 3, 1, 2)  # (T, C, H, W)

def augment_video(video):
    # Simple augmentation: horizontal flip
    return np.flip(video, axis=2)
import cv2
import numpy as np

def load_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Pad if less than num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros_like(frames[0]))
    
    return np.array(frames)
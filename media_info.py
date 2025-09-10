import json, subprocess
from pathlib import Path

def media_info(video_path: str) -> dict:
    # Use OpenCV for broad compatibility (no ffprobe dependency)
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps else 0.0
    cap.release()
    return {"fps": fps, "width": w, "height": h, "frames": frames, "duration": duration}

from typing import List, Dict
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path: str, threshold: float = 30.0) -> list[dict]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    out = []
    for i, (start_tc, end_tc) in enumerate(scenes, 1):
        s = start_tc.get_seconds(); e = end_tc.get_seconds()
        out.append({"id": i, "start": s, "end": e, "duration": round(e - s, 3)})
    return out

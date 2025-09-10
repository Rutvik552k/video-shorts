# main_legacy.py — works with older LangGraph (no writes/reads)
import os, json, sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from nodes.media_info import media_info
from nodes.transcribe import transcribe
from nodes.scene_detect import detect_scenes
from nodes.select_highlights import select_highlights
from nodes.cut_clips import cv2_cut_with_audio

load_dotenv()

# simple dict state
def n_media_info(state: dict) -> dict:
    state = dict(state)
    state["media"] = media_info(state["video_path"])
    return state

def n_transcribe(state: dict) -> dict:
    state = dict(state)
    out_dir = Path(state["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    state["transcript"] = transcribe(
        state["video_path"],
        out_dir,
        os.getenv("ASR_MODEL_SIZE", "medium"),
        os.getenv("ASR_LANG") or None
    )
    return state

def n_scene_detect(state: dict) -> dict:
    state = dict(state)
    state["scenes"] = detect_scenes(state["video_path"], threshold=30.0)
    return state

def n_select_highlights(state: dict) -> dict:
    state = dict(state)
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hl = select_highlights(state["transcript"], state.get("scenes", []), model)
    Path(state["out_dir"], "highlights.json").write_text(
        json.dumps(hl, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    state["highlights"] = hl
    return state

def n_cut(state: dict) -> dict:
    cv2_cut_with_audio(
        state["video_path"],
        state["highlights"],
        state["transcript"],
        Path(state["out_dir"])/"shorts",
        aspect=os.getenv("OUTPUT_ASPECT", "9:16")
    )
    return state  # final node; side effects only

graph = StateGraph(dict)
graph.add_node("media_info", n_media_info)
graph.add_node("transcribe", n_transcribe)
graph.add_node("scene_detect", n_scene_detect)
graph.add_node("select_highlights", n_select_highlights)
graph.add_node("cut", n_cut)

graph.set_entry_point("media_info")
graph.add_edge("media_info", "transcribe")
graph.add_edge("transcribe", "scene_detect")
graph.add_edge("scene_detect", "select_highlights")
graph.add_edge("select_highlights", "cut")
graph.add_edge("cut", END)

app = graph.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_legacy.py <path/to/source.mp4> [out_dir]")
        sys.exit(2)
    video_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    init = {"video_path": video_path, "out_dir": out_dir}
    final_state = app.invoke(init)
    print("✅ Done. Check:", Path(out_dir, "shorts").resolve())

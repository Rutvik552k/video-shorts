import os, json
from pathlib import Path
from faster_whisper import WhisperModel

def transcribe(video_path: str, out_dir: Path, model_size: str, lang: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "transcript.json"
    out_srt  = out_dir / "transcript.srt"

    model = WhisperModel(
        model_size or "medium",
        device="auto",
        compute_type="auto"
    )

    segments, info = model.transcribe(
        video_path,
        vad_filter=True,
        word_timestamps=False,
        language=(lang or None)
    )

    data = {"language": info.language, "duration": info.duration, "segments": []}
    for s in segments:
        data["segments"].append({
            "id": s.id, "start": float(s.start), "end": float(s.end), "text": s.text.strip()
        })

    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_srt(data["segments"], out_srt)
    return data

def write_srt(segments, srt_path):
    def fmt(ts: float) -> str:
        h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60)
        ms = int(round((ts - int(ts)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{(seg.get('text') or '').strip()}\n\n")

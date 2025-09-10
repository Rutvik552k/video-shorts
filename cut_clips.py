import json, textwrap, subprocess
from pathlib import Path
import cv2, imageio_ffmpeg, os

def cv2_cut_with_audio(video_path: str, highlights: dict, transcript: dict,
                       out_dir: Path, aspect: str = "9:16", crf: str = "20"):
    out_dir.mkdir(exist_ok=True)
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

    def time_to_frame(t, fps): return max(0, int(round(t * fps)))

    def center_crop_to_9x16(frame):
        h, w = frame.shape[:2]
        tgt_w = int(round(h * 9 / 16))
        if tgt_w > w:
            tgt_h = int(round(w * 16 / 9))
            y0 = (h - tgt_h) // 2
            return frame[y0:y0+tgt_h, :, :]
        x0 = (w - tgt_w) // 2
        return frame[:, x0:x0+tgt_w, :]

    def resize_for_aspect(frame, a):
        return cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA) if a == "9:16" else frame

    def pick_text(segs, t):
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return (s.get("text") or "").strip()
        return ""

    def draw_caption(frame, text, max_chars=36):
        if not text: return frame
        import textwrap as tw
        lines = tw.wrap(text, width=max_chars)[:3]
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8 if h >= 1080 else 0.6
        thick = 2
        line_h = int(28 * scale * 1.6)
        margin = int(24 * scale)
        total_h = line_h * len(lines) + margin * 2
        y0 = h - total_h - 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0 - margin//2), (w, y0 + total_h), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        for i, line in enumerate(lines):
            line_scale = scale
            while cv2.getTextSize(line, font, line_scale, thick)[0][0] > w - 2*margin and line_scale > 0.4:
                line_scale -= 0.05
            size = cv2.getTextSize(line, font, line_scale, thick)[0]
            x = (w - size[0]) // 2
            y = y0 + margin + i * line_h
            cv2.putText(frame, line, (x+2, y+2), font, line_scale, (0,0,0), thick+2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y),     font, line_scale, (255,255,255), thick, cv2.LINE_AA)
        return frame

    src = Path(video_path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened(): raise RuntimeError("OpenCV failed to open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    segs = transcript["segments"]

    for i, item in enumerate(highlights.get("shorts", []), 1):
        start, end = float(item["start"]), float(item["end"])
        if end <= start: continue
        dur = end - start
        start_f, end_f = time_to_frame(start, fps), time_to_frame(end, fps)
        total_frames = max(0, end_f - start_f)
        if total_frames == 0: continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ret, frame = cap.read()
        if not ret: continue

        if aspect == "9:16":
            frame = resize_for_aspect(center_crop_to_9x16(frame), aspect)
        out_h, out_w = frame.shape[:2]

        base = f"short_{i:02d}"
        raw_video = out_dir / f"{base}_raw.mp4"
        final_out = out_dir / f"{base}.mp4"
        thumb = out_dir / f"{base}_thumb.jpg"

        # writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_video), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            alt = out_dir / f"{base}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(alt), fourcc, fps, (out_w, out_h))
            raw_video = alt

        # first frame
        t_curr = start_f / fps
        frame = draw_caption(frame, pick_text(segs, t_curr))
        writer.write(frame)

        frames_written = 1
        while frames_written < total_frames:
            ret, frame = cap.read()
            if not ret: break
            if aspect == "9:16":
                frame = resize_for_aspect(center_crop_to_9x16(frame), aspect)
            t_curr = (start_f + frames_written) / fps
            frame = draw_caption(frame, pick_text(segs, t_curr))
            writer.write(frame)
            frames_written += 1
        writer.release()

        # audio slice
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        aac = out_dir / f"{base}.m4a"
        def run(args): subprocess.run([ff] + args, check=True)

        try:
            run(["-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src),
                 "-vn", "-acodec", "aac", "-b:a", "128k", str(aac)])
            # mux
            run(["-y", "-i", str(raw_video), "-i", str(aac),
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(final_out)])
        except subprocess.CalledProcessError:
            final_out = raw_video

        # thumbnail
        mid_f = start_f + total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, mid = cap.read()
        if ret:
            if aspect == "9:16":
                mid = resize_for_aspect(center_crop_to_9x16(mid), aspect)
            cv2.imwrite(str(thumb), mid)

    cap.release()

import os, json
from pathlib import Path
from openai import OpenAI

SCHEMA_HINT = """{
  "shorts": [
    {"title":"string","start":0.0,"end":0.0,
     "caption":"string (<=12 words)","cta":"string",
     "hashtags":["#One","#Two","#Three"],"notes":"string"}
  ]
}"""

def build_prompt(transcript: dict, scenes: list[dict], target=6, min_s=15, max_s=45, aspect="9:16") -> tuple[str,str]:
    segs = transcript["segments"]
    t_compact = "\n".join(f"[{s['start']:.2f} → {s['end']:.2f}] {s['text'].strip()}" for s in segs)
    s_compact = ""
    if scenes:
        s_compact = "SCENES:\n" + "\n".join(f"#{sc['id']}: {sc['start']:.2f}-{sc['end']:.2f} ({sc['duration']:.1f}s)"
                                            for sc in scenes)
    system = (
        "You are a precise video editor. Propose short, self-contained highlight clips.\n"
        f"- Return STRICT JSON only; match the schema exactly.\n"
        f"- {target} shorts max; each {min_s}-{max_s}s; aspect {aspect}.\n"
        "- Start/End on sentence boundaries; avoid mid-word.\n"
        "- Each item: title, start, end, caption(<=12 words), CTA, 3 hashtags, notes.\n"
        "- Use transcript timestamps; prefer staying within scene windows when provided."
    )
    user = f"TRANSCRIPT:\n{t_compact}\n\n{s_compact}\n\nJSON SCHEMA:\n{SCHEMA_HINT}\nReturn ONLY the JSON."
    return system, user

def select_highlights(transcript: dict, scenes: list[dict], model_name: str) -> dict:
    system, user = build_prompt(transcript, scenes)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    data = json.loads(resp.choices[0].message.content)
    cleaned = []
    for it in data.get("shorts", []):
        start = float(it["start"]); end = float(it["end"])
        if end > start:
            it["duration"] = round(end - start, 3)
            cleaned.append(it)
    return {"shorts": cleaned}

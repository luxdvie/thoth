# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""thoth studio — local curation UI for illustrated key-notes.

    uv run studio.py            # serves http://localhost:8511 and opens it

Reads sessions/key-notes*.md and avatars/. Every generation is saved to
generated-images/<session>/ (gitignored) with a JSON sidecar; "Promote" copies
the chosen image into gallery/<session>/ and rebuilds its gallery.md.
"""

import base64
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
AVATARS = ROOT / "avatars"
STAGING = ROOT / "generated-images"
GALLERY = ROOT / "gallery"
AUDIO_STAGING = ROOT / "generated-audio"
CUTSCENES = ROOT / "cutscenes"
PORT = 8511
STAMP_RE = re.compile(r"^(?:-|##)? ?\[(\d+):(\d\d):(\d\d)\] (.*)$")
MODELS = {  # id -> label shown in the studio picker
    "gemini-3-pro-image-preview": "Nano Banana Pro · best · ~13¢",
    "gemini-2.5-flash-image": "Nano Banana · fast · ~4¢",
}
IMAGE_MODEL = os.environ.get("THOTH_IMAGE_MODEL", "gemini-3-pro-image-preview")
NOTES_CMD = os.environ.get("THOTH_NOTES_CMD", "claude -p --model haiku")

TTS_MODEL = os.environ.get("THOTH_TTS_MODEL", "gemini-3.1-flash-tts-preview")
TTS_RATE = 24000  # Gemini TTS returns s16le mono PCM at 24 kHz
NARRATION_WPM = 95  # measured across takes: gravitas-style Charon lands 75-102 wpm; 95 centers the spread
VOICES = {  # prebuilt voice -> flavor shown in the picker
    "Charon": "deep · grave narrator",
    "Fenrir": "gravel · storm warning",
    "Kore": "warm · fireside tale",
    "Aoede": "bright · bardic",
    "Puck": "wry · trickster",
}

RECAP_PROMPT = (
    "You write voiceover narration for an illustrated D&D campaign recap — the "
    "gravitas of a 'previously on…' cold open. Below are chronicle entries covering "
    "a stretch of the session, in order. Weave them into ONE spoken recap with an "
    "arc: where things stood, what changed, where it leaves the party. Not a list — "
    "a story. AT MOST {words} words (it must fit {seconds} seconds at a slow, "
    "dramatic delivery — going over the word budget is a failure). Short sentences. "
    "Present tense. End on a hook. Keep proper nouns. Output only the script, no "
    "quotes, no stage directions.\n\n{entries}"
)

NARRATION_STYLE = (
    "Narrate with measured, dramatic gravitas — a fantasy saga's 'previously on' "
    "cold open. Deliberate pace, weight on proper nouns, let sentence ends land:\n\n"
)


def gemini_narrate(script: str, voice: str) -> tuple[bytes, float]:
    """Text → (wav bytes, duration seconds) via Gemini TTS."""
    if voice not in VOICES:
        raise RuntimeError(f"unknown voice {voice!r}")
    body = {
        "contents": [{"parts": [{"text": NARRATION_STYLE + script}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key()},
    )
    err = ""
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                parts = json.load(resp)["candidates"][0]["content"]["parts"]
                if data := next((p["inlineData"]["data"] for p in parts if "inlineData" in p), None):
                    pcm = base64.b64decode(data)
                    import struct
                    hdr = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
                           + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, TTS_RATE, TTS_RATE * 2, 2, 16)
                           + b"data" + struct.pack("<I", len(pcm)))
                    return hdr + pcm, len(pcm) / 2 / TTS_RATE
                err = "model returned no audio part"
        except Exception as e:
            err = getattr(e, "read", lambda: b"")()[-300:].decode(errors="replace") or str(e)
        if attempt == 1:
            time.sleep(10)
    raise RuntimeError(err)


ELABORATE_PROMPT = (
    "You are the art director for an illustrated D&D campaign chronicle. Turn the "
    "moment below into ONE detailed SCENE description for an image prompt: pick the "
    "single strongest instant of action, then specify composition and camera angle, "
    "what each named character ({names}) is doing and where they are in frame, and "
    "the mood of the instant. The setting below is already established separately — "
    "do NOT restate it, only the action within it. Under 130 words. Output only the "
    "scene description.\n\nSetting (already established): {context}\n\n"
    "Headline: {headline}\n\nAccount: {body}"
)

COMMAND_PROMPT = (
    "Illustrate this moment from a D&D campaign as a single dramatic fantasy scene. "
    "The attached reference images are the party's characters, in order: {names}. "
    "Keep their faces, builds, and gear recognizable. Cinematic lighting, painterly "
    "fantasy illustration style, no text or borders in the image."
)
CONTINUITY_PROMPT = (
    "The FINAL attached image is the previous keyframe of this sequence — maintain "
    "its palette, lighting, time of day, and visual continuity."
)
ASPECTS = ["16:9", "1:1", "9:16", "4:3", "3:4"]

STAGE_PROMPT = (
    "You are the stage manager for an illustrated D&D chronicle. From the transcript "
    "excerpt (noisy speech-to-text; ignore garble and out-of-game chatter), describe "
    "the physical staging during this stretch: location and environment, who is "
    "present (party members and named NPCs, with a brief visual descriptor for each "
    "NPC), spatial arrangement — who is where, doing what — time of day and weather "
    "if evident, and any ongoing state that should persist in every illustration of "
    "this stretch. 3-5 sentences, present tense, purely visual and physical facts, "
    "no plot summary. Output only the description.\n\nTranscript:\n{text}"
)


def read_bible(name: str) -> dict[str, str]:
    """avatars/party.md and avatars/npcs.md: 'Name: one visual line' per row."""
    f = AVATARS / name
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def build_prompt(context: str, scene: str, avatar_names: list[str], continuity: bool) -> str:
    party = read_bible("party.md")
    names = "; ".join(f"{n} — {party[n]}" if n in party else n for n in avatar_names)
    parts = [COMMAND_PROMPT.format(names=names)]
    text = (context + " " + scene).lower()
    npcs = {k: v for k, v in read_bible("npcs.md").items()
            if any(word.lower() in text for word in k.split()[:1])}  # only NPCs the scene names
    if npcs:
        parts.append(
            "These recurring NPCs appear in this scene without reference images — keep them "
            "consistent with these descriptions: " + "; ".join(f"{k} — {v}" for k, v in npcs.items()) + "."
        )
    if context.strip():
        parts.append("Context: " + context.strip())
    parts.append("Scene: " + scene.strip())
    if continuity:
        parts.append(CONTINUITY_PROMPT)
    return "\n\n".join(parts)


def context_file(sid: str) -> Path:
    return SESSIONS / f"context-{sid}.txt"


def stamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"[{h}:{m:02d}:{s:02d}]"


def parse_stamped_lines(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if m := STAMP_RE.match(line):
            h, mnt, s, text = m.groups()
            rows.append((int(h) * 3600 + int(mnt) * 60 + int(s), text))
    return rows


def parse_notes(path: Path) -> list[dict]:
    notes, body_lines = [], None
    for line in path.read_text().splitlines():
        if m := STAMP_RE.match(line):
            h, mnt, s, text = m.groups()
            body_lines = []
            notes.append({"t": int(h) * 3600 + int(mnt) * 60 + int(s), "headline": text, "body": body_lines})
        elif body_lines is not None and line.strip():
            body_lines.append(line.strip())
    for n in notes:
        n["body"] = " ".join(n["body"])
    return notes


def load_state() -> dict:
    sessions = []
    seen = set()
    for p in sorted(SESSIONS.glob("key-notes-*.md")):
        sid = p.stem.removeprefix("key-notes-enriched-").removeprefix("key-notes-")
        if sid in seen:
            continue
        seen.add(sid)
        enriched = SESSIONS / f"key-notes-enriched-{sid}.md"
        plain = SESSIONS / f"key-notes-{sid}.md"
        notes = parse_notes(enriched if enriched.exists() else plain)
        if enriched.exists() and plain.exists():  # headlines from plain, bodies from enriched
            bodies = {n["t"]: n["body"] for n in notes}
            notes = parse_notes(plain)
            for n in notes:
                n["body"] = bodies.get(n["t"], "")
        gdir = GALLERY / sid
        promoted_by_idx: dict[str, list[str]] = {}
        if gdir.is_dir():
            for f in sorted(gdir.glob("*.png")):
                promoted_by_idx.setdefault(f.name.split("-")[0], []).append(f"/gallery/{sid}/{f.name}")
        for i, n in enumerate(notes):
            n["idx"] = i
            n["stamp"] = stamp(n["t"])
            n["gallery"] = promoted_by_idx.get(f"{i:03d}", [])
            n["promoted"] = bool(n["gallery"])
            gen_dir = STAGING / sid
            n["generations"] = sorted(
                f"/generated/{sid}/{f.name}" for f in gen_dir.glob(f"{i:03d}-*.png")
            ) if gen_dir.is_dir() else []
        # cutscene narration takes (staged) and promoted cutscenes
        cuts: dict[str, dict] = {}
        adir = AUDIO_STAGING / sid
        if adir.is_dir():
            for f in sorted(adir.glob("cut-*.wav")):
                if not f.with_suffix(".json").exists():
                    continue
                span = "-".join(f.stem.split("-")[1:3])
                cuts.setdefault(span, {"span": span, "takes": [], "promoted": None})["takes"].append(
                    {"url": f"/generated-audio/{sid}/{f.name}", **json.loads(f.with_suffix(".json").read_text())}
                )
        cdir = CUTSCENES / sid
        if cdir.is_dir():
            for m in sorted(cdir.glob("*/manifest.json")):
                span = m.parent.name
                cuts.setdefault(span, {"span": span, "takes": [], "promoted": None})["promoted"] = {
                    "url": f"/cutscenes/{sid}/{span}/narration.wav", **json.loads(m.read_text())
                }
        sessions.append({"sid": sid, "notes": notes, "cuts": sorted(cuts.values(), key=lambda c: c["span"]),
                         "context": context_file(sid).read_text() if context_file(sid).exists() else ""})
    avatars = [
        {"name": p.stem, "url": f"/avatars/{p.name}"}
        for p in sorted(AVATARS.glob("*"))
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    return {"sessions": sessions, "avatars": avatars,
            "models": MODELS, "model": IMAGE_MODEL, "voices": VOICES, "aspects": ASPECTS}


def api_key() -> str:
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    try:  # fish universal vars aren't exported to non-fish parents
        import subprocess
        if key := subprocess.run(["fish", "-c", "echo -n $GEMINI_API_KEY"], capture_output=True, timeout=5).stdout.decode().strip():
            return key
    except Exception:
        pass
    raise RuntimeError("GEMINI_API_KEY is not set")


def image_part(p: Path) -> dict:
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else f"image/{p.suffix.lower().lstrip('.')}"
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(p.read_bytes()).decode()}}


def gemini_generate(prompt: str, avatar_names: list[str], model: str,
                    aspect: str = "16:9", continuity_path: Path | None = None) -> bytes:
    key = api_key()
    if model not in MODELS:
        raise RuntimeError(f"unknown model {model!r}")
    if aspect not in ASPECTS:
        raise RuntimeError(f"unknown aspect {aspect!r}")
    parts = []
    for name in avatar_names:
        matches = [p for p in AVATARS.glob(f"{name}.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not matches:
            raise RuntimeError(f"no avatar image for {name!r}")
        parts.append(image_part(matches[0]))
    if continuity_path is not None:
        parts.append(image_part(continuity_path))  # last, as CONTINUITY_PROMPT promises
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}], "generationConfig": {
        "responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": aspect}}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    err = ""
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                rparts = json.load(resp)["candidates"][0]["content"]["parts"]
                if data := next((p["inlineData"]["data"] for p in rparts if "inlineData" in p), None):
                    return base64.b64decode(data)
                err = "model returned no image part (likely refused the prompt)"
        except Exception as e:
            err = getattr(e, "read", lambda: b"")()[-300:].decode(errors="replace") or str(e)
        if attempt == 1:
            time.sleep(10)
    raise RuntimeError(err)


def rebuild_gallery_md(sid: str) -> None:
    gdir = GALLERY / sid
    caps_path = gdir / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    entries = []
    for f in sorted(gdir.glob("*.png")):
        c = caps.get(f.name, {})
        entries.append(f"## {c.get('stamp', '')} {c.get('headline', f.name)}\n\n![]({f.name})\n")
    (gdir / "gallery.md").write_text(f"# {sid}\n\n" + "\n".join(entries))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, base: Path, rel: str, ctype: str):
        f = (base / rel).resolve()
        if not f.is_relative_to(base) or not f.is_file():
            return self._send(404, b"{}")
        self._send(200, f.read_bytes(), ctype)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, (ROOT / "studio.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(load_state()).encode())
        elif self.path.startswith("/avatars/"):
            self._static(AVATARS, self.path.removeprefix("/avatars/"), "image/png")
        elif self.path.startswith("/generated/"):
            self._static(STAGING, self.path.removeprefix("/generated/"), "image/png")
        elif self.path.startswith("/generated-audio/"):
            self._static(AUDIO_STAGING, self.path.removeprefix("/generated-audio/"), "audio/wav")
        elif self.path.startswith("/cutscenes/"):
            self._static(CUTSCENES, self.path.removeprefix("/cutscenes/"), "audio/wav")
        elif self.path.startswith("/gallery/"):
            self._static(GALLERY, self.path.removeprefix("/gallery/"), "image/png")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        try:
            if self.path == "/api/generate":
                sid, idx = payload["sid"], int(payload["idx"])
                model = payload.get("model", IMAGE_MODEL)
                aspect = payload.get("aspect", "16:9")
                context, scene = payload.get("context", ""), payload["scene"]
                cont_path = None
                if payload.get("continuity"):
                    cont_path = (GALLERY / Path(payload["continuity"].removeprefix("/gallery/"))).resolve()
                    assert cont_path.is_relative_to(GALLERY) and cont_path.is_file()
                prompt = build_prompt(context, scene, payload.get("avatars", []), cont_path is not None)
                dest_dir = STAGING / sid
                dest_dir.mkdir(parents=True, exist_ok=True)
                serial = len(list(dest_dir.glob(f"{idx:03d}-*.png")))
                png = gemini_generate(prompt, payload.get("avatars", []), model, aspect, cont_path)
                dest = dest_dir / f"{idx:03d}-{serial:02d}.png"
                dest.write_bytes(png)
                dest.with_suffix(".json").write_text(json.dumps({
                    "prompt": prompt, "context": context, "scene": scene,
                    "avatars": payload.get("avatars", []), "aspect": aspect,
                    "continuity": payload.get("continuity"),
                    "headline": payload.get("headline", ""), "stamp": payload.get("stamp", ""),
                    "model": model,
                }, indent=2))
                context_file(sid).write_text(context)  # persist the session's setting
                self._send(200, json.dumps({"url": f"/generated/{sid}/{dest.name}"}).encode())
            elif self.path == "/api/context":
                context_file(payload["sid"]).write_text(payload.get("context", ""))
                self._send(200, b"{}")
            elif self.path == "/api/stage-context":
                import subprocess
                sid = payload["sid"]
                t0, t1 = float(payload["t0"]), float(payload["t1"])
                polished = SESSIONS / f"session-{sid}-polished.md"
                transcript_path = polished if polished.exists() else SESSIONS / f"session-{sid}.md"
                lines = [text for ts, text in parse_stamped_lines(transcript_path) if t0 - 30 <= ts <= t1 + 30]
                excerpt = "\n".join(lines)[:9000]
                if not excerpt.strip():
                    raise RuntimeError("no transcript found for that window")
                r = subprocess.run(NOTES_CMD, shell=True, capture_output=True, timeout=120,
                                   input=STAGE_PROMPT.format(text=excerpt).encode())
                text = r.stdout.decode().strip()
                if r.returncode != 0 or not text:
                    raise RuntimeError(r.stderr.decode().strip()[-200:] or "stage manager returned nothing")
                self._send(200, json.dumps({"context": text, "source": transcript_path.name}).encode())
            elif self.path == "/api/cutscene-script":
                import subprocess
                seconds = float(payload.get("seconds", 30))
                words = int(seconds * NARRATION_WPM / 60)
                entries = "\n\n".join(
                    f"{e['stamp']} {e['headline']}\n{e.get('body', '')}".strip()
                    for e in payload.get("entries", [])
                )
                prompt = RECAP_PROMPT.format(words=words, seconds=int(seconds), entries=entries)
                r = subprocess.run(NOTES_CMD, shell=True, capture_output=True, timeout=120, input=prompt.encode())
                text = r.stdout.decode().strip()
                if r.returncode != 0 or not text:
                    raise RuntimeError(r.stderr.decode().strip()[-200:] or "scriptwriter returned nothing")
                self._send(200, json.dumps({"script": text, "words": len(text.split()), "budget": words}).encode())
            elif self.path == "/api/narrate":
                sid, span = payload["sid"], payload["span"]  # e.g. "003-008"
                assert re.fullmatch(r"\d{3}-\d{3}", span)
                dest_dir = AUDIO_STAGING / sid
                dest_dir.mkdir(parents=True, exist_ok=True)
                serial = len(list(dest_dir.glob(f"cut-{span}-*.wav")))
                wav, duration = gemini_narrate(payload["script"], payload.get("voice", "Charon"))
                dest = dest_dir / f"cut-{span}-{serial:02d}.wav"
                dest.write_bytes(wav)
                meta = {"script": payload["script"], "voice": payload.get("voice", "Charon"),
                        "duration": round(duration, 2), "target": payload.get("seconds", 30),
                        "span": span, "model": TTS_MODEL}
                dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
                self._send(200, json.dumps({"url": f"/generated-audio/{sid}/{dest.name}", **meta}).encode())
            elif self.path == "/api/promote-cutscene":
                sid, span = payload["sid"], payload["span"]
                assert re.fullmatch(r"\d{3}-\d{3}", span)
                src = (AUDIO_STAGING / sid / Path(payload["file"]).name).resolve()
                assert src.is_relative_to(AUDIO_STAGING) and src.is_file()
                cdir = CUTSCENES / sid / span
                cdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, cdir / "narration.wav")
                meta = json.loads(src.with_suffix(".json").read_text())
                meta["keyframes"] = payload.get("keyframes", [])  # promoted gallery paths in span, at promote time
                meta["stamps"] = payload.get("stamps", [])
                (cdir / "manifest.json").write_text(json.dumps(meta, indent=2))
                self._send(200, json.dumps({"promoted": f"/cutscenes/{sid}/{span}/narration.wav"}).encode())
            elif self.path == "/api/elaborate":
                import subprocess
                prompt = ELABORATE_PROMPT.format(
                    names=", ".join(payload.get("avatars", [])),
                    context=payload.get("context", "") or "(none)",
                    headline=payload.get("headline", ""), body=payload.get("body", "") or "(none)",
                )
                r = subprocess.run(NOTES_CMD, shell=True, capture_output=True, timeout=120, input=prompt.encode())
                text = r.stdout.decode().strip()
                if r.returncode != 0 or not text:
                    raise RuntimeError(r.stderr.decode().strip()[-200:] or "elaborator returned nothing")
                self._send(200, json.dumps({"scene": text}).encode())
            elif self.path == "/api/promote":
                sid, idx = payload["sid"], int(payload["idx"])
                src = (STAGING / sid / Path(payload["file"]).name).resolve()
                assert src.is_relative_to(STAGING) and src.is_file()
                gdir = GALLERY / sid
                gdir.mkdir(parents=True, exist_ok=True)
                dest = gdir / f"{idx:03d}-{payload.get('stamp', '').strip('[]').replace(':', '-')}.png"
                shutil.copy2(src, dest)
                caps_path = gdir / "captions.json"
                caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
                caps[dest.name] = {"stamp": payload.get("stamp", ""), "headline": payload.get("headline", "")}
                caps_path.write_text(json.dumps(caps, indent=2))
                rebuild_gallery_md(sid)
                self._send(200, json.dumps({"promoted": f"/gallery/{sid}/{dest.name}"}).encode())
            else:
                self._send(404, b"{}")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)[:400]}).encode())


def main() -> None:
    if not SESSIONS.is_dir():
        sys.exit("run from the thoth repo (no sessions/ here)")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"𓅝 thoth studio → http://localhost:{PORT}  (Ctrl-C to stop)")
    threading.Timer(0.4, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

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
    "scene description.{canon}\n\nSetting (already established): {context}\n\n"
    "Headline: {headline}\n\nAccount: {body}"
)

COMMAND_PROMPT = (
    "Illustrate this moment from a D&D campaign as a single dramatic fantasy scene. "
    "The attached reference images are the party's characters, in order: {names}. "
    "IDENTITY LOCK: each character's face, head shape, skin, species, and body "
    "proportions must match their reference image exactly — the references override "
    "any wording below. Never blend a party member with a creature in the scene "
    "(kobolds, beasts, monsters) and never add snouts, horns, tails, or fur they "
    "don't have in their reference. Keep their builds and gear recognizable. "
    "Cinematic lighting, painterly fantasy illustration style. No text, names, "
    "labels, lettering, or borders anywhere in the image."
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


CANON_CLAUSE = (
    "\n\nCharacter canon — the source of truth for how each character looks and "
    "what they own. Every feature, mark, item, and power belongs ONLY to the "
    "character whose canon lists it: when the transcript mentions one (\"my "
    "tattoo\", \"the crystal\", \"the scarf\"), attribute it to its owner here, never "
    "to whoever is nearby in the scene, and never give a character a feature their "
    "canon doesn't have. Use the canon to make descriptions specific (exact "
    "garments, weapon, colors). Before answering, check every attribution against "
    "it.\n{sheets}"
)


def canon_block(names: list[str]) -> str:
    """avatars/sheets/<Name>.md dossiers for the given characters (empty if none)."""
    sheets = [(AVATARS / "sheets" / f"{n}.md") for n in names]
    text = "\n\n".join(p.read_text().strip() for p in sheets if p.exists())
    return CANON_CLAUSE.format(sheets=text) if text else ""


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


def discord_webhook() -> str:
    """THOTH_DISCORD_WEBHOOK (env, or fish universal var like the Gemini key). '' = posting off."""
    if url := os.environ.get("THOTH_DISCORD_WEBHOOK"):
        return url
    try:
        import subprocess
        return subprocess.run(["fish", "-c", "echo -n $THOTH_DISCORD_WEBHOOK"],
                              capture_output=True, timeout=5).stdout.decode().strip()
    except Exception:
        return ""


def discord_post(png: Path, caption: str) -> None:
    """Upload one image + caption to the webhook's channel (multipart, stdlib only)."""
    url = discord_webhook()
    if not url:
        raise RuntimeError("THOTH_DISCORD_WEBHOOK is not set")
    boundary = f"thoth{int(time.time() * 1000)}"
    payload = json.dumps({"content": caption[:1900], "username": "thoth",
                          "allowed_mentions": {"parse": []}})
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
            f"Content-Type: application/json\r\n\r\n{payload}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; filename=\"{png.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + png.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "thoth (local, 1.0)"})
    err = ""
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=60):
                return
        except Exception as e:
            err = getattr(e, "read", lambda: b"")()[-300:].decode(errors="replace") or str(e)
        if attempt == 1:
            time.sleep(5)
    raise RuntimeError(f"discord: {err}")


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


# ── live stage: auto-illustrate the running session (second-monitor view at /live) ──

LIVE_CAST = [n.strip() for n in os.environ.get("THOTH_LIVE_CAST", "Corvus,Punchy,Stone").split(",") if n.strip()]
LIVE_MODEL = os.environ.get("THOTH_LIVE_MODEL", IMAGE_MODEL)
LIVE_WINDOW = float(os.environ.get("THOTH_LIVE_WINDOW", 900))  # seconds of transcript the director reads per beat
LIVE_CHARS = 30000  # cap on that excerpt (newest kept) — haiku reads it in ~10 s
LIVE_BEAT = float(os.environ.get("THOTH_LIVE_BEAT", 90))  # paint at least every N s of transcript, key-note or not

DIRECTOR_PROMPT = (
    "You are the art director for a LIVE illustrated D&D session — each beat you "
    "paint becomes the image on the table's display. The party in every image: "
    "{names}. Below: the story so far (the session's key-notes), what you already "
    "painted recently, the current established setting, and the last ~15 minutes "
    "of noisy timestamped speech-to-text (ignore garble and out-of-game chatter). "
    "Use the older material to stay in sync — who is present, where they are, what "
    "they carry, what just happened — but paint what is happening at the END of the "
    "transcript (the last ~90 seconds). Don't repaint a moment already painted "
    "unless nothing new happened; advance the story.{direction}\n\nReturn exactly three sections:\n"
    "HEADLINE: one caption line, at most 12 words, present tense — what is happening.\n"
    "CONTEXT: 2-4 sentences of persistent physical staging — location, "
    "environment, time of day, weather, NPCs present (with a brief visual "
    "descriptor). Keep the established setting unless the story clearly moved.\n"
    "SCENE: under 120 words — the single strongest instant of action right now: "
    "composition and camera angle, what each of {names} is doing and where they "
    "are in frame, mood. Purely visual.{glossary}{canon}\n\n"
    "Story so far:\n{story}\n\nRecently painted:\n{painted}\n\n"
    "Established setting: {context}\n\nLatest headline: {headline}\n\n"
    "Transcript:\n{text}"
)
DIRECTION_CLAUSE = (
    "\n\nThe game master has given a DIRECTION for this image — it overrides "
    "everything else, follow it exactly: {direction}"
)


def live_session() -> str | None:
    """Newest live transcript (session-<sid>.md, not a polished/attributed derivative)."""
    cands = [p for p in SESSIONS.glob("session-*.md")
             if re.fullmatch(r"session-\d{4}-\d\d-\d\d-\d{4}", p.stem)]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime).stem.removeprefix("session-")


def run_notes_cmd(prompt: str, timeout: int = 120) -> str:
    import subprocess
    r = subprocess.run(NOTES_CMD, shell=True, capture_output=True, timeout=timeout, input=prompt.encode())
    text = r.stdout.decode().strip()
    if r.returncode != 0 or not text:
        raise RuntimeError(r.stderr.decode().strip()[-200:] or "notes command returned nothing")
    return text


class LiveDirector:
    """Watches the newest session's key-notes; every new note (thoth only writes
    one when something new happened — SKIP otherwise) becomes a job: director LLM
    → context+scene → Gemini image of LIVE_CAST, with the previous live frame as
    continuity. One job at a time; a newer pending job replaces an older one, and
    a GM direction always beats an auto beat. Frames land in generated-images/
    <sid>/NNN-SS.png (NNN = note index), so the studio can promote them later."""

    def __init__(self, auto: bool):
        self.lock = threading.Condition()
        self.auto = auto
        self.continuity = True
        self.discord = bool(discord_webhook())  # auto-post each frame when a webhook is configured
        self.discord_error = ""
        self.fresh_setting = False  # GM just changed the setting: skip continuity once
        self.pinned = False  # GM-saved setting: director may not rewrite it (and continuity resets on change)
        self.model = LIVE_MODEL
        self.cast = [n for n in LIVE_CAST if self._avatar(n)]
        self.sid: str | None = None
        self.seen_notes = 0
        self.last_beat_t = 0.0  # transcript time of the last self-triggered beat (no-notes mode)
        self.pending: dict | None = None
        self.status = "idle"
        self.error = ""
        self.history: list[dict] = []
        threading.Thread(target=self._watch, daemon=True).start()
        threading.Thread(target=self._work, daemon=True).start()

    @staticmethod
    def _avatar(name: str) -> Path | None:
        return next((p for p in AVATARS.glob(f"{name}.*")
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")), None)

    def _load_history(self, sid: str) -> list[dict]:
        out = []
        d = STAGING / sid
        for j in sorted(d.glob("*.json")) if d.is_dir() else []:
            meta = json.loads(j.read_text())
            if meta.get("live") and j.with_suffix(".png").exists():
                out.append({"url": f"/generated/{sid}/{j.stem}.png", **meta})
        return sorted(out, key=lambda m: m.get("made", 0))

    def _notes(self, sid: str) -> list[dict]:
        p = SESSIONS / f"key-notes-{sid}.md"
        return parse_notes(p) if p.exists() else []

    def _watch(self):
        while True:
            try:
                sid = live_session()
                with self.lock:
                    if sid and sid != self.sid:  # new session: adopt it, don't backfill old notes
                        self.sid = sid
                        self.history = self._load_history(sid)
                        self.seen_notes = len(self._notes(sid))
                        self.last_beat_t = self._transcript_end(sid)
                        self.pinned = (SESSIONS / f"context-{sid}.pinned").exists()
                    if self.sid:
                        fire = False
                        notes = self._notes(self.sid)
                        if len(notes) > self.seen_notes:  # a key-note = something new happened
                            self.seen_notes, fire = len(notes), True
                        end = self._transcript_end(self.sid)
                        if end - self.last_beat_t >= LIVE_BEAT:  # …and never go stale between notes
                            fire = True
                        if fire:
                            self.last_beat_t = end
                        if fire and self.auto and (not self.pending or self.pending["source"] == "auto"):
                            self.pending = {"source": "auto", "direction": ""}
                            self.lock.notify_all()
            except Exception as e:
                self.error = f"watcher: {e}"
            time.sleep(3)

    @staticmethod
    def _transcript_end(sid: str) -> float:
        p = SESSIONS / f"session-{sid}.md"
        rows = parse_stamped_lines(p) if p.exists() else []
        return rows[-1][0] if rows else 0.0

    def set_context(self, text: str) -> None:
        if self.sid:
            context_file(self.sid).write_text(text.strip())
            self._set_pinned(bool(text.strip()))
            self.fresh_setting = True

    def _set_pinned(self, on: bool) -> None:
        self.pinned = on
        if self.sid:  # marker file so a stage restart keeps the GM's pin
            marker = SESSIONS / f"context-{self.sid}.pinned"
            marker.touch() if on else marker.unlink(missing_ok=True)

    def request(self, direction: str, literal: bool, context: str | None = None) -> None:
        with self.lock:
            if not self.sid:
                raise RuntimeError("no session found in sessions/ — start thoth first")
            if context is not None:
                self.set_context(context)
            self.last_beat_t = self._transcript_end(self.sid)  # a GM paint resets the auto cadence
            self.pending = {"source": "direct", "direction": direction.strip(), "literal": literal}
            self.lock.notify_all()

    def _work(self):
        while True:
            with self.lock:
                while not self.pending:
                    self.lock.wait()
                job, self.pending = self.pending, None
            try:
                self._run(job)
                self.error = ""
            except Exception as e:
                self.error = str(e)[:400]
            self.status = "idle"

    def _run(self, job: dict):
        sid = self.sid
        notes = self._notes(sid)
        note = dict(notes[-1]) if notes else {"t": self._transcript_end(sid), "headline": "", "body": ""}
        if job["source"] == "direct" or not notes:
            note["t"] = self._transcript_end(sid)  # stamp manual/self-cut beats at "now"
        idx = max(len(notes) - 1, 0)
        cf = context_file(sid)
        context = cf.read_text().strip() if cf.exists() else ""
        prev = self.history[-1] if self.history else None

        if job.get("literal") and job["direction"]:
            scene = job["direction"]  # GM wrote the scene verbatim; keep the setting
        else:
            self.status = "directing"
            transcript = SESSIONS / f"session-{sid}.md"
            rows = parse_stamped_lines(transcript) if transcript.exists() else []
            end = rows[-1][0] if rows else 0
            text = "\n".join(f"{stamp(ts)} {t}" for ts, t in rows if ts >= end - LIVE_WINDOW)[-LIVE_CHARS:]
            story = "\n".join(f"{stamp(n['t'])} {n['headline']}" for n in notes) or "(none yet)"
            painted = "\n".join(f"- {h.get('headline', '')}" for h in self.history[-4:]) or "(nothing yet)"
            gloss = AVATARS / "glossary.md"
            out = run_notes_cmd(DIRECTOR_PROMPT.format(
                names=", ".join(self.cast),
                direction=DIRECTION_CLAUSE.format(direction=job["direction"]) if job["direction"] else "",
                glossary=("\n\nGlossary (correct spellings):\n" + gloss.read_text().strip()) if gloss.exists() else "",
                context=(context + "\n(PINNED by the game master — this is where the party IS now; "
                         "return it unchanged as CONTEXT, ignore older locations in the transcript)")
                        if self.pinned and context else context or "(none yet — infer it)",
                headline=note["headline"] or "(none yet)",
                text=text or "(no transcript yet)", story=story, painted=painted,
                canon=canon_block(self.cast),
            ), timeout=180)
            out = out.replace("**", "").replace("__", "")  # haiku likes markdown-bolding section labels
            m = re.search(r"CONTEXT:\s*(.*?)\s*SCENE:\s*(.*)", out, re.S | re.I)
            if m:
                scene = m.group(2).strip()
                if not self.pinned:
                    context = m.group(1).strip()
            else:
                scene = out
            if h := re.search(r"HEADLINE:\s*(.+)", out, re.I):
                note["headline"] = h.group(1).strip().strip('"')  # director's caption is fresher than the last note
            cf.write_text(context)

        self.status = "painting"
        cont_path = None
        # continuity anchors palette AND place — feeding the old location's frame after a
        # setting change drags the party back there (field bug: stuck in the mine in town)
        # (only on a GM setting change — the director rewords CONTEXT every beat, so a
        # text-equality check silently disabled continuity for every auto frame)
        if self.continuity and prev and not self.fresh_setting:
            p = STAGING / sid / Path(prev["url"]).name
            cont_path = p if p.is_file() else None
        prompt = build_prompt(context, scene, self.cast, cont_path is not None)
        png = gemini_generate(prompt, self.cast, self.model, "16:9", cont_path)
        self.fresh_setting = False
        dest_dir = STAGING / sid
        dest_dir.mkdir(parents=True, exist_ok=True)
        serial = len(list(dest_dir.glob(f"{idx:03d}-*.png")))
        dest = dest_dir / f"{idx:03d}-{serial:02d}.png"
        dest.write_bytes(png)
        meta = {"prompt": prompt, "context": context, "scene": scene, "avatars": self.cast,
                "aspect": "16:9", "continuity": prev["url"] if cont_path else None,
                "headline": note["headline"], "stamp": stamp(note["t"]), "model": self.model,
                "live": True, "source": job["source"], "direction": job["direction"], "made": time.time()}
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        with self.lock:
            self.history.append({"url": f"/generated/{sid}/{dest.name}", **meta})
        if self.discord:  # off the worker thread: a slow Discord never delays the next frame
            caption = f"**{meta['stamp']}** {meta['headline']}".strip()
            if job["source"] == "direct" and job["direction"]:
                caption += f"\n> *{job['direction'][:300]}*"
            threading.Thread(target=self._post, args=(dest, caption), daemon=True).start()

    def _post(self, png: Path, caption: str) -> None:
        try:
            discord_post(png, caption)
            self.discord_error = ""
        except Exception as e:
            self.discord_error = str(e)[:300]

    def state(self) -> dict:
        cf = context_file(self.sid) if self.sid else None
        notes = self._notes(self.sid) if self.sid else []
        return {
            "sid": self.sid, "auto": self.auto, "continuity": self.continuity, "pinned": self.pinned, "model": self.model,
            "discord": self.discord, "discord_ready": bool(discord_webhook()), "discord_error": self.discord_error,
            "models": MODELS, "cast": self.cast, "status": self.status, "error": self.error,
            "queued": bool(self.pending), "notes": len(notes), "beat": LIVE_BEAT,
            "trigger": "key-notes" if self.sid and (SESSIONS / f"key-notes-{self.sid}.md").exists() else "transcript",
            "headline": notes[-1]["headline"] if notes else "",
            "context": cf.read_text() if cf and cf.exists() else "",
            "history": [{k: h.get(k) for k in ("url", "headline", "stamp", "scene", "source", "direction", "made")}
                        for h in self.history[-60:]],
        }


DIRECTOR: LiveDirector | None = None


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
        elif self.path in ("/live", "/live/"):
            self._send(200, (ROOT / "live.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/live":
            self._send(200, json.dumps(DIRECTOR.state()).encode())
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
            if self.path == "/api/live/direct":
                DIRECTOR.request(payload.get("direction", ""), bool(payload.get("literal")), payload.get("context"))
                self._send(200, b"{}")
            elif self.path == "/api/live/settings":
                for k in ("auto", "continuity"):
                    if k in payload:
                        setattr(DIRECTOR, k, bool(payload[k]))
                if payload.get("model") in MODELS:
                    DIRECTOR.model = payload["model"]
                if "context" in payload:
                    DIRECTOR.set_context(payload["context"])
                if "discord" in payload:
                    DIRECTOR.discord = bool(payload["discord"]) and bool(discord_webhook())
                if "pinned" in payload:
                    DIRECTOR._set_pinned(bool(payload["pinned"]))
                self._send(200, json.dumps(DIRECTOR.state()).encode())
            elif self.path == "/api/generate":
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
                    canon=canon_block(payload.get("avatars", [])),
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
    import argparse
    ap = argparse.ArgumentParser(description="thoth studio — curation UI + live stage")
    ap.add_argument("--live", action="store_true",
                    help=f"auto-paint {', '.join(LIVE_CAST)} on every new key-note of the running session; opens /live")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser tab (restarts under a running stage)")
    args = ap.parse_args()
    if not SESSIONS.is_dir():
        sys.exit("run from the thoth repo (no sessions/ here)")
    global DIRECTOR
    DIRECTOR = LiveDirector(auto=args.live)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    page = "/live" if args.live else ""
    print(f"𓅝 thoth studio → http://localhost:{PORT}   live stage → http://localhost:{PORT}/live  (Ctrl-C to stop)")
    if args.live:
        print(f"   auto-paint ON · cast {', '.join(DIRECTOR.cast)} · {DIRECTOR.model}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(f"http://localhost:{PORT}{page}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

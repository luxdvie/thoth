# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "parakeet-mlx>=0.3",
#     "sounddevice>=0.5",
#     "sherpa-onnx>=1.12",
#     "numpy",
#     "numba>=0.60",
# ]
# ///
"""thoth — realtime mic transcription to terminal + disk, with speaker labels.

Usage:
    uv run thoth.py                # new session file in ./sessions/
    uv run thoth.py --out ~/dnd    # choose output dir
    Ctrl-C to stop; transcript is flushed continuously, so a crash loses nothing.
"""

import argparse
import base64
import datetime
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio

CHUNK_SECONDS = 2.0
MIN_EMBED_SECONDS = 0.5
EMBED_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)
CACHE_DIR = Path.home() / ".cache" / "thoth"
SPEAKER_COLORS = ["\x1b[96m", "\x1b[93m", "\x1b[92m", "\x1b[95m", "\x1b[94m", "\x1b[91m"]
RESET = "\x1b[0m"


def stamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"[{h}:{m:02d}:{s:02d}]"


def ensure_embed_model() -> Path:
    dest = CACHE_DIR / "titanet_small.onnx"
    if not dest.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading speaker model (40 MB, one time) …", file=sys.stderr)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(EMBED_MODEL_URL, tmp)
        tmp.rename(dest)
    return dest


class SpeakerLog:
    """Online speaker identification over sentence-sized audio slices.

    Two-tier matching: a loose floor decides which existing speaker a slice
    belongs to; only confident matches (floor + margin) update that speaker's
    centroid, so noisy far-field segments can't drift the voiceprints."""

    def __init__(self, model_path: Path, rate: int, threshold: float, max_speakers: int):
        import sherpa_onnx

        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2)
        )
        self.rate = rate
        self.assign_floor = threshold
        self.update_floor = threshold + 0.15
        self.max_speakers = max_speakers
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.last = 0

    def label(self, samples: np.ndarray) -> int:
        if len(samples) < int(MIN_EMBED_SECONDS * self.rate):
            return self.last  # too short to embed reliably; assume same voice
        stream = self.extractor.create_stream()
        stream.accept_waveform(self.rate, samples)
        stream.input_finished()
        emb = np.array(self.extractor.compute(stream))
        emb /= np.linalg.norm(emb)

        if self.centroids:
            sims = [float(np.dot(emb, c)) for c in self.centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.assign_floor or len(self.centroids) >= self.max_speakers:
                if sims[best] >= self.update_floor:
                    n = self.counts[best]
                    self.centroids[best] = (self.centroids[best] * n + emb) / (n + 1)
                    self.centroids[best] /= np.linalg.norm(self.centroids[best])
                    self.counts[best] += 1
                self.last = best
                return best
        self.centroids.append(emb)
        self.counts.append(1)
        self.last = len(self.centroids) - 1
        return self.last


def render(rows: list[tuple[float, int | None, str]], live: str) -> str:
    lines = [
        f"{stamp(start)} **Speaker {label + 1}:** {text}" if label is not None else f"{stamp(start)} {text}"
        for start, label, text in rows
    ]
    if live:
        lines.append(f"… {live}")
    return "\n".join(lines) + "\n"


def tprint(start: float, label: int | None, text: str) -> None:
    if label is None:
        print(f"\x1b[2K\r{stamp(start)} {text}")
        return
    color = SPEAKER_COLORS[label % len(SPEAKER_COLORS)]
    print(f"\x1b[2K\r{stamp(start)} {color}Speaker {label + 1}:{RESET} {text}")


NOTE_PROMPT = (
    "You are live-posting a D&D session from a transcript excerpt. In one short "
    "sentence of at most 15 words, in the style 'the group is fishing' / 'the group "
    "found a poison needle', say what the party is doing right now. The transcript "
    "is noisy speech-to-text; ignore garbled fragments. If nothing meaningfully new "
    "has happened since the previous post, output exactly SKIP. Output only the "
    "sentence, no quotes.\n\nPrevious post: {prev}\n\nTranscript excerpt:\n{text}"
)


ENRICH_PROMPT = (
    "You are the chronicler of a D&D campaign. Below is a one-line headline from a "
    "live session, and the raw speech-to-text transcript from those minutes (noisy; "
    "ignore garbled fragments). Write a vivid 2-4 sentence account of what actually "
    "happened — keep proper nouns, dice outcomes, decisions, and good table jokes. "
    "Write in past tense. Output only the account, no headline, no quotes.\n\n"
    "Headline: {headline}\n\nTranscript:\n{text}"
)
STAMP_RE = re.compile(r"^(?:-|##)? ?\[(\d+):(\d\d):(\d\d)\] (.*)$")


def parse_stamped(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if m := STAMP_RE.match(line):
            h, mnt, s, text = m.groups()
            rows.append((int(h) * 3600 + int(mnt) * 60 + int(s), text))
    return rows


def enrich(notes_path: Path, cmd: str, window: float) -> None:
    """Expand each key-note headline into a rich paragraph, grounded in the
    transcript of its window. Prefers the polished transcript when present."""
    sid = notes_path.stem.removeprefix("key-notes-")
    polished = notes_path.parent / f"session-{sid}-polished.md"
    transcript_path = polished if polished.exists() else notes_path.parent / f"session-{sid}.md"
    if not transcript_path.exists():
        sys.exit(f"no transcript found for {notes_path.name} (looked for session-{sid}[-polished].md)")
    transcript = parse_stamped(transcript_path)
    headlines = parse_stamped(notes_path)
    out = notes_path.parent / f"key-notes-enriched-{sid}.md"
    print(f"Enriching {len(headlines)} notes from {transcript_path.name} …", file=sys.stderr)

    chunks = []
    for i, (t, headline) in enumerate(headlines):
        t_end = headlines[i + 1][0] if i + 1 < len(headlines) else t + window
        excerpt = "\n".join(text for ts, text in transcript if t - 15 <= ts < min(t_end, t + window) + 15)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, timeout=300,
            input=ENRICH_PROMPT.format(headline=headline, text=excerpt).encode(),
        )
        body = result.stdout.decode().strip() if result.returncode == 0 else ""
        if not body:
            body = f"*(enrichment failed: {result.stderr.decode().strip()[-120:]})*"
        chunks.append(f"## {stamp(t)} {headline}\n\n{body}\n")
        print(f"\x1b[2K\r[{i + 1}/{len(headlines)}] {headline[:60]}", end="", file=sys.stderr, flush=True)
        out.write_text("\n".join(chunks))  # flush as we go
    print(f"\x1b[2K\rEnriched notes: {out}", file=sys.stderr)


IMAGE_PROMPT = (
    "Illustrate this moment from a D&D campaign as a single dramatic fantasy scene. "
    "The attached reference images are the party's characters ({names}) — keep their "
    "faces, builds, and gear recognizable. Cinematic lighting, painterly fantasy "
    "illustration style, no text or borders in the image.\n\nScene: {scene}"
)


def imagine(notes_path: Path, avatars_dir: Path, model: str, post_cmd: str | None) -> None:
    """Generate one scene image per key-note via the Gemini image API, using the
    avatar images as character references. Images land in gallery/<session>/."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:  # fish universal vars aren't exported to non-fish parents
        try:
            key = subprocess.run(["fish", "-c", "echo -n $GEMINI_API_KEY"], capture_output=True, timeout=5).stdout.decode().strip()
        except Exception:
            pass
    if not key:
        sys.exit("GEMINI_API_KEY is not set (create one at https://aistudio.google.com/apikey)")
    avatars = sorted(p for p in avatars_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not avatars:
        sys.exit(f"no avatar images found in {avatars_dir}/")
    party = {}  # avatars/party.md: "Name: one visual line" — same format the studio uses
    if (avatars_dir / "party.md").exists():
        for line in (avatars_dir / "party.md").read_text().splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                party[k.strip()] = v.strip()
    names = "; ".join(f"{p.stem} — {party[p.stem]}" if p.stem in party else p.stem for p in avatars)
    avatar_parts = [
        {"inline_data": {
            "mime_type": "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else f"image/{p.suffix.lower().lstrip('.')}",
            "data": base64.b64encode(p.read_bytes()).decode(),
        }}
        for p in avatars
    ]

    sid = notes_path.stem.removeprefix("key-notes-enriched-").removeprefix("key-notes-")
    notes = parse_stamped(notes_path)
    gallery = Path("gallery") / sid
    gallery.mkdir(parents=True, exist_ok=True)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    print(f"Imagining {len(notes)} scenes with {len(avatars)} character references → {gallery}/", file=sys.stderr)

    captions = []
    for i, (t, headline) in enumerate(notes):
        dest = gallery / f"{i:03d}-{stamp(t).strip('[]').replace(':', '-')}.png"
        if dest.exists():  # resumable: rerun skips finished scenes
            captions.append((dest, t, headline))
            continue
        body = {
            "contents": [{"parts": avatar_parts + [
                {"text": IMAGE_PROMPT.format(names=names, scene=headline)}
            ]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        image = err = None
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    parts = json.load(resp)["candidates"][0]["content"]["parts"]
                    image = next((p["inlineData"]["data"] for p in parts if "inlineData" in p), None)
                    err = None if image else "model returned no image part"
            except Exception as e:
                err = getattr(e, "read", lambda: b"")()[-300:].decode(errors="replace") or str(e)
            if image:
                break
            if attempt == 1:
                time.sleep(15)
        if image is None:
            print(f"\n[imagine] scene {i} failed: {err}", file=sys.stderr)
            continue
        dest.write_bytes(base64.b64decode(image))
        captions.append((dest, t, headline))
        print(f"\x1b[2K\r[{i + 1}/{len(notes)}] {dest.name}", end="", file=sys.stderr, flush=True)
        (gallery / "gallery.md").write_text(
            f"# {sid}\n\n" + "\n".join(f"## {stamp(tt)} {h}\n\n![{h}]({d.name})\n" for d, tt, h in captions)
        )
        if post_cmd:
            subprocess.run(post_cmd, shell=True, input=f"{dest}\n{headline}".encode(), capture_output=True)
    print(f"\x1b[2K\rGallery: {gallery}/gallery.md  ({len(captions)}/{len(notes)} scenes)", file=sys.stderr)


ATTRIBUTE_PROMPT = (
    "You attribute speakers in a D&D session transcript (noisy speech-to-text — "
    "ignore garble). The cast:\n{cast}\n\n"
    "Dialogue is self-identifying: the DM narrates, voices NPCs, and calls for "
    "rolls; players reference their own abilities, sheets, and dice; people address "
    "each other by name. Use those clues plus continuity with the previous lines.\n\n"
    "Previously attributed lines:\n{prev}\n\n"
    "Short reactions ('Yeah.', 'Nice.') usually belong to whoever the DM is "
    "addressing or whoever spoke last in that exchange — infer from conversational "
    "flow. Prefer your best inference; reserve '?' for lines with no usable signal "
    "at all.\n\n"
    "Now attribute these {n} numbered lines. Output EXACTLY {n} lines: line k is "
    "the speaker of input line k — a name from the cast, or '?'. No numbering, no "
    "commentary, nothing else.\n\n{lines}"
)
ATTRIBUTE_CHUNK = 120


def attribute(transcript: Path, cast_path: Path, cmd: str) -> None:
    """Label each transcript line with a speaker via dialogue-context inference —
    the diarization that actually works on single-mic recordings."""
    if not cast_path.exists():
        sys.exit(f"no cast file at {cast_path} — list the table as 'Name: how to recognize them'")
    cast = "\n".join(l for l in cast_path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))
    rows = parse_stamped(transcript)
    if not rows:
        sys.exit(f"no stamped lines found in {transcript}")
    out = transcript.with_name(transcript.stem + "-attributed.md")
    labels: list[str] = []
    if out.exists():  # resume a partial run
        done = parse_stamped(out)
        if [t for t, _ in done] == [t for t, _ in rows[: len(done)]]:
            labels = [txt.split(":**", 1)[0].lstrip("*") for _, txt in done]
            print(f"[attribute] resuming at line {len(labels)}", file=sys.stderr)
    for i in range(len(labels), len(rows), ATTRIBUTE_CHUNK):
        block = rows[i : i + ATTRIBUTE_CHUNK]
        prev = "\n".join(
            f"{stamp(t)} {lab}: {txt}"
            for (t, txt), lab in list(zip(rows, labels))[-12:]
        ) or "(session start)"
        lines = "\n".join(f"{j + 1}. {txt}" for j, (t, txt) in enumerate(block))
        prompt = ATTRIBUTE_PROMPT.format(cast=cast, prev=prev, n=len(block), lines=lines)
        got: list[str] = []
        for attempt in (1, 2, 3):
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, timeout=300, input=prompt.encode())
                got = [l.strip().lstrip("0123456789. ") for l in r.stdout.decode().strip().splitlines() if l.strip()]
                if r.returncode == 0 and got:
                    break
                got = []
            except subprocess.TimeoutExpired:
                print(f"\n[attribute] chunk {i // ATTRIBUTE_CHUNK}: attempt {attempt} timed out", file=sys.stderr)
            time.sleep(10)
        if len(got) != len(block):
            print(f"\n[attribute] chunk {i // ATTRIBUTE_CHUNK}: got {len(got)} labels for {len(block)} lines — padding", file=sys.stderr)
        got = (got + ["?"] * len(block))[: len(block)]
        labels.extend(got)
        out.write_text("\n".join(
            f"{stamp(t)} **{lab}:** {txt}" for (t, txt), lab in zip(rows, labels)
        ) + "\n")
        print(f"\x1b[2K\r[attribute] {len(labels)}/{len(rows)} lines", end="", file=sys.stderr, flush=True)
    n = len({l for l in labels if l != "?"})
    unsure = labels.count("?")
    print(f"\x1b[2K\rAttributed transcript: {out}  ({n} speakers, {unsure} unsure)", file=sys.stderr)


class NoteTaker:
    """Periodic 'twitter post' summaries of the transcript. Each note is produced
    by piping a prompt into `cmd` (stdin → stdout) on a background thread, so the
    audio loop never blocks; completed notes are drained by the main loop."""

    def __init__(self, path: Path, cmd: str, interval: float):
        self.path = path
        self.cmd = cmd
        self.interval = interval
        self.last_t = 0.0
        self.mark = 0  # sentence index already summarized
        self.prev_note = "(none yet)"
        self.results: queue.Queue[tuple[float, str]] = queue.Queue()
        self.threads: list[threading.Thread] = []

    def maybe_fire(self, stream_t: float, sentences, starts: list[float]) -> None:
        if stream_t - self.last_t < self.interval or len(sentences) <= self.mark:
            return
        # Stamp the note where its content BEGAN, not when it was summarized —
        # a fire-time stamp reads one whole interval late against the transcript.
        t0 = starts[self.mark] if self.mark < len(starts) else stream_t
        text = "\n".join(s.text.strip() for s in sentences[self.mark :])
        self.last_t, self.mark = stream_t, len(sentences)
        t = threading.Thread(target=self._summarize, args=(t0, text), daemon=True)
        self.threads.append(t)
        t.start()

    def _summarize(self, t: float, text: str) -> None:
        err = ""
        for attempt in (1, 2):
            try:
                out = subprocess.run(
                    self.cmd, shell=True, capture_output=True, timeout=120,
                    input=NOTE_PROMPT.format(prev=self.prev_note, text=text).encode(),
                )
                note = out.stdout.decode().strip()
                if out.returncode == 0 and note:
                    note = note.splitlines()[-1].strip()
                    if note != "SKIP":
                        self.prev_note = note
                        self.results.put((t, note))
                    return
                err = out.stderr.decode().strip()[-200:] or f"exit {out.returncode}, empty output"
            except Exception as e:
                err = str(e)
            if attempt == 1:
                time.sleep(10)
        print(f"\n[notes] {stamp(t)} summarizer failed twice, note lost: {err}", file=sys.stderr)

    def drain(self) -> None:
        while not self.results.empty():
            t, note = self.results.get()
            with open(self.path, "a") as f:
                f.write(f"- {stamp(t)} {note}\n")
            print(f"\x1b[2K\r🐦 {stamp(t)} \x1b[3m{note}\x1b[0m")

    def finish(self, stream_t: float, sentences, starts: list[float]) -> None:
        self.last_t = -self.interval  # force one closing note
        self.maybe_fire(stream_t, sentences, starts)
        for t in self.threads:
            t.join(timeout=280)
        self.drain()


class WavWriter:
    """Incremental 16-bit mono WAV writer. The RIFF/data sizes in the header are
    re-patched after every write, so the file is playable even after a crash."""

    def __init__(self, path: Path, rate: int):
        self.f = open(path, "wb")
        self.rate = rate
        self.frames = 0
        self._patch_header()

    def _patch_header(self) -> None:
        data = self.frames * 2
        self.f.seek(0)
        self.f.write(b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE")
        self.f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, self.rate, self.rate * 2, 2, 16))
        self.f.write(b"data" + struct.pack("<I", data))

    def write(self, samples: np.ndarray) -> None:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        self.f.seek(0, 2)
        self.f.write(pcm.tobytes())
        self.frames += len(pcm)
        self._patch_header()
        self.f.flush()

    def close(self) -> None:
        self._patch_header()
        self.f.close()


def polish(src: Path, model, rate: int, speakers: SpeakerLog | None) -> list[tuple[float, int | None, str]]:
    """Offline full-context re-transcription of a session recording. Unlike the
    streaming decode, sentence timestamps here are absolute and trustworthy."""
    out = src.with_name(src.stem + "-polished.md")

    def progress(cur, total):
        print(f"\x1b[2K\rPolishing … {min(100, int(cur / total * 100))}%", end="", file=sys.stderr, flush=True)

    result = model.transcribe(src, chunk_duration=120.0, chunk_callback=progress)
    rows = []
    samples = np.array(load_audio(src, rate, mx.float32)) if speakers else None
    for s in result.sentences:
        text = s.text.strip()
        if not text:
            continue
        label = None
        if speakers is not None:
            label = speakers.label(samples[int(s.start * rate) : int(s.end * rate)])
        rows.append((s.start, label, text))
    out.write_text(render(rows, ""))
    print(f"\x1b[2K\rPolished transcript: {out}  ({len(rows)} lines)", file=sys.stderr)
    return rows


def offline_notes(rows: list[tuple[float, int | None, str]], path: Path, cmd: str, interval: float) -> None:
    """Key-notes for an already-transcribed session (--polish --notes): window the
    transcript by interval seconds and run the same summarizer as live --notes,
    synchronously — offline there's no audio loop to keep unblocked, and serial
    calls preserve the previous-post chaining."""
    if path.exists():
        print(f"{path.name} already exists — delete it to regenerate", file=sys.stderr)
        return
    windows: list[tuple[float, list[str]]] = []
    cur: list[str] = []
    t0 = 0.0
    for t, _label, text in rows:
        if not cur:
            t0 = t
        elif t - t0 >= interval:
            windows.append((t0, cur))
            cur, t0 = [], t
        cur.append(text)
    if cur:
        windows.append((t0, cur))

    nt = NoteTaker(path, cmd, interval)
    for i, (t, texts) in enumerate(windows):
        print(f"\x1b[2K\rSummarizing window {i + 1}/{len(windows)} …", end="", file=sys.stderr, flush=True)
        nt._summarize(t, "\n".join(texts))
        nt.drain()
    print(f"\x1b[2K\rKey notes: {path}", file=sys.stderr)


def mic_chunks(rate: int, device):
    import sounddevice as sd

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=int(rate * CHUNK_SECONDS),
        callback=on_audio,
    ):
        while True:
            yield audio_q.get()


def wav_chunks(path: Path, rate: int):
    import wave

    with wave.open(str(path)) as w:
        assert w.getframerate() == rate, f"need {rate} Hz wav, got {w.getframerate()}"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    step = int(rate * CHUNK_SECONDS)
    for i in range(0, len(audio), step):
        yield audio[i : i + step]


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime local transcription (Parakeet MLX).")
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--out", type=Path, default=Path("sessions"))
    parser.add_argument("--device", default=None, help="Input device name or index (see `python -m sounddevice`)")
    parser.add_argument("--speakers", action="store_true", help="Experimental: label sentences by voice (Speaker 1/2/…)")
    parser.add_argument("--speaker-threshold", type=float, default=0.45, help="Same-speaker similarity floor (lower = fewer, broader speakers)")
    parser.add_argument("--max-speakers", type=int, default=8, help="Never mint more than N speakers; extras snap to the nearest voice")
    parser.add_argument("--notes", action="store_true", help="Periodic one-line 'what's happening' posts to key-notes-<session>.md (uses the claude CLI by default; combine with --polish to distill an existing recording)")
    parser.add_argument("--notes-interval", type=float, default=180, metavar="SEC", help="Seconds between notes (default 180)")
    parser.add_argument("--notes-cmd", default="claude -p --model haiku", help="Shell command that reads a prompt on stdin and prints the post (default: %(default)s)")
    parser.add_argument("--save-audio", action="store_true", help="Also record the session to a wav next to the transcript (~110 MB/hour), enabling --polish later")
    parser.add_argument("--polish", type=Path, default=None, metavar="AUDIO", help="Re-transcribe a session recording offline with full context (better accuracy) and exit")
    parser.add_argument("--enrich", type=Path, default=None, metavar="KEYNOTES", help="Expand a key-notes file into rich paragraphs (key-notes-enriched-*.md) using its session transcript, and exit")
    parser.add_argument("--attribute", type=Path, default=None, metavar="TRANSCRIPT", help="Label speakers by dialogue context (LLM pass; works where --speakers can't) and exit")
    parser.add_argument("--cast", type=Path, default=Path("avatars/cast.md"), help="Cast list for --attribute: 'Name: how to recognize them' per line")
    parser.add_argument("--imagine", type=Path, default=None, metavar="KEYNOTES", help="Generate a gallery image per key-note (Gemini API, needs GEMINI_API_KEY) and exit")
    parser.add_argument("--avatars", type=Path, default=Path("avatars"), help="Directory of character reference images for --imagine (default avatars/)")
    parser.add_argument("--image-model", default="gemini-2.5-flash-image", help="Gemini image model for --imagine (default: %(default)s)")
    parser.add_argument("--post-cmd", default=None, help="Optional hook run per generated image: gets '<path>\\n<caption>' on stdin (future: post to an API)")
    parser.add_argument("--wav", type=Path, default=None, help=argparse.SUPPRESS)  # testing: 16k mono wav instead of mic
    args = parser.parse_args()

    if args.enrich:  # no ASR model needed
        enrich(args.enrich, args.notes_cmd, args.notes_interval * 1.5)
        return
    if args.attribute:  # no ASR model needed
        attribute(args.attribute, args.cast, args.notes_cmd)
        return
    if args.imagine:  # no ASR model needed
        imagine(args.imagine, args.avatars, args.image_model, args.post_cmd)
        return

    if args.notes and shutil.which(args.notes_cmd.split()[0]) is None:
        sys.exit(f"--notes needs `{args.notes_cmd.split()[0]}` on PATH (or pass --notes-cmd)")

    print(f"Loading {args.model} …", file=sys.stderr)
    model = from_pretrained(args.model)
    rate = model.preprocessor_config.sample_rate
    speakers = SpeakerLog(
        ensure_embed_model(), rate, args.speaker_threshold, args.max_speakers
    ) if args.speakers else None

    if args.polish:
        rows = polish(args.polish, model, rate, speakers)
        if args.notes:
            notes_path = args.polish.parent / f"key-notes-{args.polish.stem.removeprefix('session-')}.md"
            offline_notes(rows, notes_path, args.notes_cmd, args.notes_interval)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    session = f"session-{datetime.datetime.now():%Y-%m-%d-%H%M}"
    outfile = args.out / f"{session}.md"
    recorder = WavWriter(args.out / f"{session}.wav", rate) if args.save_audio else None
    notes = None
    if args.notes:
        notes = NoteTaker(args.out / f"key-notes-{session.removeprefix('session-')}.md", args.notes_cmd, args.notes_interval)

    # Parakeet's streaming token timestamps are window-relative — once the cache
    # drops old frames they no longer reflect stream time (every stamp reads
    # 0:00:00 in a long session). So thoth keeps its own clock: samples fed ÷
    # rate, and stamps each sentence when it first appears in the decode.
    stream_t = 0.0
    starts: list[float] = []  # our timestamp per sentence index
    labels: list[int | None] = []  # speaker per printed sentence
    printed = 0
    buf = np.zeros(0, dtype=np.float32)  # retained audio for unlabeled speech
    buf_start = 0  # absolute sample index of buf[0]

    def commit(i: int, text: str, hi_time: float) -> None:
        """Print sentence i once, labeling its [starts[i], hi_time] audio if enabled."""
        nonlocal buf, buf_start
        label = None
        if speakers is not None:
            lo = max(0, int(starts[i] * rate) - buf_start)
            hi = max(0, int(hi_time * rate) - buf_start)
            label = speakers.label(buf[lo:hi])
            buf = buf[hi:]
            buf_start += hi
        labels.append(label)
        tprint(starts[i], label, text)

    chunks = wav_chunks(args.wav, rate) if args.wav else mic_chunks(rate, args.device)
    with model.transcribe_stream(context_size=(256, 256), depth=1) as streamer:
        print(f"Recording. Transcript → {outfile}  (Ctrl-C to stop)", file=sys.stderr)
        try:
            for chunk in chunks:
                stream_t += len(chunk) / rate
                if recorder is not None:
                    recorder.write(chunk)
                if speakers is not None:
                    buf = np.concatenate([buf, chunk])
                streamer.add_audio(mx.array(chunk))
                sentences = streamer.result.sentences
                while len(starts) < len(sentences):  # new sentence began ~now
                    starts.append(max(0.0, stream_t - CHUNK_SECONDS))
                printed = min(printed, max(0, len(sentences) - 1))
                if notes is not None:
                    notes.maybe_fire(stream_t, sentences, starts)
                    notes.drain()

                # Terminal: print all but the still-mutating last sentence, once.
                while printed < len(sentences) - 1:
                    commit(printed, sentences[printed].text.strip(), starts[printed + 1])
                    printed += 1
                if sentences:
                    width = shutil.get_terminal_size().columns - 4
                    live = sentences[-1].text.strip()
                    print(f"\x1b[2K\r… {live[-width:]}", end="", flush=True)

                # Disk: rewrite the whole file every chunk from the current decode —
                # crash-safe, and late refinements self-correct on disk.
                rows = [
                    (starts[i], labels[i] if i < len(labels) else None, s.text.strip())
                    for i, s in enumerate(sentences[:-1])
                ]
                outfile.write_text(render(rows, sentences[-1].text.strip() if sentences else ""))
        except KeyboardInterrupt:
            pass

        # Final flush: the draft region is now as final as it will ever be.
        sentences = streamer.result.sentences
        while printed < len(sentences):
            hi_time = starts[printed + 1] if printed + 1 < len(starts) else stream_t
            commit(printed, sentences[printed].text.strip(), hi_time)
            printed += 1
        if sentences:
            rows = [
                (starts[i], labels[i] if i < len(labels) else None, s.text.strip())
                for i, s in enumerate(sentences)
            ]
            outfile.write_text(render(rows, ""))
        tail = ""
        if speakers is not None:
            n = len({label for label in labels if label is not None})
            tail = f"  ({n} speaker{'s' if n != 1 else ''})"
        if notes is not None and sentences:
            print("\x1b[2K\rWaiting for final note …", end="", file=sys.stderr, flush=True)
            notes.finish(stream_t, sentences, starts)
            print(f"\x1b[2K\rKey notes: {notes.path}", file=sys.stderr)
        print(f"\x1b[2K\rSession saved: {outfile}{tail}", file=sys.stderr)
        if recorder is not None:
            recorder.close()
            print(f"Audio saved: {recorder.f.name} — refine with: uv run thoth.py --polish {recorder.f.name}", file=sys.stderr)


if __name__ == "__main__":
    main()

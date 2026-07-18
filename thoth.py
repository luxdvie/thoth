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
import datetime
import queue
import shutil
import sys
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.alignment import tokens_to_sentences

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


def render(committed: list[tuple[float, int | None, str]], live: str) -> str:
    lines = [
        f"{stamp(start)} **Speaker {label + 1}:** {text}" if label is not None else f"{stamp(start)} {text}"
        for start, label, text in committed
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
    parser.add_argument("--wav", type=Path, default=None, help=argparse.SUPPRESS)  # testing: 16k mono wav instead of mic
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"session-{datetime.datetime.now():%Y-%m-%d-%H%M}.md"

    print(f"Loading {args.model} …", file=sys.stderr)
    model = from_pretrained(args.model)
    rate = model.preprocessor_config.sample_rate
    speakers = SpeakerLog(
        ensure_embed_model(), rate, args.speaker_threshold, args.max_speakers
    ) if args.speakers else None

    committed: list[tuple[float, int | None, str]] = []  # (start, speaker, text) — never mutated
    buf = np.zeros(0, dtype=np.float32)  # retained audio for not-yet-committed speech
    buf_start = 0  # absolute sample index of buf[0]

    def commit(sentence) -> None:
        """Label a stable sentence from retained audio, print it, trim the buffer."""
        nonlocal buf, buf_start
        lo = max(0, int(sentence.start * rate) - buf_start)
        hi = max(0, int(sentence.end * rate) - buf_start)
        label = speakers.label(buf[lo:hi]) if speakers else None
        buf = buf[hi:]
        buf_start += hi
        committed.append((sentence.start, label, sentence.text.strip()))
        tprint(*committed[-1])

    chunks = wav_chunks(args.wav, rate) if args.wav else mic_chunks(rate, args.device)
    with model.transcribe_stream(context_size=(256, 256), depth=1) as streamer:
        print(f"Recording. Transcript → {outfile}  (Ctrl-C to stop)", file=sys.stderr)
        try:
            for chunk in chunks:
                buf = np.concatenate([buf, chunk])
                streamer.add_audio(mx.array(chunk))

                # Only sentences built purely from finalized tokens are immutable;
                # the last of them may still grow, so hold it back too.
                stable = tokens_to_sentences(
                    streamer.finalized_tokens, streamer.decoding_config.sentence
                )
                for sentence in stable[len(committed) : -1]:
                    commit(sentence)

                # Live line: everything after the committed prefix, draft included.
                done = stable[len(committed) - 1].end if committed else 0.0
                live = " ".join(
                    s.text.strip() for s in streamer.result.sentences if s.end > done
                ).strip()
                width = shutil.get_terminal_size().columns - 4
                print(f"\x1b[2K\r… {live[-width:]}", end="", flush=True)

                # Disk: rewrite the whole file every chunk — crash-safe.
                outfile.write_text(render(committed, live))
        except KeyboardInterrupt:
            pass

        # Final flush: everything left, including the draft region, is now final.
        done = committed[-1][0] if committed else -1.0
        for sentence in streamer.result.sentences:
            if sentence.start > done and sentence.text.strip():
                commit(sentence)
        if committed:
            outfile.write_text(render(committed, ""))
        tail = ""
        if speakers is not None:
            n = len({label for _, label, _ in committed})
            tail = f"  ({n} speaker{'s' if n != 1 else ''})"
        print(f"\x1b[2K\rSession saved: {outfile}{tail}", file=sys.stderr)


if __name__ == "__main__":
    main()

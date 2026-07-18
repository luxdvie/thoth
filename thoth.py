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
import sys
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained

CHUNK_SECONDS = 2.0
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
    """Online speaker identification: embed each finalized sentence, match against
    running centroids by cosine similarity, mint a new speaker below threshold."""

    def __init__(self, model_path: Path, rate: int, threshold: float):
        import sherpa_onnx

        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2)
        )
        self.rate = rate
        self.threshold = threshold
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.last = 0

    def label(self, samples: np.ndarray) -> int:
        if len(samples) < int(0.3 * self.rate):
            return self.last  # too short to embed reliably; assume same voice
        stream = self.extractor.create_stream()
        stream.accept_waveform(self.rate, samples)
        stream.input_finished()
        emb = np.array(self.extractor.compute(stream))
        emb /= np.linalg.norm(emb)

        if self.centroids:
            sims = [float(np.dot(emb, c)) for c in self.centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
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


def render(sentences, labels) -> str:
    lines = []
    for i, s in enumerate(sentences):
        if i < len(labels):
            lines.append(f"{stamp(s.start)} **Speaker {labels[i] + 1}:** {s.text.strip()}")
        else:
            lines.append(f"{stamp(s.start)} … {s.text.strip()}")
    return "\n".join(lines) + "\n"


def tprint(sentence, label: int) -> None:
    color = SPEAKER_COLORS[label % len(SPEAKER_COLORS)]
    print(
        f"\x1b[2K\r{stamp(sentence.start)} {color}Speaker {label + 1}:{RESET} {sentence.text.strip()}"
    )


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
    parser.add_argument("--no-speakers", action="store_true", help="Disable speaker labeling")
    parser.add_argument("--speaker-threshold", type=float, default=0.55, help="Cosine similarity cutoff for 'same speaker' (lower = fewer, broader speakers)")
    parser.add_argument("--wav", type=Path, default=None, help=argparse.SUPPRESS)  # testing: 16k mono wav instead of mic
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"session-{datetime.datetime.now():%Y-%m-%d-%H%M}.md"

    print(f"Loading {args.model} …", file=sys.stderr)
    model = from_pretrained(args.model)
    rate = model.preprocessor_config.sample_rate
    speakers = None if args.no_speakers else SpeakerLog(ensure_embed_model(), rate, args.speaker_threshold)

    labels: list[int] = []  # labels[i] belongs to sentences[i], assigned on finalize
    buf = np.zeros(0, dtype=np.float32)  # retained audio for pending sentences
    buf_start = 0  # absolute sample index of buf[0]

    def commit(sentence, buf, buf_start):
        """Label a finalized sentence from retained audio; trim the buffer past it."""
        if speakers is None:
            labels.append(0)
            return buf, buf_start
        lo = max(0, int(sentence.start * rate) - buf_start)
        hi = max(0, int(sentence.end * rate) - buf_start)
        labels.append(speakers.label(buf[lo:hi]))
        return buf[hi:], buf_start + hi

    chunks = wav_chunks(args.wav, rate) if args.wav else mic_chunks(rate, args.device)
    with model.transcribe_stream(context_size=(256, 256), depth=1) as streamer:
        print(f"Recording. Transcript → {outfile}  (Ctrl-C to stop)", file=sys.stderr)
        try:
            for chunk in chunks:
                buf = np.concatenate([buf, chunk])
                streamer.add_audio(mx.array(chunk))
                sentences = streamer.result.sentences
                if not sentences:
                    continue

                # Commit all but the still-mutating last sentence.
                while len(labels) < len(sentences) - 1:
                    buf, buf_start = commit(sentences[len(labels)], buf, buf_start)
                    tprint(sentences[len(labels) - 1], labels[-1])
                live = sentences[-1].text.strip()[-100:]
                print(f"\x1b[2K\r… {live}", end="", flush=True)

                # Disk: rewrite the whole file every chunk — crash-safe.
                outfile.write_text(render(sentences, labels))
        except KeyboardInterrupt:
            pass

        sentences = streamer.result.sentences
        while len(labels) < len(sentences):  # label the tail, including the draft
            buf, buf_start = commit(sentences[len(labels)], buf, buf_start)
            tprint(sentences[len(labels) - 1], labels[-1])
        if sentences:
            outfile.write_text(render(sentences, labels))
        n_speakers = len(set(labels)) if labels else 0
        print(f"\x1b[2K\rSession saved: {outfile}  ({n_speakers} speaker{'s' if n_speakers != 1 else ''})", file=sys.stderr)


if __name__ == "__main__":
    main()

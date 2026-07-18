# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "parakeet-mlx>=0.3",
#     "sounddevice>=0.5",
#     "numpy",
#     "numba>=0.60",
# ]
# ///
"""thoth — realtime mic transcription to terminal + disk.

Usage:
    uv run thoth.py                # new session file in ./sessions/
    uv run thoth.py --out ~/dnd    # choose output dir
    Ctrl-C to stop; transcript is flushed continuously, so a crash loses nothing.
"""

import argparse
import datetime
import queue
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import sounddevice as sd
from parakeet_mlx import from_pretrained

CHUNK_SECONDS = 2.0


def stamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"[{h}:{m:02d}:{s:02d}]"


def render(sentences) -> str:
    return "\n".join(f"{stamp(s.start)} {s.text.strip()}" for s in sentences) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime local transcription (Parakeet MLX).")
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--out", type=Path, default=Path("sessions"))
    parser.add_argument("--device", default=None, help="Input device name or index (see `python -m sounddevice`)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"session-{datetime.datetime.now():%Y-%m-%d-%H%M}.md"

    print(f"Loading {args.model} …", file=sys.stderr)
    model = from_pretrained(args.model)
    rate = model.preprocessor_config.sample_rate

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    printed = 0  # sentences already committed to the terminal
    with model.transcribe_stream(context_size=(256, 256), depth=1) as streamer:
        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=args.device,
            blocksize=int(rate * CHUNK_SECONDS),
            callback=on_audio,
        ):
            print(f"Recording. Transcript → {outfile}  (Ctrl-C to stop)", file=sys.stderr)
            try:
                while True:
                    streamer.add_audio(mx.array(audio_q.get()))
                    sentences = streamer.result.sentences
                    if not sentences:
                        continue

                    # Terminal: commit all but the still-mutating last sentence.
                    while printed < len(sentences) - 1:
                        print(f"\x1b[2K\r{stamp(sentences[printed].start)} {sentences[printed].text.strip()}")
                        printed += 1
                    live = sentences[-1].text.strip()[-100:]
                    print(f"\x1b[2K\r… {live}", end="", flush=True)

                    # Disk: rewrite the whole file every chunk — crash-safe.
                    outfile.write_text(render(sentences))
            except KeyboardInterrupt:
                pass

        sentences = streamer.result.sentences
        if sentences:
            outfile.write_text(render(sentences))
        print(f"\x1b[2K\rSession saved: {outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()

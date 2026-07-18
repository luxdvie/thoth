# thoth — agent notes

Realtime local speech-to-text for tabletop sessions. One file of product code: `thoth.py`. Keep it that way unless a feature genuinely can't live there.

## Architecture

- **PEP 723 single-file script.** Dependencies live in the inline `# /// script` block at the top of `thoth.py`. There is no pyproject, no lockfile, no package. Run with `uv run thoth.py`.
- **Pipeline:** sounddevice `InputStream` (16 kHz mono float32, 2 s blocks) → queue → `parakeet_mlx` `StreamingParakeet.add_audio()` → `result.sentences` (each has `.text`, `.start`, `.end`).
- **Speaker layer:** raw audio is retained in `buf` (trimmed as sentences finalize). When a sentence finalizes, its `[start, end]` slice is embedded via sherpa-onnx TitaNet (auto-downloaded to `~/.cache/thoth/`, 40 MB, ungated) and matched against running speaker centroids by cosine similarity (`SpeakerLog`); below `--speaker-threshold` mints a new speaker. Sub-0.3 s slices inherit the previous speaker. This is speaker ID by sentence clustering, not true diarization — no overlap handling.
- **Output contract:** finalized sentences print once with `[H:MM:SS]` stamps and color-coded `Speaker N:` prefixes; the last (still-mutating) sentence renders on a live line via `\x1b[2K\r` with no label. The whole transcript file is rewritten every chunk — that's deliberate crash-safety, not inefficiency; don't "optimize" it into append-only.

## Constraints learned the hard way

- Keep the `numba>=0.60` pin. Without it uv's resolver picks numba 0.53 (via librosa), which cannot build on modern Python.
- `add_audio` wants a 1-D `mx.array` at `model.preprocessor_config.sample_rate` (16 kHz). Resample anything else before feeding it.
- Apple Silicon only (MLX). Don't add cross-platform shims speculatively.

## Testing without a microphone

Synthesize speech and feed it through the real pipeline:

```sh
say -o test.aiff "Roll for initiative."
afconvert -f WAVE -d LEI16@16000 -c 1 test.aiff test.wav
```

Then run the real pipeline against it with the hidden test flag:

```sh
uv run thoth.py --wav test.wav --out /tmp/testout
```

For speaker-label testing, synthesize with two voices (`say -v Samantha …`, `say -v Daniel …`), concatenate with ~0.6 s silence gaps, and assert alternating labels. TTS voices separate at cosine ~0.2 vs ~0.9 same-voice, so threshold regressions show up clearly. Model weights cache under `~/.cache/huggingface` (~600 MB) and `~/.cache/thoth/` (40 MB).

## Known gaps (intentional, v2 territory)

- Speaker ID, not diarization: no overlapping-speech separation; similar voices can merge.
- English-tuned default model.

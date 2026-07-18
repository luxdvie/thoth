# thoth — agent notes

Realtime local speech-to-text for tabletop sessions. One file of product code: `thoth.py`. Keep it that way unless a feature genuinely can't live there.

## Architecture

- **PEP 723 single-file script.** Dependencies live in the inline `# /// script` block at the top of `thoth.py`. There is no pyproject, no lockfile, no package. Run with `uv run thoth.py`.
- **Pipeline:** sounddevice `InputStream` (16 kHz mono float32, 2 s blocks) → queue → `parakeet_mlx` `StreamingParakeet.add_audio()` → `result.sentences` (each has `.text`, `.start`, `.end`).
- **Timestamps (load-bearing):** Parakeet's streaming token times are *window-relative* — once the KV cache drops old frames they stop tracking stream time, and every stamp reads `0:00:00` (real 14-min session proved it; short tests hide it, so any timestamp change must be tested on 5+ minutes of audio). thoth therefore never reads `token.start`: it keeps its own clock (`stream_t` = samples fed ÷ rate) and stamps each sentence when it first appears in the decode (~±2 s accurate). Speaker audio slicing uses the same clock.
- **Commit semantics (chosen tradeoff):** all but the last of `result.sentences` print promptly, one line per sentence; the draft region can occasionally revise an already-printed sentence, and the disk file — rewritten from the *current* decode every chunk — self-corrects, so the file is the source of truth, the terminal is a view. The alternative (committing only finalized tokens, f7f6829) never rewrites but trails realtime by ~15–20 s and pools speech into blobs — tried at a real table, rejected (e5e108d follow-up). Don't reintroduce it without solving the lag.
- **Speaker layer (opt-in via `--speakers`, experimental — merges everyone into Speaker 1 on far-field mics; needs tuning):** raw audio is retained in `buf` (trimmed on commit). Each committed sentence's `[start, end]` slice is embedded via sherpa-onnx TitaNet (auto-downloaded to `~/.cache/thoth/`, 40 MB, ungated) and matched against running centroids (`SpeakerLog`). Two-tier thresholds: `--speaker-threshold` (default 0.45) decides assignment; only matches ≥ threshold + 0.15 update the centroid, so noisy far-field segments can't drift a voiceprint. `--max-speakers` caps minting; sub-0.5 s slices inherit the previous speaker. Speaker ID by sentence clustering, not true diarization — no overlap handling.
- **Output contract:** each sentence prints to the terminal once with an `[H:MM:SS]` stamp (color-coded `Speaker N:` prefix in `--speakers` mode); the in-flight last sentence renders on one live line via `\x1b[2K\r`, truncated to terminal width. The whole transcript file is rewritten from the current decode every chunk — deliberate crash-safety *and* self-correction; don't "optimize" it into append-only.

- **Two-pass:** `--save-audio` records the mic to `session-<date>.wav` via `WavWriter`, which re-patches the RIFF header sizes after every write so a crash still leaves a playable file (stdlib `wave` only fixes the header on close). `--polish <wav>` re-transcribes offline via `model.transcribe(chunk_duration=120)` — full context beats the streaming decode, and offline sentence timestamps ARE absolute/trustworthy (unlike streaming), so `--speakers` slicing is exact there.

- **Notes layer (`--notes`):** `NoteTaker` fires every `--notes-interval` seconds of *stream time* (so `--wav` tests behave deterministically), summarizing sentences since the last fire via `--notes-cmd` (prompt on stdin → post on stdout; default `claude -p --model haiku`). Summaries run on daemon threads and are drained by the main loop — never call the summarizer synchronously in the audio path. The prompt asks for `SKIP` on no-news; notes append to `key-notes-<ts>.md` and never rewrite.

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

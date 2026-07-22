# thoth — agent notes

Realtime local speech-to-text for tabletop (D&D) sessions, plus a storytelling pipeline on top: headlines → chronicle → illustrated scenes → (roadmap) session films.

## Repo map

```
thoth.py         core CLI: live transcribe / --polish / --enrich / --imagine (PEP 723, one file — keep it that way)
studio.py        stdlib-only HTTP server (:8511) for the curation UI; deliberately zero-dep so it starts instantly
studio.html      the UI (single page, scriptorium-noir; Cinzel/Cormorant/JetBrains Mono via Google Fonts)
aggregate.py     stitches multi-part sessions into campaign-* files (md concat with part headers, wav concat)
sessions/        (gitignored) transcripts, recordings, key-notes, chronicles — per-session timestamp ids
avatars/         (gitignored) party reference portraits; filename stem = character name used in prompts
generated-images/(gitignored) every studio/imagine take + JSON sidecar (prompt, avatars, headline)
gallery/         (gitignored) PROMOTED scenes only + captions.json + gallery.md — the curated record
```

`thoth.py` stays one file. The studio is a separate surface (server + page) on purpose; it duplicates the small Gemini REST call from `thoth.py` rather than importing it, because importing `thoth` would drag in mlx/parakeet and add seconds to server startup. If you change one Gemini call, change both.

## Data flow

```
mic ─ thoth.py ──► session-<ts>.md  + session-<ts>.wav + key-notes-<ts>.md      (live)
      --polish <wav>  ──► session-<ts>-polished.md                              (accuracy pass)
      --enrich <keynotes> ──► key-notes-enriched-<ts>.md                        (chronicle)
      --imagine <keynotes> / studio.py ──► generated-images/<ts>/ ──promote──► gallery/<ts>/
```

## Architecture

- **PEP 723 single-file script.** Dependencies live in the inline `# /// script` block at the top of `thoth.py`. There is no pyproject, no lockfile, no package. Run with `uv run thoth.py`.
- **Pipeline:** sounddevice `InputStream` (16 kHz mono float32, 2 s blocks) → queue → `parakeet_mlx` `StreamingParakeet.add_audio()` → `result.sentences` (each has `.text`, `.start`, `.end`).
- **Timestamps (load-bearing):** Parakeet's streaming token times are *window-relative* — once the KV cache drops old frames they stop tracking stream time, and every stamp reads `0:00:00` (real 14-min session proved it; short tests hide it, so any timestamp change must be tested on 5+ minutes of audio). thoth therefore never reads `token.start`: it keeps its own clock (`stream_t` = samples fed ÷ rate) and stamps each sentence when it first appears in the decode (~±2 s accurate). Speaker audio slicing uses the same clock.
- **Commit semantics (chosen tradeoff):** all but the last of `result.sentences` print promptly, one line per sentence; the draft region can occasionally revise an already-printed sentence, and the disk file — rewritten from the *current* decode every chunk — self-corrects, so the file is the source of truth, the terminal is a view. The alternative (committing only finalized tokens, f7f6829) never rewrites but trails realtime by ~15–20 s and pools speech into blobs — tried at a real table, rejected (e5e108d follow-up). Don't reintroduce it without solving the lag.
- **Speaker layer (opt-in via `--speakers`, experimental — merges everyone into Speaker 1 on far-field mics; needs tuning):** raw audio is retained in `buf` (trimmed on commit). Each committed sentence's `[start, end]` slice is embedded via sherpa-onnx TitaNet (auto-downloaded to `~/.cache/thoth/`, 40 MB, ungated) and matched against running centroids (`SpeakerLog`). Two-tier thresholds: `--speaker-threshold` (default 0.45) decides assignment; only matches ≥ threshold + 0.15 update the centroid, so noisy far-field segments can't drift a voiceprint. `--max-speakers` caps minting; sub-0.5 s slices inherit the previous speaker. Speaker ID by sentence clustering, not true diarization — no overlap handling.
- **Output contract:** each sentence prints to the terminal once with an `[H:MM:SS]` stamp (color-coded `Speaker N:` prefix in `--speakers` mode); the in-flight last sentence renders on one live line via `\x1b[2K\r`, truncated to terminal width. The whole transcript file is rewritten from the current decode every chunk — deliberate crash-safety *and* self-correction; don't "optimize" it into append-only.

- **Two-pass:** `--save-audio` records the mic to `session-<date>.wav` via `WavWriter`, which re-patches the RIFF header sizes after every write so a crash still leaves a playable file (stdlib `wave` only fixes the header on close). `--polish <wav>` re-transcribes offline via `model.transcribe(chunk_duration=120)` — full context beats the streaming decode, and offline sentence timestamps ARE absolute/trustworthy (unlike streaming), so `--speakers` slicing is exact there.

- **Notes layer (`--notes`):** `NoteTaker` fires every `--notes-interval` seconds of *stream time* (so `--wav` tests behave deterministically), summarizing sentences since the last fire via `--notes-cmd` (prompt on stdin → post on stdout; default `claude -p --model haiku`). Summaries run on daemon threads and are drained by the main loop — never call the summarizer synchronously in the audio path. The prompt asks for `SKIP` on no-news; notes append to `key-notes-<ts>.md` and never rewrite. Notes are stamped at the *start* of the window they summarize (fire-time stamps read one interval late against the transcript — real-session bug). Summarizer failures retry once after 10 s and then log loudly; never drop a note silently (a 3.75 h real session lost ~20 min gaps to silent CLI failures). `--polish --notes` runs `offline_notes()` instead: same windows/prompt/chaining over the polished rows, but synchronous (no audio loop to protect) — it refuses to touch an existing key-notes file. Unit tests: `uv run tests/test_notes.py` (stubbed summarizer cmds, no ASR inference).

- **Glossary (`--glossary`, default `avatars/glossary.md`, Zach's design — issue #1):** free-form user-maintained file (campaign names, places, known mishearing→correct mappings) injected via `glossary_block()` into the NOTE/ENRICH/ATTRIBUTE prompts (`{glossary}` slot in each — keep the slot when editing prompts) so garbled proper nouns get silently corrected. Missing/empty file = empty clause, zero prompt overhead. ENRICH_PROMPT also carries Zach's conservative-attribution rule: name a character only when unambiguous, else "the party" — misattribution rewrites a player's moment; vagueness doesn't.

- **Imagine layer (`--imagine`):** one Gemini `generateContent` REST call per key-note (no SDK dep — stdlib urllib), avatar images inlined base64 as character references, `responseModalities: ["TEXT","IMAGE"]`. Gallery writes are resumable (existing files skipped) and `gallery.md` re-renders after every image. `--post-cmd` is the future social/API hook: `<path>\n<caption>` on stdin, same stdin→stdout contract as `--notes-cmd`. `avatars/` and `gallery/` are gitignored — personal content.

## Studio (studio.py + studio.html)

- Endpoints: `GET /api/state` (sessions, notes with enriched bodies, per-note generation history, avatars), `POST /api/generate` (`{sid, idx, stamp, headline, prompt, avatars[]}` → saves png+json sidecar to staging, returns url), `POST /api/elaborate` (`{headline, body, avatars[]}` → art-director prompt via the notes CLI, shown in the textarea for editing), `POST /api/promote` (`{…, file}` → copies staging png to `gallery/<sid>/NNN-<stamp>.png`, updates `captions.json`, rebuilds `gallery.md`). Static: `/avatars/`, `/generated/`, `/gallery/`.
- Testable headless: start the server, `curl /api/state`, POST a generate with a real key. A generate costs ~$0.04 — one is fine for verification, don't loop.
- Note identity = (session id, index within its key-notes file). Promotion detection = `NNN-` filename prefix in `gallery/<sid>/`. If key-notes files are ever edited/reordered, indices shift — don't edit them in place.
- When both `key-notes-<ts>.md` and `key-notes-enriched-<ts>.md` exist, headlines come from the plain file and bodies from the enriched one, joined on timestamp.
- **Cutscenes (design decision, deliberate):** scenes are per-note (a moment → keyframes); cutscenes are per-SPAN (an arc → 6 keyframes + one recap narration). The UI separates them as sidebar modes because cutscenes CONSUME promoted scenes — paint stills first, then score the sequence. Recap narration sources from the enriched key-notes in the span (the editorial layer), NOT raw/polished transcript — a 30-min span of transcript is thousands of noisy words. Promoted cutscene = `cutscenes/<sid>/<span>/{narration.wav, manifest.json}`; the manifest (keyframes, stamps, script, voice, duration) is the video-assembly contract.
- **Narration:** `POST /api/cutscene-script` ({entries: [{stamp, headline, body}], seconds} → word-budgeted arc recap via notes CLI), `POST /api/narrate` ({sid, span: "NNN-MMM", script, voice} → Gemini TTS `gemini-3.1-flash-tts-preview`, returns 24 kHz wav to `generated-audio/<sid>/cut-<span>-NN.wav` + sidecar with measured duration), `POST /api/promote-cutscene` → `cutscenes/<sid>/<span>/`. TTS returns raw s16le PCM — the wav header is built by hand; duration = pcm_bytes/2/24000. NARRATION_WPM=95 is *measured* (gravitas style runs 75–102 wpm, take-dependent — do not "fix" it back to 150); exact video fit is ffmpeg `atempo`'s job at assembly, not the TTS call's.

## Image prompt structure (learned from Austin's hand-tuned prompts — keep it)

Prompts are assembled server-side in `build_prompt()` as **Command / Setting(Context) / Scene** — separating persistent world-state from the per-note beat is the single biggest quality lever after model tier (evidence: generated-images/2026-07-18-1801/018-*, 020-* vs earlier takes). Rules:
- `avatars/party.md` + `avatars/npcs.md`: "Name: one visual line" per row; party lines always injected, NPC lines ONLY when the scene/context names them (unconditional injection makes NPCs photobomb — real failure, 001-01).
- Session Setting persists in `sessions/context-<sid>.txt`; `/api/stage-context` derives it from the polished transcript window around the note (staging facts only the transcript knows — bite wounds, bait on the rail).
- Aspect ratio via `generationConfig.imageConfig.aspectRatio` (default 16:9 — keyframe sets must share one AR). Continuity: previous promoted keyframe appended as the LAST image part + CONTINUITY_PROMPT clause.
- Sidecars record context/scene/prompt/aspect/continuity/model separately — never collapse them back into one prompt string.

## Constraints learned the hard way

- Keep the `numba>=0.60` pin. Without it uv's resolver picks numba 0.53 (via librosa), which cannot build on modern Python.
- `add_audio` wants a 1-D `mx.array` at `model.preprocessor_config.sample_rate` (16 kHz). Resample anything else before feeding it.
- Apple Silicon only (MLX). Don't add cross-platform shims speculatively.
- Austin's `GEMINI_API_KEY` is a **fish universal variable** — invisible to non-fish parent shells. Both Gemini call sites fall back to `fish -c 'echo -n $GEMINI_API_KEY'`; keep that fallback when touching key handling.
- Image quality lesson: flash-tier Nano Banana underwhelmed Austin vs ChatGPT; the fix was BOTH the Pro model (`gemini-3-pro-image-preview`, studio default, ~13¢/image — flash stays default for batch `--imagine`) AND an elaborate step (short headlines waste a good image model; detailed art-director prompts are half the quality). The choice of Gemini over gpt-image-1 was deliberate: better multi-reference character consistency, cheaper (~$0.04/image), no org verification. Override with `--image-model` / `THOTH_IMAGE_MODEL`.

## Diarization guidance (field results — don't relearn this)

Zach's session-1 test on a single far-field iPhone, pushing `--speakers` hard: raising `--max-speakers` (6/10/25) only fragments the same people across more labels, and over-split labels are LEAKY (one label held DM narration + several players) so fragments can't be remapped to people. Root cause: overlapping speech on a far mic is noise, not a voiceprint — no parameter and no better single mic fixes it. What works, in order of bang/buck: (1) `--attribute` — LLM speaker labeling from dialogue context over the polished transcript with `avatars/cast.md` (validated: 240 lines → ~5% unsure; filling in real player names + verbal tells improves it further); (2) one recording track per speaker — everyone's phone runs Voice Memos, clap to sync, transcribe per-track, merge by timestamp = diarization perfect by construction (roadmap: `--merge-tracks`); (3) calibration phase for acoustic --speakers remains a nice-to-have, not the ceiling-raiser.

## Roadmap (agreed with Austin, in order)

1. **Session films**: 6 promoted gallery scenes → ComfyUI keyframe template (https://comfy.org/workflows/templates-6-key-frames-920c6926e747/) per clip → `ffmpeg concat` of clips. Images are the hard currency; video comes after.
2. **Posting hooks**: `--post-cmd` / `--notes-cmd` are stdin→stdout shell contracts, currently noop — wire to a social API or stream overlay without touching core code.
3. **Diarization v2**: calibration phase (each player says a line at session start), stronger embedding model; `--speakers` is opt-in experimental until then.

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

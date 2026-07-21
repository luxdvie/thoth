# 𓅝 thoth

> *Scribe of the gods. Keeper of the record. Now taking notes at your D&D table.*

**Realtime, fully-local transcription for tabletop sessions.** Mic → live terminal text → timestamped markdown on disk. Runs [NVIDIA Parakeet](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) on your Mac's Neural Engine via [MLX](https://github.com/ml-explore/mlx).

🔒 **Transcription is fully local** — no cloud, no keys, your table talk never leaves the machine. The optional storytelling layers ([notes](#-live-session-feed), [chronicle](#-the-full-ritual), [illustration](#-scene-studio)) call out to LLM/image APIs of your choosing.

```
[0:00:01] The party enters the dungeon.
[0:00:03] Roll for initiative.
[0:00:05] I check the door for traps before anyone touches it.
[0:00:08] The goblin attacks the wizard with a rusty dagger.
… and I swear if you crit me again I'm flipping the tab
```

## ⚡ Quickstart

```sh
uv run thoth.py                          # bare: live transcript only
uv run thoth.py --save-audio --notes     # the full game-night loadout
```

That's it. First run pulls the model (~600 MB), then you're live:

- ✍️ Sentences print promptly, one per line, with `[H:MM:SS]` stamps
- 🔮 The in-flight sentence updates on a live ticker line as the model changes its mind — and the transcript file continuously self-corrects as the decode refines
- 🎭 `--speakers` *(experimental)* — voice-fingerprints each sentence (TitaNet embeddings, online clustering) and tags it `Speaker N`, color-coded. Works on clean audio; still being tuned for far-field party chaos
- 💾 Full transcript rewritten to `sessions/session-<date>.md` every ~2 seconds — **a crash loses nothing**
- 🛑 `Ctrl-C` ends the session

Four hours of table time becomes a searchable, timestamped campaign log. Feed it to your favorite LLM for session recaps, quote your rogue's exact words back at them, settle the "you never told us about the trapdoor" dispute with receipts.

## 🔧 Setup

Transcription needs **nothing** — no keys, no accounts, first run downloads the model and you're live. The optional storytelling layers have two prerequisites:

1. **🤖 The `claude` CLI, installed and logged in** — all text-LLM work (`--notes`, `--enrich`, `--attribute`, the studio's Elaborate/Derive/Draft-recap buttons) shells out to [Claude Code](https://claude.com/claude-code) as `claude -p`. Any stdin→stdout command can stand in via `--notes-cmd` (e.g. `ollama run llama3`).
2. **🎨 A Gemini API key** (a.k.a. Nano Banana) — image generation and TTS narration (`--imagine`, the studio's Conjure/Narrate). Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), attach billing and add a few bucks — the good image model (`gemini-3-pro-image-preview`) is paid-tier, ~13¢/image (regular Nano Banana ~4¢, TTS ~1¢/take). Then:

   ```sh
   set -Ux GEMINI_API_KEY "…"     # fish
   export GEMINI_API_KEY="…"      # bash/zsh — put it in your profile
   ```

## 🎛️ Options

| Flag | Does |
|------|------|
| `--out DIR` | Output directory (default `./sessions`) |
| `--device NAME` | Pick a mic — list with `uv run --with sounddevice python -m sounddevice` |
| `--model ID` | Any parakeet-mlx-compatible Hugging Face model |
| `--notes` | 🐦 Live-post the session: a one-liner every few minutes to `key-notes-<session>.md` |
| `--notes-interval SEC` | Seconds between posts (default `180`) |
| `--notes-cmd CMD` | Summarizer command, prompt on stdin → post on stdout (default `claude -p --model haiku`) |
| `--save-audio` | Also record the session to a `.wav` next to the transcript (~110 MB/hour) |
| `--polish AUDIO` | Re-transcribe a recording offline with full context — noticeably more accurate than the live pass. Add `--notes` to also distill a key-notes file from the recording |
| `--attribute TRANSCRIPT` | 🗣️ Label speakers by dialogue context (LLM pass over the transcript + `avatars/cast.md`) — the diarization that actually works on one mic |
| `--cast FILE` | Cast list for `--attribute` (default `avatars/cast.md`): `Name: how to recognize them` |
| `--enrich KEYNOTES` | Expand a key-notes file into a rich chronicle (`key-notes-enriched-*.md`), grounded in the transcript |
| `--imagine KEYNOTES` | 🎨 Generate a scene image per key-note into `gallery/<session>/`, with `avatars/` as character references (Gemini, needs `GEMINI_API_KEY`) |
| `--post-cmd CMD` | Hook run per generated image (`<path>\n<caption>` on stdin) — wire it to a poster later |
| `--speakers` | Enable experimental speaker labeling (downloads 40 MB TitaNet model) |
| `--speaker-threshold X` | Same-speaker similarity floor (default `0.45`) |
| `--max-speakers N` | Hard cap on distinct speakers (default `8`) — set it to your table size |

## ✨ Two-pass mode (best accuracy)

Streaming decode trades a little accuracy for immediacy. Get both:

```sh
uv run thoth.py --save-audio                     # live transcript + session recording
uv run thoth.py --polish sessions/session-….wav  # then: full-context re-transcription
```

The polished pass re-reads the whole recording with full context — same model, better output, exact timestamps, minutes for a multi-hour session. `--speakers` works here too, and better than live (offline timestamps make the voice slicing precise). Live transcript for the table, polished one for the campaign log. 📖

Forgot `--notes` during the session (or only have a recording)? `--polish … --notes` generates the key-notes file offline from the polished transcript, same summarizer, same format.

## 🐦 Live session feed

`--notes` turns thoth into a play-by-play commentator. Every few minutes it hands the latest transcript to a summarizer and appends a one-line post:

```
- [0:12:00] The group is fishing for knucklehead trout.
- [0:15:00] The group is arguing about who touched the trapped door.
- [0:18:00] The group found the lake monster. It found them first.
```

Summaries run on a background thread — the transcription loop never blocks. The default summarizer is the `claude` CLI (Haiku), but `--notes-cmd` accepts any shell command that reads a prompt on stdin and prints a line: a local model via `ollama run`, or someday a script that posts straight to a stream overlay or social feed.

## 📚 The full ritual

A session leaves a family of artifacts, each derived from the last:

```
session-<ts>.md              live transcript        (always)
session-<ts>.wav             recording              (--save-audio)
key-notes-<ts>.md            headline feed          (--notes)
session-<ts>-polished.md     full-context re-pass   (--polish <wav>)
key-notes-enriched-<ts>.md   vivid chronicle        (--enrich <key-notes>)
gallery/<ts>/*.png           illustrated scenes     (--imagine <key-notes>)
cutscenes/<ts>/<span>/       narration.wav + keyframe manifest (studio 🎬)
```

Drop your party's character portraits in `avatars/` (one image per character, named after them) and `--imagine` paints each key-note with the party in it. Runs are resumable — existing scenes are skipped, so a failed run just continues.

Stopped and restarted mid-session? `aggregate.py` stitches the parts into `campaign-*` files:

```sh
uv run aggregate.py sessions campaign-2026-07-18
```

The full post-session ritual, in order:

```sh
uv run thoth.py --polish sessions/session-<ts>.wav       # 1. accurate transcript
uv run thoth.py --enrich sessions/key-notes-<ts>.md      # 2. vivid chronicle
uv run studio.py                                         # 3. illustrate & curate
```

## 🎨 Scene studio

```sh
uv run studio.py        # → http://localhost:8511
```

A local curation UI for turning key-notes into illustrated scenes:

- 📜 Browse every key-note from every session (chronicle text inline), pick one to work on
- 🧝 Toggle which party members ride along as character references
- 🎭 Prompts assemble from three layers — **Command** (style + party bible) / **Setting** (persistent scene state) / **Scene** (the beat) — so staging stays continuous across a whole stretch of session
- 📍 **Derive from transcript** — a stage-manager pass reads the polished transcript around the note and drafts the Setting (location, who's present, who's where, ongoing state); edit it once, it persists until the party moves
- ✨ **Elaborate** asks an art director (the `claude` CLI) to expand the beat into composition, camera, and action — without restating the Setting
- 🧬 **Party & NPC bibles** — `avatars/party.md` (one visual line per character, injected beside their reference image) and `avatars/npcs.md` (recurring NPCs stay consistent, injected only when the scene names them)
- 🎞️ **16:9 by default** (aspect picker for 1:1/9:16/…) and 🔗 **continuity chaining** — the previous promoted keyframe rides along as a reference so a cutscene's frames share palette and light
- 🖼️ **Conjure** sends it to Nano Banana **Pro** (`gemini-3-pro-image-preview`, ~13¢, ChatGPT-tier) or regular Nano Banana (~4¢) — picker in the UI. Every take is kept in gitignored `generated-images/<session>/` with a JSON sidecar of the exact prompt, party, and model used
- ✦ **Promote** copies your favorite take into `gallery/<session>/` (the curated record); **Skip** moves on, takes stay in staging
- 🎬 **Cutscenes mode** — the second tab. A cutscene is an *arc*, not a moment: click a start note and an end note, and the studio shows the promoted keyframes inside that span (with a 6-frame target for the ComfyUI template) and a recap-narration workbench. **Draft recap** weaves the span's chronicle entries into one "previously on…" story at a word budget, pick a voice (Charon the grave narrator, Fenrir, Kore, Aoede, Puck), **Narrate** renders it via Gemini TTS (~1¢/take) into gitignored `generated-audio/<session>/`, and **Promote 🎬** writes `cutscenes/<session>/<span>/` — `narration.wav` plus a `manifest.json` listing keyframes, script, voice, and duration: everything video assembly needs

**How narration length works**: duration = word count ÷ speaking pace, and pace is a *property of the delivery*, not a setting — the dramatic style runs 75–102 wpm across takes (vs ~150 for plain speech). So the studio budgets words for your target (30 s ≈ 48 words), shows each take's measured duration against target, and the last ±10% is snapped exactly at video-assembly time with `ffmpeg -af atempo` (imperceptible). Aim a hair under; padding silence is free, cutting isn't.

One-time setup: the [two prerequisites above](#-setup) (Gemini key + `claude` CLI), plus party portraits — one image per character in `avatars/`, filename = character name (`corvus.png` → "Corvus" in prompts).

For unattended batch generation of a whole session, `uv run thoth.py --imagine sessions/key-notes-<ts>.md` does the same thing without the curation step.

## 🗺️ Roadmap

- 🎬 **Session films** — feed 6 promoted gallery scenes into a [ComfyUI keyframe workflow](https://comfy.org/workflows/templates-6-key-frames-920c6926e747/) per clip, `ffmpeg concat` the clips into a session film
- 📡 **Posting hooks** — `--post-cmd` (images) and `--notes-cmd` (headlines) are stdin→stdout shell contracts; point them at a social API or stream overlay when the time comes
- 🎭 **Better diarization** — calibration phase (each player says a line at session start), stronger embedding models

## 🧱 What it is (and isn't)

Three small PEP 723 scripts — no venv, no pyproject, no build step; `uv` handles everything:

| File | Role |
|------|------|
| `thoth.py` | The scribe: transcribe, record, polish, notes, enrich, batch-imagine |
| `studio.py` + `studio.html` | The atelier: local web UI for curated scene generation |
| `aggregate.py` | The binder: stitch multi-part sessions into `campaign-*` files |

- 🍎 Apple Silicon only (MLX)
- 🗣️ Acoustic `--speakers` is per-sentence clustering and **cannot work on a single far-field mic** (field-tested: more `--max-speakers` only fragments people across labels, and overlap makes labels leaky). Use `--attribute` instead — dialogue is self-identifying ("maul damage" = the barbarian), so an LLM pass over the polished transcript beats voiceprints on one mic. The acoustic ceiling-raiser isn't a $300 mic: it's **one track per speaker** (everyone's phone records Voice Memos, sync by a clap) — per-track transcription makes diarization perfect by construction
- 🇬🇧 English-tuned default model; puns in Common only

## 📜 License

MIT. Go forth and transcribe.

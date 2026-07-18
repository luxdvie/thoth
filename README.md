# 𓅝 thoth

> *Scribe of the gods. Keeper of the record. Now taking notes at your D&D table.*

**Realtime, fully-local transcription for tabletop sessions.** Mic → live terminal text → timestamped markdown on disk. Runs [NVIDIA Parakeet](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) on your Mac's Neural Engine via [MLX](https://github.com/ml-explore/mlx).

🔒 **No cloud. No API keys. No subscription.** Your table talk never leaves the machine.

```
[0:00:01] The party enters the dungeon.
[0:00:03] Roll for initiative.
[0:00:05] I check the door for traps before anyone touches it.
[0:00:08] The goblin attacks the wizard with a rusty dagger.
… and I swear if you crit me again I'm flipping the tab
```

## ⚡ Quickstart

```sh
uv run thoth.py
```

That's it. First run pulls the model (~600 MB), then you're live:

- ✍️ Finalized sentences print with `[H:MM:SS]` stamps — once committed, a line **never changes** (finalization trails realtime by ~15–20 s; the live line stays current)
- 🔮 The in-flight text updates on a live ticker line as the model changes its mind
- 🎭 `--speakers` *(experimental)* — voice-fingerprints each sentence (TitaNet embeddings, online clustering) and tags it `Speaker N`, color-coded. Works on clean audio; still being tuned for far-field party chaos
- 💾 Full transcript rewritten to `sessions/session-<date>.md` every ~2 seconds — **a crash loses nothing**
- 🛑 `Ctrl-C` ends the session

Four hours of table time becomes a searchable, timestamped campaign log. Feed it to your favorite LLM for session recaps, quote your rogue's exact words back at them, settle the "you never told us about the trapdoor" dispute with receipts.

## 🎛️ Options

| Flag | Does |
|------|------|
| `--out DIR` | Output directory (default `./sessions`) |
| `--device NAME` | Pick a mic — list with `uv run --with sounddevice python -m sounddevice` |
| `--model ID` | Any parakeet-mlx-compatible Hugging Face model |
| `--speakers` | Enable experimental speaker labeling (downloads 40 MB TitaNet model) |
| `--speaker-threshold X` | Same-speaker similarity floor (default `0.45`) |
| `--max-speakers N` | Hard cap on distinct speakers (default `8`) — set it to your table size |

## 🧱 What it is (and isn't)

One file. ~250 lines. PEP 723 inline deps — no venv, no pyproject, no build step. `uv` handles everything.

- 🍎 Apple Silicon only (MLX)
- 🗣️ Speaker ID is per-sentence clustering, not full diarization — two people talking over each other land in one line, and very similar voices may merge (tune `--speaker-threshold`)
- 🇬🇧 English-tuned default model; puns in Common only

## 📜 License

MIT. Go forth and transcribe.

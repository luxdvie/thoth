# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Aggregate thoth session artifacts into campaign-* files.

Usage: uv run aggregate.py <sessions_dir> <campaign_name> [--pattern session|key-notes|key-notes-enriched|wav ...]
Concatenates matching files in chronological (filename) order with part headers.
"""
import struct
import sys
from pathlib import Path

sessions = Path(sys.argv[1])
name = sys.argv[2]
patterns = sys.argv[3:] or ["session", "key-notes", "key-notes-enriched", "wav"]


def concat_md(glob: str, out: Path, title: str, exclude: tuple[str, ...] = ()) -> None:
    parts = sorted(
        p for p in sessions.glob(glob)
        if p != out and not p.stem.startswith("campaign-") and not any(x in p.stem for x in exclude)
    )
    if not parts:
        print(f"  (no files for {glob})")
        return
    blocks = [f"# {title}\n"]
    for p in parts:
        blocks.append(f"\n## Part: {p.stem}\n\n{p.read_text().strip()}\n")
    out.write_text("\n".join(blocks))
    print(f"  {out.name} <- {len(parts)} parts")


def concat_wav(out: Path) -> None:
    parts = sorted(sessions.glob("session-*.wav"))
    parts = [p for p in parts if p != out]
    if not parts:
        return
    total = 0
    with open(out, "wb") as f:
        f.write(b"\x00" * 44)
        for p in parts:
            with open(p, "rb") as src:
                src.seek(44)
                while chunk := src.read(1 << 22):
                    f.write(chunk)
                    total += len(chunk)
        f.seek(0)
        f.write(b"RIFF" + struct.pack("<I", 36 + total) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
        f.write(b"data" + struct.pack("<I", total))
    print(f"  {out.name} <- {len(parts)} parts ({total / 32000 / 3600:.2f} h)")


for pat in patterns:
    if pat == "session":
        concat_md("session-*.md", sessions / f"{name}.md", f"{name} — full transcript", exclude=("-polished",))
    elif pat == "session-polished":
        concat_md("session-*-polished.md", sessions / f"{name}-polished.md", f"{name} — polished transcript")
    elif pat == "key-notes":
        concat_md("key-notes-2*.md", sessions / f"{name}-key-notes.md", f"{name} — key notes")
    elif pat == "key-notes-enriched":
        concat_md("key-notes-enriched-*.md", sessions / f"{name}-key-notes-enriched.md", f"{name} — enriched chronicle")
    elif pat == "wav":
        concat_wav(sessions / f"{name}.wav")

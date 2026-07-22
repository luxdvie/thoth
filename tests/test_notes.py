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
"""Unit tests for the notes layer (NoteTaker + offline_notes).

    uv run tests/test_notes.py

No microphone, no ASR inference, no API keys: summarizer commands are stubbed
with shell one-liners. `tail -n 1` echoes the last transcript line of each
window back as the "note", which lets assertions see exactly which window each
summarizer call received.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thoth  # noqa: E402

thoth.time.sleep = lambda s: None  # retry backoff is irrelevant in tests

ROWS = [  # (t, speaker, text) — 3 windows at interval=180: [0,60] [200] [390]
    (0.0, None, "a1"),
    (60.0, None, "a2"),
    (200.0, None, "b1"),
    (390.0, None, "c1"),
]


class S:
    """Stub for a parakeet sentence (only .text is read by NoteTaker)."""

    def __init__(self, text: str):
        self.text = text


def test_offline_notes_windows_and_stamps(tmp: Path):
    out = tmp / "key-notes-x.md"
    cap = tmp / "prompts.txt"
    thoth.offline_notes(ROWS, out, f"tee -a {cap} | tail -n 1", 180)
    assert thoth.parse_stamped(out) == [(0, "a2"), (200, "b1"), (390, "c1")], out.read_text()
    prompts = cap.read_text()
    assert "a1\na2" in prompts, "first window should contain both sentences"
    assert "Previous post: a2" in prompts, "second call should chain the previous note"


def test_offline_notes_skip_produces_no_file(tmp: Path):
    out = tmp / "key-notes-x.md"
    thoth.offline_notes(ROWS, out, "echo SKIP", 180)
    assert not out.exists(), "SKIP notes must not be written"


def test_offline_notes_never_overwrites(tmp: Path):
    out = tmp / "key-notes-x.md"
    out.write_text("- [0:00:00] precious\n")
    thoth.offline_notes(ROWS, out, "echo clobber", 180)
    assert out.read_text() == "- [0:00:00] precious\n"


def test_offline_notes_survives_summarizer_failure(tmp: Path):
    out = tmp / "key-notes-x.md"
    thoth.offline_notes(ROWS, out, "false", 180)
    assert not out.exists(), "failed windows must not write garbage"


def test_glossary_reaches_summarizer_prompt(tmp: Path):
    gpath = tmp / "glossary.md"
    gpath.write_text("Auril: the Frostmaiden. Heard as 'Oh Reel'.\n")
    out = tmp / "key-notes-x.md"
    cap = tmp / "prompts.txt"
    thoth.offline_notes(ROWS, out, f"tee -a {cap} | tail -n 1", 180, thoth.glossary_block(gpath))
    prompts = cap.read_text()
    assert "Oh Reel" in prompts, "glossary entries should be injected into the prompt"
    assert "Glossary" in prompts, "glossary clause header missing"


def test_missing_or_empty_glossary_adds_no_clause(tmp: Path):
    assert thoth.glossary_block(tmp / "nope.md") == ""
    empty = tmp / "empty.md"
    empty.write_text("  \n")
    assert thoth.glossary_block(empty) == ""
    out = tmp / "key-notes-x.md"
    cap = tmp / "prompts.txt"
    thoth.offline_notes(ROWS, out, f"tee -a {cap} | tail -n 1", 180)
    assert "Glossary" not in cap.read_text()


def test_all_prompts_have_glossary_slot(tmp: Path):
    for tpl in (thoth.NOTE_PROMPT, thoth.ENRICH_PROMPT, thoth.ATTRIBUTE_PROMPT):
        assert "{glossary}" in tpl


def test_live_notetaker_stamps_at_window_start(tmp: Path):
    out = tmp / "key-notes-x.md"
    nt = thoth.NoteTaker(out, "tail -n 1", 10)
    sentences, starts = [S("one"), S("two")], [1.0, 5.0]
    nt.maybe_fire(4.0, sentences, starts)  # 4s in: below interval, must not fire
    nt.finish(12.0, sentences, starts)
    # Stamped where the window's content BEGAN (1.0s), not at fire time.
    assert thoth.parse_stamped(out) == [(1, "two")], out.read_text()


def test_live_notetaker_takes_last_stdout_line(tmp: Path):
    out = tmp / "key-notes-x.md"
    nt = thoth.NoteTaker(out, "printf 'Some CLI preamble\\nthe real note\\n'", 10)
    nt.finish(60.0, [S("hi")], [0.0])
    assert thoth.parse_stamped(out) == [(0, "the real note")], out.read_text()


def main() -> None:
    failures = 0
    for name, fn in sorted(t for t in globals().items() if t[0].startswith("test_")):
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"  ok  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    if failures:
        sys.exit(1)
    print("all green")


if __name__ == "__main__":
    main()

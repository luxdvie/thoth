# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""thoth studio — local curation UI for illustrated key-notes.

    uv run studio.py            # serves http://localhost:8511 and opens it

Reads sessions/key-notes*.md and avatars/. Every generation is saved to
generated-images/<session>/ (gitignored) with a JSON sidecar; "Promote" copies
the chosen image into gallery/<session>/ and rebuilds its gallery.md.
"""

import base64
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
AVATARS = ROOT / "avatars"
STAGING = ROOT / "generated-images"
GALLERY = ROOT / "gallery"
PORT = 8511
STAMP_RE = re.compile(r"^(?:-|##)? ?\[(\d+):(\d\d):(\d\d)\] (.*)$")
IMAGE_MODEL = os.environ.get("THOTH_IMAGE_MODEL", "gemini-2.5-flash-image")

IMAGE_PROMPT = (
    "Illustrate this moment from a D&D campaign as a single dramatic fantasy scene. "
    "The attached reference images are the party's characters ({names}) — keep their "
    "faces, builds, and gear recognizable. Cinematic lighting, painterly fantasy "
    "illustration style, no text or borders in the image.\n\nScene: {scene}"
)


def stamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"[{h}:{m:02d}:{s:02d}]"


def parse_notes(path: Path) -> list[dict]:
    notes, body_lines = [], None
    for line in path.read_text().splitlines():
        if m := STAMP_RE.match(line):
            h, mnt, s, text = m.groups()
            body_lines = []
            notes.append({"t": int(h) * 3600 + int(mnt) * 60 + int(s), "headline": text, "body": body_lines})
        elif body_lines is not None and line.strip():
            body_lines.append(line.strip())
    for n in notes:
        n["body"] = " ".join(n["body"])
    return notes


def load_state() -> dict:
    sessions = []
    seen = set()
    for p in sorted(SESSIONS.glob("key-notes-*.md")):
        sid = p.stem.removeprefix("key-notes-enriched-").removeprefix("key-notes-")
        if sid in seen:
            continue
        seen.add(sid)
        enriched = SESSIONS / f"key-notes-enriched-{sid}.md"
        plain = SESSIONS / f"key-notes-{sid}.md"
        notes = parse_notes(enriched if enriched.exists() else plain)
        if enriched.exists() and plain.exists():  # headlines from plain, bodies from enriched
            bodies = {n["t"]: n["body"] for n in notes}
            notes = parse_notes(plain)
            for n in notes:
                n["body"] = bodies.get(n["t"], "")
        promoted = {f.name.split("-")[0] for f in (GALLERY / sid).glob("*.png")} if (GALLERY / sid).is_dir() else set()
        for i, n in enumerate(notes):
            n["idx"] = i
            n["stamp"] = stamp(n["t"])
            n["promoted"] = f"{i:03d}" in promoted
            gen_dir = STAGING / sid
            n["generations"] = sorted(
                f"/generated/{sid}/{f.name}" for f in gen_dir.glob(f"{i:03d}-*.png")
            ) if gen_dir.is_dir() else []
        sessions.append({"sid": sid, "notes": notes})
    avatars = [
        {"name": p.stem, "url": f"/avatars/{p.name}"}
        for p in sorted(AVATARS.glob("*"))
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    return {"sessions": sessions, "avatars": avatars, "prompt_template": IMAGE_PROMPT, "model": IMAGE_MODEL}


def api_key() -> str:
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    try:  # fish universal vars aren't exported to non-fish parents
        import subprocess
        if key := subprocess.run(["fish", "-c", "echo -n $GEMINI_API_KEY"], capture_output=True, timeout=5).stdout.decode().strip():
            return key
    except Exception:
        pass
    raise RuntimeError("GEMINI_API_KEY is not set")


def gemini_generate(prompt: str, avatar_names: list[str]) -> bytes:
    key = api_key()
    parts = []
    for name in avatar_names:
        matches = [p for p in AVATARS.glob(f"{name}.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not matches:
            raise RuntimeError(f"no avatar image for {name!r}")
        p = matches[0]
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else f"image/{p.suffix.lower().lstrip('.')}"
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(p.read_bytes()).decode()}})
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    err = ""
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                rparts = json.load(resp)["candidates"][0]["content"]["parts"]
                if data := next((p["inlineData"]["data"] for p in rparts if "inlineData" in p), None):
                    return base64.b64decode(data)
                err = "model returned no image part (likely refused the prompt)"
        except Exception as e:
            err = getattr(e, "read", lambda: b"")()[-300:].decode(errors="replace") or str(e)
        if attempt == 1:
            time.sleep(10)
    raise RuntimeError(err)


def rebuild_gallery_md(sid: str) -> None:
    gdir = GALLERY / sid
    caps_path = gdir / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    entries = []
    for f in sorted(gdir.glob("*.png")):
        c = caps.get(f.name, {})
        entries.append(f"## {c.get('stamp', '')} {c.get('headline', f.name)}\n\n![]({f.name})\n")
    (gdir / "gallery.md").write_text(f"# {sid}\n\n" + "\n".join(entries))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, base: Path, rel: str, ctype: str):
        f = (base / rel).resolve()
        if not f.is_relative_to(base) or not f.is_file():
            return self._send(404, b"{}")
        self._send(200, f.read_bytes(), ctype)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, (ROOT / "studio.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(load_state()).encode())
        elif self.path.startswith("/avatars/"):
            self._static(AVATARS, self.path.removeprefix("/avatars/"), "image/png")
        elif self.path.startswith("/generated/"):
            self._static(STAGING, self.path.removeprefix("/generated/"), "image/png")
        elif self.path.startswith("/gallery/"):
            self._static(GALLERY, self.path.removeprefix("/gallery/"), "image/png")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        try:
            if self.path == "/api/generate":
                sid, idx = payload["sid"], int(payload["idx"])
                dest_dir = STAGING / sid
                dest_dir.mkdir(parents=True, exist_ok=True)
                serial = len(list(dest_dir.glob(f"{idx:03d}-*.png")))
                png = gemini_generate(payload["prompt"], payload.get("avatars", []))
                dest = dest_dir / f"{idx:03d}-{serial:02d}.png"
                dest.write_bytes(png)
                dest.with_suffix(".json").write_text(json.dumps({
                    "prompt": payload["prompt"], "avatars": payload.get("avatars", []),
                    "headline": payload.get("headline", ""), "stamp": payload.get("stamp", ""),
                }, indent=2))
                self._send(200, json.dumps({"url": f"/generated/{sid}/{dest.name}"}).encode())
            elif self.path == "/api/promote":
                sid, idx = payload["sid"], int(payload["idx"])
                src = (STAGING / sid / Path(payload["file"]).name).resolve()
                assert src.is_relative_to(STAGING) and src.is_file()
                gdir = GALLERY / sid
                gdir.mkdir(parents=True, exist_ok=True)
                dest = gdir / f"{idx:03d}-{payload.get('stamp', '').strip('[]').replace(':', '-')}.png"
                shutil.copy2(src, dest)
                caps_path = gdir / "captions.json"
                caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
                caps[dest.name] = {"stamp": payload.get("stamp", ""), "headline": payload.get("headline", "")}
                caps_path.write_text(json.dumps(caps, indent=2))
                rebuild_gallery_md(sid)
                self._send(200, json.dumps({"promoted": f"/gallery/{sid}/{dest.name}"}).encode())
            else:
                self._send(404, b"{}")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)[:400]}).encode())


def main() -> None:
    if not SESSIONS.is_dir():
        sys.exit("run from the thoth repo (no sessions/ here)")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"𓅝 thoth studio → http://localhost:{PORT}  (Ctrl-C to stop)")
    threading.Timer(0.4, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

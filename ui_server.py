#!/usr/bin/env python3
"""Local page for Amanah. Serves the UI and runs the existing pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from transcribe_audio import DEFAULT_MODEL, WHISPER_ENGINE


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8741

_lock = threading.Lock()
_job: Optional[dict[str, Any]] = None


def _paths_for(video: Path) -> tuple[Path, Path, Path]:
    workdir = video.with_name(f"{video.stem}_amanah")
    report = workdir / f"{video.stem}_report.json"
    output = video.with_name(f"{video.stem}_clean{video.suffix or '.mp4'}")
    return workdir, report, output


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def health() -> dict[str, Any]:
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "ollama": _ollama_up(),
        "whisper": WHISPER_ENGINE,
        "whisper_model": DEFAULT_MODEL,
    }


def pick_film() -> Optional[str]:
    """Open a native file dialog. Returns None when the user cancels."""
    if sys.platform == "darwin":
        script = 'POSIX path of (choose file with prompt "Choose a film")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    if sys.platform == "win32":
        command = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.OpenFileDialog; "
            "$d.Filter = 'Video|*.mp4;*.mkv;*.mov;*.m4v;*.avi;*.webm|All files|*.*'; "
            "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    return None


def _note(job: dict[str, Any], line: str) -> None:
    job["log"].append(line)
    if len(job["log"]) > 400:
        del job["log"][:-400]
    if "1/3 Transcribe" in line:
        job["step"] = "transcribe"
    elif "2/3 Analyze" in line or "Analyzing scene" in line:
        job["step"] = "analyze"
    elif "3/3 Remove" in line or "Encoding cleaned" in line:
        job["step"] = "cut"
    elif "PIPELINE COMPLETE" in line or "Skipping cut" in line:
        job["step"] = "done"


def _load_report(job: dict[str, Any]) -> None:
    path = Path(job["report_path"])
    if not path.is_file():
        return
    try:
        job["report"] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return


def _run(job: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "amanah.py"),
        job["video"],
        "--llm-model",
        job["model"],
    ]
    if job["skip_cut"]:
        cmd.append("--skip-cut")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        return

    assert proc.stdout is not None
    for raw in proc.stdout:
        _note(job, raw.rstrip("\n"))
    code = proc.wait()
    _load_report(job)
    if code == 0:
        job["status"] = "done"
        job["step"] = "done"
    else:
        job["status"] = "error"
        job["error"] = f"Pipeline stopped (exit {code})."


def start_job(video: Path, *, model: str, skip_cut: bool) -> dict[str, Any]:
    global _job
    _, report, output = _paths_for(video)
    job = {
        "status": "running",
        "step": "transcribe",
        "video": str(video),
        "model": model,
        "skip_cut": skip_cut,
        "report_path": str(report),
        "output": None if skip_cut else str(output),
        "report": None,
        "error": None,
        "log": [],
    }
    with _lock:
        if _job and _job["status"] == "running":
            raise RuntimeError("A film is already being processed.")
        _job = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


def public_job(job: Optional[dict[str, Any]]) -> dict[str, Any]:
    if job is None:
        return {"status": "idle"}
    if job["status"] == "running":
        _load_report(job)
    return {
        "status": job["status"],
        "step": job["step"],
        "video": job["video"],
        "skip_cut": job["skip_cut"],
        "output": job["output"],
        "report": job["report"],
        "error": job["error"],
        "log": job["log"],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 1_000_000))
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._json(404, {"error": "Not found."})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(WEB / "index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file(WEB / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(WEB / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/api/health":
            self._json(200, health())
            return
        if path == "/api/job":
            with _lock:
                job = _job
            self._json(200, public_job(job))
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except (json.JSONDecodeError, ValueError) as exc:
            self._json(400, {"error": str(exc)})
            return

        if path == "/api/pick":
            chosen = pick_film()
            self._json(200, {"path": chosen})
            return

        if path == "/api/job":
            raw = str(data.get("video") or "").strip().strip('"')
            if not raw:
                self._json(400, {"error": "Choose a film first."})
                return
            video = Path(raw).expanduser().resolve()
            if not video.is_file():
                self._json(400, {"error": f"Film not found: {video}"})
                return
            model = str(data.get("model") or "qwen2.5:7b").strip() or "qwen2.5:7b"
            skip_cut = bool(data.get("skip_cut"))
            try:
                job = start_job(video, model=model, skip_cut=skip_cut)
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
                return
            self._json(200, public_job(job))
            return

        if path == "/api/reveal":
            raw = str(data.get("path") or "").strip()
            target = Path(raw).expanduser()
            if not raw or not target.exists():
                self._json(400, {"error": "Nothing to show yet."})
                return
            reveal(target)
            self._json(200, {"ok": True})
            return

        self._json(404, {"error": "Not found."})


def main() -> int:
    if not (WEB / "index.html").is_file():
        print(f"Missing {WEB / 'index.html'}", file=sys.stderr)
        return 1
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print(f"Cannot listen on {HOST}:{PORT}: {exc}", file=sys.stderr)
        return 1
    url = f"http://{HOST}:{PORT}"
    print(f"Amanah is at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

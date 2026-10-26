#!/usr/bin/env python3
"""Transcribe spoken English from a video or audio file.

Reads the audio track and writes the words that were spoken. The model is
forced to transcribe English. It does not translate the dialogue.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


MLX_MODEL = "mlx-community/whisper-small-mlx"
FASTER_WHISPER_MODEL = "small"
DEFAULT_LANGUAGE = "en"


def uses_faster_whisper() -> bool:
    """Windows uses faster-whisper. macOS keeps mlx-whisper."""
    return sys.platform == "win32"


WHISPER_ENGINE = "faster-whisper" if uses_faster_whisper() else "mlx-whisper"
DEFAULT_MODEL = FASTER_WHISPER_MODEL if uses_faster_whisper() else MLX_MODEL


def project_venv_python() -> Path:
    """Python inside the project virtual environment, for this operating system."""
    root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _engine_module_name() -> str:
    return "faster_whisper" if uses_faster_whisper() else "mlx_whisper"


@dataclass(frozen=True)
class Cue:
    """One spoken line with start and end times in seconds."""

    index: int
    start: float
    end: float
    text: str


def srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp, HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cues_to_srt(cues: list[Cue]) -> str:
    parts: list[str] = []
    for cue in cues:
        parts.append(
            f"{cue.index}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{cue.text}\n"
        )
    return "\n".join(parts) + ("\n" if parts else "")


def cues_to_text(cues: list[Cue]) -> str:
    return "\n".join(cue.text for cue in cues) + ("\n" if cues else "")


def engine_is_installed() -> bool:
    try:
        __import__(_engine_module_name())
    except ImportError:
        return False
    return True


def install_hint() -> str:
    if sys.platform == "win32":
        return (
            "py -3 -m venv .venv\n"
            "  .venv\\Scripts\\python -m pip install -r requirements.txt"
        )
    return "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"


def reexec_project_venv_if_needed() -> None:
    """Run again with the project venv when the speech engine is installed there."""
    if os.environ.get("AMANAH_TRANSCRIBE_REEXEC") == "1":
        return
    if engine_is_installed():
        return
    venv_python = project_venv_python()
    if venv_python.is_file():
        os.environ["AMANAH_TRANSCRIBE_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


def extract_wav(
    source: Path,
    wav_path: Path,
    *,
    start: Optional[float],
    end: Optional[float],
) -> None:
    """Extract mono 16 kHz PCM audio, the format Whisper expects."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(source)])
    if start is not None and end is not None:
        cmd.extend(["-t", f"{max(0.0, end - start):.3f}"])
    elif end is not None:
        cmd.extend(["-to", f"{end:.3f}"])
    cmd.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to read audio from {source}") from exc


def transcribe_english(wav_path: Path, model: str, *, verbose: bool) -> dict[str, Any]:
    """Transcribe English speech. The task is transcribe, never translate."""
    try:
        if uses_faster_whisper():
            return _transcribe_faster(wav_path, model, verbose=verbose)
        return _transcribe_mlx(wav_path, model, verbose=verbose)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc


def _transcribe_mlx(wav_path: Path, model: str, *, verbose: bool) -> dict[str, Any]:
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model,
        language=DEFAULT_LANGUAGE,
        task="transcribe",
        verbose=verbose,
        condition_on_previous_text=True,
        initial_prompt="English dialogue.",
    )


def _transcribe_faster(wav_path: Path, model: str, *, verbose: bool) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device="auto", compute_type="default")
    segments, info = whisper.transcribe(
        str(wav_path),
        language=DEFAULT_LANGUAGE,
        task="transcribe",
        condition_on_previous_text=True,
        initial_prompt="English dialogue.",
    )
    collected: list[dict[str, Any]] = []
    for segment in segments:
        text = segment.text
        if verbose:
            print(f"[{segment.start:.2f} -> {segment.end:.2f}] {text.strip()}")
        collected.append({"start": segment.start, "end": segment.end, "text": text})
    return {"language": info.language, "segments": collected}


def segments_to_cues(result: dict[str, Any], time_offset: float) -> list[Cue]:
    cues: list[Cue] = []
    for segment in result.get("segments") or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0)) + time_offset
        end = float(segment.get("end", start)) + time_offset
        if end < start:
            end = start
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=text))
    return cues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe spoken English from a video or audio file. "
            "Writes the original English words, not a translation."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Video or audio file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="SRT output path (default: <source>_audio.srt)",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Also write plain text to this path",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Also write timed cues as JSON",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"{WHISPER_ENGINE} model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-segment progress",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    reexec_project_venv_if_needed()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source: Path = args.source
    if not source.is_file():
        print(f"Error: file not found: {source}", file=sys.stderr)
        return 1
    if args.start is not None and args.end is not None and args.end <= args.start:
        print("Error: --end must be greater than --start", file=sys.stderr)
        return 1

    if not engine_is_installed():
        print(
            f"Error: {WHISPER_ENGINE} is not installed.\n"
            f"  {install_hint()}",
            file=sys.stderr,
        )
        return 1

    output = args.output or source.with_name(f"{source.stem}_audio.srt")
    offset = args.start or 0.0

    print(f"Source    : {source}")
    print("Language  : English transcription")
    print(f"Engine    : {WHISPER_ENGINE}")
    print(f"Model     : {args.model}")
    print(f"SRT       : {output}")

    try:
        with tempfile.TemporaryDirectory(prefix="amanah_audio_") as tmp:
            wav_path = Path(tmp) / "speech.wav"
            print("Extracting audio...")
            extract_wav(source, wav_path, start=args.start, end=args.end)
            print("Transcribing English speech...")
            result = transcribe_english(wav_path, args.model, verbose=not args.quiet)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    detected = result.get("language")
    if detected and detected != DEFAULT_LANGUAGE:
        print(
            f"Error: model reported language {detected!r}, expected English. "
            "Refusing to write a non-English transcript.",
            file=sys.stderr,
        )
        return 1

    cues = segments_to_cues(result, time_offset=offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cues_to_srt(cues), encoding="utf-8")

    txt_path = args.txt
    if txt_path is None and args.output is None:
        txt_path = source.with_name(f"{source.stem}_audio.txt")
    if txt_path is not None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(cues_to_text(cues), encoding="utf-8")

    if args.json_path is not None:
        payload = {
            "source": str(source),
            "language": DEFAULT_LANGUAGE,
            "task": "transcribe",
            "model": args.model,
            "cue_count": len(cues),
            "cues": [asdict(cue) for cue in cues],
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"\nWrote {len(cues)} English cues to {output}")
    if txt_path is not None:
        print(f"Plain text: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

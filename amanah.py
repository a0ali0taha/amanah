#!/usr/bin/env python3
"""Amanah: local family-film cleaner.

Takes a video you already have and writes a shorter cut by:
  1. Transcribing English dialogue (mlx-whisper on Mac, faster-whisper on Windows)
  2. Reviewing scenes for content unsuitable for children (Ollama)
  3. Removing flagged scenes with ffmpeg

Example:
  python3 amanah.py "/path/to/film.mp4"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from transcribe_audio import DEFAULT_MODEL, WHISPER_ENGINE, project_venv_python


ROOT = Path(__file__).resolve().parent
TRANSCRIBE_SCRIPT = ROOT / "transcribe_audio.py"
ANALYZE_SCRIPT = ROOT / "analyze_transcript.py"
REMOVE_SCRIPT = ROOT / "remove_unsafe_scenes.py"


def resolve_python_for_transcribe() -> Path:
    """Prefer the project virtual environment, where the speech engine is installed."""
    venv_python = project_venv_python()
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


def run_step(name: str, cmd: list[str], *, cwd: Path) -> None:
    print("\n" + "=" * 72)
    print(f"STEP: {name}")
    print("=" * 72)
    print(" ".join(cmd))
    print()
    started = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({name}) with exit code {result.returncode}")
    print(f"\n[{name}] completed in {elapsed / 60:.1f} min")


def summarize_report(report_path: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    flagged = data.get("flagged") or []
    print("\n" + "-" * 72)
    print("ANALYSIS SUMMARY")
    print("-" * 72)
    print(f"Scenes analyzed : {data.get('total_scenes')}")
    print(f"Flagged scenes  : {len(flagged)}")
    if flagged:
        print("Sample reasons:")
        for item in flagged[:8]:
            print(
                f"  #{item.get('scene_number')} "
                f"[{item.get('start')} --> {item.get('end')}] "
                f"{item.get('reason')}"
            )
        if len(flagged) > 8:
            print(f"  ... and {len(flagged) - 8} more")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Amanah: transcribe a film, review dialogue for family suitability, "
            "and write a cleaned video with flagged scenes removed."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Input video file",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory for SRT/report/EDL (default: <video>_amanah next to the video)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Cleaned video path (default: <video_stem>_clean.mp4)",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_MODEL,
        help=f"{WHISPER_ENGINE} model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:7b",
        help="Ollama model for scene analysis (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.25,
        help="Extra seconds to cut before/after each flagged scene",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Analyze only first N scenes (smoke test)",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset for cleaned encode (default: veryfast)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="x264 CRF for cleaned encode (default: 20)",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Reuse existing SRT if present",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Reuse existing report JSON if present",
    )
    parser.add_argument(
        "--skip-cut",
        action="store_true",
        help="Stop after analysis (do not encode cleaned video)",
    )
    parser.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="Explicit SRT path (skips transcription if file exists with --skip-transcribe)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    video: Path = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    for script in (TRANSCRIBE_SCRIPT, ANALYZE_SCRIPT, REMOVE_SCRIPT):
        if not script.is_file():
            print(f"Error: missing script {script}", file=sys.stderr)
            return 1

    workdir = (args.workdir or video.with_name(f"{video.stem}_amanah")).expanduser()
    if not workdir.is_absolute():
        workdir = (Path.cwd() / workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    srt_path = (args.srt or workdir / f"{video.stem}.srt").expanduser().resolve()
    report_path = workdir / f"{video.stem}_report.json"
    edl_path = workdir / f"{video.stem}_edl.txt"
    output = (
        args.output.expanduser().resolve()
        if args.output
        else video.with_name(f"{video.stem}_clean{video.suffix or '.mp4'}")
    )

    print("=" * 72)
    print("AMANAH")
    print("=" * 72)
    print(f"Video   : {video}")
    print(f"Workdir : {workdir}")
    print(f"SRT     : {srt_path}")
    print(f"Report  : {report_path}")
    print(f"Output  : {output}")
    print(f"Whisper : {WHISPER_ENGINE} ({args.whisper_model})")
    print(f"LLM     : {args.llm_model}")

    pipeline_started = time.time()

    try:
        # 1) Transcribe
        if args.skip_transcribe and srt_path.is_file():
            print(f"\nSkipping transcription; using {srt_path}")
        else:
            py = resolve_python_for_transcribe()
            run_step(
                "1/3 Transcribe English audio",
                [
                    str(py),
                    str(TRANSCRIBE_SCRIPT),
                    str(video),
                    "-o",
                    str(srt_path),
                    "--txt",
                    str(workdir / f"{video.stem}.txt"),
                    "--model",
                    args.whisper_model,
                ],
                cwd=ROOT,
            )

        if not srt_path.is_file():
            raise RuntimeError(f"SRT was not created: {srt_path}")

        # 2) Analyze
        if args.skip_analyze and report_path.is_file():
            print(f"\nSkipping analysis; using {report_path}")
        else:
            analyze_cmd = [
                sys.executable,
                str(ANALYZE_SCRIPT),
                str(srt_path),
                "--model",
                args.llm_model,
                "-o",
                str(report_path),
            ]
            if args.limit_scenes is not None:
                analyze_cmd.extend(["--limit-scenes", str(args.limit_scenes)])
            run_step("2/3 Analyze unsafe scenes", analyze_cmd, cwd=ROOT)

        if not report_path.is_file():
            raise RuntimeError(f"Report was not created: {report_path}")
        summarize_report(report_path)

        # 3) Cut
        if args.skip_cut:
            print("\nSkipping cut (--skip-cut).")
        else:
            run_step(
                "3/3 Remove unsafe scenes",
                [
                    sys.executable,
                    str(REMOVE_SCRIPT),
                    str(video),
                    "-r",
                    str(report_path),
                    "-o",
                    str(output),
                    "--pad",
                    str(args.pad),
                    "--preset",
                    args.preset,
                    "--crf",
                    str(args.crf),
                    "--edl",
                    str(edl_path),
                ],
                cwd=ROOT,
            )
    except RuntimeError as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    total = time.time() - pipeline_started
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"SRT     : {srt_path}")
    print(f"Report  : {report_path}")
    if not args.skip_cut:
        print(f"Clean   : {output}")
        print(f"EDL     : {edl_path}")
    print(f"Total   : {total / 60:.1f} min")
    return 0


if __name__ == "__main__":
    # Ensure child steps can find local modules when needed
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())

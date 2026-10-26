#!/usr/bin/env python3
"""Remove flagged unsafe scenes from a video using an analysis report.

Reads timestamps from a JSON report produced by analyze_transcript.py, then
builds a cleaned video that keeps only the safe intervals via ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from analyze_transcript import seconds_to_timestamp, timestamp_to_seconds


@dataclass(frozen=True)
class Interval:
    """Half-open-ish closed time range in seconds [start, end]."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def expand(self, pad: float, video_duration: Optional[float] = None) -> "Interval":
        start = max(0.0, self.start - pad)
        end = self.end + pad
        if video_duration is not None:
            end = min(video_duration, end)
        return Interval(start=start, end=max(start, end))


def load_report(path: Path) -> dict[str, Any]:
    """Load and lightly validate an analysis report JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "flagged" not in data or not isinstance(data["flagged"], list):
        raise ValueError(f"Report missing 'flagged' list: {path}")
    return data


def flagged_intervals(report: dict[str, Any]) -> list[Interval]:
    """Extract unsafe intervals from report flagged scenes."""
    intervals: list[Interval] = []
    for item in report["flagged"]:
        start = timestamp_to_seconds(str(item["start"]))
        end = timestamp_to_seconds(str(item["end"]))
        if end <= start:
            continue
        intervals.append(Interval(start=start, end=end))
    return intervals


def merge_intervals(intervals: list[Interval], join_gap: float = 0.0) -> list[Interval]:
    """Merge overlapping / near-adjacent intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: (i.start, i.end))
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start <= last.end + join_gap:
            merged[-1] = Interval(start=last.start, end=max(last.end, current.end))
        else:
            merged.append(current)
    return merged


def invert_intervals(
    cut: list[Interval],
    video_duration: float,
    min_keep: float = 0.05,
) -> list[Interval]:
    """Return keep-segments = full timeline minus cut intervals."""
    if video_duration <= 0:
        raise ValueError("Video duration must be positive")

    keep: list[Interval] = []
    cursor = 0.0
    for interval in cut:
        start = max(0.0, min(interval.start, video_duration))
        end = max(0.0, min(interval.end, video_duration))
        if start > cursor + min_keep:
            keep.append(Interval(start=cursor, end=start))
        cursor = max(cursor, end)
    if video_duration > cursor + min_keep:
        keep.append(Interval(start=cursor, end=video_duration))
    return keep


def probe_duration(video: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {video}: {exc}") from exc
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe: {out!r}") from exc


def has_audio_stream(video: Path) -> bool:
    """Return True if the input has at least one audio stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return bool(out)


def format_seconds(seconds: float) -> str:
    return seconds_to_timestamp(seconds)


def build_filter_complex(keep: list[Interval], with_audio: bool) -> tuple[str, str]:
    """Build ffmpeg filter_complex and -map labels for keep segments."""
    if not keep:
        raise ValueError("No keep segments remain after removing unsafe scenes")

    parts: list[str] = []
    concat_inputs: list[str] = []
    for idx, segment in enumerate(keep):
        parts.append(
            f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},"
            f"setpts=PTS-STARTPTS[v{idx}]"
        )
        if with_audio:
            parts.append(
                f"[0:a]atrim=start={segment.start:.3f}:end={segment.end:.3f},"
                f"asetpts=PTS-STARTPTS[a{idx}]"
            )
            concat_inputs.append(f"[v{idx}][a{idx}]")
        else:
            concat_inputs.append(f"[v{idx}]")

    n = len(keep)
    if with_audio:
        parts.append("".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]")
        return ";".join(parts), "[outv] [outa]"
    parts.append("".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]")
    return ";".join(parts), "[outv]"


def run_ffmpeg_cut(
    video: Path,
    output: Path,
    keep: list[Interval],
    *,
    with_audio: bool,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Encode a cleaned video containing only keep segments."""
    filter_complex, maps = build_filter_complex(keep, with_audio=with_audio)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(video),
        "-filter_complex",
        filter_complex,
    ]
    map_labels = maps.split()
    for label in map_labels:
        cmd.extend(["-map", label])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
        ]
    )
    if with_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")
    cmd.append(str(output))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def write_edl(path: Path, cut: list[Interval], keep: list[Interval]) -> None:
    """Write a simple text edit decision list for inspection."""
    lines = ["# Cut intervals (removed)", ""]
    for i, interval in enumerate(cut, start=1):
        lines.append(
            f"CUT {i:03d}: {format_seconds(interval.start)} --> "
            f"{format_seconds(interval.end)}  ({interval.duration:.2f}s)"
        )
    lines.extend(["", "# Keep intervals (retained)", ""])
    for i, interval in enumerate(keep, start=1):
        lines.append(
            f"KEEP {i:03d}: {format_seconds(interval.start)} --> "
            f"{format_seconds(interval.end)}  ({interval.duration:.2f}s)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove unsafe scenes from a video using a flagged-scene report "
            "from analyze_transcript.py."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the source video file provided by the user",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        required=True,
        help="Path to the analysis report JSON from analyze_transcript.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output cleaned video path (default: <video>_clean.mp4)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Extra seconds to cut before/after each flagged scene",
    )
    parser.add_argument(
        "--join-gap",
        type=float,
        default=0.25,
        help="Merge cut ranges closer than this many seconds",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF quality (lower = better, default 18)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset (ultrafast..veryslow, default medium)",
    )
    parser.add_argument(
        "--edl",
        type=Path,
        default=None,
        help="Optional path to write a human-readable cut/keep list",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cuts without writing a video",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("Error: ffmpeg/ffprobe are required but not found in PATH.", file=sys.stderr)
        return 1

    video: Path = args.video
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1
    if not args.report.is_file():
        print(f"Error: report not found: {args.report}", file=sys.stderr)
        return 1

    output = args.output or video.with_name(f"{video.stem}_clean{video.suffix or '.mp4'}")
    if output.suffix == "":
        output = output.with_suffix(".mp4")

    try:
        report = load_report(args.report)
        duration = probe_duration(video)
        audio = has_audio_stream(video)

        raw_cuts = flagged_intervals(report)
        padded = [c.expand(args.pad, video_duration=duration) for c in raw_cuts]
        cuts = merge_intervals(padded, join_gap=args.join_gap)
        keep = invert_intervals(cuts, video_duration=duration)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
        print(f"Error preparing cut plan: {exc}", file=sys.stderr)
        return 1

    removed = sum(c.duration for c in cuts)
    kept = sum(k.duration for k in keep)

    print("=" * 64)
    print("REMOVE UNSAFE SCENES")
    print("=" * 64)
    print(f"Video     : {video}")
    print(f"Report    : {args.report}")
    print(f"Output    : {output}")
    print(f"Duration  : {format_seconds(duration)} ({duration / 60:.2f} min)")
    print(f"Flagged   : {len(report['flagged'])} scenes -> {len(cuts)} cut ranges")
    print(f"Removing  : {removed:.1f}s ({removed / 60:.2f} min)")
    print(f"Keeping   : {kept:.1f}s ({kept / 60:.2f} min)")
    print(f"Audio     : {'yes' if audio else 'no'}")
    print()
    for i, interval in enumerate(cuts, start=1):
        print(
            f"  CUT {i:03d}: {format_seconds(interval.start)} --> "
            f"{format_seconds(interval.end)}  ({interval.duration:.1f}s)"
        )

    if args.edl:
        write_edl(args.edl, cuts, keep)
        print(f"\nEDL written to {args.edl}")

    if args.dry_run:
        print("\nDry run only — no video written.")
        return 0

    if not keep:
        print("Error: nothing left to keep after cuts.", file=sys.stderr)
        return 1

    print("\nEncoding cleaned video with ffmpeg...")
    try:
        # Work in a temp dir so partial outputs are not left as final names
        with tempfile.TemporaryDirectory(prefix="amanah_") as tmp:
            tmp_out = Path(tmp) / output.name
            run_ffmpeg_cut(
                video,
                tmp_out,
                keep,
                with_audio=audio,
                crf=args.crf,
                preset=args.preset,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_out), str(output))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Cleaned video saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

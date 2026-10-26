#!/usr/bin/env python3
"""Analyze movie transcripts for child-inappropriate content via Ollama (qwen).

Flags scenes that are unsuitable for children under Islamic ethics, including
sexual dialogue, vulgarity, shirk, witchcraft, divination, intoxicants, and
gambling. Uses Ollama's local REST API with strict JSON output enforcement.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_LINES_PER_SCENE = 8
DEFAULT_GAP_SECONDS = 8.0

SYSTEM_INSTRUCTIONS = """\
You are a content-moderation assistant evaluating movie dialogue for family \
suitability according to Islamic ethics for children.

Flag the whole scene as UNSAFE (safe=false) if ANY line contains one of these. \
A joke, a song, or plot dialogue still counts.

- Sexual or romantic content: sexual dialogue, innuendo, kissing, honeymoon, \
  intimate implications, or objectifying body language or lyrics.
- Dating: courtship, a boyfriend or girlfriend, a crush, asking someone out, \
  a breeding program, or lines about meeting somebody, settling down, or having \
  a romantic relationship. Friendship and family bonds are safe.
- Vulgarity: profanity, crude insults, or bathroom humor meant as crude.
- Immodesty: suggestive dancing lyrics, or "hot" and other body-focused lines.
- Shirk: worship, prayer, sacrifice, offerings, or thanks directed to any god, \
  idol, or being other than Allah. This includes "gods" as deities and calling \
  created things gods, such as water gods or volcano gods. The English word \
  "God" means Allah. Never flag it as shirk.
- Witchcraft (sihr): witch, witch doctor, spell, curse, sorcery, or magic used \
  as supernatural or occult power. Call this witchcraft, not shirk. Do not flag \
  "magic" when it only means wonderful or delightful.
- Divination: fortune teller, horoscope, palm reading, omens used to know the \
  future, or any claim to know the unseen.
- Seeking help from creation: asking jinn, the dead, or charms and amulets for \
  protection or aid.
- Misuse of Allah's name: swearing by something other than Allah; a throwaway \
  exclamation such as "God, do you feel it?"; mocking Allah, the prophets, the \
  Quran, prayer, or angels. A respectful mention is safe.
- Intoxicants: drinking or offering alcohol or drugs, being drunk, or a bar, \
  wine, beer, or cocktail. The line does not need to glorify it. Do not flag \
  ordinary food or non-alcoholic drinks, such as water, juice, sushi, sugar, \
  or vinegar.
- Gambling: a real wager, casino, poker for stakes, or a lottery. Do not flag \
  "you bet", "I bet", or "bet you" when they only mean "I'm sure" or "of course".

Do NOT flag mild cartoon fighting, parental discipline, animal comedy, ordinary \
food, generic adventure tension, or performance idioms such as "knocked them \
dead" and "killed it". Those exceptions never cover the categories above.

Respond with ONLY a single JSON object matching this schema exactly:
{"safe": <boolean>, "reason": "<short explanation>"}
If safe is true, reason must be a brief confirmation (e.g. "No inappropriate content.").
If safe is false, reason must name the category and the offending words.
"""


@dataclass(frozen=True)
class Cue:
    """A single timed subtitle cue."""

    index: int
    start: str
    end: str
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Scene:
    """A chunk of consecutive cues treated as one analysis unit."""

    scene_number: int
    start: str
    end: str
    text: str
    cue_indices: tuple[int, ...]


@dataclass
class SceneVerdict:
    """Model verdict for one scene."""

    scene_number: int
    start: str
    end: str
    text_preview: str
    safe: bool
    reason: str
    raw_response: str = ""
    parse_error: Optional[str] = None


@dataclass
class AnalysisReport:
    """Aggregate report for a transcript run."""

    source: str
    model: str
    total_scenes: int
    flagged: list[SceneVerdict] = field(default_factory=list)
    all_verdicts: list[SceneVerdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "total_scenes": self.total_scenes,
            "flagged_count": len(self.flagged),
            "flagged": [asdict(v) for v in self.flagged],
            "errors": self.errors,
        }


def timestamp_to_seconds(ts: str) -> float:
    """Convert SRT/VTT-like timestamps to seconds."""
    ts = ts.strip().replace(",", ".")
    # Support H:MM:SS.mmm, HH:MM:SS.mmm, MM:SS.mmm
    parts = ts.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unrecognized timestamp: {ts!r}")


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:06.3f}"


def _clean_cue_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    # Unescape common markdown escapes from scraped Subsaga pages
    text = re.sub(r"\\([#*_`\[\]])", r"\1", text)
    text = text.replace("\\-", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt(content: str) -> list[Cue]:
    """Parse SubRip (.srt) content into cues."""
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    cues: list[Cue] = []
    time_re = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
    )
    for block in blocks:
        lines = [ln.strip("\ufeff") for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # Optional numeric index on first line
        if re.fullmatch(r"\d+", lines[0]):
            time_line = lines[1]
            text_lines = lines[2:]
            index = int(lines[0])
        else:
            time_line = lines[0]
            text_lines = lines[1:]
            index = len(cues) + 1
        match = time_re.search(time_line)
        if not match or not text_lines:
            continue
        start, end = match.group(1), match.group(2)
        text = _clean_cue_text(" ".join(text_lines))
        if not text:
            continue
        cues.append(
            Cue(
                index=index,
                start=start.replace(",", "."),
                end=end.replace(",", "."),
                text=text,
                start_seconds=timestamp_to_seconds(start),
                end_seconds=timestamp_to_seconds(end),
            )
        )
    return cues


def parse_vtt(content: str) -> list[Cue]:
    """Parse WebVTT (.vtt) content into cues."""
    body = content.replace("\r\n", "\n")
    body = re.sub(r"^WEBVTT.*?\n", "", body, count=1, flags=re.IGNORECASE)
    # Drop NOTE / STYLE blocks roughly by splitting on blank lines
    blocks = re.split(r"\n\s*\n", body.strip())
    cues: list[Cue] = []
    time_re = re.compile(
        r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})"
    )
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() and not ln.startswith("NOTE")]
        if not lines:
            continue
        time_idx = 0
        if "-->" not in lines[0] and len(lines) > 1:
            time_idx = 1
        if time_idx >= len(lines) or "-->" not in lines[time_idx]:
            continue
        match = time_re.search(lines[time_idx])
        if not match:
            continue
        text = _clean_cue_text(" ".join(lines[time_idx + 1 :]))
        if not text:
            continue
        start, end = match.group(1), match.group(2)
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start.replace(",", "."),
                end=end.replace(",", "."),
                text=text,
                start_seconds=timestamp_to_seconds(start),
                end_seconds=timestamp_to_seconds(end),
            )
        )
    return cues


def parse_plain_text(content: str, lines_per_scene: int = DEFAULT_LINES_PER_SCENE) -> list[Cue]:
    """Parse plain dialogue lines into synthetic timed cues."""
    lines = [ln.strip() for ln in content.replace("\r\n", "\n").split("\n") if ln.strip()]
    cues: list[Cue] = []
    for i, line in enumerate(lines, start=1):
        start_s = float((i - 1) * 3)
        end_s = start_s + 2.5
        cues.append(
            Cue(
                index=i,
                start=seconds_to_timestamp(start_s),
                end=seconds_to_timestamp(end_s),
                text=_clean_cue_text(line),
                start_seconds=start_s,
                end_seconds=end_s,
            )
        )
    return cues


def parse_markdown_table(content: str) -> list[Cue]:
    """Parse Subsaga-style markdown tables: | Line | From | To |."""
    cues: list[Cue] = []
    row_re = re.compile(
        r"^\|\s*(.*?)\s*\|\s*(\d{1,2}:\d{2}:\d{2})\s*\|\s*(\d{1,2}:\d{2}:\d{2})\s*\|$"
    )
    for raw_line in content.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        match = row_re.match(line)
        if not match:
            continue
        text, start, end = match.group(1), match.group(2), match.group(3)
        # Skip header / separator rows
        if text.lower() in {"line", "---", ":---", "----"} or set(text) <= {"-", ":", " "}:
            continue
        text = _clean_cue_text(text.replace("\\|", "|"))
        if not text:
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start + ".000",
                end=end + ".000",
                text=text,
                start_seconds=timestamp_to_seconds(start),
                end_seconds=timestamp_to_seconds(end),
            )
        )
    return cues


def parse_html_subtitle_table(content: str) -> list[Cue]:
    """Parse Subsaga HTML pages with <tr>...<td><p>text</p></td><td>start</td><td>end</td>."""
    row_re = re.compile(
        r"<tr[^>]*>\s*"
        r"<td[^>]*>.*?</td>\s*"
        r"<td[^>]*>\s*(?:<p[^>]*>)?(.*?)(?:</p>)?\s*</td>\s*"
        r"<td[^>]*>\s*(\d{1,2}:\d{2}:\d{2})\s*</td>\s*"
        r"<td[^>]*>\s*(\d{1,2}:\d{2}:\d{2})\s*</td>\s*"
        r"</tr>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    cues: list[Cue] = []
    for match in row_re.finditer(content):
        raw_text, start, end = match.group(1), match.group(2), match.group(3)
        text = _clean_cue_text(re.sub(r"<[^>]+>", "", raw_text))
        if not text or text.lower() == "line":
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start + ".000",
                end=end + ".000",
                text=text,
                start_seconds=timestamp_to_seconds(start),
                end_seconds=timestamp_to_seconds(end),
            )
        )
    return cues


def parse_transcript(content: str, filename_hint: str = "") -> list[Cue]:
    """Detect format and parse transcript content into cues."""
    stripped = content.lstrip("\ufeff").strip()
    lower_name = filename_hint.lower()

    # Subsaga HTML pages (URL often ends in .srt but returns HTML)
    if "<html" in stripped[:500].lower() or "subtitles-style" in stripped:
        cues = parse_html_subtitle_table(stripped)
        if cues:
            return cues

    if lower_name.endswith(".vtt") or stripped.upper().startswith("WEBVTT"):
        cues = parse_vtt(stripped)
        if cues:
            return cues

    if lower_name.endswith(".srt") or re.search(
        r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}",
        stripped,
    ):
        cues = parse_srt(stripped)
        if cues:
            return cues

    # Subsaga / markdown table pages
    if "| Line |" in stripped or re.search(
        r"\|\s*.+?\s*\|\s*\d{1,2}:\d{2}:\d{2}\s*\|\s*\d{1,2}:\d{2}:\d{2}\s*\|",
        stripped,
    ):
        cues = parse_markdown_table(stripped)
        if cues:
            return cues

    return parse_plain_text(stripped)


def chunk_cues_into_scenes(
    cues: Iterable[Cue],
    lines_per_scene: int = DEFAULT_LINES_PER_SCENE,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[Scene]:
    """Group cues into scenes by max line count and silence gaps."""
    cue_list = list(cues)
    if not cue_list:
        return []

    scenes: list[Scene] = []
    bucket: list[Cue] = []

    def flush() -> None:
        nonlocal bucket
        if not bucket:
            return
        scenes.append(
            Scene(
                scene_number=len(scenes) + 1,
                start=bucket[0].start,
                end=bucket[-1].end,
                text=" ".join(c.text for c in bucket),
                cue_indices=tuple(c.index for c in bucket),
            )
        )
        bucket = []

    for cue in cue_list:
        if bucket:
            gap = cue.start_seconds - bucket[-1].end_seconds
            if gap >= gap_seconds or len(bucket) >= lines_per_scene:
                flush()
        bucket.append(cue)
    flush()
    return scenes


def build_scene_prompt(scene: Scene) -> str:
    """Build the user prompt that enforces JSON-only answers."""
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Scene #{scene.scene_number}\n"
        f"Timestamp: {scene.start} --> {scene.end}\n"
        f"Dialogue:\n{scene.text}\n\n"
        'Return ONLY JSON: {"safe": boolean, "reason": string}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from model output."""
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def query_ollama(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Send a prompt to Ollama's /api/generate endpoint and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Failed to reach Ollama at {api_url}. Is `ollama serve` running? ({exc})"
        ) from exc

    if "error" in body:
        raise RuntimeError(f"Ollama API error: {body['error']}")
    if "response" not in body:
        raise RuntimeError(f"Unexpected Ollama response keys: {list(body)}")
    return str(body["response"])


def analyze_scene(
    scene: Scene,
    *,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SceneVerdict:
    """Analyze one scene and return a structured verdict with parse-error handling."""
    prompt = build_scene_prompt(scene)
    preview = scene.text if len(scene.text) <= 160 else scene.text[:157] + "..."
    raw = ""
    try:
        raw = query_ollama(prompt, model=model, api_url=api_url, timeout=timeout)
        parsed = extract_json_object(raw)
        if "safe" not in parsed:
            raise KeyError("Missing required key 'safe'")
        safe = bool(parsed["safe"])
        reason = str(parsed.get("reason", "")).strip() or (
            "No reason provided." if safe else "Flagged without reason."
        )
        return SceneVerdict(
            scene_number=scene.scene_number,
            start=scene.start,
            end=scene.end,
            text_preview=preview,
            safe=safe,
            reason=reason,
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return SceneVerdict(
            scene_number=scene.scene_number,
            start=scene.start,
            end=scene.end,
            text_preview=preview,
            safe=False,
            reason="Model returned unparseable output; treat as needing review.",
            raw_response=raw,
            parse_error=str(exc),
        )


def analyze_transcript(
    content: str,
    *,
    source: str = "<memory>",
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_OLLAMA_URL,
    lines_per_scene: int = DEFAULT_LINES_PER_SCENE,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    limit_scenes: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    progress: bool = True,
) -> AnalysisReport:
    """Parse a transcript, analyze scenes sequentially, and return a report."""
    cues = parse_transcript(content, filename_hint=source)
    scenes = chunk_cues_into_scenes(
        cues, lines_per_scene=lines_per_scene, gap_seconds=gap_seconds
    )
    if limit_scenes is not None:
        scenes = scenes[: max(0, limit_scenes)]

    report = AnalysisReport(source=source, model=model, total_scenes=len(scenes))
    if not scenes:
        report.errors.append("No subtitle cues found in transcript.")
        return report

    for i, scene in enumerate(scenes, start=1):
        if progress:
            print(
                f"[{i}/{len(scenes)}] Analyzing scene {scene.scene_number} "
                f"({scene.start} --> {scene.end})...",
                file=sys.stderr,
            )
        try:
            verdict = analyze_scene(
                scene, model=model, api_url=api_url, timeout=timeout
            )
        except (ConnectionError, RuntimeError, TimeoutError) as exc:
            report.errors.append(f"Scene {scene.scene_number}: {exc}")
            verdict = SceneVerdict(
                scene_number=scene.scene_number,
                start=scene.start,
                end=scene.end,
                text_preview=scene.text[:160],
                safe=False,
                reason=f"Analysis failed: {exc}",
                parse_error=str(exc),
            )
        report.all_verdicts.append(verdict)
        if not verdict.safe or verdict.parse_error:
            report.flagged.append(verdict)
    return report


def load_transcript(path_or_url: str) -> tuple[str, str]:
    """Load transcript text from a local path or HTTP(S) URL."""
    if re.match(r"^https?://", path_or_url, flags=re.IGNORECASE):
        request = urllib.request.Request(
            path_or_url,
            headers={"User-Agent": "amanah/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8", errors="replace")
        return content, path_or_url

    path = Path(path_or_url)
    content = path.read_text(encoding="utf-8")
    return content, str(path)


def format_report(report: AnalysisReport) -> str:
    """Pretty-print a human-readable analysis report."""
    lines = [
        "=" * 72,
        "CONTENT SAFETY REPORT (Islamic ethics / children)",
        "=" * 72,
        f"Source : {report.source}",
        f"Model  : {report.model}",
        f"Scenes : {report.total_scenes}",
        f"Flagged: {len(report.flagged)}",
        "",
    ]
    if report.errors:
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")
        lines.append("")

    if not report.flagged:
        lines.append("No flagged scenes.")
    else:
        lines.append("Flagged scenes:")
        for v in report.flagged:
            lines.extend(
                [
                    "-" * 72,
                    f"Scene #{v.scene_number}  [{v.start} --> {v.end}]",
                    f"Preview : {v.text_preview}",
                    f"Reason  : {v.reason}",
                ]
            )
            if v.parse_error:
                lines.append(f"Parse   : {v.parse_error}")
        lines.append("-" * 72)
    return "\n".join(lines)


def cues_to_srt(cues: list[Cue]) -> str:
    """Serialize cues back to SubRip format."""
    parts: list[str] = []
    for cue in cues:
        start = cue.start.replace(".", ",")
        end = cue.end.replace(".", ",")
        # Ensure milliseconds present
        if "," not in start:
            start += ",000"
        if "," not in end:
            end += ",000"
        parts.append(f"{cue.index}\n{start} --> {end}\n{cue.text}\n")
    return "\n".join(parts)


SAMPLE_TRANSCRIPT = """1
00:00:01,000 --> 00:00:04,000
Welcome to the jungle adventure!

2
00:00:05,000 --> 00:00:08,000
Look at that beautiful waterfall.

3
00:00:20,000 --> 00:00:25,000
That girl has a hot body, shake it baby!

4
00:00:26,000 --> 00:00:30,000
Let's go find the treasure map.

5
00:00:40,000 --> 00:00:45,000
Damn it, get your damn hands off me, you bastard!

6
00:00:46,000 --> 00:00:50,000
We should rest and eat some fruit.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze movie transcripts with Ollama/qwen for content "
            "inappropriate for children under Islamic ethics."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path or URL to an SRT/VTT/TXT/markdown transcript",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate endpoint (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--lines-per-scene",
        type=int,
        default=DEFAULT_LINES_PER_SCENE,
        help="Max subtitle lines per scene chunk",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        help="Silence gap (seconds) that starts a new scene",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Only analyze the first N scenes (useful for smoke tests)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--export-srt",
        default=None,
        help="If source is markdown/table, also export normalized .srt here",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against built-in sample transcript data",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scene progress on stderr",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo or args.source is None:
        content, source = SAMPLE_TRANSCRIPT, "<sample>"
        if args.source is None and not args.demo:
            print(
                "No source provided; running built-in sample demo. "
                "Pass a file/URL or --demo explicitly.\n",
                file=sys.stderr,
            )
    else:
        try:
            content, source = load_transcript(args.source)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            print(f"Failed to load transcript: {exc}", file=sys.stderr)
            return 1

    cues = parse_transcript(content, filename_hint=source)
    if args.export_srt:
        Path(args.export_srt).write_text(cues_to_srt(cues), encoding="utf-8")
        print(f"Exported {len(cues)} cues to {args.export_srt}", file=sys.stderr)

    started = time.time()
    report = analyze_transcript(
        content,
        source=source,
        model=args.model,
        api_url=args.api_url,
        lines_per_scene=args.lines_per_scene,
        gap_seconds=args.gap_seconds,
        limit_scenes=args.limit_scenes,
        progress=not args.quiet,
    )
    elapsed = time.time() - started

    print(format_report(report))
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report written to {args.output}", file=sys.stderr)

    return 0 if not report.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

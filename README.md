# Amanah

Amanah is a local tool for a film you already have. It transcribes the English dialogue, reviews each scene for family suitability, and writes a shorter cut with the flagged scenes removed. Audio, transcripts, and the finished video stay on your machine.

The name is أمانة: the trust of choosing what a family watches.

## What it does

1. **Transcribe.** `ffmpeg` pulls the audio and a local Whisper model writes an English subtitle file. On a Mac the engine is [mlx-whisper](https://github.com/ml-explore/mlx-examples). On Windows it is [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Both write the English words that were spoken.
2. **Review.** A local [Ollama](https://ollama.com) model reads the dialogue in short scenes and marks lines that are unsuitable for children. The guidelines cover sexual or romantic content, dating, vulgarity, immodesty, shirk, witchcraft, divination, seeking help from creation, misuse of Allah's name, intoxicants, and gambling.
3. **Cut.** `ffmpeg` keeps the remaining intervals and writes `<name>_clean.mp4`, plus a JSON report and an edit list.

Mild cartoon action, ordinary food, family bonds, and the English word "God" (meaning Allah) are left in. Friendship is not treated as dating.

## Requirements

- Python 3.11 or newer
- [ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on your `PATH`)
- [Ollama](https://ollama.com), with a model pulled locally
- A speech engine chosen for the operating system:
  - **Mac (Apple Silicon):** mlx-whisper, model `mlx-community/whisper-small-mlx`
  - **Windows:** faster-whisper, model `small`. An NVIDIA GPU is used when CUDA is available. Otherwise transcription runs on the CPU.

### Mac

```bash
brew install ffmpeg
brew install ollama
ollama pull qwen2.5:7b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

Install [Ollama for Windows](https://ollama.com/download) and ffmpeg, then pull the review model:

```powershell
winget install Gyan.FFmpeg
ollama pull qwen2.5:7b
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pip install -r requirements.txt` installs mlx-whisper on macOS and faster-whisper on Windows. If faster-whisper fails to import, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

To install the `amanah` command into the virtual environment, on either system:

```bash
pip install -e .
```

## Run

On a Mac:

```bash
python3 amanah.py "/path/to/film.mp4"
```

On Windows:

```powershell
python amanah.py "C:\Videos\film.mp4"
```

Or, after `pip install -e .`:

```bash
amanah "/path/to/film.mp4"
```

Amanah creates a folder next to the film named `<film>_amanah/` with the subtitle, plain transcript, and report. The cleaned video is `<film>_clean.mp4` beside the original.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--skip-transcribe` | Reuse an existing subtitle file |
| `--skip-analyze` | Reuse an existing report |
| `--skip-cut` | Stop after the report |
| `--limit-scenes N` | Review only the first N scenes |
| `--whisper-model` | Speech model. Mac default: `mlx-community/whisper-small-mlx`. Windows default: `small` |
| `--llm-model` | Ollama model (default `qwen2.5:7b`) |
| `--pad 0.25` | Extra seconds removed around each flagged scene |
| `-o path.mp4` | Where to write the cleaned video |

## Run one step

```bash
python3 transcribe_audio.py "/path/to/film.mp4" -o film.srt
python3 analyze_transcript.py film.srt -o film_report.json
python3 remove_unsafe_scenes.py "/path/to/film.mp4" -r film_report.json --dry-run
python3 remove_unsafe_scenes.py "/path/to/film.mp4" -r film_report.json -o film_clean.mp4
```

`analyze_transcript.py --demo` runs the reviewer on a short built-in transcript so you can check that Ollama is running before you process a full film.

`examples/sample_report.json` is the shape of a report. It is sample text, not a real film.

## What is not in this repository

Films, cleaned videos, subtitles, reports, and logs are ignored by git. Bring your own copy of a film you have the right to edit for private family viewing.

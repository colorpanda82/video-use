"""Local-Whisper transcription fallback, drop-in for the ElevenLabs Scribe JSON.

Extracts mono 16kHz audio (shared with transcribe.py), runs the openai-whisper
CLI with word timestamps, and reshapes its output into the exact Scribe schema
that pack_transcripts.py and the other helpers expect:

    {"language_code": str, "text": str, "words": [
        {"type": "word",    "text": str, "start": float, "end": float, "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ", "start": float, "end": float, "speaker_id": "speaker_0"},
        ...
    ]}

Whisper has no diarization, so every entry is "speaker_0". "spacing" entries sit
between consecutive words and carry the real silence gap (prev word end -> next
word start) — that timing is what pack_transcripts uses to break phrases. No
"audio_event" entries are emitted (whisper can't produce them; downstream
tolerates their absence).

The default model is `small` (the English sweet spot per house convention),
overridable via the VIDEO_USE_WHISPER_MODEL env var.

Usage:
    python helpers/transcribe_whisper.py <video_path>
    python helpers/transcribe_whisper.py <video_path> --model base --language en
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from transcribe import extract_audio, transcript_path


DEFAULT_WHISPER_BIN = "/opt/homebrew/bin/whisper"
DEFAULT_MODEL = "small"


def resolve_whisper_bin() -> str:
    """Prefer whisper on PATH, fall back to the homebrew location."""
    return shutil.which("whisper") or DEFAULT_WHISPER_BIN


def resolve_model(model: str | None) -> str:
    return model or os.environ.get("VIDEO_USE_WHISPER_MODEL") or DEFAULT_MODEL


def whisper_json_to_scribe(whisper_obj: dict) -> dict:
    """Reshape an openai-whisper JSON dict into the Scribe drop-in schema.

    Pure function — no I/O — so it is unit-testable without running whisper.
    Flattens segments[].words[] in order, inserting a 'spacing' entry whenever a
    real gap exists between a word's end and the next word's start. Words missing
    start/end timestamps are skipped.
    """
    words_out: list[dict] = []
    prev_end: float | None = None

    for segment in whisper_obj.get("segments", []) or []:
        for w in segment.get("words", []) or []:
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            ws = float(ws)
            we = float(we)

            if prev_end is not None and ws > prev_end:
                words_out.append({
                    "type": "spacing",
                    "text": " ",
                    "start": prev_end,
                    "end": ws,
                    "speaker_id": "speaker_0",
                })

            words_out.append({
                "type": "word",
                "text": (w.get("word") or "").strip(),
                "start": ws,
                "end": we,
                "speaker_id": "speaker_0",
            })
            prev_end = we

    return {
        "language_code": whisper_obj.get("language", "en"),
        "text": (whisper_obj.get("text") or "").strip(),
        "words": words_out,
    }


def run_whisper(audio_path: Path, out_dir: Path, model: str, language: str | None) -> dict:
    """Run the whisper CLI and return its parsed JSON output."""
    whisper_bin = resolve_whisper_bin()
    cmd = [
        whisper_bin, str(audio_path),
        "--model", model,
        "--output_format", "json",
        "--word_timestamps", "True",
        "--output_dir", str(out_dir),
    ]
    if language:
        cmd += ["--language", language]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"whisper failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:500]}"
        )

    out_json = out_dir / f"{audio_path.stem}.json"
    if not out_json.exists():
        raise RuntimeError(f"whisper produced no JSON at {out_json}")
    return json.loads(out_json.read_text())


def transcribe_one_whisper(
    video: Path,
    edit_dir: Path,
    *,
    model: str | None = None,
    language: str | None = None,
    verbose: bool = True,
    audio_track: int = 0,
) -> Path:
    """Transcribe a single video with local Whisper. Returns the transcript path.

    Writes the Scribe-compatible JSON to transcript_path(edit_dir, video, audio_track).
    Cached: returns the existing path immediately if the transcript already exists.
    """
    out_path = transcript_path(edit_dir, video, audio_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    model = resolve_model(model)
    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio = tmp_dir / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)
        if verbose:
            size_mb = audio.stat().st_size / (1024 * 1024)
            print(f"  whisper ({model}) on {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
        whisper_obj = run_whisper(audio, tmp_dir, model, language)

    payload = whisper_json_to_scribe(whisper_obj)
    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {sum(1 for w in payload['words'] if w['type'] == 'word')}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with local Whisper (Scribe-compatible JSON)")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Whisper model (default: $VIDEO_USE_WHISPER_MODEL or '{DEFAULT_MODEL}')",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    transcribe_one_whisper(
        video=video,
        edit_dir=edit_dir,
        model=args.model,
        language=args.language,
    )


if __name__ == "__main__":
    main()

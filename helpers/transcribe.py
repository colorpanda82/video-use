"""Transcribe a video with ElevenLabs Scribe.

Extracts mono 16kHz audio via ffmpeg, uploads to Scribe with verbatim +
diarize + audio events + word-level timestamps, writes the full response
to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the upload is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ScribeHTTPError(RuntimeError):
    """Raised when Scribe returns a non-200. Carries the HTTP status code so
    callers can distinguish auth/payment/quota (401/402/429) from other errors."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def load_api_key(required: bool = False) -> str:
    """Return the ElevenLabs API key, or "" when none is configured.

    With required=True, exit instead of returning "" (the original behavior,
    kept available for callers that genuinely cannot proceed without Scribe).
    When the key is absent, transcribe_one falls back to local Whisper, so the
    default is non-fatal."""
    # 1) Canonical secrets store (workspace convention): one plain key per file.
    secret_file = Path.home() / ".jarvis-secrets" / "elevenlabs-api-key.txt"
    if secret_file.exists():
        key = secret_file.read_text().strip().strip('"').strip("'")
        if key:
            return key
    # 2) Repo-local .env (ELEVENLABS_API_KEY=...).
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ELEVENLABS_API_KEY":
                    return v.strip().strip('"').strip("'")
    # 3) Environment variable.
    v = os.environ.get("ELEVENLABS_API_KEY", "")
    if v:
        return v
    if required:
        sys.exit(
            "ELEVENLABS_API_KEY not found in ~/.jarvis-secrets/elevenlabs-api-key.txt, .env, or environment"
        )
    return ""


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise ScribeHTTPError(
            resp.status_code, f"Scribe returned {resp.status_code}: {resp.text[:500]}"
        )

    return resp.json()


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    # --- Transcriber routing -------------------------------------------------
    # Use local Whisper when forced (VIDEO_USE_TRANSCRIBER=whisper) or when no
    # ElevenLabs key is configured. Whisper writes the same Scribe-compatible
    # JSON to the same path, so downstream helpers are unaffected.
    forced = os.environ.get("VIDEO_USE_TRANSCRIBER", "").strip().lower()
    if forced == "whisper" or not api_key:
        from transcribe_whisper import transcribe_one_whisper
        if verbose and forced != "whisper":
            print("  no ElevenLabs key available, using local Whisper", flush=True)
        return transcribe_one_whisper(video, edit_dir, language=language, verbose=verbose,
                                          audio_track=audio_track)

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / f"{video.stem}.wav"
            extract_audio(video, audio, audio_track)

            # Uploading silence costs the same as uploading speech and returns
            # nothing, so catch the wrong-track case before paying for it.
            peak = peak_dbfs(audio)
            if peak < -60.0:
                raise RuntimeError(
                    f"track {audio_track + 1} of {video.name} is silent "
                    f"(peak {peak:.1f} dBFS) - not uploading. "
                    + (f"The file has {n_tracks} audio tracks; try --audio-track "
                       + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                       if n_tracks > 1 else "Check the source audio.")
                )

            size_mb = audio.stat().st_size / (1024 * 1024)
            if verbose:
                print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            payload = call_scribe(audio, api_key, language, num_speakers)
    except ScribeHTTPError as e:
        # Auth/payment/quota: fall back to local Whisper. Other HTTP errors raise.
        if e.status_code in (401, 402, 429):
            print(
                f"⚠️ Scribe unavailable (HTTP {e.status_code}), falling back to local Whisper",
                flush=True,
            )
            from transcribe_whisper import transcribe_one_whisper
            return transcribe_one_whisper(video, edit_dir, language=language, verbose=verbose,
                                          audio_track=audio_track)
        raise

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with ElevenLabs Scribe")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
    )


if __name__ == "__main__":
    main()

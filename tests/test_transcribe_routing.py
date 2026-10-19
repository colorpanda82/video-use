"""Routing tests for transcribe_one's Scribe-vs-Whisper fallback.

Deterministic — no real Scribe call, no API key, no video. Whisper and the
Scribe call are stubbed so we can assert *which* path transcribe_one takes
under each trigger condition.

Run:
    uv run python tests/test_transcribe_routing.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))

import transcribe  # noqa: E402
import transcribe_whisper  # noqa: E402


class _Stubs:
    """Install/remove stubs for extract_audio, call_scribe, and the whisper fn."""

    def __init__(self, *, scribe_result=None, scribe_error=None):
        self.scribe_result = scribe_result
        self.scribe_error = scribe_error
        self.whisper_called = False
        self.scribe_called = False
        self._orig = {}

    def _fake_extract_audio(self, video, dest, audio_track=0):
        Path(dest).write_bytes(b"\x00\x00")  # non-empty so .stat() works

    def _fake_call_scribe(self, audio, api_key, language=None, num_speakers=None):
        self.scribe_called = True
        if self.scribe_error is not None:
            raise self.scribe_error
        return self.scribe_result

    def _fake_whisper(self, video, edit_dir, *, model=None, language=None, verbose=True, audio_track=0):
        self.whisper_called = True
        self.whisper_audio_track = audio_track
        out = Path(edit_dir) / "transcripts" / f"{Path(video).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"language_code": "en", "text": "via-whisper", "words": []}))
        return out

    def __enter__(self):
        self._orig["extract_audio"] = transcribe.extract_audio
        self._orig["call_scribe"] = transcribe.call_scribe
        self._orig["whisper"] = transcribe_whisper.transcribe_one_whisper
        self._orig["count_audio_tracks"] = transcribe.count_audio_tracks
        self._orig["peak_dbfs"] = transcribe.peak_dbfs
        # The fake video and 2-byte wav are not real media: one track, audible peak.
        transcribe.count_audio_tracks = lambda video: 1
        transcribe.peak_dbfs = lambda wav: -3.0
        transcribe.extract_audio = self._fake_extract_audio
        transcribe.call_scribe = self._fake_call_scribe
        # transcribe_one does a lazy `from transcribe_whisper import ...`, so
        # patching the module attribute is enough.
        transcribe_whisper.transcribe_one_whisper = self._fake_whisper
        return self

    def __exit__(self, *exc):
        transcribe.extract_audio = self._orig["extract_audio"]
        transcribe.call_scribe = self._orig["call_scribe"]
        transcribe_whisper.transcribe_one_whisper = self._orig["whisper"]
        transcribe.count_audio_tracks = self._orig["count_audio_tracks"]
        transcribe.peak_dbfs = self._orig["peak_dbfs"]


def _edit_dir():
    return Path(tempfile.mkdtemp())


def test_forced_whisper_env_uses_whisper_even_with_key():
    os.environ["VIDEO_USE_TRANSCRIBER"] = "whisper"
    try:
        with _Stubs(scribe_result={"words": []}) as s:
            out = transcribe.transcribe_one(
                Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key", verbose=False
            )
        assert s.whisper_called is True
        assert s.scribe_called is False
        assert json.loads(out.read_text())["text"] == "via-whisper"
    finally:
        os.environ.pop("VIDEO_USE_TRANSCRIBER", None)


def test_missing_key_falls_through_to_whisper():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    with _Stubs(scribe_result={"words": []}) as s:
        out = transcribe.transcribe_one(
            Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="", verbose=False
        )
    assert s.whisper_called is True
    assert s.scribe_called is False
    assert json.loads(out.read_text())["text"] == "via-whisper"


def test_scribe_quota_429_falls_back_to_whisper():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    err = transcribe.ScribeHTTPError(429, "Scribe returned 429: quota")
    with _Stubs(scribe_error=err) as s:
        out = transcribe.transcribe_one(
            Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key", verbose=False
        )
    assert s.scribe_called is True
    assert s.whisper_called is True
    assert json.loads(out.read_text())["text"] == "via-whisper"


def test_whisper_fallback_keeps_the_audio_track():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    err = transcribe.ScribeHTTPError(429, "Scribe returned 429: quota")
    with _Stubs(scribe_error=err) as s:
        transcribe.transcribe_one(
            Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key",
            verbose=False, audio_track=1,
        )
    assert s.whisper_audio_track == 1


def test_scribe_auth_401_and_payment_402_fall_back():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    for code in (401, 402):
        err = transcribe.ScribeHTTPError(code, f"Scribe returned {code}")
        with _Stubs(scribe_error=err) as s:
            transcribe.transcribe_one(
                Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key", verbose=False
            )
        assert s.whisper_called is True, f"HTTP {code} should fall back"


def test_scribe_other_error_500_reraises():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    err = transcribe.ScribeHTTPError(500, "Scribe returned 500: boom")
    raised = False
    with _Stubs(scribe_error=err) as s:
        try:
            transcribe.transcribe_one(
                Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key", verbose=False
            )
        except transcribe.ScribeHTTPError as e:
            raised = True
            assert e.status_code == 500
    assert raised is True
    assert s.whisper_called is False


def test_happy_scribe_path_writes_scribe_payload():
    os.environ.pop("VIDEO_USE_TRANSCRIBER", None)
    payload = {"language_code": "en", "text": "via-scribe", "words": [{"type": "word", "text": "hi"}]}
    with _Stubs(scribe_result=payload) as s:
        out = transcribe.transcribe_one(
            Path("/nonexistent/clip.mp4"), _edit_dir(), api_key="real-key", verbose=False
        )
    assert s.scribe_called is True
    assert s.whisper_called is False
    assert json.loads(out.read_text())["text"] == "via-scribe"


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run())

"""Unit tests for the Whisper → Scribe-JSON transform.

Runs without whisper, ffmpeg, or any video: feeds a synthetic openai-whisper
JSON dict into `whisper_json_to_scribe` and checks the drop-in Scribe schema.

Run:
    uv run python tests/test_whisper_transform.py
(also discoverable by pytest as test_* functions if pytest is installed)
"""

from __future__ import annotations

import sys
from pathlib import Path

# helpers/ uses bare imports (e.g. `from transcribe import ...`), so put it on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))

from transcribe_whisper import whisper_json_to_scribe  # noqa: E402


def _sample_whisper() -> dict:
    """2 segments, 4 words, with a real 0.7s gap between 'world' and 'Goodbye'.

    Note leading spaces on the `word` field — whisper's CLI emits them and the
    transform must strip them. The 'Hello'->'world' and 'Goodbye'->'now'
    boundaries touch (no gap) so they must NOT yield spacing entries.
    """
    return {
        "text": "Hello world Goodbye now",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 0.8,
                "text": " Hello world",
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.99},
                    {"word": " world", "start": 0.4, "end": 0.8, "probability": 0.98},
                ],
            },
            {
                "start": 1.5,
                "end": 2.3,
                "text": " Goodbye now",
                "words": [
                    {"word": " Goodbye", "start": 1.5, "end": 2.0, "probability": 0.97},
                    {"word": " now", "start": 2.0, "end": 2.3, "probability": 0.96},
                ],
            },
        ],
    }


def test_top_level_shape():
    out = whisper_json_to_scribe(_sample_whisper())
    assert set(["language_code", "text", "words"]).issubset(out.keys())
    assert out["language_code"] == "en"
    assert out["text"] == "Hello world Goodbye now"
    assert isinstance(out["words"], list)


def test_words_in_order_and_stripped():
    out = whisper_json_to_scribe(_sample_whisper())
    word_texts = [e["text"] for e in out["words"] if e["type"] == "word"]
    assert word_texts == ["Hello", "world", "Goodbye", "now"]
    # leading whisper space is stripped
    assert all(t == t.strip() for t in word_texts)


def test_spacing_entry_carries_gap_timing():
    out = whisper_json_to_scribe(_sample_whisper())
    spacings = [e for e in out["words"] if e["type"] == "spacing"]
    # exactly one real gap: world.end=0.8 -> Goodbye.start=1.5
    assert len(spacings) == 1
    s = spacings[0]
    assert s["text"] == " "
    assert s["start"] == 0.8
    assert s["end"] == 1.5
    # touching boundaries (0.4==0.4, 2.0==2.0) must not create spacing
    assert all(not (e["type"] == "spacing" and e["start"] == e["end"]) for e in out["words"])


def test_all_speaker_zero():
    out = whisper_json_to_scribe(_sample_whisper())
    assert out["words"], "expected non-empty words"
    assert all(e["speaker_id"] == "speaker_0" for e in out["words"])


def test_times_monotonic_non_decreasing():
    out = whisper_json_to_scribe(_sample_whisper())
    prev = 0.0
    for e in out["words"]:
        assert e["start"] >= prev, f"start went backwards at {e}"
        assert e["end"] >= e["start"], f"end before start at {e}"
        prev = e["start"]


def test_spacing_sits_between_words():
    out = whisper_json_to_scribe(_sample_whisper())
    # full ordered sequence: word, word, spacing, word, word
    types = [e["type"] for e in out["words"]]
    assert types == ["word", "word", "spacing", "word", "word"]
    # spacing start == preceding word end, spacing end == following word start
    seq = out["words"]
    sp_idx = types.index("spacing")
    assert seq[sp_idx]["start"] == seq[sp_idx - 1]["end"]
    assert seq[sp_idx]["end"] == seq[sp_idx + 1]["start"]


def test_words_missing_timestamps_are_skipped():
    obj = _sample_whisper()
    # inject a malformed word with no start/end between the two good segments
    obj["segments"][0]["words"].append({"word": " glitch", "probability": 0.1})
    out = whisper_json_to_scribe(obj)
    word_texts = [e["text"] for e in out["words"] if e["type"] == "word"]
    assert "glitch" not in word_texts
    assert word_texts == ["Hello", "world", "Goodbye", "now"]


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

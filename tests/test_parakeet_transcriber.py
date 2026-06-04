"""Tests for the Parakeet (FluidAudio) transcription backend."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


def _words(*specs):
    """Build Word objects from (text, start, end, speaker) tuples."""
    from ownscribe.transcription.models import Word

    return [Word(text=t, start=s, end=e, speaker=spk, score=0.9) for (t, s, e, spk) in specs]


class TestSegmentation:
    def test_splits_on_sentence_punctuation(self):
        from ownscribe.transcription.parakeet_transcriber import _segment_words

        words = _words(
            ("Hello", 0.0, 0.4, None),
            ("there.", 0.4, 0.8, None),
            ("How", 0.9, 1.1, None),
            ("are", 1.1, 1.3, None),
            ("you?", 1.3, 1.6, None),
        )
        segs = _segment_words(words)
        assert [s.text for s in segs] == ["Hello there.", "How are you?"]
        assert segs[0].start == 0.0
        assert segs[0].end == 0.8

    def test_splits_on_speaker_change(self):
        from ownscribe.transcription.parakeet_transcriber import _segment_words

        words = _words(
            ("Hi", 0.0, 0.3, "Speaker 1"),
            ("there", 0.3, 0.6, "Speaker 1"),
            ("yes", 0.7, 1.0, "Speaker 2"),
        )
        segs = _segment_words(words)
        assert len(segs) == 2
        assert segs[0].speaker == "Speaker 1"
        assert segs[1].speaker == "Speaker 2"
        assert segs[1].text == "yes"

    def test_splits_on_long_pause(self):
        from ownscribe.transcription.parakeet_transcriber import _segment_words

        words = _words(
            ("one", 0.0, 0.3, None),
            ("two", 0.3, 0.6, None),
            ("three", 5.0, 5.3, None),  # >1s gap
        )
        segs = _segment_words(words)
        assert len(segs) == 2
        assert segs[0].text == "one two"
        assert segs[1].text == "three"

    def test_splits_on_max_duration(self):
        from ownscribe.transcription.parakeet_transcriber import _MAX_SEGMENT_SECONDS, _segment_words

        # No punctuation, no pauses, no speaker change — only the duration cap can split.
        words = _words(
            ("a", 0.0, 1.0, None),
            ("b", 1.0, 2.0, None),
            ("c", _MAX_SEGMENT_SECONDS + 1.0, _MAX_SEGMENT_SECONDS + 2.0, None),
        )
        segs = _segment_words(words)
        assert len(segs) >= 2

    def test_empty_words(self):
        from ownscribe.transcription.parakeet_transcriber import _segment_words

        assert _segment_words([]) == []


class TestParseWords:
    def test_parses_json_words_without_speakers(self):
        from ownscribe.transcription.parakeet_transcriber import _parse_words

        data = {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.95},
                {"word": "there.", "start": 0.4, "end": 0.8, "confidence": 0.9},
            ]
        }
        words = _parse_words(data)
        assert [w.text for w in words] == ["Hello", "there."]
        assert words[0].score == 0.95
        assert all(w.speaker is None for w in words)  # speakers assigned later via pyannote


class TestAssignSpeakers:
    def test_overlap_assignment_by_midpoint(self):
        from ownscribe.transcription.parakeet_transcriber import _assign_speakers

        words = _words(
            ("Hello", 0.0, 0.4, None),
            ("there", 0.4, 0.8, None),
            ("yes", 13.0, 13.4, None),
        )
        spans = [("SPEAKER_00", 0.0, 12.3), ("SPEAKER_01", 12.3, 18.1)]
        _assign_speakers(words, spans)
        assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]

    def test_word_in_gap_uses_nearest_span(self):
        from ownscribe.transcription.parakeet_transcriber import _assign_speakers

        words = _words(("um", 10.0, 10.2, None))  # falls in the silent gap
        spans = [("SPEAKER_00", 0.0, 5.0), ("SPEAKER_01", 11.0, 20.0)]
        _assign_speakers(words, spans)
        assert words[0].speaker == "SPEAKER_01"  # nearest

    def test_no_spans_leaves_speakers_none(self):
        from ownscribe.transcription.parakeet_transcriber import _assign_speakers

        words = _words(("hi", 0.0, 0.4, None))
        _assign_speakers(words, [])
        assert words[0].speaker is None


class TestBinaryDiscovery:
    def test_missing_binary_exits(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(TranscriptionConfig(), None)
        tx._binary = None
        with pytest.raises(SystemExit):
            tx._require_binary()

    def test_model_version_default_and_override(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        assert ParakeetTranscriber(TranscriptionConfig())._model_version == "v2"
        assert ParakeetTranscriber(TranscriptionConfig(parakeet_model="v3"))._model_version == "v3"


class TestTranscribeFlow:
    def _patch_run(self, tx, payload):
        """Return a side_effect for _run_with_progress that writes payload as JSON."""

        def fake_run(cmd):
            tx._last_cmd = cmd
            Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(payload))
            return []

        return fake_run

    def test_asr_only_command_no_diarize_flag(self, tmp_path):
        # Diarization is handled by pyannote in Python, never via the binary flag.
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(
            TranscriptionConfig(engine="parakeet"),
            DiarizationConfig(enabled=True, hf_token="hf_x"),
        )
        tx._binary = Path("/fake/ownscribe-transcribe")
        payload = {"text": "Hi.", "language": "en", "duration": 1.0,
                   "words": [{"word": "Hi.", "start": 0.0, "end": 0.5, "confidence": 0.9}]}

        with (
            mock.patch.object(tx, "_run_with_progress", side_effect=self._patch_run(tx, payload)),
            mock.patch.object(tx, "_diarize_pyannote", return_value=[("SPEAKER_00", 0.0, 1.0)]) as diar,
        ):
            result = tx.transcribe(tmp_path / "audio.wav")

        assert "--diarize" not in tx._last_cmd
        assert "--model" in tx._last_cmd
        diar.assert_called_once()
        assert result.segments[0].text == "Hi."
        assert result.segments[0].speaker == "SPEAKER_00"

    def test_no_diarization_without_token(self, tmp_path):
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(
            TranscriptionConfig(engine="parakeet"),
            DiarizationConfig(enabled=True, hf_token=""),  # no token -> no pyannote
        )
        tx._binary = Path("/fake/ownscribe-transcribe")
        payload = {"text": "Hi.", "language": "en", "duration": 1.0,
                   "words": [{"word": "Hi.", "start": 0.0, "end": 0.5, "confidence": 0.9}]}

        with (
            mock.patch.object(tx, "_run_with_progress", side_effect=self._patch_run(tx, payload)),
            mock.patch.object(tx, "_diarize_pyannote") as diar,
        ):
            result = tx.transcribe(tmp_path / "audio.wav")

        diar.assert_not_called()
        assert result.segments[0].speaker is None

    def test_transcribe_no_output_exits(self, tmp_path):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(TranscriptionConfig(), None)
        tx._binary = Path("/fake/ownscribe-transcribe")

        # _run_with_progress returns without writing the output file -> failure path.
        with (
            mock.patch.object(tx, "_run_with_progress", return_value=["[ERROR] boom"]),
            pytest.raises(SystemExit),
        ):
            tx.transcribe(tmp_path / "audio.wav")


class TestEngineSelection:
    def test_pipeline_selects_parakeet(self):
        from ownscribe.config import Config
        from ownscribe.pipeline import _create_transcriber
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        config = Config()
        config.transcription.engine = "parakeet"
        assert isinstance(_create_transcriber(config), ParakeetTranscriber)

    def test_pipeline_selects_whisperx(self):
        from ownscribe.config import Config
        from ownscribe.pipeline import _create_transcriber
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        config = Config()
        config.transcription.engine = "whisperx"
        assert isinstance(_create_transcriber(config), WhisperXTranscriber)

    def test_diarization_needs_token_both_engines(self):
        # Both engines diarize via pyannote, which requires an HF token.
        from ownscribe.config import Config
        from ownscribe.pipeline import _diarization_enabled

        for engine in ("parakeet", "whisperx"):
            config = Config()
            config.transcription.engine = engine
            config.diarization.enabled = True
            config.diarization.hf_token = ""
            assert _diarization_enabled(config) is False, engine
            config.diarization.hf_token = "hf_x"
            assert _diarization_enabled(config) is True, engine

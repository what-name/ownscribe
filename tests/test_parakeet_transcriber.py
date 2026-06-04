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


class TestNormalizeSpeaker:
    def test_numeric_ids_become_speaker_labels(self):
        from ownscribe.transcription.parakeet_transcriber import _normalize_speaker

        assert _normalize_speaker("1") == "Speaker 1"
        assert _normalize_speaker("2") == "Speaker 2"
        assert _normalize_speaker("Speaker 1") == "Speaker 1"
        assert _normalize_speaker(None) is None
        assert _normalize_speaker("") is None


class TestBuildResult:
    def test_parses_json_into_transcript_result(self):
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(
            TranscriptionConfig(engine="parakeet"),
            DiarizationConfig(enabled=True),
            progress=None,
        )
        data = {
            "text": "Hello there. Yes.",
            "language": "en",
            "duration": 3.2,
            "diarized": True,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.95, "speaker": "Speaker 1"},
                {"word": "there.", "start": 0.4, "end": 0.8, "confidence": 0.9, "speaker": "Speaker 1"},
                {"word": "Yes.", "start": 1.0, "end": 1.3, "confidence": 0.8, "speaker": "Speaker 2"},
            ],
        }
        result = tx._build_result(data)
        assert result.language == "en"
        assert result.duration == 3.2
        assert result.has_speakers
        assert len(result.segments) == 2
        assert result.segments[0].words[0].score == 0.95
        assert "Hello there." in result.full_text


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
    def test_transcribe_invokes_binary_and_parses_output(self, tmp_path):
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(
            TranscriptionConfig(engine="parakeet"),
            DiarizationConfig(enabled=True),
        )
        tx._binary = Path("/fake/ownscribe-transcribe")

        captured_cmd = {}

        def fake_run(cmd):
            captured_cmd["cmd"] = cmd
            out_idx = cmd.index("--output") + 1
            out_path = Path(cmd[out_idx])
            out_path.write_text(
                json.dumps(
                    {
                        "text": "Hi.",
                        "language": "en",
                        "duration": 1.0,
                        "words": [{"word": "Hi.", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": "Speaker 1"}],
                    }
                )
            )
            return []

        with mock.patch.object(tx, "_run_with_progress", side_effect=fake_run):
            result = tx.transcribe(tmp_path / "audio.wav")

        assert "--diarize" in captured_cmd["cmd"]
        assert "--model" in captured_cmd["cmd"]
        assert result.segments[0].text == "Hi."
        assert result.segments[0].speaker == "Speaker 1"

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

    def test_no_diarize_flag_when_disabled(self, tmp_path):
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.parakeet_transcriber import ParakeetTranscriber

        tx = ParakeetTranscriber(
            TranscriptionConfig(engine="parakeet"),
            DiarizationConfig(enabled=False),
        )
        tx._binary = Path("/fake/ownscribe-transcribe")

        captured_cmd = {}

        def fake_run(cmd):
            captured_cmd["cmd"] = cmd
            out_path = Path(cmd[cmd.index("--output") + 1])
            out_path.write_text(json.dumps({"text": "", "language": "en", "duration": 0.0, "words": []}))
            return []

        with mock.patch.object(tx, "_run_with_progress", side_effect=fake_run):
            tx.transcribe(tmp_path / "audio.wav")

        assert "--diarize" not in captured_cmd["cmd"]


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

    def test_diarization_enabled_parakeet_needs_no_token(self):
        from ownscribe.config import Config
        from ownscribe.pipeline import _diarization_enabled

        config = Config()
        config.transcription.engine = "parakeet"
        config.diarization.enabled = True
        config.diarization.hf_token = ""
        assert _diarization_enabled(config) is True

    def test_diarization_enabled_whisperx_needs_token(self):
        from ownscribe.config import Config
        from ownscribe.pipeline import _diarization_enabled

        config = Config()
        config.transcription.engine = "whisperx"
        config.diarization.enabled = True
        config.diarization.hf_token = ""
        assert _diarization_enabled(config) is False
        config.diarization.hf_token = "hf_x"
        assert _diarization_enabled(config) is True

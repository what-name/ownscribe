"""Parakeet-based transcription via the FluidAudio Swift helper (ownscribe-transcribe).

This backend shells out to the `ownscribe-transcribe` binary (built from
``swift/transcribe/``), which runs Parakeet TDT ASR and, optionally, FluidAudio
speaker diarization fully on-device. The binary writes a JSON file of word-level
timings (with speaker labels); this module parses it into the shared
``TranscriptResult`` model, segmenting the flat word stream into sentence-like
segments on punctuation, speaker changes, and pauses.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from ownscribe.config import DiarizationConfig, TranscriptionConfig
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult, Word

# Stderr markers emitted by the Swift helper (keep in sync with main.swift).
_MARKER_MODEL_LOADING = "[MODEL_LOADING]"
_MARKER_TRANSCRIBING = "[TRANSCRIBING]"
_MARKER_DIARIZING = "[DIARIZING]"
_MARKER_DONE = "[DONE]"

# Segmentation tuning.
_SENTENCE_ENDINGS = (".", "!", "?")
# Parakeet emits punctuation, so sentence endings drive most splits; the pause
# gap only catches genuine silences (kept high to avoid orphaning words across
# natural mid-sentence pauses).
_PAUSE_GAP_SECONDS = 2.0
_MAX_SEGMENT_SECONDS = 30.0  # hard cap so a punctuation-free stream still splits

# Binary discovery — mirror audio/coreaudio.py.
_BINARY_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "bin" / "ownscribe-transcribe",  # dev: repo root
    Path(sys.prefix) / "bin" / "ownscribe-transcribe",
]


def _find_binary() -> Path | None:
    import shutil

    for candidate in _BINARY_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate
    found = shutil.which("ownscribe-transcribe")
    return Path(found) if found else None


class ParakeetTranscriber(Transcriber):
    """Transcribes audio using Parakeet (FluidAudio) via the Swift helper binary."""

    def __init__(
        self,
        transcription_config: TranscriptionConfig,
        diarization_config: DiarizationConfig | None = None,
        progress: NullProgress | None = None,
    ) -> None:
        self._tx_config = transcription_config
        self._diar_config = diarization_config
        self._progress = progress or NullProgress()
        self._binary = _find_binary()

    @property
    def _model_version(self) -> str:
        version = getattr(self._tx_config, "parakeet_model", "v2") or "v2"
        return "v3" if str(version).lower() == "v3" else "v2"

    @property
    def _diarize(self) -> bool:
        return bool(self._diar_config and self._diar_config.enabled)

    def _require_binary(self) -> Path:
        if self._binary is None:
            click.echo(
                "Error: ownscribe-transcribe binary not found.\nBuild it with: bash swift/transcribe/build.sh",
                err=True,
            )
            raise SystemExit(1)
        return self._binary

    def prepare_models(self, language: str | None = None) -> None:
        """Prefetch the Parakeet ASR model by running the helper on a short clip.

        The helper downloads CoreML models inside ``AsrModels.downloadAndLoad``
        before any transcription runs, so even a trivial clip warms the cache.
        Diarization models are fetched lazily on the first real diarized run.
        """
        _ = language
        binary = self._require_binary()
        progress = self._progress
        progress.begin("preparing_models")
        try:
            progress.set_detail("preparing_models", f"Loading Parakeet model ({self._model_version})")
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "warmup.wav"
                _write_warmup_wav(wav)
                out = Path(tmp) / "warmup.json"
                # ASR only — keeps warmup fast; diarization models load on first use.
                subprocess.run(
                    [str(binary), str(wav), "--output", str(out), "--model", self._model_version],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        binary = self._require_binary()
        progress = self._progress

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            cmd = [
                str(binary),
                str(audio_path),
                "--output",
                str(out_path),
                "--model",
                self._model_version,
            ]
            if self._diarize:
                cmd.append("--diarize")

            progress.begin("transcribing")
            progress.set_detail("transcribing", f"Loading Parakeet model ({self._model_version})")

            stderr_lines = self._run_with_progress(cmd)

            if not out_path.exists():
                progress.fail("transcribing")
                detail = "\n".join(stderr_lines).strip()
                click.echo(
                    "Error: transcription failed (no output produced)." + (f"\n{detail}" if detail else ""),
                    err=True,
                )
                raise SystemExit(1)

            data = json.loads(out_path.read_text())

        progress.complete("transcribing")
        return self._build_result(data)

    def _run_with_progress(self, cmd: list[str]) -> list[str]:
        """Run the helper, mapping its stderr markers onto progress detail lines."""
        progress = self._progress
        stderr_lines: list[str] = []
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip("\n")
            if not line:
                continue
            stderr_lines.append(line)
            if line == _MARKER_MODEL_LOADING:
                progress.set_detail("transcribing", f"Loading Parakeet model ({self._model_version})")
            elif line == _MARKER_TRANSCRIBING:
                progress.set_detail("transcribing", "Transcribing")
            elif line == _MARKER_DIARIZING:
                progress.set_detail("transcribing", "Identifying speakers")
            elif line == _MARKER_DONE:
                progress.set_detail("transcribing", None)
        proc.wait()
        return stderr_lines

    def _build_result(self, data: dict) -> TranscriptResult:
        words_raw = data.get("words", [])
        words = [
            Word(
                text=str(w.get("word", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker=_normalize_speaker(w.get("speaker")),
                score=float(w.get("confidence", 0.0)),
            )
            for w in words_raw
        ]
        segments = _segment_words(words)
        return TranscriptResult(
            segments=segments,
            language=str(data.get("language", "")),
            duration=float(data.get("duration", 0.0)),
        )


def _normalize_speaker(speaker: object) -> str | None:
    """Render FluidAudio's bare numeric speaker ids (e.g. "1") as "Speaker 1"."""
    if speaker is None:
        return None
    text = str(speaker).strip()
    if not text:
        return None
    return f"Speaker {text}" if text.isdigit() else text


def _segment_words(words: list[Word]) -> list[Segment]:
    """Group a flat word stream into sentence-like segments.

    Splits whenever the speaker changes, a sentence-ending punctuation mark is
    reached, a pause longer than ``_PAUSE_GAP_SECONDS`` occurs, or the segment
    would exceed ``_MAX_SEGMENT_SECONDS``. This keeps output close to Whisper's
    segment granularity regardless of whether the model emits punctuation.
    """
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(w.text.strip() for w in current if w.text.strip()).strip()
        segments.append(
            Segment(
                text=text,
                start=current[0].start,
                end=current[-1].end,
                speaker=current[0].speaker,
                words=list(current),
            )
        )
        current.clear()

    for word in words:
        if current:
            prev = current[-1]
            speaker_changed = word.speaker != current[0].speaker
            long_pause = word.start - prev.end > _PAUSE_GAP_SECONDS
            too_long = word.end - current[0].start > _MAX_SEGMENT_SECONDS
            if speaker_changed or long_pause or too_long:
                flush()
        current.append(word)
        if word.text.strip().endswith(_SENTENCE_ENDINGS):
            flush()

    flush()
    return segments


def _write_warmup_wav(path: Path) -> None:
    """Write ~1s of low-amplitude noise so the helper can warm the ASR model."""
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(sample_rate) * 1e-3).astype("float32")
    sf.write(path, data, sample_rate, subtype="FLOAT")

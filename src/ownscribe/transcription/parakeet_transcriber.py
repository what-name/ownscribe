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

    def _should_diarize(self) -> bool:
        # Diarization runs via pyannote (better separation than FluidAudio on
        # meeting audio), which needs a HuggingFace token.
        return bool(self._diar_config and self._diar_config.enabled and self._diar_config.hf_token)

    def _diarization_device(self) -> str:
        cfg = self._diar_config.device if self._diar_config else "auto"
        if cfg == "auto":
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        return cfg

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
            # ASR only — diarization is handled separately by pyannote below.
            cmd = [
                str(binary),
                str(audio_path),
                "--output",
                str(out_path),
                "--model",
                self._model_version,
            ]

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

        words = _parse_words(data)
        if self._should_diarize():
            spans = self._diarize_pyannote(audio_path)
            _assign_speakers(words, spans)

        return TranscriptResult(
            segments=_segment_words(words),
            language=str(data.get("language", "")),
            duration=float(data.get("duration", 0.0)),
        )

    def _diarize_pyannote(self, audio_path: Path) -> list[tuple[str, float, float]]:
        """Run pyannote diarization and return (speaker, start, end) spans.

        Calling pyannote directly (rather than through whisperx's word-assignment)
        avoids the phantom-speaker artifacts that layer introduced.
        """
        import contextlib
        import os
        import warnings

        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if not (self._diar_config and self._diar_config.telemetry):
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

        progress = self._progress
        progress.begin("diarizing")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import torch
                import whisperx
                from whisperx.diarize import DiarizationPipeline

                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    pipeline = DiarizationPipeline(
                        token=self._diar_config.hf_token, device=self._diarization_device()
                    )
                    audio = whisperx.load_audio(str(audio_path))
                    audio_data = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": 16000}

                    kwargs: dict = {}
                    if self._diar_config.min_speakers > 0:
                        kwargs["min_speakers"] = self._diar_config.min_speakers
                    if self._diar_config.max_speakers > 0:
                        kwargs["max_speakers"] = self._diar_config.max_speakers
                    hook = getattr(progress, "diarization_hook", None)
                    diarization = pipeline.model(audio_data, hook=hook, **kwargs)

            spans = [
                (str(speaker), float(segment.start), float(segment.end))
                for segment, _label, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
            ]
            progress.complete("diarizing")
            return spans
        except Exception:
            progress.fail("diarizing")
            raise

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

def _parse_words(data: dict) -> list[Word]:
    """Build Word objects from the helper's JSON (speakers assigned separately)."""
    return [
        Word(
            text=str(w.get("word", "")),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
            speaker=None,
            score=float(w.get("confidence", 0.0)),
        )
        for w in data.get("words", [])
    ]


def _assign_speakers(words: list[Word], spans: list[tuple[str, float, float]]) -> None:
    """Label each word with a speaker by overlapping its midpoint with diarization
    spans. Words whose midpoint falls in no span are given the nearest span's speaker.
    Mutates ``words`` in place.
    """
    if not spans:
        return

    def gap(mid: float, span: tuple[str, float, float]) -> float:
        _, start, end = span
        if mid < start:
            return start - mid
        if mid > end:
            return mid - end
        return 0.0

    for word in words:
        mid = (word.start + word.end) / 2.0
        match = next((s for s in spans if s[1] <= mid <= s[2]), None)
        if match is None:
            match = min(spans, key=lambda s: gap(mid, s))
        word.speaker = match[0]


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

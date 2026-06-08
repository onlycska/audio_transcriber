"""Unit tests for TranscribeAudioUseCase."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.service.models.audio import AudioWaveform
from src.service.models.pipeline import PipelineConfig
from src.service.models.segment import (
    DialogueLine,
    DiarizationTurn,
    TranscriptionSegment,
    WordTimestamp,
)
from src.service.transcribe_audio import TranscribeAudioUseCase


class _AudioLoader:
    def load(self, audio_path: Path) -> AudioWaveform:
        return AudioWaveform(samples=np.array([[0.0], [0.1]], dtype=np.float32), sample_rate=16000)


class _Diarizer:
    def diarize(self, audio: AudioWaveform, *, num_speakers: int) -> list[DiarizationTurn]:
        return [
            DiarizationTurn(0.0, 0.5, "raw_0"),
            DiarizationTurn(0.5, 1.0, "raw_1"),
        ]


class _Transcriber:
    def transcribe(self, audio: AudioWaveform, **kwargs: object) -> list[TranscriptionSegment]:
        words = (
            WordTimestamp(0.0, 0.4, "Привет"),
            WordTimestamp(0.5, 0.7, "да"),
            WordTimestamp(0.7, 1.0, "конечно"),
        )
        return [TranscriptionSegment(0.0, 1.0, "Привет да конечно", 0.1, -0.1, 1.0, words)]


class _Writer:
    def __init__(self) -> None:
        self.lines: list[DialogueLine] = []

    def write(self, lines: list[DialogueLine], *, audio_stem: str) -> Path:
        self.lines = lines
        return Path(f"/tmp/{audio_stem}.txt")


def test_execute_positive() -> None:
    """Positive test: use case orchestrates all dependencies and returns output path."""
    # Arrange
    writer = _Writer()
    use_case = TranscribeAudioUseCase(
        audio_loader=_AudioLoader(),
        diarizer=_Diarizer(),
        transcriber=_Transcriber(),
        writer=writer,
    )
    config = PipelineConfig(
        num_speakers=2,
        language="ru",
        initial_prompt="prompt",
        aggressive_dialogue_postprocess=True,
        postprocess_debug=False,
        flicker_max_duration=0.5,
        fallback_max_gap=0.8,
    )

    # Act
    output_path = use_case.execute(audio_path=Path("/tmp/input.wav"), config=config)

    # Assert
    assert (output_path, [line.speaker for line in writer.lines]) == (
        Path("/tmp/input.txt"),
        ["SPEAKER_00", "SPEAKER_01"],
    )

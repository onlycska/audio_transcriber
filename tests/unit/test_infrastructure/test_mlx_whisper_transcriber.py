"""Unit tests for the MLX Whisper transcriber adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.infrastructure.mlx_whisper_transcriber import MlxWhisperTranscriber
from src.service.models.audio import AudioWaveform


def test_transcribe_parses_words_and_metrics_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive test: maps raw payload to segments with stripped word timestamps."""
    # Arrange
    captured: dict[str, Any] = {}

    def _fake_transcribe(audio: Any, **kwargs: Any) -> dict[str, Any]:
        captured["audio"] = audio
        captured["kwargs"] = kwargs
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Привет мир ",
                    "no_speech_prob": 0.2,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.4,
                    "words": [
                        {"word": " Привет", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": " мир", "start": 0.5, "end": 1.0, "probability": 0.8},
                    ],
                },
            ],
        }

    monkeypatch.setattr(
        "src.infrastructure.mlx_whisper_transcriber.mlx_whisper.transcribe",
        _fake_transcribe,
    )
    transcriber = MlxWhisperTranscriber(model_name="mlx-community/whisper-tiny")
    audio = AudioWaveform(
        samples=np.array([[0.0], [0.1], [0.2]], dtype=np.float32),
        sample_rate=16000,
    )

    # Act
    segments = transcriber.transcribe(audio, language="ru", initial_prompt="prompt")

    # Assert
    assert len(segments) == 1
    segment = segments[0]
    assert (
        segment.text,
        segment.no_speech_prob,
        [(word.text, word.start, word.end) for word in segment.words],
    ) == ("Привет мир", 0.2, [("Привет", 0.0, 0.5), ("мир", 0.5, 1.0)])


def test_transcribe_requests_word_timestamps_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive test: forwards mono audio and word_timestamps flag to the backend."""
    # Arrange
    captured: dict[str, Any] = {}

    def _fake_transcribe(audio: Any, **kwargs: Any) -> dict[str, Any]:
        captured["audio"] = audio
        captured["kwargs"] = kwargs
        return {"segments": []}

    monkeypatch.setattr(
        "src.infrastructure.mlx_whisper_transcriber.mlx_whisper.transcribe",
        _fake_transcribe,
    )
    transcriber = MlxWhisperTranscriber(model_name="mlx-community/whisper-tiny")
    audio = AudioWaveform(
        samples=np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        sample_rate=16000,
    )

    # Act
    transcriber.transcribe(audio, language="ru", initial_prompt=None)

    # Assert
    assert captured["audio"].ndim == 1
    assert captured["audio"].tolist() == [2.0, 3.0]
    assert (
        captured["kwargs"]["word_timestamps"],
        captured["kwargs"]["language"],
        captured["kwargs"]["path_or_hf_repo"],
    ) == (True, "ru", "mlx-community/whisper-tiny")


def test_transcribe_rejects_non_16k_audio_negative() -> None:
    """Negative test: raises ValueError when the waveform is not 16 kHz."""
    # Arrange
    transcriber = MlxWhisperTranscriber(model_name="mlx-community/whisper-tiny")
    audio = AudioWaveform(
        samples=np.array([[0.0], [0.1]], dtype=np.float32),
        sample_rate=44100,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="16000 Hz"):
        transcriber.transcribe(audio, language="ru", initial_prompt=None)

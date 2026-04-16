"""Unit tests for ffmpeg-based audio conversion helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.infrastructure.audio_conversion import ensure_wav_for_pipeline


def test_ensure_wav_for_pipeline_returns_input_when_wav_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive test: returns input path when source file is already WAV."""
    # Arrange
    source = tmp_path / "sample.wav"
    source.write_bytes(b"fake")
    called = {"value": False}

    def _fake_run(*args: object, **kwargs: object) -> None:
        called["value"] = True

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act
    result = ensure_wav_for_pipeline(source)

    # Assert
    assert result == source.resolve()
    assert called["value"] is False


def test_ensure_wav_for_pipeline_runs_ffmpeg_for_non_wav_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive test: runs ffmpeg conversion for non-WAV input."""
    # Arrange
    source = tmp_path / "sample.m4a"
    source.write_bytes(b"fake")
    captured: dict[str, object] = {}

    def _fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act
    result = ensure_wav_for_pipeline(source)

    # Assert
    assert result == (tmp_path / "sample_converted.wav").resolve()
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == "ffmpeg"
    assert "-ar" in command and "16000" in command
    assert "-ac" in command and "1" in command
    assert "-af" in command and "highpass=f=200,lowpass=f=3000" in command


def test_ensure_wav_for_pipeline_reuses_cached_output_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive test: uses cached converted WAV when it is up to date."""
    # Arrange
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"src")
    cached = tmp_path / "sample_converted.wav"
    cached.write_bytes(b"dst")
    os.utime(source, (100.0, 100.0))
    os.utime(cached, (200.0, 200.0))
    called = {"value": False}

    def _fake_run(*args: object, **kwargs: object) -> None:
        called["value"] = True

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act
    result = ensure_wav_for_pipeline(source)

    # Assert
    assert result == cached.resolve()
    assert called["value"] is False


def test_ensure_wav_for_pipeline_ffmpeg_missing_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative test: raises RuntimeError when ffmpeg executable is missing."""
    # Arrange
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"src")

    def _fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act & Assert
    with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
        ensure_wav_for_pipeline(source)


def test_ensure_wav_for_pipeline_ffmpeg_failed_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative test: raises RuntimeError and includes ffmpeg stderr."""
    # Arrange
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"src")

    def _fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "ffmpeg", stderr="decode failed")

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act & Assert
    with pytest.raises(RuntimeError, match="decode failed"):
        ensure_wav_for_pipeline(source)


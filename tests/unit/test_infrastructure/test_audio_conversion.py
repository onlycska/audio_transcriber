"""Unit tests for ffmpeg-based audio conversion helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.infrastructure.audio_conversion import (
    ensure_wav_for_pipeline,
    resolve_audio_input_path,
)


def test_resolve_audio_input_path_relative_under_base_positive(tmp_path: Path) -> None:
    """Positive test: relative audio_file resolves under base_dir."""
    # Arrange
    base = tmp_path / "root"
    base.mkdir()
    rel = "audio/rec.wav"

    # Act
    result = resolve_audio_input_path(rel, base_dir=base)

    # Assert
    assert result == (base / "audio" / "rec.wav").resolve()


def test_resolve_audio_input_path_absolute_ignores_base_positive(tmp_path: Path) -> None:
    """Positive test: absolute audio_file ignores base_dir."""
    # Arrange
    audio = tmp_path / "abs" / "a.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x")
    wrong_base = tmp_path / "other"

    # Act
    result = resolve_audio_input_path(str(audio), base_dir=wrong_base)

    # Assert
    assert result == audio.resolve()


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
    assert (result, called["value"]) == (source.resolve(), False)


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
    assert result == (tmp_path / "sample_converted_cf334d4c78.wav").resolve()
    command = captured["command"]
    assert isinstance(command, list)
    assert (
        command[0],
        "-ar" in command,
        "16000" in command,
        "-ac" in command,
        "1" in command,
        "-af" in command,
        "highpass=f=80,lowpass=f=8000" in command,
    ) == ("ffmpeg", True, True, True, True, True, True)


def test_ensure_wav_for_pipeline_reuses_cached_output_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive test: uses cached converted WAV when it is up to date."""
    # Arrange
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"src")
    cached = tmp_path / "sample_converted_5d44848a26.wav"
    cached.write_bytes(b"dst")
    called = {"value": False}

    def _fake_run(*args: object, **kwargs: object) -> None:
        called["value"] = True

    monkeypatch.setattr("src.infrastructure.audio_conversion.subprocess.run", _fake_run)

    # Act
    result = ensure_wav_for_pipeline(source)

    # Assert
    assert (result, called["value"]) == (cached.resolve(), False)


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

"""Unit tests for environment-backed pipeline settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.settings import PipelineSettings


def test_from_env_loads_required_paths_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive test: loads required env vars and resolves results directory."""
    # Arrange
    results_dir = tmp_path / "out"
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("AUDIO_FILE", "audios/input.wav")
    monkeypatch.setenv("RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("NUM_SPEAKERS", "1")

    # Act
    settings = PipelineSettings.from_env()

    # Assert
    assert (
        settings.hf_token,
        settings.audio_file,
        settings.results_dir,
        settings.config.num_speakers,
        settings.config.initial_prompt,
    ) == (
        "hf_test",
        "audios/input.wav",
        results_dir.resolve(),
        1,
        "Разговор 1 человека на русском языке.",
    )


def test_from_env_missing_hf_token_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative test: raises ValueError when HF_TOKEN is missing."""
    # Arrange
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("AUDIO_FILE", "audios/input.wav")

    # Act & Assert
    with pytest.raises(ValueError, match="HF_TOKEN"):
        PipelineSettings.from_env()


def test_from_env_missing_audio_file_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative test: raises ValueError when AUDIO_FILE is missing."""
    # Arrange
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.delenv("AUDIO_FILE", raising=False)

    # Act & Assert
    with pytest.raises(ValueError, match="AUDIO_FILE"):
        PipelineSettings.from_env()

"""Unit tests for PyAnnote diarizer adapter."""

from __future__ import annotations

import pytest

from src.infrastructure.pyannote_diarizer import PyannoteDiarizer


def test_init_pipeline_load_failure_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative test: raises RuntimeError when PyAnnote returns no pipeline."""
    # Arrange
    monkeypatch.setattr(
        "src.infrastructure.pyannote_diarizer.Pipeline.from_pretrained",
        lambda *args, **kwargs: None,
    )

    # Act & Assert
    with pytest.raises(RuntimeError, match="Failed to load PyAnnote pipeline"):
        PyannoteDiarizer(hf_token="hf_test", model_name="missing-model")

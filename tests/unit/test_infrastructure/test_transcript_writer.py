"""Unit tests for file-system transcript writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.infrastructure.transcript_writer import FileTranscriptWriter
from src.service.models.segment import DialogueLine


def test_write_uses_injected_clock_positive(tmp_path: Path) -> None:
    """Positive test: writes deterministic transcript file with injected clock."""
    # Arrange
    writer = FileTranscriptWriter(
        results_dir=tmp_path,
        clock=lambda: datetime(2026, 4, 25, 12, 30, 5),
    )
    lines = [
        DialogueLine(speaker="SPEAKER_00", text="Привет."),
        DialogueLine(speaker="SPEAKER_01", text="Здравствуйте."),
    ]

    # Act
    output_path = writer.write(lines, audio_stem="call")

    # Assert
    assert (output_path.name, output_path.read_text(encoding="utf-8")) == (
        "transcript_call_2026-04-25_12-30-05.txt",
        "[SPEAKER_00] Привет.\n[SPEAKER_01] Здравствуйте.",
    )

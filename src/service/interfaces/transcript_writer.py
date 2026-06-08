"""Service interface for transcript output persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.service.models.segment import DialogueLine


class ITranscriptWriter(Protocol):
    """Writes final dialogue lines to storage."""

    def write(self, lines: list[DialogueLine], *, audio_stem: str) -> Path:
        """Persist transcript and return output path.

        Args:
            lines: Final dialogue lines.
            audio_stem: Source audio file stem.

        Returns:
            Path to persisted transcript.
        """
        ...

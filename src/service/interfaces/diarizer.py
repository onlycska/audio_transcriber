"""Service interface for speaker diarization providers."""

from __future__ import annotations

from typing import Protocol

from src.service.models.audio import AudioWaveform
from src.service.models.segment import DiarizationTurn


class IDiarizer(Protocol):
    """Resolves speaker turns for an input waveform."""

    def diarize(self, audio: AudioWaveform, *, num_speakers: int) -> list[DiarizationTurn]:
        """Run diarization and return speaker turns.

        Args:
            audio: Loaded waveform and sample rate.
            num_speakers: Expected speaker count.

        Returns:
            Ordered speaker turns.
        """
        ...

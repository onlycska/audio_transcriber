"""Service interface for loading raw audio data."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.service.models.audio import AudioWaveform


class IAudioLoader(Protocol):
    """Loads waveform and sample rate from audio file."""

    def load(self, audio_path: Path) -> AudioWaveform:
        """Load audio from file path.

        Args:
            audio_path: Path to audio file.

        Returns:
            Loaded waveform and sample rate.
        """
        ...

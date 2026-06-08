"""Infrastructure audio loader based on soundfile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.service.interfaces.audio_loader import IAudioLoader
from src.service.models.audio import AudioWaveform


class SoundfileAudioLoader(IAudioLoader):
    """Load waveform and sample rate with soundfile."""

    def load(self, audio_path: Path) -> AudioWaveform:
        """Load audio file as float32 waveform.

        Args:
            audio_path: Path to readable audio file.

        Returns:
            Loaded waveform and sample rate.
        """
        waveform_raw, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = np.asarray(waveform_raw, dtype=np.float32)
        return AudioWaveform(samples=waveform, sample_rate=sample_rate)

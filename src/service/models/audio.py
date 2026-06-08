"""Audio-related service models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AudioSamples = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioWaveform:
    """In-memory audio payload used across service helpers.

    Keeps decoded audio and its sampling rate together so service-layer helpers
    can process audio without depending on a concrete loader implementation.

    Attributes:
        samples: Audio samples in shape ``(samples,)`` for mono or
            ``(samples, channels)`` for multichannel waveform.
        sample_rate: Sampling rate in Hz.
    """

    samples: AudioSamples
    sample_rate: int

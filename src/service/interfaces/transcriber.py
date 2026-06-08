"""Service interface for ASR transcribers."""

from __future__ import annotations

from typing import Protocol

from src.service.models.audio import AudioWaveform
from src.service.models.segment import TranscriptionSegment


class ITranscriber(Protocol):
    """Transcribes audio into timed text segments with word-level timestamps."""

    def transcribe(
        self,
        audio: AudioWaveform,
        *,
        language: str,
        initial_prompt: str | None,
    ) -> list[TranscriptionSegment]:
        """Transcribe a full waveform and return ordered segments.

        Implementations should populate
        :attr:`~src.service.models.segment.TranscriptionSegment.words` so that
        downstream speaker assignment can match each word to a diarization turn.

        Args:
            audio: Loaded waveform and sample rate.
            language: Whisper language code.
            initial_prompt: Optional prompt passed to the model.

        Returns:
            Ordered ASR segments, each carrying word-level timestamps.
        """
        ...

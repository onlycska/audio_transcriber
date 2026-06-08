"""MLX-backed Whisper transcriber for Apple Silicon.

Uses ``mlx-whisper`` which runs Whisper on the Apple GPU through the MLX
framework. Unlike ``openai-whisper`` (CPU/fp32 on macOS), this backend is
faster than real time on M-series chips and emits word-level timestamps,
which are required by the word-level speaker assignment step.
"""

from __future__ import annotations

from typing import Any, cast

import mlx_whisper
import numpy as np
from numpy.typing import NDArray

from src.service.interfaces.transcriber import ITranscriber
from src.service.models.audio import AudioWaveform
from src.service.models.segment import TranscriptionSegment, WordTimestamp

_TEMPERATURE_SCHEDULE: tuple[float, ...] = (0.0, 0.2, 0.4)
_EXPECTED_SAMPLE_RATE = 16000


class MlxWhisperTranscriber(ITranscriber):
    """Transcriber adapter over ``mlx-whisper``.

    The whole waveform is transcribed in a single call; ``mlx-whisper`` performs
    its own internal 30-second windowing, so no manual chunking is needed and
    no speaker artifacts appear at chunk boundaries.

    Args:
        model_name: MLX Whisper repository id, for example
            ``"mlx-community/whisper-large-v3-mlx"``.
    """

    def __init__(self, *, model_name: str) -> None:
        """Store the MLX model repository id.

        Args:
            model_name: MLX Whisper repository id passed to
                :func:`mlx_whisper.transcribe`.
        """
        self._model_name = model_name

    def transcribe(
        self,
        audio: AudioWaveform,
        *,
        language: str,
        initial_prompt: str | None,
    ) -> list[TranscriptionSegment]:
        """Transcribe a full waveform with word-level timestamps.

        Args:
            audio: Loaded waveform and sample rate. Multichannel input is mixed
                down to mono. Samples are assumed to be sampled at 16 kHz, as
                produced by the pipeline audio conversion step.
            language: Whisper language code.
            initial_prompt: Optional prompt passed to the model.

        Returns:
            Ordered ASR segments, each carrying word-level timestamps.

        Raises:
            ValueError: If the waveform is not sampled at 16 kHz, which
                ``mlx-whisper`` silently assumes for raw array input.
        """
        if audio.sample_rate != _EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"mlx-whisper expects {_EXPECTED_SAMPLE_RATE} Hz audio, "
                f"got {audio.sample_rate} Hz. Convert the input first.",
            )
        mono = _mixdown_to_mono(audio.samples)
        result = cast(
            "dict[str, Any]",
            mlx_whisper.transcribe(
                mono,
                path_or_hf_repo=self._model_name,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=_TEMPERATURE_SCHEDULE,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                verbose=None,
            ),
        )
        return _parse_segments(result.get("segments", []))


def _mixdown_to_mono(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a multi-channel waveform to a contiguous mono float32 array.

    Args:
        waveform: Mono ``(samples,)`` or multichannel ``(samples, channels)``
            waveform.

    Returns:
        Contiguous mono float32 waveform expected by ``mlx-whisper``.
    """
    mono = waveform if waveform.ndim == 1 else waveform.mean(axis=1)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _parse_segments(raw_segments: list[dict[str, Any]]) -> list[TranscriptionSegment]:
    """Convert raw ``mlx-whisper`` segment payloads to service segments.

    Args:
        raw_segments: Segment dictionaries emitted by :func:`mlx_whisper.transcribe`.

    Returns:
        Ordered service-layer transcription segments with word timestamps.
    """
    segments: list[TranscriptionSegment] = []
    for raw in raw_segments:
        words = tuple(
            WordTimestamp(
                start=float(word["start"]),
                end=float(word["end"]),
                text=str(word["word"]).strip(),
            )
            for word in raw.get("words", [])
            if str(word.get("word", "")).strip()
        )
        segments.append(
            TranscriptionSegment(
                start=float(raw["start"]),
                end=float(raw["end"]),
                text=str(raw["text"]).strip(),
                no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
                avg_logprob=float(raw.get("avg_logprob", 0.0)),
                compression_ratio=float(raw.get("compression_ratio", 0.0)),
                words=words,
            ),
        )
    return segments

"""Service-level audio transcription helpers.

This module contains pure or side-effect-free helpers for audio processing,
chunked Whisper transcription, speaker assignment and segment merging.
The goal is to keep this code testable and independent from CLI/runtime wiring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    import whisper


def mixdown_to_mono(waveform: Any) -> Any:
    """Convert multi-channel audio to mono for Whisper.

    Args:
        waveform: Numpy-like array with shape ``(samples,)`` or
            ``(samples, channels)``.

    Returns:
        Waveform collapsed to a single channel. If the input is already
        one-dimensional, it is returned unchanged.
    """
    if getattr(waveform, "ndim", None) == 1:
        return waveform
    return waveform.mean(axis=1)


def transcribe_in_chunks(
    model: whisper.Whisper,
    audio_mono: Any,
    sample_rate: int,
    *,
    chunk_seconds: int,
    overlap_seconds: float,
    language: str,
    initial_prompt: str | None,
) -> list[dict[str, Any]]:
    """Transcribe long audio in overlapping chunks without dropping boundaries.

    Audio is processed in fixed-size windows with optional overlap. For each
    Whisper chunk only segments whose mid-point falls into the primary
    non-overlapping window are kept, which removes duplicates on the overlap.

    Args:
        model: Loaded Whisper model instance.
        audio_mono: Mono waveform array.
        sample_rate: Audio sample rate in Hz.
        chunk_seconds: Target length of the primary window in seconds.
        overlap_seconds: Additional context before and after the window.
        language: Language code for Whisper.
        initial_prompt: Optional initial prompt passed to Whisper.

    Returns:
        List of segment dicts with ``start``, ``end`` and ``text`` keys.

    Raises:
        ValueError: If ``chunk_seconds`` is not positive or
            ``overlap_seconds`` is negative.
    """
    if chunk_seconds <= 0:
        raise ValueError("CHUNK_SECONDS must be greater than 0.")
    if overlap_seconds < 0:
        raise ValueError("OVERLAP_SECONDS must be greater than or equal to 0.")

    total_samples = len(audio_mono)
    total_duration = total_samples / sample_rate
    collected_segments: list[dict[str, Any]] = []
    chunk_step = float(chunk_seconds)
    current = 0.0
    total_chunks = max(1, math.ceil(total_duration / chunk_step))
    chunk_index = 0

    logger.info(
        f"Chunk transcription started: duration={total_duration:.2f}s, "
        f"chunk_seconds={chunk_seconds}, overlap_seconds={overlap_seconds}, "
        f"total_chunks={total_chunks}",
    )

    while current < total_duration:
        chunk_index += 1
        remaining_chunks = max(0, total_chunks - chunk_index)
        chunk_start = max(0.0, current - overlap_seconds)
        chunk_end = min(total_duration, current + chunk_seconds + overlap_seconds)
        logger.info(
            f"Chunk progress: processing chunk №{chunk_index} out of {total_chunks}, "
            f"remaining={remaining_chunks}, window={chunk_start:.2f}-{chunk_end:.2f}s",
        )

        start_idx = int(chunk_start * sample_rate)
        end_idx = int(chunk_end * sample_rate)
        chunk_audio = audio_mono[start_idx:end_idx]

        result = model.transcribe(
            chunk_audio,
            language=language,
            fp16=False,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            temperature=0.0,
        )

        primary_start = current
        primary_end = min(total_duration, current + chunk_step)
        accepted_segments = 0

        for seg_raw in result["segments"]:
            seg = cast(dict[str, Any], seg_raw)
            seg_start = float(seg["start"]) + chunk_start
            seg_end = float(seg["end"]) + chunk_start
            seg_mid = (seg_start + seg_end) / 2

            # Берём только середину сегмента из "ядра" чанка, чтобы убрать дубли на overlap.
            if primary_start <= seg_mid < primary_end:
                accepted_segments += 1
                collected_segments.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg["text"],
                    },
                )
        logger.info(
            f"Chunk completed: processed={chunk_index}/{total_chunks}, "
            f"accepted_segments={accepted_segments}",
        )

        current += chunk_step

    logger.info(f"Chunk transcription completed: collected_segments={len(collected_segments)}")
    return collected_segments


def get_speaker(start: float, end: float, diarization: Any) -> str:
    """Find speaker label with maximum overlap with a segment.

    Args:
        start: Segment start in seconds.
        end: Segment end in seconds.
        diarization: Pyannote ``Annotation``-like object that provides
            ``itertracks(yield_label=True)`` yielding
            ``(segment, _, speaker_label)``.

    Returns:
        Speaker label (for example ``\"SPEAKER_00\"``) or ``\"Unknown\"`` when
        no overlap is found.
    """
    max_overlap = 0.0
    best_speaker = "Unknown"
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        overlap = min(segment.end, end) - max(segment.start, start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker
    return best_speaker


def merge_same_speaker(
    tagged: list[dict[str, str]],
    segments: Sequence[dict[str, float | str]],
    *,
    min_merge_gap: float = 1.5,
) -> list[dict[str, str]]:
    """Merge adjacent utterances of the same speaker if gap is small enough.

    Args:
        tagged: List of dicts with ``speaker`` and ``text`` keys.
        segments: Corresponding Whisper segments with ``start``/``end`` keys.
        min_merge_gap: Maximum allowed pause in seconds to merge segments.

    Returns:
        New list of merged utterances in the same order.
    """
    if not tagged:
        return []

    merged: list[dict[str, str]] = [
        {"speaker": tagged[0]["speaker"], "text": tagged[0]["text"]},
    ]

    for i in range(1, len(tagged)):
        gap = float(segments[i]["start"]) - float(segments[i - 1]["end"])
        same_speaker = tagged[i]["speaker"] == merged[-1]["speaker"]

        if same_speaker and gap <= min_merge_gap:
            merged[-1]["text"] += " " + tagged[i]["text"]
        else:
            merged.append({"speaker": tagged[i]["speaker"], "text": tagged[i]["text"]})

    return merged


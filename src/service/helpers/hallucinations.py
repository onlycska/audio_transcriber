"""Hallucination detection helpers for Whisper segments."""

from __future__ import annotations

import re

from src.service.helpers.speakers import normalize_token
from src.service.models.segment import TranscriptionSegment

_KNOWN_PATTERNS = [
    re.compile(r"dima\s*torzok", re.IGNORECASE),
    re.compile(r"продолжение\s+следует", re.IGNORECASE),
    re.compile(r"спасибо\s+за\s+просмотр", re.IGNORECASE),
    re.compile(
        r"субтитры\s+(создавал|сделал|подготовил|сделаны|корректор|редактор)",
        re.IGNORECASE,
    ),
    re.compile(r"редактор\s+субтитров", re.IGNORECASE),
]
_SERVICE_WORDS = {"spasibo", "prodolzhenie", "sleduet", "subtitry", "poka", "ugu"}


def is_hallucination_segment(
    segment: TranscriptionSegment,
    *,
    initial_prompt: str | None,
) -> bool:
    """Check whether ASR segment is likely hallucinated.

    Args:
        segment: ASR segment with Whisper confidence metrics.
        initial_prompt: Prompt used for the transcription run.

    Returns:
        ``True`` when the segment should be dropped as likely noise.
    """
    text = segment.text.strip()
    if not text:
        return True

    if any(pattern.search(text) for pattern in _KNOWN_PATTERNS):
        return True

    if initial_prompt and normalize_token(text) == normalize_token(initial_prompt):
        return True

    words = [normalize_token(part) for part in re.split(r"\s+", text) if part]
    if 0 < len(words) <= 3 and all(word in _SERVICE_WORDS for word in words):
        return True

    if (
        segment.no_speech_prob >= 0.6
        and (segment.avg_logprob <= -1.0 or segment.compression_ratio >= 2.4)
    ):
        return True

    return False

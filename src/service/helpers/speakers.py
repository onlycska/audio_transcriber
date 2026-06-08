"""Speaker assignment helpers.

Speakers are assigned at the **word level**: every word emitted by the ASR
backend is matched to the diarization turn it overlaps with, then consecutive
words sharing a speaker are grouped into a single replica. This avoids the
coarse "split a segment by time ratio" heuristic that frequently attributed
words to the wrong speaker.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.service.models.segment import (
    DiarizationTurn,
    TaggedSegment,
    TranscriptionSegment,
    WordTimestamp,
)


def normalize_token(token: str) -> str:
    """Normalize token to ASCII-like form for lightweight heuristics.

    Args:
        token: Source token or phrase.

    Returns:
        Lowercase compact transliteration suitable for fuzzy comparisons.
    """
    mapping = str.maketrans(
        {
            "а": "a",
            "б": "b",
            "в": "v",
            "г": "g",
            "д": "d",
            "е": "e",
            "ё": "e",
            "ж": "zh",
            "з": "z",
            "и": "i",
            "й": "i",
            "к": "k",
            "л": "l",
            "м": "m",
            "н": "n",
            "о": "o",
            "п": "p",
            "р": "r",
            "с": "s",
            "т": "t",
            "у": "u",
            "ф": "f",
            "х": "h",
            "ц": "ts",
            "ч": "ch",
            "ш": "sh",
            "щ": "sh",
            "ы": "y",
            "э": "e",
            "ю": "yu",
            "я": "ya",
        },
    )
    compact = re.sub(r"[^\w]+", "", token.lower(), flags=re.UNICODE)
    return compact.translate(mapping)


def is_short_backchannel(text: str, *, max_words: int = 4, max_chars: int = 24) -> bool:
    """Check whether text is likely a short backchannel response.

    Args:
        text: Segment text to classify.
        max_words: Maximum number of words allowed. Defaults to ``4``.
        max_chars: Maximum character length allowed. Defaults to ``24``.

    Returns:
        ``True`` for compact acknowledgements such as short fillers.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > max_chars:
        return False
    words = [w for w in re.split(r"\s+", stripped) if w]
    if not words or len(words) > max_words:
        return False
    normalized = [normalize_token(word) for word in words]
    return not any(len(token) > 4 for token in normalized)


def get_speaker_with_fallback(
    start: float,
    end: float,
    turns: Sequence[DiarizationTurn],
    *,
    fallback_max_gap: float = 0.8,
) -> str:
    """Find speaker label by overlap with nearest-turn fallback.

    Args:
        start: Span start time in seconds.
        end: Span end time in seconds.
        turns: Diarization turns to match against.
        fallback_max_gap: Maximum midpoint gap for nearest-turn fallback.
            Defaults to ``0.8``.

    Returns:
        Raw speaker label or ``"Unknown"``.
    """
    best_overlap = 0.0
    best_speaker = "Unknown"
    midpoint = (start + end) / 2
    nearest_gap = float("inf")
    nearest_speaker = "Unknown"

    for turn in turns:
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

        if turn.start <= midpoint <= turn.end:
            gap = 0.0
        elif midpoint < turn.start:
            gap = turn.start - midpoint
        else:
            gap = midpoint - turn.end
        if gap < nearest_gap:
            nearest_gap = gap
            nearest_speaker = turn.speaker

    if best_overlap > 0.0:
        return best_speaker
    if nearest_gap <= fallback_max_gap:
        return nearest_speaker
    return "Unknown"


def _to_canonical_speaker(raw_speaker: str, speaker_registry: dict[str, str]) -> str:
    """Map a raw speaker label to a stable transcript label.

    Args:
        raw_speaker: Raw diarization speaker label.
        speaker_registry: Mutable mapping of raw labels to canonical labels.

    Returns:
        Canonical speaker label or ``"Unknown"``.
    """
    if raw_speaker == "Unknown":
        return raw_speaker
    if raw_speaker not in speaker_registry:
        speaker_registry[raw_speaker] = f"SPEAKER_{len(speaker_registry):02d}"
    return speaker_registry[raw_speaker]


def _iter_assigned_words(
    segments: Sequence[TranscriptionSegment],
    turns: Sequence[DiarizationTurn],
    *,
    fallback_max_gap: float,
) -> list[tuple[WordTimestamp, str]]:
    """Pair every transcribed word with its diarization speaker.

    Words are taken from
    :attr:`~src.service.models.segment.TranscriptionSegment.words`. When a
    segment carries no word timestamps, the whole segment is treated as a single
    span so that no text is lost.

    Args:
        segments: ASR segments to assign.
        turns: Diarization turns.
        fallback_max_gap: Maximum midpoint gap for nearest-turn fallback.

    Returns:
        Ordered ``(word, raw_speaker)`` pairs with non-empty word text.
    """
    assigned: list[tuple[WordTimestamp, str]] = []
    for segment in segments:
        words: Sequence[WordTimestamp] = segment.words
        if not words:
            text = segment.text.strip()
            if text:
                words = (WordTimestamp(start=segment.start, end=segment.end, text=text),)
            else:
                continue
        for word in words:
            text = word.text.strip()
            if not text:
                continue
            raw_speaker = get_speaker_with_fallback(
                word.start,
                word.end,
                turns,
                fallback_max_gap=fallback_max_gap,
            )
            assigned.append((WordTimestamp(start=word.start, end=word.end, text=text), raw_speaker))
    return assigned


def build_tagged_timeline(
    segments: Sequence[TranscriptionSegment],
    turns: Sequence[DiarizationTurn],
    *,
    fallback_max_gap: float = 0.8,
) -> list[TaggedSegment]:
    """Build a speaker-tagged timeline from word-level ASR output.

    Each word is matched to its diarization turn, then consecutive words with
    the same raw speaker are grouped into one :class:`TaggedSegment`. Canonical
    labels (``SPEAKER_00``, ``SPEAKER_01``, ...) are assigned in first-seen
    order.

    Args:
        segments: ASR segments carrying word-level timestamps.
        turns: Diarization turns.
        fallback_max_gap: Maximum midpoint gap for nearest-turn fallback.
            Defaults to ``0.8``.

    Returns:
        Timeline entries with canonical speaker labels.
    """
    speaker_registry: dict[str, str] = {}
    timeline: list[TaggedSegment] = []

    for word, raw_speaker in _iter_assigned_words(
        segments,
        turns,
        fallback_max_gap=fallback_max_gap,
    ):
        if timeline and timeline[-1].raw_speaker == raw_speaker:
            previous = timeline[-1]
            timeline[-1] = TaggedSegment(
                start=previous.start,
                end=word.end,
                speaker=previous.speaker,
                raw_speaker=raw_speaker,
                text=f"{previous.text} {word.text}".strip(),
            )
            continue
        timeline.append(
            TaggedSegment(
                start=word.start,
                end=word.end,
                speaker=_to_canonical_speaker(raw_speaker, speaker_registry),
                raw_speaker=raw_speaker,
                text=word.text,
            ),
        )
    return timeline

"""Unit tests for hallucination detection rules."""

from __future__ import annotations

from src.service.helpers.hallucinations import is_hallucination_segment
from src.service.models.segment import TranscriptionSegment


def _segment(
    text: str,
    *,
    no_speech: float = 0.0,
    logprob: float = 0.0,
    ratio: float = 1.0,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        start=0.0,
        end=1.0,
        text=text,
        no_speech_prob=no_speech,
        avg_logprob=logprob,
        compression_ratio=ratio,
    )


def test_is_hallucination_segment_regex_positive() -> None:
    """Positive test: drops known service phrase 'Продолжение следует'."""
    # Arrange
    segment = _segment("Продолжение следует...")

    # Act
    result = is_hallucination_segment(segment, initial_prompt=None)

    # Assert
    assert result is True


def test_is_hallucination_segment_metrics_positive() -> None:
    """Positive test: drops segment on high no_speech and poor confidence."""
    # Arrange
    segment = _segment("что-то", no_speech=0.9, logprob=-1.4, ratio=1.1)

    # Act
    result = is_hallucination_segment(segment, initial_prompt=None)

    # Assert
    assert result is True


def test_is_hallucination_segment_regular_phrase_negative() -> None:
    """Negative test: keeps meaningful sentence with normal metrics."""
    # Arrange
    segment = _segment(
        "Добрый день, начнем интервью.",
        no_speech=0.1,
        logprob=-0.2,
        ratio=1.1,
    )

    # Act
    result = is_hallucination_segment(segment, initial_prompt=None)

    # Assert
    assert result is False

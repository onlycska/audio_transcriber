"""Unit tests for word-level speaker assignment helpers."""

from __future__ import annotations

from src.service.helpers.speakers import (
    build_tagged_timeline,
    get_speaker_with_fallback,
)
from src.service.models.segment import DiarizationTurn, TranscriptionSegment, WordTimestamp


def _segment(
    start: float,
    end: float,
    text: str,
    words: tuple[WordTimestamp, ...] = (),
) -> TranscriptionSegment:
    """Build a transcription segment with neutral confidence metrics."""
    return TranscriptionSegment(
        start=start,
        end=end,
        text=text,
        no_speech_prob=0.1,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        words=words,
    )


def test_get_speaker_with_fallback_nearest_turn_positive() -> None:
    """Positive test: falls back to nearest turn when gap is acceptable."""
    # Arrange
    turns = [DiarizationTurn(start=0.0, end=1.0, speaker="spk_a")]

    # Act
    speaker = get_speaker_with_fallback(1.2, 1.4, turns, fallback_max_gap=0.8)

    # Assert
    assert speaker == "spk_a"


def test_build_tagged_timeline_stable_registry_positive() -> None:
    """Positive test: keeps canonical labels stable across the timeline."""
    # Arrange
    turns = [
        DiarizationTurn(start=0.0, end=1.0, speaker="raw_a"),
        DiarizationTurn(start=1.0, end=2.0, speaker="raw_b"),
        DiarizationTurn(start=2.0, end=3.0, speaker="raw_a"),
    ]
    segments = [
        _segment(0.1, 0.8, "one", (WordTimestamp(0.1, 0.8, "one"),)),
        _segment(1.1, 1.8, "two", (WordTimestamp(1.1, 1.8, "two"),)),
        _segment(2.1, 2.8, "three", (WordTimestamp(2.1, 2.8, "three"),)),
    ]

    # Act
    tagged = build_tagged_timeline(segments, turns)

    # Assert
    assert [item.speaker for item in tagged] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_build_tagged_timeline_splits_words_by_speaker_positive() -> None:
    """Positive test: assigns words inside one segment to overlapping speakers."""
    # Arrange
    turns = [
        DiarizationTurn(start=0.0, end=1.0, speaker="raw_a"),
        DiarizationTurn(start=1.0, end=2.0, speaker="raw_b"),
    ]
    words = (
        WordTimestamp(0.0, 0.4, "хорошо"),
        WordTimestamp(0.4, 0.9, "слышно"),
        WordTimestamp(1.1, 1.5, "да"),
        WordTimestamp(1.5, 1.9, "отлично"),
    )
    segments = [_segment(0.0, 2.0, "хорошо слышно да отлично", words)]

    # Act
    tagged = build_tagged_timeline(segments, turns)

    # Assert
    assert [(item.speaker, item.text) for item in tagged] == [
        ("SPEAKER_00", "хорошо слышно"),
        ("SPEAKER_01", "да отлично"),
    ]


def test_build_tagged_timeline_groups_consecutive_same_speaker_positive() -> None:
    """Positive test: merges consecutive words of the same speaker into one line."""
    # Arrange
    turns = [DiarizationTurn(start=0.0, end=2.0, speaker="raw_a")]
    words = (
        WordTimestamp(0.0, 0.5, "это"),
        WordTimestamp(0.5, 1.0, "один"),
        WordTimestamp(1.0, 1.5, "спикер"),
    )
    segments = [_segment(0.0, 1.5, "это один спикер", words)]

    # Act
    tagged = build_tagged_timeline(segments, turns)

    # Assert
    assert len(tagged) == 1
    assert (tagged[0].speaker, tagged[0].text, tagged[0].start, tagged[0].end) == (
        "SPEAKER_00",
        "это один спикер",
        0.0,
        1.5,
    )


def test_build_tagged_timeline_segment_without_words_positive() -> None:
    """Positive test: falls back to whole-segment span when words are missing."""
    # Arrange
    turns = [DiarizationTurn(start=0.0, end=1.0, speaker="raw_a")]
    segments = [_segment(0.1, 0.9, "без таймстампов слов")]

    # Act
    tagged = build_tagged_timeline(segments, turns)

    # Assert
    assert len(tagged) == 1
    assert (tagged[0].speaker, tagged[0].text) == ("SPEAKER_00", "без таймстампов слов")


def test_build_tagged_timeline_no_overlap_unknown_negative() -> None:
    """Negative test: marks a word Unknown when no turn is within fallback gap."""
    # Arrange
    turns = [DiarizationTurn(start=10.0, end=11.0, speaker="raw_a")]
    segments = [_segment(0.0, 0.4, "тест", (WordTimestamp(0.0, 0.4, "тест"),))]

    # Act
    tagged = build_tagged_timeline(segments, turns)

    # Assert
    assert (len(tagged), tagged[0].speaker, tagged[0].raw_speaker) == (1, "Unknown", "Unknown")

"""Unit tests for dialogue timeline postprocessing."""

from __future__ import annotations

from src.service.helpers.postprocess import postprocess_dialogue_timeline
from src.service.models.segment import TaggedSegment


def test_postprocess_unknown_between_different_neighbors_positive() -> None:
    """Positive test: reassigns Unknown to nearest neighbor by time gap."""
    # Arrange
    timeline = [
        TaggedSegment(0.0, 1.0, "SPEAKER_00", "raw_a", "вопрос"),
        TaggedSegment(1.0, 1.2, "Unknown", "Unknown", "да"),
        TaggedSegment(1.5, 2.0, "SPEAKER_01", "raw_b", "ответ"),
    ]

    # Act
    merged, audit = postprocess_dialogue_timeline(timeline, aggressive_mode=True)

    # Assert
    assert (audit["unknown_reassigned"], merged[0].speaker) == (1, "SPEAKER_00")


def test_postprocess_trailing_prune_positive() -> None:
    """Positive test: removes repeated trailing service phrases."""
    # Arrange
    timeline = [
        TaggedSegment(0.0, 0.8, "SPEAKER_00", "raw_a", "основная мысль"),
        TaggedSegment(0.9, 1.0, "Unknown", "Unknown", "Спасибо"),
        TaggedSegment(1.0, 1.1, "Unknown", "Unknown", "Спасибо спасибо"),
    ]

    # Act
    merged, audit = postprocess_dialogue_timeline(timeline, aggressive_mode=True)

    # Assert
    assert (
        audit["removed_unknown_lines"],
        audit["removed_trailing_noise"],
        merged == [merged[0]],
        "основная мысль" in merged[0].text,
    ) == (2, 0, True, True)


def test_postprocess_keeps_meaningful_last_negative() -> None:
    """Negative test: does not remove meaningful final replica."""
    # Arrange
    timeline = [TaggedSegment(0.0, 1.0, "SPEAKER_00", "raw_a", "Договорились, до встречи завтра.")]

    # Act
    merged, audit = postprocess_dialogue_timeline(timeline, aggressive_mode=True)

    # Assert
    assert (audit["removed_trailing_noise"], merged[0].text) == (
        0,
        "Договорились, до встречи завтра.",
    )


def test_postprocess_drops_all_unknown_positive() -> None:
    """Positive test: removes all Unknown lines and keeps speaker lines."""
    # Arrange
    timeline = [
        TaggedSegment(0.0, 0.4, "Unknown", "Unknown", "случайный шум"),
        TaggedSegment(0.4, 1.0, "SPEAKER_00", "raw_a", "осмысленная реплика"),
        TaggedSegment(1.0, 1.3, "Unknown", "Unknown", "продолжение следует"),
    ]

    # Act
    merged, audit = postprocess_dialogue_timeline(timeline, aggressive_mode=True)

    # Assert
    assert (audit["removed_unknown_lines"], len(merged), merged[0].speaker) == (
        2,
        1,
        "SPEAKER_00",
    )


def test_postprocess_flicker_reassigned_positive() -> None:
    """Positive test: reassigns short A-B-A flicker to dominant speaker."""
    # Arrange
    timeline = [
        TaggedSegment(0.0, 1.0, "SPEAKER_00", "raw_a", "длинная реплика"),
        TaggedSegment(1.0, 1.2, "SPEAKER_01", "raw_b", "угу"),
        TaggedSegment(1.2, 2.0, "SPEAKER_00", "raw_a", "продолжение"),
    ]

    # Act
    merged, audit = postprocess_dialogue_timeline(
        timeline,
        aggressive_mode=True,
        flicker_max_duration=0.5,
    )

    # Assert
    assert audit["flicker_reassigned"] == 1
    assert audit["short_backchannel_reassigned"] == 1
    assert len(merged) == 1
    assert merged[0].speaker == "SPEAKER_00"


def test_postprocess_short_answer_after_question_kept_positive() -> None:
    """Positive test: keeps short backchannel when previous line is a question."""
    # Arrange
    timeline = [
        TaggedSegment(0.0, 1.0, "SPEAKER_00", "raw_a", "Тебе видно?"),
        TaggedSegment(1.0, 1.2, "SPEAKER_01", "raw_b", "Да"),
        TaggedSegment(1.2, 2.0, "SPEAKER_00", "raw_a", "Отлично, продолжаем."),
    ]

    # Act
    merged, audit = postprocess_dialogue_timeline(
        timeline,
        aggressive_mode=True,
        flicker_max_duration=0.5,
    )

    # Assert
    assert audit["short_backchannel_kept"] == 1
    assert len(merged) == 3
    assert merged[1].speaker == "SPEAKER_01"

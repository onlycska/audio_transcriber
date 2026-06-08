"""Segment-level service models for diarization and transcription."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Single transcribed word with its time span.

    Word-level timestamps are the basis for accurate speaker assignment: each
    word can be matched to the diarization turn it overlaps with, instead of
    splitting a whole segment by a coarse time ratio.

    Attributes:
        start: Word start time in seconds.
        end: Word end time in seconds.
        text: Word text without surrounding whitespace.
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Single ASR segment enriched with Whisper confidence metrics.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text.
        no_speech_prob: Whisper no-speech probability.
        avg_logprob: Whisper average log probability.
        compression_ratio: Whisper compression ratio.
        words: Ordered word-level timestamps for this segment. Empty when the
            ASR backend did not produce word timestamps.
    """

    start: float
    end: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    words: tuple[WordTimestamp, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class DiarizationTurn:
    """Single diarization turn with speaker label and time boundaries.

    Attributes:
        start: Turn start time in seconds.
        end: Turn end time in seconds.
        speaker: Raw speaker label from diarization provider.
    """

    start: float
    end: float
    speaker: str


@dataclass(frozen=True, slots=True)
class TaggedSegment:
    """ASR segment paired with resolved canonical and raw speaker labels.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        speaker: Canonical speaker label used in transcript output.
        raw_speaker: Raw speaker label before canonical mapping.
        text: Segment text.
    """

    start: float
    end: float
    speaker: str
    raw_speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class DialogueLine:
    """Final transcript line grouped by a speaker.

    Attributes:
        speaker: Canonical speaker label.
        text: Merged transcript text for this speaker line.
    """

    speaker: str
    text: str

"""Unit tests for service transcription helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.service.transcription import (
    get_speaker,
    merge_same_speaker,
    mixdown_to_mono,
    transcribe_in_chunks,
)


class _MonoArray:
    def __init__(self, values: list[float]) -> None:
        self.ndim = 1
        self._values = values

    def __len__(self) -> int:
        return len(self._values)


class _StereoArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self.ndim = 2
        self._rows = rows

    def mean(self, axis: int) -> list[float]:
        if axis != 1:
            raise ValueError("Expected axis=1")
        return [sum(row) / len(row) for row in self._rows]


class _ArrayWithoutNdim:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def mean(self, axis: int) -> list[float]:
        if axis != 1:
            raise ValueError("Expected axis=1")
        return [sum(row) / len(row) for row in self._rows]


class _FakeModel:
    def __init__(self, responses: list[list[dict[str, float | str]]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        chunk_audio: object,
        *,
        language: str,
        fp16: bool,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
        temperature: float,
    ) -> dict[str, list[dict[str, float | str]]]:
        call_idx = len(self.calls)
        self.calls.append(
            {
                "chunk_audio": chunk_audio,
                "language": language,
                "fp16": fp16,
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": condition_on_previous_text,
                "temperature": temperature,
            },
        )
        return {"segments": self._responses[call_idx]}


@dataclass
class _Segment:
    start: float
    end: float


class _FakeDiarization:
    def __init__(self, tracks: list[tuple[_Segment, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        if not yield_label:
            return
        for segment, speaker in self._tracks:
            yield segment, None, speaker


def test_mixdown_to_mono_1d_input_positive() -> None:
    """Positive test: returns the same object for mono waveform."""
    # Arrange
    mono = _MonoArray(values=[0.1, 0.2, 0.3])

    # Act
    result = mixdown_to_mono(mono)

    # Assert
    assert result is mono


def test_mixdown_to_mono_2d_input_positive() -> None:
    """Positive test: averages channels for stereo-like waveform."""
    # Arrange
    stereo = _StereoArray(rows=[[2.0, 4.0], [3.0, 5.0], [0.0, 2.0]])

    # Act
    result = mixdown_to_mono(stereo)

    # Assert
    assert result == [3.0, 4.0, 1.0]


def test_mixdown_to_mono_no_ndim_attribute_positive() -> None:
    """Positive test: handles object without ndim via mean(axis=1)."""
    # Arrange
    stereo = _ArrayWithoutNdim(rows=[[1.0, 3.0], [8.0, 4.0]])

    # Act
    result = mixdown_to_mono(stereo)

    # Assert
    assert result == [2.0, 6.0]


def test_transcribe_in_chunks_single_chunk_positive() -> None:
    """Positive test: returns expected segments for one chunk."""
    # Arrange
    model = _FakeModel(
        responses=[
            [
                {"start": 0.1, "end": 0.5, "text": "one"},
                {"start": 0.6, "end": 0.9, "text": "two"},
            ],
        ],
    )
    audio_mono = [0.0] * 10

    # Act
    result = transcribe_in_chunks(
        model=model,  # type: ignore[arg-type]
        audio_mono=audio_mono,
        sample_rate=10,
        chunk_seconds=2,
        overlap_seconds=0.0,
        language="ru",
        initial_prompt=None,
    )

    # Assert
    assert result == [
        {"start": 0.1, "end": 0.5, "text": "one"},
        {"start": 0.6, "end": 0.9, "text": "two"},
    ]


def test_transcribe_in_chunks_two_chunks_no_overlap_positive() -> None:
    """Positive test: combines ordered segments from two chunks."""
    # Arrange
    model = _FakeModel(
        responses=[
            [{"start": 0.2, "end": 0.7, "text": "first"}],
            [{"start": 0.1, "end": 0.6, "text": "second"}],
        ],
    )
    audio_mono = [0.0] * 40

    # Act
    result = transcribe_in_chunks(
        model=model,  # type: ignore[arg-type]
        audio_mono=audio_mono,
        sample_rate=10,
        chunk_seconds=2,
        overlap_seconds=0.0,
        language="ru",
        initial_prompt=None,
    )

    # Assert
    assert result == [
        {"start": 0.2, "end": 0.7, "text": "first"},
        {"start": 2.1, "end": 2.6, "text": "second"},
    ]


def test_transcribe_in_chunks_with_overlap_deduplicates_segments_positive() -> None:
    """Positive test: keeps segment once when overlap creates duplicate."""
    # Arrange
    model = _FakeModel(
        responses=[
            [{"start": 0.9, "end": 1.1, "text": "dup"}],
            [{"start": 0.1, "end": 0.3, "text": "dup"}],
        ],
    )
    audio_mono = [0.0] * 40

    # Act
    result = transcribe_in_chunks(
        model=model,  # type: ignore[arg-type]
        audio_mono=audio_mono,
        sample_rate=10,
        chunk_seconds=2,
        overlap_seconds=1.0,
        language="ru",
        initial_prompt=None,
    )

    # Assert
    assert result == [{"start": 0.9, "end": 1.1, "text": "dup"}]


def test_transcribe_in_chunks_invalid_chunk_seconds_negative() -> None:
    """Negative test: raises ValueError for non-positive chunk size."""
    # Arrange
    model = _FakeModel(responses=[])
    audio_mono = [0.0] * 10

    # Act & Assert
    with pytest.raises(ValueError, match="CHUNK_SECONDS"):
        transcribe_in_chunks(
            model=model,  # type: ignore[arg-type]
            audio_mono=audio_mono,
            sample_rate=10,
            chunk_seconds=0,
            overlap_seconds=0.0,
            language="ru",
            initial_prompt=None,
        )


def test_transcribe_in_chunks_negative_overlap_seconds_negative() -> None:
    """Negative test: raises ValueError for negative overlap."""
    # Arrange
    model = _FakeModel(responses=[])
    audio_mono = [0.0] * 10

    # Act & Assert
    with pytest.raises(ValueError, match="OVERLAP_SECONDS"):
        transcribe_in_chunks(
            model=model,  # type: ignore[arg-type]
            audio_mono=audio_mono,
            sample_rate=10,
            chunk_seconds=2,
            overlap_seconds=-0.1,
            language="ru",
            initial_prompt=None,
        )


def test_transcribe_in_chunks_empty_audio_positive() -> None:
    """Positive test: returns empty list for empty audio."""
    # Arrange
    model = _FakeModel(responses=[])

    # Act
    result = transcribe_in_chunks(
        model=model,  # type: ignore[arg-type]
        audio_mono=[],
        sample_rate=10,
        chunk_seconds=2,
        overlap_seconds=0.0,
        language="ru",
        initial_prompt=None,
    )

    # Assert
    assert result == []


def test_transcribe_in_chunks_passes_language_and_prompt_positive() -> None:
    """Positive test: forwards language, prompt and decoding flags."""
    # Arrange
    model = _FakeModel(responses=[[{"start": 0.0, "end": 0.2, "text": "x"}]])
    audio_mono = [0.0] * 10

    # Act
    transcribe_in_chunks(
        model=model,  # type: ignore[arg-type]
        audio_mono=audio_mono,
        sample_rate=10,
        chunk_seconds=2,
        overlap_seconds=0.0,
        language="en",
        initial_prompt="custom prompt",
    )

    # Assert
    assert model.calls[0] == {
        "chunk_audio": audio_mono,
        "language": "en",
        "fp16": False,
        "initial_prompt": "custom prompt",
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }


def test_get_speaker_single_segment_positive() -> None:
    """Positive test: returns speaker for full overlap with one track."""
    # Arrange
    diarization = _FakeDiarization([(_Segment(0.0, 5.0), "SPEAKER_00")])

    # Act
    speaker = get_speaker(1.0, 2.0, diarization)

    # Assert
    assert speaker == "SPEAKER_00"


def test_get_speaker_choose_max_overlap_positive() -> None:
    """Positive test: picks speaker with maximum overlap."""
    # Arrange
    diarization = _FakeDiarization(
        [
            (_Segment(0.0, 1.4), "SPEAKER_00"),
            (_Segment(0.5, 2.0), "SPEAKER_01"),
        ],
    )

    # Act
    speaker = get_speaker(1.0, 1.8, diarization)

    # Assert
    assert speaker == "SPEAKER_01"


def test_get_speaker_no_overlap_negative() -> None:
    """Negative test: returns Unknown when there is no overlap."""
    # Arrange
    diarization = _FakeDiarization([(_Segment(0.0, 1.0), "SPEAKER_00")])

    # Act
    speaker = get_speaker(2.0, 3.0, diarization)

    # Assert
    assert speaker == "Unknown"


def test_merge_same_speaker_empty_input_positive() -> None:
    """Positive test: returns empty list for empty tagged input."""
    # Arrange
    tagged: list[dict[str, str]] = []
    segments: list[dict[str, float | str]] = []

    # Act
    merged = merge_same_speaker(tagged, segments)

    # Assert
    assert merged == []


def test_merge_same_speaker_single_utterance_positive() -> None:
    """Positive test: keeps single utterance unchanged."""
    # Arrange
    tagged = [{"speaker": "SPEAKER_00", "text": "hello"}]
    segments: list[dict[str, float | str]] = [{"start": 0.0, "end": 0.4}]

    # Act
    merged = merge_same_speaker(tagged, segments)

    # Assert
    assert merged == [{"speaker": "SPEAKER_00", "text": "hello"}]


def test_merge_same_speaker_merge_when_gap_small_positive() -> None:
    """Positive test: merges same speaker utterances for small gap."""
    # Arrange
    tagged = [
        {"speaker": "SPEAKER_00", "text": "hello"},
        {"speaker": "SPEAKER_00", "text": "world"},
    ]
    segments: list[dict[str, float | str]] = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.4, "end": 2.0},
    ]

    # Act
    merged = merge_same_speaker(tagged, segments, min_merge_gap=0.5)

    # Assert
    assert merged == [{"speaker": "SPEAKER_00", "text": "hello world"}]


def test_merge_same_speaker_do_not_merge_when_gap_large_positive() -> None:
    """Positive test: does not merge when pause exceeds threshold."""
    # Arrange
    tagged = [
        {"speaker": "SPEAKER_00", "text": "hello"},
        {"speaker": "SPEAKER_00", "text": "world"},
    ]
    segments: list[dict[str, float | str]] = [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 3.0},
    ]

    # Act
    merged = merge_same_speaker(tagged, segments, min_merge_gap=0.5)

    # Assert
    assert merged == [
        {"speaker": "SPEAKER_00", "text": "hello"},
        {"speaker": "SPEAKER_00", "text": "world"},
    ]


def test_merge_same_speaker_do_not_merge_different_speakers_positive() -> None:
    """Positive test: does not merge utterances of different speakers."""
    # Arrange
    tagged = [
        {"speaker": "SPEAKER_00", "text": "hello"},
        {"speaker": "SPEAKER_01", "text": "world"},
    ]
    segments: list[dict[str, float | str]] = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.1, "end": 2.0},
    ]

    # Act
    merged = merge_same_speaker(tagged, segments, min_merge_gap=0.5)

    # Assert
    assert merged == [
        {"speaker": "SPEAKER_00", "text": "hello"},
        {"speaker": "SPEAKER_01", "text": "world"},
    ]

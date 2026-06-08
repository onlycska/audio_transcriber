"""Configuration model for transcription pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime options for one transcription run.

    Attributes:
        num_speakers: Expected speaker count for diarization.
        language: Whisper language code.
        initial_prompt: Optional prompt passed to Whisper.
        aggressive_dialogue_postprocess: Whether to enable additional speaker
            cleanup heuristics.
        postprocess_debug: Whether postprocessing should emit debug audit logs.
        flicker_max_duration: Maximum duration of a short A-B-A flicker that may
            be reassigned to the surrounding speaker.
        fallback_max_gap: Maximum midpoint gap (seconds) used when assigning a
            word to the nearest diarization turn without direct overlap.
    """

    num_speakers: int
    language: str
    initial_prompt: str | None
    aggressive_dialogue_postprocess: bool
    postprocess_debug: bool
    flicker_max_duration: float
    fallback_max_gap: float

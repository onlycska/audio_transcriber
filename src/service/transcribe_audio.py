"""Use case for full audio transcription pipeline."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.service.helpers.hallucinations import is_hallucination_segment
from src.service.helpers.postprocess import postprocess_dialogue_timeline
from src.service.helpers.speakers import build_tagged_timeline
from src.service.interfaces.audio_loader import IAudioLoader
from src.service.interfaces.diarizer import IDiarizer
from src.service.interfaces.transcriber import ITranscriber
from src.service.interfaces.transcript_writer import ITranscriptWriter
from src.service.models.pipeline import PipelineConfig


class TranscribeAudioUseCase:
    """Execute end-to-end transcription and speaker assignment.

    Args:
        audio_loader: Audio loading implementation.
        diarizer: Speaker diarization implementation.
        transcriber: Chunked ASR implementation.
        writer: Transcript persistence implementation.
    """

    def __init__(
        self,
        *,
        audio_loader: IAudioLoader,
        diarizer: IDiarizer,
        transcriber: ITranscriber,
        writer: ITranscriptWriter,
    ) -> None:
        """Initialize use case dependencies.

        Args:
            audio_loader: Audio loading implementation.
            diarizer: Speaker diarization implementation.
            transcriber: Chunked ASR implementation.
            writer: Transcript persistence implementation.
        """
        self._audio_loader = audio_loader
        self._diarizer = diarizer
        self._transcriber = transcriber
        self._writer = writer

    def execute(self, *, audio_path: Path, config: PipelineConfig) -> Path:
        """Run full transcription pipeline and return path to transcript file.

        Args:
            audio_path: Path to normalized input audio.
            config: Runtime transcription options.

        Returns:
            Path to the written transcript.
        """
        audio = self._audio_loader.load(audio_path)
        turns = self._diarizer.diarize(audio, num_speakers=config.num_speakers)
        segments = self._transcriber.transcribe(
            audio,
            language=config.language,
            initial_prompt=config.initial_prompt,
        )
        filtered_segments = [
            item
            for item in segments
            if not is_hallucination_segment(item, initial_prompt=config.initial_prompt)
        ]
        timeline = build_tagged_timeline(
            filtered_segments,
            turns,
            fallback_max_gap=config.fallback_max_gap,
        )
        merged, audit = postprocess_dialogue_timeline(
            timeline,
            aggressive_mode=config.aggressive_dialogue_postprocess,
            debug=config.postprocess_debug,
            flicker_max_duration=config.flicker_max_duration,
        )
        dialogue_duration = 0.0
        if filtered_segments:
            dialogue_duration = max(item.end for item in filtered_segments)
        speaker_switches = max(0, len(merged) - 1)
        speaker_switches_per_minute = (
            speaker_switches / max(1e-6, dialogue_duration / 60.0)
            if dialogue_duration > 0
            else 0.0
        )
        replicas_short_le_2_tokens = sum(1 for line in merged if len(line.text.split()) <= 2)
        logger.bind(
            segments=len(segments),
            filtered_segments=len(filtered_segments),
            merged_replicas=len(merged),
            speaker_switches_per_minute=round(speaker_switches_per_minute, 2),
            replicas_short_le_2_tokens=replicas_short_le_2_tokens,
            unknown_reassigned=audit["unknown_reassigned"],
            removed_unknown_lines=audit["removed_unknown_lines"],
            collapsed_stutter_loop=audit["collapsed_stutter_loop"],
            removed_trailing_noise=audit["removed_trailing_noise"],
            merged_adjacent_same_speaker=audit["merged_adjacent_same_speaker"],
            flicker_reassigned=audit["flicker_reassigned"],
            short_backchannel_kept=audit["short_backchannel_kept"],
            short_backchannel_reassigned=audit["short_backchannel_reassigned"],
        ).info("Transcription completed")
        return self._writer.write(merged, audio_stem=audio_path.stem)

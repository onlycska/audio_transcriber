"""CLI composition root for transcription pipeline."""

from __future__ import annotations

from loguru import logger

from src.infrastructure.audio_conversion import ensure_wav_for_pipeline
from src.infrastructure.audio_loader import SoundfileAudioLoader
from src.infrastructure.logging import configure_logging
from src.infrastructure.mlx_whisper_transcriber import MlxWhisperTranscriber
from src.infrastructure.pyannote_diarizer import PyannoteDiarizer
from src.infrastructure.settings import PipelineSettings
from src.infrastructure.transcript_writer import FileTranscriptWriter
from src.service.transcribe_audio import TranscribeAudioUseCase


def main() -> None:
    """Run transcription pipeline from environment configuration.

    Raises:
        ValueError: If required environment variables are missing.
        RuntimeError: If audio conversion or model initialization fails.
    """
    configure_logging()
    settings = PipelineSettings.from_env()
    logger.info("Application startup")

    source_audio_path = settings.resolve_audio_path()
    audio_path = ensure_wav_for_pipeline(
        source_audio_path,
        ffmpeg_audio_filter=settings.ffmpeg_audio_filter,
    )
    logger.bind(audio_path=str(audio_path)).info("Resolved audio input")

    use_case = TranscribeAudioUseCase(
        audio_loader=SoundfileAudioLoader(),
        diarizer=PyannoteDiarizer(hf_token=settings.hf_token),
        transcriber=MlxWhisperTranscriber(model_name=settings.whisper_model),
        writer=FileTranscriptWriter(results_dir=settings.results_dir),
    )

    output_path = use_case.execute(audio_path=audio_path, config=settings.config)
    logger.bind(output_path=str(output_path)).success("Saved transcript")


if __name__ == "__main__":
    main()

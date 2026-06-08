"""Environment-backed settings for transcription CLI pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.infrastructure.audio_conversion import resolve_audio_input_path
from src.service.models.pipeline import PipelineConfig

_DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Raw infrastructure settings loaded from environment variables.

    Attributes:
        hf_token: Hugging Face access token for PyAnnote.
        audio_file: Configured audio file path or file name.
        whisper_model: Whisper model name.
        audio_base_dir: Repository root used for relative audio paths.
        results_dir: Directory where transcript files are written.
        ffmpeg_audio_filter: ffmpeg audio filter chain.
        config: Service-layer transcription options.
    """

    hf_token: str
    audio_file: str
    whisper_model: str
    audio_base_dir: Path
    results_dir: Path
    ffmpeg_audio_filter: str
    config: PipelineConfig

    @classmethod
    def from_env(cls) -> PipelineSettings:
        """Load pipeline settings from environment variables.

        Returns:
            Validated pipeline settings.

        Raises:
            ValueError: If required environment variables are missing.
        """
        repo_root = Path(__file__).resolve().parents[2]
        hf_token = _required_env("HF_TOKEN")
        audio_file = _required_env("AUDIO_FILE")
        num_speakers = int(os.getenv("NUM_SPEAKERS", "2"))
        people_word = "человека" if num_speakers == 1 else "людей"
        default_prompt = f"Разговор {num_speakers} {people_word} на русском языке."
        initial_prompt_raw = os.getenv("INITIAL_PROMPT")
        if initial_prompt_raw is None:
            initial_prompt = default_prompt
        else:
            initial_prompt = initial_prompt_raw.strip() or None

        return cls(
            hf_token=hf_token,
            audio_file=audio_file,
            whisper_model=os.getenv("WHISPER_MODEL", _DEFAULT_WHISPER_MODEL),
            audio_base_dir=repo_root,
            results_dir=resolve_audio_input_path(
                os.getenv("RESULTS_DIR", "results"),
                base_dir=repo_root,
            ),
            ffmpeg_audio_filter=os.getenv(
                "AUDIO_FFMPEG_FILTER",
                "highpass=f=80,lowpass=f=8000",
            ),
            config=PipelineConfig(
                num_speakers=num_speakers,
                language=os.getenv("LANGUAGE", "ru"),
                initial_prompt=initial_prompt,
                aggressive_dialogue_postprocess=os.getenv(
                    "AGGRESSIVE_DIALOGUE_POSTPROCESS",
                    "true",
                ).lower()
                in {"1", "true", "yes"},
                postprocess_debug=os.getenv("POSTPROCESS_DEBUG", "false").lower()
                in {"1", "true", "yes"},
                flicker_max_duration=float(os.getenv("FLICKER_MAX_DURATION", "0.5")),
                fallback_max_gap=float(os.getenv("FALLBACK_MAX_GAP", "0.8")),
            ),
        )

    def resolve_audio_path(self) -> Path:
        """Resolve configured input audio path.

        Returns:
            Canonical input audio path.
        """
        return resolve_audio_input_path(self.audio_file, base_dir=self.audio_base_dir)


def _required_env(name: str) -> str:
    """Read a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Stripped environment variable value.

    Raises:
        ValueError: If the variable is unset or blank.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} environment variable is required.")
    return value.strip()

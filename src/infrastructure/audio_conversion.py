"""Infrastructure helpers for audio format normalization via ffmpeg."""

from __future__ import annotations

from pathlib import Path
import subprocess

from loguru import logger


def ensure_wav_for_pipeline(
    input_path: Path,
    *,
    target_sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_audio_filter: str = "highpass=f=200,lowpass=f=3000",
) -> Path:
    """Ensure audio file is available as normalized WAV for the pipeline.

    Args:
        input_path: Source audio path.
        target_sample_rate: Target sample rate for output WAV.
        channels: Target number of channels.
        ffmpeg_audio_filter: ffmpeg audio filter chain.

    Returns:
        Path to original WAV file or converted WAV file.

    Raises:
        RuntimeError: If ffmpeg is missing or conversion fails.
    """
    source_path = input_path.expanduser().resolve()
    if source_path.suffix.lower() == ".wav":
        return source_path

    output_path = source_path.with_name(f"{source_path.stem}_converted.wav")
    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        logger.info(f"Using cached converted audio: input={source_path}, output={output_path}")
        return output_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ar",
        str(target_sample_rate),
        "-ac",
        str(channels),
        "-af",
        ffmpeg_audio_filter,
        str(output_path),
    ]

    logger.info(
        f"Converting audio for pipeline: input={source_path}, output={output_path}, "
        f"sample_rate={target_sample_rate}, channels={channels}, filter={ffmpeg_audio_filter}",
    )

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not available in PATH. Install ffmpeg to enable audio conversion.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}") from exc

    return output_path


"""Infrastructure helpers for audio format normalization via ffmpeg."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from loguru import logger


def _build_content_hash_suffix(source_path: Path, length: int = 10) -> str:
    """Return trailing hash suffix for file content-based cache keys.

    Args:
        source_path: File whose content should be hashed.
        length: Number of trailing hex characters to keep.

    Returns:
        Stable hash suffix for converted file names.
    """
    digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[-length:]


def resolve_audio_input_path(
    audio_file: str | Path,
    base_dir: str | Path,
) -> Path:
    """Resolve filesystem path to an audio file for pipeline input.

    After :meth:`~pathlib.Path.expanduser`, paths that are absolute are
    canonicalized with :meth:`~pathlib.Path.resolve` and returned unchanged
    in meaning (no *base_dir* join). Relative paths are joined to *base_dir*
    then resolved.

    Args:
        audio_file: File name or path from environment or configuration.
        base_dir: Directory used when *audio_file* is relative.

    Returns:
        Canonical path suitable for :func:`ensure_wav_for_pipeline`.
    """
    candidate = Path(audio_file).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(base_dir).expanduser().resolve() / candidate).resolve()


def ensure_wav_for_pipeline(
    input_path: Path,
    *,
    target_sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_audio_filter: str = "highpass=f=80,lowpass=f=8000",
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

    hash_suffix = _build_content_hash_suffix(source_path)
    output_path = source_path.with_name(f"{source_path.stem}_converted_{hash_suffix}.wav")
    if output_path.exists():
        logger.bind(input_path=str(source_path), output_path=str(output_path)).info(
            "Using cached converted audio from disk",
        )
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

    logger.bind(
        input_path=str(source_path),
        output_path=str(output_path),
        sample_rate=target_sample_rate,
        channels=channels,
        audio_filter=ffmpeg_audio_filter,
    ).info("Converting audio for pipeline")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not available in PATH. "
            "Install ffmpeg to enable audio conversion.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}") from exc

    return output_path

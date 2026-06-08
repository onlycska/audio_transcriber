"""File-system transcript writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from src.service.interfaces.transcript_writer import ITranscriptWriter
from src.service.models.segment import DialogueLine


class Clock(Protocol):
    """Clock dependency used to generate transcript timestamps."""

    def __call__(self) -> datetime:
        """Return current datetime.

        Args:
            self: Clock implementation instance.

        Returns:
            Current datetime.
        """
        ...


class FileTranscriptWriter(ITranscriptWriter):
    """Persist transcript lines in the results directory.

    Args:
        results_dir: Directory where transcript files are written.
        clock: Callable returning current datetime. Defaults to
            :meth:`~datetime.datetime.now`.
    """

    def __init__(self, *, results_dir: Path, clock: Clock = datetime.now) -> None:
        """Initialize writer.

        Args:
            results_dir: Directory where transcript files are written.
            clock: Callable returning current datetime.
        """
        self._results_dir = results_dir
        self._clock = clock

    def write(self, lines: list[DialogueLine], *, audio_stem: str) -> Path:
        """Write dialogue lines to timestamped UTF-8 text file.

        Args:
            lines: Final transcript lines.
            audio_stem: Source audio file stem used in output file name.

        Returns:
            Path to the written transcript file.
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._clock().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = self._results_dir / f"transcript_{audio_stem}_{timestamp}.txt"
        output_path.write_text(
            "\n".join(f"[{item.speaker}] {item.text}" for item in lines),
            encoding="utf-8",
        )
        return output_path

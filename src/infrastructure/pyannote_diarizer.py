"""Pyannote-based implementation of speaker diarization interface."""

from __future__ import annotations

import torch
from pyannote.audio import Pipeline

from src.service.interfaces.diarizer import IDiarizer
from src.service.models.audio import AudioWaveform
from src.service.models.segment import DiarizationTurn


class PyannoteDiarizer(IDiarizer):
    """Diarizer that delegates segmentation to pyannote pipeline.

    Args:
        hf_token: Hugging Face access token accepted by PyAnnote.
        model_name: PyAnnote model id. Defaults to
            ``"pyannote/speaker-diarization-3.1"``.
    """

    def __init__(
        self,
        *,
        hf_token: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        """Initialize PyAnnote pipeline.

        Args:
            hf_token: Hugging Face access token.
            model_name: PyAnnote model id.

        Raises:
            RuntimeError: If PyAnnote cannot load the requested model.
        """
        loaded_pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        if loaded_pipeline is None:
            raise RuntimeError(f"Failed to load PyAnnote pipeline: {model_name}")
        self._pipeline = loaded_pipeline
        if torch.backends.mps.is_available():
            self._pipeline.to(torch.device("mps"))

    def diarize(self, audio: AudioWaveform, *, num_speakers: int) -> list[DiarizationTurn]:
        """Run pyannote diarization for loaded waveform.

        Args:
            audio: Loaded waveform and sample rate.
            num_speakers: Expected number of speakers.

        Returns:
            Ordered diarization turns.
        """
        waveform_tensor = torch.tensor(audio.samples.T)
        diarization = self._pipeline(
            {"waveform": waveform_tensor, "sample_rate": audio.sample_rate},
            num_speakers=num_speakers,
        ).speaker_diarization
        return [
            DiarizationTurn(
                start=float(segment.start),
                end=float(segment.end),
                speaker=str(speaker),
            )
            for segment, _, speaker in diarization.itertracks(yield_label=True)
        ]

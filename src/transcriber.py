"""Audio transcription with diarization and overlap-safe chunking."""

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
import whisper
from loguru import logger
from pyannote.audio import Pipeline

from src.infrastructure.audio_conversion import ensure_wav_for_pipeline
from src.infrastructure.logging import configure_logging
from src.service.transcription import mixdown_to_mono, transcribe_in_chunks

HF_TOKEN = os.getenv("HF_TOKEN", "your_token")
AUDIO_FILE = os.getenv("AUDIO_FILE", "/path/to/audiofile.wav")
NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "300"))
OVERLAP_SECONDS = float(os.getenv("OVERLAP_SECONDS", "3.0"))
people_word = "человека" if NUM_SPEAKERS == 1 else "людей"
DEFAULT_INITIAL_PROMPT = f"Разговор {NUM_SPEAKERS} {people_word} на русском языке."
INITIAL_PROMPT = os.getenv("INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT).strip() or None
LANGUAGE = os.getenv("LANGUAGE", "ru")

configure_logging()
logger.info("Application startup")

# 1. Загружаем pipeline
logger.info("Загружаем diarization модель...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
)

if torch.backends.mps.is_available():
    diarization_pipeline.to(torch.device("mps"))
    logger.info("Используем MPS")
else:
    logger.info("Используем CPU")

# 2. Предзагружаем аудио через soundfile → обходим torchcodec
logger.info("Определяем спикеров...")
audio_path = ensure_wav_for_pipeline(Path(AUDIO_FILE))
waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
# soundfile возвращает (samples, channels), pyannote ждёт (channels, samples)
waveform_tensor = torch.tensor(waveform.T)
audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

diarization = diarization_pipeline(audio_input, num_speakers=NUM_SPEAKERS).speaker_diarization

# 3. Whisper
logger.info(
    f"Транскрибируем: model={WHISPER_MODEL}, language={LANGUAGE}, "
    f"initial_prompt_set={INITIAL_PROMPT is not None}"
)
model = whisper.load_model(WHISPER_MODEL)
audio_mono = mixdown_to_mono(waveform)
segments = transcribe_in_chunks(
    model,
    audio_mono,
    sample_rate,
    chunk_seconds=CHUNK_SECONDS,
    overlap_seconds=OVERLAP_SECONDS,
    language=LANGUAGE,
    initial_prompt=INITIAL_PROMPT,
)


# 4. Сопоставление спикеров
def get_speaker(start: float, end: float, diarization) -> str:
    """
    Находит спикера с максимальным перекрытием с заданным сегментом.

    :param start: начало сегмента (сек)
    :param end: конец сегмента (сек)
    :param diarization: pyannote Annotation объект
    :return: метка спикера, например 'SPEAKER_00'
    """
    max_overlap = 0
    best_speaker = "Unknown"
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        overlap = min(segment.end, end) - max(segment.start, start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker
    return best_speaker


# 5. Назначаем спикера каждому сегменту
tagged = [
    {"speaker": get_speaker(seg["start"], seg["end"], diarization), "text": seg["text"].strip()}
    for seg in segments
]


# 6. Склейка соседних реплик одного спикера
def merge_same_speaker(
    tagged: list[dict[str, str]],
    segments: Sequence[dict[str, float | str]],
    min_merge_gap: float = 1.5,
) -> list[dict[str, str]]:
    """
    Склеивает соседние реплики одного спикера если пауза между ними <= min_merge_gap.

    :param tagged: список dict с ключами 'speaker' и 'text'
    :param segments: список сегментов whisper с 'start'/'end'
    :param min_merge_gap: максимальная пауза (сек) для склейки
    :return: список объединённых реплик
    """
    if not tagged:
        return []

    merged = [{"speaker": tagged[0]["speaker"], "text": tagged[0]["text"]}]

    for i in range(1, len(tagged)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        same_speaker = tagged[i]["speaker"] == merged[-1]["speaker"]

        if same_speaker and gap <= min_merge_gap:
            merged[-1]["text"] += " " + tagged[i]["text"]
        else:
            merged.append({"speaker": tagged[i]["speaker"], "text": tagged[i]["text"]})

    return merged


merged = merge_same_speaker(tagged, segments)

# 7. Вывод и сохранение
lines = [f"[{item['speaker']}] {item['text']}" for item in merged]
logger.info(f"Транскрипция готова: segments={len(segments)}, merged_replicas={len(merged)}")
logger.info(f"\n=== ТРАНСКРИПЦИЯ ===\n{'\n'.join(lines)}")

results_dir = Path("../results")
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = results_dir / f"transcript_{audio_path.stem}_{timestamp}.txt"
output_path.write_text("\n".join(lines), encoding="utf-8")
logger.success(f"Сохранено в {output_path}")

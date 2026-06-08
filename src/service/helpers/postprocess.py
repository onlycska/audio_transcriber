"""Postprocessing helpers for speaker-tagged dialogue timeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from loguru import logger

from src.service.helpers.speakers import is_short_backchannel, normalize_token
from src.service.models.segment import DialogueLine, TaggedSegment

_TRAILING_SERVICE_PATTERNS = [
    re.compile(r"^спасибо(?:\s+спасибо)*[.!?]*$", re.IGNORECASE),
    re.compile(r"^пока[.!?]*$", re.IGNORECASE),
    re.compile(r"^угу[.!?]*$", re.IGNORECASE),
    re.compile(r"^продолжение\s+следует[.!?]*$", re.IGNORECASE),
]


def collapse_repeated_phrases(text: str, *, max_repeats: int = 2) -> tuple[str, int]:
    """Collapse phrase loops like 'Спасибо. Спасибо. Спасибо.'.

    Args:
        text: Source text.
        max_repeats: Maximum adjacent repeats to keep. Defaults to ``2``.

    Returns:
        Tuple with cleaned text and number of removed repeats.
    """
    parts = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
    if len(parts) > 1:
        collapsed: list[str] = []
        repeats_removed = 0
        previous = ""
        count = 0
        for part in parts:
            normalized = normalize_token(part)
            if normalized == previous:
                count += 1
                if count > max_repeats:
                    repeats_removed += 1
                    continue
            else:
                previous = normalized
                count = 1
            collapsed.append(part)
        if collapsed:
            return ". ".join(collapsed) + ".", repeats_removed

    words = [item for item in re.split(r"\s+", text.strip()) if item]
    if not words:
        return text, 0

    repeats_removed = 0
    collapsed_words: list[str] = []
    previous = ""
    count = 0
    for word in words:
        normalized = normalize_token(word)
        if normalized == previous:
            count += 1
            if count > max_repeats:
                repeats_removed += 1
                continue
        else:
            previous = normalized
            count = 1
        collapsed_words.append(word)
    return " ".join(collapsed_words), repeats_removed


def _is_trailing_service_text(text: str) -> bool:
    """Check whether text is a service phrase near transcript end.

    Args:
        text: Candidate text.

    Returns:
        ``True`` when text matches known trailing service noise.
    """
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in _TRAILING_SERVICE_PATTERNS)


def _prune_trailing_noise(items: list[TaggedSegment]) -> tuple[list[TaggedSegment], int]:
    """Remove known service-noise lines from transcript tail.

    Args:
        items: Speaker-tagged segments after unknown-line filtering.

    Returns:
        Tuple with kept segments and number of removed tail segments.
    """
    if not items:
        return [], 0
    start_guard = max(0, len(items) - 10)
    trailing_removed = 0
    kept = list(items)
    while len(kept) > start_guard:
        current = kept[-1]
        if not _is_trailing_service_text(current.text):
            break
        trailing_removed += 1
        kept.pop()
    return kept, trailing_removed


def postprocess_dialogue_timeline(
    timeline: Sequence[TaggedSegment],
    *,
    aggressive_mode: bool = False,
    debug: bool = False,
    flicker_max_duration: float = 0.5,
) -> tuple[list[DialogueLine], dict[str, int]]:
    """Clean and merge speaker-tagged timeline for final transcript output.

    Args:
        timeline: Speaker-tagged ASR timeline.
        aggressive_mode: Whether to reassign short unknown lines. Defaults to
            ``False``.
        debug: Whether to emit postprocessing audit logs. Defaults to
            ``False``.

    Returns:
        Tuple with final dialogue lines and audit counters.
    """
    audit = {
        "unknown_reassigned": 0,
        "collapsed_stutter_loop": 0,
        "merged_adjacent_same_speaker": 0,
        "removed_trailing_noise": 0,
        "removed_unknown_lines": 0,
        "flicker_reassigned": 0,
        "short_backchannel_kept": 0,
        "short_backchannel_reassigned": 0,
    }

    cleaned: list[TaggedSegment] = []
    for item in timeline:
        text = item.text.strip()
        collapsed_text, removed_count = collapse_repeated_phrases(text)
        if removed_count > 0:
            audit["collapsed_stutter_loop"] += removed_count
            if debug:
                logger.bind(
                    postprocess_event="collapsed_stutter_loop",
                    removed=removed_count,
                    before=text,
                    after=collapsed_text,
                ).info("Collapsed repeated phrase loop")
        cleaned.append(
            TaggedSegment(
                start=item.start,
                end=item.end,
                speaker=item.speaker,
                raw_speaker=item.raw_speaker,
                text=collapsed_text.strip(),
            ),
        )

    if aggressive_mode and len(cleaned) >= 3:
        for idx in range(1, len(cleaned) - 1):
            current = cleaned[idx]
            if current.speaker != "Unknown":
                continue
            left = cleaned[idx - 1]
            right = cleaned[idx + 1]
            left_gap = current.start - left.end
            right_gap = right.start - current.end
            if left_gap > 1.5 and right_gap > 1.5:
                continue
            if left.speaker == right.speaker:
                assigned = left.speaker
            else:
                assigned = left.speaker if left_gap <= right_gap else right.speaker
            cleaned[idx] = TaggedSegment(
                start=current.start,
                end=current.end,
                speaker=assigned,
                raw_speaker=current.raw_speaker,
                text=current.text,
            )
            audit["unknown_reassigned"] += 1

    if aggressive_mode and cleaned:
        for idx, current in enumerate(cleaned):
            if current.speaker != "Unknown":
                continue
            if not is_short_backchannel(current.text, max_words=8, max_chars=80):
                continue

            nearest_speaker = "Unknown"
            nearest_gap = float("inf")
            for step in (-1, 1):
                pos = idx + step
                while 0 <= pos < len(cleaned):
                    candidate = cleaned[pos]
                    if candidate.speaker == "Unknown":
                        pos += step
                        continue
                    gap = (
                        current.start - candidate.end
                        if step == -1
                        else candidate.start - current.end
                    )
                    if gap < nearest_gap:
                        nearest_gap = gap
                        nearest_speaker = candidate.speaker
                    break
            if nearest_speaker != "Unknown" and nearest_gap <= 1.5:
                cleaned[idx] = TaggedSegment(
                    start=current.start,
                    end=current.end,
                    speaker=nearest_speaker,
                    raw_speaker=current.raw_speaker,
                    text=current.text,
                )
                audit["unknown_reassigned"] += 1

    if aggressive_mode and len(cleaned) >= 3:
        for idx in range(1, len(cleaned) - 1):
            left = cleaned[idx - 1]
            middle = cleaned[idx]
            right = cleaned[idx + 1]
            if left.speaker != right.speaker or middle.speaker == left.speaker:
                continue
            middle_duration = middle.end - middle.start
            is_short = is_short_backchannel(middle.text, max_words=4, max_chars=20)
            if not is_short or middle_duration > flicker_max_duration:
                continue

            question_context = left.text.strip().endswith("?")
            if question_context:
                audit["short_backchannel_kept"] += 1
                continue

            cleaned[idx] = TaggedSegment(
                start=middle.start,
                end=middle.end,
                speaker=left.speaker,
                raw_speaker=middle.raw_speaker,
                text=middle.text,
            )
            audit["flicker_reassigned"] += 1
            audit["short_backchannel_reassigned"] += 1

    filtered_unknown: list[TaggedSegment] = []
    for idx, item in enumerate(cleaned):
        if item.speaker == "Unknown":
            audit["removed_unknown_lines"] += 1
            if debug:
                duration = max(0.0, item.end - item.start)
                left_neighbor = cleaned[idx - 1].speaker if idx > 0 else None
                right_neighbor = cleaned[idx + 1].speaker if idx + 1 < len(cleaned) else None
                logger.bind(
                    postprocess_event="drop_unknown",
                    reason="drop_unknown",
                    start=round(item.start, 2),
                    end=round(item.end, 2),
                    duration=round(duration, 2),
                    left_neighbor=left_neighbor,
                    right_neighbor=right_neighbor,
                    text=item.text,
                ).info("Dropped unknown speaker line")
            continue
        filtered_unknown.append(item)

    cleaned, removed_trailing = _prune_trailing_noise(filtered_unknown)
    audit["removed_trailing_noise"] = removed_trailing

    merged: list[DialogueLine] = []
    for item in cleaned:
        text = item.text.strip()
        if not text:
            continue
        if not merged:
            merged.append(DialogueLine(speaker=item.speaker, text=text))
            continue
        prev = merged[-1]
        if prev.speaker == item.speaker:
            combined = f"{prev.text.rstrip('.!?')} {text.lstrip()}"
            if not combined.endswith((".", "!", "?")):
                combined += "."
            merged[-1] = DialogueLine(speaker=prev.speaker, text=combined)
            audit["merged_adjacent_same_speaker"] += 1
        else:
            merged.append(DialogueLine(speaker=item.speaker, text=text))

    return merged, audit

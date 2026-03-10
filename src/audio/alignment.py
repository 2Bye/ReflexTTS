"""Alignment utilities: timestamp → mel-frame index conversion.

Converts WhisperX word-level timestamps (ms) into mel-spectrogram
frame indices for precise masking during latent inpainting.

Usage:
    frames = ms_to_mel_frames(start_ms=2450.0, end_ms=3100.0,
                               sample_rate=22050, hop_length=256)
    mask = create_error_regions(errors, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class MelRegion:
    """A region in mel-spectrogram space.

    Attributes:
        start_frame: Start frame index (inclusive).
        end_frame: End frame index (exclusive).
        start_ms: Original start time in ms.
        end_ms: Original end time in ms.
        word_expected: Word that should have been spoken.
        word_actual: Word that was actually recognized.
    """

    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    word_expected: str = ""
    word_actual: str = ""


def ms_to_mel_frames(
    start_ms: float,
    end_ms: float,
    sample_rate: int = 22050,
    hop_length: int = 256,
    padding_ms: float = 50.0,
) -> tuple[int, int]:
    """Convert millisecond timestamps to mel-spectrogram frame indices.

    Adds configurable padding around the region for smoother
    inpainting boundaries.

    Args:
        start_ms: Error region start time in milliseconds.
        end_ms: Error region end time in milliseconds.
        sample_rate: Audio sample rate in Hz.
        hop_length: STFT hop length (frames between mel columns).
        padding_ms: Padding around the error region in ms.

    Returns:
        Tuple of (start_frame, end_frame) — 0-indexed, end exclusive.
    """
    # Convert ms to samples
    start_sample = int((start_ms - padding_ms) * sample_rate / 1000)
    end_sample = int((end_ms + padding_ms) * sample_rate / 1000)

    # Clamp to non-negative
    start_sample = max(0, start_sample)

    # Convert samples to frames
    start_frame = start_sample // hop_length
    end_frame = (end_sample + hop_length - 1) // hop_length  # Ceiling division

    return start_frame, end_frame


def create_error_regions(
    errors: list[dict[str, object]],
    sample_rate: int = 22050,
    hop_length: int = 256,
    padding_ms: float = 50.0,
) -> list[MelRegion]:
    """Convert Critic errors to mel-spectrogram regions for masking.

    Args:
        errors: List of error dicts from GraphState.errors.
        sample_rate: Audio sample rate.
        hop_length: STFT hop length.
        padding_ms: Padding around each error.

    Returns:
        List of MelRegion objects, sorted by start_frame.
    """
    regions: list[MelRegion] = []

    for error in errors:
        start_ms = float(error.get("start_ms", 0.0))  # type: ignore[arg-type]
        end_ms = float(error.get("end_ms", 0.0))  # type: ignore[arg-type]

        if end_ms <= start_ms:
            continue

        start_frame, end_frame = ms_to_mel_frames(
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate=sample_rate,
            hop_length=hop_length,
            padding_ms=padding_ms,
        )

        regions.append(
            MelRegion(
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=start_ms,
                end_ms=end_ms,
                word_expected=str(error.get("word_expected", "")),
                word_actual=str(error.get("word_actual", "")),
            )
        )

    # Sort by start frame
    regions.sort(key=lambda r: r.start_frame)

    # Merge overlapping regions
    merged = _merge_overlapping(regions)

    logger.debug(
        "alignment_regions_created",
        raw_count=len(regions),
        merged_count=len(merged),
    )
    return merged


def _merge_overlapping(regions: list[MelRegion]) -> list[MelRegion]:
    """Merge overlapping or adjacent mel regions."""
    if not regions:
        return []

    merged: list[MelRegion] = [regions[0]]

    for region in regions[1:]:
        last = merged[-1]
        if region.start_frame <= last.end_frame:
            # Overlapping — extend the last region
            merged[-1] = MelRegion(
                start_frame=last.start_frame,
                end_frame=max(last.end_frame, region.end_frame),
                start_ms=min(last.start_ms, region.start_ms),
                end_ms=max(last.end_ms, region.end_ms),
                word_expected=f"{last.word_expected} {region.word_expected}".strip(),
                word_actual=f"{last.word_actual} {region.word_actual}".strip(),
            )
        else:
            merged.append(region)

    return merged

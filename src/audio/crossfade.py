"""Spectral cross-fade for chunk-based audio repair.

Fallback correction path when latent inpainting is not available.
Regenerates the entire error chunk and blends it with the original
via equal-power cross-fade at splice boundaries.
"""

from __future__ import annotations

import numpy as np

from src.log import get_logger

logger = get_logger(__name__)


def crossfade_chunks(
    original: np.ndarray,
    replacement: np.ndarray,
    start_sample: int,
    end_sample: int,
    *,
    crossfade_samples: int = 512,
) -> np.ndarray:
    """Replace a chunk in the original audio with cross-faded boundaries.

    Args:
        original: Original full waveform (float32, mono).
        replacement: Replacement audio for the error region.
        start_sample: Start sample index of the region to replace.
        end_sample: End sample index (exclusive).
        crossfade_samples: Samples for cross-fade (default 512).

    Returns:
        New waveform with the replacement blended in.
    """
    result = original.copy()
    chunk_length = end_sample - start_sample

    # Resize replacement to match chunk
    if len(replacement) != chunk_length:
        replacement = _resize_audio(replacement, chunk_length)

    # Simply place the replacement
    result[start_sample:end_sample] = replacement

    # Left cross-fade: blend original→replacement at start boundary
    left_n = min(crossfade_samples, start_sample, chunk_length // 2)
    if left_n > 0:
        fade = np.linspace(0.0, 1.0, left_n, dtype=np.float32)
        zone_start = start_sample - left_n
        zone_end = start_sample
        result[zone_start:zone_end] = (
            original[zone_start:zone_end] * (1.0 - fade)
            + replacement[:left_n] * fade
        )

    # Right cross-fade: blend replacement→original at end boundary
    right_n = min(crossfade_samples, len(original) - end_sample, chunk_length // 2)
    if right_n > 0:
        fade = np.linspace(1.0, 0.0, right_n, dtype=np.float32)
        zone_start = end_sample
        zone_end = end_sample + right_n
        result[zone_start:zone_end] = (
            replacement[-right_n:] * fade
            + original[zone_start:zone_end] * (1.0 - fade)
        )

    logger.debug(
        "crossfade_applied",
        start_sample=start_sample,
        end_sample=end_sample,
        crossfade_samples=crossfade_samples,
    )
    return result


def _equal_power_fade(length: int, *, fade_in: bool = True) -> np.ndarray:
    """Generate an equal-power (cosine) fade curve.

    Args:
        length: Number of samples.
        fade_in: If True, 0->1. If False, 1->0.

    Returns:
        Fade curve as float32 array.
    """
    t = np.linspace(0, np.pi / 2, length, dtype=np.float32)
    if fade_in:
        return np.sin(t)
    return np.cos(t)


def _resize_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Resize audio to target length by truncating or zero-padding."""
    if len(audio) >= target_length:
        return audio[:target_length]
    padded = np.zeros(target_length, dtype=audio.dtype)
    padded[: len(audio)] = audio
    return padded

"""Binary masking for mel-spectrogram inpainting.

Creates masks that specify which mel-spectrogram regions to
regenerate (0) vs preserve (1) during Flow Matching inpainting.

Usage:
    mask = build_inpainting_mask(regions, total_frames=500, n_mels=80)
    # mask.shape == (80, 500), values in {0.0, 1.0}
"""

from __future__ import annotations

import numpy as np

from src.audio.alignment import MelRegion
from src.log import get_logger

logger = get_logger(__name__)


def build_inpainting_mask(
    regions: list[MelRegion],
    total_frames: int,
    n_mels: int = 80,
    *,
    taper_frames: int = 4,
) -> np.ndarray:
    """Build a binary inpainting mask for mel-spectrogram.

    The mask has value 1.0 where the original mel should be kept,
    and 0.0 where Flow Matching should regenerate.
    Optionally applies a cosine taper at region boundaries for
    smoother transitions (avoids hard edges).

    Args:
        regions: Mel regions to mask (from alignment).
        total_frames: Total number of mel frames.
        n_mels: Number of mel frequency bins.
        taper_frames: Number of frames for cosine taper (0 to disable).

    Returns:
        Mask array of shape (n_mels, total_frames), float32.
        1.0 = keep original, 0.0 = regenerate.
    """
    mask = np.ones((n_mels, total_frames), dtype=np.float32)

    for region in regions:
        start = max(0, region.start_frame)
        end = min(total_frames, region.end_frame)

        if start >= end:
            continue

        # Zero out the error region
        mask[:, start:end] = 0.0

        # Apply cosine taper at boundaries for smooth transitions
        if taper_frames > 0:
            _apply_taper(mask, start, end, total_frames, taper_frames)

    masked_frames = int(np.sum(mask[0, :] < 1.0))
    logger.debug(
        "mask_built",
        total_frames=total_frames,
        masked_frames=masked_frames,
        mask_ratio=f"{masked_frames / total_frames:.2%}" if total_frames > 0 else "0%",
    )
    return mask


def _apply_taper(
    mask: np.ndarray,
    start: int,
    end: int,
    total_frames: int,
    taper_frames: int,
) -> None:
    """Apply cosine taper at mask boundaries (in-place).

    Creates a smooth transition from 1→0 at the start and 0→1 at
    the end of the masked region, preventing spectral artifacts.
    """
    # Left taper: 1 → 0
    left_start = max(0, start - taper_frames)
    left_end = start
    if left_end > left_start:
        n = left_end - left_start
        taper = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n)))  # 1 → 0
        mask[:, left_start:left_end] *= taper.astype(np.float32)

    # Right taper: 0 → 1
    right_start = end
    right_end = min(total_frames, end + taper_frames)
    if right_end > right_start:
        n = right_end - right_start
        taper = 0.5 * (1 + np.cos(np.linspace(np.pi, 0, n)))  # 0 → 1
        mask[:, right_start:right_end] *= taper.astype(np.float32)


def apply_mask_to_mel(
    mel_original: np.ndarray,
    mel_generated: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend original and generated mel-spectrograms using mask.

    Implements: result = mask * mel_original + (1 - mask) * mel_generated

    This is the core inpainting operation:
    - Where mask=1 → keep the original audio
    - Where mask=0 → use the regenerated audio

    Args:
        mel_original: Original mel-spectrogram.
        mel_generated: New mel from Flow Matching.
        mask: Inpainting mask (same shape as mels).

    Returns:
        Blended mel-spectrogram.
    """
    result: np.ndarray = (mask * mel_original + (1 - mask) * mel_generated).astype(np.float32)
    return result

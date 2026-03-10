"""Convergence metrics for audio quality assessment.

Combines multiple signal-quality metrics into a single convergence
score used by the orchestrator to decide whether inpainting succeeded.

Score = a*(1 - WER) + b*SECS + g*normalized_PESQ

Usage:
    score = convergence_score(wer=0.02, secs=0.92, pesq=3.8)
    # → 0.93 (pass threshold: 0.85)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.log import get_logger

logger = get_logger(__name__)

# Default weights for convergence score components
ALPHA_WER: float = 0.5   # Word Error Rate weight (dominant)
BETA_SECS: float = 0.3   # Speaker Embedding Cosine Similarity weight
GAMMA_PESQ: float = 0.2  # PESQ perceptual quality weight

# Thresholds
CONVERGENCE_THRESHOLD: float = 0.85
PESQ_MAX: float = 4.5    # Maximum PESQ score (ITU-T P.862)


@dataclass
class QualityMetrics:
    """Collection of audio quality metrics.

    Attributes:
        wer: Word Error Rate (0-1, lower is better).
        secs: Speaker Embedding Cosine Similarity (0-1, higher is better).
        pesq: PESQ score (1.0-4.5, higher is better).
        convergence_score: Weighted composite score (0-1, higher is better).
        is_converged: Whether the score exceeds the threshold.
    """

    wer: float = 1.0
    secs: float = 0.0
    pesq: float = 1.0
    convergence_score: float = 0.0
    is_converged: bool = False


def convergence_score(
    wer: float,
    secs: float = 1.0,
    pesq: float = 4.5,
    *,
    alpha: float = ALPHA_WER,
    beta: float = BETA_SECS,
    gamma: float = GAMMA_PESQ,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> QualityMetrics:
    """Calculate weighted convergence score.

    Score = a*(1 - WER) + b*SECS + g*(PESQ / PESQ_MAX)

    Args:
        wer: Word Error Rate (0-1).
        secs: Speaker Embedding Cosine Similarity (0-1).
        pesq: PESQ score (1.0-4.5).
        alpha: Weight for WER component.
        beta: Weight for SECS component.
        gamma: Weight for PESQ component.
        threshold: Minimum score for convergence.

    Returns:
        QualityMetrics with all scores and convergence flag.
    """
    # Normalize components to [0, 1]
    wer_score = 1.0 - min(1.0, max(0.0, wer))
    secs_score = min(1.0, max(0.0, secs))
    pesq_score = min(1.0, max(0.0, (pesq - 1.0) / (PESQ_MAX - 1.0)))

    score = alpha * wer_score + beta * secs_score + gamma * pesq_score

    metrics = QualityMetrics(
        wer=wer,
        secs=secs,
        pesq=pesq,
        convergence_score=float(np.clip(score, 0.0, 1.0)),
        is_converged=score >= threshold,
    )

    logger.debug(
        "convergence_calculated",
        wer=f"{wer:.3f}",
        secs=f"{secs:.3f}",
        pesq=f"{pesq:.2f}",
        score=f"{metrics.convergence_score:.3f}",
        converged=metrics.is_converged,
    )
    return metrics


def compute_secs(
    embedding_original: np.ndarray,
    embedding_repaired: np.ndarray,
) -> float:
    """Compute Speaker Embedding Cosine Similarity (SECS).

    Measures whether the repaired audio maintains the same
    speaker identity as the original.

    Args:
        embedding_original: Speaker embedding of original audio.
        embedding_repaired: Speaker embedding of repaired audio.

    Returns:
        Cosine similarity in range [-1, 1] (typically > 0.85 for same speaker).
    """
    norm_a = np.linalg.norm(embedding_original)
    norm_b = np.linalg.norm(embedding_repaired)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    similarity = float(np.dot(embedding_original, embedding_repaired) / (norm_a * norm_b))
    return similarity

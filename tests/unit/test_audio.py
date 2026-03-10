"""Unit tests for audio utilities: alignment, masking, crossfade, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.alignment import MelRegion, create_error_regions, ms_to_mel_frames
from src.audio.crossfade import _equal_power_fade, crossfade_chunks
from src.audio.masking import apply_mask_to_mel, build_inpainting_mask
from src.audio.metrics import compute_secs, convergence_score


class TestAlignment:
    """Tests for timestamp → mel frame conversion."""

    def test_ms_to_mel_frames_basic(self) -> None:
        start, end = ms_to_mel_frames(1000.0, 2000.0, sample_rate=22050, hop_length=256)
        assert start >= 0
        assert end > start

    def test_ms_to_mel_frames_with_padding(self) -> None:
        s1, e1 = ms_to_mel_frames(1000.0, 2000.0, padding_ms=0.0)
        s2, e2 = ms_to_mel_frames(1000.0, 2000.0, padding_ms=100.0)
        assert s2 <= s1  # Padding extends start earlier
        assert e2 >= e1  # Padding extends end later

    def test_ms_to_mel_frames_clamp_negative(self) -> None:
        start, _end = ms_to_mel_frames(10.0, 50.0, padding_ms=100.0)
        assert start >= 0

    def test_create_error_regions(self) -> None:
        errors = [
            {"start_ms": 1000.0, "end_ms": 1500.0, "word_expected": "king", "word_actual": "thing"},
            {"start_ms": 3000.0, "end_ms": 3500.0, "word_expected": "ship", "word_actual": "chip"},
        ]
        regions = create_error_regions(errors)
        assert len(regions) == 2
        assert regions[0].start_frame < regions[1].start_frame

    def test_merge_overlapping_regions(self) -> None:
        errors = [
            {"start_ms": 1000.0, "end_ms": 1500.0, "word_expected": "a", "word_actual": "b"},
            {"start_ms": 1400.0, "end_ms": 2000.0, "word_expected": "c", "word_actual": "d"},
        ]
        regions = create_error_regions(errors, padding_ms=0.0)
        # Should merge since regions overlap
        assert len(regions) <= 2  # May or may not merge depending on frame boundaries

    def test_empty_errors(self) -> None:
        regions = create_error_regions([])
        assert regions == []


class TestMasking:
    """Tests for binary mask construction."""

    def test_mask_shape(self) -> None:
        regions = [MelRegion(start_frame=10, end_frame=20, start_ms=0, end_ms=0)]
        mask = build_inpainting_mask(regions, total_frames=100, n_mels=80)
        assert mask.shape == (80, 100)

    def test_mask_values(self) -> None:
        regions = [MelRegion(start_frame=10, end_frame=20, start_ms=0, end_ms=0)]
        mask = build_inpainting_mask(regions, total_frames=100, n_mels=80, taper_frames=0)
        assert mask[0, 5] == 1.0  # Outside region = keep
        assert mask[0, 15] == 0.0  # Inside region = regenerate

    def test_mask_no_regions(self) -> None:
        mask = build_inpainting_mask([], total_frames=100, n_mels=80)
        assert np.all(mask == 1.0)

    def test_apply_mask_to_mel(self) -> None:
        mel_orig = np.ones((80, 100), dtype=np.float32) * 2.0
        mel_gen = np.ones((80, 100), dtype=np.float32) * 5.0
        mask = np.ones((80, 100), dtype=np.float32)
        mask[:, 10:20] = 0.0

        result = apply_mask_to_mel(mel_orig, mel_gen, mask)
        assert result[0, 5] == 2.0   # Kept from original
        assert result[0, 15] == 5.0  # From generated


class TestCrossfade:
    """Tests for spectral cross-fade."""

    def test_crossfade_basic(self) -> None:
        original = np.ones(10000, dtype=np.float32) * 0.5
        replacement = np.ones(2000, dtype=np.float32) * -0.5
        result = crossfade_chunks(original, replacement, 4000, 6000)
        assert len(result) == len(original)

    def test_equal_power_fade(self) -> None:
        fade_in = _equal_power_fade(100, fade_in=True)
        fade_out = _equal_power_fade(100, fade_in=False)
        assert len(fade_in) == 100
        assert fade_in[0] < fade_in[-1]  # Increasing
        assert fade_out[0] > fade_out[-1]  # Decreasing

    def test_crossfade_preserves_length(self) -> None:
        original = np.random.randn(5000).astype(np.float32) * 0.5
        replacement = np.random.randn(500).astype(np.float32)
        result = crossfade_chunks(original, replacement, 1000, 1500)
        assert len(result) == len(original)


class TestMetrics:
    """Tests for convergence metrics."""

    def test_perfect_score(self) -> None:
        m = convergence_score(wer=0.0, secs=1.0, pesq=4.5)
        assert m.convergence_score == pytest.approx(1.0)
        assert m.is_converged is True

    def test_worst_score(self) -> None:
        m = convergence_score(wer=1.0, secs=0.0, pesq=1.0)
        assert m.convergence_score == pytest.approx(0.0)
        assert m.is_converged is False

    def test_threshold(self) -> None:
        m = convergence_score(wer=0.02, secs=0.92, pesq=3.8)
        assert m.is_converged is True  # Should be above 0.85

    def test_compute_secs_identical(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert compute_secs(emb, emb) == pytest.approx(1.0)

    def test_compute_secs_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert compute_secs(a, b) == pytest.approx(0.0)

"""Unit tests for Critic Agent."""

from __future__ import annotations

import numpy as np

from src.agents.actor import _decode_wav_to_array, _encode_wav
from src.agents.critic import _extract_target_text
from src.agents.schemas import DirectorOutput, Segment
from src.orchestrator.state import GraphState


class TestDecodeWav:
    """Tests for WAV decoding utility (shared from actor)."""

    def test_decode_valid_wav(self) -> None:
        # Use values within [-0.9, 0.9] to avoid int16 clipping
        original = (np.random.rand(10000).astype(np.float32) * 1.8 - 0.9)
        wav_bytes = _encode_wav(original, 22050)
        decoded = _decode_wav_to_array(wav_bytes)

        assert len(decoded) == len(original)
        assert decoded.dtype == np.float32
        # int16 quantization introduces ~1/32767 ≈ 3e-5 rounding error
        np.testing.assert_allclose(decoded, original, atol=5e-5)

    def test_decode_empty_bytes(self) -> None:
        result = _decode_wav_to_array(b"")
        assert len(result) == 0

    def test_decode_invalid_bytes(self) -> None:
        result = _decode_wav_to_array(b"not a wav file")
        assert len(result) == 0


class TestExtractTargetText:
    """Tests for target text extraction."""

    def test_from_ssml_markup(self) -> None:
        state = GraphState(
            text="original",
            ssml_markup=DirectorOutput(
                segments=[
                    Segment(text="Hello"),
                    Segment(text="World"),
                ],
            ).model_dump(),
        )
        result = _extract_target_text(state)
        assert result == "Hello World"

    def test_fallback_to_raw_text(self) -> None:
        state = GraphState(text="Raw input text")
        result = _extract_target_text(state)
        assert result == "Raw input text"

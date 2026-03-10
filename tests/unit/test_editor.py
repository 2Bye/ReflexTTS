"""Unit tests for Editor Agent."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.actor import _encode_wav
from src.agents.editor import run_editor
from src.config import CosyVoiceConfig
from src.inference.tts_client import TTSClient
from src.orchestrator.state import DetectedError, ErrorSeverity, GraphState


@pytest.fixture
def tts_client() -> TTSClient:
    return TTSClient(CosyVoiceConfig(model_dir="test"))


class TestEditor:
    """Tests for Editor Agent."""

    @pytest.mark.asyncio
    async def test_skip_no_errors(self, tts_client: TTSClient) -> None:
        """Editor skips when no errors."""
        state = GraphState(text="Hello", errors=[])
        result = await run_editor(state, tts_client)
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_skip_all_hotfixable(self, tts_client: TTSClient) -> None:
        """Editor skips when all errors are hotfixable."""
        state = GraphState(
            text="Hello",
            errors=[
                DetectedError(
                    word_expected="test",
                    word_actual="tset",
                    start_ms=100,
                    end_ms=500,
                    severity=ErrorSeverity.WARNING,
                    can_hotfix=True,
                    hotfix_hint="[t][e][s][t]",
                ),
            ],
        )
        result = await run_editor(state, tts_client)
        # Should not modify audio — hotfix is handled by Director
        assert result.audio_bytes == state.audio_bytes

    @pytest.mark.asyncio
    async def test_editor_with_non_hotfix_errors(
        self, tts_client: TTSClient
    ) -> None:
        """Editor attempts repair for non-hotfixable errors.

        Without a loaded model, this should raise but be caught.
        """
        wav_bytes = _encode_wav(
            np.zeros(22050, dtype=np.float32), 22050
        )
        state = GraphState(
            text="Hello world",
            audio_bytes=wav_bytes,
            sample_rate=22050,
            errors=[
                DetectedError(
                    word_expected="world",
                    word_actual="wold",
                    start_ms=500,
                    end_ms=1000,
                    severity=ErrorSeverity.CRITICAL,
                    can_hotfix=False,
                ),
            ],
        )
        # Editor will try inpainting then chunk regen — both will fail
        # without a loaded model, but won't crash (graceful fallback)
        result = await run_editor(state, tts_client)
        # Audio should still exist (unchanged since regen fails)
        assert len(result.audio_bytes) > 0

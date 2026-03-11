"""Unit tests for Editor Agent."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.actor import _encode_wav
from src.agents.editor import (
    _get_failed_segments,
    _rebuild_combined_audio,
    run_editor,
)
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


class TestGetFailedSegments:
    """Tests for segment failure detection."""

    def test_from_segment_approved(self) -> None:
        state = GraphState(
            text="Hello world",
            segment_approved=[True, False, True, False],
            errors=[],
        )
        failed = _get_failed_segments(state)
        assert failed == [1, 3]

    def test_from_error_segment_index(self) -> None:
        state = GraphState(
            text="Hello world",
            segment_approved=[True, True],
            errors=[
                DetectedError(
                    word_expected="world",
                    word_actual="wold",
                    start_ms=500,
                    end_ms=1000,
                    severity=ErrorSeverity.CRITICAL,
                    can_hotfix=False,
                    segment_index=1,
                ),
            ],
        )
        failed = _get_failed_segments(state)
        assert failed == [1]

    def test_combined_sources(self) -> None:
        state = GraphState(
            text="Hello world test",
            segment_approved=[False, True, False],
            errors=[
                DetectedError(
                    word_expected="test",
                    word_actual="tset",
                    start_ms=500,
                    end_ms=1000,
                    severity=ErrorSeverity.CRITICAL,
                    can_hotfix=False,
                    segment_index=2,
                ),
            ],
        )
        failed = _get_failed_segments(state)
        assert failed == [0, 2]

    def test_no_failures(self) -> None:
        state = GraphState(
            text="Hello",
            segment_approved=[True, True],
            errors=[],
        )
        failed = _get_failed_segments(state)
        assert failed == []

    def test_hotfix_errors_excluded(self) -> None:
        """Hotfixable errors don't count as segment failures."""
        state = GraphState(
            text="Hello",
            segment_approved=[True, True],
            errors=[
                DetectedError(
                    word_expected="hello",
                    word_actual="hallo",
                    start_ms=0,
                    end_ms=500,
                    severity=ErrorSeverity.WARNING,
                    can_hotfix=True,
                    segment_index=0,
                ),
            ],
        )
        failed = _get_failed_segments(state)
        assert failed == []


class TestRebuildCombinedAudio:
    """Tests for combined audio rebuild from segments."""

    def test_rebuild_basic(self) -> None:
        """Rebuild concatenates segment audio."""
        wav1 = _encode_wav(np.ones(1000, dtype=np.float32) * 0.5, 22050)
        wav2 = _encode_wav(np.ones(500, dtype=np.float32) * 0.3, 22050)
        state = GraphState(
            text="Hello world",
            segment_audio=[wav1, wav2],
            sample_rate=22050,
        )
        segments = [
            {"text": "Hello", "pause_before_ms": 0},
            {"text": "world", "pause_before_ms": 0},
        ]
        _rebuild_combined_audio(state, segments)
        assert len(state.audio_bytes) > 0

    def test_rebuild_with_pause(self) -> None:
        """Rebuild inserts pause samples between segments."""
        wav1 = _encode_wav(np.ones(2205, dtype=np.float32) * 0.5, 22050)
        state = GraphState(
            text="Hello",
            segment_audio=[wav1],
            sample_rate=22050,
        )
        segments = [
            {"text": "Hello", "pause_before_ms": 100},  # 100ms pause
        ]
        _rebuild_combined_audio(state, segments)
        # Audio should include pause + segment
        assert len(state.audio_bytes) > len(wav1)

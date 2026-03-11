"""Unit tests for the ASR client (HTTP microservice mode).

Tests use mocks — no WhisperX service or GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import WhisperXConfig
from src.inference.asr_client import (
    ASRClient,
    ASRModelNotLoadedError,
    TranscriptionResult,
    WordTimestamp,
)


@pytest.fixture
def asr_config() -> WhisperXConfig:
    """Create a test ASR config."""
    return WhisperXConfig(
        base_url="http://localhost:9999",
        model_name="tiny",
        device="cpu",
        compute_type="float32",
    )


@pytest.fixture
def client(asr_config: WhisperXConfig) -> ASRClient:
    """Create a test client (not connected)."""
    return ASRClient(asr_config)


class TestWordTimestamp:
    """Tests for WordTimestamp dataclass."""

    def test_creation(self) -> None:
        """WordTimestamp stores word and time boundaries."""
        wt = WordTimestamp(word="hello", start_ms=100.0, end_ms=500.0, score=0.95)
        assert wt.word == "hello"
        assert wt.start_ms == 100.0
        assert wt.end_ms == 500.0
        assert wt.score == 0.95

    def test_default_score(self) -> None:
        """Score defaults to 0.0."""
        wt = WordTimestamp(word="test", start_ms=0.0, end_ms=100.0)
        assert wt.score == 0.0


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_creation(self) -> None:
        """TranscriptionResult stores text and timestamps."""
        result = TranscriptionResult(
            text="hello world",
            word_timestamps=[
                WordTimestamp(word="hello", start_ms=0.0, end_ms=500.0),
                WordTimestamp(word="world", start_ms=510.0, end_ms=1000.0),
            ],
            language="en",
        )
        assert result.text == "hello world"
        assert len(result.word_timestamps) == 2
        assert result.language == "en"

    def test_empty_defaults(self) -> None:
        """Empty defaults work."""
        result = TranscriptionResult(text="")
        assert result.word_timestamps == []
        assert result.language == ""


class TestASRClient:
    """Test suite for ASRClient (HTTP mode)."""

    @pytest.mark.asyncio
    async def test_transcribe_raises_if_not_loaded(
        self, client: ASRClient
    ) -> None:
        """transcribe() raises ASRModelNotLoadedError."""
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(ASRModelNotLoadedError):
            await client.transcribe(audio)

    @pytest.mark.asyncio
    async def test_health_check_not_loaded(self, client: ASRClient) -> None:
        """health_check() returns False when not connected."""
        client._base_url = "http://localhost:1"
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_no_url(self) -> None:
        """health_check() returns False with empty base_url."""
        config = WhisperXConfig(base_url="")
        client = ASRClient(config)
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_close_resets_state(self, client: ASRClient) -> None:
        """close() releases client and resets state."""
        client._loaded = True
        client._client = None

        await client.close()

        assert client._loaded is False
        assert client._client is None

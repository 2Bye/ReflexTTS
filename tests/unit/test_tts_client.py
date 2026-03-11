"""Unit tests for the TTS client (HTTP microservice mode).

Tests use mocks — no CosyVoice service or GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CosyVoiceConfig
from src.inference.tts_client import (
    AudioResult,
    TTSClient,
    TTSGenerationError,
    TTSModelNotLoadedError,
)


@pytest.fixture
def tts_config() -> CosyVoiceConfig:
    """Create a test TTS config."""
    return CosyVoiceConfig(
        base_url="http://localhost:9999",
        model_dir="test_models/cosyvoice",
        load_vllm=False,
        load_trt=False,
    )


@pytest.fixture
def client(tts_config: CosyVoiceConfig) -> TTSClient:
    """Create a test client (not connected)."""
    return TTSClient(tts_config)


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_duration_calculated(self) -> None:
        """Duration is auto-calculated from waveform length."""
        waveform = np.zeros(22050, dtype=np.float32)  # 1 second at 22050Hz
        result = AudioResult(waveform=waveform, sample_rate=22050)
        assert abs(result.duration_seconds - 1.0) < 0.01

    def test_empty_waveform(self) -> None:
        """Empty waveform has zero duration."""
        result = AudioResult(waveform=np.array([], dtype=np.float32), sample_rate=22050)
        assert result.duration_seconds == 0.0


class TestTTSClient:
    """Test suite for TTSClient (HTTP mode)."""

    def test_not_loaded_raises(self, client: TTSClient) -> None:
        """Operations fail if service not connected."""
        assert client._loaded is False

    @pytest.mark.asyncio
    async def test_synthesize_raises_if_not_loaded(
        self, client: TTSClient
    ) -> None:
        """synthesize() raises TTSModelNotLoadedError."""
        with pytest.raises(TTSModelNotLoadedError):
            await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_clone_raises_if_not_loaded(
        self, client: TTSClient
    ) -> None:
        """clone_voice() raises TTSModelNotLoadedError."""
        with pytest.raises(TTSModelNotLoadedError):
            await client.clone_voice("Hello", b"\x00" * 44, "ref text")

    @pytest.mark.asyncio
    async def test_synthesize_unknown_voice_raises(
        self, client: TTSClient
    ) -> None:
        """synthesize() with unknown voice_id raises TTSGenerationError."""
        # Force loaded state for this test
        client._loaded = True
        client._client = object()  # Mock HTTP client

        with pytest.raises(TTSGenerationError, match="Unknown voice_id"):
            await client.synthesize("Hello", voice_id="nonexistent_speaker")

    @pytest.mark.asyncio
    async def test_health_check_not_loaded(self, client: TTSClient) -> None:
        """health_check() returns False when not connected."""
        # Point to non-existent service
        client._base_url = "http://localhost:1"
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_no_url(self) -> None:
        """health_check() returns False with empty base_url."""
        config = CosyVoiceConfig(base_url="")
        client = TTSClient(config)
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_close_resets_state(self, client: TTSClient) -> None:
        """close() releases client and resets state."""
        client._loaded = True
        client._client = None  # No real client to close

        await client.close()

        assert client._loaded is False
        assert client._client is None

    def test_sample_rate_default(self, client: TTSClient) -> None:
        """sample_rate returns config default."""
        assert client.sample_rate == client.config.sample_rate

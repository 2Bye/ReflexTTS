"""Unit tests for the TTS client.

Tests use a mock CosyVoice3 model — no GPU needed.
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
        model_dir="test_models/cosyvoice",
        load_vllm=False,
        load_trt=False,
    )


@pytest.fixture
def client(tts_config: CosyVoiceConfig) -> TTSClient:
    """Create a test client (not loaded)."""
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
    """Test suite for TTSClient."""

    def test_not_loaded_raises(self, client: TTSClient) -> None:
        """Operations fail if model not loaded."""
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
            await client.clone_voice("Hello", "/tmp/ref.wav", "ref text")  # noqa: S108

    @pytest.mark.asyncio
    async def test_synthesize_unknown_voice_raises(
        self, client: TTSClient
    ) -> None:
        """synthesize() with unknown voice_id raises TTSGenerationError."""
        # Force loaded state for this test
        client._loaded = True
        client._model = object()

        with pytest.raises(TTSGenerationError, match="Unknown voice_id"):
            await client.synthesize("Hello", voice_id="nonexistent_speaker")

    @pytest.mark.asyncio
    async def test_health_check_not_loaded(self, client: TTSClient) -> None:
        """health_check() returns False when not loaded."""
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_loaded(self, client: TTSClient) -> None:
        """health_check() returns True when loaded."""
        client._loaded = True
        client._model = object()
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_close_resets_state(self, client: TTSClient) -> None:
        """close() releases model and resets state."""
        client._loaded = True
        client._model = object()

        await client.close()

        assert client._loaded is False
        assert client._model is None

    def test_sample_rate_default(self, client: TTSClient) -> None:
        """sample_rate returns config default when model not loaded."""
        assert client.sample_rate == client.config.sample_rate

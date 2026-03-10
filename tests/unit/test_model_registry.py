"""Unit tests for the model registry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig
from src.inference.model_registry import HealthStatus, ModelRegistry


@pytest.fixture
def config() -> AppConfig:
    """Create a test config."""
    return AppConfig()


@pytest.fixture
def registry(config: AppConfig) -> ModelRegistry:
    """Create a test registry."""
    return ModelRegistry(config)


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_all_healthy(self) -> None:
        """all_healthy is True only when all backends are up."""
        status = HealthStatus(vllm=True, tts=True, asr=True)
        assert status.all_healthy is True

    def test_not_all_healthy(self) -> None:
        """all_healthy is False if any backend is down."""
        status = HealthStatus(vllm=True, tts=False, asr=True)
        assert status.all_healthy is False

    def test_to_dict(self) -> None:
        """to_dict() includes all fields + all_healthy."""
        status = HealthStatus(vllm=True, tts=True, asr=False)
        d = status.to_dict()
        assert d == {
            "vllm": True,
            "tts": True,
            "asr": False,
            "all_healthy": False,
        }


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_creates_clients(self, registry: ModelRegistry) -> None:
        """Registry creates all three clients."""
        assert registry.vllm is not None
        assert registry.tts is not None
        assert registry.asr is not None

    @pytest.mark.asyncio
    async def test_health_returns_status(self, registry: ModelRegistry) -> None:
        """health() returns HealthStatus with all backends."""
        registry.vllm.health_check = AsyncMock(return_value=False)
        registry.tts.health_check = AsyncMock(return_value=False)
        registry.asr.health_check = AsyncMock(return_value=False)

        status = await registry.health()
        assert status.vllm is False
        assert status.tts is False
        assert status.asr is False

    @pytest.mark.asyncio
    async def test_initialize_skip_gpu(self, registry: ModelRegistry) -> None:
        """initialize(skip_gpu_models=True) only checks vLLM."""
        registry.vllm.health_check = AsyncMock(return_value=False)
        registry.tts.health_check = AsyncMock(return_value=False)
        registry.asr.health_check = AsyncMock(return_value=False)

        status = await registry.initialize(skip_gpu_models=True)
        assert status.vllm is False
        assert status.tts is False

    @pytest.mark.asyncio
    async def test_shutdown_closes_all(self, registry: ModelRegistry) -> None:
        """shutdown() closes all clients."""
        registry.vllm.close = AsyncMock()
        registry.tts.close = AsyncMock()
        registry.asr.close = AsyncMock()

        await registry.shutdown()

        registry.vllm.close.assert_called_once()
        registry.tts.close.assert_called_once()
        registry.asr.close.assert_called_once()

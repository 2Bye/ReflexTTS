"""Unit tests for the vLLM client.

All tests use mocked OpenAI client — no vLLM server needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.config import VLLMConfig
from src.inference.vllm_client import (
    VLLMClient,
    VLLMConnectionError,
    VLLMResponseError,
)


class _MockDirectorOutput(BaseModel):
    """Test model for structured output."""

    segments: list[str]
    emotion: str


@pytest.fixture
def vllm_config() -> VLLMConfig:
    """Create a test vLLM config."""
    return VLLMConfig(
        base_url="http://test:8000/v1",
        model_name="test-model",
        max_retries=2,
        timeout_seconds=5,
    )


@pytest.fixture
def client(vllm_config: VLLMConfig) -> VLLMClient:
    """Create a test client."""
    return VLLMClient(vllm_config)


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock ChatCompletion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.usage = MagicMock()
    mock.usage.total_tokens = 42
    return mock


class TestVLLMClient:
    """Test suite for VLLMClient."""

    @pytest.mark.asyncio
    async def test_chat_returns_text(self, client: VLLMClient) -> None:
        """chat() returns text response."""
        mock_resp = _make_mock_response("Hello world")
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.chat("You are helpful", "Say hello")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_chat_json_parses_model(self, client: VLLMClient) -> None:
        """chat_json() parses response into Pydantic model."""
        json_response = '{"segments": ["hello", "world"], "emotion": "happy"}'
        mock_resp = _make_mock_response(json_response)
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.chat_json(
            "Return JSON", "test", response_model=_MockDirectorOutput
        )
        assert result.segments == ["hello", "world"]
        assert result.emotion == "happy"

    @pytest.mark.asyncio
    async def test_chat_json_raises_on_invalid_json(
        self, client: VLLMClient
    ) -> None:
        """chat_json() raises VLLMResponseError on bad JSON."""
        mock_resp = _make_mock_response("not valid json {{{")
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with pytest.raises(VLLMResponseError, match="Failed to parse"):
            await client.chat_json(
                "Return JSON", "test", response_model=_MockDirectorOutput
            )

    @pytest.mark.asyncio
    async def test_chat_retries_on_connection_error(
        self, client: VLLMClient
    ) -> None:
        """Client retries on connection errors."""
        from openai import APIConnectionError

        mock_ok = _make_mock_response("recovered")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=MagicMock()),
                mock_ok,
            ]
        )

        result = await client.chat("sys", "msg")
        assert result == "recovered"
        assert client._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_raises_after_max_retries(
        self, client: VLLMClient
    ) -> None:
        """Client raises VLLMConnectionError after max retries."""
        from openai import APIConnectionError

        client._client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        with pytest.raises(VLLMConnectionError, match="Failed to connect"):
            await client.chat("sys", "msg")

        assert client._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_raises_on_empty_response(
        self, client: VLLMClient
    ) -> None:
        """Client raises VLLMResponseError on None content."""
        mock = _make_mock_response("")
        mock.choices[0].message.content = None
        client._client.chat.completions.create = AsyncMock(return_value=mock)

        with pytest.raises(VLLMResponseError, match="Empty response"):
            await client.chat("sys", "msg")

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: VLLMClient) -> None:
        """health_check() returns True when server is up."""
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="test-model")]
        client._client.models.list = AsyncMock(return_value=mock_models)

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client: VLLMClient) -> None:
        """health_check() returns False when server is down."""
        client._client.models.list = AsyncMock(side_effect=Exception("down"))

        assert await client.health_check() is False

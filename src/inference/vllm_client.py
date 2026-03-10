"""Async client for vLLM server (OpenAI-compatible API).

Provides structured JSON output for Director and Critic agents.
Connects to a single vLLM instance running Qwen3-8B-Instruct.

Usage:
    client = VLLMClient(config.vllm)
    result = await client.chat(
        system_prompt="You are a text analyst...",
        user_message="Analyze this text",
        response_model=DirectorOutput,
    )
"""

from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from src.config import VLLMConfig
from src.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class VLLMError(Exception):
    """Base exception for vLLM client errors."""


class VLLMConnectionError(VLLMError):
    """Cannot connect to vLLM server."""


class VLLMResponseError(VLLMError):
    """Invalid or unparseable response from vLLM."""


class VLLMClient:
    """Async client for vLLM server with structured output support.

    Wraps the OpenAI-compatible API provided by vLLM.
    Supports retry logic and structured JSON responses via Pydantic.

    Attributes:
        config: vLLM connection configuration.
    """

    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=float(config.timeout_seconds),
        )
        logger.info(
            "vllm_client_initialized",
            base_url=config.base_url,
            model=config.model_name,
        )

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the text response.

        Args:
            system_prompt: System prompt defining agent behavior.
            user_message: User input message.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The assistant's response text.

        Raises:
            VLLMConnectionError: Cannot connect to vLLM server.
            VLLMResponseError: Invalid response.
        """
        return await self._request_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            json_mode=False,
        )

    async def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Send a chat request and parse the response as a Pydantic model.

        Args:
            system_prompt: System prompt (should instruct JSON output).
            user_message: User input message.
            response_model: Pydantic model class to parse the response into.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            VLLMConnectionError: Cannot connect to vLLM server.
            VLLMResponseError: Response cannot be parsed as the model.
        """
        raw = await self._request_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                "vllm_json_parse_error",
                error=str(e),
                raw_response=raw[:500],
                model=response_model.__name__,
            )
            raise VLLMResponseError(
                f"Failed to parse response as {response_model.__name__}: {e}"
            ) from e

    async def _request_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Execute a chat request with exponential backoff retry.

        Args:
            system_prompt: System prompt.
            user_message: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            json_mode: Whether to request JSON output format.

        Returns:
            Raw response text.

        Raises:
            VLLMConnectionError: After all retries exhausted.
            VLLMResponseError: Empty response.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"response_format": {"type": "json_object"}} if json_mode else None,
                )

                content = response.choices[0].message.content
                if content is None:
                    raise VLLMResponseError("Empty response from vLLM")

                logger.debug(
                    "vllm_request_success",
                    attempt=attempt,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                )
                return str(content)

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    "vllm_request_retry",
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(wait_time)

            except APIStatusError as e:
                logger.error(
                    "vllm_api_error",
                    status_code=e.status_code,
                    error=str(e),
                )
                raise VLLMResponseError(f"vLLM API error {e.status_code}: {e}") from e

        raise VLLMConnectionError(
            f"Failed to connect to vLLM after {self.config.max_retries} retries: {last_error}"
        )

    async def health_check(self) -> bool:
        """Check if the vLLM server is reachable and healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            models = await self._client.models.list()
            available = [m.id for m in models.data]
            logger.info("vllm_health_ok", models=available)
            return len(available) > 0
        except Exception as e:
            logger.error("vllm_health_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

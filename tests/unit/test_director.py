"""Unit tests for Director Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.director import _apply_hotfix_hints, run_director
from src.agents.schemas import DirectorOutput, Segment
from src.config import VLLMConfig
from src.inference.vllm_client import VLLMClient
from src.orchestrator.state import DetectedError, ErrorSeverity, GraphState


def _make_director_output() -> DirectorOutput:
    return DirectorOutput(
        segments=[
            Segment(text="Hello world"),
            Segment(text="How are you"),
        ],
        voice_id="speaker_1",
    )


@pytest.fixture
def vllm_mock() -> VLLMClient:
    client = VLLMClient(VLLMConfig(base_url="http://test:8000/v1"))
    client.chat_json = AsyncMock(return_value=_make_director_output())
    return client


class TestDirector:
    """Test suite for Director Agent."""

    @pytest.mark.asyncio
    async def test_director_produces_segments(self, vllm_mock: VLLMClient) -> None:
        state = GraphState(text="Hello world. How are you?", voice_id="speaker_2")
        result = await run_director(state, vllm_mock)

        assert result.ssml_markup is not None
        assert len(result.ssml_markup["segments"]) == 2
        # Voice override
        assert result.ssml_markup["voice_id"] == "speaker_2"

    @pytest.mark.asyncio
    async def test_director_logs_action(self, vllm_mock: VLLMClient) -> None:
        state = GraphState(text="Test")
        result = await run_director(state, vllm_mock)

        assert len(result.agent_log) == 1
        assert result.agent_log[0]["agent"] == "director"

    def test_apply_hotfix_hints(self) -> None:
        output = DirectorOutput(
            segments=[Segment(text="高管对报道予好评")],
        )
        state = GraphState(
            text="高管对报道予好评",
            iteration=1,
            errors=[
                DetectedError(
                    word_expected="予",
                    word_actual="与",
                    start_ms=1000,
                    end_ms=1500,
                    severity=ErrorSeverity.WARNING,
                    can_hotfix=True,
                    hotfix_hint="[j][ǐ]",
                ),
            ],
        )

        result = _apply_hotfix_hints(output, state)
        assert "[j][ǐ]" in result.segments[0].text
